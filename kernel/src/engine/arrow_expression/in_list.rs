use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{types::*, ArrowNumericType, GenericListArray, OffsetSizeTrait, PrimitiveArray};
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_cast::cast;
use arrow_ord::comparison::{in_list, in_list_utf8};
use arrow_schema::{DataType as ArrowDataType, IntervalUnit, TimeUnit};

use super::{evaluate_expression, extract_column, wrap_comparison_result};
use crate::error::{DeltaResult, Error};
use crate::expressions::{ArrayData, Expression, Scalar, VariadicOperator};
use crate::predicates::PredicateEvaluatorDefaults;
use crate::schema::PrimitiveType;

macro_rules! prim_array_cmp {
    ( $left_arr: ident, $right_arr: ident, $(($data_ty: pat, $prim_ty: ty)),+ ) => {
        match $left_arr.data_type() {
        $(
            $data_ty => {
                let prim_array = $left_arr.as_primitive_opt::<$prim_ty>()
                    .ok_or(Error::invalid_expression(
                        format!("Cannot cast to primitive array: {}", $left_arr.data_type()))
                    )?;
                match $right_arr.data_type() {
                    ArrowDataType::List(_) => {
                        eval_arrow_in_list(prim_array, $right_arr.as_list::<i32>())
                    },
                    ArrowDataType::LargeList(_) => {
                        eval_arrow_in_list(prim_array, $right_arr.as_list::<i64>())
                    },
                    // TODO: LargeListView - not fully supported by arrow yet
                    _ => {
                        Err(Error::invalid_expression(format!(
                            "Expected right hand side to be list column, got: {:?}",
                            $right_arr.data_type()
                        )))
                    }
                }
            }
        )+
            _ => Err(Error::invalid_expression(
                format!("Bad Comparison between: {:?} and {:?}",
                    $left_arr.data_type(),
                    $right_arr.data_type())
                )
            )
        }
    }
}

pub(super) fn eval_in_list(
    batch: &RecordBatch,
    left: &Expression,
    right: &Expression,
) -> Result<Arc<dyn Array>, Error> {
    use Expression::*;

    match (left, right) {
        (Literal(lit), Column(_)) => {
            if lit.is_null() {
                return Ok(Arc::new(BooleanArray::new_null(batch.num_rows())));
            }

            let right_arr = evaluate_expression(right, batch, None)?;
            let list_field = match right_arr.data_type() {
                ArrowDataType::List(list_field) => list_field,
                ArrowDataType::LargeList(list_field) => list_field,
                // TODO: LargeListView - not fully supported by arrow yet
                _ => {
                    return Err(Error::invalid_expression(format!(
                        "Expected right hand side to be list column, got: {:?}",
                        right_arr.data_type()
                    )));
                }
            };

            let left_arr =
                evaluate_expression(left, batch, Some(&list_field.data_type().try_into()?))?;

            if matches!(
                list_field.data_type(),
                ArrowDataType::Utf8 | ArrowDataType::LargeUtf8
            ) {
                if let Some(right_arr) = right_arr.as_list_opt::<i32>() {
                    // Note: since delta types are purely logical and arrow types also consider the
                    // the physical layout, we may need to cast between different offsets.
                    let left_arr = if left_arr.data_type() == list_field.data_type() {
                        left_arr
                    } else {
                        cast(left_arr.as_ref(), list_field.data_type())
                            .map_err(Error::generic_err)?
                    };

                    let result = match left_arr.data_type() {
                        ArrowDataType::Utf8 => in_list_utf8(left_arr.as_string::<i32>(), right_arr),
                        ArrowDataType::LargeUtf8 => {
                            in_list_utf8(left_arr.as_string::<i64>(), right_arr)
                        }
                        _ => unreachable!(),
                    }
                    .map_err(Error::generic_err)?;
                    return Ok(fix_arrow_in_list_result(result, right_arr.iter()));
                }
            }

            prim_array_cmp! {
                left_arr, right_arr,
                (ArrowDataType::Int8, Int8Type),
                (ArrowDataType::Int16, Int16Type),
                (ArrowDataType::Int32, Int32Type),
                (ArrowDataType::Int64, Int64Type),
                (ArrowDataType::UInt8, UInt8Type),
                (ArrowDataType::UInt16, UInt16Type),
                (ArrowDataType::UInt32, UInt32Type),
                (ArrowDataType::UInt64, UInt64Type),
                (ArrowDataType::Float16, Float16Type),
                (ArrowDataType::Float32, Float32Type),
                (ArrowDataType::Float64, Float64Type),
                (ArrowDataType::Timestamp(TimeUnit::Second, _), TimestampSecondType),
                (ArrowDataType::Timestamp(TimeUnit::Millisecond, _), TimestampMillisecondType),
                (ArrowDataType::Timestamp(TimeUnit::Microsecond, _), TimestampMicrosecondType),
                (ArrowDataType::Timestamp(TimeUnit::Nanosecond, _), TimestampNanosecondType),
                (ArrowDataType::Date32, Date32Type),
                (ArrowDataType::Date64, Date64Type),
                (ArrowDataType::Time32(TimeUnit::Second), Time32SecondType),
                (ArrowDataType::Time32(TimeUnit::Millisecond), Time32MillisecondType),
                (ArrowDataType::Time64(TimeUnit::Microsecond), Time64MicrosecondType),
                (ArrowDataType::Time64(TimeUnit::Nanosecond), Time64NanosecondType),
                (ArrowDataType::Duration(TimeUnit::Second), DurationSecondType),
                (ArrowDataType::Duration(TimeUnit::Millisecond), DurationMillisecondType),
                (ArrowDataType::Duration(TimeUnit::Microsecond), DurationMicrosecondType),
                (ArrowDataType::Duration(TimeUnit::Nanosecond), DurationNanosecondType),
                (ArrowDataType::Interval(IntervalUnit::DayTime), IntervalDayTimeType),
                (ArrowDataType::Interval(IntervalUnit::YearMonth), IntervalYearMonthType),
                (ArrowDataType::Interval(IntervalUnit::MonthDayNano), IntervalMonthDayNanoType),
                (ArrowDataType::Decimal128(_, _), Decimal128Type),
                (ArrowDataType::Decimal256(_, _), Decimal256Type)
            }
        }
        (Column(name), Literal(Scalar::Array(ad))) => {
            fn op<T: ArrowPrimitiveType>(
                values: &dyn Array,
                from: fn(T::Native) -> Scalar,
            ) -> impl IntoIterator<Item = Option<Scalar>> + '_ {
                values.as_primitive::<T>().iter().map(move |v| v.map(from))
            }

            fn str_op<'a>(
                column: impl IntoIterator<Item = Option<&'a str>> + 'a,
            ) -> impl IntoIterator<Item = Option<Scalar>> + 'a {
                column.into_iter().map(|v| v.map(Scalar::from))
            }

            fn binary_op<'a>(
                column: impl IntoIterator<Item = Option<&'a [u8]>> + 'a,
            ) -> impl IntoIterator<Item = Option<Scalar>> + 'a {
                column.into_iter().map(|v| v.map(Scalar::from))
            }

            let column = extract_column(batch, name)?;
            let data_type = ad
                .array_type()
                .element_type()
                .as_primitive_opt()
                .ok_or_else(|| {
                    Error::invalid_expression(format!(
                        "IN only supports array literals with primitive elements, got: '{:?}'",
                        ad.array_type().element_type()
                    ))
                })?;

            // safety: as_* methods on arrow arrays can panic, but we checked the data type before applying.
            let arr = match (column.data_type(), data_type) {
                (ArrowDataType::Utf8, PrimitiveType::String) => {
                    is_in_list(ad, str_op(column.as_string::<i32>()))
                }
                (ArrowDataType::LargeUtf8, PrimitiveType::String) => {
                    is_in_list(ad, str_op(column.as_string::<i64>()))
                }
                (ArrowDataType::Utf8View, PrimitiveType::String) => {
                    is_in_list(ad, str_op(column.as_string_view()))
                }
                (ArrowDataType::Int8, PrimitiveType::Byte) => {
                    is_in_list(ad, op::<Int8Type>(&column, Scalar::from))
                }
                (ArrowDataType::Int16, PrimitiveType::Short) => {
                    is_in_list(ad, op::<Int16Type>(&column, Scalar::from))
                }
                (ArrowDataType::Int32, PrimitiveType::Integer) => {
                    is_in_list(ad, op::<Int32Type>(&column, Scalar::from))
                }
                (ArrowDataType::Int64, PrimitiveType::Long) => {
                    is_in_list(ad, op::<Int64Type>(&column, Scalar::from))
                }
                (ArrowDataType::Float32, PrimitiveType::Float) => {
                    is_in_list(ad, op::<Float32Type>(&column, Scalar::from))
                }
                (ArrowDataType::Float64, PrimitiveType::Double) => {
                    is_in_list(ad, op::<Float64Type>(&column, Scalar::from))
                }
                (ArrowDataType::Boolean, PrimitiveType::Boolean) => {
                    let iter = column.as_boolean().into_iter().map(|v| v.map(Scalar::from));
                    is_in_list(ad, iter)
                }
                (ArrowDataType::Binary, PrimitiveType::Binary) => {
                    is_in_list(ad, binary_op(column.as_binary::<i32>()))
                }
                (ArrowDataType::LargeBinary, PrimitiveType::Binary) => {
                    is_in_list(ad, binary_op(column.as_binary::<i64>()))
                }
                (ArrowDataType::BinaryView, PrimitiveType::Binary) => {
                    is_in_list(ad, binary_op(column.as_binary_view()))
                }
                (ArrowDataType::Date32, PrimitiveType::Date) => {
                    is_in_list(ad, op::<Date32Type>(&column, Scalar::Date))
                }
                (
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, Some(_)),
                    PrimitiveType::Timestamp,
                ) => is_in_list(
                    ad,
                    op::<TimestampMicrosecondType>(column.as_ref(), Scalar::Timestamp),
                ),
                (
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                    PrimitiveType::TimestampNtz,
                ) => is_in_list(
                    ad,
                    op::<TimestampMicrosecondType>(column.as_ref(), Scalar::TimestampNtz),
                ),
                (l, r) => {
                    return Err(Error::invalid_expression(format!(
                        "Cannot check if value of type '{l}' is in array with value type '{r}'"
                    )))
                }
            };
            Ok(wrap_comparison_result(arr))
        }
        (Literal(lit), Literal(Scalar::Array(ad))) => {
            let res = is_in_list(ad, Some(Some(lit.clone())));
            let exists = res.is_valid(0).then(|| res.value(0));
            Ok(Arc::new(BooleanArray::from(vec![exists; batch.num_rows()])))
        }
        (l, r) => Err(Error::invalid_expression(format!(
            "Invalid right value for (NOT) IN comparison, left is: {l} right is: {r}"
        ))),
    }
}

fn is_in_list(ad: &ArrayData, values: impl IntoIterator<Item = Option<Scalar>>) -> BooleanArray {
    #[allow(deprecated)]
    let inlist = ad.array_elements();
    // `v IN (k1, ..., kN)` is logically equivalent to `v = k1 OR ... OR v = kN`, so evaluate
    // it as such, ensuring correct handling of NULL inputs (including `Scalar::Null`).
    values
        .into_iter()
        .map(|v| {
            PredicateEvaluatorDefaults::finish_eval_variadic(
                VariadicOperator::Or,
                inlist
                    .iter()
                    .map(|k| Some(v.as_ref()?.partial_cmp(k)? == Ordering::Equal)),
                false,
            )
        })
        .collect()
}

// helper function to make arrow in_list* kernel results comliant with SQL NULL semantics.
// Specifically, if an item is not found in the in-list, but the in-list contains NULLs, the
// result should be NULL (UNKNOWN) as well.
fn fix_arrow_in_list_result(
    results: BooleanArray,
    in_lists: impl IntoIterator<Item = Option<ArrayRef>>,
) -> ArrayRef {
    wrap_comparison_result(
        results
            .iter()
            .zip(in_lists)
            .map(|(res, arr)| match (res, arr) {
                (Some(false), Some(arr)) if arr.null_count() > 0 => None,
                _ => res,
            })
            .collect(),
    )
}

fn eval_arrow_in_list<T: ArrowNumericType, O: OffsetSizeTrait>(
    left: &PrimitiveArray<T>,
    right: &GenericListArray<O>,
) -> DeltaResult<ArrayRef> {
    Ok(fix_arrow_in_list_result(
        in_list(left, right).map_err(Error::generic_err)?,
        right.iter(),
    ))
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        BinaryArray, BinaryViewArray, Date32Array, Float32Array, Float64Array, Int16Array,
        Int32Array, Int64Array, Int8Array, LargeBinaryArray, LargeListArray, LargeStringArray,
        ListArray, StringArray, StringViewArray, TimestampMicrosecondArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::Field;

    use super::*;
    use crate::{
        expressions::column_expr,
        schema::{ArrayType, DataType},
    };

    #[test]
    fn test_fix_arrow_in_list_result() {
        let in_lists = [
            Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
        ]
        .map(Some);

        let results = BooleanArray::from(vec![Some(false), Some(false)]);
        let expected = Arc::new(BooleanArray::from(vec![None, Some(false)]));
        assert_eq!(
            fix_arrow_in_list_result(results, in_lists.clone()).as_ref(),
            expected.as_ref()
        );

        let results = BooleanArray::from(vec![Some(true), Some(true)]);
        let expected = Arc::new(BooleanArray::from(vec![Some(true), Some(true)]));
        assert_eq!(
            fix_arrow_in_list_result(results, in_lists).as_ref(),
            expected.as_ref()
        );
    }

    #[test]
    fn test_is_in_list() {
        let values = [
            Some(Scalar::Integer(1)),
            Some(Scalar::Integer(3)),
            Some(Scalar::Null(DataType::INTEGER)),
        ];

        let in_list = ArrayData::new(
            ArrayType::new(DataType::INTEGER, false),
            [Scalar::Integer(1), Scalar::Integer(2)],
        );
        let expected = BooleanArray::from(vec![Some(true), Some(false), None]);
        assert_eq!(is_in_list(&in_list, values.clone()), expected);

        let in_list = ArrayData::new(
            ArrayType::new(DataType::INTEGER, true),
            [Scalar::Integer(1), Scalar::Null(DataType::INTEGER)],
        );
        let expected = BooleanArray::from(vec![Some(true), None, None]);
        assert_eq!(is_in_list(&in_list, values), expected);
    }

    #[test]
    fn test_eval_in_list_lit_in_col_large() {
        let values = Int32Array::from(vec![Some(1), None, Some(2), None, Some(1), Some(2)]);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6]));
        let field = Arc::new(Field::new("item", ArrowDataType::Int32, true));
        let arr_field = Arc::new(Field::new(
            "item",
            ArrowDataType::LargeList(field.clone()),
            true,
        ));
        let schema = arrow_schema::Schema::new([arr_field.clone()]);
        let array = LargeListArray::new(field.clone(), offsets, Arc::new(values), None);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

        let result = eval_in_list(
            &batch,
            &Expression::literal(Scalar::Integer(1)),
            &column_expr!("item"),
        )
        .unwrap();
        let expected = Arc::new(BooleanArray::from(vec![Some(true), None, Some(true)]));
        assert_eq!(result.as_ref(), expected.as_ref());

        let result = eval_in_list(
            &batch,
            &Expression::literal(Scalar::Integer(3)),
            &column_expr!("item"),
        )
        .unwrap();
        let expected = Arc::new(BooleanArray::from(vec![None, None, Some(false)]));
        assert_eq!(result.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_eval_in_list() {
        let cases = [
            (
                DataType::STRING,
                Arc::new(StringArray::from(vec![Some("one"), None, Some("three")])) as ArrayRef,
                Scalar::from("one"),
            ),
            (
                DataType::STRING,
                Arc::new(LargeStringArray::from(vec![
                    Some("one"),
                    None,
                    Some("three"),
                ])),
                Scalar::from("one"),
            ),
            (
                DataType::STRING,
                Arc::new(StringViewArray::from(vec![
                    Some("one"),
                    None,
                    Some("three"),
                ])),
                Scalar::from("one"),
            ),
            (
                DataType::INTEGER,
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
                Scalar::Integer(1),
            ),
            (
                DataType::SHORT,
                Arc::new(Int16Array::from(vec![Some(1), None, Some(3)])),
                Scalar::Short(1),
            ),
            (
                DataType::LONG,
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])),
                Scalar::Long(1),
            ),
            (
                DataType::BYTE,
                Arc::new(Int8Array::from(vec![Some(1), None, Some(3)])),
                Scalar::Byte(1),
            ),
            (
                DataType::FLOAT,
                Arc::new(Float32Array::from(vec![Some(1.0), None, Some(3.0)])),
                Scalar::Float(1.0),
            ),
            (
                DataType::DOUBLE,
                Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)])),
                Scalar::Double(1.0),
            ),
            (
                DataType::BOOLEAN,
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
                Scalar::Boolean(true),
            ),
            (
                DataType::BINARY,
                Arc::new(BinaryArray::from_opt_vec(vec![
                    Some(b"one"),
                    None,
                    Some(b"three"),
                ])),
                Scalar::Binary(b"one".to_vec()),
            ),
            (
                DataType::BINARY,
                Arc::new(LargeBinaryArray::from_opt_vec(vec![
                    Some(b"one"),
                    None,
                    Some(b"three"),
                ])),
                Scalar::Binary(b"one".to_vec()),
            ),
            (
                DataType::BINARY,
                Arc::new(BinaryViewArray::from_iter(vec![
                    Some(b"one".to_vec()),
                    None,
                    Some(b"three".to_vec()),
                ])),
                Scalar::Binary(b"one".to_vec()),
            ),
            // TODO: decimal
            (
                DataType::DATE,
                Arc::new(Date32Array::from(vec![Some(1), None, Some(3)])),
                Scalar::Date(1),
            ),
            (
                DataType::TIMESTAMP,
                Arc::new(
                    TimestampMicrosecondArray::from(vec![Some(1), None, Some(3)])
                        .with_timezone_utc(),
                ),
                Scalar::Timestamp(1),
            ),
            (
                DataType::TIMESTAMP_NTZ,
                Arc::new(TimestampMicrosecondArray::from(vec![
                    Some(1),
                    None,
                    Some(3),
                ])),
                Scalar::TimestampNtz(1),
            ),
        ];

        // unsupported types due to lack of arrow kernel support
        let unsupported = &[
            ArrowDataType::Utf8View,
            ArrowDataType::Binary,
            ArrowDataType::LargeBinary,
            ArrowDataType::BinaryView,
            ArrowDataType::Boolean,
        ];

        for (data_type, values, list_value) in cases {
            let field = Arc::new(Field::new("item", values.data_type().clone(), true));
            let schema = arrow_schema::Schema::new([field.clone()]);
            let batch = RecordBatch::try_new(Arc::new(schema), vec![values.clone()]).unwrap();

            // test "col IN (lit1, lit2, ...)"
            let rhs = Expression::literal(Scalar::Array(ArrayData::new(
                ArrayType::new(data_type.clone(), true),
                [list_value.clone(), Scalar::Null(data_type)],
            )));
            let result = eval_in_list(&batch, &column_expr!("item"), &rhs).unwrap();
            let expected = Arc::new(BooleanArray::from(vec![Some(true), None, None]));
            assert_eq!(result.as_ref(), expected.as_ref());

            // test "lit IN (col)"
            let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2, 3]));
            let list_field = Arc::new(Field::new("item", ArrowDataType::List(field.clone()), true));
            let schema = arrow_schema::Schema::new([list_field.clone()]);
            let array = ListArray::new(field.clone(), offsets, values.clone(), None);
            let batch =
                RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();
            let result = eval_in_list(
                &batch,
                &Expression::literal(list_value),
                &column_expr!("item"),
            );
            if unsupported.contains(values.data_type()) {
                assert!(result.is_err());
            } else {
                let result = result.unwrap();
                let expected = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
                assert_eq!(result.as_ref(), expected.as_ref());
            }
        }
    }

    #[test]
    fn test_eval_in_list_lit_in_lit() {
        let dummy = arrow_schema::Schema::new(vec![Field::new("a", ArrowDataType::Boolean, true)]);
        let dummy_batch =
            RecordBatch::try_new(Arc::new(dummy), vec![Arc::new(BooleanArray::new_null(1))])
                .unwrap();

        let rhs = Expression::literal(Scalar::Array(ArrayData::new(
            ArrayType::new(PrimitiveType::Integer.into(), true),
            [Scalar::Integer(1), Scalar::Null(DataType::INTEGER)],
        )));

        let result =
            eval_in_list(&dummy_batch, &Expression::literal(Scalar::Integer(1)), &rhs).unwrap();
        let expected = Arc::new(BooleanArray::from(vec![Some(true)]));
        assert_eq!(result.as_ref(), expected.as_ref());

        let result = eval_in_list(
            &dummy_batch,
            &Expression::literal(Scalar::Integer(1)),
            &Expression::literal(Scalar::Integer(1)),
        );
        assert!(result.is_err());
    }
}
