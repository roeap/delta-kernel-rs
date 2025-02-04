use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use arrow_ord::comparison::{in_list, in_list_utf8};
use arrow_schema::{DataType as ArrowDataType, IntervalUnit, TimeUnit};

use super::{evaluate_expression, extract_column, wrap_comparison_result};
use crate::error::Error;
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
                let list_array = $right_arr.as_list_opt::<i32>()
                    .ok_or(Error::invalid_expression(
                        format!("Cannot cast to list array: {}", $right_arr.data_type()))
                    )?;
                Ok(fix_arrow_in_list_result(
                    in_list(prim_array, list_array).map_err(Error::generic_err)?,
                    list_array.iter(),
                ))
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
            let left_arr = evaluate_expression(left, batch, None)?;
            let right_arr = evaluate_expression(right, batch, None)?;
            if let Some(string_arr) = left_arr.as_string_opt::<i32>() {
                if let Some(right_arr) = right_arr.as_list_opt::<i32>() {
                    return Ok(fix_arrow_in_list_result(
                        in_list_utf8(string_arr, right_arr).map_err(Error::generic_err)?,
                        right_arr.iter(),
                    ));
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
                    (ArrowDataType::Utf8, PrimitiveType::String) => is_in_list(
                        ad, str_op(column.as_string::<i32>())
                    ),
                    (ArrowDataType::LargeUtf8, PrimitiveType::String) => is_in_list(
                        ad, str_op(column.as_string::<i64>())
                    ),
                    (ArrowDataType::Utf8View, PrimitiveType::String) => is_in_list(
                        ad, str_op(column.as_string_view())
                    ),
                    (ArrowDataType::Int8, PrimitiveType::Byte) =>  is_in_list(
                        ad,op::<Int8Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Int16, PrimitiveType::Short) => is_in_list(
                        ad,op::<Int16Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Int32, PrimitiveType::Integer) => is_in_list(
                        ad,op::<Int32Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Int64, PrimitiveType::Long) => is_in_list(
                        ad,op::<Int64Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Float32, PrimitiveType::Float) => is_in_list(
                        ad,op::<Float32Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Float64, PrimitiveType::Double) => is_in_list(
                        ad,op::<Float64Type>(&column, Scalar::from)
                    ),
                    (ArrowDataType::Date32, PrimitiveType::Date) =>  is_in_list(
                        ad,op::<Date32Type>(&column, Scalar::Date)
                    ),
                    (
                        ArrowDataType::Timestamp(TimeUnit::Microsecond, Some(_)),
                        PrimitiveType::Timestamp,
                    ) => is_in_list(
                        ad, op::<TimestampMicrosecondType>(column.as_ref(), Scalar::Timestamp)
                    ),
                    (
                        ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                        PrimitiveType::TimestampNtz,
                    ) => is_in_list(
                        ad, op::<TimestampMicrosecondType>(column.as_ref(), Scalar::TimestampNtz)
                    ),
                    (l, r) => {
                        return Err(Error::invalid_expression(format!(
                        "Cannot check if value of type '{l}' is contained in array with values of type '{r}'"
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

#[cfg(test)]
mod tests {
    use arrow_array::{GenericStringArray, Int32Array, ListArray};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::Field;

    use super::*;
    use crate::{
        expressions::column_expr,
        schema::{ArrayType, DataType},
    };

    #[test]
    fn test_fix_arrow_in_list_result() {
        let in_lists: [Option<ArrayRef>; 2] = [
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
    fn test_eval_in_list_lit_in_col() {
        let values = GenericStringArray::<i32>::from(vec![
            Some("one"),
            None,
            Some("two"),
            None,
            Some("one"),
            Some("two"),
        ]);
        let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6]));
        let field = Arc::new(Field::new("item", ArrowDataType::Utf8, true));
        let arr_field = Arc::new(Field::new("item", ArrowDataType::List(field.clone()), true));
        let schema = arrow_schema::Schema::new([arr_field.clone()]);
        let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

        let result = eval_in_list(
            &batch,
            &Expression::literal(Scalar::from("one")),
            &column_expr!("item"),
        )
        .unwrap();
        let expected = Arc::new(BooleanArray::from(vec![Some(true), None, Some(true)]));
        assert_eq!(result.as_ref(), expected.as_ref());

        let result = eval_in_list(
            &batch,
            &Expression::literal(Scalar::from("three")),
            &column_expr!("item"),
        )
        .unwrap();
        let expected = Arc::new(BooleanArray::from(vec![None, None, Some(false)]));
        assert_eq!(result.as_ref(), expected.as_ref());
    }

    #[test]
    fn test_eval_in_list_col_in_lit() {
        let values = GenericStringArray::<i32>::from(vec![Some("one"), None, Some("three")]);
        let field = Arc::new(Field::new("item", ArrowDataType::Utf8, true));
        let schema = arrow_schema::Schema::new([field.clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();

        let rhs = Expression::literal(Scalar::Array(ArrayData::new(
            ArrayType::new(PrimitiveType::String.into(), true),
            ["one".into(), Scalar::Null(DataType::STRING)],
        )));
        let result = eval_in_list(&batch, &column_expr!("item"), &rhs).unwrap();
        let expected = Arc::new(BooleanArray::from(vec![Some(true), None, None]));
        assert_eq!(result.as_ref(), expected.as_ref());
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
