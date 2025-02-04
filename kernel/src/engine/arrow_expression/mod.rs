//! Expression handling based on arrow-rs compute kernels.
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_arith::boolean::{and_kleene, is_null, not, or_kleene};
use arrow_arith::numeric::{add, div, mul, sub};
use arrow_array::cast::AsArray;
use arrow_array::MapArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Datum, Decimal128Array, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, ListArray, RecordBatch,
    StringArray, StructArray, TimestampMicrosecondArray,
};
use arrow_buffer::OffsetBuffer;
use arrow_ord::cmp::{distinct, eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::{
    ArrowError, DataType as ArrowDataType, Field as ArrowField, Fields, Schema as ArrowSchema,
};
use arrow_select::concat::concat;
use itertools::Itertools;

use super::arrow_conversion::LIST_ARRAY_ROOT;
use super::arrow_utils::make_arrow_error;
use crate::engine::arrow_data::ArrowEngineData;
use crate::engine::ensure_data_types::ensure_data_types;
use crate::error::{DeltaResult, Error};
use crate::expressions::{
    BinaryExpression, BinaryOperator, Expression, Scalar, UnaryExpression, UnaryOperator,
    VariadicExpression, VariadicOperator,
};
use crate::schema::{ArrayType, DataType, MapType, PrimitiveType, Schema, SchemaRef, StructField};
use crate::{EngineData, ExpressionEvaluator, ExpressionHandler};

mod in_list;
#[cfg(test)]
mod tests;

// TODO leverage scalars / Datum

fn downcast_to_bool(arr: &dyn Array) -> DeltaResult<&BooleanArray> {
    arr.as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| Error::generic("expected boolean array"))
}

impl Scalar {
    /// Convert scalar to arrow array.
    pub fn to_array(&self, num_rows: usize) -> DeltaResult<ArrayRef> {
        use Scalar::*;
        let arr: ArrayRef = match self {
            Integer(val) => Arc::new(Int32Array::from_value(*val, num_rows)),
            Long(val) => Arc::new(Int64Array::from_value(*val, num_rows)),
            Short(val) => Arc::new(Int16Array::from_value(*val, num_rows)),
            Byte(val) => Arc::new(Int8Array::from_value(*val, num_rows)),
            Float(val) => Arc::new(Float32Array::from_value(*val, num_rows)),
            Double(val) => Arc::new(Float64Array::from_value(*val, num_rows)),
            String(val) => Arc::new(StringArray::from(vec![val.clone(); num_rows])),
            Boolean(val) => Arc::new(BooleanArray::from(vec![*val; num_rows])),
            Timestamp(val) => {
                Arc::new(TimestampMicrosecondArray::from_value(*val, num_rows).with_timezone("UTC"))
            }
            TimestampNtz(val) => Arc::new(TimestampMicrosecondArray::from_value(*val, num_rows)),
            Date(val) => Arc::new(Date32Array::from_value(*val, num_rows)),
            Binary(val) => Arc::new(BinaryArray::from(vec![val.as_slice(); num_rows])),
            Decimal(val, precision, scale) => Arc::new(
                Decimal128Array::from_value(*val, num_rows)
                    .with_precision_and_scale(*precision, *scale as i8)?,
            ),
            Struct(data) => {
                let arrays = data
                    .values()
                    .iter()
                    .map(|val| val.to_array(num_rows))
                    .try_collect()?;
                let fields: Fields = data
                    .fields()
                    .iter()
                    .map(ArrowField::try_from)
                    .try_collect()?;
                Arc::new(StructArray::try_new(fields, arrays, None)?)
            }
            Array(data) => {
                #[allow(deprecated)]
                let values = data.array_elements();
                let vecs: Vec<_> = values.iter().map(|v| v.to_array(num_rows)).try_collect()?;
                let values: Vec<_> = vecs.iter().map(|x| x.as_ref()).collect();
                let offsets: Vec<_> = vecs.iter().map(|v| v.len()).collect();
                let offset_buffer = OffsetBuffer::from_lengths(offsets);
                let field = ArrowField::try_from(data.array_type())?;
                Arc::new(ListArray::new(
                    Arc::new(field),
                    offset_buffer,
                    concat(values.as_slice())?,
                    None,
                ))
            }
            Null(data_type) => match data_type {
                DataType::Primitive(primitive) => match primitive {
                    PrimitiveType::Byte => Arc::new(Int8Array::new_null(num_rows)),
                    PrimitiveType::Short => Arc::new(Int16Array::new_null(num_rows)),
                    PrimitiveType::Integer => Arc::new(Int32Array::new_null(num_rows)),
                    PrimitiveType::Long => Arc::new(Int64Array::new_null(num_rows)),
                    PrimitiveType::Float => Arc::new(Float32Array::new_null(num_rows)),
                    PrimitiveType::Double => Arc::new(Float64Array::new_null(num_rows)),
                    PrimitiveType::String => Arc::new(StringArray::new_null(num_rows)),
                    PrimitiveType::Boolean => Arc::new(BooleanArray::new_null(num_rows)),
                    PrimitiveType::Timestamp => {
                        Arc::new(TimestampMicrosecondArray::new_null(num_rows).with_timezone("UTC"))
                    }
                    PrimitiveType::TimestampNtz => {
                        Arc::new(TimestampMicrosecondArray::new_null(num_rows))
                    }
                    PrimitiveType::Date => Arc::new(Date32Array::new_null(num_rows)),
                    PrimitiveType::Binary => Arc::new(BinaryArray::new_null(num_rows)),
                    PrimitiveType::Decimal(precision, scale) => Arc::new(
                        Decimal128Array::new_null(num_rows)
                            .with_precision_and_scale(*precision, *scale as i8)?,
                    ),
                },
                DataType::Struct(t) => {
                    let fields: Fields = t.fields().map(ArrowField::try_from).try_collect()?;
                    Arc::new(StructArray::new_null(fields, num_rows))
                }
                DataType::Array(t) => {
                    let field =
                        ArrowField::new(LIST_ARRAY_ROOT, t.element_type().try_into()?, true);
                    Arc::new(ListArray::new_null(Arc::new(field), num_rows))
                }
                DataType::Map { .. } => unimplemented!(),
            },
        };
        Ok(arr)
    }
}

fn wrap_comparison_result(arr: BooleanArray) -> ArrayRef {
    Arc::new(arr) as _
}

trait ProvidesColumnByName {
    fn column_by_name(&self, name: &str) -> Option<&ArrayRef>;
}

impl ProvidesColumnByName for RecordBatch {
    fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.column_by_name(name)
    }
}

impl ProvidesColumnByName for StructArray {
    fn column_by_name(&self, name: &str) -> Option<&ArrayRef> {
        self.column_by_name(name)
    }
}

// Given a RecordBatch or StructArray, recursively probe for a nested column path and return the
// corresponding column, or Err if the path is invalid. For example, given the following schema:
// ```text
// root: {
//   a: int32,
//   b: struct {
//     c: int32,
//     d: struct {
//       e: int32,
//       f: int64,
//     },
//   },
// }
// ```
// The path ["b", "d", "f"] would retrieve the int64 column while ["a", "b"] would produce an error.
fn extract_column(mut parent: &dyn ProvidesColumnByName, col: &[String]) -> DeltaResult<ArrayRef> {
    let mut field_names = col.iter();
    let Some(mut field_name) = field_names.next() else {
        return Err(ArrowError::SchemaError("Empty column path".to_string()))?;
    };
    loop {
        let child = parent
            .column_by_name(field_name)
            .ok_or_else(|| ArrowError::SchemaError(format!("No such field: {field_name}")))?;
        field_name = match field_names.next() {
            Some(name) => name,
            None => return Ok(child.clone()),
        };
        parent = child
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| ArrowError::SchemaError(format!("Not a struct: {field_name}")))?;
    }
}

fn evaluate_expression(
    expression: &Expression,
    batch: &RecordBatch,
    result_type: Option<&DataType>,
) -> DeltaResult<ArrayRef> {
    use BinaryOperator::*;
    use Expression::*;
    match (expression, result_type) {
        (Literal(scalar), _) => Ok(scalar.to_array(batch.num_rows())?),
        (Column(name), _) => extract_column(batch, name),
        (Struct(fields), Some(DataType::Struct(output_schema))) => {
            let columns = fields
                .iter()
                .zip(output_schema.fields())
                .map(|(expr, field)| evaluate_expression(expr, batch, Some(field.data_type())));
            let output_cols: Vec<ArrayRef> = columns.try_collect()?;
            let output_fields: Vec<ArrowField> = output_cols
                .iter()
                .zip(output_schema.fields())
                .map(|(output_col, output_field)| -> DeltaResult<_> {
                    Ok(ArrowField::new(
                        output_field.name(),
                        output_col.data_type().clone(),
                        output_col.is_nullable(),
                    ))
                })
                .try_collect()?;
            let result = StructArray::try_new(output_fields.into(), output_cols, None)?;
            Ok(Arc::new(result))
        }
        (Struct(_), _) => Err(Error::generic(
            "Result type is required to evaluate struct expressions",
        )),
        (Unary(UnaryExpression { op, expr }), _) => {
            let arr = evaluate_expression(expr.as_ref(), batch, None)?;
            Ok(match op {
                UnaryOperator::Not => Arc::new(not(downcast_to_bool(&arr)?)?),
                UnaryOperator::IsNull => Arc::new(is_null(&arr)?),
            })
        }
        (
            Binary(BinaryExpression {
                op: In,
                left,
                right,
            }),
            None | Some(&DataType::BOOLEAN),
        ) => in_list::eval_in_list(batch, left, right),
        (
            Binary(BinaryExpression {
                op: NotIn,
                left,
                right,
            }),
            None | Some(&DataType::BOOLEAN),
        ) => {
            let reverse_op = Expression::binary(In, *left.clone(), *right.clone());
            let reverse_expr = evaluate_expression(&reverse_op, batch, None)?;
            not(reverse_expr.as_boolean())
                .map(wrap_comparison_result)
                .map_err(Error::generic_err)
        }
        (Binary(BinaryExpression { op: In | NotIn, .. }), Some(_)) => {
            Err(Error::invalid_expression(format!(
                "(NOT) IN expression is expected to return boolean results, got: {result_type:?}"
            )))
        }
        (Binary(BinaryExpression { op, left, right }), _) => {
            let left_arr = evaluate_expression(left.as_ref(), batch, None)?;
            let right_arr = evaluate_expression(right.as_ref(), batch, None)?;

            type Operation = fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>;
            let eval: Operation = match op {
                Plus => add,
                Minus => sub,
                Multiply => mul,
                Divide => div,
                LessThan => |l, r| lt(l, r).map(wrap_comparison_result),
                LessThanOrEqual => |l, r| lt_eq(l, r).map(wrap_comparison_result),
                GreaterThan => |l, r| gt(l, r).map(wrap_comparison_result),
                GreaterThanOrEqual => |l, r| gt_eq(l, r).map(wrap_comparison_result),
                Equal => |l, r| eq(l, r).map(wrap_comparison_result),
                NotEqual => |l, r| neq(l, r).map(wrap_comparison_result),
                Distinct => |l, r| distinct(l, r).map(wrap_comparison_result),
                // NOTE: [Not]In was already covered above
                In | NotIn => return Err(Error::generic("Invalid expression given")),
            };

            eval(&left_arr, &right_arr).map_err(Error::generic_err)
        }
        (Variadic(VariadicExpression { op, exprs }), None | Some(&DataType::BOOLEAN)) => {
            type Operation = fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>;
            let (reducer, default): (Operation, _) = match op {
                VariadicOperator::And => (and_kleene, true),
                VariadicOperator::Or => (or_kleene, false),
            };
            exprs
                .iter()
                .map(|expr| evaluate_expression(expr, batch, result_type))
                .reduce(|l, r| {
                    Ok(reducer(downcast_to_bool(&l?)?, downcast_to_bool(&r?)?)
                        .map(wrap_comparison_result)?)
                })
                .unwrap_or_else(|| {
                    evaluate_expression(&Expression::literal(default), batch, result_type)
                })
        }
        (Variadic(_), _) => {
            // NOTE: Update this error message if we add support for variadic operations on other types
            Err(Error::Generic(format!(
                "Variadic {expression:?} is expected to return boolean results, got {result_type:?}"
            )))
        }
    }
}

// Apply a schema to an array. The array _must_ be a `StructArray`. Returns a `RecordBatch where the
// names of fields, nullable, and metadata in the struct have been transformed to match those in
// schema specified by `schema`
fn apply_schema(array: &dyn Array, schema: &DataType) -> DeltaResult<RecordBatch> {
    let DataType::Struct(struct_schema) = schema else {
        return Err(Error::generic(
            "apply_schema at top-level must be passed a struct schema",
        ));
    };
    let applied = apply_schema_to_struct(array, struct_schema)?;
    Ok(applied.into())
}

// helper to transform an arrow field+col into the specified target type. If `rename` is specified
// the field will be renamed to the contained `str`.
fn new_field_with_metadata(
    field_name: &str,
    data_type: &ArrowDataType,
    nullable: bool,
    metadata: Option<HashMap<String, String>>,
) -> ArrowField {
    let mut field = ArrowField::new(field_name, data_type.clone(), nullable);
    if let Some(metadata) = metadata {
        field.set_metadata(metadata);
    };
    field
}

// A helper that is a wrapper over `transform_field_and_col`. This will take apart the passed struct
// and use that method to transform each column and then put the struct back together. Target types
// and names for each column should be passed in `target_fields`. The number of elements in
// the `target_fields` iterator _must_ be the same as the number of columns in
// `struct_array`. The transformation is ordinal. That is, the order of fields in `target_fields`
// _must_ match the order of the columns in `struct_array`.
fn transform_struct(
    struct_array: &StructArray,
    target_fields: impl Iterator<Item = impl Borrow<StructField>>,
) -> DeltaResult<StructArray> {
    let (_, arrow_cols, nulls) = struct_array.clone().into_parts();
    let input_col_count = arrow_cols.len();
    let result_iter =
        arrow_cols
            .into_iter()
            .zip(target_fields)
            .map(|(sa_col, target_field)| -> DeltaResult<_> {
                let target_field = target_field.borrow();
                let transformed_col = apply_schema_to(&sa_col, target_field.data_type())?;
                let transformed_field = new_field_with_metadata(
                    &target_field.name,
                    transformed_col.data_type(),
                    target_field.nullable,
                    Some(target_field.metadata_with_string_values()),
                );
                Ok((transformed_field, transformed_col))
            });
    let (transformed_fields, transformed_cols): (Vec<ArrowField>, Vec<ArrayRef>) =
        result_iter.process_results(|iter| iter.unzip())?;
    if transformed_cols.len() != input_col_count {
        return Err(Error::InternalError(format!(
            "Passed struct had {input_col_count} columns, but transformed column has {}",
            transformed_cols.len()
        )));
    }
    Ok(StructArray::try_new(
        transformed_fields.into(),
        transformed_cols,
        nulls,
    )?)
}

// Transform a struct array. The data is in `array`, and the target fields are in `kernel_fields`.
fn apply_schema_to_struct(array: &dyn Array, kernel_fields: &Schema) -> DeltaResult<StructArray> {
    let Some(sa) = array.as_struct_opt() else {
        return Err(make_arrow_error(
            "Arrow claimed to be a struct but isn't a StructArray",
        ));
    };
    transform_struct(sa, kernel_fields.fields())
}

// deconstruct the array, then rebuild the mapped version
fn apply_schema_to_list(
    array: &dyn Array,
    target_inner_type: &ArrayType,
) -> DeltaResult<ListArray> {
    let Some(la) = array.as_list_opt() else {
        return Err(make_arrow_error(
            "Arrow claimed to be a list but isn't a ListArray",
        ));
    };
    let (field, offset_buffer, values, nulls) = la.clone().into_parts();

    let transformed_values = apply_schema_to(&values, &target_inner_type.element_type)?;
    let transformed_field = ArrowField::new(
        field.name(),
        transformed_values.data_type().clone(),
        target_inner_type.contains_null,
    );
    Ok(ListArray::try_new(
        Arc::new(transformed_field),
        offset_buffer,
        transformed_values,
        nulls,
    )?)
}

// deconstruct a map, and rebuild it with the specified target kernel type
fn apply_schema_to_map(array: &dyn Array, kernel_map_type: &MapType) -> DeltaResult<MapArray> {
    let Some(ma) = array.as_map_opt() else {
        return Err(make_arrow_error(
            "Arrow claimed to be a map but isn't a MapArray",
        ));
    };
    let (map_field, offset_buffer, map_struct_array, nulls, ordered) = ma.clone().into_parts();
    let target_fields = map_struct_array
        .fields()
        .iter()
        .zip([&kernel_map_type.key_type, &kernel_map_type.value_type])
        .zip([false, kernel_map_type.value_contains_null])
        .map(|((arrow_field, target_type), nullable)| {
            StructField::new(arrow_field.name(), target_type.clone(), nullable)
        });

    // Arrow puts the key type/val as the first field/col and the value type/val as the second. So
    // we just transform like a 'normal' struct, but we know there are two fields/cols and we
    // specify the key/value types as the target type iterator.
    let transformed_map_struct_array = transform_struct(&map_struct_array, target_fields)?;

    let transformed_map_field = ArrowField::new(
        map_field.name().clone(),
        transformed_map_struct_array.data_type().clone(),
        map_field.is_nullable(),
    );
    Ok(MapArray::try_new(
        Arc::new(transformed_map_field),
        offset_buffer,
        transformed_map_struct_array,
        nulls,
        ordered,
    )?)
}

// apply `schema` to `array`. This handles renaming, and adjusting nullability and metadata. if the
// actual data types don't match, this will return an error
fn apply_schema_to(array: &ArrayRef, schema: &DataType) -> DeltaResult<ArrayRef> {
    use DataType::*;
    let array: ArrayRef = match schema {
        Struct(stype) => Arc::new(apply_schema_to_struct(array, stype)?),
        Array(atype) => Arc::new(apply_schema_to_list(array, atype)?),
        Map(mtype) => Arc::new(apply_schema_to_map(array, mtype)?),
        _ => {
            ensure_data_types(schema, array.data_type(), true)?;
            array.clone()
        }
    };
    Ok(array)
}

#[derive(Debug)]
pub struct ArrowExpressionHandler;

impl ExpressionHandler for ArrowExpressionHandler {
    fn get_evaluator(
        &self,
        schema: SchemaRef,
        expression: Expression,
        output_type: DataType,
    ) -> Arc<dyn ExpressionEvaluator> {
        Arc::new(DefaultExpressionEvaluator {
            input_schema: schema,
            expression: Box::new(expression),
            output_type,
        })
    }
}

#[derive(Debug)]
pub struct DefaultExpressionEvaluator {
    input_schema: SchemaRef,
    expression: Box<Expression>,
    output_type: DataType,
}

impl ExpressionEvaluator for DefaultExpressionEvaluator {
    fn evaluate(&self, batch: &dyn EngineData) -> DeltaResult<Box<dyn EngineData>> {
        let batch = batch
            .any_ref()
            .downcast_ref::<ArrowEngineData>()
            .ok_or_else(|| Error::engine_data_type("ArrowEngineData"))?
            .record_batch();
        let _input_schema: ArrowSchema = self.input_schema.as_ref().try_into()?;
        // TODO: make sure we have matching schemas for validation
        // if batch.schema().as_ref() != &input_schema {
        //     return Err(Error::Generic(format!(
        //         "input schema does not match batch schema: {:?} != {:?}",
        //         input_schema,
        //         batch.schema()
        //     )));
        // };
        let array_ref = evaluate_expression(&self.expression, batch, Some(&self.output_type))?;
        let batch: RecordBatch = if let DataType::Struct(_) = self.output_type {
            apply_schema(&array_ref, &self.output_type)?
        } else {
            let array_ref = apply_schema_to(&array_ref, &self.output_type)?;
            let arrow_type: ArrowDataType = ArrowDataType::try_from(&self.output_type)?;
            let schema = ArrowSchema::new(vec![ArrowField::new("output", arrow_type, true)]);
            RecordBatch::try_new(Arc::new(schema), vec![array_ref])?
        };
        Ok(Box::new(ArrowEngineData::new(batch)))
    }
}
