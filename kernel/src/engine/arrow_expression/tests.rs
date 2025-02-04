use std::ops::{Add, Div, Mul, Sub};

use arrow_array::{GenericStringArray, Int32Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field, Fields, Schema};

use super::*;
use crate::expressions::*;
use crate::schema::ArrayType;
use crate::DataType as DeltaDataTypes;

#[test]
fn test_array_column() {
    let values = Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 6, 9]));
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let arr_field = Arc::new(Field::new("item", DataType::List(field.clone()), true));

    let schema = Schema::new([arr_field.clone()]);

    let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

    let not_op = Expression::binary(BinaryOperator::NotIn, 5, column_expr!("item"));
    let result = evaluate_expression(&not_op, &batch, None).unwrap();
    let expected = BooleanArray::from(vec![true, false, true]);
    assert_eq!(result.as_ref(), &expected);

    let in_op = Expression::binary(BinaryOperator::In, 5, column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, None).unwrap();
    let in_expected = BooleanArray::from(vec![false, true, false]);
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_bad_right_type_array() {
    let values = Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();

    let in_op = Expression::binary(BinaryOperator::NotIn, 5, column_expr!("item"));

    let in_result = evaluate_expression(&in_op, &batch, None);

    assert!(in_result.is_err());
    assert_eq!(
        in_result.unwrap_err().to_string(),
        "Invalid expression evaluation: Cannot cast to list array: Int32"
    );
}

#[test]
fn test_literal_type_array_empty() {
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::new_empty(Arc::new(schema));

    let in_op = Expression::binary(
        BinaryOperator::NotIn,
        5,
        Scalar::Array(ArrayData::new(
            ArrayType::new(DeltaDataTypes::INTEGER, false),
            vec![Scalar::Integer(1), Scalar::Integer(2)],
        )),
    );

    let in_result = evaluate_expression(&in_op, &batch, None).unwrap();
    let in_expected = BooleanArray::from(Vec::<Option<bool>>::new());
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_invalid_array_sides() {
    let values = Int32Array::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 6, 9]));
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let arr_field = Arc::new(Field::new("item", DataType::List(field.clone()), true));

    let schema = Schema::new([arr_field.clone()]);

    let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

    let in_op = Expression::binary(
        BinaryOperator::NotIn,
        column_expr!("item"),
        column_expr!("item"),
    );

    let in_result = evaluate_expression(&in_op, &batch, None);

    assert!(in_result.is_err());
    assert_eq!(
        in_result.unwrap_err().to_string(),
        "Invalid expression evaluation: Invalid right value for (NOT) IN comparison, left is: Column(item) right is: Column(item)".to_string()
    )
}

#[test]
fn test_str_arrays() {
    let values = GenericStringArray::<i32>::from(vec![
        "hi", "bye", "hi", "hi", "bye", "bye", "hi", "bye", "hi",
    ]);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 3, 6, 9]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let arr_field = Arc::new(Field::new("item", DataType::List(field.clone()), true));
    let schema = Schema::new([arr_field.clone()]);
    let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

    let str_not_op = Expression::binary(BinaryOperator::NotIn, "bye", column_expr!("item"));

    let str_in_op = Expression::binary(BinaryOperator::In, "hi", column_expr!("item"));

    let result = evaluate_expression(&str_in_op, &batch, None).unwrap();
    let expected = BooleanArray::from(vec![true, true, true]);
    assert_eq!(result.as_ref(), &expected);

    let in_result = evaluate_expression(&str_not_op, &batch, None).unwrap();
    let in_expected = BooleanArray::from(vec![false, false, false]);
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_str_arrays_with_null() {
    let values = GenericStringArray::<i32>::from(vec![
        Some("one"),
        None,
        Some("two"),
        None,
        Some("one"),
        Some("two"),
    ]);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6]));
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let arr_field = Arc::new(Field::new("item", DataType::List(field.clone()), true));
    let schema = Schema::new([arr_field.clone()]);
    let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

    let in_op = Expression::binary(BinaryOperator::In, "one", column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(true), None, Some(true)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(BinaryOperator::In, "two", column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, Some(true), Some(true)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(BinaryOperator::In, "three", column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, None, Some(false)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(
        BinaryOperator::In,
        Scalar::Null(DeltaDataTypes::STRING),
        column_expr!("item"),
    );
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, None, None]);
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_arrays_with_null() {
    let values = Int32Array::from(vec![Some(1), None, Some(2), None, Some(1), Some(2)]);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 4, 6]));
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let arr_field = Arc::new(Field::new("item", DataType::List(field.clone()), true));
    let schema = Schema::new([arr_field.clone()]);
    let array = ListArray::new(field.clone(), offsets, Arc::new(values), None);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array.clone())]).unwrap();

    let in_op = Expression::binary(BinaryOperator::In, 1, column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(true), None, Some(true)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(BinaryOperator::In, 2, column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, Some(true), Some(true)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(BinaryOperator::In, 3, column_expr!("item"));
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, None, Some(false)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_op = Expression::binary(
        BinaryOperator::In,
        Scalar::Null(DeltaDataTypes::INTEGER),
        column_expr!("item"),
    );
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, None, None]);
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_column_in_array() {
    let values = Int32Array::from(vec![0, 1, 2, 3]);
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Integer.into(), false),
        [Scalar::Integer(1), Scalar::Integer(3)],
    )));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();

    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![false, true, false, true]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let not_in_op = Expression::binary(BinaryOperator::NotIn, column_expr!("item"), rhs);
    let not_in_result =
        evaluate_expression(&not_in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let not_in_expected = BooleanArray::from(vec![true, false, true, false]);
    assert_eq!(not_in_result.as_ref(), &not_in_expected);

    let in_expected = BooleanArray::from(vec![false, true, false, true]);

    // Date arrays
    let values = Date32Array::from(vec![0, 1, 2, 3]);
    let field = Arc::new(Field::new("item", DataType::Date32, true));
    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Date.into(), false),
        [Scalar::Date(1), Scalar::Date(3)],
    )));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);

    // Timestamp arrays
    let values = TimestampMicrosecondArray::from(vec![0, 1, 2, 3]).with_timezone("UTC");
    let field = Arc::new(Field::new(
        "item",
        (&DeltaDataTypes::TIMESTAMP).try_into().unwrap(),
        true,
    ));
    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Timestamp.into(), false),
        [Scalar::Timestamp(1), Scalar::Timestamp(3)],
    )));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);

    // Timestamp NTZ arrays
    let values = TimestampMicrosecondArray::from(vec![0, 1, 2, 3]);
    let field = Arc::new(Field::new(
        "item",
        (&DeltaDataTypes::TIMESTAMP_NTZ).try_into().unwrap(),
        true,
    ));
    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::TimestampNtz.into(), false),
        [Scalar::TimestampNtz(1), Scalar::TimestampNtz(3)],
    )));
    let schema = Schema::new([field.clone()]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(values.clone())]).unwrap();
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_column_in_array_with_null() {
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let values = Int32Array::from(vec![Some(1), Some(2), None]);
    let schema = Schema::new([field.clone()]);
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values.clone())]).unwrap();

    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Integer.into(), true),
        [Scalar::Integer(1), Scalar::Null(DeltaDataTypes::INTEGER)],
    )));

    // item IN (1, NULL) -- TRUE, NULL, NULL
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(true), None, None]);
    assert_eq!(in_result.as_ref(), &in_expected);

    // 1 IN (1, NULL) -- TRUE
    let in_op = Expression::binary(BinaryOperator::In, Scalar::Integer(1), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(true), Some(true), Some(true)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    // 1 NOT IN (1, NULL) -- FALSE
    let in_op = Expression::binary(BinaryOperator::NotIn, Scalar::Integer(1), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(false), Some(false), Some(false)]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Integer.into(), true),
        [Scalar::Integer(2), Scalar::Null(DeltaDataTypes::INTEGER)],
    )));

    // item IN (2, NULL) -- NULL, TRUE, NULL
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![None, Some(true), None]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_expected = BooleanArray::from(vec![None, None, None]);

    // 1 IN (2, NULL) -- NULL
    let in_op = Expression::binary(BinaryOperator::In, Scalar::Integer(1), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);

    // 1 NOT IN (2, NULL) -- NULL
    let in_op = Expression::binary(BinaryOperator::NotIn, Scalar::Integer(1), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);

    let rhs = Expression::literal(Scalar::Array(ArrayData::new(
        ArrayType::new(PrimitiveType::Integer.into(), true),
        [Scalar::Integer(1), Scalar::Integer(2)],
    )));

    // item IN (1, 2) -- TRUE, TRUE, NULL
    let in_op = Expression::binary(BinaryOperator::In, column_expr!("item"), rhs.clone());
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let in_expected = BooleanArray::from(vec![Some(true), Some(true), None]);
    assert_eq!(in_result.as_ref(), &in_expected);

    let in_expected = BooleanArray::from(vec![None, None, None]);

    // NULL IN (1, 2) -- NULL
    let in_op = Expression::binary(
        BinaryOperator::In,
        Scalar::Null(DeltaDataTypes::INTEGER),
        rhs.clone(),
    );
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);

    // NULL NOT IN (1, 2) -- NULL
    let in_op = Expression::binary(
        BinaryOperator::NotIn,
        Scalar::Null(DeltaDataTypes::INTEGER),
        rhs.clone(),
    );
    let in_result = evaluate_expression(&in_op, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    assert_eq!(in_result.as_ref(), &in_expected);
}

#[test]
fn test_extract_column() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let values = Int32Array::from(vec![1, 2, 3]);
    let batch =
        RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values.clone())]).unwrap();
    let column = column_expr!("a");

    let results = evaluate_expression(&column, &batch, None).unwrap();
    assert_eq!(results.as_ref(), &values);

    let schema = Schema::new(vec![Field::new(
        "b",
        DataType::Struct(Fields::from(vec![Field::new("a", DataType::Int32, false)])),
        false,
    )]);

    let struct_values: ArrayRef = Arc::new(values.clone());
    let struct_array = StructArray::from(vec![(
        Arc::new(Field::new("a", DataType::Int32, false)),
        struct_values,
    )]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(struct_array.clone())],
    )
    .unwrap();
    let column = column_expr!("b.a");
    let results = evaluate_expression(&column, &batch, None).unwrap();
    assert_eq!(results.as_ref(), &values);
}

#[test]
fn test_binary_op_scalar() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let values = Int32Array::from(vec![1, 2, 3]);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();
    let column = column_expr!("a");

    let expression = column.clone().add(1);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![2, 3, 4]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().sub(1);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![0, 1, 2]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().mul(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![2, 4, 6]));
    assert_eq!(results.as_ref(), expected.as_ref());

    // TODO handle type casting
    let expression = column.div(1);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![1, 2, 3]));
    assert_eq!(results.as_ref(), expected.as_ref())
}

#[test]
fn test_binary_op() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, false),
    ]);
    let values = Int32Array::from(vec![1, 2, 3]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(values.clone()), Arc::new(values)],
    )
    .unwrap();
    let column_a = column_expr!("a");
    let column_b = column_expr!("b");

    let expression = column_a.clone().add(column_b.clone());
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![2, 4, 6]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column_a.clone().sub(column_b.clone());
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![0, 0, 0]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column_a.clone().mul(column_b);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(Int32Array::from(vec![1, 4, 9]));
    assert_eq!(results.as_ref(), expected.as_ref());
}

#[test]
fn test_binary_cmp() {
    let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
    let values = Int32Array::from(vec![1, 2, 3]);
    let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(values)]).unwrap();
    let column = column_expr!("a");

    let expression = column.clone().lt(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, false, false]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().lt_eq(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, true, false]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().gt(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![false, false, true]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().gt_eq(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![false, true, true]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().eq(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![false, true, false]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column.clone().ne(2);
    let results = evaluate_expression(&expression, &batch, None).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, false, true]));
    assert_eq!(results.as_ref(), expected.as_ref());
}

#[test]
fn test_logical() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Boolean, false),
        Field::new("b", DataType::Boolean, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![
            Arc::new(BooleanArray::from(vec![true, false])),
            Arc::new(BooleanArray::from(vec![false, true])),
        ],
    )
    .unwrap();
    let column_a = column_expr!("a");
    let column_b = column_expr!("b");

    let expression = column_a.clone().and(column_b.clone());
    let results = evaluate_expression(&expression, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![false, false]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column_a.clone().and(true);
    let results = evaluate_expression(&expression, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, false]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column_a.clone().or(column_b);
    let results = evaluate_expression(&expression, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, true]));
    assert_eq!(results.as_ref(), expected.as_ref());

    let expression = column_a.clone().or(false);
    let results = evaluate_expression(&expression, &batch, Some(&DeltaDataTypes::BOOLEAN)).unwrap();
    let expected = Arc::new(BooleanArray::from(vec![true, false]));
    assert_eq!(results.as_ref(), expected.as_ref());
}
