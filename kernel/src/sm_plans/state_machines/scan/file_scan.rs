//! Scan-time SSA state machines and the shared data-phase projection helper.
//!
//! Hosts the `impl Scan { ... }` block that exposes the SSA-flavored coroutine state
//! machines for metadata-only and combined metadata + data scans. The actual SSA plan
//! body is built by [`super::ssa_scan::build_scan_ssa`].

use std::collections::HashSet;
use std::sync::Arc;

use super::ssa_scan::build_scan_ssa;
use crate::delta_error;
use crate::expressions::{col, Expression, ExpressionStructPatchBuilder};
use crate::scan::log_replay::FILE_CONSTANT_VALUES_NAME;
use crate::scan::state_info::StateInfo;
use crate::scan::transform_spec::{row_id_coalesce_expr, FieldTransformSpec};
use crate::scan::Scan;
use crate::schema::{DataType, MetadataColumnSpec, StructField};
use crate::sm_plans::errors::{DeltaError, DeltaErrorCode, DeltaResultExt};
use crate::sm_plans::ir::plan::ResultPlan;
use crate::sm_plans::state_machines::framework::coroutine::driver::CoroutineSM;
use crate::sm_plans::state_machines::framework::plan_context::Context as SsaContext;

// === SM entry points ======================================================================

impl Scan {
    /// CoroutineSM SM for metadata-only scan execution.
    ///
    /// Builds the canonical scan pipeline (FSR reconciliation + flat `scan_file_row`
    /// projection) as a single SSA program against the builder API and yields a single
    /// [`ResultPlan`]. Engines drive this through `drive_ssa_to_dataframe`.
    pub fn scan_metadata_state_machine(&self) -> Result<CoroutineSM<ResultPlan>, DeltaError> {
        let scan = self.clone();
        CoroutineSM::new("scan_metadata_ssa", move |mut engine, _sm_id| async move {
            let ctx = SsaContext::new();
            let live_actions = build_scan_ssa(
                &ctx,
                &mut engine,
                &scan,
                /* with_data= */ false,
                /* with_stats= */ false,
            )
            .await?;
            ctx.into_result_plan(live_actions)
        })
    }

    /// CoroutineSM SM for metadata-only scan execution that **retains per-file stats**.
    ///
    /// Identical to [`Self::scan_metadata_state_machine`] except the flat `scan_file_row`
    /// terminal appends a top-level `stats: STRUCT<physical_stats_schema>?` column (carrying the
    /// reconciled `add.stats_parsed`) so an engine-free consumer can collect a `path -> stats`
    /// map on the same SSA driver.
    ///
    /// The `stats` column only appears when the scan was built requesting struct stats
    /// (`ScanBuilder::with_stats(StatsOptions::all_struct())` — there is no `Scan`-level
    /// shortcut). Otherwise `physical_stats_schema()` is `None` and this is a no-op: no extra
    /// stats I/O, and the terminal is byte-identical to [`Self::scan_metadata_state_machine`].
    pub fn scan_stats_metadata_state_machine(&self) -> Result<CoroutineSM<ResultPlan>, DeltaError> {
        let scan = self.clone();
        CoroutineSM::new(
            "scan_stats_metadata_ssa",
            move |mut engine, _sm_id| async move {
                let ctx = SsaContext::new();
                let live_actions = build_scan_ssa(
                    &ctx,
                    &mut engine,
                    &scan,
                    /* with_data= */ false,
                    /* with_stats= */ true,
                )
                .await?;
                ctx.into_result_plan(live_actions)
            },
        )
    }

    /// CoroutineSM SM for combined metadata + data scan execution.
    ///
    /// Pipeline: shape-resolution yields, then reconciliation + flat scan-file
    /// projection + per-file Load + logical projection are appended into a single
    /// SSA program.
    pub fn scan_state_machine(&self) -> Result<CoroutineSM<ResultPlan>, DeltaError> {
        let scan = self.clone();
        CoroutineSM::new("scan_ssa", move |mut engine, _sm_id| async move {
            let ctx = SsaContext::new();
            let data = build_scan_ssa(
                &ctx,
                &mut engine,
                &scan,
                /* with_data= */ true,
                /* with_stats= */ false,
            )
            .await?;
            ctx.into_result_plan(data)
        })
    }
}

// === Data-phase projection ===============================================================

/// Build the per-logical-field projection list that converts raw Load output (physical schema +
/// broadcast file-constant struct + parquet-synthesized metadata columns) into the scan's
/// logical schema. Emits one expression per logical field, in order.
///
/// Classification is driven by [`StateInfo::transform_spec`] (the same source of truth as the
/// visitor path's `get_transform_expr`) plus each field's [`MetadataColumnSpec`]:
///
/// - Partition columns ([`FieldTransformSpec::MetadataDerivedColumn`]) ->
///   `fileConstantValues.partitionValues_parsed.<physical_name>`.
/// - `RowId` (paired with [`FieldTransformSpec::GenerateRowId`]) -> `coalesce(materialized,
///   baseRowId + row_index)` via [`row_id_coalesce_expr`], with `baseRowId` read from the
///   file-constant struct (vs. a literal in the visitor path).
/// - `RowIndex` -> `col([field.physical_name])`, the parquet-synthesized column under the user's
///   chosen name (not the kernel default `_metadata.row_index`).
/// - Regular fields -> `col([physical_name])`, except struct-typed fields which emit an identity
///   [`Expression::StructPatch`] for engines to lower to struct-reshape against the projection's
///   declared output schema.
///
/// Returns [`DeltaErrorCode::DeltaCommandInvariantViolation`] for shapes the SM data stage
/// doesn't (yet) support: [`FieldTransformSpec::DynamicColumn`] (CDF-only),
/// [`MetadataColumnSpec::FilePath`] (no DF synthesizer yet), and
/// [`MetadataColumnSpec::RowCommitVersion`] (rejected at `StateInfo::try_new`).
pub(super) fn scan_data_projection(
    state_info: &StateInfo,
) -> Result<Vec<Arc<Expression>>, DeltaError> {
    let logical_schema = &state_info.logical_schema;
    let column_mapping_mode = state_info.column_mapping_mode;

    // Index the spec once for O(1) per-field dispatch. Partition columns are keyed by
    // `field_index`; the at-most-one `GenerateRowId` entry feeds the `RowId` arm.
    // `StaticInsert` / `StaticDrop` don't affect the logical projection (the scan classifier
    // emits neither today, and `StaticDrop` targets columns this projection doesn't read).
    let mut partition_indices: HashSet<usize> = HashSet::new();
    let mut row_id_spec: Option<&FieldTransformSpec> = None;
    if let Some(spec) = state_info.transform_spec.as_deref() {
        for entry in spec {
            match entry {
                FieldTransformSpec::MetadataDerivedColumn { field_index, .. } => {
                    partition_indices.insert(*field_index);
                }
                FieldTransformSpec::GenerateRowId { .. } => row_id_spec = Some(entry),
                FieldTransformSpec::StaticInsert { .. } | FieldTransformSpec::StaticDrop { .. } => {
                }
                FieldTransformSpec::DynamicColumn { .. } => {
                    return Err(delta_error!(
                        DeltaErrorCode::DeltaCommandInvariantViolation,
                        "fsr::scan_data_projection: DynamicColumn entries are emitted only \
                         by the CDF classifier; the scan data stage does not run for CDF",
                    ));
                }
            }
        }
    }

    logical_schema
        .fields()
        .enumerate()
        .map(|(field_index, field)| {
            let expr = match field.get_metadata_column_spec() {
                Some(MetadataColumnSpec::RowIndex) => {
                    // Metadata columns aren't subject to column mapping, so `physical_name`
                    // returns the user-supplied field name -- the same name the parquet
                    // reader stamps onto the synthesized column.
                    col([field.physical_name(column_mapping_mode)])
                }
                Some(MetadataColumnSpec::RowId) => row_id_expr(field, row_id_spec)?,
                Some(
                    spec @ (MetadataColumnSpec::FilePath | MetadataColumnSpec::RowCommitVersion),
                ) => return Err(unsupported_metadata_column(field, spec)),
                None if partition_indices.contains(&field_index) => col([
                    FILE_CONSTANT_VALUES_NAME,
                    "partitionValues_parsed",
                    field.physical_name(column_mapping_mode),
                ]),
                None => {
                    // Struct fields wrap in an identity `StructPatch` so engines reshape to the
                    // declared output schema (logical names + nullability/order). Non-structs
                    // are renamed by the engine's per-field cast.
                    let physical_name = field.physical_name(column_mapping_mode);
                    if matches!(field.data_type(), DataType::Struct(_)) {
                        Expression::struct_patch(ExpressionStructPatchBuilder::new_nested([
                            physical_name,
                        ]))
                        .or_delta(DeltaErrorCode::DeltaCommandInvariantViolation)?
                    } else {
                        col([physical_name])
                    }
                }
            };
            Ok(expr.into())
        })
        .collect()
}

/// Build the `RowId` projection expression for `field`, matched against the spec's at-most-one
/// [`FieldTransformSpec::GenerateRowId`] entry. `baseRowId` is read from the per-file
/// file-constant struct (the SM pipeline doesn't know the value at projection-build time).
fn row_id_expr(
    field: &StructField,
    row_id_spec: Option<&FieldTransformSpec>,
) -> Result<Expression, DeltaError> {
    let Some(FieldTransformSpec::GenerateRowId {
        field_name,
        row_index_field_name,
    }) = row_id_spec
    else {
        return Err(delta_error!(
            DeltaErrorCode::DeltaCommandInvariantViolation,
            "fsr::scan_data_projection: RowId column `{}` has no GenerateRowId entry in \
             the transform spec",
            field.name(),
        ));
    };
    Ok(row_id_coalesce_expr(
        field_name,
        row_index_field_name,
        col([FILE_CONSTANT_VALUES_NAME, "baseRowId"]),
    ))
}

/// Typed error for metadata-column specs the scan SM data stage doesn't (yet) support; see
/// [`scan_data_projection`]'s doc for the full rationale.
fn unsupported_metadata_column(field: &StructField, spec: MetadataColumnSpec) -> DeltaError {
    delta_error!(
        DeltaErrorCode::DeltaCommandInvariantViolation,
        "fsr::scan_data_projection: metadata column `{}` ({:?}) is not supported in the \
         scan data stage yet",
        field.name(),
        spec,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::expressions::BinaryExpressionOp;
    use crate::scan::state_info::tests::{
        get_simple_state_info, get_state_info, ROW_TRACKING_FEATURES,
    };
    use crate::schema::{ColumnMetadataKey, MetadataValue, StructField, StructType};

    // === Helpers ==========================================================================

    /// Build a `StructField` annotated with the given physical name + column ID, suitable for
    /// `column_mapping_mode = Name`.
    fn annotated_field(
        logical_name: &str,
        physical_name: &str,
        column_id: i64,
        data_type: DataType,
        nullable: bool,
    ) -> StructField {
        StructField::new(logical_name, data_type, nullable).with_metadata([
            (
                ColumnMetadataKey::ColumnMappingPhysicalName.as_ref(),
                MetadataValue::String(physical_name.to_string()),
            ),
            (
                ColumnMetadataKey::ColumnMappingId.as_ref(),
                MetadataValue::Number(column_id),
            ),
        ])
    }

    /// Metadata configuration that turns on column mapping mode `Name`.
    fn cm_name_metadata() -> HashMap<String, String> {
        HashMap::from([("delta.columnMapping.mode".to_string(), "name".to_string())])
    }

    /// Metadata configuration required by `StateInfo::try_new` to accept a `RowId` metadata
    /// column, parameterised by the materialized row-id column name.
    fn row_tracking_metadata(materialized_row_id_col: &str) -> HashMap<String, String> {
        HashMap::from([
            ("delta.enableRowTracking".to_string(), "true".to_string()),
            (
                "delta.rowTracking.materializedRowIdColumnName".to_string(),
                materialized_row_id_col.to_string(),
            ),
            (
                "delta.rowTracking.materializedRowCommitVersionColumnName".to_string(),
                "some_row_commit_version_col".to_string(),
            ),
        ])
    }

    // === Tests ============================================================================

    /// Verifies the per-shape projection emitted for non-metadata, non-partition columns:
    ///
    /// - Primitive / list / map fields: a single top-level `col([physical_root])` reference.
    /// - Struct fields: an identity [`Expression::StructPatch`] rooted at the physical column.
    #[rstest::rstest]
    #[case::primitive(DataType::LONG, false)]
    #[case::nested_struct(DataType::Struct(Box::new(StructType::try_new(vec![
        annotated_field("inner1", "phys-inner1", 91, DataType::STRING, true),
        annotated_field("inner2", "phys-inner2", 92, DataType::INTEGER, true),
    ]).unwrap())), true)]
    fn scan_data_projection_emits_expected_shape(
        #[case] data_type: DataType,
        #[case] expect_transform: bool,
    ) {
        let schema = Arc::new(
            StructType::try_new(vec![annotated_field(
                "field", "col-phys", 1, data_type, true,
            )])
            .unwrap(),
        );
        let state_info =
            get_state_info(schema, vec![], None, &[], cm_name_metadata(), vec![]).unwrap();
        let exprs = scan_data_projection(&state_info).unwrap();
        assert_eq!(exprs.len(), 1);
        if expect_transform {
            let expected =
                Expression::struct_patch(ExpressionStructPatchBuilder::new_nested(["col-phys"]))
                    .unwrap();
            assert_eq!(exprs[0].as_ref(), &expected);
        } else {
            assert_eq!(exprs[0].as_ref(), &col(["col-phys"]));
        }
    }

    #[test]
    fn scan_data_projection_partition_column() {
        let schema = Arc::new(
            StructType::try_new(vec![
                StructField::nullable("id", DataType::STRING),
                StructField::nullable("date", DataType::DATE),
            ])
            .unwrap(),
        );
        let state_info = get_simple_state_info(schema, vec!["date".to_string()]).unwrap();
        let exprs = scan_data_projection(&state_info).unwrap();
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].as_ref(), &col(["id"]));
        assert_eq!(
            exprs[1].as_ref(),
            &col([FILE_CONSTANT_VALUES_NAME, "partitionValues_parsed", "date"]),
        );
    }

    #[test]
    fn scan_data_projection_file_path_errors() {
        let schema = Arc::new(
            StructType::try_new(vec![StructField::nullable("id", DataType::STRING)]).unwrap(),
        );
        let state_info = get_state_info(
            schema,
            vec![],
            None,
            &[],
            HashMap::new(),
            vec![("my_path", MetadataColumnSpec::FilePath)],
        )
        .unwrap();
        let err = scan_data_projection(&state_info)
            .expect_err("FilePath metadata column should not be projected today");
        assert_eq!(err.code, DeltaErrorCode::DeltaCommandInvariantViolation);
        assert!(
            err.message.contains("metadata column `my_path` (FilePath)"),
            "unexpected error message: {}",
            err.message,
        );
    }

    /// Regression test: the previous implementation always emitted `col(["_metadata.row_index"])`
    /// regardless of the metadata column's actual name.
    #[test]
    fn scan_data_projection_user_named_row_index() {
        let schema = Arc::new(
            StructType::try_new(vec![StructField::nullable("id", DataType::STRING)]).unwrap(),
        );
        let state_info = get_state_info(
            schema,
            vec![],
            None,
            &[],
            HashMap::new(),
            vec![("my_row_idx", MetadataColumnSpec::RowIndex)],
        )
        .unwrap();
        let exprs = scan_data_projection(&state_info).unwrap();
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].as_ref(), &col(["id"]));
        assert_eq!(exprs[1].as_ref(), &col(["my_row_idx"]));
    }

    /// Regression test: the previous implementation emitted `baseRowId +
    /// col("_metadata.row_index")` and ignored the materialized row-id column. Asserts the
    /// `coalesce(materialized, baseRowId + row_index)` form, with the classifier-synthesized
    /// row-index column name from the spec.
    #[test]
    fn scan_data_projection_row_id_synthesized_index() {
        let schema = Arc::new(
            StructType::try_new(vec![StructField::nullable("id", DataType::STRING)]).unwrap(),
        );
        let state_info = get_state_info(
            schema,
            vec![],
            None,
            ROW_TRACKING_FEATURES,
            row_tracking_metadata("some_row_id_col"),
            vec![("row_id", MetadataColumnSpec::RowId)],
        )
        .unwrap();
        let exprs = scan_data_projection(&state_info).unwrap();
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].as_ref(), &col(["id"]));
        let expected_row_id = Expression::coalesce([
            col(["some_row_id_col"]),
            Expression::binary(
                BinaryExpressionOp::Plus,
                col([FILE_CONSTANT_VALUES_NAME, "baseRowId"]),
                col(["row_indexes_for_row_id_0"]),
            ),
        ]);
        assert_eq!(exprs[1].as_ref(), &expected_row_id);
    }

    /// `RowId` reuses the user's `RowIndex` column when both are requested -- the
    /// `GenerateRowId` spec carries the same `row_index_field_name`. Previously both branches
    /// hardcoded the kernel default `_metadata.row_index`.
    #[test]
    fn scan_data_projection_row_id_with_explicit_index() {
        let schema = Arc::new(
            StructType::try_new(vec![StructField::nullable("id", DataType::STRING)]).unwrap(),
        );
        let state_info = get_state_info(
            schema,
            vec![],
            None,
            ROW_TRACKING_FEATURES,
            row_tracking_metadata("some_row_id_col"),
            vec![
                ("row_id", MetadataColumnSpec::RowId),
                ("row_index", MetadataColumnSpec::RowIndex),
            ],
        )
        .unwrap();
        let exprs = scan_data_projection(&state_info).unwrap();
        assert_eq!(exprs.len(), 3);
        assert_eq!(exprs[0].as_ref(), &col(["id"]));
        let expected_row_id = Expression::coalesce([
            col(["some_row_id_col"]),
            Expression::binary(
                BinaryExpressionOp::Plus,
                col([FILE_CONSTANT_VALUES_NAME, "baseRowId"]),
                col(["row_index"]),
            ),
        ]);
        assert_eq!(exprs[1].as_ref(), &expected_row_id);
        assert_eq!(exprs[2].as_ref(), &col(["row_index"]));
    }

    // === SM drive tests (real tables, SyncEngine snapshot) ================================

    mod drive {
        use std::collections::HashMap;

        use url::Url;

        use super::super::*;
        use crate::engine::arrow_conversion::TryIntoKernel;
        use crate::engine::sync::SyncEngine;
        use crate::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use crate::schema::SchemaRef;
        use crate::sm_plans::ir::nodes::FileType;
        use crate::sm_plans::ir::plan::{NodeKind, PlanNode};
        use crate::sm_plans::state_machines::framework::state_machine::{NextStep, StateMachine};
        use crate::sm_plans::state_machines::framework::step::EngineRequest;
        use crate::sm_plans::state_machines::framework::step_payload::EngineResponse;
        use crate::snapshot::Snapshot;
        use crate::SnapshotRef;

        fn snapshot_for(table_rel_path: &str) -> SnapshotRef {
            let path = std::fs::canonicalize(table_rel_path).unwrap();
            let url = Url::from_directory_path(path).unwrap();
            Snapshot::builder_for(url)
                .build(&SyncEngine::new())
                .unwrap()
        }

        /// Answer a footer [`EngineRequest::SchemaQuery`] by actually reading the parquet
        /// footer of the referenced file, mirroring what a real engine executor does.
        fn footer_schema(file_path: &str) -> SchemaRef {
            let path = Url::parse(file_path).unwrap().to_file_path().unwrap();
            let file = std::fs::File::open(path).unwrap();
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
            let kernel_schema: crate::schema::StructType =
                builder.schema().as_ref().try_into_kernel().unwrap();
            Arc::new(kernel_schema)
        }

        /// Drive `sm` through the get_step/submit loop to completion, answering schema
        /// queries with real footer reads. Panics on `Consume` steps -- callers pick tables
        /// whose pipelines resolve without consumer drains. Returns the terminal result and
        /// the number of schema queries answered.
        fn drive_schema_only<R>(sm: &mut CoroutineSM<R>) -> (R, usize) {
            let mut answered = 0;
            loop {
                let resp = match sm.get_step() {
                    Ok(EngineRequest::SchemaQuery(q)) => {
                        answered += 1;
                        EngineResponse::Schema(footer_schema(&q.file_path))
                    }
                    Ok(EngineRequest::Consume { .. }) => {
                        panic!("test table should not require consume steps")
                    }
                    // Zero-yield SM: no pending operation; submit the priming value to
                    // collect the terminal result.
                    Err(_) => EngineResponse::Empty,
                };
                match sm.submit(Ok(resp)).unwrap() {
                    NextStep::Continue => continue,
                    NextStep::Done(r) => return (r, answered),
                }
            }
        }

        /// Collect `(file_type, file paths)` for every file source in the plan. Sources appear
        /// either as direct `Scan` nodes (files inline) or as `Values(path rows) -> Load` chains
        /// (the reconciliation's file-listing shape).
        fn scan_sources(result: &ResultPlan) -> Vec<(FileType, Vec<String>)> {
            let by_output: HashMap<_, _> = result
                .plan
                .stmts
                .iter()
                .map(|node| (node.output, node))
                .collect();
            let values_paths = |node: &PlanNode| -> Vec<String> {
                let NodeKind::Values(values) = &node.kind else {
                    return Vec::new();
                };
                let Some(path_idx) = values
                    .schema
                    .fields()
                    .position(|f| f.name() == "path" || f.name() == "file_path")
                else {
                    return Vec::new();
                };
                values
                    .rows
                    .iter()
                    .filter_map(|row| match &row[path_idx] {
                        crate::expressions::Scalar::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect()
            };
            result
                .plan
                .stmts
                .iter()
                .filter_map(|node| match &node.kind {
                    NodeKind::Scan(scan) => Some((
                        scan.file_type,
                        scan.files.iter().map(|f| f.location.to_string()).collect(),
                    )),
                    NodeKind::Load(load) => {
                        let paths: Vec<String> = node
                            .inputs
                            .iter()
                            .filter_map(|r| by_output.get(r))
                            .flat_map(|n| values_paths(n))
                            .collect();
                        Some((load.file_type, paths))
                    }
                    _ => None,
                })
                .collect()
        }

        #[test]
        fn scan_metadata_sm_commit_only_table_yields_plan_over_commit_files() {
            let snapshot = snapshot_for("./tests/data/basic_partitioned");
            let scan = snapshot.scan_replay_builder().build_replay().unwrap();
            let mut sm = scan.scan_metadata_state_machine().unwrap();

            let (result, schema_queries) = drive_schema_only(&mut sm);
            assert_eq!(
                schema_queries, 0,
                "commit-only table must not need footer probes"
            );

            // The reconciliation must source exactly the snapshot's two commit files as JSON.
            let sources = scan_sources(&result);
            let json_files: Vec<_> = sources
                .iter()
                .filter(|(ft, _)| *ft == FileType::Json)
                .flat_map(|(_, files)| files.iter())
                .collect();
            assert!(
                json_files
                    .iter()
                    .any(|f| f.ends_with("00000000000000000000.json"))
                    && json_files
                        .iter()
                        .any(|f| f.ends_with("00000000000000000001.json")),
                "expected both commit files in the plan, got: {json_files:?}"
            );
            assert!(
                !sources.iter().any(|(ft, _)| *ft == FileType::Parquet),
                "commit-only table must not scan parquet checkpoints"
            );
        }

        #[test]
        fn scan_metadata_sm_checkpoint_table_probes_footer_and_scans_checkpoint() {
            let snapshot = snapshot_for("./tests/data/with_checkpoint_no_last_checkpoint");
            let scan = snapshot.scan_replay_builder().build_replay().unwrap();
            let mut sm = scan.scan_metadata_state_machine().unwrap();

            let (result, schema_queries) = drive_schema_only(&mut sm);
            assert!(
                schema_queries >= 1,
                "checkpoint without a usable hint schema requires a footer probe"
            );

            let sources = scan_sources(&result);
            let parquet_files: Vec<_> = sources
                .iter()
                .filter(|(ft, _)| *ft == FileType::Parquet)
                .flat_map(|(_, files)| files.iter())
                .collect();
            assert!(
                parquet_files
                    .iter()
                    .any(|f| f.ends_with("00000000000000000002.checkpoint.parquet")),
                "expected the checkpoint parquet in the plan, got: {parquet_files:?}"
            );
            // Commits after the checkpoint must still be replayed as JSON.
            let json_files: Vec<_> = sources
                .iter()
                .filter(|(ft, _)| *ft == FileType::Json)
                .flat_map(|(_, files)| files.iter())
                .collect();
            assert!(
                json_files
                    .iter()
                    .any(|f| f.ends_with("00000000000000000003.json")),
                "expected post-checkpoint commit in the plan, got: {json_files:?}"
            );
        }
    }
}
