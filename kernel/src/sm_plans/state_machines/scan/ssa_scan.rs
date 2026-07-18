//! SSA-flavored scan plan builder.
//!
//! Mirrors [`super::file_scan::Scan::build_plans`] / [`super::file_scan::Scan::do_data_stage`]
//! but builds against the [`super::super::framework::plan_context`] (`Context` / `PlanBuilder`) API
//! and the SSA IR. The scan-side terminal projection (reconciled rows -> flat `scan_file_row`)
//! and the data-phase Load + logical projection are expressed as builder chains; reconciliation
//! upstream of the terminal is shared with FSR via [`super::ssa_reconciliation`].
//!
//! After PR8 deletes the legacy registry-based pipeline, the SMs in [`super::file_scan`]
//! will be retired in favor of the SSA SMs declared on `Scan` here.

use std::sync::Arc;

use super::file_scan::scan_data_projection;
use super::ssa_reconciliation::{
    execute_reconciliation_ssa, retention_timestamps, scan_file_dedup_key, SCAN_BASE,
};
use crate::actions::deletion_vector::DeletionVectorDescriptor;
use crate::actions::ADD_NAME;
use crate::expressions::{col, ColumnName, Expression, Predicate};
use crate::scan::data_skipping::as_ssa_add_stats_skipping_predicate;
use crate::scan::log_replay::FILE_CONSTANT_VALUES_NAME;
use crate::scan::{PhysicalPredicate, Scan};
use crate::schema::{DataType, MapType, SchemaRef, StructField, StructType, ToSchema};
use crate::sm_plans::errors::DeltaError;
use crate::sm_plans::ir::nodes::{default_scan_file_columns, DvRef, FileType};
use crate::sm_plans::state_machines::framework::coroutine::context::Engine;
use crate::sm_plans::state_machines::framework::plan_context::{Context, LoadSpec, PlanBuilder};

// ============================================================================
// SSA scan builder
// ============================================================================

/// Build the SSA scan pipeline against `ctx`. Mirrors [`Scan::build_plans`] -- runs the
/// shared SSA reconciliation, then projects to the flat scan-file row shape and (when
/// `with_data`) appends the data phase.
///
/// Returns a [`PlanBuilder`] terminating on either:
/// - the live-actions stream (`with_data == false`), or
/// - the per-row data stream after parquet Load + logical-schema projection (`with_data == true`).
///
/// The caller wraps the returned builder with [`Context::into_result_plan`] to mint the
/// SM's terminal value.
pub(super) async fn build_scan_ssa(
    ctx: &Context,
    engine: &mut Engine,
    scan: &Scan,
    with_data: bool,
    with_stats: bool,
) -> Result<PlanBuilder, DeltaError> {
    let stats = scan.physical_stats_schema();
    // When `with_stats`, the flat terminal retains `add.stats_parsed` as a top-level `stats`
    // column (see `project_scan_file_row`). Capture the schema here: `stats` is *moved* into
    // `execute_reconciliation_ssa` below, so the terminal needs its own copy. `None` when stats
    // were not requested (`physical_stats_schema()` is `None`), which keeps the terminal
    // byte-identical to the plain metadata SM.
    let terminal_stats = if with_stats { stats.clone() } else { None };
    // The data stage needs the full partition schema (see `data_stage_partition_schema`
    // doc on [`Scan`]); the predicate-narrowed `physical_partition_schema()` only works
    // for metadata-only execution.
    let parts = if with_data {
        scan.data_stage_partition_schema()
    } else {
        scan.physical_partition_schema()
    };

    // The data stage feeds the flat row into a parquet `Load` whose column-mapping
    // rename matches on *physical* names, so it must keep physical naming. Metadata-only
    // terminals are user-facing (log inspection) and expose the `stats` /
    // `partitionValues_parsed` sub-structs with *logical* (table-facing) leaf names — the
    // values still come from the physical `add.stats_parsed` / `add.partitionValues_parsed`,
    // positionally, so only the declared field names change. See [`NameSpace`]. Resolve the
    // terminal's declared schemas here so the terminal itself is naming-agnostic.
    let name_space = if with_data {
        NameSpace::Physical
    } else {
        NameSpace::Logical
    };
    let terminal_parts = parts
        .as_ref()
        .map(|p| relabel_partitions(scan, p, name_space));
    let terminal_stats = terminal_stats
        .as_ref()
        .map(|s| relabel_stats(scan, s, name_space));

    // === Stages 1-5: shared reconciliation -> reconciled builder =========================
    let snapshot = scan.snapshot().as_ref();
    let reconciled = execute_reconciliation_ssa(
        ctx,
        engine,
        snapshot.log_segment(),
        &SCAN_BASE,
        stats,
        parts.clone(),
        Arc::new(scan_file_dedup_key()),
        Some(retention_timestamps(snapshot)?),
    )
    .await?;

    // === Data-skipping: prune the live-add rows by their per-file stats ================
    // Engine-free / declarative: when the scan carries a predicate and stats are present, append a
    // `FilterNode` over `add.stats_parsed` (evaluated lazily by the same engine that runs every
    // other reconciliation filter). Stats-only (min/max/nullCount); a no-predicate scan is
    // byte-identical to before. See `apply_data_skipping_ssa`.
    let reconciled = apply_data_skipping_ssa(reconciled, scan)?;

    // === Scan-specific terminal projection: reconciled -> flat scan_file_row ==========
    let live_actions =
        project_scan_file_row(reconciled, terminal_parts.as_ref(), terminal_stats.as_ref())?;

    // === Stage 6 (optional): data phase =================================================
    if with_data {
        do_data_stage_ssa(scan, live_actions)
    } else {
        Ok(live_actions)
    }
}

/// Append a data-skipping [`crate::sm_plans::ir::nodes::FilterNode`] over the reconciled live-add
/// rows, keying on `add.stats_parsed.{minValues,maxValues,nullCount}` / `numRecords`.
///
/// This is the `sm_plans` (engine-free, declarative) counterpart to the classic scan path's
/// [`crate::scan::log_replay`] `DataSkippingFilter`: instead of an eager evaluator wrapper, it adds
/// a plan node the SSA executor evaluates lazily like every other reconciliation filter, so it works
/// on wasm / async without an extra blocking evaluation path.
///
/// Behavior by predicate state (read directly off `StateInfo`, since `Scan::physical_predicate()`
/// collapses `StaticSkipAll` to `None`):
/// - `PhysicalPredicate::None` → no filter (byte-identical to the pre-skipping plan);
/// - `PhysicalPredicate::StaticSkipAll` → a constant-`false` filter (the predicate can never hold,
///   so no file survives) — mirrors the classic path's skip-all short-circuit;
/// - `PhysicalPredicate::Some(pred, _)` → the stats-only skipping predicate from
///   [`as_ssa_add_stats_skipping_predicate`], or no filter when the predicate is not eligible for
///   data skipping (conservative: keep every file).
fn apply_data_skipping_ssa(
    reconciled: PlanBuilder,
    scan: &Scan,
) -> Result<PlanBuilder, DeltaError> {
    let state_info = scan.state_info();
    match &state_info.physical_predicate {
        PhysicalPredicate::None => Ok(reconciled),
        PhysicalPredicate::StaticSkipAll => reconciled.filter(Predicate::literal(false)),
        PhysicalPredicate::Some(predicate, _) => {
            match as_ssa_add_stats_skipping_predicate(predicate, &state_info.physical_stats_columns)
            {
                Some(skipping) => reconciled.filter(skipping),
                None => Ok(reconciled),
            }
        }
    }
}

/// Append the SSA data stage onto the live-actions builder and return the data-stream
/// builder. Mirrors [`Scan::do_data_stage`].
///
/// Splits per-file: the upstream builder emits one row per surviving file (flat
/// `scan_file_row` shape), and [`PlanBuilder::load`] expands each row into the file's
/// per-record stream while broadcasting `path` and the `fileConstantValues` struct via
/// `passthrough_columns`. The trailing projection translates physical -> logical column
/// names per the scan's column-mapping mode.
fn do_data_stage_ssa(scan: &Scan, live_actions: PlanBuilder) -> Result<PlanBuilder, DeltaError> {
    let logical_schema = scan.logical_schema().clone();
    let logical_projection = scan_data_projection(scan.state_info())?;

    // Per-file parquet read; broadcasts the file-constant struct and `path` to every
    // emitted record-row. Output schema = physical_schema ++ {path, fileConstantValues}.
    let raw_data = live_actions.load(LoadSpec {
        file_schema: scan.physical_schema().clone(),
        file_type: FileType::Parquet,
        base_url: Some(scan.snapshot().table_root().clone()),
        passthrough_columns: vec![
            ColumnName::new([FILE_CONSTANT_VALUES_NAME]),
            ColumnName::new(["path"]),
        ],
        file_meta: default_scan_file_columns(),
        dv_ref: Some(DvRef::skip(ColumnName::new(["deletionVector"]))),
    })?;

    // Surviving projection: physical -> logical (column-mapping rename + metadata-column
    // synthesis). The kernel evaluator validates the expressions against the declared
    // `logical_schema` at lower time -- inference here would be insufficient (Transform
    // / RowId coalesce / etc. fall outside narrow inference's supported set).
    raw_data.project_with_schema(logical_projection, logical_schema)
}

// ============================================================================
// Scan terminal: reconciled builder -> flat scan_file_row
// ============================================================================

/// Project the reconciled action stream into the flat `scan_file_row` shape consumed by
/// the SSA scan data stage:
///
/// ```text
/// {
///   path: STRING NOT NULL,
///   size: LONG NOT NULL,
///   deletionVector: DV?,
///   fileConstantValues: STRUCT<
///     baseRowId: LONG?,
///     defaultRowCommitVersion: LONG?,
///     tags: MAP<STRING,STRING>?,
///     clusteringProvider: STRING?,
///     partitionValues_parsed?: STRUCT<...>,  // present iff `partitions` is Some
///   >?,
///   stats?: STRUCT<...>,  // present iff `physical_stats_schema` is Some (retains add.stats_parsed)
/// }
/// ```
///
/// **Stats:** when `physical_stats_schema` is `Some`, a top-level `stats` column (a *sibling* of
/// `fileConstantValues`, not nested) is appended, carrying the reconciled `add.stats_parsed`
/// struct verbatim (physical leaf names). It is `None` unless the scan was built requesting struct
/// stats (`StatsOptions::all_struct()` / `struct_columns`), in which case the terminal is
/// byte-identical to the four-field shape. The metadata-only SM
/// (`scan_stats_metadata_state_machine`) passes it through; the data-stage SMs pass `None`.
///
/// **Invariant:** when `partitions` is `Some(parts)`, the upstream reconciliation pipeline
/// must have already replaced `add.partitionValues` with `add.partitionValues_parsed`
/// (see `ssa_reconciliation::ReconciliationPlanBuilder::with_partitions_parsed`). The terminal
/// reads `col(["add", "partitionValues_parsed"])` directly -- no per-row
/// `map_to_struct(add.partitionValues)` here. The raw Map form has no downstream consumer
/// in the SSA path, so it is omitted from `fileConstantValues` entirely (parsing happens
/// once upstream, not per file row in the data phase).
fn project_scan_file_row(
    builder: PlanBuilder,
    partitions: Option<&SchemaRef>,
    stats_schema: Option<&SchemaRef>,
) -> Result<PlanBuilder, DeltaError> {
    let schema = scan_file_row_schema_parts(partitions, stats_schema);

    // Projection expressions read the *physical* action columns regardless of how `schema` names
    // its fields; `project_with_schema` supplies the output field names positionally (so a
    // logically-relabeled `schema` only renames — it never changes which value each field carries).
    let mut file_constant_exprs: Vec<Arc<Expression>> = vec![
        col([ADD_NAME, "baseRowId"]).into(),
        col([ADD_NAME, "defaultRowCommitVersion"]).into(),
        col([ADD_NAME, "tags"]).into(),
        col([ADD_NAME, "clusteringProvider"]).into(),
    ];
    if partitions.is_some() {
        file_constant_exprs.push(col([ADD_NAME, "partitionValues_parsed"]).into());
    }
    let mut exprs: Vec<Arc<Expression>> = vec![
        col([ADD_NAME, "path"]).into(),
        col([ADD_NAME, "size"]).into(),
        col([ADD_NAME, "deletionVector"]).into(),
        Arc::new(Expression::struct_from(file_constant_exprs)),
    ];
    if stats_schema.is_some() {
        exprs.push(col([ADD_NAME, "stats_parsed"]).into());
    }

    builder.project_with_schema(exprs, schema)
}

/// Whether the flat `scan_file_row` terminal declares its `stats` / `partitionValues_parsed`
/// sub-structs with **physical** (on-disk) or **logical** (table-facing) leaf names.
///
/// The data-stage terminal must be [`Physical`](Self::Physical): the downstream parquet `Load`
/// projection matches its column-mapping rename on physical names. The metadata-only terminals are
/// user-facing (log inspection) and use [`Logical`](Self::Logical) so callers see table column
/// names, not `col-…` / field-id-mangled physical names. Naming is the *only* difference — the
/// projection reads the same physical `add.*` columns either way, so `Physical` reproduces the
/// historical terminal byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameSpace {
    Physical,
    Logical,
}

/// Compute the flat `scan_file_row` schema the **metadata** terminals emit, without driving the SM.
///
/// The single source of truth behind [`Scan::scan_file_row_schema`]. Uses the metadata-terminal
/// inputs: the predicate-narrowed `physical_partition_schema()`, `physical_stats_schema()` (so the
/// `stats` sibling is present iff the scan requested struct stats — matching
/// `scan_stats_metadata_state_machine`), relabeled to [`NameSpace::Logical`] to match the terminal.
///
/// [`Scan::scan_file_row_schema`]: crate::scan::Scan::scan_file_row_schema
pub(crate) fn scan_file_row_schema(scan: &Scan) -> SchemaRef {
    let parts = scan
        .physical_partition_schema()
        .map(|p| relabel_partitions(scan, &p, NameSpace::Logical));
    let stats = scan
        .physical_stats_schema()
        .map(|s| relabel_stats(scan, &s, NameSpace::Logical));
    scan_file_row_schema_parts(parts.as_ref(), stats.as_ref())
}

/// Assemble the flat `scan_file_row` output [`StructType`] — the single source of truth shared by
/// the terminal projection ([`project_scan_file_row`]) and the public
/// [`Scan::scan_file_row_schema`] accessor, so the declared schema and the projected columns cannot
/// drift.
///
/// Pure function of the (already name-resolved) sub-schemas: `partitions` gates
/// `partitionValues_parsed`, `stats_schema` gates the top-level `stats` sibling. Naming-agnostic —
/// the caller relabels the sub-schemas (via [`relabel_partitions`] / [`relabel_stats`]) before
/// passing them, so this function is identical for both name spaces. The four fixed
/// `fileConstantValues` scalars and `path`/`size`/`deletionVector` are naming-invariant.
fn scan_file_row_schema_parts(
    partitions: Option<&SchemaRef>,
    stats_schema: Option<&SchemaRef>,
) -> SchemaRef {
    let tags = MapType::new(DataType::STRING, DataType::STRING, true);
    let mut file_constant_fields = vec![
        StructField::nullable("baseRowId", DataType::LONG),
        StructField::nullable("defaultRowCommitVersion", DataType::LONG),
        StructField::nullable("tags", tags),
        StructField::nullable("clusteringProvider", DataType::STRING),
    ];
    if let Some(parts) = partitions {
        file_constant_fields.push(StructField::nullable(
            "partitionValues_parsed",
            parts.as_ref().clone(),
        ));
    }

    let mut top_fields = vec![
        StructField::not_null("path", DataType::STRING),
        StructField::not_null("size", DataType::LONG),
        StructField::nullable("deletionVector", DeletionVectorDescriptor::to_schema()),
        StructField::nullable(
            FILE_CONSTANT_VALUES_NAME,
            StructType::new_unchecked(file_constant_fields),
        ),
    ];
    // Stage K: retain per-file stats as a top-level `stats` column (sibling of
    // `fileConstantValues`) so an engine-free consumer can read `add.stats_parsed` off the flat
    // scan_file_row. Appended in lockstep with the projection expr so field/expr counts match.
    if let Some(stats_schema) = stats_schema {
        top_fields.push(StructField::nullable(
            "stats",
            stats_schema.as_ref().clone(),
        ));
    }

    Arc::new(StructType::new_unchecked(top_fields))
}

/// Resolve the declared field names of a `partitionValues_parsed` [`StructType`] for `name_space`.
///
/// [`Physical`](NameSpace::Physical) returns the input unchanged (the data stage needs physical
/// names for its parquet-Load projection). [`Logical`](NameSpace::Logical) relabels to
/// table-facing partition column names.
///
/// Partition columns live in the *logical* schema but **not** the file read schema
/// (`Scan::physical_schema` excludes them — their values come from the Add action, not the file),
/// so the data-schema relation cannot be used. Instead map each physical partition field to its
/// logical name by position: the physical partition schema is the ordered projection of the table's
/// partition columns that resolve in the logical schema (see
/// `TableConfiguration::partition_schema_with_physical_names`), so the two align.
fn relabel_partitions(scan: &Scan, parts: &SchemaRef, name_space: NameSpace) -> SchemaRef {
    if name_space == NameSpace::Physical {
        return parts.clone();
    }
    let tc = scan.snapshot().table_configuration();
    let logical = tc.logical_schema();
    let logical_names: Vec<String> = tc
        .partition_columns()
        .iter()
        .filter(|c| logical.field(c.as_str()).is_some())
        .cloned()
        .collect();
    let fields = parts.fields().enumerate().map(|(i, pf)| {
        let name = logical_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| pf.name().clone());
        StructField::new(name, pf.data_type().clone(), pf.is_nullable())
    });
    Arc::new(StructType::new_unchecked(fields))
}

/// Resolve the declared leaf names of a stats [`StructType`]
/// (`{numRecords, nullCount:{…}, minValues:{…}, maxValues:{…}, tightBounds}`) for `name_space`.
///
/// [`Physical`](NameSpace::Physical) returns the input unchanged. [`Logical`](NameSpace::Logical)
/// renames the per-column leaves inside `nullCount` / `minValues` / `maxValues` to logical names;
/// the bookkeeping fields (`numRecords`, `tightBounds`) and the wrapper names are naming-invariant.
///
/// Stats never cover partition columns (they are built from the *data* schema with partition
/// columns excluded — see `TableConfiguration::physical_data_schema_without_partition_columns`), so
/// the rename map is derived from the **partition-excluded** logical/physical data-schema pair,
/// which are aligned by position (`make_physical` preserves order + structure). Using the full
/// `Scan::{logical,physical}_schema` pair would misalign, because the logical schema interleaves
/// partition columns the physical read schema omits.
fn relabel_stats(scan: &Scan, stats: &SchemaRef, name_space: NameSpace) -> SchemaRef {
    if name_space == NameSpace::Physical {
        return stats.clone();
    }
    let tc = scan.snapshot().table_configuration();
    let rename = physical_to_logical_leaf_names(
        &tc.logical_schema_without_partition_columns(),
        &tc.physical_data_schema_without_partition_columns(),
    );
    Arc::new(relabel_struct(stats, &rename))
}

/// Build a `physical field name -> logical field name` map (per struct level) by walking the
/// logical and physical schemas in positional lockstep. Because the physical schema is
/// `logical.make_physical(mode)` — same order and structure with renamed fields — pairing by
/// position is exact and needs no field-id join. Nested structs contribute their own entries under
/// the same map (names are unique within a level).
///
/// Callers must pass a **positionally aligned** pair (same field set + order). In particular, use
/// the partition-*excluded* data-schema pair for stats: the full scan logical schema interleaves
/// partition columns that the physical read schema omits.
fn physical_to_logical_leaf_names(
    logical: &StructType,
    physical: &StructType,
) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    fn walk(
        logical: &StructType,
        physical: &StructType,
        map: &mut std::collections::HashMap<String, String>,
    ) {
        for (lf, pf) in logical.fields().zip(physical.fields()) {
            map.insert(pf.name().clone(), lf.name().clone());
            if let (DataType::Struct(l_inner), DataType::Struct(p_inner)) =
                (lf.data_type(), pf.data_type())
            {
                walk(l_inner, p_inner, map);
            }
        }
    }
    walk(logical, physical, &mut map);
    map
}

/// Rename every field of `schema` (recursing into nested structs) via `rename`, leaving a field's
/// name unchanged when it has no entry (e.g. the stats bookkeeping fields `numRecords` /
/// `tightBounds` / the `nullCount` / `minValues` / `maxValues` wrappers, which are not data-column
/// leaves). Types and nullability are preserved.
fn relabel_struct(
    schema: &StructType,
    rename: &std::collections::HashMap<String, String>,
) -> StructType {
    let fields = schema.fields().map(|f| {
        let name = rename
            .get(f.name())
            .cloned()
            .unwrap_or_else(|| f.name().clone());
        let data_type = match f.data_type() {
            DataType::Struct(inner) => DataType::Struct(Box::new(relabel_struct(inner, rename))),
            other => other.clone(),
        };
        StructField::new(name, data_type, f.is_nullable())
    });
    StructType::new_unchecked(fields)
}

// Helper module-internal tests are exercised via the engine's scan integration tests; the
// `scan_data_projection` logic itself is unit-tested in [`super::file_scan::tests`].

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{Remove, REMOVE_NAME};
    use crate::schema::arc_schema;

    /// Build a synthetic input builder whose schema mimics the reconciled action stream
    /// (post-`with_partitions_parsed`): `add` carries `path`/`size`/`deletionVector`/
    /// `baseRowId`/etc., plus `partitionValues_parsed` when partitions are present. The
    /// terminal projection reads only those field names regardless of how the rest of the
    /// upstream looks.
    fn make_input_builder(parts: Option<&SchemaRef>) -> (Context, PlanBuilder) {
        make_input_builder_with_stats(parts, None)
    }

    /// Like [`make_input_builder`] but optionally stamps an `add.stats_parsed` struct field on the
    /// synthetic reconciled `add` slot, so the terminal's `col([ADD_NAME, "stats_parsed"])`
    /// reference resolves when exercising the stats-retention path.
    fn make_input_builder_with_stats(
        parts: Option<&SchemaRef>,
        stats: Option<&SchemaRef>,
    ) -> (Context, PlanBuilder) {
        let tags = MapType::new(DataType::STRING, DataType::STRING, true);
        let mut add_fields = vec![
            StructField::nullable("path", DataType::STRING),
            StructField::nullable("size", DataType::LONG),
            StructField::nullable("deletionVector", DeletionVectorDescriptor::to_schema()),
            StructField::nullable("baseRowId", DataType::LONG),
            StructField::nullable("defaultRowCommitVersion", DataType::LONG),
            StructField::nullable("tags", tags),
            StructField::nullable("clusteringProvider", DataType::STRING),
        ];
        if let Some(p) = parts {
            add_fields.push(StructField::nullable(
                "partitionValues_parsed",
                p.as_ref().clone(),
            ));
        }
        if let Some(s) = stats {
            add_fields.push(StructField::nullable("stats_parsed", s.as_ref().clone()));
        }
        let schema = arc_schema([
            StructField::nullable(ADD_NAME, StructType::new_unchecked(add_fields)),
            StructField::nullable(REMOVE_NAME, Remove::to_schema()),
        ]);
        let ctx = Context::new();
        let builder = ctx
            .values(schema, Vec::<Vec<crate::expressions::Scalar>>::new())
            .unwrap();
        (ctx, builder)
    }

    /// Without partitions: the terminal emits the four-field `fileConstantValues` and
    /// drops the `partitionValues` Map slot.
    #[test]
    fn project_scan_file_row_drops_partition_values_map_when_no_parts() {
        let (_ctx, builder) = make_input_builder(None);
        let out = project_scan_file_row(builder, None, None).unwrap();
        let schema = out.schema().unwrap();
        let fields: Vec<_> = schema.fields().map(|f| f.name().clone()).collect();
        assert_eq!(
            fields,
            ["path", "size", "deletionVector", FILE_CONSTANT_VALUES_NAME]
        );
        let DataType::Struct(fcv) = schema.field(FILE_CONSTANT_VALUES_NAME).unwrap().data_type()
        else {
            panic!("fileConstantValues must be a struct");
        };
        let fcv_fields: Vec<_> = fcv.fields().map(|f| f.name().as_str()).collect();
        assert_eq!(
            fcv_fields,
            [
                "baseRowId",
                "defaultRowCommitVersion",
                "tags",
                "clusteringProvider",
            ],
            "partitionValues Map slot must be dropped when no partitions",
        );
    }

    /// With partitions: `fileConstantValues.partitionValues_parsed` is appended as a
    /// passthrough column ref (not `map_to_struct`); the Map slot remains absent.
    #[test]
    fn project_scan_file_row_appends_partitions_parsed_when_some() {
        let parts = arc_schema([StructField::nullable("p", DataType::STRING)]);
        let (_ctx, builder) = make_input_builder(Some(&parts));
        let out = project_scan_file_row(builder, Some(&parts), None).unwrap();
        let schema = out.schema().unwrap();
        let DataType::Struct(fcv) = schema.field(FILE_CONSTANT_VALUES_NAME).unwrap().data_type()
        else {
            panic!("fileConstantValues must be a struct");
        };
        let fcv_fields: Vec<_> = fcv.fields().map(|f| f.name().as_str()).collect();
        assert_eq!(
            fcv_fields,
            [
                "baseRowId",
                "defaultRowCommitVersion",
                "tags",
                "clusteringProvider",
                "partitionValues_parsed",
            ],
            "partitions present => partitionValues_parsed appended; Map slot stays absent",
        );
    }

    /// With `physical_stats_schema` = `Some`: a top-level `stats` column (sibling of
    /// `fileConstantValues`, carrying `add.stats_parsed`) is appended; passing `None` leaves the
    /// terminal byte-identical to the four-field shape.
    #[test]
    fn project_scan_file_row_appends_stats_when_requested() {
        let stats = arc_schema([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable(
                "minValues",
                StructType::new_unchecked([StructField::nullable("col-abc", DataType::LONG)]),
            ),
            StructField::nullable("tightBounds", DataType::BOOLEAN),
        ]);

        // stats requested => `stats` appended as a top-level sibling of `fileConstantValues`.
        let (_ctx, builder) = make_input_builder_with_stats(None, Some(&stats));
        let out = project_scan_file_row(builder, None, Some(&stats)).unwrap();
        let schema = out.schema().unwrap();
        let fields: Vec<_> = schema.fields().map(|f| f.name().clone()).collect();
        assert_eq!(
            fields,
            [
                "path",
                "size",
                "deletionVector",
                FILE_CONSTANT_VALUES_NAME,
                "stats"
            ],
            "stats requested => top-level `stats` column appended after fileConstantValues",
        );
        let DataType::Struct(stats_struct) = schema.field("stats").unwrap().data_type() else {
            panic!("`stats` must be a struct");
        };
        let stats_fields: Vec<_> = stats_struct.fields().map(|f| f.name().as_str()).collect();
        assert_eq!(
            stats_fields,
            ["numRecords", "minValues", "tightBounds"],
            "`stats` carries the physical_stats_schema verbatim (physical leaf names)",
        );

        // stats NOT requested => terminal is byte-identical to the four-field shape even though
        // the input carries `stats_parsed`.
        let (_ctx, builder) = make_input_builder_with_stats(None, Some(&stats));
        let out = project_scan_file_row(builder, None, None).unwrap();
        let fields: Vec<_> = out
            .schema()
            .unwrap()
            .fields()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            fields,
            ["path", "size", "deletionVector", FILE_CONSTANT_VALUES_NAME],
            "stats not requested => no `stats` column, four-field shape unchanged",
        );
    }

    // === Logical relabeling (name-only, structure-preserving) ============================

    /// `physical_to_logical_leaf_names` pairs a logical/physical schema by position and records one
    /// entry per field (recursing into nested structs), keyed by physical name.
    #[test]
    fn physical_to_logical_leaf_names_maps_flat_and_nested() {
        let logical = StructType::new_unchecked([
            StructField::nullable("id", DataType::LONG),
            StructField::nullable(
                "outer",
                StructType::new_unchecked([StructField::nullable("inner", DataType::STRING)]),
            ),
        ]);
        let physical = StructType::new_unchecked([
            StructField::nullable("col-a", DataType::LONG),
            StructField::nullable(
                "col-outer",
                StructType::new_unchecked([StructField::nullable("col-b", DataType::STRING)]),
            ),
        ]);
        let map = physical_to_logical_leaf_names(&logical, &physical);
        assert_eq!(map.get("col-a").map(String::as_str), Some("id"));
        assert_eq!(map.get("col-outer").map(String::as_str), Some("outer"));
        assert_eq!(map.get("col-b").map(String::as_str), Some("inner"));
    }

    /// `relabel_struct` renames leaves that have a mapping (recursing into structs) and leaves
    /// unmapped fields — e.g. the stats bookkeeping/wrapper names — untouched, preserving types and
    /// nullability.
    #[test]
    fn relabel_struct_renames_only_mapped_leaves() {
        // A stats-shaped struct: physical leaf `col-a` under min/max/nullCount wrappers.
        let inner = || StructType::new_unchecked([StructField::nullable("col-a", DataType::LONG)]);
        let stats = StructType::new_unchecked([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable("nullCount", inner()),
            StructField::nullable("minValues", inner()),
            StructField::nullable("maxValues", inner()),
            StructField::nullable("tightBounds", DataType::BOOLEAN),
        ]);
        let rename = [("col-a".to_string(), "id".to_string())]
            .into_iter()
            .collect();
        let out = relabel_struct(&stats, &rename);

        // Top-level wrapper/bookkeeping names are unchanged...
        let top: Vec<_> = out.fields().map(|f| f.name().as_str()).collect();
        assert_eq!(
            top,
            [
                "numRecords",
                "nullCount",
                "minValues",
                "maxValues",
                "tightBounds"
            ],
        );
        // ...but the per-column leaf inside each wrapper is relabeled logical.
        for wrapper in ["nullCount", "minValues", "maxValues"] {
            let DataType::Struct(w) = out.field(wrapper).unwrap().data_type() else {
                panic!("{wrapper} must be a struct");
            };
            let leaves: Vec<_> = w.fields().map(|f| f.name().as_str()).collect();
            assert_eq!(leaves, ["id"], "{wrapper} leaf must be relabeled logical");
        }
    }

    /// An empty rename map (no column mapping / `none` mode) is an identity relabel.
    #[test]
    fn relabel_struct_identity_when_no_mapping() {
        let s = StructType::new_unchecked([
            StructField::nullable("numRecords", DataType::LONG),
            StructField::nullable(
                "minValues",
                StructType::new_unchecked([StructField::nullable("id", DataType::LONG)]),
            ),
        ]);
        let out = relabel_struct(&s, &std::collections::HashMap::new());
        assert_eq!(out, s, "no mapping => byte-identical");
    }
}
