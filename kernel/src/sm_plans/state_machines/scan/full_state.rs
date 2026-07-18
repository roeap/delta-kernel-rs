//! Canonical Full Snapshot Read (FSR) declarative plans — *window-on-commits +
//! anti-join-on-checkpoint*.
//!
//! Mirrors Delta log-replay semantics by composing three or four declarative plans:
//!
//! 1. **commit_load** materializes the raw per-commit action stream into `fsr.commit_raw`, covering
//!    `ascending_commit_files ∪ ascending_compaction_files` so downstream steps can `ORDER BY
//!    version DESC` to recover "newest action wins" semantics.
//! 2. **commit_dedup** runs a `row_number PARTITION BY key ORDER BY version DESC` window over the
//!    commit tail to pick the winners, materialized into `fsr.commit_dedup`.
//! 3. **sidecar_load** (only when `has_sidecars`) materializes V2-multipart sidecar parquet actions
//!    into `fsr.sidecar_actions`.
//! 4. **results** anti-joins the (top-level checkpoint ∪ sidecar_actions) stream against
//!    `commit_dedup`, unions in the winners, applies retention, and binds the final reconciled
//!    action stream to `fsr.results`.
//!
//! The window applies only to the (small) commit tail; the (large) checkpoint stream
//! goes through a single hash anti-join, avoiding per-key orderings over the full snapshot.
//!
//! `dv_unique_id` is computed as `If(storageType IS NULL, NULL,
//! ToJson(Array(storageType, pathOrInlineDv)))` — equality-equivalent (not byte-equivalent)
//! to `DeletionVectorDescriptor::unique_id_from_parts`.
//! `offset` is omitted because kernel lacks an int-to-string cast; inline DVs sharing
//! `pathOrInlineDv` would already imply identical byte payloads.

use std::sync::Arc;

use super::ssa_reconciliation::{
    execute_reconciliation_ssa, fsr_dedup_key, retention_timestamps, stats_parsed_file_schema,
    FSR_BASE,
};
use crate::expressions::{col, Expression};
use crate::scan::state_info::StateInfo;
use crate::scan::{PartitionValuesOptions, StatsOptions};
use crate::schema::SchemaRef;
use crate::sm_plans::errors::{DeltaError, KernelErrAsDelta};
use crate::sm_plans::ir::plan::ResultPlan;
use crate::sm_plans::state_machines::framework::coroutine::driver::CoroutineSM;
use crate::sm_plans::state_machines::framework::plan_context::Context as SsaContext;
use crate::snapshot::Snapshot;

/// Configured FSR plan source. Construct via [`FullState::for_table`] (or
/// [`Snapshot::full_state_builder`](crate::snapshot::Snapshot::full_state_builder)), call
/// [`FullStateBuilder::build`], then drive [`Self::state_machine`] through an engine. The
/// snapshot is the sole source of truth for the table's log segment and `_last_checkpoint`
/// hint; the resolver derives shape from those, no override hook.
#[derive(Debug, Clone)]
pub struct FullState {
    snapshot: Arc<Snapshot>,
    /// `Some` iff [`FullStateBuilder::with_stats`] was called. Derived at `build()` time.
    state_info: Option<Arc<StateInfo>>,
}

/// Builder for canonical Full State Reconstruction plans.
#[derive(Debug, Clone)]
pub struct FullStateBuilder {
    snapshot: Arc<Snapshot>,
    with_stats: bool,
}

impl FullState {
    /// Start building canonical FSR plans for `snapshot`.
    pub fn for_table(snapshot: Arc<Snapshot>) -> FullStateBuilder {
        FullStateBuilder {
            snapshot,
            with_stats: false,
        }
    }

    /// CoroutineSM SM driving the FSR pipeline end-to-end.
    ///
    /// Builds against the [`crate::sm_plans::state_machines::framework::plan_context::Context`]
    /// (SSA / PlanBuilder API) and yields a single
    /// [`ResultPlan`] containing the entire
    /// reconciliation as one flat SSA program. Engines drive this through
    /// `drive_ssa_to_dataframe`.
    pub fn state_machine(&self) -> Result<CoroutineSM<ResultPlan>, DeltaError> {
        let snapshot = self.snapshot.clone();
        let state_info = self.state_info.clone();
        CoroutineSM::new("fsr_ssa", move |mut engine, _sm_id| async move {
            let ctx = SsaContext::new();
            let stats = state_info
                .as_ref()
                .and_then(|si| si.physical_stats_schema.clone());
            let terminal_schema = Self::output_schema_for(stats.as_ref());
            let reconciled = execute_reconciliation_ssa(
                &ctx,
                &mut engine,
                snapshot.log_segment(),
                &FSR_BASE,
                stats,
                /* parts= */ None,
                Arc::new(fsr_dedup_key()),
                Some(retention_timestamps(snapshot.as_ref())?),
            )
            .await?;
            // Terminal projection to the declared action-slot schema. The reconciliation pipeline
            // threads synthetic bookkeeping columns (the `__fsr_join_k` dedup key, and — once an
            // engine compiles the SSA — per-node presence markers) alongside the six action slots;
            // an explicit identity projection here drops everything but the declared slots, exactly
            // as the scan SM terminates on `project_scan_file_row`. This makes the driven terminal
            // byte-match [`Self::output_schema`], so an engine-free consumer can declare its output
            // schema up front. Identity `col([slot])` per slot — the `add.stats -> add.stats_parsed`
            // swap already happened upstream (`with_json_stats_parsed`), so `col(["add"])` carries it.
            let exprs: Vec<Arc<Expression>> = terminal_schema
                .fields()
                .map(|f| col([f.name().as_str()]).into())
                .collect();
            let projected = reconciled.project_with_schema(exprs, terminal_schema)?;
            ctx.into_result_plan(projected)
        })
    }

    /// The schema of the reconciled action stream [`Self::state_machine`] emits, computed as a
    /// pure function of the table configuration — no SM drive, no engine, no I/O.
    ///
    /// This is the [`FullState`] analog of [`Scan::scan_file_row_schema`](crate::scan::Scan::scan_file_row_schema):
    /// an engine-free consumer (a DataFusion `TableProvider` over the log, say) needs to declare the
    /// output schema up front, before it drives the coroutine. The [`Self::state_machine`] terminal
    /// projects to exactly this schema, so the two cannot drift.
    ///
    /// The shape is [`FSR_BASE`] — the six reconciled action slots
    /// `{add, remove, protocol, metaData, domainMetadata, txn}` — with one in-place edit that mirrors
    /// exactly what the pipeline applies: when the state was built [`FullStateBuilder::with_stats`],
    /// `add.stats: STRING` is replaced by `add.stats_parsed: STRUCT<…>` (the same swap
    /// `ReconciliationPlanBuilder::with_json_stats_parsed` performs, reproduced here via
    /// [`stats_parsed_file_schema`]). `FullState` never requests partition parsing
    /// (`parts = None` in [`Self::state_machine`]), so `add.partitionValues` is left as-is. The
    /// synthetic `version` / join-key / presence-marker columns the pipeline threads internally are
    /// dropped by the terminal projection, so they do not appear here.
    ///
    /// [`stats_parsed_file_schema`]: super::ssa_reconciliation::stats_parsed_file_schema
    pub fn output_schema(&self) -> SchemaRef {
        // Mirror `state_machine()`'s stats resolution exactly, so the declared schema and the driven
        // terminal cannot drift.
        let stats = self
            .state_info
            .as_ref()
            .and_then(|si| si.physical_stats_schema.clone());
        Self::output_schema_for(stats.as_ref())
    }

    /// Shared derivation behind [`Self::output_schema`] and the [`Self::state_machine`] terminal
    /// projection, so the declared schema and the driven terminal are built from one source.
    fn output_schema_for(stats: Option<&SchemaRef>) -> SchemaRef {
        match stats {
            Some(stats_schema) => stats_parsed_file_schema(&FSR_BASE, stats_schema),
            None => FSR_BASE.clone(),
        }
    }
}

impl FullStateBuilder {
    /// Surface `add.stats_parsed` in FSR output. `build()` then derives the
    /// `physical_stats_schema` driving native-stats detection and projection.
    pub fn with_stats(mut self) -> Self {
        self.with_stats = true;
        self
    }

    /// Finalize into a [`FullState`]. When [`Self::with_stats`] was called, constructs a
    /// `StateInfo` from the snapshot's logical schema with no predicate and
    /// [`StatsOptions::all_struct`]; otherwise no `StateInfo` is built.
    pub fn build(self) -> Result<FullState, DeltaError> {
        let state_info = if self.with_stats {
            let logical_schema = self.snapshot.schema();
            let table_configuration = self.snapshot.table_configuration();
            let si = StateInfo::try_new(
                logical_schema.clone(),
                logical_schema,
                table_configuration,
                None,
                &StatsOptions::all_struct(),
                &PartitionValuesOptions::default(),
                (),
            )
            .map_err(|e| e.into_delta_default())?;
            Some(Arc::new(si))
        } else {
            None
        };
        Ok(FullState {
            snapshot: self.snapshot,
            state_info,
        })
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::actions::{ADD_NAME, METADATA_NAME, PROTOCOL_NAME, REMOVE_NAME};
    use crate::engine::sync::SyncEngine;
    use crate::schema::DataType;
    use crate::snapshot::Snapshot;

    fn snapshot_for(table_rel_path: &str) -> Arc<Snapshot> {
        let path = std::fs::canonicalize(table_rel_path).unwrap();
        let url = Url::from_directory_path(path).unwrap();
        Snapshot::builder_for(url)
            .build(&SyncEngine::new())
            .unwrap()
    }

    /// The `add` slot's inner fields, or `None` if `add` is missing / not a struct.
    fn add_inner_names(schema: &SchemaRef) -> Option<Vec<String>> {
        let DataType::Struct(add) = schema.field(ADD_NAME)?.data_type() else {
            return None;
        };
        Some(add.fields().map(|f| f.name().clone()).collect())
    }

    /// Without `with_stats`, `output_schema` is exactly `FSR_BASE`: the six reconciled action
    /// slots, `add` untouched (`stats` stays the raw JSON string, no `stats_parsed`).
    #[test]
    fn output_schema_without_stats_is_fsr_base() {
        let snapshot = snapshot_for("./tests/data/basic_partitioned");
        let fs = FullState::for_table(snapshot).build().unwrap();
        let schema = fs.output_schema();

        assert_eq!(
            schema.as_ref(),
            FSR_BASE.as_ref(),
            "no stats => FSR_BASE verbatim"
        );

        let top: Vec<_> = schema.fields().map(|f| f.name().as_str()).collect();
        assert!(top.contains(&ADD_NAME) && top.contains(&REMOVE_NAME));
        assert!(top.contains(&PROTOCOL_NAME) && top.contains(&METADATA_NAME));

        let add = add_inner_names(&schema).expect("add slot present");
        assert!(
            add.iter().any(|n| n == "stats"),
            "raw `stats` retained without stats"
        );
        assert!(
            !add.iter().any(|n| n == "stats_parsed"),
            "no `stats_parsed` without with_stats()",
        );
    }

    /// With `with_stats`, the accessor reproduces the pipeline's in-place `add.stats: STRING ->
    /// add.stats_parsed: STRUCT` swap — and nothing else changes relative to `FSR_BASE`.
    #[test]
    fn output_schema_with_stats_swaps_add_stats_for_parsed() {
        let snapshot = snapshot_for("./tests/data/basic_partitioned");
        let fs = FullState::for_table(snapshot).with_stats().build().unwrap();
        let schema = fs.output_schema();

        // Same top-level slot set as FSR_BASE.
        let top: Vec<_> = schema.fields().map(|f| f.name().clone()).collect();
        let base_top: Vec<_> = FSR_BASE.fields().map(|f| f.name().clone()).collect();
        assert_eq!(
            top, base_top,
            "with_stats only edits inside `add`, not the top-level slots"
        );

        let add = add_inner_names(&schema).expect("add slot present");
        assert!(
            add.iter().any(|n| n == "stats_parsed"),
            "with_stats() => add.stats_parsed present, got {add:?}",
        );
        assert!(
            !add.iter().any(|n| n == "stats"),
            "the raw JSON `stats` column is replaced, not kept alongside",
        );

        // Faithful to the exact edit the pipeline applies (`with_json_stats_parsed` ==
        // `stats_parsed_file_schema`), so the declared schema matches the driven terminal.
        let stats_schema = fs
            .state_info
            .as_ref()
            .and_then(|si| si.physical_stats_schema.clone())
            .expect("with_stats() builds a StateInfo carrying physical_stats_schema");
        assert_eq!(
            schema.as_ref(),
            stats_parsed_file_schema(&FSR_BASE, &stats_schema).as_ref(),
        );
    }
}
