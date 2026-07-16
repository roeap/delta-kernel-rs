//! `SnapshotPm` — the Protocol & Metadata resolution state machine.
//!
//! Resolves a table's effective [`Protocol`] and [`Metadata`] from a bare
//! [`LogSegment`] by driving the shared reconciliation pipeline
//! ([`execute_reconciliation_ssa`]) over the `{protocol, metaData}` base
//! ([`PM_BASE`]) and draining the reconciled action stream through the
//! [`MetadataProtocolReader`] consumer.
//!
//! This is the engine-free, async-native counterpart to the synchronous
//! `LogSegment::read_protocol_metadata` / `replay_for_pm` log replay: the SM itself is CPU-only
//! and IO-free (it yields `SchemaQuery` / `Consume` requests the engine services against its own
//! async object store), so it drives snapshot construction without a kernel `Engine` and without
//! priming the log up front.
//!
//! The terminal value is a typed `(Protocol, Metadata)` — unlike the scan / FSR SMs, whose
//! terminal is a [`ResultPlan`](crate::sm_plans::ir::plan::ResultPlan). The caller assembles the
//! `Snapshot` from the resolved pair plus the log segment via
//! [`Snapshot::from_parts`](crate::snapshot::Snapshot::from_parts).

use std::sync::Arc;

use super::super::scan::{execute_reconciliation_ssa, pm_dedup_key, PM_BASE};
use crate::actions::{Metadata, Protocol};
use crate::log_segment::LogSegment;
use crate::sm_plans::errors::DeltaError;
use crate::sm_plans::kernel_consumers::MetadataProtocolReader;
use crate::sm_plans::state_machines::framework::coroutine::driver::CoroutineSM;
use crate::sm_plans::state_machines::framework::plan_context::Context as SsaContext;

/// Configured Protocol & Metadata resolution over a [`LogSegment`].
///
/// Construct via [`Self::for_log_segment`], then drive [`Self::state_machine`] through an engine
/// (e.g. the DataFusion `sm_plans` executor) to obtain the effective `(Protocol, Metadata)`.
#[derive(Debug, Clone)]
pub struct SnapshotPm {
    log_segment: Arc<LogSegment>,
}

impl SnapshotPm {
    /// Resolve Protocol & Metadata over `log_segment`.
    pub fn for_log_segment(log_segment: Arc<LogSegment>) -> Self {
        Self { log_segment }
    }

    /// Build the coroutine SM that drives P&M resolution end-to-end, yielding the effective
    /// `(Protocol, Metadata)`.
    ///
    /// The SM builds the shared reconciliation pipeline over [`PM_BASE`] (retention is a no-op —
    /// the base carries no tombstone/txn rows — so it passes `(0, None)`) and drains the reconciled
    /// stream through a [`MetadataProtocolReader`], which captures the newest Protocol and Metadata
    /// (`max_by_version` already collapses each singleton to its winning row) and stops as soon as
    /// both are present.
    pub fn state_machine(&self) -> Result<CoroutineSM<(Protocol, Metadata)>, DeltaError> {
        let log_segment = self.log_segment.clone();
        CoroutineSM::new("snapshot_pm_ssa", move |mut engine, _sm_id| async move {
            let ctx = SsaContext::new();
            let reconciled = execute_reconciliation_ssa(
                &ctx,
                &mut engine,
                log_segment.as_ref(),
                &PM_BASE,
                /* stats= */ None,
                /* parts= */ None,
                Arc::new(pm_dedup_key()),
                /* retention= */ (0, None),
            )
            .await?;
            ctx.consume(
                &mut engine,
                reconciled,
                MetadataProtocolReader::new(),
                "SnapshotPm::resolve::pm_drain",
            )
            .await
        })
    }
}
