//! Declarative Full Snapshot Read (FSR): [`full_state`] builds the multi-phase
//! [`CoroutineSM`](crate::sm_plans::state_machines::framework::coroutine::driver::CoroutineSM)
//! consumed by [`crate::snapshot::Snapshot::full_state_builder`].

mod file_scan;
pub mod full_state;
mod ssa_reconciliation;
pub(crate) mod ssa_scan;

pub use full_state::{FullState, FullStateBuilder};
pub use ssa_reconciliation::CommitFileMeta;
// Shared reconciliation pipeline pieces reused by the snapshot-construction SMs
// (`super::snapshot`): the P&M base schema, its dedup key, and the async pipeline driver.
pub(crate) use ssa_reconciliation::{execute_reconciliation_ssa, pm_dedup_key, PM_BASE};
