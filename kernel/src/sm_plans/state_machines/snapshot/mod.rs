//! Snapshot-construction state machines.
//!
//! Where the [`scan`](super::scan) SMs consume an already-built [`Snapshot`](crate::snapshot::Snapshot)
//! (Protocol, Metadata, and log segment already resolved), these SMs run *before* a snapshot
//! exists: they take a bare [`LogSegment`](crate::log_segment::LogSegment) and resolve the
//! Protocol + Metadata by driving an SSA log-replay plan through an engine — the engine-free,
//! async-native counterpart to the synchronous `LogSegment::read_protocol_metadata` replay.
//!
//! [`SnapshotPm`] is the Protocol & Metadata resolver: it drives the shared
//! [`execute_reconciliation_ssa`](super::scan) pipeline over the `{protocol, metaData}` base and
//! drains it through the
//! [`MetadataProtocolReader`](crate::sm_plans::kernel_consumers::MetadataProtocolReader)
//! consumer, yielding a typed `(Protocol, Metadata)`.

pub mod pm;

pub use pm::SnapshotPm;
