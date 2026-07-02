//! Time primitives that work on both native and `wasm32-unknown-unknown` targets.
//!
//! `std::time::Instant` and `std::time::SystemTime` panic on `wasm32-unknown-unknown`
//! because that target has no clock. Kernel uses these types only for metrics/timing and
//! for deriving commit timestamps -- never for control flow -- so we route every such use
//! through this module. On native targets it re-exports `std`; on wasm it provides minimal
//! shims backed by JavaScript's `Date.now()` (via `js-sys`), which is sufficient for coarse
//! metric durations and Unix-epoch timestamps.
//!
//! `Duration` is target-agnostic and continues to come from `std::time`.
//!
//! We deliberately avoid the `web-time` crate here: its wasm-family target dependencies
//! perturb Cargo's resolver-v3 feature unification for this workspace and strip required
//! features off the shared `object_store` build. A direct `js-sys` shim avoids that.

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::{Instant, SystemTime};

#[cfg(target_arch = "wasm32")]
pub(crate) use wasm::{Instant, SystemTime};

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::time::Duration;

    /// Reads wall-clock milliseconds since the Unix epoch from JS `Date.now()`.
    fn now_ms() -> f64 {
        js_sys::Date::now()
    }

    /// Monotonic-ish timer for measuring elapsed durations in metrics. Backed by wall-clock
    /// time, so it is not strictly monotonic across NTP adjustments; adequate for coarse
    /// timing where kernel only reports the delta.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct Instant {
        millis: f64,
    }

    impl Instant {
        pub(crate) fn now() -> Self {
            Self { millis: now_ms() }
        }

        pub(crate) fn elapsed(&self) -> Duration {
            Duration::from_secs_f64(((now_ms() - self.millis) / 1000.0).max(0.0))
        }
    }

    /// Wall-clock time backed by JS `Date.now()`, exposing the subset of the std
    /// `SystemTime` API kernel uses: `now()`, `duration_since()`, and `UNIX_EPOCH`.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct SystemTime {
        /// Milliseconds since the Unix epoch.
        millis: f64,
    }

    impl SystemTime {
        pub(crate) const UNIX_EPOCH: SystemTime = SystemTime { millis: 0.0 };

        pub(crate) fn now() -> Self {
            Self { millis: now_ms() }
        }

        /// Mirrors `std::time::SystemTime::duration_since`: `Err` when `earlier` is later
        /// than `self`, matching how kernel maps it to a "time before Unix epoch" error.
        pub(crate) fn duration_since(
            &self,
            earlier: SystemTime,
        ) -> Result<Duration, SystemTimeError> {
            let delta = self.millis - earlier.millis;
            if delta < 0.0 {
                Err(SystemTimeError)
            } else {
                Ok(Duration::from_secs_f64(delta / 1000.0))
            }
        }
    }

    /// Error returned by [`SystemTime::duration_since`] when the argument is later than
    /// `self`. Mirrors `std::time::SystemTimeError` closely enough for kernel's `Display`.
    #[derive(Debug)]
    pub(crate) struct SystemTimeError;

    impl std::fmt::Display for SystemTimeError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("second time provided was later than self")
        }
    }

    impl std::error::Error for SystemTimeError {}
}
