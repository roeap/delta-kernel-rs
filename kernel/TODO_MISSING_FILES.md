# Missing Files That Need Import Alias

The following files still need `use buoyant_kernel as delta_kernel;` to be added:

The list is large and includes many integration test files. The pattern is:
- All files in `tests/integration/**/*` that use `delta_kernel`
- All files in `tests/*.rs` that use `delta_kernel`
- Example files in `examples/**/*.rs`
- Benchmark files in `benches/*.rs`

Key files still needing updates:
- `tests/commit_range.rs` ✅ DONE
- `tests/incremental_scan.rs` ✅ DONE
- Many files in `tests/integration/` (features, write, read, etc.)
- Various other test and example files

## Verification Status
- ✅ `cargo nextest r --lib` passes (3,111 tests)
- ✅ Basic library compilation works
- ❌ Full test suite compilation still failing due to missing imports

## Next Steps
I will systematically add the import alias to all remaining files that need it, starting with the most critical test files that are causing compilation errors.