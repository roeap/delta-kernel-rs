# Buoyant Kernel Migration Plan

## Goal
Migrate from `delta_kernel` to `buoyant_kernel` and create a new `buoyant_engine` crate, implementing the changes from patch `54dc148d09afc7773b6721f545435e1e66bcdb4f` without cherry-picking.

## Current State Analysis
- Repository: delta-kernel-rs
- Current branch: buoyant/dev
- Main crate: `delta_kernel` at ./kernel/Cargo.toml
- Engine crate: `delta_kernel_default_engine` at ./default-engine/Cargo.toml
- Total files referencing `delta_kernel`: ~200
- Total files referencing `delta_kernel_default_engine`: ~110

## Changes Required

### 1. Rename Kernel Crate (`delta_kernel` → `buoyant_kernel`)

#### Kernel Crate Changes (kernel/Cargo.toml)
1. Change `[package]name` from `delta_kernel` to `buoyant_kernel`
2. Update description and documentation URLs
3. Rename dependencies: `delta_kernel_derive` → `buoyant_kernel_derive`
4. Update derive crate reference in dependencies
5. Fix dev-dependencies to use the new package names

#### Files to Modify
- `kernel/Cargo.toml` - core crate metadata
- `Cargo.toml` (workspace) - add buoyant_kernel to members
- `derive-macros/Cargo.toml` - rename derive crate

### 2. Create New Engine Crate (`buoyant_engine`)

#### New Engine Crate (buoyant-engine/)
1. Create new directory `buoyant-engine/`
2. Create `buoyant-engine/Cargo.toml` with:
   - name: `buoyant_engine`
   - dependency on `buoyant_kernel`
   - copy functionality from `delta_kernel_default_engine`
3. Copy engine-specific code from `default-engine/` to `buoyant-engine/`

#### Migrating from delta_kernel_default_engine
1. Analyze which parts of default-engine are kernel-core vs engine-specific
2. Keep kernel-common components in `buoyant_kernel`
3. Move engine-specific implementations to `buoyant_engine`

### 3. Update Code References

#### Add Alias Imports
Most files will need: `use buoyant_kernel as delta_kernel;`

#### Files Categories to Fix:

**A. Kernel Internal Files** (`kernel/src/**/*.rs`):
- Replace `use delta_kernel_derive` → `use buoyant_kernel_derive`
- Add `use buoyant_kernel as delta_kernel;` to files using `delta_kernel` in types/paths
- Fix re-exports and public API

**B. Engine Implementation Files** (`default-engine/src/**/*.rs`):
- Replace all `delta_kernel` → `buoyant_kernel` imports
- Fix any `delta_kernel_default_engine` references
- Update internal dependency references

**C. Examples** (`kernel/examples/**/*.rs`):
- Replace `use delta_kernel` → `use buoyant_kernel as delta_kernel`
- Update `use delta_kernel_default_engine` → `use buoyant_engine`

**D. Tests** (`kernel/tests/**/*.rs`, `test-utils/**/*.rs`):
- Replace `use delta_kernel` → `use buoyant_kernel as delta_kernel`
- Update engine-related imports

**E. Other Crates**:
- `ffi/` - update imports and dependency references
- `delta-kernel-unity-catalog/` - update imports
- All other workspace members

### 4. Build System Updates

#### Cargo.toml Updates
1. Workspace `Cargo.toml`: Add `buoyant-engine` to members
2. Update all `Cargo.toml` files that depend on `delta_kernel` or `delta_kernel_default_engine`
3. Update `path` dependencies to point to new locations

#### Repository Metadata
1. Review `README.md`, `CHANGELOG.md`, and documentation
2. Update any hardcoded `delta_kernel` references
3. Update repository URLs if needed

## Implementation Strategy

### Phase 1: Crate Renaming (2-4 hours)
1. ✅ Analyze current structure (DONE)
2. Rename `kernel/Cargo.toml` → `buoyant_kernel`
3. Rename `derive-macros/Cargo.toml` → `buoyant_kernel_derive`
4. Test basic cargo build for kernel crate

### Phase 2: Engine Crate Creation (4-6 hours)
1. Create `buoyant-engine/` directory structure
2. Copy relevant code from `default-engine/`
3. Create `buoyant-engine/Cargo.toml` with proper dependencies
4. Test engine compilation separately

### Phase 3: Import Updates (8-12 hours)
1. Scripted find/replace for `delta_kernel` → `buoyant_kernel as delta_kernel`
2. Manual review of ~200 Rust files
3. Fix compilation errors iteratively
4. Test basic functionality after each category

### Phase 4: Test Migration (4-6 hours)
1. Update test files for new imports
2. Run cargo test phases gradually
3. Fix test-specific issues
4. Validate no regressions

### Phase 5: Integration & Cleanup (3-5 hours)
1. Update all workspace dependencies
2. Review Cargo.lock compatibility
3. Clean build artifacts
4. Document migration changes

## Risk Factors & Challenges

### Dependency Hell
- Circular dependencies if not careful with import order
- Cargo may have trouble resolving `buoyant_kernel as delta_kernel` in complex macro scenarios

### Macro Complexity
- `delta_kernel_derive` macros expanded with `delta_kernel` types
- Need to ensure derive macros work with renamed imports

### Test Environment
- Some tests may depend on exact module paths for reflection/debugging
- CI/CD pipelines may need updates
- Need to ensure test isolation still works

### Migration Cost
- Large-scale text substitution risking syntactic errors
- Manual review will be tedious but necessary
- Code search/replace tools may have false positives

## Success Criteria

1. ✅ Cargo build succeeds for all workspace members
2. ✅ All tests pass with new crate layout
3. ✅ No `delta_kernel` string references remain in source code
4. ✅ `buoyant_engine` crate builds and provides same functionality
5. ✅ Executables and examples work identically to before
6. ✅ Documentation and examples updated

## Timeline Estimate
- Total: ~20-30 hours (3-4 days)
- Could be parallelized by team
- Automated tooling can reduce manual effort

## Tools to Use
- `ripgrep` (rg) for finding references
- `sed` for bulk replacements
- `cargo check --no-deps` for quick syntax validation
- `git diff` for change review
-Hermes file tools for atomic changes