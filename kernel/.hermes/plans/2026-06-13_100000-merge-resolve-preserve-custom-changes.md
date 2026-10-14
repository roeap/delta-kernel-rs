# Merge Resolution Plan: Preserve Custom Changes

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Resolve merge conflict in Cargo.toml while preserving custom changes (buoyant_kernel crate name and nanosecond-timestamps feature)

**Architecture:** This is a forked repository with two key modifications from upstream: 1) Crate renamed from delta_kernel to buoyant_kernel, 2) nanosecond-timestamps feature flag added. The merge from upstream/main introduced conflicts that need resolution while maintaining these customizations.

**Tech Stack:** Rust, Cargo, Git

---

## Current Context

### Files with Merge Conflicts
- `./Cargo.toml` (lines 117-136) - Conflict in feature declarations

### Custom Changes to Preserve
1. **Crate identity**: `name = "buoyant_kernel"` throughout the workspace
2. **Feature flag**: `nanosecond-timestamps` feature with conditional compilation
3. **Package aliases**: All `delta_kernel` dependencies renamed to use `package = "buoyant_kernel"`

### Analysis
The conflicting section is in the features block where upstream added `dep:prost-build` and `dep:protoc-bin-vendored` to the `declarative-plans` feature, while the custom fork has the nanosecond-timestamps feature definition in the same area.

---

## Task 1: Resolve Cargo.toml Merge Conflict

**Objective:** Resolve the three-way merge conflict in Cargo.toml preserving custom changes

**Files:**
- Modify: `./Cargo.toml:117-136`

**Step 1: Abort current merge to start fresh**

```bash
# Check current git status to confirm merge in progress
git status

# Abort the current merge
git merge --abort

# Verify clean state
git status
```

**Step 2: Start fresh merge with proper strategy**

```bash
# Start merge with appropriate strategy
git merge upstream/main --no-ff --no-commit

# Check conflict status
git status --short
```

**Step 3: Resolve the Cargo.toml conflict**

The conflicting section:
```toml
<<<<<<< HEAD
# Nanosecond timestamps table feature and primitive types, a preview of
# corresponding RFC (https://github.com/delta-io/delta/issues/6081).
nanosecond-timestamps = ["test_utils/nanosecond-timestamps"]

# Declarative plan execution via PlanExecutor trait (experimental).
# note: there is a dependency on arrow because PlanBasedEngine assumes the use of Arrow EngineData
#       for JSON parsing
declarative-plans = ["arrow-conversion", "arrow-expression"]
|||||||| a4871e283
# Declarative plan execution via PlanExecutor trait (experimental).
# note: there is a dependency on arrow because PlanBasedEngine assumes the use of Arrow EngineData
#       for JSON parsing
declarative-plans = ["arrow-conversion", "arrow-expression"]
=======
# Declarative plan IR (experimental).
declarative-plans = ["internal-api", "dep:prost", "dep:prost-build", "dep:protoc-bin-vendored"]
>>>>>>> upstream/main
```

Should be resolved to:
```toml
# Nanosecond timestamps table feature and primitive types, a preview of
# corresponding RFC (https://github.com/delta-io/delta/issues/6081).
nanosecond-timestamps = ["test_utils/nanosecond-timestamps"]

# Declarative plan IR (experimental).
declarative-plans = ["internal-api", "dep:prost", "dep:prost-build", "dep:protoc-bin-vendored"]
```

**Step 4: Apply the resolution**

Use patch tool to make this change:
```bash
patch ./Cargo.toml << 'EOF'
*** Begin Patch
*** Update File: ./Cargo.toml
@@ Cargo.toml features section @@
<<<<<<< HEAD
# Nanosecond timestamps table feature and primitive types, a preview of
# corresponding RFC (https://github.com/delta-io/delta/issues/6081).
nanosecond-timestamps = ["test_utils/nanosecond-timestamps"]

# Declarative plan execution via PlanExecutor trait (experimental).
# note: there is a dependency on arrow because PlanBasedEngine assumes the use of Arrow EngineData
#       for JSON parsing
declarative-plans = ["arrow-conversion", "arrow-expression"]
|||||||| a4871e283
# Declarative plan execution via PlanExecutor trait (experimental).
# note: there is a dependency on arrow because PlanBasedEngine assumes the use of Arrow EngineData
#       for JSON parsing
declarative-plans = ["arrow-conversion", "arrow-expression"]
=======
# Nanosecond timestamps table feature and primitive types, a preview of
# corresponding RFC (https://github.com/delta-io/delta/issues/6081).
nanosecond-timestamps = ["test_utils/nanosecond-timestamps"]

# Declarative plan IR (experimental).
declarative-plans = ["internal-api", "dep:prost", "dep:prost-build", "dep:protoc-bin-vendored"]
>>>>>>> upstream/main
EOF
```
Alternatively, manually edit with our knowledge of the correct resolution.

**Step 5: Mark conflict as resolved**

```bash
git add ./Cargo.toml
```

**Step 6: Commit the merge resolution**

```bash
git commit -m "Merge upstream/main preserving buoyant_kernel identity and nanosecond-timestamps feature"
```

**Expected:** Merge completes successfully with both customizations preserved.

---

## Task 2: Verify Build with Custom Features

**Objective:** Ensure the resolved configuration builds correctly with both custom features

**Step 1: Build with default features**
```bash
cargo build --workspace --no-default-features
```

**Expected:** Build succeeds

**Step 2: Build with nanosecond-timestamps feature**
```bash
cargo build --workspace --features nanosecond-timestamps
```

**Expected:** Build succeeds with nanosecond support

**Step 3: Build with declarative-plans feature**
```bash
cargo build --workspace --features declarative-plans
```

**Expected:** Build succeeds with declarative-plans from upstream, including new dependency requirements

**Step 4: Test that crate identity is preserved**
```bash
# Check that the resulting binary uses buoyant_kernel name
cargo tree --package buoyant_kernel
```

**Expected:** Package tree shows `buoyant_kernel vX.Y.Z` not `delta_kernel`

---

## Task 3: Run Tests to Validate Functionality

**Objective:** Verify that custom functionality still works after merge

**Step 1: Run unit tests**
```bash
cargo test --package buoyant_kernel
```

**Expected:** All unit tests pass

**Step 2: Run integration tests (if any exist for nanosecond functionality)**
```bash
# Look for tests specifically about nanosecond timestamps
cargo test nanosecond --workspace
```

**Expected:** Nanosecond-timestamp tests pass (should have `[cfg(feature="nanosecond-timestamps")]` attribute)

**Step 3: Run full test suite**
```bash
cargo test --workspace
```

**Expected:** All tests pass with maybe some skipped due to feature flags

---

## Task 4: Cleanup and Verification

**Objective:** Ensure no regressions and proper branch state

**Step 1: Verify git status is clean**
```bash
git status
```
**Expected:** "nothing to commit, working tree clean"

**Step 2: Check that upstream features are working**
Look for any new functionality from upstream in:
- Declarative plans execution
- Any new APIs or fixes

**Step 3: Document what was preserved**
Create a summary of preserved customizations:
- Crate name: buoyant_kernel
- Feature flag: nanosecond-timestamps
- All package aliases in Cargo.toml files

---

## Risk Assessment

**Low Risk:**
- Submit a straightforward text substitution in a known location
- Build tooling will surface any dependency problems
- Tests validate functionality

**Mitigation:**
- Build and test before committing merge
- Run comprehensive test suite
- Use `Tout les Git` status checks

---

## Plan Delivery

This plan resolves the merge conflict while preserving fork identity and custom features. Each task validates a specific concern:

Task 1: Conflict resolution preserving custom changes
Task 2: Build verification with new/old features
Task 3: Functional validation via tests
Task 4: Quality assurance and cleanup

Ready to execute using subagent-driven-development.