# Final Doc Test Fix - Packaging Resolution

> **For Hermes:** Execute this plan task-by-task using delegate_task.

**Goal:** Resolve all outstanding doc test failures (7/54 failing: 2 from commit_range, 4 from schema, 1 from table_changes).

**Architecture:** Three phases:
1. Fix remaining crate:: import resolution issues in schema/mod.rs
2. Update commit_range/mod.rs to use proper crate naming
3. Apply same packaging discipline to table_changes/mod.rs

**Tech Stack:** Cargo/Rust, Doc Test System

---

## Analysis Results

From `cargo test --doc` output analysis:
* **Total tests:** 50+ doc tests
* **Passed:** 54 ✓  
* **Failed:** 7 ✗
* **Ignored:** 34 ⚠️

**Failures breakdown:**
- `kernel/src/commit_range/mod.rs`- `commit_range` (line 19): Compile failure (already fixed unresolved imports, may be CI-related~
- `kernel/src/schema/mod.rs` - 4 doc test failures: root cause unresolved crate::Error and crate::schema in examples
- `kernel/src/table_changes/mod.rs` - 2 doc test failures: similar packaging issues

**Root Causes Identified:**
1. **Module naming conflicts:** Doc examples use `delta_kernel` crate name when package is published as `buoyant_kernel`
2. **Hybrid CI context:** Some examples work in CI (external crate) vs dev (crate:: access)
3. **Missing imports:** `crate::Error` doesn't exist at root level
4. **Schema module conflicts:** `crate::schema::{...}` vs `buoyant_kernel::schema::{...}` inconsistency

---

## Phase 1: Schema Crate Import Resolution

### Task 1.1: Fix crate::Error resolution issues

**Objective:** Replace problematic `crate::Error` imports with working alternatives
**Files:** `src/schema/mod.rs:1357`, `src/schema/mod.rs:1431`

**Step 1:** Identify correct Error type usage in context
```bash
cd /home/tyler/source/github/delta-io/delta-kernel-rs/kernel
grep -n "use.*Error" src/schema/mod.rs | head -10
```

**Step 2:** Replace `use crate::Error;` with `use buoyant_kernel::Error;`
```bash
# Existing import line 1357 and 1431
use crate::Error;
# Should become
use buoyant_kernel::Error;
```

**Step 3:** Verify no similar errors remain
```bash
grep -n "crate::Error" src/**/*.rs || echo "No crate::Error imports found"
```

**Verification:**
```bash
cargo test --doc --quiet 2>&1 | grep -c "unresolved import\|crate::Error"
# Expected: 0
```

### Task 1.2: Fix crate::schema resolution issues

**Objective:** Replace problematic `use crate::schema::{...};` imports
**Files:** `src/schema/mod.rs:1358`, `src/schema/mod.rs:1432`

**Step 1:** Check if schema module exports structs locally
```bash
grep -A 5 "pub.*struct.*StructType\|pub.*mod.*schema" src/schema/mod.rs | head -10
```

**Step 2:** Replace with `use buoyant_kernel::schema::{StructType, StructField, DataType};`
```bash
# Lines 1358 and 1432
use crate::schema::{StructType, StructField, DataType};
# Should become
use buoyant_kernel::schema::{StructType, StructField, DataType};
```

**Step 3:** Clean up any redundant imports
```bash
cargo check --lib
```

**Verification:**
```bash
cargo test --doc --quiet 2>&1 | grep -c "crate::schema"
# Expected: 0
```

---

## Phase 2: Table Changes Doc Test Fixes

### Task 2.1: Identify table_changes failures

**Objective:** Analyze(`line 4` and `line 97`) doc test failures
**Files:** `src/table_changes/mod.rs:4`, `src/table_changes/mod.rs:97`

**Step 1:** Check failing functions
```bash
sed -n '1,10p' src/table_changes/mod.rs
sed -n '95,105p' src/table_changes/mod.rs
```

**Step 2:** Look for `delta_kernel` imports in doc comments
```bash
grep -n "delta_kernel::" src/table_changes/mod.rs
grep -n "crate::" src/table_changes/mod.rs
```

**Step 3:** Apply same fix: replace `delta_kernel::` with `crate::` or `buoyant_kernel::`
```bash
# If found, apply sed or patch
```

**Verification:**
```bash
cargo test --doc --table-changes 2>&1 | grep -c "unresolved import"
```

### Task 2.2: Test targeted module

```bash
cargo test --doc table_changes
# Expect: test result pass
```

---

## Phase 3: Comprehensive Resolution

### Task 3.1: Global pattern replacement

**Objective:** Ensure no `delta_kernel` doc imports remain unfixed

```bash
# Find all remaining delta_kernel imports in doc comments
find src -name "*.rs" -exec grep -l "use delta_kernel" {} \;
```

**Action:** Apply fix systematically:
- In public-facing doc examples (published artefact): `use buoyant_kernel::`  
- In internal-only doc tests (crate context): `use crate::`

**Decision Rule:**
```
# For lines with: use buoyant_kernel as delta_kernel;
# Use: delta_kernel::module::Type or buoyant_kernel::module::Type
```

**Step:** Globally replace `delta_kernel::` with system-determined correct path

**Verification:**
```bash
cargo test --doc > /tmp/doc_test_results.log
grep -c "unresolved import" /tmp/doc_test_results.log
```

---

## Validation & Verification

### Task 4.1: Full suite execution

**Command:**
```bash
cd /home/tyler/source/github/delta-io/delta-kernel-rs/kernel
date; cargo test --doc 2>&1 | tee /tmp/final_doc_test.log
```

**Check output:**
```bash
# Parse results
grep "test result:" /tmp/final_doc_test.log
grep -c FAILED /tmp/final_doc_test.log
echo "FAILED COUNT: $(grep -c FAILED /tmp/final_doc_test.log)"
echo "PASSED COUNT: $(grep -c "test.*ok$" /tmp/final_doc_test.log)"
```

**Expected:** 60+ passed, 0 failed (down from 7)

### Task 4.2: Documentation Maintenance Guide

**Create guide:**
```bash
cat > docs/DOC_TEST_GUIDE.md << 'EOF'
# Committing Doc Test Changes

## Rules for Imports
- **Never** use `delta_kernel` in doc examples (CI-only artifact name)
- Use **`buoyant_kernel`** for public-facing examples (crate name)  
- Use **`crate::`** for internal-only dev examples

## Testing Locally
```bash
cargo test --doc
cargo test --doc --quiet
```

## Context Switching
- In dev: examples resolve to `crate::`
- In CI: examples resolve to external `buoyant_kernel` crate
- Use `#[cfg_attr(doc, doc = include_str!("../../FOOBAR"))]` to load different content

EOF
```

---

## Current State & Progress

**Already Fixed:** ✓
- `src/commit_range/mod.rs`: delta_kernel → crate for imports, health warnings removed  
- `src/schema/mod.rs`: Multiple delta_kernel references → crate::, but introduced `crate::Error` regression

**Pending:**  ✗
- `crate::Error` → `buoyant_kernel::Error` in schema examples
- `crate::schema::` → `buoyant_kernel::schema::` consistency
- Table changes module similar fixes

## Execution Constraint

Maximum per-turn limit: **50 tool calls**. Use batch mode sparingly. This plan breaks tasks into **7 separate execute_code** windows, each under 10 calls.

## Next Actions

 Ready to execute **Phase 1 > Task 1.1** (Fix crate::Error). Call `@agent/delegate_task()` with the step-by-step plan above.
