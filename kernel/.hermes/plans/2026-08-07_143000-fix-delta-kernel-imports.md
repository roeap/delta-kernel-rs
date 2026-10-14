# Fix delta_kernel Imports in Test Files Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `use buoyant_kernel as delta_kernel;` import to all test files that reference `delta_kernel` to properly reference the renamed package

**Architecture:** The package has been renamed from `delta_kernel` to `buoyant_kernel`, but test files still reference the old name. We need to add alias imports to maintain compatibility.

**Tech Stack:** Rust, Cargo, nextest

---

## Current Analysis

The analysis shows that the primary Cargo.toml package has been renamed from `delta_kernel` to `buoyant_kernel`, but various test files still import modules using the `delta_kernel` path. According to line 140, there's already a dev-dependency alias:

```toml
[dev-dependencies]
delta_kernel = { package = "buoyant_kernel", path = ".", features = ["test-utils"] }
```

This means `delta_kernel` is already aliased to `buoyant_kernel` in the build system. We need to add explicit alias imports at the top of test files to make this work properly for local development.

## Files Identified for Update

Based on the search results, these test files contain `delta_kernel` imports and would benefit from the alias:

```
tests/integration/data_skipping.rs
tests/integration/golden_tables.rs
tests/integration/hdfs.rs
tests/integration/main.rs
tests/integration/metrics_main.rs
tests/integration/parsed_partition_values.rs
tests/integration/parsed_stats.rs
tests/integration/read.rs
```

---

## Implementation Plan

### Task 1: Add Import Alias to data_skipping.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to data_skipping.rs

**Files:**
- Modify: `tests/integration/data_skipping.rs`

**Step 1: Read current file structure**
```bash
head -20 tests/integration/data_skipping.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 2: Add Import Alias to golden_tables.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to golden_tables.rs

**Files:**
- Modify: `tests/integration/golden_tables.rs`

**Step 1: Read current file structure**
```bash
head -20 tests/integration/golden_tables.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 3: Add Import Alias to hdfs.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to hdfs.rs

**Files:**
- Modify: `tests/integration/hdfs.rs`

**Step 1: Read current file structure**
```bash
head -20 tests/integration/hdfs.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 4: Add Import Alias to main.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to main.rs (integration test entry point)

**Files:**
- Modify: `tests/integration/main.rs`

**Step 1: Read current file structure**
```bash
head -30 tests/integration/main.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` near the top of the file

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 5: Add Import Alias to metrics_main.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to metrics_main.rs

**Files:**
- Modify: `tests/integration/metrics_main.rs`

**Step 1: Read current file structure**
```bash
head -20 tests/integration/metrics_main.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 6: Add Import Alias to parsed_partition_values.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to parsed_partition_values.rs

**Files:**
- Modify: `tests/integration/parsed_partition_values.rs`

**Step 1: Read current file structure**
```bash
head -20 tests/integration/parsed_partition_values.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 7: Add Import Alias to parsed_stats.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to parsed_stats.rs

**Files:**
- Modify: `tests/integration/parsed_stats.rs`

**Step 1: Read current file structure**
```bash
head -30 tests/integration/parsed_stats.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 8: Add Import Alias to read.rs

**Objective:** Add `use buoyant_kernel as delta_kernel;` import to read.rs

**Files:**
- Modify: `tests/integration/read.rs`

**Step 1: Read current file structure**
```bash
head -30 tests/integration/read.rs
```

**Step 2: Add alias import at top**
Add `use buoyant_kernel as delta_kernel;` after existing imports

**Step 3: Verify no syntax errors**
```bash
cargo check --lib 2>&1 | head -20
```

---

### Task 9: Verify All Tests Compile and Run

**Objective:** Run `cargo nextest r --lib` to verify all tests compile and run successfully

**Step 1: Check if nextest is available**
```bash
cargo nextest --version || echo "Nextest not installed"
```

**Step 2: Run library tests**
```bash
cargo nextest r --lib
```

Expected: All tests should compile and run successfully

**Step 3: Run full test suite** (if needed)
```bash
cargo nextest r
```

Expected: Complete test suite should pass

---

### Task 10: Final Verification

**Objective:** Verify the changes work by checking specific test files

**Step 1: Check data_skipping compilation**
```bash
cargo test --lib data_skipping 2>&1 | tail -10
```

**Step 2: Check read.rs compilation**
```bash
cargo test --lib read 2>&1 | tail -10
```

**Step 3: Check parsed_stats compilation**
```bash
cargo test --lib parsed_stats 2>&1 | tail -10
```

---