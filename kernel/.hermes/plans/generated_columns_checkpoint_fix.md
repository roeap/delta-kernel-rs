# Generated Columns Checkpoint Fix Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Allow checkpoint creation for tables with `generatedColumns` writer feature while maintaining the restriction on writing data to such tables

**Architecture:** The fix will separate feature validation for checkpoint operations from regular write operations, allowing unsupported writer features that don't affect checkpoint integrity.

**Tech Stack:** Rust, Delta Lake Kernel, Delta Protocol

---

## Current Context

- Commit 6699a2565 introduced validation that prevents checkpoint creation on tables with unsupported writer features
- `generatedColumns` is marked as `KernelSupport::NotSupported` in `IDENTITY_COLUMNS_INFO`  
- This validation is too strict for checkpoint operations, which only need read capability
- The kernel should allow checkpointing but not writing data to tables with unsupported writer-only features

## Proposed Approach

1. Create a new validation method specifically for checkpoint operations
2. Modify checkpoint-related code to use this new validation
3. Update existing tests to reflect the new behavior
4. Ensure regular write operations still reject unsupported features

---

## Task 1: Create Checkpoint-Specific Feature Validation

**Objective:** Add a new validation method that allows unsupported writer-only features for checkpoint operations

**Files:**
- Create: `kernel/src/table_configuration.rs` (modify existing file)
- Test: `kernel/tests/unit/table_configuration_tests.rs`

**Step 1: Write failing test for checkpoint feature validation**

```rust
#[test]
fn test_ensure_checkpoint_supported_allows_generated_columns() {
    let table_config = MockTableConfigurationBuilder::new()
        .with_protocol(
            MockProtocolBuilder::new()
                .with_features([TableFeature::GeneratedColumns])
                .build(),
        )
        .build();
    
    // Should succeed for checkpoint operations
    let result = table_config.ensure_checkpoint_supported();
    assert!(result.is_ok());
    
    // Should still fail for regular write operations
    let write_result = table_config.ensure_operation_supported(Operation::Write);
    assert_result_error_with_message(write_result, "Feature 'generatedColumns' is not supported");
}
```

**Step 2: Run test to verify failure**

Run: `cargo test test_ensure_checkpoint_supported_allows_generated_columns -p delta_kernel --lib`
Expected: FAIL - "method not found" or similar

**Step 3: Implement ensure_checkpoint_supported method**

```rust
impl TableConfiguration {
    /// Ensures Kernel supports checkpoint operations on this table.
    /// This is more permissive than ensure_write_supported, allowing tables with
    /// unsupported writer-only features since checkpointing only requires read capability.
    pub(crate) fn ensure_checkpoint_supported(&self) -> DeltaResult<()> {
        // Checkpoint operations only need read support
        self.ensure_read_supported(Operation::Scan)
        
        // Writer features are not validated for checkpoint operations since
        // checkpoints only read existing data and don't write new table data
    }
}
```

**Step 4: Run test to verify pass**

Run: `cargo test test_ensure_checkpoint_supported_allows_generated_columns -p delta_kernel --lib`
Expected: PASS

**Step 5: Commit**

```bash
git add kernel/src/table_configuration.rs
cargo test test_ensure_checkpoint_supported_allows_generated_columns -p delta_kernel --lib
git commit -m "feat: add checkpoint-specific feature validation"
```

---

## Task 2: Update Checkpoint Code to Use New Validation

**Objective:** Modify checkpoint creation to use the new checkpoint-specific validation

**Files:**
- Modify: `kernel/src/checkpoint/mod.rs`
- Modify: `kernel/src/snapshot/mod.rs`
- Test: `kernel/tests/integration/features/maintenance_ops.rs`

**Step 1: Write failing integration test for generatedColumns checkpoint**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_checkpoint_allows_generated_columns() -> Result<(), Box<dyn std::error::Error>> {
    let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
    let table_url = Url::from_directory_path(&table_path).unwrap();
    let store = LocalFileSystem::new();
    let schema_string = json!({
        "type": "struct",
        "fields": [{
            "name": "id",
            "type": "integer",
            "nullable": true,
            "metadata": {},
        }],
    })
    .to_string();
    
    let commit = [
        json!({
            "protocol": {
                "minReaderVersion": 3,
                "minWriterVersion": 7,
                "readerFeatures": [],
                "writerFeatures": ["generatedColumns"],
            }
        }),
        json!({
            "metaData": {
                "id": "generated-columns-checkpoint-test",
                "format": { "provider": "parquet", "options": {} },
                "schemaString": schema_string,
                "partitionColumns": [],
                "configuration": {},
                "createdTime": 1_700_000_000_000i64,
            }
        }),
    ]
    .map(|action| action.to_string())
    .join("\n");
    
    add_commit(table_url.as_str(), &store, 0, commit).await?;

    let snapshot = Snapshot::builder_for(table_url).build(engine.as_ref())?;
    
    // This should succeed - checkpointing should work with generatedColumns
    let checkpoint_result = snapshot.checkpoint(engine.as_ref(), None);
    assert!(checkpoint_result.is_ok());
    
    // Writing data should still be rejected
    let mut tx = snapshot.subscribe(engine.as_ref())?.transaction();
    let temp_file = temp_dir.path().join("temp.parquet");
    let mut file_w = FileWriter::try_new(&temp_file)?;
    let write_result = tx.write_data(&[AddAction::test_add()], &mut file_w);
    assert_result_error_with_message(write_result, "generatedColumns");
    
    Ok(())
}
```

**Step 2: Run test to verify failure**

Run: `cargo test test_checkpoint_allows_generated_columns --test maintenance_ops`
Expected: FAIL - should reject generatedColumns

**Step 3: Update checkpoint writer to use new validation**

Modify `kernel/src/checkpoint/mod.rs`:
```rust
impl CheckpointWriter {
    pub(crate) fn try_new(snapshot: SnapshotRef, engine: &dyn Engine) -> DeltaResult<Self> {
        // Replaced: snapshot.table_configuration().ensure_read_write_supported()?;
        snapshot.table_configuration().ensure_checkpoint_supported()?;
        
        // Rest of method continues...
```

Modify `kernel/src/snapshot/mod.rs`:
```rust
impl Snapshot {
    pub fn write_checksum(&self, engine: &dyn Engine) -> DeltaResult<ChecksumWriteResult> {
        // Replaced: self.table_configuration().ensure_read_write_supported()?;
        self.table_configuration().ensure_checkpoint_supported()?;
        
        // Rest of method continues...
```

**Step 4: Run test to verify pass**

Run: `cargo test test_checkpoint_allows_generated_columns --test maintenance_ops`
Expected: PASS

**Step 5: Commit**

```bash
git add kernel/src/checkpoint/mod.rs kernel/src/snapshot/mod.rs
git commit -m "feat: use checkpoint-specific validation for maintenance operations"
```

---

## Task 3: Verify Existing Tests Still Pass

**Objective:** Ensure the change doesn't break existing functionality

**Files:**
- Test: All existing tests

**Step 1: Run integration tests for existing checkpoint functionality**

```bash
cargo nextest r checkpoint -p delta_kernel
```

**Step 2: Run all maintenance operations tests**

```bash
cargo nextest r maintenance_ops -p delta_kernel
```

**Step 3: Run table configuration tests**

```bash
cargo nextest r table_configuration -p delta_kernel
```

**Step 4: Verify all tests pass**

**Step 5: Commit verification**

```bash
git commit -m "test: verify checkpoint validation changes"
```

---

## Task 4: Add Additional Test Cases for Other Unsupported Features

**Objective:** Ensure the fix works for similar writer-only features

**Files:**
- Test: `kernel/tests/integration/features/maintenance_ops.rs`

**Step 1: Create comprehensive test for multiple unsupported writer features**

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_checkpoint_allows_unsupported_writer_features() -> Result<(), Box<dyn std::error::Error>> {
    let unsupported_writer_features = [
        "generatedColumns",
        "identityColumns",
        "checkConstraints",
    ];
    
    for &feature in &unsupported_writer_features {
        let (_temp_dir, table_path, engine) = test_table_setup_mt()?;
        let table_url = Url::from_directory_path(&table_path).unwrap();
        let store = LocalFileSystem::new();
        
        let schema_string = json!({
            "type": "struct",
            "fields": [{
                "name": "id",
                "type": "integer",
                "nullable": true,
                "metadata": {},
            }],
        })
        .to_string();
        
        let commit = [
            json!({
                "protocol": {
                    "minReaderVersion": 3,
                    "minWriterVersion": 7,
                    "readerFeatures": [],
                    "writerFeatures": [feature],
                }
            }),
            json!({
                "metaData": {
                    "id": "unsupported-features-checkpoint-test",
                    "format": { "provider": "parquet", "options": {} },
                    "schemaString": schema_string,
                    "partitionColumns": [],
                    "configuration": {},
                    "createdTime": 1_700_000_000_000i64,
                }
            }),
        ]
        .map(|action| action.to_string())
        .join("\n");
        
        add_commit(table_url.as_str(), &store, 0, commit).await?;

        let snapshot = Snapshot::builder_for(table_url).build(engine.as_ref())?;
        
        // Checkpoint should succeed
        let checkpoint_result = snapshot.checkpoint(engine.as_ref(), None);
        assert!(checkpoint_result.is_ok(), "Checkpoint failed for feature: {}", feature);
        
        // Checksum should also succeed
        let checksum_result = snapshot.write_checksum(engine.as_ref());
        assert!(checksum_result.is_ok(), "Checksum failed for feature: {}", feature);
        
        // Writing should still be rejected
        let mut tx = snapshot.subscribe(engine.as_ref())?.transaction();
        let temp_file = temp_dir.path().join("temp.parquet");
        let mut file_w = FileWriter::try_new(&temp_file)?;
        let write_result = tx.write_data(&[AddAction::test_add()], &mut file_w);
        assert_result_error_with_message(write_result, feature);
    }
    
    Ok(())
}
```

**Step 2: Run comprehensive test**

```bash
cargo test test_checkpoint_allows_unsupported_writer_features --test maintenance_ops
```

**Step 3: Verify all tests pass**

**Step 4: Commit**

```bash
git commit -m "test: add comprehensive tests for unsupported writer features"
```

---

## Task 5: Documentation Update

**Objective:** Update documentation to reflect the new behavior

**Files:**
- Modify: `kernel/src/table_configuration.rs` (decorations)

**Step 1: Add comprehensive documentation**

```rust
/// Ensures Kernel supports checkpoint operations on this table.
/// 
/// This validation is distinct from `ensure_write_supported` and `ensure_read_write_supported`
/// because checkpoint operations are fundamentally read-only operations that only validate
/// the integrity of existing data. They do not need to understand writer-only features
/// that wouldn't affect the correctness of reading existing table files.
///
/// This method validates:
/// - Reader protocol version is supported
/// - Reader features are supported by kernel
/// - Table is in a valid state for snapshot operations
///
/// Unlike regular write operations, this does NOT validate:
/// - Writer features (since checkpoints only read existing data)
/// - Schema invariants for write compatibility
///
/// Example: A table with the `generatedColumns` writer feature can be checkpointed even though
/// the kernel cannot write new data to such tables, because the existing data can still be
/// read and validated correctly.
///
/// # Returns
/// 
/// `Ok(())` if checkpoint operations are supported, or an `Error` describing the specific
/// incompatibility.
pub(crate) fn ensure_checkpoint_supported(&self) -> DeltaResult<()> {
    self.ensure_read_supported(Operation::Scan)
}
```

**Step 2: Update related method documentation**

```rust
pub(crate) fn ensure_read_write_supported(&self) -> DeltaResult<()> {
    /// Ensures Kernel supports both scanning and writing this table.
    ///
    /// This is used for operations that require both read and write capability,
    /// such as transaction commits or data insertion. For checkpoint/cleanup operations
    /// that only read existing data, use `ensure_checkpoint_supported` instead.
    ///
    /// See also: [`ensure_checkpoint_supported`] for maintenance operations.
    self.ensure_operation_supported(Operation::Scan)?;
    self.ensure_operation_supported(Operation::Write)
}
```

**Step 3: Commit**

```bash
git commit -m "docs: document checkpoint validation behavior"
```

---

## Verification and Testing Strategy

### Unit Tests
1. `test_ensure_checkpoint_supported_allows_generated_columns` - Core functional test
2. `test_checkpoint_allows_generated_columns` - Integration test  
3. `test_checkpoint_allows_unsupported_writer_features` - Comprehensive test for similar features

### Integration Tests
1. Run all existing checkpoint tests to ensure no regressions
2. Verify that regular write operations still reject unsupported features
3. Test checkpoint creation with various writer-only features

### Manual Verification
```bash
# Create a test table with generatedColumns feature
# Attempt to create a checkpoint (should succeed)
# Attempt to write data (should fail)
# Verify checksum creation works (should succeed)
```

---

## Risks and Trade-offs

### Risks
1. **Overly permissive validation**: The new method could allow checkpoints on tables with truly problematic features
2. **Missing validation**: Some writer features might actually affect checkpoint integrity
3. **Breaking existing behavior**: The change modifies established validation patterns

### Mitigations
1. Thorough testing of various feature combinations
2. Careful analysis of feature semantics before applying the new validation
3. Maintaining strict validation for regular write operations

### Trade-offs
1. **Safety vs Functionality**: Relaxing validation for checkpoint operations improves usability but could allow edge cases
2. **Complexity vs Maintainability**: Adding a new validation method increases code complexity but improves clarity
3. **Backward Compatibility**: The change could affect existing code that expects strict validation

---

## Open Questions

1. Are there any writer-only features that should actually block checkpoint operations?
2. Should this approach be extended to other maintenance operations (e.g., OPTIMIZE, ZORDER)?
3. Does this change need to be reflected in the public API documentation?

---

## Implementation Summary

This plan proposes a targeted fix that separates feature validation for checkpoint operations from regular write operations, allowing the Delta Kernel to create checkpoints for tables with unsupported writer-only features like `generatedColumns` while maintaining strict validation for actual data write operations.

The approach preserves safety by still requiring read support for checkpoint operations while recognizing that checkpoint operations are fundamentally read-only and don't need to understand writer features that wouldn't affect data integrity.

Estimated implementation time: 2-3 hours
