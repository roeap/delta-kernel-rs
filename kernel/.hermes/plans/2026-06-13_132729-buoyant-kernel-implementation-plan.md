# Buoyant Kernel Migration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reimplement patch `54dc148d09afc7773b6721f545435e1e66bcdb4f` to migrate from `delta_kernel` to `buoyant_kernel` and create `buoyant_engine` crate.

**Architecture:** Clean separation between kernel (data structures, traits, core logic) and engine (concrete implementations, async runtime, storage). Use module aliases to minimize code disruption.

**Tech Stack:** Rust 1.91+, Cargo workspaces, procedural macros, async/await, object_store, Arrow, Tokio.

---

## Task 1: Analyze Dependency Graph and Create Migration Script

**Objective:** Understand current dependency relationships and create tools for bulk migration.

**Files:**
- Create: `.hermes/scripts/migrate_imports.py`
- Create: `.hermes/scripts/bzip2_deps.sh`
- Read: `kernel/Cargo.toml`
- Read: `default-engine/Cargo.toml`

**Step 1: Create dependency analysis script**

```bash
#!/bin/bash
# bzip2_deps.sh
echo "=== Package Dependencies ==="
grep -n "^\[package\]" --include="Cargo.toml" -A 2 -r .

echo "=== Internal Dependencies ==="
grep -n "path.*kernel" --include="Cargo.toml" -r .

echo "=== External Dependencies ==="
grep -n "^\w\+ = \{.*path" --include="Cargo.toml" -r .
```

**Step 2: Run cargo tree to understand dependency structure**

```bash
cargo tree --depth 3 --workspace --invert | grep -E "(delta_kernel|default_engine)" | head -50
```

**Step 3: Commit analysis tooling**

```bash
git add .hermes/scripts/
git commit -m "feat: add migration analysis scripting tools"
```

---

## Task 2: Create Derive Macro Duplicate with Buoyant Name

**Objective:** Establish `buoyant_kernel_derive` alongside existing `delta_kernel_derive`

**Files:**
- Create: `buoyant-derive-macros/` directory
- Create: `buoyant-derive-macros/Cargo.toml`
- Modify: `derive-macros/Cargo.toml`

**Step 1: Create new derive crate**

```toml
# buoyant-derive-macros/Cargo.toml
[package]
name = "buoyant_kernel_derive"
description = "Procedural macros for buoyant_kernel"
edition = "2021"
version = "0.24.0" // match existing version

[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full"] }
quote = "1"
proc-macro2 = "1"

// Match the original macro functionality exactly
// by copying the source files
```

**Step 2: Update existing derive crate to provide both names**

```toml
[package]
name = "delta_kernel_derive"
description = "Procedural macros for delta_kernel (legacy name)"
# Add alias crate feature
[package.metadata.aliases]
buoyant = { package = "buoyant_kernel_derive", path = "../buoyant-derive-macros" }
```

**Step 3: Build derive crates**

```bash
cargo build -p buoyant_kernel_derive
cargo build -p delta_kernel_derive
```

---

## Task 3: Rename Kernel Crate and Update Metadata

**Objective:** Change `kernel/Cargo.toml` from `delta_kernel` to `buoyant_kernel`

**Files:**
- Modify: `kernel/Cargo.toml`
- Modify: `Cargo.toml` (workspace)

**Step 1: Backup original workload configuration**

```bash
cp kernel/Cargo.toml kernel/Cargo.toml.backup
```

**Step 2: Rename main package**

```toml
[package]
name = "buoyant_kernel"
description = "Buoyant Data distribution of delta-kernel"
documentation = "https://docs.rs/buoyant_kernel"
```

**Step 3: Update internal dependencies**

```toml
[dependencies]
delta_kernel_derive = { package = "buoyant_kernel_derive", path = "../buoyant-derive-macros", version = "1" }

[dev-dependencies]
delta_kernel = { package = "buoyant_kernel", path = ".", features = ["test-utils"] }
```

**Step 4: Update workspace to include new crate**

```toml
[members]
    "acceptance",
    "benchmarks",
    "buoyant-engine",
    "buoyant-derive-macros",
    "derive-macros",  # keep for transition
    # ... other members
```

**Step 5: Build kernel crate**

```bash
cd kernel
cargo build --no-default-features
```

---

## Task 4: Create Buoyant Engine Crate

**Objective:** Create `buoyant_engine` crate to replace `delta_kernel_default_engine`

**Files:**
- Create: `buoyant-engine/`
- Create: `buoyant-engine/Cargo.toml`
- Create: `buoyant-engine/src/lib.rs`
- Modify: `Cargo.toml` (workspace)

**Step 1: Create new engine directory structure**

```bash
mkdir -p buoyant-engine/src
mkdir -p buoyant-engine/test-utils
```

**Step 2: Create Cargo.toml for engine**

```toml
# buoyant-engine/Cargo.toml
[package]
name = "buoyant_engine"
description = "Default Arrow/Tokio-based engine implementation for buoyant_kernel."
edition = "2021"
version = "0.24.0"

[dependencies]
buoyant_kernel = { path = "../kernel", version = "0.24.0", features = [
  "default-engine-base",
  "internal-api",
] }
bytes = "1.10"
tokio = { version = "1", features = ["rt-multi-thread"] }
object_store = { version = "0.13", features = ["aws", "azure", "gcp", "http"] }
arrow = { version = "58", features = ["chrono-tz", "ffi", "json"] }
parquet = { version = "58", features = ["async", "object_store"] }
reqwest = { version = "0.13", default-features = false, features = ["rustls"] }

[features]
default = ["arrow-conversion", "arrow-expression"]
arrow-conversion = []
arrow-expression = []

[dev-dependencies]
buoyant_engine_test_utils = { path = "./test-utils" }
test_utils = { path = "../test-utils" }
```

**Step 3: Copy core engine functionality**

```bash
CLI VoR -enamed-crate=df6ault-engine/src/  milename-Buoyant-engine-affect-the-Engine/TTrait-Impls-integers
rsync -av --exclude='target/' default-engine/src/ buoyant-engine/src/
```

**Step 4: Update imports in copied files**

```bash
find buoyant-engine/src -name "*.rs" -exec sed -i 's/delta_kernel/buoyant_kernel/g' {} \;
find buoyant-engine/src -name "*.rs" -exec sed -i 's/delta_kernel_default_engine/buoyant_engine/g' {} \;
```

**Step 5: Build new engine crate**

```bash
cd buoyant-engine
cargo build
```

---

## Task 5: Create Migration Script for Source Code Updates

**Objective:** Automate the majority of `delta_kernel` → `buoyant_kernel as delta_kernel` replacements

**Files:**
- Create: `.hermes/scripts/migrate_imports.py`

**Step 1: Create Python migration script**

```python
#!/usr/bin/env python3
import os
import re
import argparse

def migrate_file(filepath):
    """Migrate delta_kernel imports to buoyant_kernel aliases in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern 1: Simple use statements
    content = re.sub(
        r'usefunction delta_kernel::([^;]+);',
        r'use buoyant_kernel as delta_kernel;
 use delta_kernel::$1;',
        content
    )
    
    # Pattern 2: Simple delta_kernel imports
    content = re.sub(
        r'usefunction delta_kernel;`,
        r'use buoyant_kernel as delta_kernel;',
        content
    )
    
    # Pattern 3: In doc comments
    content = re.sub(
        r'(///\s*#\s*)use delta_kernel',
        r'\1use buoyant_kernel as delta_kernel;\n\1use delta_kernel',
        content
    )
    
    # Pattern 4: External crate references in test-ils
    if re.search(r'(test_lib|test_utils)::delta_kernel', content):
        content = re.sub(
            r'(test_\w+)::delta_kernel',
            r'\1::buoyant_kernel',
            content
        )
    
    # Pattern 5: Use delta_kernel_default_engine
    content = re.sub(
        r'use delta_kernel_default_engine',
        r'use buoyant_engine',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', help='Paths to migrate')
    args = parser.parse_args()
    
    for path in args.paths:
        for root, dirs, files in os.walk(path):
            # Skip target directories
            if 'target' in root:
                continue
            
            for file in files:
                if file.endswith('.rs'):
                    filepath = os.path.join(root, file)
                    print(f"Migrating {filepath}")
                    migrate_file(filepath)
```

**Step 2: Make script executable**

```bash
chmod +x .hermes/scripts/migrate_imports.py
```

**Step 3: Test migration on a single file**

```bash
python .hermes/scripts/migrate_imports.py kernel/examples/common/src/lib.rs
git diff kernel/examples/common/src/lib.rs
```

---

## Task 6: Apply Migration to Kernel Examples

**Objective:** Migrate all example code to use new import structure

**Files:**
- Modify: `kernel/examples/common/src/lib.rs`
- Modify: `kernel/examples/**/*.rs`

**Step 1: Run migration on examples**

```bash
python .hermes/scripts/migrate_imports.py kernel/examples/
```

**Step 2: Verify compilation**

```bash
cd kernel/examples/common
cargo build
```

**Step 3: Build all examples to ensure compatibility**

```bash
for example in kernel/examples/*; do
  if [ -d "$example" ]; then
    echo "Building $(basename $example)"
    (cd "$example" && cargo build) || break
  fi
done
```

---

## Task 7: Update Test Suite Imports

**Objective:** Migrate test files to use new crate names while preserving functionality

**Files:**
- Create: `fix_test_imports.sh`
- Modify: `kernel/tests/**/*.rs`
- Modify: `test-utils/src/**/*.rs`

**Step 1: Create test-specific migration script**

```bash
#!/bin/bash
# fix_test_imports.sh
find kernel/tests test-utils/src -name "*.rs" -print0 | while IFS= read -r -d '' file; do
    echo "Processing $file"
    
    # Fix test utils imports first
    sed -i 's/test_utils::delta_kernel/test_utils::buoyant_kernel/g' "$file"
    
    # Then fix regular delta kernel imports
    sed -i 's/use delta_kernel/use buoyant_kernel as delta_kernel;/g' "$file"
    
    # Fix user's delta kernel default engine imports with proper path
    sed -i 's/use delta_kernel_default_engine/use buoyant_engine/g' "$file"
    
    # Add alias if file tries to use delta_kernel directly
    if grep -q "delta_kernel::" "$file" && ! grep -q "use buoyant_kernel as delta_kernel" "$file"; then
        # Insert at import section (after use statements)
        sed -i '/^use.*;$/{
            H
            /\n$/!{
                s/./\n&/g
                suse delta_kernel::/buoyant_kernel::/g
            }
        }
        s/^import section/add alias sedang pindah mana-mana/' "$file"
        
        # More robust approach: add alias to beginning of file if needed
        if ! grep -q "use buoyant_kernel as delta_kernel" "$file"; then
            echo "Adding explicit alias to $file"
            (echo "use buoyant_kernel as delta_kernel;" && cat "$file") > "$file.tmp"
            mv "$file.tmp" "$file"
        fi
    fi
done
```

**Step 2: Run migration on tests**

```bash
chmod +x fix_test_imports.sh
./fix_test_imports.sh
```

**Step 3: Verify key test files compile**

```bash
cd kernel
export RUSTFLAGS="--cfg tokio_unstable"
mkdir -p tests/tmp
 COPYFILE_DISABLE=true cargo check --test incremental_scan 2>&1 | head -20
```

---

## Task 8: Migrate Integration Tests and Acceptance Suite

**Objective:** Update full test suite to work with new import structure

**Files:**
- Modify: `acceptance/**/*.rs`
- Modify: `benchmarks/**/*.rs`
- Modify: `integration-tests/**/*.rs`

**Step 1: Run migration scripts on integration components**

```bash
python .hermes/scripts/migrate_imports.py acceptance benchmarks integration-tests
```

**Step 2: Manually review high-risk areas**

```bash
# Review files that heavily use FFI/bindings
python -m pip install --user pygments
grep -r "cbindgen\|extern" --include="*.rs" ffi/ | head -10
```

**Step 3: Cross-check with Cargo.lock for version consistency**

```bash
# Analyse lock file symbols to find any delta_kernel references that need attention
grep -o "delta_kernel.*version" Cargo.lock | sort | uniq
```

---

## Task 9: Update FFI Bindings and Bridge Code

**Objective:** Ensure foreign function interface layers work with new crate layout

**Files:**
- Modify: `ffi/src/lib.rs`
- Modify: `ffi/Cargo.toml`

**Step 1: Migrate FFI Cargo.toml dependencies**

```bash
cd ffi
find . -name Cargo.toml -exec sed -i '' -e 's/delta_kernel/buoyant_kernel/g' {} \;
sed -i 's/delta_kernel_derive/buoyant_kernel_derive/g' Cargo.toml
```

**Step 2: Update FFI imports**

```bash
python .hermes/scripts/migrate_imports.py ffi/src
```

**Step 3: Build FFI crate**

```bash
cargo build -p buoyant_kernel_ffi
```

---

## Task 10: Update Documentation and Examples

**Objective:** Fix all READMEs, examples, and documentation references

**Files:**
- Modify: `*.md` files with delta_kernel references
- Modify: `kernel/examples/*/src/main.rs`

**Step 1: Find and replace in documentation**

```bash
find . -name "*.md" ! -path "*/target/*" -exec sed -i 's/delta_kernel/buoyant_kernel/g' {} \;
```

**Step 2: Create migration guide**

```markdown
# From delta_kernel to buoyant_kernel

Replace all imports: