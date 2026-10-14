#!/bin/bash
echo "=== Package Dependencies ==="
grep -n "^\[package\]" --include="Cargo.toml" -A 2 -r . 2>/dev/null | head -30

echo ""
echo "=== Internal Dependencies ==="
grep -n "path.*kernel" --include="Cargo.toml" -r . 2>/dev/null | head -20

echo ""
echo "=== External Dependencies ==="
grep -n "^\w\+ = \{.*path" --include="Cargo.toml" -r . 2>/dev/null | head -20