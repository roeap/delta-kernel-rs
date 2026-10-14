# Final Doc Test Fix - Packaging Resolution

> **For Hermes:** Execute this plan task-by-task using delegate_task.

**Goal:** Resolve all outstanding doc test failures (7/54 failing: 2 from commit_range, 4 from schema, 1 from table_changes).

**Architecture:** Two phases:
1. Fix remaining crate:: import resolution issues in schema/mod.rs
2. Update failed doc tests to use correct packaging references

**Tech Stack:** Cargo/Rust, Doc Test System

---

## Phase 1: Root Cause Analysis

* **54 passed**, 7 failed, 34 ignored
* **Pattern:**, Missing crate::Error, crate::schema in servicing imported modules
* **Root:** Incons ...[truncated]