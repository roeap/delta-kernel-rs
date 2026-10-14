#!/usr/bin/env python3

import re
import os
import subprocess

def read_file(filepath):
    """Read file content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """Write content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def restore_file(filepath):
    """Restore file from git"""
    try:
        subprocess.run(['git', 'checkout', 'HEAD', '--', filepath], cwd='.', check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def fix_doctests_in_file(filepath):
    """Fix doctests in a single file by adding 'use buoyant_kernel as delta_kernel;'"""
    content = read_file(filepath)
    
    # Look for doctest blocks (start with /// ``` or //! ```)
    # We need to add the import alias at the beginning of each doctest block
    
    # Pattern to match the start of doctest blocks
    # Match lines like: /// ```, /// ```rust, /// ```compile_fail, etc.
    doctest_pattern = r'(///|/// {[^}]+}|//!|//! {[^}]+})\s*```([^\n]*)\n(?!\s*# use buoyant_kernel as delta_kernel;)'
    
    # Foreach match, we want to insert the import alias after the opening line
    # But only if the next line isn't already the import alias
    
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts a doctest block
        doctest_match = re.match(r'^(///|/// {[^}]+}|//!|//! {[^}]+})\s*```(.*)$', line)
        
        if doctest_match:
            # This is the start of a doctest block
            # Check if this doctest is one we should modify (contains delta_kernel imports)
            
            # Look ahead to see if this doctest contains delta_kernel usage
            lookahead = i + 1
            contains_delta_kernel = False
            
            while lookahead < len(lines) and lines[lookahead].strip().startswith('///'):
                if 'delta_kernel' in lines[lookahead]:
                    contains_delta_kernel = True
                    break
                lookahead += 1
            
            if contains_delta_kernel:
                # Add a hint line after the opening before we continue
                fixed_lines.append(line)  # Original line
                
                # Check if next line is not already our import
                next_line_idx = i + 1
                if next_line_idx < len(lines):
                    next_line = lines[next_line_idx]
                    next_line_content = next_line.strip()
                    
                    if next_line_content != '/// # use buoyant_kernel as delta_kernel;':
                        # Insert our import alias
                        # Try to maintain similar indentation
                        leading_spaces = re.match(r'^(\s+)', line)
                        if leading_spaces:
                            indent = leading_spaces.group(1)
                        else:
                            indent = ''
                        
                        # Determine if this is a fixed-width doc comment
                        if '///' in line or '///' in line.split('{')[0]:
                            prefix = '/// '
                        elif '///' in line:
                            prefix = '/// '
                        elif '/!' in line:
                            prefix = '/! '
                        else:
                            prefix = '/// '
                        
                        insertion_line = f"{prefix}# use buoyant_kernel as delta_kernel;"
                        fixed_lines.append(insertion_line)
                        i += 1  # Skip the next line we haven't processed yet
                else:
                    i += 1
            else:
                # Doctest doesn't use delta_kernel, leave as-is
                fixed_lines.append(line)
                i += 1
                
        else:
            # Regular line, just add it
            fixed_lines.append(line)
            i += 1
    
    return '\n'.join(fixed_lines)

def main():
    # List of files with failing doctests (from the cargo test output)
    failing_files = [
        ("src/actions/deletion_vector_writer.rs", [23, 113, 193]),
        ("src/checkpoint/mod.rs", [35]),
        ("src/commit_range/mod.rs", [19]),
        ("src/doctests/into_engine_data.rs", [14]),
        ("src/doctests/to_schema.rs", [3, 21, 32, 58]),
        ("src/engine_data.rs", [478]),
        ("src/expressions/column_names.rs", [30, 78, 103, 44, 182, 189, 197, 249, 258, 366, 399, 450]),
        ("src/expressions/mod.rs", [42]),
        ("src/lib.rs", [289, 335, 351, 367]),
        ("src/log_compaction/mod.rs", [32]),
        ("src/metrics/mod.rs", [13, 41, 118]),
        ("src/parallel/parallel_scan_metadata.rs", [147]),
        ("src/scan/mod.rs", [1045]),
        ("src/schema/mod.rs", [74, 91, 1411, 1484]),
        ("src/snapshot/builder.rs", [20]),
        ("src/table_changes/mod.rs", [4, 97]),
        ("src/table_changes/scan.rs", [44]),
        ("src/transaction/builder/create_table.rs", [762, 805]),
        ("src/transaction/create_table.rs", [9, 67, 100]),
        ("src/transaction/update.rs", [151]),
        ("src/transforms/expression.rs", [69]),
        ("src/transforms/mod.rs", [17, 34]),
        ("src/transforms/schema.rs", [63]),
        ("src/utils.rs", [27]),
    ]
    
    # Process each file
    for filepath, lines in failing_files:
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        
        print(f"📁 Processing {filepath}")
        
        # Restore from git first to avoid duplicate modifications
        if restore_file(filepath):
            print(f"  ✅ Restored from git")
        
        # Read and fix the file
        original_content = read_file(filepath)
        fixed_content = fix_doctests_in_file(filepath)
        
        if original_content != fixed_content:
            write_file(filepath, fixed_content)
            print(f"  ✅ Fixed doctests in {filepath}")
        else:
            print(f"  ℹ️  No changes needed for {filepath}")
        
        print()

if __name__ == "__main__":
    main()