#!/bin/bash
# Pre-tool validation hook for The Analyst
# Validates file paths before Write/Edit operations

# This hook runs BEFORE tool execution to prevent:
# 1. Writes to source data directories
# 2. Writes outside allowed paths
# 3. Potentially dangerous file operations

set -e

INPUT="${1:-}"

# Check for attempts to write to source data directory
if echo "$INPUT" | grep -qE "data/raw|data\/raw"; then
    echo "ERROR: Cannot write to source data directory (data/raw/)"
    echo "Source data is read-only. Use data/processed/ or data/outputs/ instead."
    exit 1
fi

# Check for attempts to write to sensitive system paths
if echo "$INPUT" | grep -qE "^/etc|^/usr|^/bin|^/sbin|^/var|^/root"; then
    echo "ERROR: Cannot write to system directories"
    exit 1
fi

# Check for hidden config files that shouldn't be modified
if echo "$INPUT" | grep -qE "\.env|\.git/|\.ssh/|credentials|secret|password"; then
    echo "WARNING: Potential modification of sensitive file detected"
    echo "File: $INPUT"
    echo "Please confirm this is intentional."
fi

# All checks passed
exit 0
