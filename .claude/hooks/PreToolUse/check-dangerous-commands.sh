#!/bin/bash
# Pre-tool validation hook for The Analyst
# Checks for dangerous commands before Bash execution

# This hook runs BEFORE Bash execution to prevent:
# 1. Destructive file operations
# 2. System-level changes
# 3. Commands that could leak data

set -e

INPUT="${1:-}"

# Block recursive delete operations
if echo "$INPUT" | grep -qE "rm\s+-rf|rm\s+-fr|rm\s+--recursive.*--force"; then
    echo "ERROR: Recursive force delete operations are not allowed"
    exit 1
fi

# Block sudo operations
if echo "$INPUT" | grep -qE "^sudo|;\s*sudo|\|\s*sudo"; then
    echo "ERROR: sudo operations are not allowed"
    exit 1
fi

# Block operations that could modify source data
if echo "$INPUT" | grep -qE ">\s*data/raw|>>\s*data/raw|mv.*data/raw|cp.*data/raw"; then
    echo "ERROR: Cannot modify source data directory (data/raw/)"
    exit 1
fi

# Warn about external network operations
if echo "$INPUT" | grep -qE "curl.*-d|wget.*--post|curl.*POST"; then
    echo "WARNING: Detected potential data exfiltration via HTTP POST"
    echo "Command: $INPUT"
    echo "Ensure this is an authorized operation."
fi

# Warn about pip install from external sources
if echo "$INPUT" | grep -qE "pip install.*http|pip install.*git\+"; then
    echo "WARNING: Installing package from external source"
    echo "Command: $INPUT"
    echo "Ensure the source is trusted."
fi

# All checks passed
exit 0
