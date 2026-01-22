#!/bin/bash
# Post-tool verification hook for The Analyst
# Verifies that outputs meet quality standards

# This hook runs after tool executions to verify:
# 1. Source data was not modified
# 2. Outputs have required fields
# 3. Statistical claims include confidence intervals

set -e

# Get the tool output from stdin (if available)
OUTPUT="${1:-}"

# Check for data modification attempts
if echo "$OUTPUT" | grep -q "data/raw.*write\|data/raw.*modify\|data/raw.*delete"; then
    echo "ERROR: Attempted modification of source data detected"
    exit 1
fi

# Check for missing methodology in analysis outputs
if echo "$OUTPUT" | grep -qi "analysis\|statistical"; then
    if ! echo "$OUTPUT" | grep -qi "methodology\|method"; then
        echo "WARNING: Analysis output may be missing methodology section"
    fi
fi

# Check for missing confidence intervals in predictions
if echo "$OUTPUT" | grep -qi "forecast\|predict\|model"; then
    if ! echo "$OUTPUT" | grep -qi "confidence\|interval\|uncertainty"; then
        echo "WARNING: Prediction output may be missing confidence intervals"
    fi
fi

exit 0
