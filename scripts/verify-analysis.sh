#!/bin/bash
# Verification script for analysis outputs
# Ensures all outputs meet quality standards

set -e

echo "=== The Analyst - Output Verification ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Function to check file
check_file() {
    local file="$1"
    local check="$2"
    local message="$3"

    if grep -qi "$check" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $message"
        return 0
    else
        echo -e "${RED}✗${NC} $message"
        return 1
    fi
}

# Function to warn
warn_check() {
    local file="$1"
    local check="$2"
    local message="$3"

    if ! grep -qi "$check" "$file" 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} $message"
        ((WARNINGS++))
    fi
}

# Check for output files
OUTPUT_DIR="${1:-data/outputs}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Output directory not found: $OUTPUT_DIR"
    exit 0
fi

# Find recent output files
OUTPUT_FILES=$(find "$OUTPUT_DIR" -type f \( -name "*.md" -o -name "*.html" -o -name "*.json" \) -mmin -60 2>/dev/null)

if [ -z "$OUTPUT_FILES" ]; then
    echo "No recent output files found."
    exit 0
fi

echo "Checking recent output files..."
echo ""

for file in $OUTPUT_FILES; do
    echo "Checking: $file"

    # Check for methodology section
    if ! check_file "$file" "methodology\|method" "Has methodology section"; then
        ((ERRORS++))
    fi

    # Check for confidence intervals in predictions
    if grep -qi "forecast\|predict\|model" "$file" 2>/dev/null; then
        if ! check_file "$file" "confidence\|interval\|uncertainty" "Has confidence intervals"; then
            ((ERRORS++))
        fi
    fi

    # Warn if missing data source
    warn_check "$file" "source\|data from" "Should include data source"

    # Warn if missing limitations
    warn_check "$file" "limitation\|caveat\|note that" "Should include limitations"

    echo ""
done

# Summary
echo "=== Verification Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Verification FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Verification PASSED${NC}"
    exit 0
fi
