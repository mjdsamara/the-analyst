#!/bin/bash
# Data integrity verification script
# Ensures source data has not been modified

set -e

echo "=== The Analyst - Data Integrity Check ==="
echo ""

RAW_DIR="${1:-data/raw}"
CHECKSUM_FILE="${2:-memory/data/checksums.json}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

if [ ! -d "$RAW_DIR" ]; then
    echo "Raw data directory not found: $RAW_DIR"
    exit 0
fi

# Find all data files
DATA_FILES=$(find "$RAW_DIR" -type f \( -name "*.csv" -o -name "*.xlsx" -o -name "*.json" -o -name "*.parquet" \) 2>/dev/null)

if [ -z "$DATA_FILES" ]; then
    echo "No data files found in $RAW_DIR"
    exit 0
fi

# Create checksum file if it doesn't exist
if [ ! -f "$CHECKSUM_FILE" ]; then
    echo "Creating initial checksum file..."
    mkdir -p "$(dirname "$CHECKSUM_FILE")"
    echo "{}" > "$CHECKSUM_FILE"
fi

CHANGED=0
NEW=0

for file in $DATA_FILES; do
    # Compute current checksum
    CURRENT=$(shasum -a 256 "$file" | cut -d' ' -f1)
    FILENAME=$(basename "$file")

    # Check if we have a stored checksum
    STORED=$(python3 -c "
import json
import sys
try:
    with open('$CHECKSUM_FILE') as f:
        data = json.load(f)
    print(data.get('$FILENAME', ''))
except:
    print('')
" 2>/dev/null)

    if [ -z "$STORED" ]; then
        echo -e "${GREEN}NEW${NC}  $FILENAME"
        ((NEW++))
    elif [ "$CURRENT" != "$STORED" ]; then
        echo -e "${RED}CHANGED${NC} $FILENAME"
        echo "  Stored:  $STORED"
        echo "  Current: $CURRENT"
        ((CHANGED++))
    else
        echo -e "${GREEN}OK${NC}   $FILENAME"
    fi
done

echo ""
echo "=== Summary ==="
echo "New files: $NEW"
echo "Changed files: $CHANGED"

if [ $CHANGED -gt 0 ]; then
    echo ""
    echo -e "${RED}WARNING: Source data has been modified!${NC}"
    echo "This violates the read-only constraint for source data."
    exit 1
fi

exit 0
