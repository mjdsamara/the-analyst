---
description: Load data from file (CSV, Excel, JSON, Parquet) with profiling and quality checks
allowed_tools: [Read, Glob, Grep, Bash]
---

## Your Single Job

Load data from a file and generate a comprehensive data profile with quality checks.

## Usage

```
/ingest <file_path>
```

## Arguments

- `$ARGUMENTS` will contain the file path provided by the user

## Supported Formats

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- Parquet (`.parquet`)

## Output

After loading, you'll receive:

1. **Data Profile**
   - Row and column counts
   - Column names, types, and statistics
   - Sample rows

2. **Quality Report**
   - Completeness percentage
   - Duplicate detection
   - Type issues and warnings

3. **Checksum**
   - SHA-256 hash for data integrity verification

## Examples

```
/ingest data/raw/sales_2024.csv
/ingest data/raw/user_comments.xlsx
/ingest data/raw/events.json
```

## Notes

- Large files (>100MB) use chunked loading
- Source data is never modified
- Encoding auto-detection (UTF-8, ISO-8859-1)
