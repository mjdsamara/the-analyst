---
description: Start analysis workflow with options (statistical, sentiment, trend, correlation, full)
allowed_tools: [Read, Glob, Grep, Bash, Task]
---

## Your Single Job

Start an analysis workflow by parsing user intent and coordinating with the orchestrator agent.

## Usage

```
/analyze [file_path] [analysis_type]
```

## Arguments

- `$ARGUMENTS` will contain the file path and analysis type provided by the user
- `file_path` (optional): Path to the data file to analyze
- `analysis_type` (optional): Type of analysis to perform
  - `statistical`: Descriptive statistics and EDA
  - `sentiment`: Arabic sentiment analysis
  - `trend`: Time series and trend analysis
  - `correlation`: Correlation analysis
  - `full`: Comprehensive analysis (default)

## Workflow

1. **Load Data**: The retrieval agent loads and profiles the data
2. **Present Options**: You'll see 2-3 analysis approaches with recommendations
3. **Select Approach**: Choose your preferred approach
4. **Transform**: Data is cleaned and prepared
5. **Analyze**: Selected analysis is performed
6. **Review Insights**: Key findings are presented
7. **Generate Output**: Choose output format (PDF, PPTX, HTML, etc.)

## Examples

```
/analyze data/raw/viewership.csv statistical
/analyze data/raw/comments.csv sentiment
/analyze data/raw/traffic.csv trend
```

## Notes

- All source data is read-only
- High-stakes operations require confirmation
- Every analysis includes methodology and confidence intervals
