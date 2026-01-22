---
description: Generate time series forecasts using Prophet, ARIMA, or Exponential Smoothing
allowed_tools: [Read, Glob, Grep, Bash, Task]
---

## Your Single Job

Generate forecasts and predictions for time series data with proper uncertainty quantification.

## Usage

```
/forecast <file_path> <target_column> <date_column> [horizon]
```

## Arguments

- `$ARGUMENTS` will contain all parameters provided by the user
- `file_path`: Path to the data file
- `target_column`: Column to forecast
- `date_column`: Column containing dates
- `horizon` (optional): Forecast period (e.g., "30 days", "3 months")

## Models Available

1. **Prophet** (Recommended)
   - Handles seasonality automatically
   - Robust to missing data
   - Includes uncertainty intervals

2. **ARIMA**
   - Classical time series model
   - Good for short-term forecasts
   - Requires stationarity

3. **Exponential Smoothing**
   - Simple and interpretable
   - Good baseline model

## Output

- Forecast values with confidence intervals
- Model diagnostics and validation
- Feature importance (if applicable)
- Visualization of predictions

## Workflow

1. Present model options with pros/cons
2. User selects approach
3. Train and validate model
4. Generate forecasts
5. Review and refine

## Examples

```
/forecast data/raw/traffic.csv pageviews date "30 days"
/forecast data/raw/revenue.csv amount month "6 months"
```

## Notes

- High-stakes operation (requires confirmation)
- Always includes confidence intervals
- Model assumptions are checked and reported
