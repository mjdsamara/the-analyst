---
description: Generate formatted reports (PDF, PPTX, HTML, Markdown) from analysis results
allowed_tools: [Read, Write, Glob, Grep, Bash]
---

## Your Single Job

Generate a professionally formatted report from analysis results with proper structure and methodology.

## Usage

```
/report [format] [style]
```

## Arguments

- `$ARGUMENTS` will contain the format and style provided by the user
- `format` (optional): Output format
  - `pdf`: PDF document (default)
  - `pptx`: PowerPoint presentation
  - `html`: Interactive HTML dashboard
  - `md`: Markdown document

- `style` (optional): Visual style
  - `executive`: Clean, minimal, stakeholder-ready
  - `detailed`: Comprehensive with technical details
  - `dashboard`: Interactive with filters

## Report Structure

1. **Executive Summary** (1 page max)
2. **Methodology** (data sources, approach)
3. **Key Findings** (charts + insights)
4. **Detailed Analysis** (full results)
5. **Recommendations** (actionable items)
6. **Appendix** (technical details, data tables)

## Workflow

1. Review draft structure
2. Approve sections and styling
3. Generate final output
4. Save to `data/outputs/`

## Examples

```
/report pdf executive
/report pptx
/report html dashboard
```

## Notes

- Draft approval required before final generation
- All charts include titles, labels, and sources
- Confidence intervals shown for predictions
