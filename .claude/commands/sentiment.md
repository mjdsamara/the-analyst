---
description: Analyze Arabic text sentiment using MARBERT and CAMeL Tools
allowed_tools: [Read, Glob, Grep, Bash]
---

## Your Single Job

Analyze sentiment in Arabic text using specialized Arabic NLP models with proper dialect handling.

## Usage

```
/sentiment <file_path> <text_column>
```

## Arguments

- `$ARGUMENTS` will contain the file path and text column provided by the user
- `file_path`: Path to the data file containing Arabic text
- `text_column`: Name of the column containing text to analyze

## Analysis Output

1. **Dialect Detection**
   - Primary dialect (MSA, Gulf, Egyptian, Levantine, Maghrebi)
   - Confidence score
   - Code-switching detection

2. **Sentiment Analysis**
   - Overall sentiment (Positive/Negative/Neutral)
   - Sentiment distribution
   - Confidence scores

3. **Named Entities**
   - People, organizations, locations
   - Entity frequency counts

4. **Topics**
   - Extracted topics with percentages

## Models Used

- **MARBERT**: Arabic-specific BERT for sentiment
- **CAMeL Tools**: NER, POS tagging, dialect ID

## Examples

```
/sentiment data/raw/comments.csv comment_text
/sentiment data/raw/reviews.xlsx review_ar
```

## Notes

- Text is normalized before processing
- Handles mixed Arabic-English content
- Results include confidence intervals
