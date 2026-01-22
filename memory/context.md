# The Analyst - Project Context

This file provides context for agents about the project and user preferences.

## Project Overview

The Analyst is an AI-powered analytics orchestration system for media company analytics. It uses specialized agents coordinated by a main orchestrator.

## Domain Context

### Media Companies We Work With
- IMI (International Media Investments)
- Sky News Arabia
- CNN
- The National
- Other regional media outlets

### Common Analysis Types
1. **Audience Analytics**: Viewership, engagement, demographics
2. **Content Performance**: Article/video metrics, topics, sentiment
3. **Arabic NLP**: Sentiment analysis for Arabic content and comments
4. **Cross-Platform**: Comparing performance across channels
5. **Forecasting**: Traffic, revenue, engagement predictions

## User Preferences

*This section is updated based on user interactions.*

### Output Preferences
- Preferred report format: PDF
- Chart style: Clean, minimal
- Color scheme: Professional blues

### Analysis Preferences
- Default confidence level: 95%
- Preferred forecast horizon: 30 days
- Statistical rigor: High (always include methodology)

## Common Mistakes Log

*Record mistakes here to avoid repeating them.*

### Data Handling
- [Example] Remember to check for Arabic text encoding
- Always validate data types before statistical operations

### Analysis
- [Example] Always verify seasonality before forecasting
- Include effect sizes alongside p-values for meaningful interpretation
- Ensure correlation insights specify direction (positive/negative) clearly

### Development Learnings
- Ensure all dataclass output types have `.to_dict()` methods for serialization
- Use enums for categorical values (e.g., ConfidenceLevel, InsightPriority)
- Always include `format_output()` method for markdown rendering
