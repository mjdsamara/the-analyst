"""
Pydantic schemas for The Analyst platform.

Defines data models for requests, responses, and internal state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Types of analysis available."""

    STATISTICAL = "statistical"
    SENTIMENT = "sentiment"
    CORRELATION = "correlation"
    TREND = "trend"
    FORECAST = "forecast"
    CLASSIFICATION = "classification"
    FULL = "full"


class OutputFormat(str, Enum):
    """Output format options."""

    PDF = "pdf"
    PPTX = "pptx"
    HTML = "html"
    MARKDOWN = "md"
    JSON = "json"


class ConfidenceLevel(str, Enum):
    """Confidence levels for insights."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Dataset Models
# =============================================================================


class ColumnInfo(BaseModel):
    """Information about a dataset column."""

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    unique_count: int
    sample_values: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    """Information about a loaded dataset."""

    filename: str
    row_count: int
    column_count: int
    columns: list[ColumnInfo]
    missing_summary: dict[str, int] = Field(default_factory=dict)
    checksum: str
    file_size_bytes: int
    memory_usage_mb: float
    loaded_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Analysis Models
# =============================================================================


class AnalysisRequest(BaseModel):
    """Request for an analysis."""

    analysis_type: AnalysisType
    dataset_id: str | None = None
    file_path: str | None = None
    columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_format: OutputFormat = OutputFormat.MARKDOWN


class StatisticalResult(BaseModel):
    """Result from statistical analysis."""

    metric_name: str
    value: float
    unit: str | None = None
    confidence_interval: tuple[float, float] | None = None
    p_value: float | None = None
    effect_size: float | None = None
    interpretation: str | None = None


class SentimentResult(BaseModel):
    """Result from sentiment analysis."""

    text_sample: str
    dialect: str
    dialect_confidence: float
    overall_sentiment: str
    sentiment_confidence: float
    sentiment_distribution: dict[str, float]
    entities: list[dict[str, Any]] = Field(default_factory=list)
    topics: list[dict[str, Any]] = Field(default_factory=list)


class ForecastResult(BaseModel):
    """Result from forecasting."""

    target_column: str
    model_name: str
    forecast_horizon: int
    horizon_unit: str
    predictions: list[dict[str, Any]]
    confidence_level: float = 0.95
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete result from an analysis."""

    analysis_type: AnalysisType
    started_at: datetime
    completed_at: datetime
    methodology: str
    assumptions: list[str] = Field(default_factory=list)
    results: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM


# =============================================================================
# Insight Models
# =============================================================================


class InsightItem(BaseModel):
    """A single insight from analysis."""

    title: str
    finding: str
    evidence: str
    impact: str
    recommendation: str
    confidence: ConfidenceLevel
    priority: int = Field(ge=1, le=10)
    tags: list[str] = Field(default_factory=list)


class InsightsSummary(BaseModel):
    """Summary of all insights from an analysis."""

    insights: list[InsightItem]
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    questions_for_followup: list[str] = Field(default_factory=list)


# =============================================================================
# Visualization Models
# =============================================================================


class ChartType(str, Enum):
    """Types of charts available."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    BOX = "box"
    HISTOGRAM = "histogram"
    AREA = "area"


class ChartConfig(BaseModel):
    """Configuration for a chart."""

    chart_type: ChartType
    title: str
    x_column: str | None = None
    y_column: str | None = None
    color_column: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    legend_title: str | None = None
    data_source: str | None = None
    width: int = 800
    height: int = 500
    template: str = "plotly_white"
    additional_config: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Report Models
# =============================================================================


class ReportSection(BaseModel):
    """A section in a report."""

    title: str
    content: str
    charts: list[str] = Field(default_factory=list)  # Chart IDs
    order: int


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    title: str
    output_format: OutputFormat
    style: str = "executive"  # "executive", "detailed", "dashboard"
    sections: list[ReportSection] = Field(default_factory=list)
    include_executive_summary: bool = True
    include_methodology: bool = True
    include_appendix: bool = False
    author: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
