"""Tests for Pydantic schemas in The Analyst platform."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisType,
    ChartConfig,
    ChartType,
    ColumnInfo,
    ConfidenceLevel,
    DatasetInfo,
    ForecastResult,
    InsightItem,
    InsightsSummary,
    OutputFormat,
    ReportConfig,
    ReportSection,
    SentimentResult,
    StatisticalResult,
)

# =============================================================================
# Enum Tests
# =============================================================================


class TestAnalysisType:
    """Test suite for AnalysisType enum."""

    def test_analysis_type_values(self):
        """Test that all analysis types have correct string values."""
        assert AnalysisType.STATISTICAL.value == "statistical"
        assert AnalysisType.SENTIMENT.value == "sentiment"
        assert AnalysisType.CORRELATION.value == "correlation"
        assert AnalysisType.TREND.value == "trend"
        assert AnalysisType.FORECAST.value == "forecast"
        assert AnalysisType.CLASSIFICATION.value == "classification"
        assert AnalysisType.FULL.value == "full"

    def test_analysis_type_from_string(self):
        """Test creating AnalysisType from string."""
        assert AnalysisType("statistical") == AnalysisType.STATISTICAL
        assert AnalysisType("sentiment") == AnalysisType.SENTIMENT

    def test_analysis_type_invalid(self):
        """Test that invalid analysis type raises error."""
        with pytest.raises(ValueError):
            AnalysisType("invalid_type")


class TestOutputFormat:
    """Test suite for OutputFormat enum."""

    def test_output_format_values(self):
        """Test that all output formats have correct string values."""
        assert OutputFormat.PDF.value == "pdf"
        assert OutputFormat.PPTX.value == "pptx"
        assert OutputFormat.HTML.value == "html"
        assert OutputFormat.MARKDOWN.value == "md"
        assert OutputFormat.JSON.value == "json"

    def test_output_format_from_string(self):
        """Test creating OutputFormat from string."""
        assert OutputFormat("pdf") == OutputFormat.PDF
        assert OutputFormat("html") == OutputFormat.HTML


class TestConfidenceLevel:
    """Test suite for ConfidenceLevel enum."""

    def test_confidence_level_values(self):
        """Test that all confidence levels have correct string values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


class TestChartType:
    """Test suite for ChartType enum."""

    def test_chart_type_values(self):
        """Test that all chart types have correct string values."""
        assert ChartType.LINE.value == "line"
        assert ChartType.BAR.value == "bar"
        assert ChartType.SCATTER.value == "scatter"
        assert ChartType.PIE.value == "pie"
        assert ChartType.HEATMAP.value == "heatmap"
        assert ChartType.BOX.value == "box"
        assert ChartType.HISTOGRAM.value == "histogram"
        assert ChartType.AREA.value == "area"


# =============================================================================
# Dataset Model Tests
# =============================================================================


class TestColumnInfo:
    """Test suite for ColumnInfo model."""

    def test_column_info_creation(self):
        """Test creating a ColumnInfo instance."""
        column = ColumnInfo(
            name="test_column",
            dtype="int64",
            non_null_count=100,
            null_count=0,
            unique_count=50,
            sample_values=["1", "2", "3"],
        )
        assert column.name == "test_column"
        assert column.dtype == "int64"
        assert column.non_null_count == 100
        assert column.null_count == 0
        assert column.unique_count == 50
        assert len(column.sample_values) == 3

    def test_column_info_default_sample_values(self):
        """Test that sample_values defaults to empty list."""
        column = ColumnInfo(
            name="test_column",
            dtype="float64",
            non_null_count=50,
            null_count=10,
            unique_count=25,
        )
        assert column.sample_values == []

    def test_column_info_missing_required(self):
        """Test that missing required fields raises error."""
        with pytest.raises(ValidationError):
            ColumnInfo(name="test")  # type: ignore


class TestDatasetInfo:
    """Test suite for DatasetInfo model."""

    @pytest.fixture
    def sample_columns(self) -> list[ColumnInfo]:
        """Create sample column info for testing."""
        return [
            ColumnInfo(
                name="id", dtype="int64", non_null_count=100, null_count=0, unique_count=100
            ),
            ColumnInfo(
                name="value", dtype="float64", non_null_count=95, null_count=5, unique_count=80
            ),
        ]

    def test_dataset_info_creation(self, sample_columns):
        """Test creating a DatasetInfo instance."""
        dataset = DatasetInfo(
            filename="test.csv",
            row_count=100,
            column_count=2,
            columns=sample_columns,
            missing_summary={"value": 5},
            checksum="abc123",
            file_size_bytes=1024,
            memory_usage_mb=0.5,
        )
        assert dataset.filename == "test.csv"
        assert dataset.row_count == 100
        assert dataset.column_count == 2
        assert len(dataset.columns) == 2
        assert dataset.missing_summary == {"value": 5}
        assert dataset.checksum == "abc123"
        assert dataset.file_size_bytes == 1024
        assert dataset.memory_usage_mb == 0.5
        assert isinstance(dataset.loaded_at, datetime)

    def test_dataset_info_defaults(self, sample_columns):
        """Test that defaults are applied correctly."""
        dataset = DatasetInfo(
            filename="test.csv",
            row_count=100,
            column_count=2,
            columns=sample_columns,
            checksum="abc123",
            file_size_bytes=1024,
            memory_usage_mb=0.5,
        )
        assert dataset.missing_summary == {}
        assert dataset.loaded_at is not None


# =============================================================================
# Analysis Model Tests
# =============================================================================


class TestAnalysisRequest:
    """Test suite for AnalysisRequest model."""

    def test_analysis_request_creation(self):
        """Test creating an AnalysisRequest instance."""
        request = AnalysisRequest(
            analysis_type=AnalysisType.STATISTICAL,
            dataset_id="dataset-123",
            columns=["value1", "value2"],
            parameters={"method": "pearson"},
            output_format=OutputFormat.PDF,
        )
        assert request.analysis_type == AnalysisType.STATISTICAL
        assert request.dataset_id == "dataset-123"
        assert request.columns == ["value1", "value2"]
        assert request.parameters == {"method": "pearson"}
        assert request.output_format == OutputFormat.PDF

    def test_analysis_request_defaults(self):
        """Test that defaults are applied correctly."""
        request = AnalysisRequest(analysis_type=AnalysisType.CORRELATION)
        assert request.dataset_id is None
        assert request.file_path is None
        assert request.columns == []
        assert request.parameters == {}
        assert request.output_format == OutputFormat.MARKDOWN

    def test_analysis_request_with_file_path(self):
        """Test creating request with file path."""
        request = AnalysisRequest(
            analysis_type=AnalysisType.SENTIMENT,
            file_path="/data/comments.csv",
        )
        assert request.file_path == "/data/comments.csv"
        assert request.dataset_id is None


class TestStatisticalResult:
    """Test suite for StatisticalResult model."""

    def test_statistical_result_creation(self):
        """Test creating a StatisticalResult instance."""
        result = StatisticalResult(
            metric_name="mean",
            value=100.5,
            unit="views",
            confidence_interval=(95.0, 106.0),
            p_value=0.001,
            effect_size=0.8,
            interpretation="Large effect size detected",
        )
        assert result.metric_name == "mean"
        assert result.value == 100.5
        assert result.unit == "views"
        assert result.confidence_interval == (95.0, 106.0)
        assert result.p_value == 0.001
        assert result.effect_size == 0.8
        assert result.interpretation == "Large effect size detected"

    def test_statistical_result_defaults(self):
        """Test that defaults are applied correctly."""
        result = StatisticalResult(metric_name="median", value=50.0)
        assert result.unit is None
        assert result.confidence_interval is None
        assert result.p_value is None
        assert result.effect_size is None
        assert result.interpretation is None


class TestSentimentResult:
    """Test suite for SentimentResult model."""

    def test_sentiment_result_creation(self):
        """Test creating a SentimentResult instance."""
        result = SentimentResult(
            text_sample="هذا المقال رائع",
            dialect="msa",
            dialect_confidence=0.92,
            overall_sentiment="positive",
            sentiment_confidence=0.87,
            sentiment_distribution={"positive": 0.87, "neutral": 0.1, "negative": 0.03},
            entities=[{"text": "المقال", "type": "NOUN"}],
            topics=[{"topic": "article", "confidence": 0.9}],
        )
        assert result.text_sample == "هذا المقال رائع"
        assert result.dialect == "msa"
        assert result.dialect_confidence == 0.92
        assert result.overall_sentiment == "positive"
        assert result.sentiment_confidence == 0.87
        assert "positive" in result.sentiment_distribution
        assert len(result.entities) == 1
        assert len(result.topics) == 1

    def test_sentiment_result_defaults(self):
        """Test that defaults are applied correctly."""
        result = SentimentResult(
            text_sample="test",
            dialect="gulf",
            dialect_confidence=0.8,
            overall_sentiment="neutral",
            sentiment_confidence=0.7,
            sentiment_distribution={"neutral": 1.0},
        )
        assert result.entities == []
        assert result.topics == []


class TestForecastResult:
    """Test suite for ForecastResult model."""

    def test_forecast_result_creation(self):
        """Test creating a ForecastResult instance."""
        result = ForecastResult(
            target_column="views",
            model_name="Prophet",
            forecast_horizon=30,
            horizon_unit="days",
            predictions=[{"date": "2024-01-01", "value": 100, "lower": 90, "upper": 110}],
            confidence_level=0.95,
            metrics={"mape": 0.05, "rmse": 10.5},
            feature_importance={"trend": 0.7, "seasonality": 0.3},
        )
        assert result.target_column == "views"
        assert result.model_name == "Prophet"
        assert result.forecast_horizon == 30
        assert result.horizon_unit == "days"
        assert len(result.predictions) == 1
        assert result.confidence_level == 0.95
        assert result.metrics["mape"] == 0.05
        assert result.feature_importance["trend"] == 0.7

    def test_forecast_result_defaults(self):
        """Test that defaults are applied correctly."""
        result = ForecastResult(
            target_column="sales",
            model_name="ARIMA",
            forecast_horizon=7,
            horizon_unit="days",
            predictions=[],
        )
        assert result.confidence_level == 0.95
        assert result.metrics == {}
        assert result.feature_importance == {}


class TestAnalysisResult:
    """Test suite for AnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult instance."""
        now = datetime.utcnow()
        result = AnalysisResult(
            analysis_type=AnalysisType.STATISTICAL,
            started_at=now,
            completed_at=now,
            methodology="Descriptive statistics",
            assumptions=["Normal distribution"],
            results={"mean": 100.5},
            limitations=["Small sample size"],
            confidence_level=ConfidenceLevel.HIGH,
        )
        assert result.analysis_type == AnalysisType.STATISTICAL
        assert result.methodology == "Descriptive statistics"
        assert result.assumptions == ["Normal distribution"]
        assert result.results == {"mean": 100.5}
        assert result.limitations == ["Small sample size"]
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_analysis_result_defaults(self):
        """Test that defaults are applied correctly."""
        now = datetime.utcnow()
        result = AnalysisResult(
            analysis_type=AnalysisType.CORRELATION,
            started_at=now,
            completed_at=now,
            methodology="Pearson correlation",
        )
        assert result.assumptions == []
        assert result.results == {}
        assert result.limitations == []
        assert result.confidence_level == ConfidenceLevel.MEDIUM


# =============================================================================
# Insight Model Tests
# =============================================================================


class TestInsightItem:
    """Test suite for InsightItem model."""

    def test_insight_item_creation(self):
        """Test creating an InsightItem instance."""
        insight = InsightItem(
            title="Strong Correlation Found",
            finding="Views and time on page are strongly correlated",
            evidence="r = 0.85, p < 0.001",
            impact="30% increase in engagement predicted",
            recommendation="Focus content on engagement metrics",
            confidence=ConfidenceLevel.HIGH,
            priority=8,
            tags=["correlation", "engagement"],
        )
        assert insight.title == "Strong Correlation Found"
        assert insight.finding == "Views and time on page are strongly correlated"
        assert insight.evidence == "r = 0.85, p < 0.001"
        assert insight.impact == "30% increase in engagement predicted"
        assert insight.recommendation == "Focus content on engagement metrics"
        assert insight.confidence == ConfidenceLevel.HIGH
        assert insight.priority == 8
        assert "correlation" in insight.tags

    def test_insight_item_defaults(self):
        """Test that defaults are applied correctly."""
        insight = InsightItem(
            title="Test",
            finding="Test finding",
            evidence="Test evidence",
            impact="Test impact",
            recommendation="Test recommendation",
            confidence=ConfidenceLevel.MEDIUM,
            priority=5,
        )
        assert insight.tags == []

    def test_insight_item_priority_validation(self):
        """Test that priority must be between 1 and 10."""
        with pytest.raises(ValidationError):
            InsightItem(
                title="Test",
                finding="Test",
                evidence="Test",
                impact="Test",
                recommendation="Test",
                confidence=ConfidenceLevel.LOW,
                priority=11,  # Invalid: > 10
            )

        with pytest.raises(ValidationError):
            InsightItem(
                title="Test",
                finding="Test",
                evidence="Test",
                impact="Test",
                recommendation="Test",
                confidence=ConfidenceLevel.LOW,
                priority=0,  # Invalid: < 1
            )


class TestInsightsSummary:
    """Test suite for InsightsSummary model."""

    @pytest.fixture
    def sample_insight(self) -> InsightItem:
        """Create a sample insight for testing."""
        return InsightItem(
            title="Test",
            finding="Test finding",
            evidence="Test evidence",
            impact="Test impact",
            recommendation="Test recommendation",
            confidence=ConfidenceLevel.MEDIUM,
            priority=5,
        )

    def test_insights_summary_creation(self, sample_insight):
        """Test creating an InsightsSummary instance."""
        summary = InsightsSummary(
            insights=[sample_insight],
            anomalies=[{"type": "outlier", "value": 999}],
            recommended_actions=["Review data quality"],
            questions_for_followup=["What caused the spike?"],
        )
        assert len(summary.insights) == 1
        assert len(summary.anomalies) == 1
        assert "Review data quality" in summary.recommended_actions
        assert "What caused the spike?" in summary.questions_for_followup

    def test_insights_summary_defaults(self, sample_insight):
        """Test that defaults are applied correctly."""
        summary = InsightsSummary(insights=[sample_insight])
        assert summary.anomalies == []
        assert summary.recommended_actions == []
        assert summary.questions_for_followup == []


# =============================================================================
# Visualization Model Tests
# =============================================================================


class TestChartConfig:
    """Test suite for ChartConfig model."""

    def test_chart_config_creation(self):
        """Test creating a ChartConfig instance."""
        config = ChartConfig(
            chart_type=ChartType.LINE,
            title="Views Over Time",
            x_column="date",
            y_column="views",
            color_column="section",
            x_label="Date",
            y_label="Number of Views",
            legend_title="Section",
            data_source="Analytics Database",
            width=1200,
            height=600,
            template="plotly_dark",
            additional_config={"showgrid": True},
        )
        assert config.chart_type == ChartType.LINE
        assert config.title == "Views Over Time"
        assert config.x_column == "date"
        assert config.y_column == "views"
        assert config.color_column == "section"
        assert config.x_label == "Date"
        assert config.y_label == "Number of Views"
        assert config.legend_title == "Section"
        assert config.data_source == "Analytics Database"
        assert config.width == 1200
        assert config.height == 600
        assert config.template == "plotly_dark"
        assert config.additional_config["showgrid"] is True

    def test_chart_config_defaults(self):
        """Test that defaults are applied correctly."""
        config = ChartConfig(chart_type=ChartType.BAR, title="Test Chart")
        assert config.x_column is None
        assert config.y_column is None
        assert config.color_column is None
        assert config.x_label is None
        assert config.y_label is None
        assert config.legend_title is None
        assert config.data_source is None
        assert config.width == 800
        assert config.height == 500
        assert config.template == "plotly_white"
        assert config.additional_config == {}


# =============================================================================
# Report Model Tests
# =============================================================================


class TestReportSection:
    """Test suite for ReportSection model."""

    def test_report_section_creation(self):
        """Test creating a ReportSection instance."""
        section = ReportSection(
            title="Executive Summary",
            content="Key findings from the analysis...",
            charts=["chart-1", "chart-2"],
            order=1,
        )
        assert section.title == "Executive Summary"
        assert section.content == "Key findings from the analysis..."
        assert section.charts == ["chart-1", "chart-2"]
        assert section.order == 1

    def test_report_section_defaults(self):
        """Test that defaults are applied correctly."""
        section = ReportSection(title="Test", content="Test content", order=1)
        assert section.charts == []


class TestReportConfig:
    """Test suite for ReportConfig model."""

    @pytest.fixture
    def sample_section(self) -> ReportSection:
        """Create a sample section for testing."""
        return ReportSection(title="Test Section", content="Content", order=1)

    def test_report_config_creation(self, sample_section):
        """Test creating a ReportConfig instance."""
        config = ReportConfig(
            title="Analytics Report Q1 2024",
            output_format=OutputFormat.PDF,
            style="detailed",
            sections=[sample_section],
            include_executive_summary=True,
            include_methodology=True,
            include_appendix=True,
            author="Data Team",
        )
        assert config.title == "Analytics Report Q1 2024"
        assert config.output_format == OutputFormat.PDF
        assert config.style == "detailed"
        assert len(config.sections) == 1
        assert config.include_executive_summary is True
        assert config.include_methodology is True
        assert config.include_appendix is True
        assert config.author == "Data Team"
        assert isinstance(config.created_at, datetime)

    def test_report_config_defaults(self):
        """Test that defaults are applied correctly."""
        config = ReportConfig(title="Test Report", output_format=OutputFormat.MARKDOWN)
        assert config.style == "executive"
        assert config.sections == []
        assert config.include_executive_summary is True
        assert config.include_methodology is True
        assert config.include_appendix is False
        assert config.author is None
        assert config.created_at is not None


# =============================================================================
# Serialization Tests
# =============================================================================


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_analysis_request_to_dict(self):
        """Test serializing AnalysisRequest to dict."""
        request = AnalysisRequest(
            analysis_type=AnalysisType.STATISTICAL,
            columns=["value1", "value2"],
        )
        data = request.model_dump()
        assert data["analysis_type"] == "statistical"
        assert data["columns"] == ["value1", "value2"]

    def test_analysis_request_to_json(self):
        """Test serializing AnalysisRequest to JSON."""
        request = AnalysisRequest(
            analysis_type=AnalysisType.CORRELATION,
            parameters={"method": "spearman"},
        )
        json_str = request.model_dump_json()
        assert '"analysis_type":"correlation"' in json_str
        assert '"method":"spearman"' in json_str

    def test_chart_config_round_trip(self):
        """Test serializing and deserializing ChartConfig."""
        original = ChartConfig(
            chart_type=ChartType.LINE,
            title="Test Chart",
            x_column="date",
            y_column="value",
        )
        data = original.model_dump()
        restored = ChartConfig(**data)
        assert restored.chart_type == original.chart_type
        assert restored.title == original.title
        assert restored.x_column == original.x_column
        assert restored.y_column == original.y_column

    def test_dataset_info_serialization(self):
        """Test serializing DatasetInfo with nested models."""
        columns = [
            ColumnInfo(name="id", dtype="int64", non_null_count=100, null_count=0, unique_count=100)
        ]
        dataset = DatasetInfo(
            filename="test.csv",
            row_count=100,
            column_count=1,
            columns=columns,
            checksum="abc123",
            file_size_bytes=1024,
            memory_usage_mb=0.5,
        )
        data = dataset.model_dump()
        assert data["columns"][0]["name"] == "id"
        assert data["columns"][0]["dtype"] == "int64"
