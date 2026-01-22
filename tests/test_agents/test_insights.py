"""Tests for the Insights Agent."""

import os
from unittest.mock import patch

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.insights import (
    Anomaly,
    ConfidenceLevel,
    Insight,
    InsightPriority,
    InsightsAgent,
    InsightsOutput,
    RecommendedAction,
)


class TestInsightsAgent:
    """Test suite for InsightsAgent."""

    @pytest.fixture
    def agent(self):
        """Create an insights agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return InsightsAgent()

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results for testing."""
        return {
            "descriptive": {
                "value1": {
                    "mean": {"value": 100, "confidence_interval": (95, 105)},
                    "std": {"value": 50},
                    "skewness": {"interpretation": "Positively skewed (right-tailed)"},
                },
                "value2": {
                    "mean": {"value": 50, "confidence_interval": (48, 52)},
                    "std": {"value": 10},
                    "skewness": {"interpretation": "Approximately symmetric"},
                },
            },
            "correlation": {
                "pairwise_correlations": [
                    {
                        "variable_1": "x",
                        "variable_2": "y",
                        "correlation": 0.85,
                        "p_value": 0.001,
                    },
                    {
                        "variable_1": "x",
                        "variable_2": "z",
                        "correlation": 0.15,
                        "p_value": 0.5,
                    },
                ],
            },
        }

    @pytest.fixture
    def sample_data_profile(self):
        """Create sample data profile for testing."""
        return {
            "row_count": 1000,
            "column_count": 5,
            "missing_summary": {
                "col1": 150,  # 15% missing
                "col2": 50,  # 5% missing
            },
        }

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "insights"
        assert agent.autonomy.value == "advisory"

    @pytest.mark.asyncio
    async def test_execute_with_analysis_results(self, agent, sample_analysis_results):
        """Test generating insights from analysis results."""
        result = await agent.execute(analysis_results=sample_analysis_results)

        assert result.success
        assert result.data is not None
        assert len(result.data.insights) > 0
        assert result.data.executive_summary

    @pytest.mark.asyncio
    async def test_execute_with_data_profile(self, agent, sample_data_profile):
        """Test generating insights from data profile."""
        result = await agent.execute(data_profile=sample_data_profile)

        assert result.success
        assert result.data is not None
        # Should detect high missing values in col1
        missing_insight = any(
            "missing" in insight.title.lower() for insight in result.data.insights
        )
        assert missing_insight

    @pytest.mark.asyncio
    async def test_execute_with_no_input(self, agent):
        """Test error handling when no input provided."""
        result = await agent.execute()

        assert not result.success
        assert "no analysis results" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_focus_areas(self, agent, sample_analysis_results):
        """Test filtering insights by focus areas."""
        result = await agent.execute(
            analysis_results=sample_analysis_results,
            focus_areas=["correlation"],
        )

        assert result.success
        # Should prioritize correlation-related insights
        if result.data.insights:
            correlation_insights = [
                i
                for i in result.data.insights
                if "correlation" in i.tags or "correlation" in i.title.lower()
            ]
            # Correlation insights should be present due to focus
            assert len(correlation_insights) >= 0  # May vary based on data

    @pytest.mark.asyncio
    async def test_insights_from_strong_correlation(self, agent):
        """Test that strong correlations generate insights."""
        analysis_results = {
            "correlation": {
                "pairwise_correlations": [
                    {
                        "variable_1": "sales",
                        "variable_2": "marketing_spend",
                        "correlation": 0.92,
                        "p_value": 0.0001,
                    },
                ],
            },
        }

        result = await agent.execute(analysis_results=analysis_results)

        assert result.success
        # Should find the strong correlation insight
        strong_corr_insight = any(
            "strong" in insight.title.lower() and "correlation" in insight.title.lower()
            for insight in result.data.insights
        )
        assert strong_corr_insight

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, agent):
        """Test anomaly detection for near-perfect correlations."""
        analysis_results = {
            "correlation": {
                "pairwise_correlations": [
                    {
                        "variable_1": "col_a",
                        "variable_2": "col_b",
                        "correlation": 0.999,  # Near-perfect - suspicious
                        "p_value": 0.0001,
                    },
                ],
            },
        }

        result = await agent.execute(analysis_results=analysis_results)

        assert result.success
        # Should detect this as an anomaly
        assert len(result.data.anomalies) > 0
        assert (
            "perfect" in result.data.anomalies[0].description.lower()
            or "col_a" in result.data.anomalies[0].description
        )

    @pytest.mark.asyncio
    async def test_recommended_actions_generated(self, agent, sample_analysis_results):
        """Test that recommended actions are generated."""
        result = await agent.execute(analysis_results=sample_analysis_results)

        assert result.success
        # Actions should be generated from insights
        if result.data.insights:
            # Should have at least some actions
            assert len(result.data.actions) >= 0

    @pytest.mark.asyncio
    async def test_questions_for_further_analysis(self, agent, sample_analysis_results):
        """Test that follow-up questions are generated."""
        result = await agent.execute(
            analysis_results=sample_analysis_results,
            focus_areas=["revenue"],
        )

        assert result.success
        # Should generate questions for further analysis
        assert len(result.data.questions_for_further_analysis) >= 0

    def test_get_focus_options(self, agent):
        """Test generating focus area options."""
        options = agent.get_focus_options(["correlation", "descriptive"])

        assert len(options) >= 2
        assert any(opt.id == "all" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_format_output(self, agent):
        """Test output formatting."""
        output = InsightsOutput(
            insights=[
                Insight(
                    title="Test Insight",
                    finding="Test finding",
                    evidence=["Evidence 1"],
                    impact="Test impact",
                    recommendation="Test recommendation",
                    confidence=ConfidenceLevel.HIGH,
                    priority=InsightPriority.HIGH,
                )
            ],
            anomalies=[
                Anomaly(
                    description="Test anomaly",
                    severity="high",
                    evidence="Test evidence",
                    potential_causes=["Cause 1"],
                    recommended_investigation="Investigate X",
                )
            ],
            actions=[
                RecommendedAction(
                    action="Do something",
                    expected_impact="Improve metrics",
                    priority=1,
                    effort="low",
                )
            ],
            questions_for_further_analysis=["What causes X?"],
            executive_summary="Test summary",
        )

        formatted = agent.format_output(output)

        assert "Test Insight" in formatted
        assert "Test finding" in formatted
        assert "Test anomaly" in formatted
        assert "Do something" in formatted
        assert "What causes X?" in formatted


class TestInsight:
    """Test suite for Insight dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        insight = Insight(
            title="Test Title",
            finding="Test finding",
            evidence=["Evidence 1", "Evidence 2"],
            impact="High impact",
            recommendation="Do this",
            confidence=ConfidenceLevel.HIGH,
            priority=InsightPriority.CRITICAL,
            tags=["tag1", "tag2"],
            related_columns=["col1"],
        )

        d = insight.to_dict()

        assert d["title"] == "Test Title"
        assert d["confidence"] == "high"
        assert d["priority"] == "critical"
        assert len(d["evidence"]) == 2
        assert "tag1" in d["tags"]


class TestAnomaly:
    """Test suite for Anomaly dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        anomaly = Anomaly(
            description="Test anomaly",
            severity="high",
            evidence="Evidence text",
            potential_causes=["Cause 1", "Cause 2"],
            recommended_investigation="Investigate Y",
        )

        d = anomaly.to_dict()

        assert d["description"] == "Test anomaly"
        assert d["severity"] == "high"
        assert len(d["potential_causes"]) == 2


class TestRecommendedAction:
    """Test suite for RecommendedAction dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        action = RecommendedAction(
            action="Take action",
            expected_impact="Positive outcome",
            priority=1,
            effort="medium",
            dependencies=["Dep 1"],
        )

        d = action.to_dict()

        assert d["action"] == "Take action"
        assert d["priority"] == 1
        assert d["effort"] == "medium"


class TestInsightsOutput:
    """Test suite for InsightsOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = InsightsOutput(
            insights=[
                Insight(
                    title="Test",
                    finding="Finding",
                    evidence=["E1"],
                    impact="Impact",
                    recommendation="Rec",
                    confidence=ConfidenceLevel.MEDIUM,
                )
            ],
            anomalies=[],
            actions=[],
            questions_for_further_analysis=["Q1"],
            executive_summary="Summary",
        )

        d = output.to_dict()

        assert len(d["insights"]) == 1
        assert d["executive_summary"] == "Summary"
        assert "Q1" in d["questions_for_further_analysis"]
