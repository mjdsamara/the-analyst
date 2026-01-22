"""Tests for the Statistical Agent.

Marks applied:
- @pytest.mark.critical: Core agent execution tests (must always pass)
- @pytest.mark.slow: Comprehensive analysis tests (can skip for quick feedback)
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.statistical import (
    AnalysisResult,
    AnalysisType,
    StatisticalAgent,
    StatisticalAnalysisOutput,
)


class TestStatisticalAgent:
    """Test suite for StatisticalAgent."""

    @pytest.fixture
    def agent(self):
        """Create a statistical agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return StatisticalAgent()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "value1": np.random.normal(100, 15, 100),
                "value2": np.random.normal(50, 10, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @pytest.fixture
    def correlated_df(self):
        """Create a DataFrame with correlated variables."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = x * 2 + np.random.normal(0, 0.5, 100)  # Strongly correlated
        z = np.random.normal(0, 1, 100)  # Uncorrelated
        return pd.DataFrame({"x": x, "y": y, "z": z})

    @pytest.mark.critical
    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "statistical"
        assert agent.autonomy.value == "advisory"
        assert agent.toolkit is not None

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_execute_descriptive_analysis(self, agent, sample_df):
        """Test descriptive statistics analysis."""
        result = await agent.execute(
            data=sample_df,
            analysis_type=AnalysisType.DESCRIPTIVE,
            columns=["value1", "value2"],
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.analyses) == 1
        assert result.data.analyses[0].analysis_type == AnalysisType.DESCRIPTIVE

    @pytest.mark.asyncio
    async def test_execute_distribution_analysis(self, agent, sample_df):
        """Test distribution analysis."""
        result = await agent.execute(
            data=sample_df,
            analysis_type=AnalysisType.DISTRIBUTION,
            columns=["value1"],
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.analyses) == 1
        assert result.data.analyses[0].analysis_type == AnalysisType.DISTRIBUTION
        assert "normality" in str(result.data.analyses[0].methodology).lower()

    @pytest.mark.asyncio
    async def test_execute_correlation_analysis(self, agent, correlated_df):
        """Test correlation analysis."""
        result = await agent.execute(
            data=correlated_df,
            analysis_type=AnalysisType.CORRELATION,
        )

        assert result.success
        assert result.data is not None
        assert len(result.data.analyses) == 1

        # Check that correlation was detected
        analysis = result.data.analyses[0]
        assert analysis.analysis_type == AnalysisType.CORRELATION
        assert "pairwise_correlations" in analysis.results

        # x and y should be strongly correlated
        correlations = analysis.results["pairwise_correlations"]
        xy_corr = next(
            (
                c
                for c in correlations
                if (c["variable_1"] == "x" and c["variable_2"] == "y")
                or (c["variable_1"] == "y" and c["variable_2"] == "x")
            ),
            None,
        )
        assert xy_corr is not None
        assert abs(xy_corr["correlation"]) > 0.9

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_execute_comprehensive_analysis(self, agent, sample_df):
        """Test comprehensive analysis runs multiple analyses."""
        result = await agent.execute(
            data=sample_df,
            analysis_type=AnalysisType.COMPREHENSIVE,
        )

        assert result.success
        assert result.data is not None
        # Comprehensive should run multiple analyses
        assert len(result.data.analyses) >= 2
        assert result.data.summary

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_execute_with_no_data(self, agent):
        """Test error handling when no data provided."""
        result = await agent.execute(data=None)

        assert not result.success
        assert "no data" in result.error.lower()

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_execute_with_empty_data(self, agent):
        """Test error handling for empty DataFrame."""
        result = await agent.execute(data=pd.DataFrame())

        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.critical
    @pytest.mark.asyncio
    async def test_execute_with_no_numeric_columns(self, agent):
        """Test error handling when no numeric columns."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = await agent.execute(data=df)

        assert not result.success
        assert "numeric" in result.error.lower()

    @pytest.mark.asyncio
    async def test_hypothesis_testing_with_groups(self, agent):
        """Test hypothesis testing between groups."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(100, 10, 50),  # Group A
                        np.random.normal(120, 10, 50),  # Group B (different mean)
                    ]
                ),
                "group": ["A"] * 50 + ["B"] * 50,
            }
        )

        result = await agent.execute(
            data=df,
            analysis_type=AnalysisType.HYPOTHESIS_TESTING,
            target_column="value",
            group_column="group",
        )

        assert result.success
        assert result.data is not None
        analysis = result.data.analyses[0]
        assert "t_test" in analysis.results

    @pytest.mark.asyncio
    async def test_time_series_analysis(self, agent):
        """Test time series analysis."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "value": np.cumsum(np.random.randn(100)) + 100,  # Random walk
            }
        )

        result = await agent.execute(
            data=df,
            analysis_type=AnalysisType.TIME_SERIES,
            target_column="value",
        )

        assert result.success
        assert result.data is not None
        analysis = result.data.analyses[0]
        assert analysis.analysis_type == AnalysisType.TIME_SERIES
        assert "changes" in analysis.results

    def test_get_analysis_options(self, agent, sample_df):
        """Test generating analysis options."""
        options = agent.get_analysis_options(sample_df)

        assert len(options) >= 2
        assert any(opt.id == "comprehensive" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_format_output(self, agent):
        """Test output formatting."""
        output = StatisticalAnalysisOutput(
            analyses=[
                AnalysisResult(
                    analysis_type=AnalysisType.DESCRIPTIVE,
                    methodology="Test methodology",
                    assumptions_checked={"normality": True},
                    results={"test": "value"},
                    confidence_level=0.95,
                    interpretation="Test interpretation",
                    limitations=["Test limitation"],
                    reproducibility={"seed": 42},
                )
            ],
            summary="Test summary",
            recommendations=["Test recommendation"],
        )

        formatted = agent.format_output(output)

        assert "Statistical Analysis" in formatted
        assert "Methodology" in formatted
        assert "Test interpretation" in formatted
        assert "Test limitation" in formatted


class TestAnalysisResult:
    """Test suite for AnalysisResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult(
            analysis_type=AnalysisType.DESCRIPTIVE,
            methodology="Test",
            assumptions_checked={"test": True},
            results={"mean": 100},
            confidence_level=0.95,
            interpretation="Test interpretation",
            limitations=["Limitation 1"],
            reproducibility={"seed": 42},
        )

        d = result.to_dict()

        assert d["analysis_type"] == "descriptive"
        assert d["confidence_level"] == 0.95
        assert "mean" in d["results"]


class TestStatisticalAnalysisOutput:
    """Test suite for StatisticalAnalysisOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        output = StatisticalAnalysisOutput(
            analyses=[
                AnalysisResult(
                    analysis_type=AnalysisType.DESCRIPTIVE,
                    methodology="Test",
                    assumptions_checked={},
                    results={},
                    confidence_level=0.95,
                    interpretation="Test",
                    limitations=[],
                    reproducibility={},
                )
            ],
            summary="Test summary",
            recommendations=["Rec 1"],
        )

        d = output.to_dict()

        assert len(d["analyses"]) == 1
        assert d["summary"] == "Test summary"
        assert "Rec 1" in d["recommendations"]
