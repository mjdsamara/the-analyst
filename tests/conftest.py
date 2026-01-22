"""Pytest configuration and fixtures for The Analyst tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Pytest Marks Registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom pytest markers for test categorization.

    Marks:
        critical: Tests that must always pass - high coverage priority.
                  Use for core agent functionality and critical paths.
                  Example: pytest -m critical

        integration: End-to-end workflow tests spanning multiple agents.
                     These test the full orchestration pipeline.
                     Example: pytest -m integration

        slow: Tests that take longer to run (>5s). Can be skipped for
              quick feedback during development.
              Example: pytest -m "not slow"

    Usage in tests:
        @pytest.mark.critical
        def test_agent_execution():
            ...

        @pytest.mark.integration
        async def test_full_workflow():
            ...

        @pytest.mark.slow
        def test_large_dataset_processing():
            ...
    """
    config.addinivalue_line(
        "markers", "critical: Tests that must always pass - high coverage priority"
    )
    config.addinivalue_line(
        "markers", "integration: End-to-end workflow tests spanning multiple agents"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer to run - can skip with -m 'not slow'"
    )


# Set environment variables before any imports
os.environ["ANTHROPIC_API_KEY"] = "test-key-for-testing"
os.environ["HUGGINGFACE_TOKEN"] = "test-token"
os.environ["DATABASE_URL"] = "postgresql://localhost:5432/analyst_test"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    yield


@pytest.fixture(autouse=True)
def reset_middleware_singletons():
    """Reset middleware singletons before each test to ensure isolation."""
    from src.agents.base import reset_middleware

    reset_middleware()
    yield
    reset_middleware()


# ---------------------------------------------------------------------------
# Mock LLM Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing agents."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Mock LLM response")]
    # Add usage attributes for cost tracking middleware
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_llm_response():
    """Factory fixture to create custom LLM responses."""

    def _create_response(text: str = "Mock response") -> MagicMock:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=text)]
        # Add usage attributes for cost tracking middleware
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        return mock_response

    return _create_response


@pytest.fixture
def patch_anthropic():
    """Patch the AsyncAnthropic client for all agent tests."""
    with patch("src.agents.base.AsyncAnthropic") as mock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mock LLM response")]
        # Add usage attributes for cost tracking middleware
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock.return_value = mock_client
        yield mock


# ---------------------------------------------------------------------------
# Sample DataFrame Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe():
    """Create a basic sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(1, 101),
            "value1": np.random.normal(100, 15, 100),
            "value2": np.random.normal(50, 10, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "date": pd.date_range("2024-01-01", periods=100),
        }
    )


@pytest.fixture
def sample_dataframe_with_nulls():
    """Create a DataFrame with null values for testing data cleaning."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "id": range(1, 101),
            "value1": np.random.normal(100, 15, 100),
            "value2": np.random.normal(50, 10, 100),
            "category": np.random.choice(["A", "B", "C", None], 100),
        }
    )
    # Introduce nulls
    df.loc[5:10, "value1"] = np.nan
    df.loc[20:25, "value2"] = np.nan
    return df


@pytest.fixture
def sample_dataframe_with_duplicates():
    """Create a DataFrame with duplicate rows for testing."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 1, 2, 3],
            "value": [10, 20, 30, 40, 50, 10, 20, 30],
            "category": ["A", "B", "C", "A", "B", "A", "B", "C"],
        }
    )
    return df


@pytest.fixture
def time_series_dataframe():
    """Create a time series DataFrame for testing forecasting."""
    np.random.seed(42)
    n_points = 365
    dates = pd.date_range("2023-01-01", periods=n_points, freq="D")

    # Create seasonal pattern with trend
    trend = np.linspace(100, 150, n_points)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_points) / 365)
    noise = np.random.normal(0, 5, n_points)

    return pd.DataFrame(
        {
            "date": dates,
            "value": trend + seasonal + noise,
            "volume": np.random.randint(1000, 5000, n_points),
        }
    )


@pytest.fixture
def correlated_dataframe():
    """Create a DataFrame with correlated variables for testing."""
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = x * 2 + np.random.normal(0, 0.5, n)  # Strongly correlated with x
    z = np.random.normal(0, 1, n)  # Uncorrelated
    return pd.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def viewership_dataframe():
    """Create a viewership DataFrame mimicking media analytics data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "article_id": range(1, n + 1),
            "views": np.random.randint(100, 10000, n),
            "unique_visitors": np.random.randint(50, 5000, n),
            "time_on_page": np.random.uniform(30, 300, n),
            "bounce_rate": np.random.uniform(0.2, 0.8, n),
            "section": np.random.choice(["News", "Sports", "Entertainment", "Opinion"], n),
            "author": np.random.choice(["Alice", "Bob", "Charlie", "Diana"], n),
        }
    )


# ---------------------------------------------------------------------------
# Arabic Text Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arabic_text_samples():
    """Sample Arabic text for sentiment analysis testing."""
    return [
        "هذا المقال رائع جدا",  # Positive: "This article is very wonderful"
        "الخدمة سيئة للغاية",  # Negative: "The service is very bad"
        "المنتج عادي",  # Neutral: "The product is ordinary"
        "أحببت هذا الفيلم كثيرا",  # Positive: "I loved this movie a lot"
        "لم أكن راضيا عن التجربة",  # Negative: "I was not satisfied with the experience"
    ]


@pytest.fixture
def arabic_dataframe():
    """DataFrame with Arabic text for NLP testing."""
    return pd.DataFrame(
        {
            "id": range(1, 6),
            "text": [
                "هذا المقال رائع جدا",
                "الخدمة سيئة للغاية",
                "المنتج عادي",
                "أحببت هذا الفيلم كثيرا",
                "لم أكن راضيا عن التجربة",
            ],
            "category": ["article", "service", "product", "entertainment", "service"],
        }
    )


# ---------------------------------------------------------------------------
# Agent Context Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_context():
    """Create an agent context for testing."""
    from src.agents.base import AgentContext

    return AgentContext(session_id="test-session-123", user_id="test-user")


@pytest.fixture
def agent_context_with_data(sample_dataframe):
    """Create an agent context with preloaded data."""
    from src.agents.base import AgentContext

    ctx = AgentContext(session_id="test-session-123", user_id="test-user")
    ctx.set_data("loaded_data", sample_dataframe)
    ctx.set_data(
        "data_profile",
        {
            "row_count": len(sample_dataframe),
            "column_count": len(sample_dataframe.columns),
            "columns": list(sample_dataframe.columns),
        },
    )
    return ctx


# ---------------------------------------------------------------------------
# Agent Factory Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transform_agent(patch_anthropic, agent_context):
    """Create a TransformAgent instance for testing."""
    from src.agents.transform import TransformAgent

    return TransformAgent(context=agent_context)


@pytest.fixture
def statistical_agent(patch_anthropic, agent_context):
    """Create a StatisticalAgent instance for testing."""
    from src.agents.statistical import StatisticalAgent

    return StatisticalAgent(context=agent_context)


@pytest.fixture
def retrieval_agent(patch_anthropic, agent_context):
    """Create a RetrievalAgent instance for testing."""
    from src.agents.retrieval import RetrievalAgent

    return RetrievalAgent(context=agent_context)


@pytest.fixture
def modeling_agent(patch_anthropic, agent_context):
    """Create a ModelingAgent instance for testing."""
    from src.agents.modeling import ModelingAgent

    return ModelingAgent(context=agent_context)


@pytest.fixture
def insights_agent(patch_anthropic, agent_context):
    """Create an InsightsAgent instance for testing."""
    from src.agents.insights import InsightsAgent

    return InsightsAgent(context=agent_context)


@pytest.fixture
def visualization_agent(patch_anthropic, agent_context):
    """Create a VisualizationAgent instance for testing."""
    from src.agents.visualization import VisualizationAgent

    return VisualizationAgent(context=agent_context)


@pytest.fixture
def report_agent(patch_anthropic, agent_context):
    """Create a ReportAgent instance for testing."""
    from src.agents.report import ReportAgent

    return ReportAgent(context=agent_context)


# ---------------------------------------------------------------------------
# Router and Orchestrator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def intent_router():
    """Create an IntentRouter instance for testing."""
    from src.orchestrator.router import IntentRouter

    return IntentRouter()


@pytest.fixture
def mock_orchestrator(patch_anthropic):
    """Create a mock Orchestrator for testing."""
    with (
        patch("src.orchestrator.main.StateManager"),
        patch("src.orchestrator.main.Notifier"),
        patch("src.orchestrator.main.ObsidianVault"),
    ):
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()
        yield orchestrator


# ---------------------------------------------------------------------------
# File System Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_file(temp_directory, sample_dataframe):
    """Create a sample CSV file for testing file loading."""
    file_path = temp_directory / "sample_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_excel_file(temp_directory, sample_dataframe):
    """Create a sample Excel file for testing file loading."""
    file_path = temp_directory / "sample_data.xlsx"
    sample_dataframe.to_excel(file_path, index=False)
    return file_path


@pytest.fixture
def sample_json_file(temp_directory, sample_dataframe):
    """Create a sample JSON file for testing file loading."""
    file_path = temp_directory / "sample_data.json"
    sample_dataframe.to_json(file_path, orient="records", date_format="iso")
    return file_path


@pytest.fixture
def sample_parquet_file(temp_directory, sample_dataframe):
    """Create a sample Parquet file for testing file loading."""
    file_path = temp_directory / "sample_data.parquet"
    sample_dataframe.to_parquet(file_path, index=False)
    return file_path


# ---------------------------------------------------------------------------
# Mock Database Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_client():
    """Create a mock database client for testing."""
    mock_client = MagicMock()
    mock_client.execute = AsyncMock(return_value=None)
    mock_client.fetch = AsyncMock(return_value=[])
    mock_client.fetchrow = AsyncMock(return_value=None)
    mock_client.close = AsyncMock()
    return mock_client


@pytest.fixture
def mock_db_pool(mock_db_client):
    """Create a mock database connection pool."""
    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=mock_db_client)
    mock_pool.release = MagicMock()
    mock_pool.close = AsyncMock()
    return mock_pool


# ---------------------------------------------------------------------------
# Analysis Result Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_statistical_results():
    """Sample statistical analysis results for testing."""
    return {
        "analyses": [
            {
                "analysis_type": "descriptive",
                "methodology": "Basic descriptive statistics",
                "results": {
                    "mean": 100.5,
                    "std": 15.2,
                    "min": 65.3,
                    "max": 145.7,
                    "median": 99.8,
                },
                "confidence_level": 0.95,
                "interpretation": "Data shows normal distribution",
                "limitations": ["Sample size is moderate"],
            }
        ],
        "summary": "Descriptive analysis completed successfully",
        "recommendations": ["Consider segmentation analysis"],
    }


@pytest.fixture
def sample_insights_results():
    """Sample insights results for testing."""
    return {
        "key_findings": [
            {
                "finding": "Strong correlation between views and time on page",
                "confidence": 0.92,
                "evidence": "r = 0.85, p < 0.001",
            }
        ],
        "patterns": ["Seasonal trend detected with weekly peaks"],
        "recommendations": [
            {
                "recommendation": "Optimize content for peak hours",
                "priority": "high",
                "rationale": "30% more engagement during peak times",
            }
        ],
        "summary": "Analysis reveals significant patterns in user engagement",
    }


@pytest.fixture
def sample_visualization_results():
    """Sample visualization results for testing."""
    return {
        "charts": [
            {
                "chart_type": "line",
                "title": "Views Over Time",
                "data_points": 100,
                "file_path": "/tmp/views_over_time.html",
            },
            {
                "chart_type": "bar",
                "title": "Views by Section",
                "data_points": 4,
                "file_path": "/tmp/views_by_section.html",
            },
        ],
        "summary": "Generated 2 visualizations",
    }


# ---------------------------------------------------------------------------
# Transformation Operation Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_transform_operations():
    """Sample transformation operations for testing."""
    return [
        {"type": "drop_duplicates"},
        {"type": "fill_nulls", "column": "value1", "method": "mean"},
        {"type": "rename_columns", "mapping": {"value1": "primary_value"}},
        {"type": "filter_rows", "column": "category", "operator": "in", "value": ["A", "B"]},
    ]


# ---------------------------------------------------------------------------
# CLI Testing Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner():
    """Create a Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def mock_cli_orchestrator():
    """Mock orchestrator for CLI testing."""
    with patch("src.orchestrator.main.Orchestrator") as mock_class:
        mock_instance = MagicMock()
        mock_instance.load_data = AsyncMock(return_value="Data loaded successfully")
        mock_instance._run_statistical_analysis = AsyncMock(return_value="Analysis complete")
        mock_instance._run_forecast = AsyncMock(return_value="Forecast complete")
        mock_instance._run_sentiment_analysis = AsyncMock(
            return_value="Sentiment analysis complete"
        )
        mock_instance.generate_insights = AsyncMock(return_value="Insights generated")
        mock_instance.generate_report = AsyncMock(return_value="Report generated")
        mock_instance._agent_context = MagicMock()
        mock_instance._agent_context.get_data = MagicMock(return_value=pd.DataFrame())
        mock_instance._agent_context.set_data = MagicMock()
        mock_instance.conversation = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance
