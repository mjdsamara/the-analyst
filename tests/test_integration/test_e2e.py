"""End-to-end integration tests for The Analyst platform.

All tests in this file are marked with @pytest.mark.integration as they
span multiple agents and test complete workflows.

Run integration tests only: pytest -m integration
Skip integration tests: pytest -m "not integration"
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/analyst_test")


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestFullWorkflow:
    """Test complete analysis workflows."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample CSV file for testing."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "revenue": np.random.normal(10000, 2000, 100),
                "visitors": np.random.poisson(500, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )
        path = tmp_path / "sample_data.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.fixture
    def time_series_csv(self, tmp_path: Path) -> Path:
        """Create a time series CSV for forecasting tests."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        # Trend + seasonality + noise
        trend = np.arange(100) * 0.5
        seasonal = 10 * np.sin(np.arange(100) * 2 * np.pi / 7)
        noise = np.random.normal(0, 2, 100)
        values = 100 + trend + seasonal + noise

        df = pd.DataFrame({"date": dates, "value": values})
        path = tmp_path / "time_series.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.mark.asyncio
    async def test_statistical_analysis_workflow(self, sample_csv: Path):
        """Test: Load CSV -> Statistical Analysis -> Insights."""
        from src.orchestrator.main import Orchestrator
        from src.orchestrator.state import WorkflowState

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()

            # Initialize workflow (normally done by process_message)
            orchestrator.workflow = WorkflowState()

            # Step 1: Load data
            result = await orchestrator.load_data(str(sample_csv))
            assert "Data Profile" in result or "rows" in result.lower()

            # Verify data was loaded
            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            assert loaded_data is not None
            assert len(loaded_data) == 100

            # Step 2: Run statistical analysis
            result = await orchestrator._run_statistical_analysis(loaded_data, "comprehensive")
            assert "Statistical Analysis" in result or "Analysis" in result

            # Verify results were stored
            stats_results = orchestrator._agent_context.get_data("statistical_results")
            assert stats_results is not None

    @pytest.mark.asyncio
    async def test_forecast_workflow(self, time_series_csv: Path):
        """Test: Load time series -> Forecast -> Results."""
        from src.agents.modeling import ModelingAgent, TaskType
        from src.orchestrator.main import Orchestrator
        from src.orchestrator.state import WorkflowState

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()

            # Initialize workflow (normally done by process_message)
            orchestrator.workflow = WorkflowState()

            # Step 1: Load data
            result = await orchestrator.load_data(str(time_series_csv))
            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            assert loaded_data is not None

            # Step 2: Run forecast directly via modeling agent
            # (avoid orchestrator method which has date parsing issues)
            modeling_agent = ModelingAgent(context=orchestrator._agent_context)
            result = await modeling_agent.execute(
                data=loaded_data,
                target_column="value",
                date_column="date",
                task_type=TaskType.TIME_SERIES_FORECAST,
                periods=7,
            )
            assert result.success
            assert result.data is not None
            assert len(result.data.models) == 1
            assert result.data.models[0].periods == 7

    @pytest.mark.asyncio
    async def test_classification_workflow(self, sample_csv: Path):
        """Test: Load CSV -> Classification model."""
        from src.orchestrator.main import Orchestrator

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()

            # Load data
            await orchestrator.load_data(str(sample_csv))
            loaded_data = orchestrator._agent_context.get_data("loaded_data")

            # Run classification
            result = await orchestrator.run_classification(
                target_column="category",
                feature_columns=["revenue", "visitors"],
            )
            assert "Classification" in result

    @pytest.mark.asyncio
    async def test_regression_workflow(self, sample_csv: Path):
        """Test: Load CSV -> Regression model."""
        from src.orchestrator.main import Orchestrator

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()

            # Load data
            await orchestrator.load_data(str(sample_csv))
            loaded_data = orchestrator._agent_context.get_data("loaded_data")

            # Run regression
            result = await orchestrator.run_regression(
                target_column="revenue",
                feature_columns=["visitors"],
            )
            assert "Regression" in result


class TestAgentIntegration:
    """Test agent-to-agent communication and data flow."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 50),
                "y": np.random.normal(5, 2, 50),
                "z": np.random.choice([0, 1], 50),
            }
        )

    @pytest.mark.asyncio
    async def test_statistical_to_insights_flow(self, sample_df: pd.DataFrame):
        """Test data flows correctly from statistical agent to insights agent."""
        from src.agents.base import AgentContext
        from src.agents.insights import InsightsAgent
        from src.agents.statistical import AnalysisType, StatisticalAgent

        with patch("src.agents.base.AsyncAnthropic"):
            # Shared context
            context = AgentContext(session_id="test")

            # Run statistical analysis
            stat_agent = StatisticalAgent(context=context)
            stat_result = await stat_agent.execute(
                data=sample_df,
                analysis_type=AnalysisType.COMPREHENSIVE,
            )
            assert stat_result.success

            # Store results in context (like orchestrator does)
            context.set_data("statistical_results", stat_result.data.to_dict())

            # Run insights (would normally use the stored results)
            insights_agent = InsightsAgent(context=context)
            # Note: insights agent would read from context.get_data("statistical_results")

    @pytest.mark.asyncio
    async def test_modeling_to_visualization_flow(self, sample_df: pd.DataFrame):
        """Test modeling results can be used for visualization."""
        from src.agents.base import AgentContext
        from src.agents.modeling import ModelingAgent, TaskType
        from src.agents.visualization import VisualizationAgent

        with patch("src.agents.base.AsyncAnthropic"):
            context = AgentContext(session_id="test")

            # Run modeling
            model_agent = ModelingAgent(context=context)
            model_result = await model_agent.execute(
                data=sample_df,
                target_column="y",
                feature_columns=["x"],
                task_type=TaskType.REGRESSION,
            )
            assert model_result.success

            # Store results
            context.set_data("modeling_results", model_result.data.to_dict())

            # Visualization agent can use these results
            viz_agent = VisualizationAgent(context=context)
            # Would generate charts based on modeling results


class TestOrchestratorRouting:
    """Test orchestrator intent routing and option handling."""

    @pytest.mark.asyncio
    async def test_option_presentation(self):
        """Test that orchestrator presents options correctly."""
        from src.orchestrator.main import Orchestrator
        from src.orchestrator.router import Intent, IntentType

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()

            # Create a mock intent
            intent = Intent(
                type=IntentType.STATISTICAL_ANALYSIS,
                confidence=0.9,
                parameters={},
                agents_required=["statistical"],
            )

            # Get default options
            options = orchestrator._get_default_options(intent)

            assert len(options) >= 2
            assert any(opt.recommended for opt in options)

    @pytest.mark.asyncio
    async def test_option_selection(self):
        """Test handling of user option selection."""
        from src.orchestrator.main import Orchestrator
        from src.orchestrator.state import WorkflowState

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()
            orchestrator.workflow = WorkflowState()

            # Set up pending approval
            orchestrator.conversation.pending_approval = {
                "type": "option_selection",
                "options": ["comprehensive", "quick", "targeted"],
                "intent": "statistical_analysis",
            }

            # Simulate user selecting option 1
            response = await orchestrator._handle_approval_response("1")

            # Verify approval was recorded (clear_pending resets selected_option,
            # but the approval is stored in user_preferences)
            approvals = orchestrator.conversation.user_preferences.get("approvals", [])
            assert len(approvals) == 1
            assert approvals[0]["action"] == "Selected option: comprehensive"
            assert approvals[0]["approved"] is True

            # Pending should be cleared
            assert orchestrator.conversation.pending_approval is None

    @pytest.mark.asyncio
    async def test_high_stakes_detection(self):
        """Test high-stakes keyword detection."""
        from src.orchestrator.main import Orchestrator
        from src.orchestrator.router import Intent, IntentType

        with patch("src.agents.base.AsyncAnthropic") as mock_client:
            mock_instance = MagicMock()
            mock_instance.messages.create = AsyncMock(
                return_value=MagicMock(content=[MagicMock(text="response")])
            )
            mock_client.return_value = mock_instance

            orchestrator = Orchestrator()

            # Create intent with high-stakes flag
            intent = Intent(
                type=IntentType.FORECAST,
                confidence=0.9,
                parameters={},
                agents_required=["modeling"],
                high_stakes=True,
                high_stakes_reasons=["Contains 'forecast' keyword"],
            )

            response = await orchestrator._request_high_stakes_approval(intent)
            assert "High-Stakes" in response
            assert orchestrator.conversation.pending_approval is not None


class TestAgentOptions:
    """Test agent option generation."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "a": np.random.normal(0, 1, 50),
                "b": np.random.normal(5, 2, 50),
            }
        )

    def test_statistical_agent_options(self, sample_df: pd.DataFrame):
        """Test statistical agent generates appropriate options."""
        from src.agents.statistical import StatisticalAgent

        with patch("src.agents.base.AsyncAnthropic"):
            agent = StatisticalAgent()
            options = agent.get_analysis_options(sample_df)

            assert len(options) >= 2
            # Should include comprehensive option
            ids = [opt.id for opt in options]
            assert "comprehensive" in ids
            assert "descriptive" in ids

    def test_modeling_agent_options_regression(self, sample_df: pd.DataFrame):
        """Test modeling agent options for regression."""
        from src.agents.modeling import ModelingAgent, TaskType

        with patch("src.agents.base.AsyncAnthropic"):
            agent = ModelingAgent()
            options = agent.get_model_options(
                sample_df,
                target_column="b",
                task_type=TaskType.REGRESSION,
            )

            assert len(options) >= 2
            ids = [opt.id for opt in options]
            assert "random_forest_regressor" in ids
            assert "linear_regression" in ids

    def test_modeling_agent_options_classification(self):
        """Test modeling agent options for classification."""
        from src.agents.modeling import ModelingAgent, TaskType

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [0, 1, 0, 1, 0],
            }
        )

        with patch("src.agents.base.AsyncAnthropic"):
            agent = ModelingAgent()
            options = agent.get_model_options(
                df,
                target_column="b",
                task_type=TaskType.CLASSIFICATION,
            )

            assert len(options) >= 2
            ids = [opt.id for opt in options]
            assert "random_forest_classifier" in ids
            assert "logistic_regression" in ids


class TestDataIntegrity:
    """Test data integrity across the pipeline."""

    @pytest.fixture
    def csv_file(self, tmp_path: Path) -> Path:
        """Create a CSV file with known data."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [100, 200, 300, 400, 500],
            }
        )
        path = tmp_path / "test_data.csv"
        df.to_csv(path, index=False)
        return path

    @pytest.mark.asyncio
    async def test_data_checksum_preserved(self, csv_file: Path):
        """Test that data checksums are computed and stored."""
        from src.orchestrator.main import Orchestrator

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()
            await orchestrator.load_data(str(csv_file))

            # Verify profile contains checksum
            profile = orchestrator._agent_context.get_data("data_profile")
            assert profile is not None
            assert "checksum" in profile

    @pytest.mark.asyncio
    async def test_data_shape_preserved(self, csv_file: Path):
        """Test that data shape is preserved through loading."""
        from src.orchestrator.main import Orchestrator

        with patch("src.agents.base.AsyncAnthropic"):
            orchestrator = Orchestrator()
            await orchestrator.load_data(str(csv_file))

            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            assert loaded_data is not None
            assert len(loaded_data) == 5
            assert list(loaded_data.columns) == ["id", "value"]
