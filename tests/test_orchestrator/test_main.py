"""Tests for the Orchestrator main module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.base import AgentOption
from src.orchestrator.main import Orchestrator
from src.orchestrator.router import IntentType
from src.orchestrator.state import (
    ConversationState,
    StateManager,
    WorkflowPhase,
    WorkflowState,
)


class TestOrchestrator:
    """Test suite for Orchestrator."""

    @pytest.fixture
    def orchestrator(self, patch_anthropic):
        """Create an Orchestrator instance for testing."""
        with (
            patch("src.orchestrator.main.StateManager"),
            patch("src.orchestrator.main.Notifier") as mock_notifier,
            patch("src.orchestrator.main.ObsidianVault"),
            patch("src.orchestrator.main.AsyncAnthropic") as mock_anthropic,
        ):
            mock_notifier_instance = MagicMock()
            mock_notifier.return_value = mock_notifier_instance
            # Configure mock Anthropic client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Option 1\nOption 2")]
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client
            orch = Orchestrator()
            yield orch

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes correctly."""
        assert orchestrator is not None
        assert orchestrator.router is not None
        assert orchestrator.conversation is not None
        assert orchestrator.workflow is None  # No workflow until started

    def test_orchestrator_has_system_prompt(self, orchestrator):
        """Test that orchestrator has a system prompt."""
        assert orchestrator.system_prompt is not None
        assert len(orchestrator.system_prompt) > 0

    def test_orchestrator_agent_registry(self, orchestrator):
        """Test that agent registry is initialized."""
        assert orchestrator._agents == {}  # Lazy loaded

    # -------------------------------------------------------------------------
    # Agent Factory Tests
    # -------------------------------------------------------------------------

    def test_get_retrieval_agent(self, orchestrator):
        """Test getting retrieval agent."""
        agent = orchestrator._get_agent("retrieval")
        assert agent is not None
        assert agent.name == "retrieval"

    def test_get_transform_agent(self, orchestrator):
        """Test getting transform agent."""
        agent = orchestrator._get_agent("transform")
        assert agent is not None
        assert agent.name == "transform"

    def test_get_statistical_agent(self, orchestrator):
        """Test getting statistical agent."""
        agent = orchestrator._get_agent("statistical")
        assert agent is not None
        assert agent.name == "statistical"

    def test_get_insights_agent(self, orchestrator):
        """Test getting insights agent."""
        agent = orchestrator._get_agent("insights")
        assert agent is not None
        assert agent.name == "insights"

    def test_get_modeling_agent(self, orchestrator):
        """Test getting modeling agent."""
        agent = orchestrator._get_agent("modeling")
        assert agent is not None
        assert agent.name == "modeling"

    def test_get_visualization_agent(self, orchestrator):
        """Test getting visualization agent."""
        agent = orchestrator._get_agent("visualization")
        assert agent is not None
        assert agent.name == "visualization"

    def test_get_report_agent(self, orchestrator):
        """Test getting report agent."""
        agent = orchestrator._get_agent("report")
        assert agent is not None
        assert agent.name == "report"

    def test_get_arabic_nlp_agent(self, orchestrator):
        """Test getting Arabic NLP agent."""
        agent = orchestrator._get_agent("arabic_nlp")
        assert agent is not None
        assert agent.name == "arabic_nlp"

    def test_get_unknown_agent_raises_error(self, orchestrator):
        """Test that getting unknown agent raises error."""
        with pytest.raises(ValueError) as exc_info:
            orchestrator._get_agent("unknown_agent")
        assert "unknown" in str(exc_info.value).lower()

    def test_agent_caching(self, orchestrator):
        """Test that agents are cached after first creation."""
        agent1 = orchestrator._get_agent("retrieval")
        agent2 = orchestrator._get_agent("retrieval")
        assert agent1 is agent2

    # -------------------------------------------------------------------------
    # Message Processing Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_process_message_help(self, orchestrator):
        """Test processing help message."""
        response = await orchestrator.process_message("help")

        assert "help" in response.lower() or "analyst" in response.lower()
        assert orchestrator.conversation.current_intent == "help"

    @pytest.mark.asyncio
    async def test_process_message_status(self, orchestrator):
        """Test processing status message."""
        response = await orchestrator.process_message("status")

        assert "no active workflow" in response.lower() or "status" in response.lower()
        assert orchestrator.conversation.current_intent == "status"

    @pytest.mark.asyncio
    async def test_process_message_unknown(self, orchestrator):
        """Test processing unknown message."""
        response = await orchestrator.process_message("xyzzy gibberish random")

        # Should request clarification
        assert len(response) > 0
        assert orchestrator.conversation.current_intent == "unknown"

    @pytest.mark.asyncio
    async def test_process_message_adds_to_conversation(self, orchestrator):
        """Test that messages are added to conversation history."""
        await orchestrator.process_message("help")

        assert len(orchestrator.conversation.messages) >= 2  # User + assistant

    @pytest.mark.asyncio
    async def test_process_message_saves_state(self, orchestrator):
        """Test that state is saved after processing."""
        orchestrator.state_manager.save_conversation = MagicMock()

        await orchestrator.process_message("help")

        orchestrator.state_manager.save_conversation.assert_called()

    # -------------------------------------------------------------------------
    # Analysis Intent Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_statistical_analysis_intent(self, orchestrator):
        """Test handling statistical analysis intent."""
        response = await orchestrator.process_message("Analyze the data statistically")

        # Should present options
        assert "option" in response.lower() or "approach" in response.lower()
        assert orchestrator.workflow is not None
        assert orchestrator.workflow.phase == WorkflowPhase.OPTION_PRESENTATION

    @pytest.mark.asyncio
    async def test_handle_forecast_intent(self, orchestrator):
        """Test handling forecast intent."""
        response = await orchestrator.process_message("Forecast next 30 days of sales")

        # Should present options (forecast is high-stakes)
        assert len(response) > 0
        # High-stakes should trigger approval request
        assert "high-stakes" in response.lower() or "option" in response.lower()

    # -------------------------------------------------------------------------
    # Option Presentation Tests
    # -------------------------------------------------------------------------

    def test_format_options_response(self, orchestrator):
        """Test formatting options for user."""
        from src.orchestrator.router import Intent

        options = [
            AgentOption(
                id="comprehensive",
                title="Comprehensive EDA",
                description="Full analysis",
                recommended=True,
                pros=["Thorough", "Complete"],
                cons=["Takes longer"],
                estimated_complexity="medium",
            ),
            AgentOption(
                id="quick",
                title="Quick Summary",
                description="Basic stats",
                pros=["Fast"],
                cons=["Less detail"],
                estimated_complexity="low",
            ),
        ]

        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
        )

        response = orchestrator._format_options_response(intent, options)

        assert "Option 1" in response
        assert "Option 2" in response
        assert "Comprehensive EDA" in response
        assert "Recommended" in response
        assert "Thorough" in response

    def test_format_options_sets_pending_approval(self, orchestrator):
        """Test that formatting options sets pending approval."""
        from src.orchestrator.router import Intent

        options = [
            AgentOption(
                id="test",
                title="Test Option",
                description="Test",
                recommended=True,
            )
        ]

        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
        )

        orchestrator._format_options_response(intent, options)

        assert orchestrator.conversation.pending_approval is not None
        assert orchestrator.conversation.pending_approval["type"] == "option_selection"

    def test_get_default_options_statistical(self, orchestrator):
        """Test getting default options for statistical analysis."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
        )

        options = orchestrator._get_default_options(intent)

        assert len(options) >= 2
        assert any(o.id == "comprehensive" for o in options)
        assert any(o.recommended for o in options)

    def test_get_default_options_sentiment(self, orchestrator):
        """Test getting default options for sentiment analysis."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.SENTIMENT_ANALYSIS,
            confidence=1.0,
        )

        options = orchestrator._get_default_options(intent)

        assert len(options) >= 1
        assert any("marbert" in o.id.lower() for o in options)

    def test_get_default_options_forecast(self, orchestrator):
        """Test getting default options for forecasting."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.FORECAST,
            confidence=1.0,
        )

        options = orchestrator._get_default_options(intent)

        assert len(options) >= 2
        assert any("prophet" in o.id.lower() or "arima" in o.id.lower() for o in options)

    # -------------------------------------------------------------------------
    # Approval Handling Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_option_selection_by_number(self, orchestrator):
        """Test handling option selection by number."""
        orchestrator.conversation.pending_approval = {
            "type": "option_selection",
            "options": ["comprehensive", "quick", "targeted"],
            "intent": "statistical_analysis",
        }

        response = await orchestrator._handle_approval_response("1")

        assert orchestrator.conversation.selected_option == "comprehensive"
        assert orchestrator.conversation.pending_approval is None

    @pytest.mark.asyncio
    async def test_handle_option_selection_by_name(self, orchestrator):
        """Test handling option selection by name."""
        orchestrator.conversation.pending_approval = {
            "type": "option_selection",
            "options": ["comprehensive", "quick", "targeted"],
            "intent": "statistical_analysis",
        }

        response = await orchestrator._handle_approval_response("comprehensive")

        assert orchestrator.conversation.selected_option == "comprehensive"

    @pytest.mark.asyncio
    async def test_handle_invalid_option_selection(self, orchestrator):
        """Test handling invalid option selection."""
        orchestrator.conversation.pending_approval = {
            "type": "option_selection",
            "options": ["comprehensive", "quick"],
            "intent": "statistical_analysis",
        }

        response = await orchestrator._handle_approval_response("invalid_option")

        assert "didn't understand" in response.lower() or "choose from" in response.lower()
        assert orchestrator.conversation.pending_approval is not None

    @pytest.mark.asyncio
    async def test_handle_high_stakes_approval_yes(self, orchestrator):
        """Test handling high-stakes approval with yes."""
        orchestrator.conversation.pending_approval = {
            "type": "high_stakes",
            "action": "forecast",
            "reasons": ["Contains high-stakes keyword"],
        }

        response = await orchestrator._handle_approval_response("yes")

        assert "approved" in response.lower() or "proceed" in response.lower()
        assert orchestrator.conversation.pending_approval is None

    @pytest.mark.asyncio
    async def test_handle_high_stakes_approval_no(self, orchestrator):
        """Test handling high-stakes approval with no."""
        orchestrator.conversation.pending_approval = {
            "type": "high_stakes",
            "action": "delete",
            "reasons": ["Contains high-stakes keyword: delete"],
        }

        response = await orchestrator._handle_approval_response("no")

        assert "cancelled" in response.lower() or "how else" in response.lower()
        assert orchestrator.conversation.pending_approval is None

    @pytest.mark.asyncio
    async def test_handle_no_pending_approval(self, orchestrator):
        """Test handling approval response with no pending approval."""
        orchestrator.conversation.pending_approval = None

        response = await orchestrator._handle_approval_response("yes")

        assert "no pending" in response.lower()

    # -------------------------------------------------------------------------
    # High Stakes Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_request_high_stakes_approval(self, orchestrator):
        """Test requesting approval for high-stakes operations."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
            high_stakes=True,
            high_stakes_reasons=["Contains high-stakes keyword: 'delete'"],
        )

        response = await orchestrator._request_high_stakes_approval(intent)

        assert "high-stakes" in response.lower()
        assert "delete" in response.lower()
        assert orchestrator.conversation.pending_approval is not None
        assert orchestrator.conversation.pending_approval["type"] == "high_stakes"

    # -------------------------------------------------------------------------
    # Clarification Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_request_clarification(self, orchestrator):
        """Test requesting clarification from user."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.LOAD_DATA,
            confidence=0.5,
            clarifications_needed=["Which file would you like to load?"],
        )

        response = await orchestrator._request_clarification(intent)

        assert "more information" in response.lower() or "file" in response.lower()

    @pytest.mark.asyncio
    async def test_request_clarification_empty(self, orchestrator):
        """Test requesting clarification with no specific questions."""
        from src.orchestrator.router import Intent

        intent = Intent(
            type=IntentType.UNKNOWN,
            confidence=0.0,
            clarifications_needed=[],
        )

        response = await orchestrator._request_clarification(intent)

        assert "rephrase" in response.lower() or "details" in response.lower()

    # -------------------------------------------------------------------------
    # Help and Status Response Tests
    # -------------------------------------------------------------------------

    def test_generate_help_response(self, orchestrator):
        """Test generating help response."""
        response = orchestrator._generate_help_response()

        assert "help" in response.lower() or "analyst" in response.lower()
        assert "data" in response.lower()
        assert "analysis" in response.lower()

    def test_generate_status_response_no_workflow(self, orchestrator):
        """Test generating status response with no active workflow."""
        orchestrator.workflow = None

        response = orchestrator._generate_status_response()

        assert "no active workflow" in response.lower()

    def test_generate_status_response_with_workflow(self, orchestrator):
        """Test generating status response with active workflow."""
        orchestrator.workflow = WorkflowState()
        orchestrator.workflow.agents_used = ["retrieval", "statistical"]
        orchestrator.workflow.phase = WorkflowPhase.ANALYSIS

        response = orchestrator._generate_status_response()

        assert orchestrator.workflow.workflow_id in response
        assert "analysis" in response.lower()
        assert "retrieval" in response.lower()

    # -------------------------------------------------------------------------
    # Data Loading Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_load_data(self, orchestrator, sample_csv_file):
        """Test loading data from file."""
        # Mock the retrieval agent's run method
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = MagicMock()
        mock_result.data.data = pd.DataFrame({"a": [1, 2, 3]})
        mock_result.data.profile = MagicMock()
        mock_result.data.profile.to_dict.return_value = {"rows": 3}
        mock_result.data.profile.checksum = "abc123"
        mock_result.data.quality = MagicMock()
        mock_result.data.quality.to_dict.return_value = {"score": 0.9}

        retrieval_agent = orchestrator._get_agent("retrieval")
        retrieval_agent.run = AsyncMock(return_value=mock_result)
        retrieval_agent.format_profile_output = MagicMock(return_value="Profile output")
        retrieval_agent.format_quality_output = MagicMock(return_value="Quality output")

        response = await orchestrator.load_data(str(sample_csv_file))

        assert "profile" in response.lower() or "quality" in response.lower()

    @pytest.mark.asyncio
    async def test_load_data_failure(self, orchestrator):
        """Test handling data loading failure."""
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "File not found"

        retrieval_agent = orchestrator._get_agent("retrieval")
        retrieval_agent.run = AsyncMock(return_value=mock_result)

        response = await orchestrator.load_data("nonexistent.csv")

        assert "failed" in response.lower()
        assert "file not found" in response.lower()

    # -------------------------------------------------------------------------
    # Insights Generation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_insights_no_data(self, orchestrator):
        """Test generating insights with no analysis results."""
        response = await orchestrator.generate_insights()

        assert "no analysis" in response.lower() or "run an analysis" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_insights_with_data(self, orchestrator, sample_statistical_results):
        """Test generating insights with analysis results."""
        orchestrator._agent_context.set_data("statistical_results", sample_statistical_results)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = MagicMock()
        mock_result.data.to_dict.return_value = {"insights": "test"}

        insights_agent = orchestrator._get_agent("insights")
        insights_agent.run = AsyncMock(return_value=mock_result)
        insights_agent.format_output = MagicMock(return_value="Formatted insights")

        response = await orchestrator.generate_insights()

        assert "formatted insights" in response.lower()

    # -------------------------------------------------------------------------
    # Visualization Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_visualizations_no_data(self, orchestrator):
        """Test generating visualizations with no data."""
        response = await orchestrator.generate_visualizations()

        assert "no data" in response.lower() or "load data" in response.lower()

    @pytest.mark.asyncio
    async def test_get_visualization_options(self, orchestrator, sample_dataframe):
        """Test getting visualization options."""
        orchestrator._agent_context.set_data("loaded_data", sample_dataframe)

        visualization_agent = orchestrator._get_agent("visualization")
        visualization_agent.get_chart_type_options = MagicMock(
            return_value=[
                AgentOption(
                    id="line", title="Line Chart", description="Show trends", recommended=True
                )
            ]
        )

        response = await orchestrator.get_visualization_options()

        assert "option" in response.lower() or "chart" in response.lower()

    # -------------------------------------------------------------------------
    # Report Generation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_generate_report_no_content(self, orchestrator):
        """Test generating report with no content."""
        response = await orchestrator.generate_report()

        assert "no content" in response.lower() or "run analysis" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_report_draft(self, orchestrator, sample_insights_results):
        """Test generating report draft."""
        orchestrator._agent_context.set_data("insights_results", sample_insights_results)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = MagicMock()
        mock_result.data.to_dict.return_value = {"sections": []}

        report_agent = orchestrator._get_agent("report")
        report_agent.run = AsyncMock(return_value=mock_result)
        report_agent.format_output = MagicMock(return_value="Report draft")

        response = await orchestrator.generate_report(draft_only=True)

        assert "draft" in response.lower()
        assert orchestrator.conversation.pending_approval is not None

    @pytest.mark.asyncio
    async def test_get_report_options(self, orchestrator):
        """Test getting report format options."""
        report_agent = orchestrator._get_agent("report")
        report_agent.get_format_options = MagicMock(
            return_value=[
                AgentOption(id="pdf", title="PDF", description="PDF format", recommended=True)
            ]
        )
        report_agent.get_audience_options = MagicMock(
            return_value=[
                AgentOption(id="executive", title="Executive", description="High-level summary")
            ]
        )

        response = await orchestrator.get_report_options()

        assert "format" in response.lower()
        assert "audience" in response.lower()

    # -------------------------------------------------------------------------
    # Classification and Regression Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_classification_no_data(self, orchestrator):
        """Test running classification with no data."""
        response = await orchestrator.run_classification(target_column="category")

        assert "no data" in response.lower() or "load data" in response.lower()

    @pytest.mark.asyncio
    async def test_run_regression_no_data(self, orchestrator):
        """Test running regression with no data."""
        response = await orchestrator.run_regression(target_column="value")

        assert "no data" in response.lower() or "load data" in response.lower()


class TestConversationState:
    """Test suite for ConversationState."""

    def test_add_message(self):
        """Test adding a message to conversation."""
        state = ConversationState()
        state.add_message("user", "Hello")

        assert len(state.messages) == 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hello"

    def test_get_messages_for_llm(self):
        """Test getting messages formatted for LLM."""
        state = ConversationState()
        state.add_message("system", "System prompt")
        state.add_message("user", "Hello")
        state.add_message("assistant", "Hi there")

        messages = state.get_messages_for_llm()

        # Should exclude system messages
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_clear_pending(self):
        """Test clearing pending approval."""
        state = ConversationState()
        state.pending_approval = {"type": "test"}
        state.selected_option = "option1"

        state.clear_pending()

        assert state.pending_approval is None
        assert state.selected_option is None

    def test_record_approval(self):
        """Test recording approval decision."""
        state = ConversationState()
        state.record_approval("test_action", True, "Approved by user")

        assert "approvals" in state.user_preferences
        assert len(state.user_preferences["approvals"]) == 1
        assert state.user_preferences["approvals"][0]["approved"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = ConversationState()
        state.add_message("user", "Hello")

        d = state.to_dict()

        assert "session_id" in d
        assert "messages" in d
        assert len(d["messages"]) == 1

    def test_from_dict(self):
        """Test creation from dictionary."""
        state = ConversationState()
        state.add_message("user", "Hello")
        state.current_intent = "help"

        d = state.to_dict()
        restored = ConversationState.from_dict(d)

        assert restored.session_id == state.session_id
        assert len(restored.messages) == 1
        assert restored.current_intent == "help"


class TestWorkflowState:
    """Test suite for WorkflowState."""

    def test_advance_phase(self):
        """Test advancing workflow phase."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.DATA_RETRIEVAL)

        assert state.phase == WorkflowPhase.DATA_RETRIEVAL
        assert state.completed_at is None

    def test_advance_phase_completed(self):
        """Test advancing to completed phase."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.COMPLETED)

        assert state.phase == WorkflowPhase.COMPLETED
        assert state.completed_at is not None

    def test_advance_phase_failed(self):
        """Test advancing to failed phase."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.FAILED)

        assert state.phase == WorkflowPhase.FAILED
        assert state.completed_at is not None

    def test_record_agent_result(self):
        """Test recording agent result."""
        state = WorkflowState()
        state.record_agent_result("statistical", {"mean": 100})

        assert "statistical" in state.agents_used
        assert "statistical" in state.agent_results

    def test_record_error(self):
        """Test recording error."""
        state = WorkflowState()
        state.record_error("Test error")

        assert len(state.errors) == 1
        assert "Test error" in state.errors[0]

    def test_record_checksum(self):
        """Test recording data checksum."""
        state = WorkflowState()
        state.record_checksum("data.csv", "abc123")

        assert "data.csv" in state.data_checksums
        assert state.data_checksums["data.csv"] == "abc123"

    def test_record_transformation(self):
        """Test recording transformation."""
        state = WorkflowState()
        state.record_transformation("drop_duplicates", {"subset": None})

        assert len(state.transformations) == 1
        assert state.transformations[0]["operation"] == "drop_duplicates"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.ANALYSIS)

        d = state.to_dict()

        assert "workflow_id" in d
        assert d["phase"] == "analysis"

    def test_from_dict(self):
        """Test creation from dictionary."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.ANALYSIS)
        state.record_error("Test")

        d = state.to_dict()
        restored = WorkflowState.from_dict(d)

        assert restored.workflow_id == state.workflow_id
        assert restored.phase == WorkflowPhase.ANALYSIS
        assert len(restored.errors) == 1


class TestStateManager:
    """Test suite for StateManager."""

    @pytest.fixture
    def state_manager(self, temp_directory):
        """Create a StateManager instance for testing."""
        return StateManager(memory_path=temp_directory)

    def test_save_and_load_conversation(self, state_manager):
        """Test saving and loading conversation state."""
        state = ConversationState()
        state.add_message("user", "Hello")

        state_manager.save_conversation(state)
        loaded = state_manager.load_conversation(state.session_id)

        assert loaded is not None
        assert loaded.session_id == state.session_id
        assert len(loaded.messages) == 1

    def test_load_nonexistent_conversation(self, state_manager):
        """Test loading non-existent conversation."""
        loaded = state_manager.load_conversation("nonexistent")

        assert loaded is None

    def test_save_and_load_workflow(self, state_manager):
        """Test saving and loading workflow state."""
        state = WorkflowState()
        state.advance_phase(WorkflowPhase.ANALYSIS)

        state_manager.save_workflow(state)
        loaded = state_manager.load_workflow(state.workflow_id)

        assert loaded is not None
        assert loaded.workflow_id == state.workflow_id
        assert loaded.phase == WorkflowPhase.ANALYSIS

    def test_load_nonexistent_workflow(self, state_manager):
        """Test loading non-existent workflow."""
        loaded = state_manager.load_workflow("nonexistent")

        assert loaded is None

    def test_append_to_history(self, state_manager):
        """Test appending to analysis history."""
        entry = {"analysis": "test", "timestamp": "2024-01-01"}

        state_manager.append_to_history(entry)
        state_manager.append_to_history(entry)

        # Read the history file
        import json

        history_path = state_manager.memory_path / "analysis_history.json"
        with open(history_path) as f:
            history = json.load(f)

        assert len(history) == 2
