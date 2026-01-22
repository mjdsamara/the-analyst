"""Tests for the base agent classes."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.base import (
    AgentContext,
    AgentOption,
    AgentResult,
    AgentStatus,
    BaseAgent,
)
from src.config import AutonomyLevel


class TestAgentStatus:
    """Test suite for AgentStatus enum."""

    def test_status_values(self):
        """Test all status enum values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.AWAITING_APPROVAL.value == "awaiting_approval"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"

    def test_status_is_string_enum(self):
        """Test that status values can be used as strings."""
        assert str(AgentStatus.IDLE) == "AgentStatus.IDLE"
        assert AgentStatus.IDLE == "idle"


class TestAgentContext:
    """Test suite for AgentContext dataclass."""

    @pytest.fixture
    def context(self):
        """Create a basic context for testing."""
        return AgentContext(session_id="test-session-123")

    def test_initialization_defaults(self):
        """Test context initialization with defaults."""
        context = AgentContext(session_id="sess-001")

        assert context.session_id == "sess-001"
        assert context.user_id == "default"
        assert context.conversation_history == []
        assert context.shared_data == {}
        assert context.approvals == []
        assert context.metadata == {}

    def test_initialization_with_values(self):
        """Test context initialization with custom values."""
        history = [{"role": "user", "content": "hello"}]
        data = {"key": "value"}
        approvals = [{"action": "test", "approved": True}]
        metadata = {"source": "test"}

        context = AgentContext(
            session_id="sess-002",
            user_id="user-123",
            conversation_history=history,
            shared_data=data,
            approvals=approvals,
            metadata=metadata,
        )

        assert context.session_id == "sess-002"
        assert context.user_id == "user-123"
        assert len(context.conversation_history) == 1
        assert context.shared_data["key"] == "value"
        assert len(context.approvals) == 1
        assert context.metadata["source"] == "test"

    def test_add_message(self, context):
        """Test adding a message to conversation history."""
        context.add_message("user", "Hello!")

        assert len(context.conversation_history) == 1
        msg = context.conversation_history[0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello!"
        assert "timestamp" in msg

    def test_add_multiple_messages(self, context):
        """Test adding multiple messages."""
        context.add_message("user", "Question")
        context.add_message("assistant", "Answer")
        context.add_message("user", "Follow-up")

        assert len(context.conversation_history) == 3
        assert context.conversation_history[0]["role"] == "user"
        assert context.conversation_history[1]["role"] == "assistant"
        assert context.conversation_history[2]["content"] == "Follow-up"

    def test_set_data(self, context):
        """Test setting shared data."""
        context.set_data("dataset", {"rows": 100})

        assert context.shared_data["dataset"]["rows"] == 100

    def test_set_data_overwrite(self, context):
        """Test overwriting shared data."""
        context.set_data("key", "value1")
        context.set_data("key", "value2")

        assert context.shared_data["key"] == "value2"

    def test_get_data(self, context):
        """Test retrieving shared data."""
        context.shared_data["test_key"] = "test_value"

        assert context.get_data("test_key") == "test_value"

    def test_get_data_with_default(self, context):
        """Test retrieving data with default value."""
        result = context.get_data("nonexistent", default="fallback")

        assert result == "fallback"

    def test_get_data_missing_no_default(self, context):
        """Test retrieving missing data returns None."""
        result = context.get_data("nonexistent")

        assert result is None

    def test_record_approval(self, context):
        """Test recording an approval."""
        context.record_approval("delete_data", True, "User confirmed")

        assert len(context.approvals) == 1
        approval = context.approvals[0]
        assert approval["action"] == "delete_data"
        assert approval["approved"] is True
        assert approval["reason"] == "User confirmed"
        assert "timestamp" in approval

    def test_record_multiple_approvals(self, context):
        """Test recording multiple approvals."""
        context.record_approval("action1", True)
        context.record_approval("action2", False, "User declined")

        assert len(context.approvals) == 2
        assert context.approvals[0]["approved"] is True
        assert context.approvals[1]["approved"] is False


class TestAgentResult:
    """Test suite for AgentResult dataclass."""

    def test_basic_initialization(self):
        """Test basic result initialization."""
        result = AgentResult(success=True)

        assert result.success is True
        assert result.data is None
        assert result.error is None
        assert result.metadata == {}
        assert result.requires_approval is False
        assert result.approval_request is None
        assert result.logs == []
        assert result.execution_time_ms == 0.0

    def test_initialization_with_data(self):
        """Test result with data."""
        result = AgentResult(success=True, data={"result": 42})

        assert result.success is True
        assert result.data["result"] == 42

    def test_initialization_with_error(self):
        """Test result with error."""
        result = AgentResult(success=False, error="Something went wrong")

        assert result.success is False
        assert result.error == "Something went wrong"

    def test_success_result_factory(self):
        """Test success_result class method."""
        result = AgentResult.success_result(
            data={"count": 100}, operation="test", rows_processed=50
        )

        assert result.success is True
        assert result.data["count"] == 100
        assert result.metadata["operation"] == "test"
        assert result.metadata["rows_processed"] == 50

    def test_error_result_factory(self):
        """Test error_result class method."""
        result = AgentResult.error_result(error="File not found", file_path="/data/missing.csv")

        assert result.success is False
        assert result.error == "File not found"
        assert result.metadata["file_path"] == "/data/missing.csv"

    def test_approval_required_factory(self):
        """Test approval_required class method."""
        result = AgentResult.approval_required(
            action="Delete all data",
            reason="Destructive operation",
            details={"rows_affected": 1000},
        )

        assert result.success is True
        assert result.requires_approval is True
        assert result.approval_request is not None
        assert "action" in result.approval_request
        assert "reason" in result.approval_request

    def test_result_with_logs(self):
        """Test result with execution logs."""
        result = AgentResult(
            success=True,
            logs=["Started processing", "Completed step 1", "Finished"],
            execution_time_ms=1500.5,
        )

        assert len(result.logs) == 3
        assert result.execution_time_ms == 1500.5


class TestAgentOption:
    """Test suite for AgentOption dataclass."""

    def test_basic_initialization(self):
        """Test basic option initialization."""
        option = AgentOption(
            id="opt1",
            title="Option One",
            description="This is the first option",
        )

        assert option.id == "opt1"
        assert option.title == "Option One"
        assert option.description == "This is the first option"
        assert option.recommended is False
        assert option.pros == []
        assert option.cons == []
        assert option.estimated_complexity == ""

    def test_full_initialization(self):
        """Test option with all fields."""
        option = AgentOption(
            id="opt2",
            title="Advanced Analysis",
            description="Full statistical analysis",
            recommended=True,
            pros=["Comprehensive", "Accurate"],
            cons=["Slower", "More expensive"],
            estimated_complexity="high",
        )

        assert option.recommended is True
        assert len(option.pros) == 2
        assert len(option.cons) == 2
        assert option.estimated_complexity == "high"


class TestBaseAgent:
    """Test suite for BaseAgent class."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock the Anthropic client."""
        with patch("src.agents.base.AsyncAnthropic") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def retrieval_agent(self, mock_anthropic):
        """Create a concrete agent for testing (using retrieval agent)."""
        # We need to import a concrete implementation to test BaseAgent
        from src.agents.retrieval import RetrievalAgent

        return RetrievalAgent()

    def test_agent_initialization_unknown_name(self, mock_anthropic):
        """Test that unknown agent names raise ValueError."""

        # Create a minimal concrete class
        class TestAgent(BaseAgent):
            @property
            def system_prompt(self) -> str:
                return "Test prompt"

            async def execute(self, **kwargs):
                return AgentResult.success_result(None)

        with pytest.raises(ValueError, match="Unknown agent"):
            TestAgent(name="unknown_agent_name")

    def test_agent_initialization_valid(self, mock_anthropic):
        """Test valid agent initialization."""
        from src.agents.retrieval import RetrievalAgent

        agent = RetrievalAgent()

        assert agent.name == "retrieval"
        assert agent.status == AgentStatus.IDLE
        assert agent.context is not None
        assert agent.autonomy == AutonomyLevel.SUPERVISED

    def test_agent_with_custom_context(self, mock_anthropic):
        """Test agent initialization with custom context."""
        from src.agents.retrieval import RetrievalAgent

        context = AgentContext(session_id="custom-session", user_id="user-456")
        agent = RetrievalAgent(context=context)

        assert agent.context.session_id == "custom-session"
        assert agent.context.user_id == "user-456"

    def test_log_method(self, retrieval_agent):
        """Test the log method."""
        retrieval_agent.log("Test message")
        retrieval_agent.log("Warning message", level="WARNING")

        assert len(retrieval_agent._logs) == 3  # 2 + initialization log
        assert "Test message" in retrieval_agent._logs[-2]
        assert "WARNING" in retrieval_agent._logs[-1]

    def test_single_job_property(self, retrieval_agent):
        """Test the single_job property."""
        job = retrieval_agent.single_job

        assert isinstance(job, str)
        assert len(job) > 0

    def test_compute_checksum_string(self, retrieval_agent):
        """Test checksum computation for string input."""
        checksum = retrieval_agent.compute_checksum("Hello, World!")

        assert len(checksum) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_checksum_bytes(self, retrieval_agent):
        """Test checksum computation for bytes input."""
        checksum = retrieval_agent.compute_checksum(b"Hello, World!")

        assert len(checksum) == 64

    def test_compute_checksum_deterministic(self, retrieval_agent):
        """Test that checksum is deterministic."""
        data = "Test data for checksum"
        checksum1 = retrieval_agent.compute_checksum(data)
        checksum2 = retrieval_agent.compute_checksum(data)

        assert checksum1 == checksum2

    def test_verify_data_unchanged_file_not_exists(self, retrieval_agent, tmp_path):
        """Test data verification when file doesn't exist."""
        missing_path = tmp_path / "nonexistent.txt"

        result = retrieval_agent.verify_data_unchanged(missing_path, "somechecksum")

        assert result is False

    def test_verify_data_unchanged_valid(self, retrieval_agent, tmp_path):
        """Test data verification with valid checksum."""
        test_file = tmp_path / "test.txt"
        content = b"Test content for verification"
        test_file.write_bytes(content)

        expected_checksum = retrieval_agent.compute_checksum(content)
        result = retrieval_agent.verify_data_unchanged(test_file, expected_checksum)

        assert result is True

    def test_verify_data_unchanged_invalid(self, retrieval_agent, tmp_path):
        """Test data verification with invalid checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        result = retrieval_agent.verify_data_unchanged(test_file, "invalid_checksum")

        assert result is False

    def test_check_high_stakes(self, retrieval_agent):
        """Test high-stakes keyword detection."""
        text = "Please delete all production data"
        matches = retrieval_agent.check_high_stakes(text)

        assert len(matches) > 0
        assert any("delete" in m.lower() for m in matches)

    def test_check_high_stakes_no_matches(self, retrieval_agent):
        """Test high-stakes detection with no matches."""
        text = "Load the CSV file and analyze it"
        matches = retrieval_agent.check_high_stakes(text)

        assert len(matches) == 0

    def test_present_options(self, retrieval_agent):
        """Test presenting options to user."""
        options = [
            AgentOption(
                id="opt1",
                title="Option A",
                description="First option",
                recommended=True,
                pros=["Fast"],
                cons=["Less accurate"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="opt2",
                title="Option B",
                description="Second option",
                recommended=False,
                pros=["Accurate"],
                cons=["Slower"],
                estimated_complexity="high",
            ),
        ]

        result = retrieval_agent.present_options(options, context_message="Choose one")

        assert result["type"] == "options_presentation"
        assert result["agent"] == "retrieval"
        assert result["context"] == "Choose one"
        assert len(result["options"]) == 2
        assert result["options"][0]["id"] == "opt1"
        assert result["options"][0]["recommended"] is True

    def test_verify_output(self, retrieval_agent):
        """Test output verification."""
        # Base implementation returns True with empty list
        passed, failed = retrieval_agent.verify_output({"data": "test"}, ["req1", "req2"])

        assert passed is True
        assert failed == []

    @pytest.mark.asyncio
    async def test_request_approval(self, retrieval_agent):
        """Test requesting approval."""
        result = await retrieval_agent.request_approval(
            action="Delete data",
            reason="High-stakes operation",
            details={"rows": 1000},
        )

        assert retrieval_agent.status == AgentStatus.AWAITING_APPROVAL
        assert result.requires_approval is True
        assert result.approval_request is not None

    @pytest.mark.asyncio
    async def test_call_llm(self):
        """Test calling the LLM."""
        from anthropic.types import TextBlock

        from src.agents.retrieval import RetrievalAgent

        with patch("src.agents.base.AsyncAnthropic") as mock_anthropic_class:
            # Setup mock response with TextBlock
            mock_text_block = MagicMock(spec=TextBlock)
            mock_text_block.text = "LLM response text"

            mock_response = MagicMock()
            mock_response.content = [mock_text_block]
            # Add usage attributes for cost tracking middleware
            mock_response.usage = MagicMock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50

            # Set up the mock client
            mock_client = MagicMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic_class.return_value = mock_client

            # Create the agent (which will use the mocked client)
            agent = RetrievalAgent()

            messages = [{"role": "user", "content": "Hello"}]
            result = await agent.call_llm(messages)

            assert result == "LLM response text"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success(self, mock_anthropic, tmp_path):
        """Test successful agent run."""
        import pandas as pd

        from src.agents.retrieval import RetrievalAgent

        agent = RetrievalAgent()

        # Create test file
        test_file = tmp_path / "test.csv"
        pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}).to_csv(test_file, index=False)

        result = await agent.run(file_path=test_file)

        assert result.success is True
        assert agent.status == AgentStatus.COMPLETED
        assert result.execution_time_ms > 0
        assert len(result.logs) > 0

    @pytest.mark.asyncio
    async def test_run_with_error(self, mock_anthropic, tmp_path):
        """Test agent run with error."""
        from src.agents.retrieval import RetrievalAgent

        agent = RetrievalAgent()
        missing_file = tmp_path / "nonexistent.csv"

        result = await agent.run(file_path=missing_file)

        assert result.success is False
        assert agent.status == AgentStatus.FAILED


class TestBaseAgentHighStakes:
    """Test high-stakes detection in agent runs."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock the Anthropic client."""
        with patch("src.agents.base.AsyncAnthropic") as mock:
            mock_client = MagicMock()
            mock.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_run_detects_high_stakes_advisory(self, mock_anthropic):
        """Test that high-stakes keywords trigger approval for advisory agents."""
        import pandas as pd

        from src.agents.statistical import StatisticalAgent

        agent = StatisticalAgent()
        # StatisticalAgent has ADVISORY autonomy

        # The input contains "production" which is a high-stakes keyword
        result = await agent.run(
            data=pd.DataFrame({"id": [1, 2, 3]}),
            analysis_type="production forecast model",  # Contains high-stakes keyword
        )

        # Should request approval due to high-stakes keywords
        assert result.requires_approval is True
        assert agent.status == AgentStatus.AWAITING_APPROVAL
