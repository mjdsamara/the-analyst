"""Tests for autonomy middleware."""

from unittest.mock import MagicMock

import pytest

from src.config import AutonomyLevel
from src.middleware.autonomy import (
    AutonomyCheckResult,
    AutonomyConfig,
    AutonomyMiddleware,
    AutonomyViolation,
    RestrictedToolCategory,
)


@pytest.fixture(autouse=True)
def reset_middleware():
    """Reset middleware singleton before each test."""
    AutonomyMiddleware.reset_instance()
    yield
    AutonomyMiddleware.reset_instance()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent.autonomy = AutonomyLevel.ADVISORY
    agent.model = "claude-opus-4-5-20251101"
    agent.description = "Test agent"
    return agent


@pytest.fixture
def supervised_agent():
    """Create a mock supervised agent for testing."""
    agent = MagicMock()
    agent.name = "supervised_agent"
    agent.autonomy = AutonomyLevel.SUPERVISED
    agent.model = "claude-opus-4-5-20251101"
    agent.description = "Supervised test agent"
    return agent


class TestRestrictedToolCategory:
    """Tests for RestrictedToolCategory enum."""

    def test_enum_values(self):
        """Test that all expected categories exist."""
        assert RestrictedToolCategory.EXTERNAL_API == "external_api"
        assert RestrictedToolCategory.DATA_MODIFICATION == "data_modification"
        assert RestrictedToolCategory.COST_INCURRING == "cost_incurring"
        assert RestrictedToolCategory.HIGH_STAKES == "high_stakes"

    def test_enum_count(self):
        """Test that we have exactly 4 categories."""
        assert len(RestrictedToolCategory) == 4


class TestAutonomyConfig:
    """Tests for AutonomyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AutonomyConfig()
        assert RestrictedToolCategory.EXTERNAL_API in config.restricted_categories
        assert RestrictedToolCategory.DATA_MODIFICATION in config.restricted_categories
        assert RestrictedToolCategory.HIGH_STAKES in config.restricted_categories
        assert config.strict_advisory_mode is True
        assert config.block_on_violation is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = AutonomyConfig(
            restricted_categories={RestrictedToolCategory.EXTERNAL_API},
            strict_advisory_mode=False,
            block_on_violation=True,
        )
        assert len(config.restricted_categories) == 1
        assert config.strict_advisory_mode is False
        assert config.block_on_violation is True

    def test_external_api_tools(self):
        """Test default external API tool list."""
        config = AutonomyConfig()
        assert "http_request" in config.external_api_tools
        assert "api_call" in config.external_api_tools

    def test_data_modification_tools(self):
        """Test default data modification tool list."""
        config = AutonomyConfig()
        assert "write_file" in config.data_modification_tools
        assert "delete_file" in config.data_modification_tools


class TestAutonomyCheckResult:
    """Tests for AutonomyCheckResult dataclass."""

    def test_allowed_result(self):
        """Test creating an allowed result."""
        result = AutonomyCheckResult(allowed=True, reason="All checks passed")
        assert result.allowed is True
        assert result.reason == "All checks passed"
        assert result.requires_options is False
        assert result.requires_approval is False

    def test_requires_options_result(self):
        """Test creating a result requiring options."""
        result = AutonomyCheckResult(
            allowed=True,
            reason="Need options",
            requires_options=True,
        )
        assert result.requires_options is True

    def test_requires_approval_result(self):
        """Test creating a result requiring approval."""
        result = AutonomyCheckResult(
            allowed=False,
            reason="Need approval",
            requires_approval=True,
            violation_type="restricted_tool",
        )
        assert result.allowed is False
        assert result.requires_approval is True
        assert result.violation_type == "restricted_tool"


class TestAutonomyViolation:
    """Tests for AutonomyViolation exception."""

    def test_violation_message(self):
        """Test violation exception message formatting."""
        exc = AutonomyViolation(
            message="Must present options",
            agent_name="test_agent",
            autonomy_level=AutonomyLevel.ADVISORY,
        )
        assert "test_agent" in str(exc)
        assert "advisory" in str(exc)
        assert "Must present options" in str(exc)

    def test_violation_attributes(self):
        """Test violation exception attributes."""
        exc = AutonomyViolation(
            message="Test error",
            agent_name="my_agent",
            autonomy_level=AutonomyLevel.SUPERVISED,
        )
        assert exc.agent_name == "my_agent"
        assert exc.autonomy_level == AutonomyLevel.SUPERVISED


class TestAutonomyMiddleware:
    """Tests for AutonomyMiddleware class."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = AutonomyMiddleware.get_instance()
        instance2 = AutonomyMiddleware.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = AutonomyMiddleware.get_instance()
        AutonomyMiddleware.reset_instance()
        instance2 = AutonomyMiddleware.get_instance()
        assert instance1 is not instance2

    def test_register_agent(self, mock_agent):
        """Test agent registration."""
        middleware = AutonomyMiddleware()
        middleware.register_agent(mock_agent)

        state = middleware.get_agent_state(mock_agent.name)
        assert state["autonomy"] == AutonomyLevel.ADVISORY
        assert state["options_presented"] is False
        assert state["execution_count"] == 0

    def test_mark_options_presented(self, mock_agent):
        """Test marking options as presented."""
        middleware = AutonomyMiddleware()
        middleware.register_agent(mock_agent)
        session_id = "test-session"

        assert middleware.check_options_presented(mock_agent.name, session_id) is False
        middleware.mark_options_presented(mock_agent.name, session_id)
        assert middleware.check_options_presented(mock_agent.name, session_id) is True

    def test_check_pre_execute_advisory_no_options(self, mock_agent):
        """Test pre-execute check for advisory agent without options."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        result = middleware.check_pre_execute(mock_agent, session_id)

        assert result.requires_options is True
        assert "ADVISORY" in result.reason

    def test_check_pre_execute_advisory_with_options(self, mock_agent):
        """Test pre-execute check for advisory agent with options presented."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        middleware.mark_options_presented(mock_agent.name, session_id)
        result = middleware.check_pre_execute(mock_agent, session_id)

        assert result.allowed is True
        assert result.requires_options is False

    def test_check_pre_execute_supervised_no_restricted_tool(self, supervised_agent):
        """Test pre-execute check for supervised agent without restricted tool."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        result = middleware.check_pre_execute(supervised_agent, session_id, tool_name="safe_tool")

        assert result.allowed is True
        assert result.requires_approval is False

    def test_check_pre_execute_supervised_restricted_tool(self, supervised_agent):
        """Test pre-execute check for supervised agent with restricted tool."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        result = middleware.check_pre_execute(
            supervised_agent, session_id, tool_name="http_request"
        )

        assert result.requires_approval is True
        assert "restricted tool" in result.reason

    def test_clear_session(self, mock_agent):
        """Test clearing session state."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        middleware.mark_options_presented(mock_agent.name, session_id)
        assert middleware.check_options_presented(mock_agent.name, session_id) is True

        middleware.clear_session(session_id)
        assert middleware.check_options_presented(mock_agent.name, session_id) is False

    def test_validate_execution_blocking_mode(self, mock_agent):
        """Test validate_execution with blocking mode enabled."""
        config = AutonomyConfig(block_on_violation=True)
        middleware = AutonomyMiddleware(config)
        session_id = "test-session"

        with pytest.raises(AutonomyViolation):
            middleware.validate_execution(mock_agent, session_id)

    def test_validate_execution_non_blocking_mode(self, mock_agent):
        """Test validate_execution with blocking mode disabled."""
        config = AutonomyConfig(block_on_violation=False)
        middleware = AutonomyMiddleware(config)
        session_id = "test-session"

        # Should not raise, just log warning
        middleware.validate_execution(mock_agent, session_id)

    def test_get_stats(self, mock_agent):
        """Test getting middleware statistics."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        middleware.register_agent(mock_agent)
        middleware.mark_options_presented(mock_agent.name, session_id)

        stats = middleware.get_stats()
        assert stats["registered_agents"] == 1
        assert stats["options_presented_count"] == 1

    def test_execution_count_increment(self, mock_agent):
        """Test that execution count increments on pre-execute check."""
        middleware = AutonomyMiddleware()
        session_id = "test-session"

        middleware.mark_options_presented(mock_agent.name, session_id)
        middleware.check_pre_execute(mock_agent, session_id)
        middleware.check_pre_execute(mock_agent, session_id)

        state = middleware.get_agent_state(mock_agent.name)
        assert state["execution_count"] == 2

    def test_is_restricted_tool_external_api(self):
        """Test detecting external API tools."""
        middleware = AutonomyMiddleware()
        assert middleware._is_restricted_tool("http_request") is True
        assert middleware._is_restricted_tool("api_call") is True
        assert middleware._is_restricted_tool("safe_tool") is False

    def test_is_restricted_tool_data_modification(self):
        """Test detecting data modification tools."""
        middleware = AutonomyMiddleware()
        assert middleware._is_restricted_tool("write_file") is True
        assert middleware._is_restricted_tool("delete_file") is True
        assert middleware._is_restricted_tool("read_file") is False
