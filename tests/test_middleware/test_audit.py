"""Tests for audit logging middleware."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import AutonomyLevel
from src.middleware.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditMiddleware,
    sanitize_data,
    sanitize_text,
    truncate_text,
)


@pytest.fixture(autouse=True)
def reset_middleware():
    """Reset middleware singletons before each test."""
    AuditLogger.reset_instance()
    AuditMiddleware.reset_instance()
    yield
    AuditLogger.reset_instance()
    AuditMiddleware.reset_instance()


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent.autonomy = AutonomyLevel.ADVISORY
    agent.model = "claude-opus-4-5-20251101"
    agent.description = "Test agent"
    return agent


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_agent_lifecycle_events(self):
        """Test agent lifecycle event types exist."""
        assert AuditEventType.AGENT_START == "agent_start"
        assert AuditEventType.AGENT_COMPLETE == "agent_complete"
        assert AuditEventType.AGENT_ERROR == "agent_error"

    def test_llm_events(self):
        """Test LLM event types exist."""
        assert AuditEventType.LLM_CALL == "llm_call"
        assert AuditEventType.LLM_RESPONSE == "llm_response"

    def test_human_in_loop_events(self):
        """Test human-in-loop event types exist."""
        assert AuditEventType.OPTION_PRESENTED == "option_presented"
        assert AuditEventType.OPTION_SELECTED == "option_selected"
        assert AuditEventType.APPROVAL_REQUESTED == "approval_requested"

    def test_cost_events(self):
        """Test cost-related event types exist."""
        assert AuditEventType.COST_RECORDED == "cost_recorded"
        assert AuditEventType.BUDGET_EXCEEDED == "budget_exceeded"


class TestSanitizationFunctions:
    """Tests for data sanitization functions."""

    def test_sanitize_text_api_key(self):
        """Test sanitizing API keys."""
        text = "Using api_key=sk-abc123def456ghi789jkl012mno345"
        result = sanitize_text(text)
        assert "sk-abc123" not in result
        assert "REDACTED" in result

    def test_sanitize_text_password(self):
        """Test sanitizing passwords."""
        text = "password=mysecretpassword123"
        result = sanitize_text(text)
        assert "mysecretpassword123" not in result
        assert "REDACTED" in result

    def test_sanitize_text_hf_token(self):
        """Test sanitizing HuggingFace tokens."""
        text = "Using token hf_abcdefghijklmnopqrstuvwxyz123"
        result = sanitize_text(text)
        assert "hf_abcdefghijklmnopqrstuvwxyz123" not in result
        assert "REDACTED" in result

    def test_sanitize_text_no_sensitive(self):
        """Test that safe text is unchanged."""
        text = "This is safe text without secrets"
        result = sanitize_text(text)
        assert result == text

    def test_truncate_text_short(self):
        """Test truncation of short text."""
        text = "Short text"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_truncate_text_long(self):
        """Test truncation of long text."""
        text = "x" * 1500
        result = truncate_text(text, max_length=1000)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_sanitize_data_dict(self):
        """Test sanitizing nested dictionary."""
        data = {
            "key": "api_key=abc123def456ghi789jkl012mno345",
            "nested": {"password": "pwd=secret456"},
        }
        result = sanitize_data(data)
        assert "abc123def456ghi789jkl012mno345" not in str(result)
        assert "secret456" not in str(result)

    def test_sanitize_data_list(self):
        """Test sanitizing list data."""
        data = ["password=secret1", "password=secret2"]
        result = sanitize_data(data)
        assert "secret1" not in str(result)
        assert "secret2" not in str(result)

    def test_sanitize_data_long_list(self):
        """Test truncating long lists."""
        data = list(range(200))
        result = sanitize_data(data)
        assert len(result) <= 101  # 100 items + truncation message


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_START,
            timestamp=datetime.utcnow(),
            agent_name="test_agent",
            session_id="session-123",
            data={"test": "data"},
        )

        assert event.event_type == AuditEventType.AGENT_START
        assert event.agent_name == "test_agent"
        assert event.session_id == "session-123"

    def test_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime.utcnow()
        event = AuditEvent(
            event_type=AuditEventType.AGENT_START,
            timestamp=timestamp,
            agent_name="test_agent",
            session_id="session-123",
            data={"test": "data"},
        )

        d = event.to_dict()
        assert d["event_type"] == "agent_start"
        assert d["agent_name"] == "test_agent"
        assert d["timestamp"] == timestamp.isoformat()

    def test_to_json(self):
        """Test converting event to JSON."""
        event = AuditEvent(
            event_type=AuditEventType.AGENT_START,
            timestamp=datetime.utcnow(),
            agent_name="test_agent",
            session_id="session-123",
            data={"test": "data"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "agent_start"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = AuditLogger.get_instance()
        instance2 = AuditLogger.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = AuditLogger.get_instance()
        AuditLogger.reset_instance()
        instance2 = AuditLogger.get_instance()
        assert instance1 is not instance2

    def test_log_event(self, temp_log_dir):
        """Test logging an event."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file, buffer_size=100)

        event = audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-123",
            data={"test": "data"},
        )

        assert event.event_type == AuditEventType.AGENT_START
        assert log_file.exists()

    def test_log_event_sanitizes_data(self, temp_log_dir):
        """Test that logging sanitizes sensitive data."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        event = audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-123",
            data={"api_key": "sk-verysecretkey123456789012"},
        )

        assert "sk-verysecretkey" not in str(event.data)

    def test_buffer_size_limit(self, temp_log_dir):
        """Test that buffer respects size limit."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file, buffer_size=5)

        for i in range(10):
            audit_logger.log_event(
                event_type=AuditEventType.AGENT_START,
                agent_name="test_agent",
                session_id=f"session-{i}",
                data={},
            )

        assert len(audit_logger._buffer) == 5

    def test_get_recent_events(self, temp_log_dir):
        """Test getting recent events."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        for i in range(5):
            audit_logger.log_event(
                event_type=AuditEventType.AGENT_START,
                agent_name="test_agent",
                session_id=f"session-{i}",
                data={},
            )

        events = audit_logger.get_recent_events(count=3)
        assert len(events) == 3

    def test_get_recent_events_filter_by_type(self, temp_log_dir):
        """Test filtering recent events by type."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )
        audit_logger.log_event(
            event_type=AuditEventType.AGENT_COMPLETE,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )

        events = audit_logger.get_recent_events(event_type=AuditEventType.AGENT_START)
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.AGENT_START

    def test_get_session_events(self, temp_log_dir):
        """Test getting events for a session."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )
        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-2",
            data={},
        )

        events = audit_logger.get_session_events("session-1")
        assert len(events) == 1

    def test_get_event_counts(self, temp_log_dir):
        """Test getting event counts."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )
        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-2",
            data={},
        )

        counts = audit_logger.get_event_counts()
        assert counts["agent_start"] == 2

    def test_get_stats(self, temp_log_dir):
        """Test getting logger statistics."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file, buffer_size=100)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )

        stats = audit_logger.get_stats()
        assert stats["buffer_size"] == 1
        assert stats["max_buffer_size"] == 100
        assert stats["total_events"] == 1

    def test_clear_buffer(self, temp_log_dir):
        """Test clearing the buffer."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )

        audit_logger.clear_buffer()
        assert len(audit_logger._buffer) == 0

    def test_clear_session(self, temp_log_dir):
        """Test clearing session events."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)

        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-1",
            data={},
        )
        audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name="test_agent",
            session_id="session-2",
            data={},
        )

        audit_logger.clear_session("session-1")
        events = audit_logger.get_session_events("session-1")
        assert len(events) == 0


class TestAuditMiddleware:
    """Tests for AuditMiddleware class."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = AuditMiddleware.get_instance()
        instance2 = AuditMiddleware.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = AuditMiddleware.get_instance()
        AuditMiddleware.reset_instance()
        instance2 = AuditMiddleware.get_instance()
        assert instance1 is not instance2

    def test_log_agent_start(self, mock_agent, temp_log_dir):
        """Test logging agent start."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_agent_start(
            agent=mock_agent,
            session_id="session-123",
            kwargs={"test": "args"},
        )

        assert event.event_type == AuditEventType.AGENT_START
        assert event.agent_name == mock_agent.name

    def test_log_agent_complete(self, mock_agent, temp_log_dir):
        """Test logging agent completion."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_agent_complete(
            agent=mock_agent,
            session_id="session-123",
            success=True,
            execution_time_ms=150.5,
            result_summary="Completed successfully",
        )

        assert event.event_type == AuditEventType.AGENT_COMPLETE
        assert event.data["success"] is True

    def test_log_agent_error(self, mock_agent, temp_log_dir):
        """Test logging agent error."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_agent_error(
            agent=mock_agent,
            session_id="session-123",
            error="Test error message",
            execution_time_ms=50.0,
        )

        assert event.event_type == AuditEventType.AGENT_ERROR
        assert "Test error" in event.data["error"]

    def test_log_llm_call(self, mock_agent, temp_log_dir):
        """Test logging LLM call."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_llm_call(
            agent=mock_agent,
            session_id="session-123",
            message_count=3,
            system_prompt_preview="You are a helpful assistant...",
        )

        assert event.event_type == AuditEventType.LLM_CALL
        assert event.data["message_count"] == 3

    def test_log_llm_response(self, mock_agent, temp_log_dir):
        """Test logging LLM response."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_llm_response(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            response_preview="The analysis shows...",
        )

        assert event.event_type == AuditEventType.LLM_RESPONSE
        assert event.data["input_tokens"] == 1000

    def test_log_option_presented(self, mock_agent, temp_log_dir):
        """Test logging option presentation."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        options = [
            {"id": "opt1", "title": "Option 1"},
            {"id": "opt2", "title": "Option 2"},
        ]
        event = middleware.log_option_presented(
            agent=mock_agent,
            session_id="session-123",
            options=options,
            context_message="Please choose an option",
        )

        assert event.event_type == AuditEventType.OPTION_PRESENTED
        assert event.data["option_count"] == 2

    def test_log_high_stakes_detected(self, mock_agent, temp_log_dir):
        """Test logging high stakes detection."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_high_stakes_detected(
            agent=mock_agent,
            session_id="session-123",
            keywords=["delete", "production"],
            input_preview="Delete data from production...",
        )

        assert event.event_type == AuditEventType.HIGH_STAKES_DETECTED
        assert "delete" in event.data["keywords"]

    def test_log_cost_recorded(self, mock_agent, temp_log_dir):
        """Test logging cost recording."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_cost_recorded(
            agent=mock_agent,
            session_id="session-123",
            cost=0.05,
            input_tokens=1000,
            output_tokens=500,
        )

        assert event.event_type == AuditEventType.COST_RECORDED
        assert event.data["cost_usd"] == 0.05

    def test_log_budget_exceeded(self, temp_log_dir):
        """Test logging budget exceeded."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_budget_exceeded(
            session_id="session-123",
            current_cost=6.0,
            threshold=5.0,
        )

        assert event.event_type == AuditEventType.BUDGET_EXCEEDED
        assert event.data["exceeded_by_usd"] == 1.0

    def test_log_session_start(self, temp_log_dir):
        """Test logging session start."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_session_start(
            session_id="session-123",
            user_id="user-456",
        )

        assert event.event_type == AuditEventType.SESSION_START
        assert event.data["user_id"] == "user-456"

    def test_log_session_end(self, temp_log_dir):
        """Test logging session end."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        event = middleware.log_session_end(
            session_id="session-123",
            total_cost=2.5,
            agent_count=4,
        )

        assert event.event_type == AuditEventType.SESSION_END
        assert event.data["total_cost_usd"] == 2.5

    def test_get_session_audit_trail(self, mock_agent, temp_log_dir):
        """Test getting complete session audit trail."""
        log_file = temp_log_dir / "audit.jsonl"
        audit_logger = AuditLogger(log_file=log_file)
        middleware = AuditMiddleware(audit_logger)

        middleware.log_session_start("session-123", "user-1")
        middleware.log_agent_start(mock_agent, "session-123", {})
        middleware.log_agent_complete(mock_agent, "session-123", True, 100.0)

        trail = middleware.get_session_audit_trail("session-123")
        assert len(trail) == 3
        assert trail[0]["event_type"] == "session_start"
