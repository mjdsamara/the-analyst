"""Tests for cost tracking middleware."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.middleware.cost_tracking import (
    OPUS_4_5_INPUT_COST,
    OPUS_4_5_OUTPUT_COST,
    CostEntry,
    CostTracker,
    CostTrackingMiddleware,
    SessionCostSummary,
)


@pytest.fixture(autouse=True)
def reset_middleware():
    """Reset middleware singletons before each test."""
    CostTracker.reset_instance()
    CostTrackingMiddleware.reset_instance()
    yield
    CostTracker.reset_instance()
    CostTrackingMiddleware.reset_instance()


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = "test_agent"
    agent.model = "claude-opus-4-5-20251101"
    return agent


class TestCostConstants:
    """Tests for cost constants."""

    def test_opus_input_cost(self):
        """Test Opus 4.5 input token cost."""
        assert OPUS_4_5_INPUT_COST == 15.0

    def test_opus_output_cost(self):
        """Test Opus 4.5 output token cost."""
        assert OPUS_4_5_OUTPUT_COST == 75.0


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_from_usage_basic(self):
        """Test creating entry from usage."""
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            model="claude-opus-4-5-20251101",
        )

        assert entry.agent_name == "test_agent"
        assert entry.session_id == "session-123"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.model == "claude-opus-4-5-20251101"
        assert isinstance(entry.timestamp, datetime)

    def test_from_usage_cost_calculation(self):
        """Test cost calculation in from_usage."""
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=1_000_000,  # 1M input tokens
            output_tokens=1_000_000,  # 1M output tokens
            model="claude-opus-4-5-20251101",
        )

        assert entry.input_cost == 15.0  # $15/1M
        assert entry.output_cost == 75.0  # $75/1M
        assert entry.total_cost == 90.0

    def test_from_usage_small_amount(self):
        """Test cost calculation with small token amounts."""
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            model="claude-opus-4-5-20251101",
        )

        expected_input_cost = (1000 / 1_000_000) * 15.0
        expected_output_cost = (500 / 1_000_000) * 75.0

        assert abs(entry.input_cost - expected_input_cost) < 0.0001
        assert abs(entry.output_cost - expected_output_cost) < 0.0001
        assert abs(entry.total_cost - (expected_input_cost + expected_output_cost)) < 0.0001

    def test_from_usage_with_metadata(self):
        """Test creating entry with metadata."""
        metadata = {"temperature": 0.7, "max_tokens": 4096}
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            model="claude-opus-4-5-20251101",
            metadata=metadata,
        )

        assert entry.metadata == metadata


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = CostTracker.get_instance()
        instance2 = CostTracker.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = CostTracker.get_instance()
        CostTracker.reset_instance()
        instance2 = CostTracker.get_instance()
        assert instance1 is not instance2

    def test_record_cost(self):
        """Test recording a cost entry."""
        tracker = CostTracker(alert_threshold=10.0)
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            model="claude-opus-4-5-20251101",
        )

        exceeded = tracker.record_cost(entry)
        assert exceeded is False
        assert tracker.get_session_cost("session-123") == entry.total_cost

    def test_record_cost_threshold_exceeded(self):
        """Test recording cost that exceeds threshold."""
        tracker = CostTracker(alert_threshold=0.0001)  # Very low threshold
        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=10000,
            output_tokens=5000,
            model="claude-opus-4-5-20251101",
        )

        exceeded = tracker.record_cost(entry)
        assert exceeded is True

    def test_record_usage(self, mock_agent):
        """Test recording usage via agent."""
        tracker = CostTracker(alert_threshold=10.0)
        entry = tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
        )

        assert entry.agent_name == "test_agent"
        assert entry.input_tokens == 1000

    def test_get_session_summary(self, mock_agent):
        """Test getting session summary."""
        tracker = CostTracker(alert_threshold=10.0)

        # Record multiple entries
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
        )
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=2000,
            output_tokens=1000,
        )

        summary = tracker.get_session_summary("session-123")

        assert summary.session_id == "session-123"
        assert summary.call_count == 2
        assert summary.total_input_tokens == 3000
        assert summary.total_output_tokens == 1500
        assert mock_agent.name in summary.agent_breakdown

    def test_get_session_summary_empty(self):
        """Test getting summary for non-existent session."""
        tracker = CostTracker()
        summary = tracker.get_session_summary("nonexistent")

        assert summary.total_cost == 0.0
        assert summary.call_count == 0
        assert summary.first_call is None

    def test_get_agent_costs(self, mock_agent):
        """Test getting costs for specific agent."""
        tracker = CostTracker()
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-1",
            input_tokens=1000,
            output_tokens=500,
        )

        entries = tracker.get_agent_costs(mock_agent.name)
        assert len(entries) == 1
        assert entries[0].agent_name == mock_agent.name

    def test_get_total_cost(self, mock_agent):
        """Test getting total cost across all sessions."""
        tracker = CostTracker()

        tracker.record_usage(
            agent=mock_agent,
            session_id="session-1",
            input_tokens=1000,
            output_tokens=500,
        )
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-2",
            input_tokens=1000,
            output_tokens=500,
        )

        total = tracker.get_total_cost()
        assert total > 0

    def test_check_budget_within(self):
        """Test budget check when within budget."""
        tracker = CostTracker(alert_threshold=10.0)
        allowed, reason = tracker.check_budget("session-123", 0.01)
        assert allowed is True
        assert "Within budget" in reason

    def test_check_budget_exceeded(self, mock_agent):
        """Test budget check when would exceed."""
        tracker = CostTracker(alert_threshold=0.001)
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=10000,
            output_tokens=5000,
        )

        allowed, reason = tracker.check_budget("session-123", 1.0)
        assert allowed is False
        assert "exceed" in reason.lower()

    def test_clear_session(self, mock_agent):
        """Test clearing session data."""
        tracker = CostTracker()
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
        )

        assert tracker.get_session_cost("session-123") > 0
        tracker.clear_session("session-123")
        assert tracker.get_session_cost("session-123") == 0.0

    def test_register_alert_callback(self):
        """Test registering alert callbacks."""
        tracker = CostTracker(alert_threshold=0.0001)
        callback_called = []

        def callback(session_id, total, threshold):
            callback_called.append((session_id, total, threshold))

        tracker.register_alert_callback(callback)

        entry = CostEntry.from_usage(
            agent_name="test_agent",
            session_id="session-123",
            input_tokens=10000,
            output_tokens=5000,
            model="claude-opus-4-5-20251101",
        )
        tracker.record_cost(entry)

        assert len(callback_called) == 1
        assert callback_called[0][0] == "session-123"

    def test_get_stats(self, mock_agent):
        """Test getting tracker statistics."""
        tracker = CostTracker(alert_threshold=0.0001)  # Low threshold

        tracker.record_usage(
            agent=mock_agent,
            session_id="session-1",
            input_tokens=10000,
            output_tokens=5000,
        )

        stats = tracker.get_stats()
        assert stats["total_entries"] == 1
        assert stats["total_sessions"] == 1
        assert stats["total_cost"] > 0
        assert stats["sessions_over_threshold"] == 1


class TestCostTrackingMiddleware:
    """Tests for CostTrackingMiddleware class."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        instance1 = CostTrackingMiddleware.get_instance()
        instance2 = CostTrackingMiddleware.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = CostTrackingMiddleware.get_instance()
        CostTrackingMiddleware.reset_instance()
        instance2 = CostTrackingMiddleware.get_instance()
        assert instance1 is not instance2

    def test_before_llm_call_allowed(self, mock_agent):
        """Test before_llm_call returns allowed."""
        middleware = CostTrackingMiddleware()
        allowed, reason = middleware.before_llm_call(
            mock_agent, "session-123", estimated_input_tokens=1000
        )
        assert allowed is True

    def test_before_llm_call_budget_check(self, mock_agent):
        """Test before_llm_call checks budget."""
        tracker = CostTracker(alert_threshold=0.0001)  # Very low threshold
        middleware = CostTrackingMiddleware(tracker)

        # Record high cost first
        tracker.record_usage(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )

        allowed, reason = middleware.before_llm_call(
            mock_agent, "session-123", estimated_input_tokens=1_000_000
        )
        assert allowed is False

    def test_after_llm_call(self, mock_agent):
        """Test after_llm_call records cost."""
        middleware = CostTrackingMiddleware()
        entry = middleware.after_llm_call(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
            metadata={"test": "data"},
        )

        assert entry.agent_name == mock_agent.name
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500

    def test_get_session_summary(self, mock_agent):
        """Test getting session summary via middleware."""
        middleware = CostTrackingMiddleware()
        middleware.after_llm_call(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
        )

        summary = middleware.get_session_summary("session-123")
        assert summary.call_count == 1

    def test_is_within_budget(self, mock_agent):
        """Test checking if session is within budget."""
        tracker = CostTracker(alert_threshold=10.0)
        middleware = CostTrackingMiddleware(tracker)

        middleware.after_llm_call(
            agent=mock_agent,
            session_id="session-123",
            input_tokens=1000,
            output_tokens=500,
        )

        assert middleware.is_within_budget("session-123") is True


class TestSessionCostSummary:
    """Tests for SessionCostSummary dataclass."""

    def test_summary_fields(self):
        """Test summary dataclass fields."""
        summary = SessionCostSummary(
            session_id="session-123",
            total_cost=1.5,
            total_input_tokens=10000,
            total_output_tokens=5000,
            call_count=3,
            agent_breakdown={"agent1": 1.0, "agent2": 0.5},
            first_call=datetime.utcnow(),
            last_call=datetime.utcnow(),
            threshold_exceeded=False,
            threshold_value=5.0,
        )

        assert summary.session_id == "session-123"
        assert summary.total_cost == 1.5
        assert summary.call_count == 3
        assert len(summary.agent_breakdown) == 2

    def test_summary_threshold_exceeded(self):
        """Test summary with exceeded threshold."""
        summary = SessionCostSummary(
            session_id="session-123",
            total_cost=10.0,
            total_input_tokens=100000,
            total_output_tokens=50000,
            call_count=5,
            agent_breakdown={"agent1": 10.0},
            first_call=datetime.utcnow(),
            last_call=datetime.utcnow(),
            threshold_exceeded=True,
            threshold_value=5.0,
        )

        assert summary.threshold_exceeded is True
        assert summary.total_cost > summary.threshold_value
