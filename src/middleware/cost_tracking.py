"""
Cost Tracking Middleware for The Analyst.

Tracks API costs per agent per session and alerts when thresholds are exceeded.

BORIS Compliance: Runtime enforcement of cost constraints.
Claude Opus 4.5 Pricing: $15/1M input tokens, $75/1M output tokens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.config import get_settings

if TYPE_CHECKING:
    from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


# Claude Opus 4.5 pricing (per 1M tokens)
OPUS_4_5_INPUT_COST = 15.0  # $15 per 1M input tokens
OPUS_4_5_OUTPUT_COST = 75.0  # $75 per 1M output tokens


@dataclass
class CostEntry:
    """Record of a single API call cost."""

    agent_name: str
    session_id: str
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_usage(
        cls,
        agent_name: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """
        Create a cost entry from token usage.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name used
            metadata: Optional additional metadata

        Returns:
            CostEntry with calculated costs
        """
        input_cost = (input_tokens / 1_000_000) * OPUS_4_5_INPUT_COST
        output_cost = (output_tokens / 1_000_000) * OPUS_4_5_OUTPUT_COST

        return cls(
            agent_name=agent_name,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            model=model,
            metadata=metadata or {},
        )


@dataclass
class SessionCostSummary:
    """Summary of costs for a session."""

    session_id: str
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    call_count: int
    agent_breakdown: dict[str, float]  # agent_name -> total cost
    first_call: datetime | None
    last_call: datetime | None
    threshold_exceeded: bool = False
    threshold_value: float = 0.0


class CostTracker:
    """
    Tracks API costs across sessions and agents.

    Usage:
        tracker = CostTracker()
        tracker.record_cost(entry)
        summary = tracker.get_session_summary(session_id)
    """

    _instance: CostTracker | None = None

    def __init__(self, alert_threshold: float | None = None) -> None:
        """
        Initialize cost tracker.

        Args:
            alert_threshold: Cost threshold for alerts (defaults to config setting)
        """
        settings = get_settings()
        self.alert_threshold = alert_threshold or settings.cost_alert_threshold
        self._entries: list[CostEntry] = []
        self._session_totals: dict[str, float] = {}
        self._alert_callbacks: list[Any] = []

    @classmethod
    def get_instance(cls, alert_threshold: float | None = None) -> CostTracker:
        """Get singleton instance of the tracker."""
        if cls._instance is None:
            cls._instance = cls(alert_threshold)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def record_cost(self, entry: CostEntry) -> bool:
        """
        Record a cost entry.

        Args:
            entry: Cost entry to record

        Returns:
            True if threshold was exceeded
        """
        self._entries.append(entry)

        # Update session total
        current = self._session_totals.get(entry.session_id, 0.0)
        new_total = current + entry.total_cost
        self._session_totals[entry.session_id] = new_total

        logger.debug(
            f"Cost recorded: {entry.agent_name} - ${entry.total_cost:.6f} "
            f"(session total: ${new_total:.4f})"
        )

        # Check threshold
        threshold_exceeded = new_total > self.alert_threshold
        if threshold_exceeded:
            self._handle_threshold_exceeded(entry.session_id, new_total)

        return threshold_exceeded

    def record_usage(
        self,
        agent: BaseAgent,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """
        Record token usage and calculate cost.

        Args:
            agent: Agent that made the call
            session_id: Session identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Optional metadata

        Returns:
            The created cost entry
        """
        entry = CostEntry.from_usage(
            agent_name=agent.name,
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=agent.model,
            metadata=metadata,
        )
        self.record_cost(entry)
        return entry

    def _handle_threshold_exceeded(self, session_id: str, total: float) -> None:
        """
        Handle threshold exceeded event.

        Args:
            session_id: Session that exceeded threshold
            total: Current total cost
        """
        logger.warning(
            f"Cost threshold exceeded for session {session_id}: "
            f"${total:.4f} > ${self.alert_threshold:.2f}"
        )

        for callback in self._alert_callbacks:
            try:
                callback(session_id, total, self.alert_threshold)
            except Exception as e:
                logger.error(f"Cost alert callback failed: {e}")

    def register_alert_callback(self, callback: Any) -> None:
        """
        Register a callback for threshold exceeded alerts.

        Args:
            callback: Function(session_id, total, threshold)
        """
        self._alert_callbacks.append(callback)

    def get_session_summary(self, session_id: str) -> SessionCostSummary:
        """
        Get cost summary for a session.

        Args:
            session_id: Session to summarize

        Returns:
            SessionCostSummary with aggregated data
        """
        session_entries = [e for e in self._entries if e.session_id == session_id]

        if not session_entries:
            return SessionCostSummary(
                session_id=session_id,
                total_cost=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                call_count=0,
                agent_breakdown={},
                first_call=None,
                last_call=None,
            )

        # Calculate totals
        total_cost = sum(e.total_cost for e in session_entries)
        total_input = sum(e.input_tokens for e in session_entries)
        total_output = sum(e.output_tokens for e in session_entries)

        # Agent breakdown
        agent_breakdown: dict[str, float] = {}
        for entry in session_entries:
            agent_breakdown[entry.agent_name] = (
                agent_breakdown.get(entry.agent_name, 0.0) + entry.total_cost
            )

        # Time range
        timestamps = [e.timestamp for e in session_entries]
        first_call = min(timestamps)
        last_call = max(timestamps)

        return SessionCostSummary(
            session_id=session_id,
            total_cost=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            call_count=len(session_entries),
            agent_breakdown=agent_breakdown,
            first_call=first_call,
            last_call=last_call,
            threshold_exceeded=total_cost > self.alert_threshold,
            threshold_value=self.alert_threshold,
        )

    def get_agent_costs(self, agent_name: str) -> list[CostEntry]:
        """
        Get all cost entries for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of cost entries
        """
        return [e for e in self._entries if e.agent_name == agent_name]

    def get_total_cost(self) -> float:
        """Get total cost across all sessions."""
        return sum(e.total_cost for e in self._entries)

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session."""
        return self._session_totals.get(session_id, 0.0)

    def check_budget(self, session_id: str, estimated_cost: float) -> tuple[bool, str]:
        """
        Check if a new operation would exceed the budget.

        Args:
            session_id: Session identifier
            estimated_cost: Estimated cost of the operation

        Returns:
            Tuple of (allowed, reason)
        """
        current = self._session_totals.get(session_id, 0.0)
        projected = current + estimated_cost

        if projected > self.alert_threshold:
            return False, (
                f"Operation would exceed cost threshold: "
                f"${current:.4f} + ${estimated_cost:.4f} = ${projected:.4f} > "
                f"${self.alert_threshold:.2f}"
            )

        return True, "Within budget"

    def clear_session(self, session_id: str) -> None:
        """
        Clear cost data for a session.

        Args:
            session_id: Session to clear
        """
        self._entries = [e for e in self._entries if e.session_id != session_id]
        if session_id in self._session_totals:
            del self._session_totals[session_id]
        logger.debug(f"Cleared cost data for session {session_id}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get tracker statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_entries": len(self._entries),
            "total_sessions": len(self._session_totals),
            "total_cost": self.get_total_cost(),
            "alert_threshold": self.alert_threshold,
            "sessions_over_threshold": sum(
                1 for t in self._session_totals.values() if t > self.alert_threshold
            ),
        }


class CostTrackingMiddleware:
    """
    Middleware that integrates cost tracking with agent execution.

    Usage:
        middleware = CostTrackingMiddleware()
        middleware.before_llm_call(agent, session_id)
        # ... make LLM call ...
        middleware.after_llm_call(agent, session_id, input_tokens, output_tokens)
    """

    _instance: CostTrackingMiddleware | None = None

    def __init__(self, tracker: CostTracker | None = None) -> None:
        """
        Initialize cost tracking middleware.

        Args:
            tracker: Cost tracker instance (uses singleton if not provided)
        """
        self.tracker = tracker or CostTracker.get_instance()
        self._pending_calls: dict[str, dict[str, Any]] = {}  # key -> call info

    @classmethod
    def get_instance(cls, tracker: CostTracker | None = None) -> CostTrackingMiddleware:
        """Get singleton instance of the middleware."""
        if cls._instance is None:
            cls._instance = cls(tracker)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def before_llm_call(
        self,
        agent: BaseAgent,
        session_id: str,
        estimated_input_tokens: int | None = None,
    ) -> tuple[bool, str]:
        """
        Called before an LLM call.

        Args:
            agent: Agent making the call
            session_id: Session identifier
            estimated_input_tokens: Estimated input tokens

        Returns:
            Tuple of (allowed, reason)
        """
        # Create tracking key
        key = f"{session_id}:{agent.name}:{datetime.utcnow().timestamp()}"
        self._pending_calls[key] = {
            "agent": agent.name,
            "session_id": session_id,
            "start_time": datetime.utcnow(),
        }

        # Check budget if we have an estimate
        if estimated_input_tokens:
            estimated_cost = (estimated_input_tokens / 1_000_000) * OPUS_4_5_INPUT_COST
            allowed, reason = self.tracker.check_budget(session_id, estimated_cost)
            if not allowed:
                logger.warning(f"LLM call blocked: {reason}")
                return False, reason

        return True, "Allowed"

    def after_llm_call(
        self,
        agent: BaseAgent,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """
        Called after an LLM call completes.

        Args:
            agent: Agent that made the call
            session_id: Session identifier
            input_tokens: Actual input tokens used
            output_tokens: Actual output tokens generated
            metadata: Optional metadata

        Returns:
            The recorded cost entry
        """
        entry = self.tracker.record_usage(
            agent=agent,
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
        )

        logger.info(
            f"LLM call cost: {agent.name} - "
            f"in:{input_tokens} out:{output_tokens} = ${entry.total_cost:.6f}"
        )

        return entry

    def get_session_summary(self, session_id: str) -> SessionCostSummary:
        """Get session cost summary."""
        return self.tracker.get_session_summary(session_id)

    def is_within_budget(self, session_id: str) -> bool:
        """Check if session is within budget."""
        summary = self.get_session_summary(session_id)
        return not summary.threshold_exceeded
