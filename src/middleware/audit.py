"""
Comprehensive Audit Logging Middleware.

Provides structured JSON logging of all agent decision points with:
- Event categorization
- Data sanitization (redacts secrets, truncates large payloads)
- Memory buffer for recent events
- File persistence

BORIS Compliance: Complete audit trail for transparency and reproducibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"

    # LLM interactions
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"

    # Human-in-loop
    OPTION_PRESENTED = "option_presented"
    OPTION_SELECTED = "option_selected"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"

    # High-stakes detection
    HIGH_STAKES_DETECTED = "high_stakes_detected"

    # Cost tracking
    COST_RECORDED = "cost_recorded"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Data operations
    DATA_LOADED = "data_loaded"
    DATA_TRANSFORMED = "data_transformed"
    DATA_EXPORTED = "data_exported"

    # System events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONFIG_CHANGED = "config_changed"


# Patterns for sensitive data that should be redacted
SENSITIVE_PATTERNS = [
    (
        re.compile(r"(?i)(api[_-]?key|apikey)[=:\s]+['\"]?([a-zA-Z0-9_-]{20,})['\"]?"),
        r"\1=***REDACTED***",
    ),
    (re.compile(r"(?i)(password|passwd|pwd)[=:\s]+['\"]?([^\s'\"]+)['\"]?"), r"\1=***REDACTED***"),
    (
        re.compile(r"(?i)(secret|token)[=:\s]+['\"]?([a-zA-Z0-9_-]{10,})['\"]?"),
        r"\1=***REDACTED***",
    ),
    (
        re.compile(r"(?i)(authorization)[=:\s]+['\"]?(bearer\s+)?([a-zA-Z0-9_.-]+)['\"]?"),
        r"\1=***REDACTED***",
    ),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "***API_KEY_REDACTED***"),
    (re.compile(r"hf_[a-zA-Z0-9]{20,}"), "***HF_TOKEN_REDACTED***"),
]

# Maximum lengths for truncation
MAX_MESSAGE_LENGTH = 1000
MAX_DATA_LENGTH = 5000


@dataclass
class AuditEvent:
    """A single audit event."""

    event_type: AuditEventType
    timestamp: datetime
    agent_name: str | None
    session_id: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


def sanitize_text(text: str) -> str:
    """
    Sanitize text by redacting sensitive information.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    result = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def truncate_text(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated text with indicator if truncated
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, total length: {len(text)}]"


def sanitize_data(data: Any, max_length: int = MAX_DATA_LENGTH) -> Any:
    """
    Recursively sanitize data structure.

    Args:
        data: Data to sanitize
        max_length: Maximum string length

    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        return truncate_text(sanitize_text(data), max_length)
    elif isinstance(data, dict):
        return {k: sanitize_data(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 100:
            return sanitize_data(data[:100], max_length) + [f"... and {len(data) - 100} more items"]
        return [sanitize_data(item, max_length) for item in data]
    elif isinstance(data, (int, float, bool, type(None))):
        return data
    else:
        return truncate_text(str(data), max_length)


class AuditLogger:
    """
    Structured JSON audit logger with file persistence and memory buffer.

    Usage:
        logger = AuditLogger(log_file="audit.jsonl")
        logger.log_event(AuditEventType.AGENT_START, "my_agent", session_id, {...})
    """

    _instance: AuditLogger | None = None

    def __init__(
        self,
        log_file: Path | str | None = None,
        buffer_size: int = 1000,
        enable_file_logging: bool = True,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file (JSONL format)
            buffer_size: Maximum events to keep in memory buffer
            enable_file_logging: Whether to write to file
        """
        self.log_file = Path(log_file) if log_file else Path("./data/audit/audit.jsonl")
        self.buffer_size = buffer_size
        self.enable_file_logging = enable_file_logging
        self._buffer: list[AuditEvent] = []
        self._event_counts: dict[str, int] = {}

        # Ensure log directory exists
        if self.enable_file_logging:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(
        cls,
        log_file: Path | str | None = None,
        buffer_size: int = 1000,
    ) -> AuditLogger:
        """Get singleton instance of the logger."""
        if cls._instance is None:
            cls._instance = cls(log_file, buffer_size)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def log_event(
        self,
        event_type: AuditEventType,
        agent_name: str | None,
        session_id: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            agent_name: Name of the agent (if applicable)
            session_id: Session identifier
            data: Event data
            metadata: Optional metadata

        Returns:
            The created audit event
        """
        # Sanitize data
        sanitized_data = sanitize_data(data)
        sanitized_metadata = sanitize_data(metadata or {})

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            agent_name=agent_name,
            session_id=session_id,
            data=sanitized_data,
            metadata=sanitized_metadata,
        )

        # Add to buffer
        self._buffer.append(event)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)

        # Update counts
        self._event_counts[event_type.value] = self._event_counts.get(event_type.value, 0) + 1

        # Write to file
        if self.enable_file_logging:
            self._write_to_file(event)

        # Log to standard logger
        logger.debug(f"Audit: {event_type.value} - {agent_name or 'system'} - {session_id}")

        return event

    def _write_to_file(self, event: AuditEvent) -> None:
        """Write event to log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def get_recent_events(
        self,
        count: int = 100,
        event_type: AuditEventType | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[AuditEvent]:
        """
        Get recent events from the buffer.

        Args:
            count: Maximum number of events to return
            event_type: Filter by event type
            session_id: Filter by session
            agent_name: Filter by agent

        Returns:
            List of matching events (newest first)
        """
        events = self._buffer.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        if agent_name:
            events = [e for e in events if e.agent_name == agent_name]

        return list(reversed(events[-count:]))

    def get_session_events(self, session_id: str) -> list[AuditEvent]:
        """
        Get all events for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of events for the session
        """
        return [e for e in self._buffer if e.session_id == session_id]

    def get_event_counts(self) -> dict[str, int]:
        """Get counts of events by type."""
        return self._event_counts.copy()

    def get_stats(self) -> dict[str, Any]:
        """
        Get logger statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "buffer_size": len(self._buffer),
            "max_buffer_size": self.buffer_size,
            "event_counts": self._event_counts.copy(),
            "total_events": sum(self._event_counts.values()),
            "unique_sessions": len(set(e.session_id for e in self._buffer)),
            "log_file": str(self.log_file),
            "file_logging_enabled": self.enable_file_logging,
        }

    def clear_buffer(self) -> None:
        """Clear the memory buffer."""
        self._buffer.clear()

    def clear_session(self, session_id: str) -> None:
        """Clear events for a specific session from buffer."""
        self._buffer = [e for e in self._buffer if e.session_id != session_id]


class AuditMiddleware:
    """
    Middleware that integrates audit logging with agent execution.

    Usage:
        middleware = AuditMiddleware()
        middleware.log_agent_start(agent, session_id, kwargs)
        # ... agent executes ...
        middleware.log_agent_complete(agent, session_id, result)
    """

    _instance: AuditMiddleware | None = None

    def __init__(self, audit_logger: AuditLogger | None = None) -> None:
        """
        Initialize audit middleware.

        Args:
            audit_logger: Audit logger instance (uses singleton if not provided)
        """
        self.audit_logger = audit_logger or AuditLogger.get_instance()

    @classmethod
    def get_instance(cls, audit_logger: AuditLogger | None = None) -> AuditMiddleware:
        """Get singleton instance of the middleware."""
        if cls._instance is None:
            cls._instance = cls(audit_logger)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def log_agent_start(
        self,
        agent: BaseAgent,
        session_id: str,
        kwargs: dict[str, Any],
    ) -> AuditEvent:
        """Log agent start event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.AGENT_START,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "autonomy_level": agent.autonomy.value,
                "model": agent.model,
                "input_args": kwargs,
            },
            metadata={"description": agent.description},
        )

    def log_agent_complete(
        self,
        agent: BaseAgent,
        session_id: str,
        success: bool,
        execution_time_ms: float,
        result_summary: str | None = None,
    ) -> AuditEvent:
        """Log agent completion event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.AGENT_COMPLETE,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "success": success,
                "execution_time_ms": execution_time_ms,
                "result_summary": result_summary,
            },
        )

    def log_agent_error(
        self,
        agent: BaseAgent,
        session_id: str,
        error: str,
        execution_time_ms: float,
    ) -> AuditEvent:
        """Log agent error event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.AGENT_ERROR,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "error": error,
                "execution_time_ms": execution_time_ms,
            },
        )

    def log_llm_call(
        self,
        agent: BaseAgent,
        session_id: str,
        message_count: int,
        system_prompt_preview: str,
    ) -> AuditEvent:
        """Log LLM call event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.LLM_CALL,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "model": agent.model,
                "message_count": message_count,
                "system_prompt_preview": (
                    system_prompt_preview[:200] if system_prompt_preview else None
                ),
            },
        )

    def log_llm_response(
        self,
        agent: BaseAgent,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        response_preview: str,
    ) -> AuditEvent:
        """Log LLM response event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.LLM_RESPONSE,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_preview": response_preview[:500] if response_preview else None,
            },
        )

    def log_option_presented(
        self,
        agent: BaseAgent,
        session_id: str,
        options: list[dict[str, Any]],
        context_message: str,
    ) -> AuditEvent:
        """Log option presentation event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.OPTION_PRESENTED,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "option_count": len(options),
                "options": [{"id": o.get("id"), "title": o.get("title")} for o in options],
                "context_message": context_message,
            },
        )

    def log_option_selected(
        self,
        agent: BaseAgent,
        session_id: str,
        selected_option_id: str,
    ) -> AuditEvent:
        """Log option selection event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.OPTION_SELECTED,
            agent_name=agent.name,
            session_id=session_id,
            data={"selected_option_id": selected_option_id},
        )

    def log_approval_requested(
        self,
        agent: BaseAgent,
        session_id: str,
        action: str,
        reason: str,
    ) -> AuditEvent:
        """Log approval request event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.APPROVAL_REQUESTED,
            agent_name=agent.name,
            session_id=session_id,
            data={"action": action, "reason": reason},
        )

    def log_high_stakes_detected(
        self,
        agent: BaseAgent,
        session_id: str,
        keywords: list[str],
        input_preview: str,
    ) -> AuditEvent:
        """Log high-stakes detection event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.HIGH_STAKES_DETECTED,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "keywords": keywords,
                "input_preview": input_preview[:200],
            },
        )

    def log_cost_recorded(
        self,
        agent: BaseAgent,
        session_id: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
    ) -> AuditEvent:
        """Log cost recording event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.COST_RECORDED,
            agent_name=agent.name,
            session_id=session_id,
            data={
                "cost_usd": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    def log_budget_exceeded(
        self,
        session_id: str,
        current_cost: float,
        threshold: float,
    ) -> AuditEvent:
        """Log budget exceeded event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.BUDGET_EXCEEDED,
            agent_name=None,
            session_id=session_id,
            data={
                "current_cost_usd": current_cost,
                "threshold_usd": threshold,
                "exceeded_by_usd": current_cost - threshold,
            },
        )

    def log_session_start(self, session_id: str, user_id: str | None = None) -> AuditEvent:
        """Log session start event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.SESSION_START,
            agent_name=None,
            session_id=session_id,
            data={"user_id": user_id},
        )

    def log_session_end(
        self,
        session_id: str,
        total_cost: float,
        agent_count: int,
    ) -> AuditEvent:
        """Log session end event."""
        return self.audit_logger.log_event(
            event_type=AuditEventType.SESSION_END,
            agent_name=None,
            session_id=session_id,
            data={
                "total_cost_usd": total_cost,
                "agent_count": agent_count,
            },
        )

    def get_session_audit_trail(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get complete audit trail for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of audit events as dictionaries
        """
        events = self.audit_logger.get_session_events(session_id)
        return [e.to_dict() for e in events]
