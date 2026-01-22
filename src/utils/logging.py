"""
Structured logging for The Analyst platform.

Provides consistent logging across all agents and modules.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import get_settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured fields."""
        # Create timestamp from record.created
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Add extra fields if present
        extras = []
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "timestamp",
                "message",
            ):
                extras.append(f"{key}={value}")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        return (
            f"{timestamp} | {record.levelname:8} | "
            f"{record.name} | {record.getMessage()}{extra_str}"
        )


def setup_logging(
    level: str | None = None,
    log_file: Path | None = None,
) -> None:
    """
    Set up logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
    """
    settings = get_settings()
    log_level = level or settings.log_level

    # Create formatter
    formatter = StructuredFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class AgentLogger:
    """Logger wrapper for agent-specific logging with context."""

    def __init__(self, agent_name: str, session_id: str | None = None) -> None:
        """Initialize the agent logger."""
        self.agent_name = agent_name
        self.session_id = session_id
        self._logger = get_logger(f"analyst.agents.{agent_name}")
        self._entries: list[dict[str, Any]] = []

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "session_id": self.session_id,
            "level": level,
            "message": message,
            **kwargs,
        }
        self._entries.append(entry)

        log_func = getattr(self._logger, level.lower())
        log_func(message, extra=kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def get_entries(self) -> list[dict[str, Any]]:
        """Get all log entries."""
        return self._entries.copy()
