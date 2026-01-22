"""Tests for structured logging utilities."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.logging import AgentLogger, StructuredFormatter, get_logger, setup_logging


class TestStructuredFormatter:
    """Test suite for StructuredFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a StructuredFormatter instance for testing."""
        return StructuredFormatter()

    @pytest.fixture
    def sample_record(self):
        """Create a sample log record for testing."""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        return record

    def test_format_basic_record(self, formatter, sample_record):
        """Test formatting a basic log record."""
        result = formatter.format(sample_record)

        assert "test.logger" in result
        assert "INFO" in result
        assert "Test message" in result

    def test_format_includes_timestamp(self, formatter, sample_record):
        """Test that formatted output includes timestamp."""
        result = formatter.format(sample_record)

        # Timestamp should be in ISO format (YYYY-MM-DDTHH:MM:SS)
        assert "T" in result  # ISO format separator
        assert "|" in result  # Field separator

    def test_format_different_log_levels(self, formatter):
        """Test formatting with different log levels."""
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level, level_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            assert level_name in result

    def test_format_with_extra_fields(self, formatter, sample_record):
        """Test formatting with extra fields attached to record."""
        sample_record.session_id = "sess-123"
        sample_record.agent_name = "statistical"

        result = formatter.format(sample_record)

        assert "session_id=sess-123" in result
        assert "agent_name=statistical" in result

    def test_format_message_with_args(self, formatter):
        """Test formatting message with arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Value is %d",
            args=(42,),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "Value is 42" in result

    def test_format_without_extra_fields(self, formatter, sample_record):
        """Test formatting without extra fields has no brackets."""
        result = formatter.format(sample_record)

        # Should not have empty brackets at end
        assert not result.endswith("[]")


class TestSetupLogging:
    """Test suite for setup_logging function."""

    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary file for log output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            yield Path(f.name)

    def test_setup_logging_default(self):
        """Test setting up logging with defaults."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "INFO"

            setup_logging()

            root_logger = logging.getLogger()
            assert root_logger.level == logging.INFO
            assert len(root_logger.handlers) > 0

    def test_setup_logging_with_level(self):
        """Test setting up logging with specific level."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "INFO"

            setup_logging(level="DEBUG")

            root_logger = logging.getLogger()
            assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, temp_log_file):
        """Test setting up logging with file output."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "INFO"

            setup_logging(log_file=temp_log_file)

            root_logger = logging.getLogger()
            # Should have console and file handlers
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert "FileHandler" in handler_types
            assert "StreamHandler" in handler_types

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates log directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "app.log"

            with patch("src.utils.logging.get_settings") as mock_settings:
                mock_settings.return_value.log_level = "INFO"

                setup_logging(log_file=log_file)

                assert log_file.parent.exists()

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears existing handlers."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "INFO"

            # Add a dummy handler
            root_logger = logging.getLogger()
            dummy_handler = logging.NullHandler()
            root_logger.addHandler(dummy_handler)

            initial_count = len(root_logger.handlers)

            setup_logging()

            # Should have replaced handlers, not added to them
            assert len(root_logger.handlers) <= initial_count

    def test_setup_logging_mutes_noisy_libraries(self):
        """Test that noisy library loggers are set to WARNING."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "DEBUG"

            setup_logging()

            assert logging.getLogger("httpx").level == logging.WARNING
            assert logging.getLogger("anthropic").level == logging.WARNING
            assert logging.getLogger("urllib3").level == logging.WARNING


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that same name returns same logger instance."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")

        assert logger1 is logger2

    def test_get_logger_different_names_return_different_instances(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")

        assert logger1 is not logger2


class TestAgentLogger:
    """Test suite for AgentLogger class."""

    @pytest.fixture
    def agent_logger(self):
        """Create an AgentLogger instance for testing."""
        return AgentLogger(agent_name="statistical", session_id="sess-123")

    def test_agent_logger_initialization(self):
        """Test AgentLogger initialization."""
        logger = AgentLogger(agent_name="test_agent", session_id="test-session")

        assert logger.agent_name == "test_agent"
        assert logger.session_id == "test-session"
        assert logger._entries == []

    def test_agent_logger_without_session_id(self):
        """Test AgentLogger initialization without session_id."""
        logger = AgentLogger(agent_name="test_agent")

        assert logger.agent_name == "test_agent"
        assert logger.session_id is None

    def test_debug_logging(self, agent_logger):
        """Test debug level logging."""
        agent_logger.debug("Debug message", extra_field="value")

        entries = agent_logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["level"] == "DEBUG"
        assert entries[0]["message"] == "Debug message"
        assert entries[0]["extra_field"] == "value"

    def test_info_logging(self, agent_logger):
        """Test info level logging."""
        agent_logger.info("Info message")

        entries = agent_logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["level"] == "INFO"
        assert entries[0]["message"] == "Info message"

    def test_warning_logging(self, agent_logger):
        """Test warning level logging."""
        agent_logger.warning("Warning message")

        entries = agent_logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["level"] == "WARNING"
        assert entries[0]["message"] == "Warning message"

    def test_error_logging(self, agent_logger):
        """Test error level logging."""
        agent_logger.error("Error message", error_code=500)

        entries = agent_logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["level"] == "ERROR"
        assert entries[0]["message"] == "Error message"
        assert entries[0]["error_code"] == 500

    def test_log_entries_include_context(self, agent_logger):
        """Test that log entries include agent context."""
        agent_logger.info("Test message")

        entries = agent_logger.get_entries()
        assert entries[0]["agent"] == "statistical"
        assert entries[0]["session_id"] == "sess-123"

    def test_log_entries_include_timestamp(self, agent_logger):
        """Test that log entries include timestamp."""
        agent_logger.info("Test message")

        entries = agent_logger.get_entries()
        assert "timestamp" in entries[0]
        # Timestamp should be ISO format
        assert "T" in entries[0]["timestamp"]

    def test_multiple_log_entries(self, agent_logger):
        """Test logging multiple messages."""
        agent_logger.debug("First message")
        agent_logger.info("Second message")
        agent_logger.warning("Third message")

        entries = agent_logger.get_entries()
        assert len(entries) == 3
        assert entries[0]["message"] == "First message"
        assert entries[1]["message"] == "Second message"
        assert entries[2]["message"] == "Third message"

    def test_get_entries_returns_copy(self, agent_logger):
        """Test that get_entries returns a copy of entries."""
        agent_logger.info("Test message")

        entries1 = agent_logger.get_entries()
        entries2 = agent_logger.get_entries()

        assert entries1 == entries2
        assert entries1 is not entries2  # Should be different objects

    def test_get_entries_modification_does_not_affect_internal(self, agent_logger):
        """Test that modifying returned entries doesn't affect internal state."""
        agent_logger.info("Original message")

        entries = agent_logger.get_entries()
        entries.append({"message": "Injected"})

        internal_entries = agent_logger.get_entries()
        assert len(internal_entries) == 1
        assert internal_entries[0]["message"] == "Original message"

    def test_log_with_multiple_extra_kwargs(self, agent_logger):
        """Test logging with multiple extra keyword arguments."""
        agent_logger.info(
            "Complex message",
            metric_name="correlation",
            value=0.85,
            confidence=0.95,
            tags=["analysis", "statistical"],
        )

        entries = agent_logger.get_entries()
        entry = entries[0]
        assert entry["metric_name"] == "correlation"
        assert entry["value"] == 0.85
        assert entry["confidence"] == 0.95
        assert entry["tags"] == ["analysis", "statistical"]

    def test_underlying_logger_is_called(self, agent_logger):
        """Test that the underlying logger is actually called."""
        with patch.object(agent_logger._logger, "info") as mock_info:
            agent_logger.info("Test message", custom_field="test")

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[0][0] == "Test message"
            assert call_args[1]["extra"]["custom_field"] == "test"

    def test_agent_logger_name_format(self):
        """Test that agent logger has correct name format."""
        logger = AgentLogger(agent_name="my_agent")

        assert logger._logger.name == "analyst.agents.my_agent"


class TestLoggingIntegration:
    """Integration tests for the logging module."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        with patch("src.utils.logging.get_settings") as mock_settings:
            mock_settings.return_value.log_level = "DEBUG"

            # Setup logging
            setup_logging(level="DEBUG")

            # Get a logger
            logger = get_logger("test.integration")

            # Create agent logger
            agent_logger = AgentLogger("test_agent", "session-001")

            # Log messages
            logger.info("Standard logger message")
            agent_logger.info("Agent logger message", operation="test")

            # Verify agent logger entries
            entries = agent_logger.get_entries()
            assert len(entries) == 1
            assert entries[0]["operation"] == "test"

    def test_formatter_with_real_handler(self):
        """Test StructuredFormatter with a real handler."""
        import io

        # Create a stream handler with our formatter
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        # Create and configure a test logger
        test_logger = logging.getLogger("test.formatter.real")
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        # Log a message
        test_logger.info("Test message")

        # Check output
        output = stream.getvalue()
        assert "INFO" in output
        assert "test.formatter.real" in output
        assert "Test message" in output
