"""Tests for the notification system."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.notifications import (
    Notification,
    NotificationChannel,
    NotificationConfig,
    NotificationType,
    Notifier,
)


@pytest.fixture
def mock_settings():
    """Mock settings to prevent override of test config values."""
    with patch("src.utils.notifications.get_settings") as mock:
        mock_settings_instance = MagicMock()
        mock_settings_instance.obsidian_vault_path = None
        mock.return_value = mock_settings_instance
        yield mock


class TestNotificationType:
    """Tests for NotificationType enum."""

    def test_notification_types_exist(self):
        """Test that all notification types are defined."""
        assert NotificationType.INFO == "info"
        assert NotificationType.SUCCESS == "success"
        assert NotificationType.WARNING == "warning"
        assert NotificationType.ERROR == "error"

    def test_notification_type_values(self):
        """Test notification type string values."""
        assert NotificationType.INFO.value == "info"
        assert NotificationType.SUCCESS.value == "success"
        assert NotificationType.WARNING.value == "warning"
        assert NotificationType.ERROR.value == "error"


class TestNotificationChannel:
    """Tests for NotificationChannel enum."""

    def test_channels_exist(self):
        """Test that all channels are defined."""
        assert NotificationChannel.TERMINAL == "terminal"
        assert NotificationChannel.DESKTOP == "desktop"
        assert NotificationChannel.OBSIDIAN == "obsidian"

    def test_channel_values(self):
        """Test channel string values."""
        assert NotificationChannel.TERMINAL.value == "terminal"
        assert NotificationChannel.DESKTOP.value == "desktop"
        assert NotificationChannel.OBSIDIAN.value == "obsidian"


class TestNotificationConfig:
    """Tests for NotificationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NotificationConfig()
        assert config.channels == [NotificationChannel.TERMINAL]
        assert config.obsidian_vault_path is None
        assert config.enable_desktop is True
        assert config.obsidian_folder == "The Analyst"

    def test_custom_config(self):
        """Test custom configuration."""
        config = NotificationConfig(
            channels=[NotificationChannel.TERMINAL, NotificationChannel.DESKTOP],
            obsidian_vault_path=Path("/tmp/vault"),
            enable_desktop=False,
            obsidian_folder="Custom Folder",
        )
        assert len(config.channels) == 2
        assert config.obsidian_vault_path == Path("/tmp/vault")
        assert config.enable_desktop is False
        assert config.obsidian_folder == "Custom Folder"

    def test_from_env_default(self):
        """Test config from environment with defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear relevant env vars
            os.environ.pop("NOTIFICATION_CHANNELS", None)
            os.environ.pop("OBSIDIAN_VAULT_PATH", None)
            os.environ.pop("ENABLE_DESKTOP_NOTIFICATIONS", None)

            config = NotificationConfig.from_env()
            assert NotificationChannel.TERMINAL in config.channels

    def test_from_env_all_channels(self):
        """Test config from environment with all channels."""
        with patch.dict(os.environ, {"NOTIFICATION_CHANNELS": "terminal,desktop,obsidian"}):
            config = NotificationConfig.from_env()
            assert NotificationChannel.TERMINAL in config.channels
            assert NotificationChannel.DESKTOP in config.channels
            assert NotificationChannel.OBSIDIAN in config.channels

    def test_from_env_with_vault_path(self):
        """Test config from environment with Obsidian vault path."""
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": "/custom/vault"}):
            config = NotificationConfig.from_env()
            assert config.obsidian_vault_path == Path("/custom/vault")

    def test_from_env_disable_desktop(self):
        """Test config from environment with desktop disabled."""
        with patch.dict(os.environ, {"ENABLE_DESKTOP_NOTIFICATIONS": "false"}):
            config = NotificationConfig.from_env()
            assert config.enable_desktop is False


class TestNotification:
    """Tests for Notification dataclass."""

    def test_notification_creation(self):
        """Test creating a notification."""
        notif = Notification(
            title="Test Title",
            message="Test Message",
        )
        assert notif.title == "Test Title"
        assert notif.message == "Test Message"
        assert notif.type == NotificationType.INFO
        assert notif.timestamp is not None
        assert notif.metadata == {}

    def test_notification_with_type(self):
        """Test notification with custom type."""
        notif = Notification(
            title="Error",
            message="Something went wrong",
            type=NotificationType.ERROR,
        )
        assert notif.type == NotificationType.ERROR

    def test_notification_with_metadata(self):
        """Test notification with metadata."""
        notif = Notification(
            title="Analysis Complete",
            message="Done",
            metadata={"analysis_type": "statistical"},
        )
        assert notif.metadata["analysis_type"] == "statistical"

    def test_notification_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        before = datetime.utcnow()
        notif = Notification(title="Test", message="Test")
        after = datetime.utcnow()

        assert before <= notif.timestamp <= after


class TestNotifier:
    """Tests for Notifier class."""

    @pytest.fixture
    def notifier(self):
        """Create a notifier with terminal-only config."""
        config = NotificationConfig(
            channels=[NotificationChannel.TERMINAL],
            enable_desktop=False,
        )
        return Notifier(config=config)

    @pytest.fixture
    def temp_vault(self, tmp_path):
        """Create a temporary Obsidian vault."""
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        return vault_path

    def test_notifier_initialization(self, notifier):
        """Test notifier initialization."""
        assert notifier.config is not None
        assert notifier.settings is not None

    def test_notify_terminal(self, notifier, capsys):
        """Test terminal notification."""
        notifier.notify(
            title="Test Notification",
            message="This is a test",
            type=NotificationType.INFO,
            channels=[NotificationChannel.TERMINAL],
        )

        captured = capsys.readouterr()
        assert "Test Notification" in captured.out
        assert "This is a test" in captured.out

    def test_notify_terminal_success(self, notifier, capsys):
        """Test terminal notification with success type."""
        notifier.notify(
            title="Success",
            message="Operation completed",
            type=NotificationType.SUCCESS,
            channels=[NotificationChannel.TERMINAL],
        )

        captured = capsys.readouterr()
        assert "Success" in captured.out

    def test_notify_terminal_warning(self, notifier, capsys):
        """Test terminal notification with warning type."""
        notifier.notify(
            title="Warning",
            message="Please check",
            type=NotificationType.WARNING,
            channels=[NotificationChannel.TERMINAL],
        )

        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_notify_terminal_error(self, notifier, capsys):
        """Test terminal notification with error type."""
        notifier.notify(
            title="Error",
            message="Something failed",
            type=NotificationType.ERROR,
            channels=[NotificationChannel.TERMINAL],
        )

        captured = capsys.readouterr()
        assert "Error" in captured.out

    def test_notify_obsidian(self, temp_vault, mock_settings):
        """Test Obsidian notification creates a note."""
        config = NotificationConfig(
            channels=[NotificationChannel.OBSIDIAN],
            obsidian_vault_path=temp_vault,
            enable_desktop=False,
        )
        notifier = Notifier(config=config)

        notifier.notify(
            title="Test Analysis",
            message="Analysis complete",
            type=NotificationType.SUCCESS,
            channels=[NotificationChannel.OBSIDIAN],
        )

        # Check that a note was created
        notes_dir = temp_vault / "The Analyst" / "Notifications"
        assert notes_dir.exists()

        notes = list(notes_dir.glob("*.md"))
        assert len(notes) == 1

        content = notes[0].read_text()
        assert "Test Analysis" in content
        assert "Analysis complete" in content
        assert "notification_type: success" in content

    def test_notify_obsidian_with_metadata(self, temp_vault, mock_settings):
        """Test Obsidian notification includes metadata tags."""
        config = NotificationConfig(
            channels=[NotificationChannel.OBSIDIAN],
            obsidian_vault_path=temp_vault,
        )
        notifier = Notifier(config=config)

        notifier.notify(
            title="Statistical Analysis",
            message="Completed",
            type=NotificationType.SUCCESS,
            channels=[NotificationChannel.OBSIDIAN],
            analysis_type="statistical",
        )

        notes_dir = temp_vault / "The Analyst" / "Notifications"
        notes = list(notes_dir.glob("*.md"))
        content = notes[0].read_text()

        assert "statistical" in content

    @patch("subprocess.run")
    def test_notify_desktop_macos(self, mock_run, notifier):
        """Test desktop notification on macOS."""
        with patch("sys.platform", "darwin"):
            config = NotificationConfig(
                channels=[NotificationChannel.DESKTOP],
                enable_desktop=True,
            )
            notifier = Notifier(config=config)

            notifier.notify(
                title="Test",
                message="Desktop test",
                channels=[NotificationChannel.DESKTOP],
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "osascript"

    @patch("subprocess.run")
    def test_notify_desktop_linux(self, mock_run):
        """Test desktop notification on Linux."""
        with patch("sys.platform", "linux"):
            config = NotificationConfig(
                channels=[NotificationChannel.DESKTOP],
                enable_desktop=True,
            )
            notifier = Notifier(config=config)

            notifier.notify(
                title="Test",
                message="Desktop test",
                channels=[NotificationChannel.DESKTOP],
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "notify-send"

    def test_notify_all(self, temp_vault, capsys, mock_settings):
        """Test notify_all sends to all available channels."""
        config = NotificationConfig(
            channels=[NotificationChannel.TERMINAL],
            obsidian_vault_path=temp_vault,
            enable_desktop=False,
        )
        notifier = Notifier(config=config)

        notifier.notify_all(
            title="Broadcast",
            message="To all channels",
            type=NotificationType.INFO,
        )

        # Check terminal output
        captured = capsys.readouterr()
        assert "Broadcast" in captured.out

        # Check Obsidian note
        notes_dir = temp_vault / "The Analyst" / "Notifications"
        notes = list(notes_dir.glob("*.md"))
        assert len(notes) == 1

    def test_notify_analysis_complete(self, notifier, capsys):
        """Test notify_analysis_complete helper."""
        notifier.notify_analysis_complete(
            analysis_type="Statistical",
            summary="Found 5 significant correlations",
            output_path="/path/to/output.pdf",
        )

        captured = capsys.readouterr()
        assert "Analysis Complete" in captured.out
        assert "Statistical" in captured.out

    def test_notify_approval_required(self, notifier, capsys):
        """Test notify_approval_required helper."""
        notifier.notify_approval_required(
            action="Delete data",
            reason="This operation is irreversible",
        )

        captured = capsys.readouterr()
        assert "Approval Required" in captured.out
        assert "Delete data" in captured.out

    def test_notify_error(self, notifier, capsys):
        """Test notify_error helper."""
        notifier.notify_error(
            error="File not found",
            context="Loading data from source.csv",
        )

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "File not found" in captured.out

    def test_notify_with_string_channels(self, notifier, capsys):
        """Test that string channel names are accepted."""
        notifier.notify(
            title="Test",
            message="Using string channels",
            channels=["terminal"],
        )

        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_notify_invalid_channel_ignored(self, notifier, capsys):
        """Test that invalid channel names are ignored."""
        notifier.notify(
            title="Test",
            message="Invalid channel",
            channels=["invalid_channel", "terminal"],
        )

        captured = capsys.readouterr()
        assert "Test" in captured.out

    def test_notify_exception_handling(self, temp_vault):
        """Test that notification errors don't crash the notifier."""
        config = NotificationConfig(
            channels=[NotificationChannel.OBSIDIAN],
            obsidian_vault_path=Path("/nonexistent/path/that/will/fail"),
        )
        notifier = Notifier(config=config)

        # This should not raise an exception
        notifier.notify(
            title="Test",
            message="Should handle error gracefully",
            channels=[NotificationChannel.OBSIDIAN],
        )
