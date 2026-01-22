"""
Notification system for The Analyst platform.

Supports terminal, desktop, and Obsidian notifications.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_settings


class NotificationType(str, Enum):
    """Types of notifications (severity levels)."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class NotificationChannel(str, Enum):
    """Available notification channels."""

    TERMINAL = "terminal"
    DESKTOP = "desktop"
    OBSIDIAN = "obsidian"


@dataclass
class NotificationConfig:
    """Configuration for the notification system."""

    channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.TERMINAL]
    )
    obsidian_vault_path: Path | None = None
    enable_desktop: bool = True
    obsidian_folder: str = "The Analyst"

    @classmethod
    def from_env(cls) -> NotificationConfig:
        """Create config from environment variables."""
        channels_str = os.environ.get("NOTIFICATION_CHANNELS", "terminal")
        channel_names = [c.strip().lower() for c in channels_str.split(",")]

        channels = []
        for name in channel_names:
            if name == "terminal":
                channels.append(NotificationChannel.TERMINAL)
            elif name == "desktop":
                channels.append(NotificationChannel.DESKTOP)
            elif name == "obsidian":
                channels.append(NotificationChannel.OBSIDIAN)

        # Default to terminal if nothing valid found
        if not channels:
            channels = [NotificationChannel.TERMINAL]

        # Get Obsidian vault path
        vault_path_str = os.environ.get("OBSIDIAN_VAULT_PATH")
        obsidian_vault_path = None
        if vault_path_str:
            obsidian_vault_path = Path(vault_path_str)
        else:
            # Default to ~/Documents/Obsidian if it exists
            default_path = Path.home() / "Documents" / "Obsidian"
            if default_path.exists():
                obsidian_vault_path = default_path

        return cls(
            channels=channels,
            obsidian_vault_path=obsidian_vault_path,
            enable_desktop=os.environ.get("ENABLE_DESKTOP_NOTIFICATIONS", "true").lower() == "true",
        )


@dataclass
class Notification:
    """A notification message."""

    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize defaults."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class Notifier:
    """
    Handles notifications across multiple channels.

    Channels:
    - Terminal: Rich console output
    - Desktop: Native OS notifications
    - Obsidian: Note creation in vault
    """

    def __init__(self, config: NotificationConfig | None = None) -> None:
        """
        Initialize the notifier.

        Args:
            config: Optional notification configuration. If not provided,
                   will load from environment/settings.
        """
        self.settings = get_settings()
        self.config = config or NotificationConfig.from_env()

        # Override config from settings if available
        if self.settings.obsidian_vault_path:
            self.config.obsidian_vault_path = self.settings.obsidian_vault_path
        if hasattr(self.settings, "enable_desktop_notifications"):
            self.config.enable_desktop = self.settings.enable_desktop_notifications

    def notify(
        self,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        channels: Sequence[str | NotificationChannel] | None = None,
        **metadata: Any,
    ) -> None:
        """
        Send a notification to specified channels.

        Args:
            title: Notification title
            message: Notification message
            type: Type of notification
            channels: List of channels ("terminal", "desktop", "obsidian") or NotificationChannel enums
            **metadata: Additional metadata
        """
        notification = Notification(
            title=title,
            message=message,
            type=type,
            metadata=metadata,
        )

        # Default to configured channels
        if channels is None:
            target_channels = self.config.channels.copy()
        else:
            # Normalize to NotificationChannel enums
            target_channels = []
            for ch in channels:
                if isinstance(ch, NotificationChannel):
                    target_channels.append(ch)
                elif isinstance(ch, str):
                    try:
                        target_channels.append(NotificationChannel(ch.lower()))
                    except ValueError:
                        continue

        for channel in target_channels:
            try:
                if channel == NotificationChannel.TERMINAL:
                    self._notify_terminal(notification)
                elif channel == NotificationChannel.DESKTOP:
                    self._notify_desktop(notification)
                elif channel == NotificationChannel.OBSIDIAN:
                    self._notify_obsidian(notification)
            except Exception as e:
                # Log but don't fail on notification errors
                print(f"Notification error ({channel.value}): {e}", file=sys.stderr)

    def notify_all(
        self,
        title: str,
        message: str,
        type: NotificationType = NotificationType.INFO,
        **metadata: Any,
    ) -> None:
        """
        Send a notification to all available channels.

        Args:
            title: Notification title
            message: Notification message
            type: Type of notification
            **metadata: Additional metadata
        """
        all_channels = [NotificationChannel.TERMINAL]

        # Add desktop if enabled and available
        if self.config.enable_desktop:
            all_channels.append(NotificationChannel.DESKTOP)

        # Add Obsidian if vault path is configured
        if self.config.obsidian_vault_path:
            all_channels.append(NotificationChannel.OBSIDIAN)

        self.notify(title, message, type, channels=all_channels, **metadata)

    def _notify_terminal(self, notification: Notification) -> None:
        """Send terminal notification using rich formatting."""
        # Color mapping
        colors = {
            NotificationType.INFO: "\033[94m",  # Blue
            NotificationType.SUCCESS: "\033[92m",  # Green
            NotificationType.WARNING: "\033[93m",  # Yellow
            NotificationType.ERROR: "\033[91m",  # Red
        }
        reset = "\033[0m"

        # Icons
        icons = {
            NotificationType.INFO: "ℹ️",
            NotificationType.SUCCESS: "✅",
            NotificationType.WARNING: "⚠️",
            NotificationType.ERROR: "❌",
        }

        color = colors.get(notification.type, "")
        icon = icons.get(notification.type, "")

        print(f"\n{color}{'─' * 60}{reset}")
        print(f"{icon} {color}{notification.title}{reset}")
        print(f"{notification.message}")
        print(f"{color}{'─' * 60}{reset}\n")

    def _notify_desktop(self, notification: Notification) -> None:
        """Send desktop notification."""
        if sys.platform == "darwin":
            # macOS
            script = f"""
            display notification "{notification.message}" with title "{notification.title}"
            """
            subprocess.run(["osascript", "-e", script], capture_output=True)

        elif sys.platform == "linux":
            # Linux (requires notify-send)
            subprocess.run(
                ["notify-send", notification.title, notification.message],
                capture_output=True,
            )

        elif sys.platform == "win32":
            # Windows (requires win10toast or similar)
            try:
                from win10toast import ToastNotifier

                toaster = ToastNotifier()
                toaster.show_toast(
                    notification.title,
                    notification.message,
                    duration=5,
                    threaded=True,
                )
            except ImportError:
                pass  # win10toast not installed

    def _notify_obsidian(self, notification: Notification) -> None:
        """Create notification note in Obsidian vault."""
        vault_path = self.config.obsidian_vault_path
        if not vault_path:
            return

        # Use configured folder or default
        folder_name = self.config.obsidian_folder or "The Analyst"
        notifications_dir = vault_path / folder_name / "Notifications"
        notifications_dir.mkdir(parents=True, exist_ok=True)

        # Create note filename
        ts = notification.timestamp or datetime.utcnow()
        timestamp_str = ts.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp_str}_{notification.type.value}.md"
        filepath = notifications_dir / filename

        # Build metadata tags from notification metadata
        tags = ["notification", notification.type.value, "the-analyst"]
        if notification.metadata:
            if "analysis_type" in notification.metadata:
                tags.append(notification.metadata["analysis_type"].replace(" ", "-").lower())

        # Create note content with YAML frontmatter
        content = f"""---
type: notification
notification_type: {notification.type.value}
timestamp: {ts.isoformat()}
tags: [{", ".join(tags)}]
---

# {notification.title}

{notification.message}

---
*Generated by The Analyst at {ts.strftime("%Y-%m-%d %H:%M:%S")}*
"""

        filepath.write_text(content)

    def notify_analysis_complete(
        self,
        analysis_type: str,
        summary: str,
        output_path: str | None = None,
    ) -> None:
        """
        Send notification for completed analysis.

        Args:
            analysis_type: Type of analysis completed
            summary: Brief summary of findings
            output_path: Path to output file (if any)
        """
        message = f"{summary}"
        if output_path:
            message += f"\n\nOutput saved to: {output_path}"

        self.notify(
            title=f"Analysis Complete: {analysis_type}",
            message=message,
            type=NotificationType.SUCCESS,
            analysis_type=analysis_type,
            output_path=output_path,
        )

    def notify_approval_required(
        self,
        action: str,
        reason: str,
    ) -> None:
        """
        Send notification for approval request.

        Args:
            action: Action requiring approval
            reason: Reason approval is needed
        """
        self.notify(
            title="Approval Required",
            message=f"Action: {action}\nReason: {reason}",
            type=NotificationType.WARNING,
            action=action,
            reason=reason,
        )

    def notify_error(
        self,
        error: str,
        context: str | None = None,
    ) -> None:
        """
        Send error notification.

        Args:
            error: Error message
            context: Additional context
        """
        message = error
        if context:
            message = f"{context}\n\n{error}"

        self.notify(
            title="Error",
            message=message,
            type=NotificationType.ERROR,
            error=error,
            context=context,
        )
