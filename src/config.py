"""
Configuration module for The Analyst.

Handles environment variables, constraints enforcement, and system-wide settings.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AutonomyLevel(str, Enum):
    """Agent autonomy levels determining human-in-loop requirements."""

    SUPERVISED = "supervised"  # Proceeds, reports actions
    ADVISORY = "advisory"  # Presents options, waits for approval


class AgentModel(str, Enum):
    """Available Claude models for agents."""

    OPUS_4_5 = "claude-opus-4-5-20251101"  # Latest frontier model


# =============================================================================
# NEVER-DO Constraints (Hard-coded, cannot be overridden)
# =============================================================================

NEVER_DO: list[str] = [
    "Modify source data files (read-only access always)",
    "Share user data with external services",
    "Make analytical decisions without showing reasoning",
    "Execute cost-incurring API calls without approval",
    "Proceed with analysis without presenting options first",
    "Skip verification steps for statistical claims",
    "Output results without confidence intervals where applicable",
]

# Keywords that trigger high-stakes confirmation
HIGH_STAKES_KEYWORDS: list[str] = [
    "delete",
    "remove",
    "drop",
    "send",
    "share",
    "export to",
    "production",
    "stakeholder",
    "executive",
    "forecast",
    "predict",
    "model",
    "large dataset",
    "batch process",
]


# =============================================================================
# Agent Configuration
# =============================================================================

AGENT_CONFIG: dict[str, dict[str, Any]] = {
    "orchestrator": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.ADVISORY,
        "description": "Route requests, coordinate agents, ensure human approval",
    },
    "retrieval": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.SUPERVISED,
        "description": "Ingest data from files and APIs",
    },
    "transform": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.SUPERVISED,
        "description": "Clean, reshape, and prepare data",
    },
    "statistical": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.ADVISORY,
        "description": "Perform statistical analysis and EDA",
    },
    "arabic_nlp": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.SUPERVISED,
        "description": "Process Arabic text: sentiment, NER, topics",
    },
    "modeling": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.ADVISORY,
        "description": "Build and evaluate predictive models",
    },
    "insights": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.ADVISORY,
        "description": "Synthesize findings into actionable insights",
    },
    "visualization": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.SUPERVISED,
        "description": "Create charts and interactive visuals",
    },
    "report": {
        "model": AgentModel.OPUS_4_5,
        "autonomy": AutonomyLevel.ADVISORY,
        "description": "Generate formatted outputs",
    },
}


# =============================================================================
# Settings
# =============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars like SENTRY_DSN
    )

    # API Keys
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    huggingface_token: str = Field(default="", description="HuggingFace token for Arabic models")

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/analyst",
        description="PostgreSQL connection string",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Thresholds (Enhanced for Opus 4.5)
    cost_alert_threshold: float = Field(
        default=5.0, description="Cost threshold for API call confirmation (USD)"
    )
    max_rows_without_confirm: int = Field(
        default=5000, description="Row count threshold requiring confirmation"
    )

    # Models
    orchestrator_model: str = Field(
        default=AgentModel.OPUS_4_5.value, description="Model for orchestrator"
    )
    analysis_model: str = Field(
        default=AgentModel.OPUS_4_5.value, description="Model for analysis agents"
    )
    utility_model: str = Field(
        default=AgentModel.OPUS_4_5.value, description="Model for utility agents"
    )

    # Paths
    data_path: Path = Field(default=Path("./data"), description="Base path for data storage")
    output_path: Path = Field(default=Path("./data/outputs"), description="Path for outputs")
    obsidian_vault_path: Path | None = Field(
        default=None, description="Obsidian vault path for notifications"
    )

    # Notifications
    enable_desktop_notifications: bool = Field(
        default=True, description="Enable desktop notifications"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @property
    def raw_data_path(self) -> Path:
        """Path to raw data directory (read-only)."""
        return self.data_path / "raw"

    @property
    def processed_data_path(self) -> Path:
        """Path to processed data directory."""
        return self.data_path / "processed"


# =============================================================================
# Constraint Enforcement
# =============================================================================


class ConstraintViolation(Exception):
    """Raised when a NEVER-DO constraint is violated."""

    pass


def check_high_stakes(text: str) -> list[str]:
    """
    Check if text contains high-stakes keywords.

    Args:
        text: Text to check for high-stakes keywords

    Returns:
        List of matched high-stakes keywords found in text
    """
    text_lower = text.lower()
    return [kw for kw in HIGH_STAKES_KEYWORDS if kw in text_lower]


def enforce_read_only(path: Path) -> None:
    """
    Enforce read-only access for source data files.

    Args:
        path: Path to validate

    Raises:
        ConstraintViolation: If attempting to modify source data
    """
    settings = get_settings()
    raw_path = settings.raw_data_path.resolve()

    if path.resolve().is_relative_to(raw_path):
        raise ConstraintViolation(
            f"NEVER-DO Violation: Cannot modify source data at {path}. "
            f"Source data is read-only. Use processed/ directory for outputs."
        )


def require_confirmation(
    action: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a confirmation request for high-stakes actions.

    Args:
        action: Description of the action requiring confirmation
        reason: Why confirmation is required
        details: Additional details about the action

    Returns:
        Confirmation request dictionary
    """
    return {
        "requires_confirmation": True,
        "action": action,
        "reason": reason,
        "details": details or {},
        "message": f"This action requires confirmation: {action}\nReason: {reason}",
    }


# =============================================================================
# Singleton Settings Instance
# =============================================================================

_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
