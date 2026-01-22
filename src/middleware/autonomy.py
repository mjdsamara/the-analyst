"""
Autonomy Level Enforcement Middleware.

Validates that agents operate within their configured autonomy levels:
- ADVISORY agents must present options before executing
- SUPERVISED agents need approval for restricted tools

BORIS Compliance: Runtime enforcement of human-in-loop requirements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.config import AutonomyLevel

if TYPE_CHECKING:
    from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class RestrictedToolCategory(str, Enum):
    """Categories of tools that require approval for supervised agents."""

    EXTERNAL_API = "external_api"  # Calls to external APIs
    DATA_MODIFICATION = "data_modification"  # Any data writes
    COST_INCURRING = "cost_incurring"  # Operations with API costs
    HIGH_STAKES = "high_stakes"  # Operations matching high-stakes keywords


@dataclass
class AutonomyConfig:
    """Configuration for autonomy enforcement."""

    # Tool categories requiring approval for SUPERVISED autonomy
    restricted_categories: set[RestrictedToolCategory] = field(
        default_factory=lambda: {
            RestrictedToolCategory.EXTERNAL_API,
            RestrictedToolCategory.DATA_MODIFICATION,
            RestrictedToolCategory.HIGH_STAKES,
        }
    )

    # Whether to strictly enforce advisory mode (options must be presented)
    strict_advisory_mode: bool = True

    # Whether to block execution on violations or just warn
    block_on_violation: bool = False

    # List of tool names considered external APIs
    external_api_tools: set[str] = field(
        default_factory=lambda: {
            "http_request",
            "api_call",
            "external_fetch",
            "webhook",
        }
    )

    # List of tool names considered data modification
    data_modification_tools: set[str] = field(
        default_factory=lambda: {
            "write_file",
            "delete_file",
            "update_database",
            "insert_database",
            "delete_database",
        }
    )


class AutonomyViolation(Exception):
    """Raised when an autonomy constraint is violated."""

    def __init__(self, message: str, agent_name: str, autonomy_level: AutonomyLevel) -> None:
        self.agent_name = agent_name
        self.autonomy_level = autonomy_level
        super().__init__(f"[{agent_name}] Autonomy violation ({autonomy_level.value}): {message}")


@dataclass
class AutonomyCheckResult:
    """Result of an autonomy check."""

    allowed: bool
    reason: str
    requires_options: bool = False
    requires_approval: bool = False
    violation_type: str | None = None


class AutonomyMiddleware:
    """
    Middleware for enforcing autonomy level constraints.

    Usage:
        middleware = AutonomyMiddleware()
        result = middleware.check_pre_execute(agent)
        if not result.allowed:
            raise AutonomyViolation(result.reason)
    """

    _instance: AutonomyMiddleware | None = None

    def __init__(self, config: AutonomyConfig | None = None) -> None:
        """
        Initialize autonomy middleware.

        Args:
            config: Configuration for autonomy enforcement
        """
        self.config = config or AutonomyConfig()
        self._options_presented: dict[str, bool] = {}  # session_id -> presented
        self._agent_states: dict[str, dict[str, Any]] = {}  # agent_name -> state

    @classmethod
    def get_instance(cls, config: AutonomyConfig | None = None) -> AutonomyMiddleware:
        """Get singleton instance of the middleware."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the middleware.

        Args:
            agent: Agent to register
        """
        self._agent_states[agent.name] = {
            "autonomy": agent.autonomy,
            "options_presented": False,
            "execution_count": 0,
        }
        logger.debug(f"Registered agent {agent.name} with autonomy {agent.autonomy.value}")

    def mark_options_presented(self, agent_name: str, session_id: str) -> None:
        """
        Mark that options have been presented for an advisory agent.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier
        """
        key = f"{session_id}:{agent_name}"
        self._options_presented[key] = True
        if agent_name in self._agent_states:
            self._agent_states[agent_name]["options_presented"] = True
        logger.debug(f"Options marked as presented for {agent_name} in session {session_id}")

    def check_options_presented(self, agent_name: str, session_id: str) -> bool:
        """
        Check if options have been presented for an advisory agent.

        Args:
            agent_name: Name of the agent
            session_id: Session identifier

        Returns:
            True if options were presented
        """
        key = f"{session_id}:{agent_name}"
        return self._options_presented.get(key, False)

    def clear_session(self, session_id: str) -> None:
        """
        Clear autonomy state for a session.

        Args:
            session_id: Session to clear
        """
        keys_to_remove = [k for k in self._options_presented if k.startswith(f"{session_id}:")]
        for key in keys_to_remove:
            del self._options_presented[key]
        logger.debug(f"Cleared autonomy state for session {session_id}")

    def check_pre_execute(
        self,
        agent: BaseAgent,
        session_id: str,
        tool_name: str | None = None,
    ) -> AutonomyCheckResult:
        """
        Check autonomy constraints before agent execution.

        Args:
            agent: Agent about to execute
            session_id: Session identifier
            tool_name: Optional tool being called

        Returns:
            AutonomyCheckResult indicating if execution is allowed
        """
        # Ensure agent is registered
        if agent.name not in self._agent_states:
            self.register_agent(agent)

        # ADVISORY mode: must present options first
        if agent.autonomy == AutonomyLevel.ADVISORY:
            if self.config.strict_advisory_mode:
                if not self.check_options_presented(agent.name, session_id):
                    return AutonomyCheckResult(
                        allowed=not self.config.block_on_violation,
                        reason="ADVISORY agents must present options before execution",
                        requires_options=True,
                        violation_type="advisory_no_options",
                    )

        # SUPERVISED mode: check restricted tools
        if agent.autonomy == AutonomyLevel.SUPERVISED and tool_name:
            restricted = self._is_restricted_tool(tool_name)
            if restricted:
                return AutonomyCheckResult(
                    allowed=not self.config.block_on_violation,
                    reason=f"SUPERVISED agent using restricted tool: {tool_name}",
                    requires_approval=True,
                    violation_type="supervised_restricted_tool",
                )

        # Update execution count
        self._agent_states[agent.name]["execution_count"] += 1

        return AutonomyCheckResult(
            allowed=True,
            reason="Autonomy check passed",
        )

    def _is_restricted_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is restricted.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is restricted
        """
        if tool_name in self.config.external_api_tools:
            return RestrictedToolCategory.EXTERNAL_API in self.config.restricted_categories

        if tool_name in self.config.data_modification_tools:
            return RestrictedToolCategory.DATA_MODIFICATION in self.config.restricted_categories

        return False

    def validate_execution(
        self,
        agent: BaseAgent,
        session_id: str,
        tool_name: str | None = None,
    ) -> None:
        """
        Validate autonomy constraints and raise if violated.

        Args:
            agent: Agent to validate
            session_id: Session identifier
            tool_name: Optional tool being called

        Raises:
            AutonomyViolation: If constraints are violated and blocking is enabled
        """
        result = self.check_pre_execute(agent, session_id, tool_name)

        if not result.allowed:
            raise AutonomyViolation(
                message=result.reason,
                agent_name=agent.name,
                autonomy_level=agent.autonomy,
            )

        if result.requires_options:
            logger.warning(
                f"Agent {agent.name} should present options before execution "
                f"(ADVISORY autonomy)"
            )

        if result.requires_approval:
            logger.warning(
                f"Agent {agent.name} using restricted tool without approval "
                f"(SUPERVISED autonomy)"
            )

    def get_agent_state(self, agent_name: str) -> dict[str, Any]:
        """
        Get the current state of an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent state dictionary
        """
        return self._agent_states.get(agent_name, {})

    def get_stats(self) -> dict[str, Any]:
        """
        Get middleware statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "registered_agents": len(self._agent_states),
            "active_sessions": len(set(k.split(":")[0] for k in self._options_presented.keys())),
            "options_presented_count": sum(1 for v in self._options_presented.values() if v),
            "total_executions": sum(
                s.get("execution_count", 0) for s in self._agent_states.values()
            ),
        }
