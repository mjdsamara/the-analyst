"""
Base Agent class for The Analyst platform.

All specialized agents inherit from this class, which provides:
- Common verification methods
- Logging and audit trail
- Option presentation for human-in-loop
- Approval workflow management
- Middleware integration (autonomy, cost tracking, audit)
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock

from src.config import (
    AGENT_CONFIG,
    AutonomyLevel,
    check_high_stakes,
    get_settings,
    require_confirmation,
)

# Middleware imports - lazy loading to avoid circular imports
if TYPE_CHECKING:
    from src.middleware.audit import AuditMiddleware
    from src.middleware.autonomy import AutonomyMiddleware
    from src.middleware.cost_tracking import CostTrackingMiddleware

logger = logging.getLogger(__name__)

# Middleware singletons (initialized lazily)
_autonomy_middleware: AutonomyMiddleware | None = None
_cost_middleware: CostTrackingMiddleware | None = None
_audit_middleware: AuditMiddleware | None = None


def get_autonomy_middleware() -> AutonomyMiddleware:
    """Get or create the autonomy middleware singleton."""
    global _autonomy_middleware
    if _autonomy_middleware is None:
        from src.middleware.autonomy import AutonomyMiddleware

        _autonomy_middleware = AutonomyMiddleware.get_instance()
    return _autonomy_middleware


def get_cost_middleware() -> CostTrackingMiddleware:
    """Get or create the cost tracking middleware singleton."""
    global _cost_middleware
    if _cost_middleware is None:
        from src.middleware.cost_tracking import CostTrackingMiddleware

        _cost_middleware = CostTrackingMiddleware.get_instance()
    return _cost_middleware


def get_audit_middleware() -> AuditMiddleware:
    """Get or create the audit middleware singleton."""
    global _audit_middleware
    if _audit_middleware is None:
        from src.middleware.audit import AuditMiddleware

        _audit_middleware = AuditMiddleware.get_instance()
    return _audit_middleware


def reset_middleware() -> None:
    """Reset all middleware singletons (for testing)."""
    global _autonomy_middleware, _cost_middleware, _audit_middleware
    _autonomy_middleware = None
    _cost_middleware = None
    _audit_middleware = None


T = TypeVar("T")


class AgentStatus(str, Enum):
    """Status of agent execution."""

    IDLE = "idle"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentContext:
    """Context passed between agents during workflow execution."""

    session_id: str
    user_id: str = "default"
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    shared_data: dict[str, Any] = field(default_factory=dict)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append(
            {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
        )

    def set_data(self, key: str, value: Any) -> None:
        """Store data in shared context."""
        self.shared_data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from shared context."""
        return self.shared_data.get(key, default)

    def record_approval(self, action: str, approved: bool, reason: str = "") -> None:
        """Record an approval decision."""
        self.approvals.append(
            {
                "action": action,
                "approved": approved,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


@dataclass
class AgentResult(Generic[T]):
    """Result from agent execution."""

    success: bool
    data: T | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    approval_request: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @classmethod
    def success_result(cls, data: T, **metadata: Any) -> AgentResult[T]:
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_result(cls, error: str, **metadata: Any) -> AgentResult[T]:
        """Create an error result."""
        return cls(success=False, error=error, metadata=metadata)

    @classmethod
    def approval_required(
        cls, action: str, reason: str, details: dict[str, Any] | None = None
    ) -> AgentResult[T]:
        """Create a result requiring approval."""
        return cls(
            success=True,
            requires_approval=True,
            approval_request=require_confirmation(action, reason, details),
        )


@dataclass
class AgentOption:
    """An option to present to the user."""

    id: str
    title: str
    description: str
    recommended: bool = False
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    estimated_complexity: str = ""  # "low", "medium", "high"


class BaseAgent(ABC):
    """
    Base class for all agents in The Analyst platform.

    Provides common functionality for:
    - LLM interaction via Anthropic API
    - Verification and validation
    - Logging and audit trails
    - Human-in-loop approval workflows
    - Option presentation
    """

    def __init__(
        self,
        name: str,
        context: AgentContext | None = None,
    ) -> None:
        """
        Initialize the agent.

        Args:
            name: Agent identifier (must match AGENT_CONFIG key)
            context: Shared context for the workflow
        """
        self.name = name
        self.context = context or AgentContext(session_id="default")
        self.status = AgentStatus.IDLE
        self._logs: list[str] = []

        # Load configuration
        if name not in AGENT_CONFIG:
            raise ValueError(f"Unknown agent: {name}. Must be one of {list(AGENT_CONFIG.keys())}")

        config = AGENT_CONFIG[name]
        self.model = config["model"].value
        self.autonomy = config["autonomy"]
        self.description = config["description"]

        # Initialize Anthropic client
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)

        self.log(f"Initialized {name} agent with {self.autonomy.value} autonomy")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for this agent. Must be implemented by subclasses."""
        pass

    @property
    def single_job(self) -> str:
        """The single job this agent is responsible for."""
        return str(self.description)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] [{self.name}] [{level}] {message}"
        self._logs.append(log_entry)

        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.name}] {message}")

    async def call_llm(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """
        Call the Claude API with the agent's configuration.

        Args:
            messages: Conversation messages
            system: System prompt override (uses self.system_prompt if not provided)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            The assistant's response text
        """
        self.log(f"Calling LLM with {len(messages)} messages")

        # Audit: Log LLM call
        audit = get_audit_middleware()
        system_prompt = system or self.system_prompt
        audit.log_llm_call(
            agent=self,
            session_id=self.context.session_id,
            message_count=len(messages),
            system_prompt_preview=system_prompt,
        )

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=cast(list[MessageParam], messages),
        )

        content = ""
        if response.content and isinstance(response.content[0], TextBlock):
            content = response.content[0].text
        self.log(f"LLM response received: {len(content)} chars")

        # Get token usage from response
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)

        # Cost tracking: Record the LLM call cost
        cost_middleware = get_cost_middleware()
        entry = cost_middleware.after_llm_call(
            agent=self,
            session_id=self.context.session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"temperature": temperature, "max_tokens": max_tokens},
        )

        # Audit: Log LLM response with cost
        audit.log_llm_response(
            agent=self,
            session_id=self.context.session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_preview=content,
        )
        audit.log_cost_recorded(
            agent=self,
            session_id=self.context.session_id,
            cost=entry.total_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return content

    def verify_output(self, output: Any, requirements: list[str]) -> tuple[bool, list[str]]:
        """
        Verify that output meets requirements.

        Args:
            output: The output to verify
            requirements: List of requirement descriptions

        Returns:
            Tuple of (all_passed, list of failed requirements)
        """
        failed: list[str] = []
        # Subclasses should implement specific verification logic
        return len(failed) == 0, failed

    def check_high_stakes(self, text: str) -> list[str]:
        """
        Check if text contains high-stakes keywords requiring confirmation.

        Args:
            text: Text to check

        Returns:
            List of matched high-stakes keywords
        """
        return check_high_stakes(text)

    def present_options(
        self,
        options: list[AgentOption],
        context_message: str = "",
    ) -> dict[str, Any]:
        """
        Present options to the user for selection.

        Args:
            options: List of options to present
            context_message: Additional context about the options

        Returns:
            Formatted options presentation
        """
        formatted = {
            "type": "options_presentation",
            "agent": self.name,
            "context": context_message,
            "options": [],
            "requires_selection": self.autonomy == AutonomyLevel.ADVISORY,
        }

        for opt in options:
            formatted["options"].append(
                {
                    "id": opt.id,
                    "title": opt.title,
                    "description": opt.description,
                    "recommended": opt.recommended,
                    "pros": opt.pros,
                    "cons": opt.cons,
                    "complexity": opt.estimated_complexity,
                }
            )

        self.log(f"Presenting {len(options)} options to user")

        # Autonomy: Mark options as presented for ADVISORY mode enforcement
        autonomy = get_autonomy_middleware()
        autonomy.mark_options_presented(self.name, self.context.session_id)

        # Audit: Log option presentation
        audit = get_audit_middleware()
        audit.log_option_presented(
            agent=self,
            session_id=self.context.session_id,
            options=formatted["options"],
            context_message=context_message,
        )

        return formatted

    async def request_approval(
        self,
        action: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> AgentResult[None]:
        """
        Request approval for an action.

        Args:
            action: Description of the action
            reason: Why approval is needed
            details: Additional details

        Returns:
            AgentResult indicating approval is required
        """
        self.status = AgentStatus.AWAITING_APPROVAL
        self.log(f"Requesting approval for: {action}")

        return AgentResult.approval_required(action, reason, details)

    def compute_checksum(self, data: bytes | str) -> str:
        """
        Compute SHA-256 checksum for data integrity verification.

        Args:
            data: Data to checksum

        Returns:
            Hexadecimal checksum string
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def verify_data_unchanged(self, path: Path, expected_checksum: str) -> bool:
        """
        Verify that source data has not been modified.

        Args:
            path: Path to the data file
            expected_checksum: Expected checksum

        Returns:
            True if data is unchanged
        """
        if not path.exists():
            self.log(f"Data file not found: {path}", level="WARNING")
            return False

        actual = self.compute_checksum(path.read_bytes())
        return actual == expected_checksum

    @abstractmethod
    async def execute(self, **kwargs: Any) -> AgentResult[Any]:
        """
        Execute the agent's main task.

        Must be implemented by subclasses.

        Args:
            **kwargs: Task-specific arguments

        Returns:
            AgentResult with execution outcome
        """
        pass

    async def run(self, **kwargs: Any) -> AgentResult[Any]:
        """
        Run the agent with status tracking and error handling.

        Args:
            **kwargs: Arguments passed to execute()

        Returns:
            AgentResult with execution outcome
        """
        import time

        start_time = time.time()
        self.status = AgentStatus.RUNNING
        self.log("Starting execution")

        # Get middleware instances
        autonomy = get_autonomy_middleware()
        audit = get_audit_middleware()

        # Audit: Log agent start
        audit.log_agent_start(
            agent=self,
            session_id=self.context.session_id,
            kwargs=kwargs,
        )

        # Autonomy: Check pre-execution constraints
        autonomy_result = autonomy.check_pre_execute(
            agent=self,
            session_id=self.context.session_id,
        )
        if autonomy_result.requires_options and self.autonomy == AutonomyLevel.ADVISORY:
            self.log(f"Autonomy check: {autonomy_result.reason}", level="WARNING")
            # Note: We log but don't block - strict mode is configurable

        try:
            # Check for high-stakes keywords in input
            input_text = str(kwargs)
            high_stakes = self.check_high_stakes(input_text)
            if high_stakes and self.autonomy == AutonomyLevel.ADVISORY:
                self.log(f"High-stakes keywords detected: {high_stakes}")

                # Audit: Log high-stakes detection
                audit.log_high_stakes_detected(
                    agent=self,
                    session_id=self.context.session_id,
                    keywords=high_stakes,
                    input_preview=input_text,
                )

                # Audit: Log approval request
                audit.log_approval_requested(
                    agent=self,
                    session_id=self.context.session_id,
                    action=f"Execute {self.name} with high-stakes input",
                    reason=f"Input contains high-stakes keywords: {high_stakes}",
                )

                return await self.request_approval(
                    action=f"Execute {self.name} with high-stakes input",
                    reason=f"Input contains high-stakes keywords: {high_stakes}",
                    details={"keywords": high_stakes, "input_preview": input_text[:500]},
                )

            result = await self.execute(**kwargs)
            result.logs = self._logs.copy()
            result.execution_time_ms = (time.time() - start_time) * 1000

            if result.success and not result.requires_approval:
                self.status = AgentStatus.COMPLETED
                # Audit: Log successful completion
                audit.log_agent_complete(
                    agent=self,
                    session_id=self.context.session_id,
                    success=True,
                    execution_time_ms=result.execution_time_ms,
                    result_summary=str(result.data)[:200] if result.data else None,
                )
            elif result.requires_approval:
                self.status = AgentStatus.AWAITING_APPROVAL
            else:
                self.status = AgentStatus.FAILED
                # Audit: Log failure
                audit.log_agent_complete(
                    agent=self,
                    session_id=self.context.session_id,
                    success=False,
                    execution_time_ms=result.execution_time_ms,
                    result_summary=result.error,
                )

            self.log(f"Execution completed: success={result.success}")
            return result

        except Exception as e:
            self.status = AgentStatus.FAILED
            self.log(f"Execution failed: {e}", level="ERROR")
            result = AgentResult.error_result(str(e))
            result.logs = self._logs.copy()
            result.execution_time_ms = (time.time() - start_time) * 1000

            # Audit: Log error
            audit.log_agent_error(
                agent=self,
                session_id=self.context.session_id,
                error=str(e),
                execution_time_ms=result.execution_time_ms,
            )

            return result
