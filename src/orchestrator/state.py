"""
State management for The Analyst orchestrator.

Handles conversation state, workflow state, and memory persistence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowPhase(str, Enum):
    """Phases of the analysis workflow."""

    INTENT_PARSING = "intent_parsing"
    OPTION_PRESENTATION = "option_presentation"
    AWAITING_APPROVAL = "awaiting_approval"
    DATA_RETRIEVAL = "data_retrieval"
    DATA_TRANSFORM = "data_transform"
    ANALYSIS = "analysis"
    INSIGHTS = "insights"
    OUTPUT_SELECTION = "output_selection"
    OUTPUT_GENERATION = "output_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationState:
    """State of the current conversation."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = field(default_factory=list)
    current_intent: str | None = None
    pending_approval: dict[str, Any] | None = None
    selected_option: str | None = None
    user_preferences: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
        self.updated_at = datetime.utcnow()

    def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM API."""
        return [{"role": m.role, "content": m.content} for m in self.messages if m.role != "system"]

    def clear_pending(self) -> None:
        """Clear pending approval state."""
        self.pending_approval = None
        self.selected_option = None

    def record_approval(self, action: str, approved: bool, reason: str = "") -> None:
        """Record an approval decision in the conversation metadata."""
        if "approvals" not in self.user_preferences:
            self.user_preferences["approvals"] = []
        self.user_preferences["approvals"].append(
            {
                "action": action,
                "approved": approved,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "current_intent": self.current_intent,
            "pending_approval": self.pending_approval,
            "selected_option": self.selected_option,
            "user_preferences": self.user_preferences,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationState:
        """Create from dictionary."""
        state = cls(
            session_id=data["session_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            current_intent=data.get("current_intent"),
            pending_approval=data.get("pending_approval"),
            selected_option=data.get("selected_option"),
            user_preferences=data.get("user_preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
        return state


@dataclass
class WorkflowState:
    """State of the current analysis workflow."""

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: WorkflowPhase = WorkflowPhase.INTENT_PARSING
    agents_used: list[str] = field(default_factory=list)
    agent_results: dict[str, Any] = field(default_factory=dict)
    data_checksums: dict[str, str] = field(default_factory=dict)
    transformations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def advance_phase(self, new_phase: WorkflowPhase) -> None:
        """Advance to a new workflow phase."""
        self.phase = new_phase
        if new_phase in (WorkflowPhase.COMPLETED, WorkflowPhase.FAILED):
            self.completed_at = datetime.utcnow()

    def record_agent_result(self, agent_name: str, result: dict[str, Any]) -> None:
        """Record the result from an agent."""
        self.agents_used.append(agent_name)
        self.agent_results[agent_name] = result

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.errors.append(f"[{datetime.utcnow().isoformat()}] {error}")

    def record_checksum(self, filename: str, checksum: str) -> None:
        """Record a data file checksum."""
        self.data_checksums[filename] = checksum

    def record_transformation(self, operation: str, details: dict[str, Any]) -> None:
        """Record a data transformation."""
        self.transformations.append(
            {"operation": operation, "details": details, "timestamp": datetime.utcnow().isoformat()}
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "phase": self.phase.value,
            "agents_used": self.agents_used,
            "agent_results": self.agent_results,
            "data_checksums": self.data_checksums,
            "transformations": self.transformations,
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Create from dictionary."""
        state = cls(
            workflow_id=data["workflow_id"],
            phase=WorkflowPhase(data["phase"]),
            agents_used=data.get("agents_used", []),
            agent_results=data.get("agent_results", {}),
            data_checksums=data.get("data_checksums", {}),
            transformations=data.get("transformations", []),
            errors=data.get("errors", []),
            started_at=datetime.fromisoformat(data["started_at"]),
        )
        if data.get("completed_at"):
            state.completed_at = datetime.fromisoformat(data["completed_at"])
        return state


class StateManager:
    """Manages persistence of conversation and workflow states."""

    def __init__(self, memory_path: Path | None = None) -> None:
        """Initialize the state manager."""
        self.memory_path = memory_path or Path("memory/data")
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def save_conversation(self, state: ConversationState) -> None:
        """Save conversation state to disk."""
        filepath = self.memory_path / "conversations" / f"{state.session_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def load_conversation(self, session_id: str) -> ConversationState | None:
        """Load conversation state from disk."""
        filepath = self.memory_path / "conversations" / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return ConversationState.from_dict(json.load(f))

    def save_workflow(self, state: WorkflowState) -> None:
        """Save workflow state to disk."""
        filepath = self.memory_path / "workflows" / f"{state.workflow_id}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def load_workflow(self, workflow_id: str) -> WorkflowState | None:
        """Load workflow state from disk."""
        filepath = self.memory_path / "workflows" / f"{workflow_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return WorkflowState.from_dict(json.load(f))

    def append_to_history(self, entry: dict[str, Any]) -> None:
        """Append an entry to analysis history."""
        filepath = self.memory_path / "analysis_history.json"
        history = []
        if filepath.exists():
            with open(filepath) as f:
                history = json.load(f)
        history.append(entry)
        with open(filepath, "w") as f:
            json.dump(history, f, indent=2)
