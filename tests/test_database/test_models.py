"""Tests for database models."""

import os
from datetime import datetime
from uuid import uuid4

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/analyst_test")

from src.database.models import (
    AnalysisHistory,
    Base,
    DataChecksum,
    UserPreference,
    WorkflowState,
)


class TestAnalysisHistory:
    """Test suite for AnalysisHistory model."""

    def test_model_attributes(self):
        """Test that model has expected attributes."""
        analysis = AnalysisHistory(
            session_id=str(uuid4()),
            analysis_type="statistical",
            status="running",
            parameters={"columns": ["a", "b"]},
            results={"mean": 100},
        )

        assert analysis.session_id is not None
        assert analysis.analysis_type == "statistical"
        assert analysis.status == "running"
        assert analysis.parameters == {"columns": ["a", "b"]}
        assert analysis.results == {"mean": 100}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.utcnow()
        analysis = AnalysisHistory(
            id=1,
            session_id="test-session-123",
            analysis_type="comprehensive",
            started_at=now,
            completed_at=now,
            status="completed",
            parameters={"test": "value"},
            results={"output": "data"},
            metadata_={"version": "1.0"},
        )

        d = analysis.to_dict()

        assert d["id"] == 1
        assert d["session_id"] == "test-session-123"
        assert d["analysis_type"] == "comprehensive"
        assert d["status"] == "completed"
        assert d["parameters"] == {"test": "value"}
        assert d["results"] == {"output": "data"}
        assert d["metadata"] == {"version": "1.0"}
        assert d["started_at"] is not None
        assert d["completed_at"] is not None

    def test_to_dict_with_none_dates(self):
        """Test to_dict handles None dates."""
        analysis = AnalysisHistory(
            id=1,
            session_id="test-session",
            analysis_type="test",
            status="running",
        )
        analysis.started_at = None
        analysis.completed_at = None

        d = analysis.to_dict()

        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_tablename(self):
        """Test table name is correct."""
        assert AnalysisHistory.__tablename__ == "analysis_history"


class TestDataChecksum:
    """Test suite for DataChecksum model."""

    def test_model_attributes(self):
        """Test that model has expected attributes."""
        checksum = DataChecksum(
            filename="data.csv",
            checksum="abc123def456",
            file_size=1024,
        )

        assert checksum.filename == "data.csv"
        assert checksum.checksum == "abc123def456"
        assert checksum.file_size == 1024

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.utcnow()
        checksum = DataChecksum(
            id=1,
            filename="test.csv",
            checksum="sha256hash",
            file_size=2048,
            first_seen=now,
            last_verified=now,
        )

        d = checksum.to_dict()

        assert d["id"] == 1
        assert d["filename"] == "test.csv"
        assert d["checksum"] == "sha256hash"
        assert d["file_size"] == 2048
        assert d["first_seen"] is not None
        assert d["last_verified"] is not None

    def test_tablename(self):
        """Test table name is correct."""
        assert DataChecksum.__tablename__ == "data_checksums"


class TestUserPreference:
    """Test suite for UserPreference model."""

    def test_model_attributes(self):
        """Test that model has expected attributes."""
        pref = UserPreference(
            user_id="user123",
            preferences={"theme": "dark", "language": "en"},
        )

        assert pref.user_id == "user123"
        assert pref.preferences == {"theme": "dark", "language": "en"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.utcnow()
        pref = UserPreference(
            id=1,
            user_id="test_user",
            preferences={"key": "value"},
            created_at=now,
            updated_at=now,
        )

        d = pref.to_dict()

        assert d["id"] == 1
        assert d["user_id"] == "test_user"
        assert d["preferences"] == {"key": "value"}
        assert d["created_at"] is not None
        assert d["updated_at"] is not None

    def test_tablename(self):
        """Test table name is correct."""
        assert UserPreference.__tablename__ == "user_preferences"


class TestWorkflowState:
    """Test suite for WorkflowState model."""

    def test_model_attributes(self):
        """Test that model has expected attributes."""
        workflow = WorkflowState(
            workflow_id=str(uuid4()),
            session_id=str(uuid4()),
            phase="analysis",
            state={"step": 1, "data": "loaded"},
        )

        assert workflow.workflow_id is not None
        assert workflow.session_id is not None
        assert workflow.phase == "analysis"
        assert workflow.state == {"step": 1, "data": "loaded"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.utcnow()
        workflow = WorkflowState(
            id=1,
            workflow_id="wf-123",
            session_id="sess-456",
            phase="completed",
            state={"results": "done"},
            created_at=now,
            updated_at=now,
        )

        d = workflow.to_dict()

        assert d["id"] == 1
        assert d["workflow_id"] == "wf-123"
        assert d["session_id"] == "sess-456"
        assert d["phase"] == "completed"
        assert d["state"] == {"results": "done"}
        assert d["created_at"] is not None
        assert d["updated_at"] is not None

    def test_tablename(self):
        """Test table name is correct."""
        assert WorkflowState.__tablename__ == "workflow_states"


class TestBaseModel:
    """Test suite for Base model."""

    def test_base_is_declarative_base(self):
        """Test that Base is a proper declarative base."""
        from sqlalchemy.orm import DeclarativeBase

        assert issubclass(Base, DeclarativeBase)

    def test_all_models_inherit_from_base(self):
        """Test that all models inherit from Base."""
        assert issubclass(AnalysisHistory, Base)
        assert issubclass(DataChecksum, Base)
        assert issubclass(UserPreference, Base)
        assert issubclass(WorkflowState, Base)
