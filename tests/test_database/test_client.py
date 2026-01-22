"""Tests for database client."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/analyst_test")

from src.database.client import DatabaseClient, get_database_client, reset_database_client


class TestDatabaseClient:
    """Test suite for DatabaseClient."""

    def test_initialization_with_url(self):
        """Test client initialization with explicit URL."""
        client = DatabaseClient(database_url="postgresql://localhost:5432/test")
        assert client._database_url == "postgresql+asyncpg://localhost:5432/test"

    def test_initialization_converts_url(self):
        """Test that postgresql:// is converted to asyncpg."""
        client = DatabaseClient(database_url="postgresql://user:pass@host/db")
        assert "asyncpg" in client._database_url

    def test_engine_not_connected_initially(self):
        """Test that engine is None before connect."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        assert client._engine is None
        assert client._session_factory is None

    @pytest.mark.asyncio
    async def test_connect_creates_engine(self):
        """Test that connect creates the engine."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        with patch("src.database.client.create_async_engine") as mock_engine:
            mock_engine.return_value = MagicMock()

            await client.connect()

            mock_engine.assert_called_once()
            assert client._engine is not None
            assert client._session_factory is not None

    @pytest.mark.asyncio
    async def test_connect_warns_if_already_connected(self):
        """Test that connect logs warning if already connected."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        with patch("src.database.client.create_async_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            await client.connect()

            # Second connect should warn
            with patch("src.database.client.logger") as mock_logger:
                await client.connect()
                mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_disposes_engine(self):
        """Test that disconnect disposes the engine."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        mock_engine = AsyncMock()
        client._engine = mock_engine
        client._session_factory = MagicMock()

        await client.disconnect()

        mock_engine.dispose.assert_called_once()
        assert client._engine is None
        assert client._session_factory is None

    @pytest.mark.asyncio
    async def test_session_raises_if_not_connected(self):
        """Test that session context manager raises if not connected."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        with pytest.raises(RuntimeError, match="not connected"):
            async with client.session():
                pass

    @pytest.mark.asyncio
    async def test_create_tables_raises_if_not_connected(self):
        """Test that create_tables raises if not connected."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        with pytest.raises(RuntimeError, match="not connected"):
            await client.create_tables()


class TestGetDatabaseClient:
    """Test suite for get_database_client singleton."""

    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the singleton before and after each test."""
        await reset_database_client()
        yield
        await reset_database_client()

    def test_returns_database_client(self):
        """Test that get_database_client returns a DatabaseClient."""
        client = get_database_client()
        assert isinstance(client, DatabaseClient)

    def test_returns_same_instance(self):
        """Test that get_database_client returns singleton."""
        client1 = get_database_client()
        client2 = get_database_client()
        assert client1 is client2


class TestDatabaseClientOperations:
    """Test database operations with mocked sessions."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        client._engine = MagicMock()
        client._session_factory = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_save_analysis_parameters(self, mock_client):
        """Test save_analysis accepts correct parameters."""
        # The actual save_analysis would use a real session
        # Here we just test the interface exists and accepts parameters

        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock the flush and refresh
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()
            mock_session.refresh = AsyncMock()

            result = await mock_client.save_analysis(
                analysis_type="statistical",
                session_id="test-session",
                parameters={"test": "value"},
                status="running",
            )

            # Verify the session was used
            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_analysis_parameters(self, mock_client):
        """Test update_analysis returns None with no updates."""
        result = await mock_client.update_analysis(
            analysis_id=1,
            status=None,
            results=None,
            completed=False,
        )

        assert result is None

    def test_client_has_required_methods(self, mock_client):
        """Test that client has all required methods."""
        # Analysis operations
        assert hasattr(mock_client, "save_analysis")
        assert hasattr(mock_client, "update_analysis")
        assert hasattr(mock_client, "get_analysis_history")
        assert hasattr(mock_client, "get_analysis_by_id")

        # Checksum operations
        assert hasattr(mock_client, "save_checksum")
        assert hasattr(mock_client, "verify_checksum")
        assert hasattr(mock_client, "get_checksum")

        # Preference operations
        assert hasattr(mock_client, "get_preferences")
        assert hasattr(mock_client, "set_preferences")
        assert hasattr(mock_client, "delete_preference")

        # Workflow operations
        assert hasattr(mock_client, "save_workflow_state")
        assert hasattr(mock_client, "get_workflow_state")
        assert hasattr(mock_client, "get_workflow_states_by_session")
        assert hasattr(mock_client, "delete_workflow_state")

        # Connection operations
        assert hasattr(mock_client, "connect")
        assert hasattr(mock_client, "disconnect")
        assert hasattr(mock_client, "create_tables")
        assert hasattr(mock_client, "session")


class TestAnalysisHistoryOperations:
    """Test analysis history CRUD operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        client._engine = MagicMock()
        client._session_factory = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_get_analysis_history_no_filters(self, mock_client):
        """Test retrieving analysis history without filters."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_analysis_history()

            assert result == []
            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_analysis_history_with_session_filter(self, mock_client):
        """Test retrieving analysis history with session filter."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_analysis_history(session_id="test-session")

            assert result == []

    @pytest.mark.asyncio
    async def test_get_analysis_history_with_type_filter(self, mock_client):
        """Test retrieving analysis history with analysis type filter."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_analysis_history(analysis_type="statistical")

            assert result == []

    @pytest.mark.asyncio
    async def test_get_analysis_by_id(self, mock_client):
        """Test retrieving a specific analysis by ID."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_analysis_by_id(analysis_id=999)

            assert result is None

    @pytest.mark.asyncio
    async def test_update_analysis_with_status(self, mock_client):
        """Test updating analysis with status change."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.update_analysis(
                analysis_id=1,
                status="completed",
            )

            mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_analysis_with_results(self, mock_client):
        """Test updating analysis with results."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.update_analysis(
                analysis_id=1,
                results={"mean": 100.5},
                completed=True,
            )

            mock_session.execute.assert_called_once()


class TestChecksumOperations:
    """Test checksum CRUD operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        client._engine = MagicMock()
        client._session_factory = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_save_checksum_new(self, mock_client):
        """Test saving a new checksum."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            # Mock no existing checksum
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()
            mock_session.refresh = AsyncMock()

            result = await mock_client.save_checksum(
                filename="test.csv",
                checksum="abc123def456",
                file_size=1024,
            )

            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_checksum_not_found(self, mock_client):
        """Test verifying checksum when file not in database."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.verify_checksum("test.csv", "expected123")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_checksum(self, mock_client):
        """Test getting checksum for a file."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_checksum("test.csv")

            assert result is None


class TestUserPreferenceOperations:
    """Test user preference CRUD operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        client._engine = MagicMock()
        client._session_factory = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_get_preferences_not_found(self, mock_client):
        """Test getting preferences for non-existent user."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_preferences("user-123")

            assert result == {}

    @pytest.mark.asyncio
    async def test_set_preferences_new_user(self, mock_client):
        """Test setting preferences for new user."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()
            mock_session.refresh = AsyncMock()

            result = await mock_client.set_preferences(
                user_id="user-123",
                preferences={"theme": "dark"},
            )

            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_preference_not_found(self, mock_client):
        """Test deleting preference that doesn't exist."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.delete_preference("user-123", "theme")

            assert result is False


class TestWorkflowStateOperations:
    """Test workflow state CRUD operations."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked session."""
        client = DatabaseClient(database_url="postgresql://localhost/test")
        client._engine = MagicMock()
        client._session_factory = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_save_workflow_state_new(self, mock_client):
        """Test saving a new workflow state."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)
            mock_session.add = MagicMock()
            mock_session.flush = AsyncMock()
            mock_session.refresh = AsyncMock()

            result = await mock_client.save_workflow_state(
                workflow_id="wf-123",
                session_id="sess-123",
                phase="analysis",
                state={"step": 1},
            )

            mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_state(self, mock_client):
        """Test getting a workflow state."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_workflow_state("wf-123")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_workflow_states_by_session(self, mock_client):
        """Test getting workflow states by session."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.get_workflow_states_by_session("sess-123")

            assert result == []

    @pytest.mark.asyncio
    async def test_delete_workflow_state_not_found(self, mock_client):
        """Test deleting non-existent workflow state."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.rowcount = 0
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.delete_workflow_state("wf-nonexistent")

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_workflow_state_success(self, mock_client):
        """Test successfully deleting workflow state."""
        with patch.object(mock_client, "session") as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_result = MagicMock()
            mock_result.rowcount = 1
            mock_session.execute = AsyncMock(return_value=mock_result)

            result = await mock_client.delete_workflow_state("wf-123")

            assert result is True


class TestDatabaseClientInitialization:
    """Test database client initialization edge cases."""

    def test_initialization_without_url_uses_settings(self):
        """Test that client uses settings when no URL provided."""
        with patch("src.database.client.get_settings") as mock_settings:
            mock_settings.return_value.database_url = "postgresql://from/settings"

            client = DatabaseClient()

            assert "asyncpg" in client._database_url
            assert "settings" in client._database_url

    def test_asyncpg_url_not_modified(self):
        """Test that asyncpg URLs are not double-converted."""
        client = DatabaseClient(database_url="postgresql+asyncpg://host/db")

        # Should only have one asyncpg, not duplicated
        assert client._database_url.count("asyncpg") == 1

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        """Test that disconnect does nothing when not connected."""
        client = DatabaseClient(database_url="postgresql://localhost/test")

        # Should not raise, even though not connected
        await client.disconnect()

        assert client._engine is None
