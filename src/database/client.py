"""
Database client for The Analyst platform.

Provides async database operations with connection pooling.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings
from src.database.models import (
    AnalysisHistory,
    Base,
    DataChecksum,
    UserPreference,
    WorkflowState,
)

logger = logging.getLogger(__name__)


class DatabaseClient:
    """
    Async database client for The Analyst.

    Provides connection pooling and async operations for all database models.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """
        Initialize the database client.

        Args:
            database_url: PostgreSQL connection string. If not provided,
                         uses DATABASE_URL from settings.
        """
        if database_url is None:
            settings = get_settings()
            database_url = settings.database_url

        # Convert postgresql:// to postgresql+asyncpg:// for async
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self._database_url = database_url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        """
        Establish database connection with pooling.

        Creates the async engine and session factory.
        """
        if self._engine is not None:
            logger.warning("Database already connected")
            return

        logger.info("Connecting to database...")

        self._engine = create_async_engine(
            self._database_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 minutes
            echo=False,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        logger.info("Database connected successfully")

    async def disconnect(self) -> None:
        """Close database connection and dispose of the engine."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database disconnected")

    async def create_tables(self) -> None:
        """Create all tables if they don't exist."""
        if self._engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database tables created/verified")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async session context manager.

        Yields:
            AsyncSession for database operations

        Example:
            async with client.session() as session:
                result = await session.execute(query)
        """
        if self._session_factory is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # =========================================================================
    # Analysis History Operations
    # =========================================================================

    async def save_analysis(
        self,
        analysis_type: str,
        session_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisHistory:
        """
        Save a new analysis record.

        Args:
            analysis_type: Type of analysis performed
            session_id: Session identifier (generated if not provided)
            parameters: Analysis parameters
            status: Current status (running, completed, failed)
            metadata: Additional metadata

        Returns:
            The created AnalysisHistory record
        """
        analysis = AnalysisHistory(
            session_id=session_id or str(uuid4()),
            analysis_type=analysis_type,
            parameters=parameters,
            status=status,
            metadata_=metadata,
        )

        async with self.session() as session:
            session.add(analysis)
            await session.flush()
            await session.refresh(analysis)

        logger.info(f"Saved analysis: {analysis.id} ({analysis_type})")
        return analysis

    async def update_analysis(
        self,
        analysis_id: int,
        status: str | None = None,
        results: dict[str, Any] | None = None,
        completed: bool = False,
    ) -> AnalysisHistory | None:
        """
        Update an existing analysis record.

        Args:
            analysis_id: ID of the analysis to update
            status: New status
            results: Analysis results
            completed: Whether to mark as completed

        Returns:
            Updated AnalysisHistory or None if not found
        """
        update_data: dict[str, Any] = {}
        if status is not None:
            update_data["status"] = status
        if results is not None:
            update_data["results"] = results
        if completed:
            update_data["completed_at"] = datetime.utcnow()

        if not update_data:
            return None

        async with self.session() as session:
            stmt = (
                update(AnalysisHistory)
                .where(AnalysisHistory.id == analysis_id)
                .values(**update_data)
                .returning(AnalysisHistory)
            )
            result = await session.execute(stmt)
            analysis = result.scalar_one_or_none()

        if analysis:
            logger.info(f"Updated analysis: {analysis_id}")

        return analysis

    async def get_analysis_history(
        self,
        session_id: str | None = None,
        analysis_type: str | None = None,
        limit: int = 100,
    ) -> list[AnalysisHistory]:
        """
        Retrieve analysis history with optional filters.

        Args:
            session_id: Filter by session ID
            analysis_type: Filter by analysis type
            limit: Maximum number of records to return

        Returns:
            List of AnalysisHistory records
        """
        async with self.session() as session:
            stmt = select(AnalysisHistory).order_by(AnalysisHistory.started_at.desc())

            if session_id:
                stmt = stmt.where(AnalysisHistory.session_id == session_id)
            if analysis_type:
                stmt = stmt.where(AnalysisHistory.analysis_type == analysis_type)

            stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_analysis_by_id(self, analysis_id: int) -> AnalysisHistory | None:
        """Get a specific analysis by ID."""
        async with self.session() as session:
            stmt = select(AnalysisHistory).where(AnalysisHistory.id == analysis_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # =========================================================================
    # Data Checksum Operations
    # =========================================================================

    async def save_checksum(
        self,
        filename: str,
        checksum: str,
        file_size: int | None = None,
    ) -> DataChecksum:
        """
        Save or update a data checksum.

        Args:
            filename: Name of the file
            checksum: SHA-256 checksum
            file_size: Optional file size in bytes

        Returns:
            The created or updated DataChecksum record
        """
        async with self.session() as session:
            # Check if exists
            stmt = select(DataChecksum).where(DataChecksum.filename == filename)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.checksum = checksum
                if file_size is not None:
                    existing.file_size = file_size
                existing.last_verified = datetime.utcnow()
                await session.flush()
                await session.refresh(existing)
                logger.info(f"Updated checksum for: {filename}")
                return existing
            else:
                # Create new
                record = DataChecksum(
                    filename=filename,
                    checksum=checksum,
                    file_size=file_size,
                )
                session.add(record)
                await session.flush()
                await session.refresh(record)
                logger.info(f"Saved new checksum for: {filename}")
                return record

    async def verify_checksum(
        self,
        filename: str,
        expected_checksum: str,
    ) -> bool:
        """
        Verify a file's checksum matches the stored value.

        Args:
            filename: Name of the file
            expected_checksum: Expected SHA-256 checksum

        Returns:
            True if checksum matches, False otherwise
        """
        async with self.session() as session:
            stmt = select(DataChecksum).where(DataChecksum.filename == filename)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if record is None:
                logger.warning(f"No checksum found for: {filename}")
                return False

            matches = record.checksum == expected_checksum

            if matches:
                # Update last_verified timestamp
                record.last_verified = datetime.utcnow()

            return matches

    async def get_checksum(self, filename: str) -> DataChecksum | None:
        """Get the stored checksum for a file."""
        async with self.session() as session:
            stmt = select(DataChecksum).where(DataChecksum.filename == filename)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    # =========================================================================
    # User Preference Operations
    # =========================================================================

    async def get_preferences(self, user_id: str) -> dict[str, Any]:
        """
        Get user preferences.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of preferences (empty dict if not found)
        """
        async with self.session() as session:
            stmt = select(UserPreference).where(UserPreference.user_id == user_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            return record.preferences if record else {}

    async def set_preferences(
        self,
        user_id: str,
        preferences: dict[str, Any],
        merge: bool = True,
    ) -> UserPreference:
        """
        Set user preferences.

        Args:
            user_id: User identifier
            preferences: Preferences to set
            merge: If True, merge with existing; if False, replace

        Returns:
            The UserPreference record
        """
        async with self.session() as session:
            stmt = select(UserPreference).where(UserPreference.user_id == user_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                if merge:
                    existing.preferences = {**existing.preferences, **preferences}
                else:
                    existing.preferences = preferences
                await session.flush()
                await session.refresh(existing)
                return existing
            else:
                record = UserPreference(
                    user_id=user_id,
                    preferences=preferences,
                )
                session.add(record)
                await session.flush()
                await session.refresh(record)
                return record

    async def delete_preference(self, user_id: str, key: str) -> bool:
        """
        Delete a specific preference key.

        Args:
            user_id: User identifier
            key: Preference key to delete

        Returns:
            True if deleted, False if not found
        """
        async with self.session() as session:
            stmt = select(UserPreference).where(UserPreference.user_id == user_id)
            result = await session.execute(stmt)
            record = result.scalar_one_or_none()

            if record and key in record.preferences:
                del record.preferences[key]
                return True

            return False

    # =========================================================================
    # Workflow State Operations
    # =========================================================================

    async def save_workflow_state(
        self,
        workflow_id: str,
        session_id: str,
        phase: str,
        state: dict[str, Any],
    ) -> WorkflowState:
        """
        Save or update a workflow state.

        Args:
            workflow_id: Unique workflow identifier
            session_id: Session identifier
            phase: Current workflow phase
            state: State data to persist

        Returns:
            The WorkflowState record
        """
        async with self.session() as session:
            # Check if exists
            stmt = select(WorkflowState).where(WorkflowState.workflow_id == workflow_id)
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing:
                existing.phase = phase
                existing.state = state
                await session.flush()
                await session.refresh(existing)
                logger.info(f"Updated workflow state: {workflow_id}")
                return existing
            else:
                record = WorkflowState(
                    workflow_id=workflow_id,
                    session_id=session_id,
                    phase=phase,
                    state=state,
                )
                session.add(record)
                await session.flush()
                await session.refresh(record)
                logger.info(f"Saved new workflow state: {workflow_id}")
                return record

    async def get_workflow_state(self, workflow_id: str) -> WorkflowState | None:
        """Get a workflow state by workflow ID."""
        async with self.session() as session:
            stmt = select(WorkflowState).where(WorkflowState.workflow_id == workflow_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_workflow_states_by_session(
        self,
        session_id: str,
    ) -> list[WorkflowState]:
        """Get all workflow states for a session."""
        async with self.session() as session:
            stmt = (
                select(WorkflowState)
                .where(WorkflowState.session_id == session_id)
                .order_by(WorkflowState.updated_at.desc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def delete_workflow_state(self, workflow_id: str) -> bool:
        """
        Delete a workflow state.

        Args:
            workflow_id: Workflow ID to delete

        Returns:
            True if deleted, False if not found
        """
        async with self.session() as session:
            stmt = delete(WorkflowState).where(WorkflowState.workflow_id == workflow_id)
            result = cast(CursorResult[Any], await session.execute(stmt))
            deleted: bool = result.rowcount > 0

        if deleted:
            logger.info(f"Deleted workflow state: {workflow_id}")

        return deleted


# Singleton instance
_database_client: DatabaseClient | None = None


def get_database_client() -> DatabaseClient:
    """
    Get the singleton database client instance.

    Returns:
        The global DatabaseClient instance
    """
    global _database_client
    if _database_client is None:
        _database_client = DatabaseClient()
    return _database_client


async def reset_database_client() -> None:
    """Reset the singleton database client (useful for testing)."""
    global _database_client
    if _database_client is not None:
        await _database_client.disconnect()
        _database_client = None
