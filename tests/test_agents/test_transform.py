"""Tests for the Transform Agent."""

import os

import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.transform import (
    TransformAgent,
    TransformationLog,
    TransformationStep,
    TransformResult,
)


class TestTransformAgent:
    """Test suite for TransformAgent."""

    @pytest.fixture
    def agent(self, patch_anthropic, agent_context):
        """Create a transform agent for testing."""
        return TransformAgent(context=agent_context)

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "transform"
        assert agent.autonomy.value == "supervised"
        assert agent.description is not None

    def test_agent_system_prompt(self, agent):
        """Test that agent has a system prompt."""
        assert agent.system_prompt is not None
        assert len(agent.system_prompt) > 0

    # -------------------------------------------------------------------------
    # Execute Tests - Basic Operations
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_with_no_data(self, agent):
        """Test error handling when no data provided."""
        result = await agent.execute(data=None)

        assert not result.success
        assert "no data" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_valid_data(self, agent, sample_dataframe):
        """Test successful execution with valid data."""
        result = await agent.execute(data=sample_dataframe)

        assert result.success
        assert result.data is not None
        assert isinstance(result.data, TransformResult)
        assert isinstance(result.data.data, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_execute_auto_operations(self, agent, sample_dataframe):
        """Test that auto operations are generated when none provided."""
        result = await agent.execute(data=sample_dataframe, operations=None)

        assert result.success
        assert result.data is not None
        # Log should exist even if no operations were needed
        assert result.data.log is not None

    @pytest.mark.asyncio
    async def test_execute_preserves_data_copy(self, agent, sample_dataframe):
        """Test that original data is not modified."""
        original_len = len(sample_dataframe)
        original_cols = list(sample_dataframe.columns)

        await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["value1"]}],
        )

        # Original should be unchanged
        assert len(sample_dataframe) == original_len
        assert list(sample_dataframe.columns) == original_cols

    # -------------------------------------------------------------------------
    # Drop Duplicates Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_drop_duplicates(self, agent, sample_dataframe_with_duplicates):
        """Test dropping duplicate rows."""
        result = await agent.execute(
            data=sample_dataframe_with_duplicates,
            operations=[{"type": "drop_duplicates"}],
        )

        assert result.success
        # Should have fewer rows after dropping duplicates
        assert len(result.data.data) < len(sample_dataframe_with_duplicates)

    @pytest.mark.asyncio
    async def test_drop_duplicates_with_subset(self, agent, sample_dataframe_with_duplicates):
        """Test dropping duplicates based on subset of columns."""
        result = await agent.execute(
            data=sample_dataframe_with_duplicates,
            operations=[{"type": "drop_duplicates", "subset": ["id"], "keep": "first"}],
        )

        assert result.success
        # IDs should be unique
        assert result.data.data["id"].is_unique

    @pytest.mark.asyncio
    async def test_drop_duplicates_keep_last(self, agent, sample_dataframe_with_duplicates):
        """Test dropping duplicates keeping last occurrence."""
        result = await agent.execute(
            data=sample_dataframe_with_duplicates,
            operations=[{"type": "drop_duplicates", "keep": "last"}],
        )

        assert result.success
        assert len(result.data.data) == 5  # Unique rows

    # -------------------------------------------------------------------------
    # Drop Columns Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_drop_columns(self, agent, sample_dataframe):
        """Test dropping specified columns."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["value1"]}],
        )

        assert result.success
        assert "value1" not in result.data.data.columns
        assert "value2" in result.data.data.columns

    @pytest.mark.asyncio
    async def test_drop_multiple_columns(self, agent, sample_dataframe):
        """Test dropping multiple columns."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["value1", "value2"]}],
        )

        assert result.success
        assert "value1" not in result.data.data.columns
        assert "value2" not in result.data.data.columns

    @pytest.mark.asyncio
    async def test_drop_nonexistent_column(self, agent, sample_dataframe):
        """Test that dropping non-existent column doesn't raise error."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["nonexistent"]}],
        )

        assert result.success
        # Original columns should still be present
        assert "value1" in result.data.data.columns

    # -------------------------------------------------------------------------
    # Drop Rows with Nulls Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_drop_rows_with_nulls(self, agent, sample_dataframe_with_nulls):
        """Test dropping rows with null values."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[{"type": "drop_rows_with_nulls"}],
        )

        assert result.success
        assert len(result.data.data) < len(sample_dataframe_with_nulls)
        # No nulls should remain in numeric columns
        assert not result.data.data["value1"].isna().any()
        assert not result.data.data["value2"].isna().any()

    @pytest.mark.asyncio
    async def test_drop_rows_with_nulls_subset(self, agent, sample_dataframe_with_nulls):
        """Test dropping rows with nulls only in specific columns."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[{"type": "drop_rows_with_nulls", "subset": ["value1"]}],
        )

        assert result.success
        assert not result.data.data["value1"].isna().any()
        # value2 might still have nulls

    @pytest.mark.asyncio
    async def test_drop_rows_with_nulls_how_all(self, agent):
        """Test dropping rows only if all values are null."""
        df = pd.DataFrame(
            {
                "a": [1, None, 3],
                "b": [None, None, 6],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "drop_rows_with_nulls", "how": "all"}],
        )

        assert result.success
        # Only row where ALL values are null should be dropped
        assert len(result.data.data) >= 2

    # -------------------------------------------------------------------------
    # Fill Nulls Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_fill_nulls_with_value(self, agent, sample_dataframe_with_nulls):
        """Test filling nulls with a specific value."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[{"type": "fill_nulls", "column": "value1", "method": "value", "value": 0}],
        )

        assert result.success
        assert not result.data.data["value1"].isna().any()
        # Check that some zeros were filled in
        assert (result.data.data["value1"] == 0).any()

    @pytest.mark.asyncio
    async def test_fill_nulls_with_mean(self, agent, sample_dataframe_with_nulls):
        """Test filling nulls with mean value."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[{"type": "fill_nulls", "column": "value1", "method": "mean"}],
        )

        assert result.success
        assert not result.data.data["value1"].isna().any()

    @pytest.mark.asyncio
    async def test_fill_nulls_with_median(self, agent, sample_dataframe_with_nulls):
        """Test filling nulls with median value."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[{"type": "fill_nulls", "column": "value1", "method": "median"}],
        )

        assert result.success
        assert not result.data.data["value1"].isna().any()

    @pytest.mark.asyncio
    async def test_fill_nulls_with_mode(self, agent):
        """Test filling nulls with mode value."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", None, None],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "fill_nulls", "column": "category", "method": "mode"}],
        )

        assert result.success
        assert not result.data.data["category"].isna().any()

    @pytest.mark.asyncio
    async def test_fill_nulls_ffill(self, agent):
        """Test forward fill for nulls."""
        df = pd.DataFrame(
            {
                "value": [1, None, None, 4, 5],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "fill_nulls", "column": "value", "method": "ffill"}],
        )

        assert result.success
        assert not result.data.data["value"].isna().any()
        assert result.data.data["value"].iloc[1] == 1  # Forward filled

    @pytest.mark.asyncio
    async def test_fill_nulls_bfill(self, agent):
        """Test backward fill for nulls."""
        df = pd.DataFrame(
            {
                "value": [1, None, None, 4, 5],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "fill_nulls", "column": "value", "method": "bfill"}],
        )

        assert result.success
        assert not result.data.data["value"].isna().any()
        assert result.data.data["value"].iloc[1] == 4  # Backward filled

    # -------------------------------------------------------------------------
    # Convert Datetime Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_convert_datetime(self, agent):
        """Test converting string column to datetime."""
        df = pd.DataFrame(
            {
                "date_str": ["2024-01-01", "2024-02-15", "2024-03-20"],
                "value": [1, 2, 3],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "convert_datetime", "column": "date_str"}],
        )

        assert result.success
        assert pd.api.types.is_datetime64_any_dtype(result.data.data["date_str"])

    @pytest.mark.asyncio
    async def test_convert_datetime_with_format(self, agent):
        """Test converting datetime with specific format."""
        df = pd.DataFrame(
            {
                "date_str": ["01/15/2024", "02/20/2024", "03/25/2024"],
                "value": [1, 2, 3],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "convert_datetime", "column": "date_str", "format": "%m/%d/%Y"}],
        )

        assert result.success
        assert pd.api.types.is_datetime64_any_dtype(result.data.data["date_str"])

    @pytest.mark.asyncio
    async def test_convert_datetime_invalid_column(self, agent, sample_dataframe):
        """Test converting non-existent column to datetime."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "convert_datetime", "column": "nonexistent"}],
        )

        assert result.success  # Should not fail, just skip

    # -------------------------------------------------------------------------
    # Convert Numeric Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_convert_numeric(self, agent):
        """Test converting string column to numeric."""
        df = pd.DataFrame(
            {
                "num_str": ["1.5", "2.7", "3.9"],
                "category": ["A", "B", "C"],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "convert_numeric", "column": "num_str"}],
        )

        assert result.success
        assert pd.api.types.is_numeric_dtype(result.data.data["num_str"])

    @pytest.mark.asyncio
    async def test_convert_numeric_with_errors(self, agent):
        """Test converting mixed string/numeric column."""
        df = pd.DataFrame(
            {
                "mixed": ["1.5", "invalid", "3.9"],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[{"type": "convert_numeric", "column": "mixed"}],
        )

        assert result.success
        # Invalid values should become NaN
        assert result.data.data["mixed"].isna().any()

    # -------------------------------------------------------------------------
    # Rename Columns Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rename_columns(self, agent, sample_dataframe):
        """Test renaming columns."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "rename_columns", "mapping": {"value1": "primary_value"}}],
        )

        assert result.success
        assert "primary_value" in result.data.data.columns
        assert "value1" not in result.data.data.columns

    @pytest.mark.asyncio
    async def test_rename_multiple_columns(self, agent, sample_dataframe):
        """Test renaming multiple columns."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {
                    "type": "rename_columns",
                    "mapping": {"value1": "primary", "value2": "secondary"},
                }
            ],
        )

        assert result.success
        assert "primary" in result.data.data.columns
        assert "secondary" in result.data.data.columns
        assert "value1" not in result.data.data.columns
        assert "value2" not in result.data.data.columns

    # -------------------------------------------------------------------------
    # Filter Rows Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_filter_rows_eq(self, agent, sample_dataframe):
        """Test filtering rows with equality operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "category", "operator": "eq", "value": "A"}
            ],
        )

        assert result.success
        assert all(result.data.data["category"] == "A")

    @pytest.mark.asyncio
    async def test_filter_rows_ne(self, agent, sample_dataframe):
        """Test filtering rows with not equal operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "category", "operator": "ne", "value": "A"}
            ],
        )

        assert result.success
        assert all(result.data.data["category"] != "A")

    @pytest.mark.asyncio
    async def test_filter_rows_gt(self, agent, sample_dataframe):
        """Test filtering rows with greater than operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "value1", "operator": "gt", "value": 100}
            ],
        )

        assert result.success
        assert all(result.data.data["value1"] > 100)

    @pytest.mark.asyncio
    async def test_filter_rows_gte(self, agent, sample_dataframe):
        """Test filtering rows with greater than or equal operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "value1", "operator": "gte", "value": 100}
            ],
        )

        assert result.success
        assert all(result.data.data["value1"] >= 100)

    @pytest.mark.asyncio
    async def test_filter_rows_lt(self, agent, sample_dataframe):
        """Test filtering rows with less than operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "value1", "operator": "lt", "value": 100}
            ],
        )

        assert result.success
        assert all(result.data.data["value1"] < 100)

    @pytest.mark.asyncio
    async def test_filter_rows_lte(self, agent, sample_dataframe):
        """Test filtering rows with less than or equal operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "value1", "operator": "lte", "value": 100}
            ],
        )

        assert result.success
        assert all(result.data.data["value1"] <= 100)

    @pytest.mark.asyncio
    async def test_filter_rows_in(self, agent, sample_dataframe):
        """Test filtering rows with 'in' operator."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "filter_rows", "column": "category", "operator": "in", "value": ["A", "B"]}
            ],
        )

        assert result.success
        assert all(result.data.data["category"].isin(["A", "B"]))

    @pytest.mark.asyncio
    async def test_filter_rows_contains(self, agent):
        """Test filtering rows with 'contains' operator."""
        df = pd.DataFrame(
            {
                "text": ["hello world", "goodbye", "hello there", "test"],
            }
        )

        result = await agent.execute(
            data=df,
            operations=[
                {"type": "filter_rows", "column": "text", "operator": "contains", "value": "hello"}
            ],
        )

        assert result.success
        assert len(result.data.data) == 2
        assert all("hello" in text for text in result.data.data["text"])

    # -------------------------------------------------------------------------
    # Create Column Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_column_year(self, agent, sample_dataframe):
        """Test creating year column from datetime."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "create_column", "name": "year", "source": "date", "transform": "year"}
            ],
        )

        assert result.success
        assert "year" in result.data.data.columns
        assert all(result.data.data["year"] == 2024)

    @pytest.mark.asyncio
    async def test_create_column_month(self, agent, sample_dataframe):
        """Test creating month column from datetime."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "create_column", "name": "month", "source": "date", "transform": "month"}
            ],
        )

        assert result.success
        assert "month" in result.data.data.columns

    @pytest.mark.asyncio
    async def test_create_column_dayofweek(self, agent, sample_dataframe):
        """Test creating day of week column from datetime."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {
                    "type": "create_column",
                    "name": "weekday",
                    "source": "date",
                    "transform": "dayofweek",
                }
            ],
        )

        assert result.success
        assert "weekday" in result.data.data.columns
        assert all(result.data.data["weekday"].between(0, 6))

    @pytest.mark.asyncio
    async def test_create_column_lower(self, agent, sample_dataframe):
        """Test creating lowercase column from string."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {
                    "type": "create_column",
                    "name": "category_lower",
                    "source": "category",
                    "transform": "lower",
                }
            ],
        )

        assert result.success
        assert "category_lower" in result.data.data.columns
        assert all(c.islower() for c in result.data.data["category_lower"].dropna())

    @pytest.mark.asyncio
    async def test_create_column_upper(self, agent, sample_dataframe):
        """Test creating uppercase column from string."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {
                    "type": "create_column",
                    "name": "category_upper",
                    "source": "category",
                    "transform": "upper",
                }
            ],
        )

        assert result.success
        assert "category_upper" in result.data.data.columns
        assert all(c.isupper() for c in result.data.data["category_upper"].dropna())

    @pytest.mark.asyncio
    async def test_create_column_len(self, agent, sample_dataframe):
        """Test creating length column from string."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {
                    "type": "create_column",
                    "name": "category_len",
                    "source": "category",
                    "transform": "len",
                }
            ],
        )

        assert result.success
        assert "category_len" in result.data.data.columns
        # All categories are single letters (A, B, C)
        assert all(result.data.data["category_len"] == 1)

    # -------------------------------------------------------------------------
    # Flag Column Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_flag_column(self, agent, sample_dataframe):
        """Test flagging a column generates a warning."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "flag_column", "column": "value1", "reason": "High variance detected"}
            ],
        )

        assert result.success
        assert len(result.data.warnings) > 0
        assert any("value1" in w for w in result.data.warnings)

    # -------------------------------------------------------------------------
    # Unknown Operation Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unknown_operation_type(self, agent, sample_dataframe):
        """Test that unknown operation type generates a warning."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "unknown_operation"}],
        )

        assert result.success
        assert len(result.data.warnings) > 0
        assert any("unknown" in w.lower() for w in result.data.warnings)

    # -------------------------------------------------------------------------
    # Multiple Operations Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_operations(self, agent, sample_dataframe_with_nulls):
        """Test applying multiple operations in sequence."""
        result = await agent.execute(
            data=sample_dataframe_with_nulls,
            operations=[
                {"type": "fill_nulls", "column": "value1", "method": "mean"},
                {"type": "rename_columns", "mapping": {"value1": "primary_value"}},
                {"type": "filter_rows", "column": "primary_value", "operator": "gt", "value": 90},
            ],
        )

        assert result.success
        assert "primary_value" in result.data.data.columns
        assert "value1" not in result.data.data.columns
        assert not result.data.data["primary_value"].isna().any()
        assert all(result.data.data["primary_value"] > 90)

    @pytest.mark.asyncio
    async def test_operation_error_handling(self, agent, sample_dataframe):
        """Test that errors in one operation don't stop others."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "convert_numeric", "column": "category"},  # Will create NaNs
                {"type": "rename_columns", "mapping": {"value1": "primary"}},  # Should still work
            ],
        )

        assert result.success
        assert "primary" in result.data.data.columns

    # -------------------------------------------------------------------------
    # Transformation Log Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_transformation_log_recorded(self, agent, sample_dataframe):
        """Test that transformation log is recorded."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["value1"]}],
        )

        assert result.success
        assert result.data.log is not None
        assert len(result.data.log.steps) >= 1

    @pytest.mark.asyncio
    async def test_transformation_log_shapes(self, agent, sample_dataframe):
        """Test that transformation log tracks shape changes."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[{"type": "drop_columns", "columns": ["value1", "value2"]}],
        )

        assert result.success
        log = result.data.log
        assert log.original_shape[1] > log.final_shape[1]
        assert log.original_shape[0] == log.final_shape[0]  # Rows unchanged

    @pytest.mark.asyncio
    async def test_transformation_log_columns(self, agent, sample_dataframe):
        """Test that transformation log tracks column changes."""
        result = await agent.execute(
            data=sample_dataframe,
            operations=[
                {"type": "drop_columns", "columns": ["value1"]},
                {"type": "create_column", "name": "new_col", "source": "date", "transform": "year"},
            ],
        )

        assert result.success
        log = result.data.log
        assert "value1" in log.original_columns
        assert "value1" not in log.final_columns
        assert "new_col" in log.final_columns

    def test_format_log_output(self, agent):
        """Test formatting transformation log for display."""
        log = TransformationLog(
            original_shape=(100, 5),
            final_shape=(90, 4),
            original_columns=["a", "b", "c", "d", "e"],
            final_columns=["a", "b", "d", "new"],
        )
        log.add_step(
            TransformationStep(
                operation="drop_duplicates",
                description="Dropped duplicate rows",
                rows_before=100,
                rows_after=90,
                columns_before=5,
                columns_after=5,
            )
        )
        log.add_step(
            TransformationStep(
                operation="drop_columns",
                description="Dropped columns",
                rows_before=90,
                rows_after=90,
                columns_before=5,
                columns_after=4,
            )
        )

        formatted = agent.format_log_output(log)

        assert "Transformation Log" in formatted
        assert "100" in formatted  # Original rows
        assert "90" in formatted  # Final rows
        assert "drop_duplicates" in formatted
        assert "Columns Added" in formatted or "Columns Removed" in formatted


class TestTransformationStep:
    """Test suite for TransformationStep dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        step = TransformationStep(
            operation="drop_duplicates",
            description="Dropped duplicate rows",
            rows_before=100,
            rows_after=90,
            columns_before=5,
            columns_after=5,
            details={"subset": None},
        )

        d = step.to_dict()

        assert d["operation"] == "drop_duplicates"
        assert d["rows_before"] == 100
        assert d["rows_after"] == 90
        assert "timestamp" in d


class TestTransformationLog:
    """Test suite for TransformationLog dataclass."""

    def test_add_step(self):
        """Test adding a step to the log."""
        log = TransformationLog(original_shape=(100, 5))

        step = TransformationStep(
            operation="test",
            description="Test step",
            rows_before=100,
            rows_after=95,
            columns_before=5,
            columns_after=5,
        )
        log.add_step(step)

        assert len(log.steps) == 1
        assert log.final_shape == (95, 5)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        log = TransformationLog(
            original_shape=(100, 5),
            final_shape=(90, 4),
            original_columns=["a", "b", "c", "d", "e"],
            final_columns=["a", "b", "c", "d"],
        )

        d = log.to_dict()

        assert d["original_shape"] == (100, 5)
        assert d["final_shape"] == (90, 4)
        assert d["total_rows_removed"] == 10
        assert d["total_columns_changed"] == -1


class TestTransformResult:
    """Test suite for TransformResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary (excluding DataFrame)."""
        log = TransformationLog(original_shape=(100, 5), final_shape=(90, 4))
        result = TransformResult(
            data=pd.DataFrame({"a": [1, 2, 3]}),
            log=log,
            warnings=["Warning 1", "Warning 2"],
        )

        d = result.to_dict()

        assert "log" in d
        assert "warnings" in d
        assert len(d["warnings"]) == 2
