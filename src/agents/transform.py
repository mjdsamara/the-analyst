"""
Transform Agent for The Analyst platform.

Handles data cleaning, reshaping, and preparation for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.agents.base import AgentContext, AgentResult, BaseAgent
from src.prompts.agents import TRANSFORM_PROMPT


@dataclass
class TransformationStep:
    """Record of a single transformation operation."""

    operation: str
    description: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "description": self.description,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class TransformationLog:
    """Complete log of all transformations applied."""

    steps: list[TransformationStep] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)
    final_shape: tuple[int, int] = (0, 0)
    original_columns: list[str] = field(default_factory=list)
    final_columns: list[str] = field(default_factory=list)

    def add_step(self, step: TransformationStep) -> None:
        """Add a transformation step to the log."""
        self.steps.append(step)
        self.final_shape = (step.rows_after, step.columns_after)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "original_columns": self.original_columns,
            "final_columns": self.final_columns,
            "total_rows_removed": self.original_shape[0] - self.final_shape[0],
            "total_columns_changed": self.final_shape[1] - self.original_shape[1],
        }


@dataclass
class TransformResult:
    """Result from the transform agent."""

    data: pd.DataFrame
    log: TransformationLog
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding DataFrame)."""
        return {
            "log": self.log.to_dict(),
            "warnings": self.warnings,
        }


class TransformAgent(BaseAgent):
    """
    Agent responsible for data cleaning and transformation.

    Single Job: Clean, reshape, and prepare data for analysis.
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the transform agent."""
        super().__init__(name="transform", context=context)
        self._transformation_log: TransformationLog | None = None

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return TRANSFORM_PROMPT

    async def execute(
        self,
        data: pd.DataFrame | None = None,
        operations: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AgentResult[TransformResult]:
        """
        Execute data transformations.

        Args:
            data: DataFrame to transform
            operations: List of transformation operations to apply

        Returns:
            AgentResult containing transformed data and transformation log
        """
        if data is None:
            return AgentResult.error_result("No data provided for transformation")

        # Initialize log
        self._transformation_log = TransformationLog(
            original_shape=(len(data), len(data.columns)),
            original_columns=list(data.columns),
        )
        self._transformation_log.final_shape = self._transformation_log.original_shape
        self._transformation_log.final_columns = list(data.columns)

        # Make a copy to avoid modifying original
        df = data.copy()
        warnings: list[str] = []

        self.log(f"Starting transformation: {len(df)} rows, {len(df.columns)} columns")

        # If no operations specified, perform auto-cleaning
        if not operations:
            operations = self._generate_auto_operations(df)
            self.log(f"Generated {len(operations)} auto-cleaning operations")

        # Apply each operation
        for op in operations:
            try:
                df, step_warnings = await self._apply_operation(df, op)
                warnings.extend(step_warnings)
            except Exception as e:
                self.log(f"Error in operation {op.get('type')}: {e}", level="ERROR")
                warnings.append(f"Skipped operation {op.get('type')}: {e}")

        # Update final state
        self._transformation_log.final_shape = (len(df), len(df.columns))
        self._transformation_log.final_columns = list(df.columns)

        result = TransformResult(
            data=df,
            log=self._transformation_log,
            warnings=warnings,
        )

        self.log(
            f"Transformation complete: {len(df)} rows, {len(df.columns)} columns, "
            f"{len(self._transformation_log.steps)} steps"
        )

        return AgentResult.success_result(result)

    def _generate_auto_operations(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate automatic cleaning operations based on data analysis."""
        operations: list[dict[str, Any]] = []

        # Check for duplicates
        if df.duplicated().any():
            operations.append({"type": "drop_duplicates"})

        # Check for columns with all nulls
        all_null_cols = [col for col in df.columns if df[col].isna().all()]
        if all_null_cols:
            operations.append({"type": "drop_columns", "columns": all_null_cols})

        # Check for string columns that should be datetime
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample, errors="raise")
                operations.append({"type": "convert_datetime", "column": col})
            except Exception:
                pass

        # Check for high missing value columns
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 50:
                operations.append(
                    {
                        "type": "flag_column",
                        "column": col,
                        "reason": f"High missing rate: {missing_pct:.1f}%",
                    }
                )

        return operations

    async def _apply_operation(
        self, df: pd.DataFrame, op: dict[str, Any]
    ) -> tuple[pd.DataFrame, list[str]]:
        """Apply a single transformation operation."""
        op_type = op.get("type", "")
        warnings: list[str] = []
        rows_before = len(df)
        cols_before = len(df.columns)

        if op_type == "drop_duplicates":
            df = self._drop_duplicates(df, op)
        elif op_type == "drop_columns":
            df = self._drop_columns(df, op)
        elif op_type == "drop_rows_with_nulls":
            df = self._drop_rows_with_nulls(df, op)
        elif op_type == "fill_nulls":
            df = self._fill_nulls(df, op)
        elif op_type == "convert_datetime":
            df = self._convert_datetime(df, op)
        elif op_type == "convert_numeric":
            df = self._convert_numeric(df, op)
        elif op_type == "rename_columns":
            df = self._rename_columns(df, op)
        elif op_type == "filter_rows":
            df = self._filter_rows(df, op)
        elif op_type == "create_column":
            df = self._create_column(df, op)
        elif op_type == "flag_column":
            # Just log, don't modify
            self.log(f"Flagged column {op.get('column')}: {op.get('reason')}")
            warnings.append(f"Column '{op.get('column')}' flagged: {op.get('reason')}")
        else:
            warnings.append(f"Unknown operation type: {op_type}")

        # Record step
        step = TransformationStep(
            operation=op_type,
            description=op.get("description", f"Applied {op_type}"),
            rows_before=rows_before,
            rows_after=len(df),
            columns_before=cols_before,
            columns_after=len(df.columns),
            details=op,
        )
        if self._transformation_log:
            self._transformation_log.add_step(step)

        return df, warnings

    def _drop_duplicates(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Drop duplicate rows."""
        subset = op.get("subset")
        keep = op.get("keep", "first")
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        self.log(f"Dropped {before - len(df)} duplicate rows")
        return df

    def _drop_columns(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Drop specified columns."""
        columns = op.get("columns", [])
        df = df.drop(columns=columns, errors="ignore")
        self.log(f"Dropped columns: {columns}")
        return df

    def _drop_rows_with_nulls(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Drop rows with null values."""
        subset = op.get("subset")
        how = op.get("how", "any")
        thresh = op.get("thresh")
        before = len(df)
        # thresh and how are mutually exclusive in pandas - use thresh if provided
        if thresh is not None:
            df = df.dropna(subset=subset, thresh=thresh)
        else:
            df = df.dropna(subset=subset, how=how)
        self.log(f"Dropped {before - len(df)} rows with nulls")
        return df

    def _fill_nulls(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Fill null values."""
        column = op.get("column")
        method = op.get("method", "value")
        value = op.get("value")

        if column:
            if method == "value" and value is not None:
                df[column] = df[column].fillna(value)
            elif method == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif method == "median":
                df[column] = df[column].fillna(df[column].median())
            elif method == "mode":
                df[column] = df[column].fillna(df[column].mode().iloc[0])
            elif method == "ffill":
                df[column] = df[column].ffill()
            elif method == "bfill":
                df[column] = df[column].bfill()
            self.log(f"Filled nulls in '{column}' using {method}")
        else:
            if value is not None:
                df = df.fillna(value)
            self.log(f"Filled all nulls with {value}")

        return df

    def _convert_datetime(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Convert column to datetime."""
        column = op.get("column")
        format_str = op.get("format")
        if column and column in df.columns:
            df[column] = pd.to_datetime(df[column], format=format_str, errors="coerce")
            self.log(f"Converted '{column}' to datetime")
        return df

    def _convert_numeric(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Convert column to numeric."""
        column = op.get("column")
        if column and column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            self.log(f"Converted '{column}' to numeric")
        return df

    def _rename_columns(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Rename columns."""
        mapping = op.get("mapping", {})
        df = df.rename(columns=mapping)
        self.log(f"Renamed columns: {mapping}")
        return df

    def _filter_rows(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Filter rows based on condition."""
        column = op.get("column")
        operator = op.get("operator")
        value = op.get("value")
        before = len(df)

        if column and column in df.columns and operator and value is not None:
            if operator == "eq":
                df = df[df[column] == value]
            elif operator == "ne":
                df = df[df[column] != value]
            elif operator == "gt":
                df = df[df[column] > value]
            elif operator == "gte":
                df = df[df[column] >= value]
            elif operator == "lt":
                df = df[df[column] < value]
            elif operator == "lte":
                df = df[df[column] <= value]
            elif operator == "in":
                df = df[df[column].isin(value)]
            elif operator == "contains":
                df = df[df[column].str.contains(value, na=False)]

            self.log(f"Filtered rows: {column} {operator} {value} ({before} -> {len(df)})")

        return df

    def _create_column(self, df: pd.DataFrame, op: dict[str, Any]) -> pd.DataFrame:
        """Create a new column."""
        name = op.get("name")
        source = op.get("source")
        transform = op.get("transform")

        if name and source and source in df.columns:
            if transform == "year":
                df[name] = pd.to_datetime(df[source]).dt.year
            elif transform == "month":
                df[name] = pd.to_datetime(df[source]).dt.month
            elif transform == "day":
                df[name] = pd.to_datetime(df[source]).dt.day
            elif transform == "dayofweek":
                df[name] = pd.to_datetime(df[source]).dt.dayofweek
            elif transform == "hour":
                df[name] = pd.to_datetime(df[source]).dt.hour
            elif transform == "lower":
                df[name] = df[source].str.lower()
            elif transform == "upper":
                df[name] = df[source].str.upper()
            elif transform == "len":
                df[name] = df[source].str.len()

            self.log(f"Created column '{name}' from '{source}' using {transform}")

        return df

    def format_log_output(self, log: TransformationLog) -> str:
        """
        Format transformation log for display.

        Args:
            log: The transformation log to format

        Returns:
            Formatted string output
        """
        lines = [
            "## Transformation Log",
            "",
            "### Summary",
            f"- **Original Shape**: {log.original_shape[0]:,} rows x {log.original_shape[1]} columns",
            f"- **Final Shape**: {log.final_shape[0]:,} rows x {log.final_shape[1]} columns",
            f"- **Rows Changed**: {log.original_shape[0] - log.final_shape[0]:,}",
            f"- **Total Steps**: {len(log.steps)}",
            "",
            "### Steps Applied",
        ]

        for i, step in enumerate(log.steps, 1):
            row_change = step.rows_after - step.rows_before
            row_str = f"+{row_change}" if row_change >= 0 else str(row_change)
            lines.append(f"{i}. **{step.operation}**: {step.description} " f"(rows: {row_str})")

        if log.original_columns != log.final_columns:
            added = set(log.final_columns) - set(log.original_columns)
            removed = set(log.original_columns) - set(log.final_columns)

            if added:
                lines.extend(["", "### Columns Added"])
                for col in added:
                    lines.append(f"- {col}")

            if removed:
                lines.extend(["", "### Columns Removed"])
                for col in removed:
                    lines.append(f"- {col}")

        return "\n".join(lines)
