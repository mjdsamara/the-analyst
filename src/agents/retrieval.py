"""
Retrieval Agent for The Analyst platform.

Handles data ingestion from various file formats with validation and profiling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.agents.base import AgentContext, AgentResult, BaseAgent
from src.config import get_settings
from src.prompts.agents import RETRIEVAL_PROMPT


@dataclass
class DataProfile:
    """Profile of a loaded dataset."""

    filename: str
    row_count: int
    column_count: int
    columns: list[dict[str, Any]]
    missing_summary: dict[str, int]
    sample_rows: list[dict[str, Any]]
    checksum: str
    file_size_bytes: int
    memory_usage_mb: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "filename": self.filename,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "missing_summary": self.missing_summary,
            "sample_rows": self.sample_rows,
            "checksum": self.checksum,
            "file_size_bytes": self.file_size_bytes,
            "memory_usage_mb": self.memory_usage_mb,
        }


@dataclass
class QualityReport:
    """Data quality assessment report."""

    completeness: float  # Percentage of non-null values
    duplicate_rows: int
    duplicate_percentage: float
    type_issues: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "completeness": self.completeness,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_percentage": self.duplicate_percentage,
            "type_issues": self.type_issues,
            "warnings": self.warnings,
        }


@dataclass
class RetrievalResult:
    """Result from the retrieval agent."""

    data: pd.DataFrame
    profile: DataProfile
    quality: QualityReport

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding DataFrame)."""
        return {
            "profile": self.profile.to_dict(),
            "quality": self.quality.to_dict(),
        }


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for data ingestion from files and APIs.

    Single Job: Ingest data from files and APIs, validate integrity, provide profiles.
    """

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    CHUNK_THRESHOLD_MB = 100

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the retrieval agent."""
        super().__init__(name="retrieval", context=context)
        self._loaded_data: dict[str, pd.DataFrame] = {}

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return RETRIEVAL_PROMPT

    async def execute(
        self,
        file_path: str | Path | None = None,
        **kwargs: Any,
    ) -> AgentResult[RetrievalResult]:
        """
        Execute data retrieval.

        Args:
            file_path: Path to the data file to load

        Returns:
            AgentResult containing the loaded data, profile, and quality report
        """
        if file_path is None:
            return AgentResult.error_result("No file path provided")

        path = Path(file_path)
        self.log(f"Loading data from: {path}")

        # Validate file exists
        if not path.exists():
            return AgentResult.error_result(f"File not found: {path}")

        # Validate format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return AgentResult.error_result(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        try:
            # Load the data
            df = await self._load_file(path)

            # Generate profile
            profile = self._generate_profile(df, path)

            # Generate quality report
            quality = self._generate_quality_report(df)

            # Store loaded data
            self._loaded_data[str(path)] = df

            # Create result
            result = RetrievalResult(data=df, profile=profile, quality=quality)

            self.log(
                f"Successfully loaded {profile.row_count} rows, " f"{profile.column_count} columns"
            )

            return AgentResult.success_result(
                result,
                file_path=str(path),
                checksum=profile.checksum,
            )

        except Exception as e:
            self.log(f"Error loading file: {e}", level="ERROR")
            return AgentResult.error_result(f"Failed to load file: {e}")

    async def _load_file(self, path: Path) -> pd.DataFrame:
        """
        Load a file into a DataFrame.

        Args:
            path: Path to the file

        Returns:
            Loaded DataFrame
        """
        _settings = get_settings()  # Reserved for future constraint checks
        file_size_mb = path.stat().st_size / (1024 * 1024)

        # Check if chunked loading is needed
        use_chunks = file_size_mb > self.CHUNK_THRESHOLD_MB

        suffix = path.suffix.lower()

        if suffix == ".csv":
            if use_chunks:
                self.log(f"Large file ({file_size_mb:.1f}MB), using chunked loading")
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100000):
                    chunks.append(chunk)
                return cast(pd.DataFrame, pd.concat(chunks, ignore_index=True))
            return pd.read_csv(path)

        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)

        elif suffix == ".json":
            return pd.read_json(path)

        elif suffix == ".parquet":
            return pd.read_parquet(path)

        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _generate_profile(self, df: pd.DataFrame, path: Path) -> DataProfile:
        """
        Generate a profile of the loaded data.

        Args:
            df: The loaded DataFrame
            path: Path to the source file

        Returns:
            DataProfile object
        """
        # Column information
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique()),
            }

            # Add sample values
            sample_vals = df[col].dropna().head(3).tolist()
            col_info["sample_values"] = [str(v)[:50] for v in sample_vals]

            columns.append(col_info)

        # Missing value summary
        missing_summary = {
            col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()
        }

        # Sample rows
        sample_rows = cast(list[dict[str, Any]], df.head(5).to_dict(orient="records"))

        # Compute checksum
        checksum = self._compute_file_checksum(path)

        # File size
        file_size = path.stat().st_size

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        return DataProfile(
            filename=path.name,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            missing_summary=missing_summary,
            sample_rows=sample_rows,
            checksum=checksum,
            file_size_bytes=file_size,
            memory_usage_mb=round(memory_mb, 2),
        )

    def _generate_quality_report(self, df: pd.DataFrame) -> QualityReport:
        """
        Generate a quality assessment report.

        Args:
            df: The loaded DataFrame

        Returns:
            QualityReport object
        """
        # Completeness
        total_cells = df.size
        non_null_cells = df.notna().sum().sum()
        completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0

        # Duplicates
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0

        # Type issues
        type_issues = []
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if column could be numeric
                numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
                if numeric_count > len(df) * 0.5:
                    type_issues.append(
                        {
                            "column": col,
                            "issue": "Potential numeric column stored as string",
                            "numeric_values": int(numeric_count),
                        }
                    )

                # Check if column could be datetime
                try:
                    datetime_count = pd.to_datetime(df[col], errors="coerce").notna().sum()
                    if datetime_count > len(df) * 0.5:
                        type_issues.append(
                            {
                                "column": col,
                                "issue": "Potential datetime column stored as string",
                                "datetime_values": int(datetime_count),
                            }
                        )
                except Exception:
                    pass

        # Warnings
        warnings = []
        if completeness < 80:
            warnings.append(f"Low data completeness: {completeness:.1f}%")
        if duplicate_percentage > 5:
            warnings.append(f"High duplicate rate: {duplicate_percentage:.1f}%")
        if type_issues:
            warnings.append(f"Found {len(type_issues)} potential type issues")

        return QualityReport(
            completeness=round(completeness, 2),
            duplicate_rows=int(duplicate_rows),
            duplicate_percentage=round(duplicate_percentage, 2),
            type_issues=type_issues,
            warnings=warnings,
        )

    def _compute_file_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_loaded_data(self, file_path: str | Path) -> pd.DataFrame | None:
        """
        Get previously loaded data.

        Args:
            file_path: Path to the loaded file

        Returns:
            DataFrame if loaded, None otherwise
        """
        return self._loaded_data.get(str(file_path))

    def format_profile_output(self, profile: DataProfile) -> str:
        """
        Format a data profile for display.

        Args:
            profile: The data profile to format

        Returns:
            Formatted string output
        """
        lines = [
            f"## Data Profile: {profile.filename}",
            "",
            "### Overview",
            f"- **Rows**: {profile.row_count:,}",
            f"- **Columns**: {profile.column_count}",
            f"- **File Size**: {profile.file_size_bytes / 1024:.1f} KB",
            f"- **Memory Usage**: {profile.memory_usage_mb:.2f} MB",
            f"- **Checksum**: `{profile.checksum[:16]}...`",
            "",
            "### Columns",
            "| Column | Type | Non-Null | Null | Unique |",
            "|--------|------|----------|------|--------|",
        ]

        for col in profile.columns:
            lines.append(
                f"| {col['name']} | {col['dtype']} | "
                f"{col['non_null_count']:,} | {col['null_count']:,} | "
                f"{col['unique_count']:,} |"
            )

        if profile.missing_summary:
            lines.extend(
                [
                    "",
                    "### Missing Values",
                ]
            )
            for col_name, count in profile.missing_summary.items():
                pct = count / profile.row_count * 100
                lines.append(f"- **{col_name}**: {count:,} ({pct:.1f}%)")

        lines.extend(
            [
                "",
                "### Sample Data (First 5 Rows)",
                "```",
            ]
        )
        for i, row in enumerate(profile.sample_rows[:5]):
            lines.append(f"Row {i+1}: {row}")
        lines.append("```")

        return "\n".join(lines)

    def format_quality_output(self, quality: QualityReport) -> str:
        """
        Format a quality report for display.

        Args:
            quality: The quality report to format

        Returns:
            Formatted string output
        """
        lines = [
            "## Data Quality Report",
            "",
            "### Summary",
            f"- **Completeness**: {quality.completeness:.1f}%",
            f"- **Duplicate Rows**: {quality.duplicate_rows:,} ({quality.duplicate_percentage:.1f}%)",
        ]

        if quality.type_issues:
            lines.extend(
                [
                    "",
                    "### Type Issues",
                ]
            )
            for issue in quality.type_issues:
                lines.append(f"- **{issue['column']}**: {issue['issue']}")

        if quality.warnings:
            lines.extend(
                [
                    "",
                    "### Warnings",
                ]
            )
            for warning in quality.warnings:
                lines.append(f"- {warning}")
        else:
            lines.extend(
                [
                    "",
                    "No quality warnings detected.",
                ]
            )

        return "\n".join(lines)
