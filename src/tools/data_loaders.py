"""
Data loading utilities for The Analyst platform.

Supports CSV, Excel, JSON, and Parquet formats with validation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.config import get_settings


class DataLoader:
    """
    Utility class for loading data from various file formats.

    Supports:
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - JSON files (.json)
    - Parquet files (.parquet)
    """

    SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
    CHUNK_SIZE = 100000
    LARGE_FILE_THRESHOLD_MB = 100

    def __init__(self) -> None:
        """Initialize the data loader."""
        self.settings = get_settings()

    def load(
        self,
        file_path: str | Path,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from a file.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to the loader

        Returns:
            Loaded DataFrame

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Check file size for chunked loading
        file_size_mb = path.stat().st_size / (1024 * 1024)
        use_chunks = file_size_mb > self.LARGE_FILE_THRESHOLD_MB

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(path, use_chunks, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            return self._load_excel(path, **kwargs)
        elif suffix == ".json":
            return self._load_json(path, **kwargs)
        elif suffix == ".parquet":
            return self._load_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _load_csv(
        self,
        path: Path,
        use_chunks: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load CSV file."""
        if use_chunks:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.CHUNK_SIZE, **kwargs):
                chunks.append(chunk)
            return cast(pd.DataFrame, pd.concat(chunks, ignore_index=True))
        return cast(pd.DataFrame, pd.read_csv(path, **kwargs))

    def _load_excel(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load Excel file."""
        return cast(pd.DataFrame, pd.read_excel(path, **kwargs))

    def _load_json(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load JSON file."""
        return cast(pd.DataFrame, pd.read_json(path, **kwargs))

    def _load_parquet(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(path, **kwargs)

    def load_chunked(
        self,
        file_path: str | Path,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks for large files.

        Args:
            file_path: Path to the data file
            chunk_size: Number of rows per chunk
            **kwargs: Additional arguments

        Yields:
            DataFrame chunks
        """
        path = Path(file_path)
        chunk_size = chunk_size or self.CHUNK_SIZE

        if path.suffix.lower() == ".csv":
            yield from pd.read_csv(path, chunksize=chunk_size, **kwargs)
        elif path.suffix.lower() == ".parquet":
            # Parquet chunked reading
            df = pd.read_parquet(path, **kwargs)
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i : i + chunk_size]
        else:
            # For other formats, load and yield in chunks
            df = self.load(path, **kwargs)
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i : i + chunk_size]

    def compute_checksum(self, file_path: str | Path) -> str:
        """
        Compute SHA-256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal checksum string
        """
        path = Path(file_path)
        sha256 = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def validate_checksum(
        self,
        file_path: str | Path,
        expected_checksum: str,
    ) -> bool:
        """
        Validate file integrity against expected checksum.

        Args:
            file_path: Path to the file
            expected_checksum: Expected checksum

        Returns:
            True if checksums match
        """
        actual = self.compute_checksum(file_path)
        return actual == expected_checksum

    def get_file_info(self, file_path: str | Path) -> dict[str, Any]:
        """
        Get basic information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        stat = path.stat()

        return {
            "name": path.name,
            "path": str(path.absolute()),
            "format": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_at": stat.st_mtime,
            "checksum": self.compute_checksum(path),
        }
