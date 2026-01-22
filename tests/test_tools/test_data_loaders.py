"""Tests for data loading utilities."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.data_loaders import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance for testing."""
        return DataLoader()

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "id": range(1, 101),
                "value": np.random.normal(100, 15, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # -------------------------------------------------------------------------
    # CSV Loading Tests
    # -------------------------------------------------------------------------

    def test_load_csv(self, loader, sample_df, temp_dir):
        """Test loading a CSV file."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert list(result.columns) == list(sample_df.columns)

    def test_load_csv_with_path_string(self, loader, sample_df, temp_dir):
        """Test loading a CSV file with string path."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        result = loader.load(str(file_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_load_csv_with_kwargs(self, loader, sample_df, temp_dir):
        """Test loading a CSV file with additional kwargs."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        result = loader.load(file_path, usecols=["id", "value"])

        assert "category" not in result.columns
        assert "id" in result.columns
        assert "value" in result.columns

    # -------------------------------------------------------------------------
    # Excel Loading Tests
    # -------------------------------------------------------------------------

    def test_load_excel_xlsx(self, loader, sample_df, temp_dir):
        """Test loading an Excel .xlsx file."""
        file_path = temp_dir / "test.xlsx"
        sample_df.to_excel(file_path, index=False)

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    @pytest.mark.skip(reason="xlwt engine not available in this environment")
    def test_load_excel_xls(self, loader, sample_df, temp_dir):
        """Test loading an Excel .xls file."""
        file_path = temp_dir / "test.xls"
        # Use xlwt engine for .xls format
        sample_df.to_excel(file_path, index=False, engine="xlwt")

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    # -------------------------------------------------------------------------
    # JSON Loading Tests
    # -------------------------------------------------------------------------

    def test_load_json(self, loader, sample_df, temp_dir):
        """Test loading a JSON file."""
        file_path = temp_dir / "test.json"
        sample_df.to_json(file_path, orient="records")

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_load_json_with_orient(self, loader, sample_df, temp_dir):
        """Test loading a JSON file with specific orient."""
        file_path = temp_dir / "test.json"
        sample_df.to_json(file_path, orient="columns")

        result = loader.load(file_path, orient="columns")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    # -------------------------------------------------------------------------
    # Parquet Loading Tests
    # -------------------------------------------------------------------------

    def test_load_parquet(self, loader, sample_df, temp_dir):
        """Test loading a Parquet file."""
        file_path = temp_dir / "test.parquet"
        sample_df.to_parquet(file_path, index=False)

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_load_parquet_with_columns(self, loader, sample_df, temp_dir):
        """Test loading specific columns from Parquet file."""
        file_path = temp_dir / "test.parquet"
        sample_df.to_parquet(file_path, index=False)

        result = loader.load(file_path, columns=["id", "value"])

        assert "category" not in result.columns
        assert "id" in result.columns

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_load_file_not_found(self, loader):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load("/nonexistent/path/file.csv")

    def test_load_unsupported_format(self, loader, temp_dir):
        """Test that ValueError is raised for unsupported formats."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(file_path)

    def test_load_unsupported_format_includes_supported_list(self, loader, temp_dir):
        """Test that error message includes list of supported formats."""
        file_path = temp_dir / "test.xml"
        file_path.write_text("<data></data>")

        with pytest.raises(ValueError) as exc_info:
            loader.load(file_path)

        assert ".csv" in str(exc_info.value)
        assert ".xlsx" in str(exc_info.value)
        assert ".json" in str(exc_info.value)
        assert ".parquet" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # Chunked Loading Tests
    # -------------------------------------------------------------------------

    def test_load_chunked_csv(self, loader, sample_df, temp_dir):
        """Test chunked loading of CSV file."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        chunks = list(loader.load_chunked(file_path, chunk_size=25))

        # Should have 4 chunks of ~25 rows each
        assert len(chunks) == 4
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == len(sample_df)

    def test_load_chunked_parquet(self, loader, sample_df, temp_dir):
        """Test chunked loading of Parquet file."""
        file_path = temp_dir / "test.parquet"
        sample_df.to_parquet(file_path, index=False)

        chunks = list(loader.load_chunked(file_path, chunk_size=30))

        # Should have 4 chunks (100 / 30 = 4 with remainder)
        assert len(chunks) == 4
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == len(sample_df)

    def test_load_chunked_excel(self, loader, sample_df, temp_dir):
        """Test chunked loading of Excel file (via load and slice)."""
        file_path = temp_dir / "test.xlsx"
        sample_df.to_excel(file_path, index=False)

        chunks = list(loader.load_chunked(file_path, chunk_size=50))

        # Should have 2 chunks of 50 rows each
        assert len(chunks) == 2
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == len(sample_df)

    def test_load_chunked_default_chunk_size(self, loader, sample_df, temp_dir):
        """Test that default chunk size is used when not specified."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        chunks = list(loader.load_chunked(file_path))

        # With 100 rows and default chunk size of 100000, should get 1 chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == len(sample_df)

    # -------------------------------------------------------------------------
    # Checksum Tests
    # -------------------------------------------------------------------------

    def test_compute_checksum(self, loader, temp_dir):
        """Test computing SHA-256 checksum."""
        file_path = temp_dir / "test.txt"
        content = b"Hello, World!"
        file_path.write_bytes(content)

        result = loader.compute_checksum(file_path)

        # Verify it's a valid SHA-256 hex string
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

        # Verify it matches expected value
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_compute_checksum_with_string_path(self, loader, temp_dir):
        """Test computing checksum with string path."""
        file_path = temp_dir / "test.txt"
        file_path.write_bytes(b"test content")

        result = loader.compute_checksum(str(file_path))

        assert len(result) == 64

    def test_compute_checksum_large_file(self, loader, temp_dir):
        """Test checksum computation for larger files (chunked reading)."""
        file_path = temp_dir / "large.bin"
        # Create a file larger than the 8192 byte read buffer
        content = b"x" * 50000
        file_path.write_bytes(content)

        result = loader.compute_checksum(file_path)

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_validate_checksum_valid(self, loader, temp_dir):
        """Test validating checksum with correct value."""
        file_path = temp_dir / "test.txt"
        content = b"test data"
        file_path.write_bytes(content)
        expected_checksum = hashlib.sha256(content).hexdigest()

        result = loader.validate_checksum(file_path, expected_checksum)

        assert result is True

    def test_validate_checksum_invalid(self, loader, temp_dir):
        """Test validating checksum with incorrect value."""
        file_path = temp_dir / "test.txt"
        file_path.write_bytes(b"test data")

        result = loader.validate_checksum(file_path, "invalid_checksum")

        assert result is False

    # -------------------------------------------------------------------------
    # File Info Tests
    # -------------------------------------------------------------------------

    def test_get_file_info(self, loader, sample_df, temp_dir):
        """Test getting file information."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        result = loader.get_file_info(file_path)

        assert result["name"] == "test.csv"
        assert result["format"] == ".csv"
        assert result["size_bytes"] > 0
        assert result["size_mb"] > 0
        assert "checksum" in result
        assert "modified_at" in result
        assert str(temp_dir) in result["path"]

    def test_get_file_info_different_formats(self, loader, sample_df, temp_dir):
        """Test file info for different formats."""
        formats = [".csv", ".json", ".parquet"]

        for fmt in formats:
            file_path = temp_dir / f"test{fmt}"
            if fmt == ".csv":
                sample_df.to_csv(file_path, index=False)
            elif fmt == ".json":
                sample_df.to_json(file_path, orient="records")
            elif fmt == ".parquet":
                sample_df.to_parquet(file_path, index=False)

            result = loader.get_file_info(file_path)

            assert result["format"] == fmt

    def test_get_file_info_with_string_path(self, loader, sample_df, temp_dir):
        """Test getting file info with string path."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        result = loader.get_file_info(str(file_path))

        assert result["name"] == "test.csv"

    # -------------------------------------------------------------------------
    # Class Attributes Tests
    # -------------------------------------------------------------------------

    def test_supported_formats_constant(self, loader):
        """Test that SUPPORTED_FORMATS contains expected formats."""
        expected_formats = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
        assert loader.SUPPORTED_FORMATS == expected_formats

    def test_chunk_size_constant(self, loader):
        """Test default chunk size constant."""
        assert loader.CHUNK_SIZE == 100000

    def test_large_file_threshold_constant(self, loader):
        """Test large file threshold constant."""
        assert loader.LARGE_FILE_THRESHOLD_MB == 100

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_load_and_validate_workflow(self, loader, sample_df, temp_dir):
        """Test complete workflow: save, load, validate checksum."""
        file_path = temp_dir / "test.csv"
        sample_df.to_csv(file_path, index=False)

        # Get file info including checksum
        info = loader.get_file_info(file_path)

        # Load the file
        loaded_df = loader.load(file_path)

        # Validate checksum
        is_valid = loader.validate_checksum(file_path, info["checksum"])

        assert len(loaded_df) == len(sample_df)
        assert is_valid is True

    def test_case_insensitive_extension(self, loader, sample_df, temp_dir):
        """Test that file extensions are handled case-insensitively."""
        file_path = temp_dir / "test.CSV"
        sample_df.to_csv(file_path, index=False)

        result = loader.load(file_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
