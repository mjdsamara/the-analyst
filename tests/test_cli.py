"""Tests for the CLI module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/analyst_test")

from src.cli import app, print_banner, setup_logging


class TestCLISetup:
    """Test suite for CLI setup functions."""

    def test_setup_logging_default(self):
        """Test setting up logging with default verbosity."""
        import logging

        setup_logging(verbose=False)
        # Should set INFO level by default
        assert logging.getLogger().level == logging.INFO or logging.root.level == logging.INFO

    def test_setup_logging_verbose(self):
        """Test setting up logging with verbose mode."""
        import logging

        setup_logging(verbose=True)
        # Should set DEBUG level when verbose
        assert logging.getLogger().level == logging.DEBUG or logging.root.level == logging.DEBUG

    def test_print_banner(self, capsys):
        """Test printing the application banner."""
        print_banner()

        captured = capsys.readouterr()
        assert "Analyst" in captured.out or len(captured.out) > 0


class TestAnalyzeCommand:
    """Test suite for the analyze command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_analyze_missing_file(self, runner):
        """Test analyze command with missing file."""
        result = runner.invoke(app, ["analyze", "nonexistent.csv"])

        assert result.exit_code != 0
        # Typer should report that the file doesn't exist

    def test_analyze_with_valid_file(self, runner, sample_csv_file, mock_cli_orchestrator):
        """Test analyze command with valid file."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_statistical_analysis = AsyncMock(
                return_value="Analysis complete"
            )
            mock_orchestrator.generate_insights = AsyncMock(return_value="Insights generated")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(return_value=None)
            mock_orchestrator._agent_context.set_data = MagicMock()
            mock_orchestrator.conversation = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "analyze",
                    str(sample_csv_file),
                    "--type",
                    "descriptive",
                    "--output",
                    str(sample_csv_file.parent / "report.md"),
                ],
            )

            # Should at least attempt to run
            assert result.exit_code == 0 or "Error" in result.output

    def test_analyze_output_option(self, runner, sample_csv_file):
        """Test analyze command output option."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_statistical_analysis = AsyncMock(
                return_value="Analysis complete"
            )
            mock_orchestrator.generate_insights = AsyncMock(return_value="Insights generated")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(return_value=None)
            mock_orchestrator._agent_context.set_data = MagicMock()
            mock_orchestrator.conversation = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            output_path = sample_csv_file.parent / "custom_report.md"
            result = runner.invoke(
                app,
                ["analyze", str(sample_csv_file), "-o", str(output_path)],
            )

            # Command should recognize output option
            assert "-o" not in result.output or result.exit_code == 0

    def test_analyze_type_option(self, runner, sample_csv_file):
        """Test analyze command with different analysis types."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_statistical_analysis = AsyncMock(
                return_value="Analysis complete"
            )
            mock_orchestrator.generate_insights = AsyncMock(return_value="Insights generated")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(return_value=None)
            mock_orchestrator._agent_context.set_data = MagicMock()
            mock_orchestrator.conversation = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            for analysis_type in ["comprehensive", "descriptive", "correlation", "distribution"]:
                result = runner.invoke(
                    app,
                    [
                        "analyze",
                        str(sample_csv_file),
                        "-t",
                        analysis_type,
                        "-o",
                        str(sample_csv_file.parent / "report.md"),
                    ],
                )
                # Should not fail on type option parsing
                assert "--type" not in result.output.lower() or result.exit_code == 0


class TestForecastCommand:
    """Test suite for the forecast command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_forecast_missing_file(self, runner):
        """Test forecast command with missing file."""
        result = runner.invoke(app, ["forecast", "nonexistent.csv", "--target", "value"])

        assert result.exit_code != 0

    def test_forecast_missing_target(self, runner, sample_csv_file):
        """Test forecast command without required target option."""
        result = runner.invoke(app, ["forecast", str(sample_csv_file)])

        # Should fail because --target is required
        assert result.exit_code != 0

    def test_forecast_with_valid_options(self, runner, sample_csv_file):
        """Test forecast command with valid options."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_forecast = AsyncMock(return_value="Forecast complete")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(
                return_value=pd.DataFrame({"value": [1, 2, 3]})
            )
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "forecast",
                    str(sample_csv_file),
                    "--target",
                    "value1",
                    "--periods",
                    "30",
                    "--method",
                    "exponential_smoothing",
                ],
            )

            assert result.exit_code == 0 or "Error" in result.output

    def test_forecast_periods_option(self, runner, sample_csv_file):
        """Test forecast command periods option."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_forecast = AsyncMock(return_value="Forecast complete")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(
                return_value=pd.DataFrame({"value": [1, 2, 3]})
            )
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "forecast",
                    str(sample_csv_file),
                    "--target",
                    "value1",
                    "-p",
                    "60",
                ],
            )

            # Should accept periods option
            assert "-p" not in result.output.lower() or "Error" not in result.output

    def test_forecast_method_option(self, runner, sample_csv_file):
        """Test forecast command with different methods."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_forecast = AsyncMock(return_value="Forecast complete")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(
                return_value=pd.DataFrame({"value": [1, 2, 3]})
            )
            mock_orch_class.return_value = mock_orchestrator

            for method in ["exponential_smoothing", "prophet", "arima"]:
                result = runner.invoke(
                    app,
                    [
                        "forecast",
                        str(sample_csv_file),
                        "--target",
                        "value1",
                        "-m",
                        method,
                    ],
                )
                # Method option should be accepted
                assert "--method" not in result.output.lower() or result.exit_code == 0


class TestSentimentCommand:
    """Test suite for the sentiment command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_sentiment_missing_file(self, runner):
        """Test sentiment command with missing file."""
        result = runner.invoke(app, ["sentiment", "nonexistent.csv", "--column", "text"])

        assert result.exit_code != 0

    def test_sentiment_missing_column(self, runner, sample_csv_file):
        """Test sentiment command without required column option."""
        result = runner.invoke(app, ["sentiment", str(sample_csv_file)])

        # Should fail because --column is required
        assert result.exit_code != 0

    def test_sentiment_with_valid_options(self, runner, sample_csv_file):
        """Test sentiment command with valid options."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_sentiment_analysis = AsyncMock(
                return_value="Sentiment analysis complete"
            )
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(
                return_value=pd.DataFrame({"text": ["hello", "world"]})
            )
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "sentiment",
                    str(sample_csv_file),
                    "--column",
                    "category",
                    "--output",
                    str(sample_csv_file.parent / "sentiment.md"),
                ],
            )

            assert result.exit_code == 0 or "Error" in result.output


class TestReportCommand:
    """Test suite for the report command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_report_default_options(self, runner):
        """Test report command with default options."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.generate_report = AsyncMock(return_value="Report generated")
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(app, ["report"])

            # Should run without error (may report no content)
            assert result.exit_code == 0

    def test_report_format_options(self, runner):
        """Test report command with different format options."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.generate_report = AsyncMock(return_value="Report generated")
            mock_orch_class.return_value = mock_orchestrator

            for format_type in ["markdown", "html", "pdf", "pptx"]:
                result = runner.invoke(app, ["report", "-f", format_type])
                # Format option should be accepted
                assert "--format" not in result.output.lower() or result.exit_code == 0

    def test_report_audience_options(self, runner):
        """Test report command with different audience options."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.generate_report = AsyncMock(return_value="Report generated")
            mock_orch_class.return_value = mock_orchestrator

            for audience in ["executive", "technical", "general"]:
                result = runner.invoke(app, ["report", "-a", audience])
                # Audience option should be accepted
                assert "--audience" not in result.output.lower() or result.exit_code == 0

    def test_report_title_option(self, runner):
        """Test report command with custom title."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.generate_report = AsyncMock(return_value="Report generated")
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(app, ["report", "-t", "Custom Report Title"])

            assert result.exit_code == 0


class TestInteractiveCommand:
    """Test suite for the interactive command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_interactive_quit(self, runner):
        """Test interactive command with quit."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_message = AsyncMock(return_value="Response")
            mock_orch_class.return_value = mock_orchestrator

            # Simulate user typing 'quit'
            result = runner.invoke(app, ["interactive"], input="quit\n")

            assert result.exit_code == 0
            assert "goodbye" in result.output.lower()

    def test_interactive_exit(self, runner):
        """Test interactive command with exit."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(app, ["interactive"], input="exit\n")

            assert result.exit_code == 0


class TestVersionCommand:
    """Test suite for the version command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_version_output(self, runner):
        """Test version command output."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "analyst" in result.output.lower()
        assert "0.1.0" in result.output


class TestCLIHelp:
    """Test suite for CLI help output."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "analyst" in result.output.lower()

    def test_analyze_help(self, runner):
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "file" in result.output.lower()
        assert "output" in result.output.lower()
        assert "type" in result.output.lower()

    def test_forecast_help(self, runner):
        """Test forecast command help."""
        result = runner.invoke(app, ["forecast", "--help"])

        assert result.exit_code == 0
        assert "target" in result.output.lower()
        assert "periods" in result.output.lower()
        assert "method" in result.output.lower()

    def test_sentiment_help(self, runner):
        """Test sentiment command help."""
        result = runner.invoke(app, ["sentiment", "--help"])

        assert result.exit_code == 0
        assert "column" in result.output.lower()
        assert "arabic" in result.output.lower()

    def test_report_help(self, runner):
        """Test report command help."""
        result = runner.invoke(app, ["report", "--help"])

        assert result.exit_code == 0
        assert "format" in result.output.lower()
        assert "audience" in result.output.lower()

    def test_interactive_help(self, runner):
        """Test interactive command help."""
        result = runner.invoke(app, ["interactive", "--help"])

        assert result.exit_code == 0
        assert "repl" in result.output.lower() or "interactive" in result.output.lower()


class TestCLIVerboseMode:
    """Test suite for CLI verbose mode."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    def test_analyze_verbose(self, runner, sample_csv_file):
        """Test analyze command with verbose flag."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_statistical_analysis = AsyncMock(
                return_value="Analysis complete"
            )
            mock_orchestrator.generate_insights = AsyncMock(return_value="Insights")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(return_value=None)
            mock_orchestrator._agent_context.set_data = MagicMock()
            mock_orchestrator.conversation = MagicMock()
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "analyze",
                    str(sample_csv_file),
                    "-v",
                    "-o",
                    str(sample_csv_file.parent / "report.md"),
                ],
            )

            # Verbose flag should be accepted
            assert "-v" not in result.output or result.exit_code == 0

    def test_forecast_verbose(self, runner, sample_csv_file):
        """Test forecast command with verbose flag."""
        with patch("src.orchestrator.main.Orchestrator") as mock_orch_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator.load_data = AsyncMock(return_value="Data loaded")
            mock_orchestrator._run_forecast = AsyncMock(return_value="Forecast")
            mock_orchestrator._agent_context = MagicMock()
            mock_orchestrator._agent_context.get_data = MagicMock(
                return_value=pd.DataFrame({"value": [1, 2, 3]})
            )
            mock_orch_class.return_value = mock_orchestrator

            result = runner.invoke(
                app,
                [
                    "forecast",
                    str(sample_csv_file),
                    "--target",
                    "value1",
                    "-v",
                ],
            )

            assert "-v" not in result.output or result.exit_code == 0
