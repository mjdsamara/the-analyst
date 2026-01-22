"""Tests for the Intent Router."""

import os

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.orchestrator.router import (
    HIGH_STAKES_KEYWORDS,
    INTENT_AGENTS,
    INTENT_PATTERNS,
    Intent,
    IntentRouter,
    IntentType,
)


class TestIntentRouter:
    """Test suite for IntentRouter."""

    @pytest.fixture
    def router(self):
        """Create an IntentRouter instance for testing."""
        return IntentRouter()

    # -------------------------------------------------------------------------
    # Initialization Tests
    # -------------------------------------------------------------------------

    def test_router_initialization(self, router):
        """Test that router initializes correctly."""
        assert router is not None
        assert hasattr(router, "_compiled_patterns")
        assert len(router._compiled_patterns) > 0

    def test_patterns_compiled(self, router):
        """Test that all patterns are compiled."""
        for intent_type, patterns in router._compiled_patterns.items():
            assert len(patterns) > 0
            for pattern in patterns:
                assert hasattr(pattern, "search")  # Compiled regex has search method

    # -------------------------------------------------------------------------
    # Load Data Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_load_data_intent(self, router):
        """Test parsing load data intents."""
        test_cases = [
            "Load the data from sales.csv",
            "Import data from file.xlsx",
            "Read the file data.json",
            "Please ingest my_data.parquet",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.LOAD_DATA, f"Failed for: {text}"
            assert intent.confidence > 0
            assert "retrieval" in intent.agents_required

    def test_parse_load_data_extracts_file(self, router):
        """Test that file paths are extracted from load data intents."""
        intent = router.parse_intent("Load data from sales_2024.csv")

        assert intent.type == IntentType.LOAD_DATA
        assert "files" in intent.parameters
        assert "sales_2024.csv" in intent.parameters["files"]

    def test_parse_load_data_multiple_files(self, router):
        """Test extracting multiple file paths."""
        intent = router.parse_intent("Load data.csv and other.xlsx")

        assert "files" in intent.parameters
        assert len(intent.parameters["files"]) >= 2

    # -------------------------------------------------------------------------
    # Explore Data Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_explore_data_intent(self, router):
        """Test parsing explore data intents."""
        test_cases = [
            "Explore the data",
            "Show me the data summary",
            "What does the data look like?",
            "Describe the data",
            "Give me an overview of the dataset",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.EXPLORE_DATA, f"Failed for: {text}"

    # -------------------------------------------------------------------------
    # Transform Data Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_transform_data_intent(self, router):
        """Test parsing transform data intents."""
        test_cases = [
            "Clean the data",
            "Transform the data for analysis",
            "Prepare the data",
            "Remove duplicates from the dataset",
            "Fill missing values",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.TRANSFORM_DATA, f"Failed for: {text}"
            assert "transform" in intent.agents_required

    # -------------------------------------------------------------------------
    # Statistical Analysis Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_statistical_analysis_intent(self, router):
        """Test parsing statistical analysis intents."""
        test_cases = [
            "Analyze the data",
            "Run statistical analysis",
            "Show me descriptive statistics",
            "What's the distribution of sales?",
            "Calculate the mean and standard deviation",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.STATISTICAL_ANALYSIS, f"Failed for: {text}"
            assert "statistical" in intent.agents_required

    # -------------------------------------------------------------------------
    # Sentiment Analysis Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_sentiment_analysis_intent(self, router):
        """Test parsing sentiment analysis intents."""
        test_cases = [
            "Run sentiment analysis",
            "Analyze the sentiment of comments",
            "Is this text positive or negative?",
            "Analyze Arabic text sentiment",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.SENTIMENT_ANALYSIS, f"Failed for: {text}"
            assert "arabic_nlp" in intent.agents_required

    # -------------------------------------------------------------------------
    # Correlation Analysis Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_correlation_analysis_intent(self, router):
        """Test parsing correlation analysis intents."""
        test_cases = [
            "Find correlation between X and Y",
            "What's the relationship between views and engagement?",
            "Are these variables correlated?",
            "How are sales associated with marketing?",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.CORRELATION_ANALYSIS, f"Failed for: {text}"

    # -------------------------------------------------------------------------
    # Trend Analysis Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_trend_analysis_intent(self, router):
        """Test parsing trend analysis intents."""
        test_cases = [
            "Show me the trend",
            "How did sales change over time?",
            "Analyze the time series",
            "What's the growth pattern?",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.TREND_ANALYSIS, f"Failed for: {text}"

    # -------------------------------------------------------------------------
    # Forecast Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_forecast_intent(self, router):
        """Test parsing forecast intents."""
        test_cases = [
            "Forecast next 30 days",
            "Predict future sales",
            "What will revenue be next month?",
            "Give me a projection",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.FORECAST, f"Failed for: {text}"
            assert "modeling" in intent.agents_required

    def test_parse_forecast_extracts_horizon(self, router):
        """Test that forecast horizon is extracted."""
        test_cases = [
            ("Forecast next 30 days", 30, "day"),
            ("Predict next 12 months", 12, "month"),
            ("Project next 4 weeks", 4, "week"),
            ("Forecast next 2 years", 2, "year"),
        ]

        for text, value, unit in test_cases:
            intent = router.parse_intent(text)
            assert "forecast_horizon" in intent.parameters, f"Failed for: {text}"
            assert intent.parameters["forecast_horizon"]["value"] == value
            assert intent.parameters["forecast_horizon"]["unit"] == unit

    def test_forecast_requires_approval(self, router):
        """Test that forecast intent requires approval."""
        intent = router.parse_intent("Forecast sales for next quarter")

        assert intent.requires_approval

    # -------------------------------------------------------------------------
    # Classification Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_classification_intent(self, router):
        """Test parsing classification intents."""
        test_cases = [
            "Classify these articles",
            "Run classification model",
            "Categorize the data",
            "Which category does this belong to?",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.CLASSIFICATION, f"Failed for: {text}"
            assert "modeling" in intent.agents_required

    # -------------------------------------------------------------------------
    # Regression Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_regression_intent(self, router):
        """Test parsing regression intents."""
        test_cases = [
            "Run regression analysis",
            "Predict a value",
            "Estimate sales based on features",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.REGRESSION, f"Failed for: {text}"

    # -------------------------------------------------------------------------
    # Visualize Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_visualize_intent(self, router):
        """Test parsing visualization intents."""
        test_cases = [
            "Visualize the data",
            "Create a chart",
            "Show me a graph",
            "Plot the results",
            "Create a visual summary",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.VISUALIZE, f"Failed for: {text}"
            assert "visualization" in intent.agents_required

    # -------------------------------------------------------------------------
    # Generate Report Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_generate_report_intent(self, router):
        """Test parsing report generation intents."""
        test_cases = [
            "Generate a report",
            "Create a report",
            "Write an executive summary",
            "Prepare a summary report",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.GENERATE_REPORT, f"Failed for: {text}"
            assert "report" in intent.agents_required

    def test_parse_report_extracts_format(self, router):
        """Test that report format is extracted."""
        test_cases = [
            ("Generate a PDF report", "pdf"),
            ("Create a PowerPoint presentation", "powerpoint"),
            ("Write report in HTML", "html"),
            ("Generate markdown report", "markdown"),
        ]

        for text, expected_format in test_cases:
            intent = router.parse_intent(text)
            assert "output_format" in intent.parameters, f"Failed for: {text}"
            assert intent.parameters["output_format"] == expected_format

    def test_report_requires_approval(self, router):
        """Test that report generation requires approval."""
        intent = router.parse_intent("Generate an executive report")

        assert intent.requires_approval

    # -------------------------------------------------------------------------
    # Export Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_export_intent(self, router):
        """Test parsing export intents."""
        test_cases = [
            "Export the results",
            "Save to file",
            "Download the data",
            "Share the report",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.EXPORT, f"Failed for: {text}"

    def test_export_requires_approval(self, router):
        """Test that export intent requires approval."""
        intent = router.parse_intent("Export results to external system")

        assert intent.requires_approval

    # -------------------------------------------------------------------------
    # Help Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_help_intent(self, router):
        """Test parsing help intents."""
        test_cases = [
            "Help",
            "How do I analyze data?",
            "What can you do?",
            "Show me your capabilities",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.HELP, f"Failed for: {text}"
            assert len(intent.agents_required) == 0

    # -------------------------------------------------------------------------
    # Status Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_status_intent(self, router):
        """Test parsing status intents."""
        test_cases = [
            "Status",
            "Show me progress",
            "Where are we in the analysis?",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert intent.type == IntentType.STATUS, f"Failed for: {text}"

    # -------------------------------------------------------------------------
    # Unknown Intent Tests
    # -------------------------------------------------------------------------

    def test_parse_unknown_intent(self, router):
        """Test parsing unknown/unclear intents."""
        intent = router.parse_intent("xyzzy gibberish random text")

        assert intent.type == IntentType.UNKNOWN
        assert intent.confidence == 0.0
        assert len(intent.clarifications_needed) > 0

    # -------------------------------------------------------------------------
    # High Stakes Keyword Detection Tests
    # -------------------------------------------------------------------------

    def test_high_stakes_delete_keyword(self, router):
        """Test detection of 'delete' keyword."""
        intent = router.parse_intent("Delete the old data")

        assert intent.high_stakes
        assert any("delete" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_send_keyword(self, router):
        """Test detection of 'send' keyword."""
        intent = router.parse_intent("Send the report to stakeholders")

        assert intent.high_stakes
        assert any("send" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_share_keyword(self, router):
        """Test detection of 'share' keyword."""
        intent = router.parse_intent("Share the analysis with the team")

        assert intent.high_stakes
        assert any("share" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_production_keyword(self, router):
        """Test detection of 'production' keyword."""
        intent = router.parse_intent("Deploy to production")

        assert intent.high_stakes
        assert any("production" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_executive_keyword(self, router):
        """Test detection of 'executive' keyword."""
        intent = router.parse_intent("Prepare executive summary")

        assert intent.high_stakes
        assert any("executive" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_forecast_keyword(self, router):
        """Test detection of 'forecast' keyword."""
        intent = router.parse_intent("Forecast revenue for Q2")

        assert intent.high_stakes
        assert any("forecast" in reason.lower() for reason in intent.high_stakes_reasons)

    def test_high_stakes_multiple_keywords(self, router):
        """Test detection of multiple high-stakes keywords."""
        intent = router.parse_intent("Delete old data and send forecast to executives")

        assert intent.high_stakes
        assert len(intent.high_stakes_reasons) >= 3

    def test_no_high_stakes(self, router):
        """Test that normal requests don't trigger high-stakes."""
        intent = router.parse_intent("Analyze the data")

        assert not intent.high_stakes
        assert len(intent.high_stakes_reasons) == 0

    # -------------------------------------------------------------------------
    # Parameter Extraction Tests
    # -------------------------------------------------------------------------

    def test_extract_date_range(self, router):
        """Test extraction of date ranges."""
        intent = router.parse_intent("Analyze data from 2024-01-01 to 2024-06-30")

        assert "date_range" in intent.parameters
        assert intent.parameters["date_range"]["start"] == "2024-01-01"
        assert intent.parameters["date_range"]["end"] == "2024-06-30"

    def test_extract_column_names(self, router):
        """Test extraction of column names."""
        intent = router.parse_intent('Analyze column "revenue" and "expenses"')

        assert "columns" in intent.parameters

    def test_extract_numeric_values(self, router):
        """Test extraction of numeric values."""
        intent = router.parse_intent("Filter data where value > 100")

        assert "numeric_values" in intent.parameters
        assert 100.0 in intent.parameters["numeric_values"]

    # -------------------------------------------------------------------------
    # Clarification Request Tests
    # -------------------------------------------------------------------------

    def test_load_data_needs_file_clarification(self, router):
        """Test that load data without file needs clarification."""
        intent = router.parse_intent("Load the data")

        assert "files" not in intent.parameters
        assert len(intent.clarifications_needed) > 0
        assert any("file" in c.lower() for c in intent.clarifications_needed)

    def test_forecast_needs_horizon_clarification(self, router):
        """Test that forecast without horizon needs clarification."""
        intent = router.parse_intent("Forecast sales")

        assert "forecast_horizon" not in intent.parameters
        assert len(intent.clarifications_needed) > 0
        assert any("forecast" in c.lower() for c in intent.clarifications_needed)

    def test_report_needs_format_clarification(self, router):
        """Test that report generation without format needs clarification."""
        intent = router.parse_intent("Generate a report")

        assert "output_format" not in intent.parameters
        assert len(intent.clarifications_needed) > 0
        assert any("format" in c.lower() for c in intent.clarifications_needed)

    # -------------------------------------------------------------------------
    # Workflow Agent Tests
    # -------------------------------------------------------------------------

    def test_get_workflow_agents_statistical(self, router):
        """Test getting workflow agents for statistical analysis."""
        intent = router.parse_intent("Run statistical analysis")
        agents = router.get_workflow_agents(intent)

        assert "statistical" in agents

    def test_get_workflow_agents_report(self, router):
        """Test getting workflow agents for report generation."""
        intent = router.parse_intent("Generate a PDF report")
        agents = router.get_workflow_agents(intent)

        assert "report" in agents
        # Report should also include insights and visualization
        assert "insights" in agents or "visualization" in agents

    def test_workflow_agents_order(self, router):
        """Test that workflow agents are returned in correct order."""
        intent = Intent(
            type=IntentType.GENERATE_REPORT,
            confidence=1.0,
            agents_required=["report", "insights", "visualization"],
        )
        agents = router.get_workflow_agents(intent)

        # Insights should come before visualization, visualization before report
        insights_idx = agents.index("insights")
        viz_idx = agents.index("visualization")
        report_idx = agents.index("report")

        assert insights_idx < viz_idx
        assert viz_idx < report_idx

    # -------------------------------------------------------------------------
    # Confidence Score Tests
    # -------------------------------------------------------------------------

    def test_confidence_score_multiple_matches(self, router):
        """Test that multiple pattern matches increase confidence."""
        # More specific request with multiple matching patterns
        intent1 = router.parse_intent("Run statistical analysis")
        # Less specific request
        intent2 = router.parse_intent("Analyze")

        assert intent1.confidence >= intent2.confidence

    def test_confidence_score_bounds(self, router):
        """Test that confidence scores are within valid range."""
        test_cases = [
            "Load data from file.csv",
            "Statistical analysis",
            "Forecast next 30 days",
            "Generate report",
            "Random gibberish text",
        ]

        for text in test_cases:
            intent = router.parse_intent(text)
            assert 0.0 <= intent.confidence <= 1.0, f"Invalid confidence for: {text}"


class TestIntent:
    """Test suite for Intent dataclass."""

    def test_intent_requires_approval_high_stakes(self):
        """Test that high-stakes intent requires approval."""
        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
            high_stakes=True,
            high_stakes_reasons=["Contains high-stakes keyword: 'delete'"],
        )

        assert intent.requires_approval

    def test_intent_requires_approval_forecast(self):
        """Test that forecast intent requires approval."""
        intent = Intent(
            type=IntentType.FORECAST,
            confidence=1.0,
        )

        assert intent.requires_approval

    def test_intent_requires_approval_classification(self):
        """Test that classification intent requires approval."""
        intent = Intent(
            type=IntentType.CLASSIFICATION,
            confidence=1.0,
        )

        assert intent.requires_approval

    def test_intent_requires_approval_regression(self):
        """Test that regression intent requires approval."""
        intent = Intent(
            type=IntentType.REGRESSION,
            confidence=1.0,
        )

        assert intent.requires_approval

    def test_intent_requires_approval_report(self):
        """Test that report generation requires approval."""
        intent = Intent(
            type=IntentType.GENERATE_REPORT,
            confidence=1.0,
        )

        assert intent.requires_approval

    def test_intent_requires_approval_export(self):
        """Test that export requires approval."""
        intent = Intent(
            type=IntentType.EXPORT,
            confidence=1.0,
        )

        assert intent.requires_approval

    def test_intent_no_approval_statistical(self):
        """Test that normal statistical analysis doesn't require approval."""
        intent = Intent(
            type=IntentType.STATISTICAL_ANALYSIS,
            confidence=1.0,
            high_stakes=False,
        )

        assert not intent.requires_approval

    def test_intent_no_approval_load_data(self):
        """Test that load data doesn't require approval."""
        intent = Intent(
            type=IntentType.LOAD_DATA,
            confidence=1.0,
        )

        assert not intent.requires_approval


class TestIntentPatterns:
    """Test suite for intent pattern definitions."""

    def test_all_intent_types_have_patterns(self):
        """Test that all intent types have patterns defined."""
        for intent_type in IntentType:
            if intent_type not in [IntentType.CLARIFICATION_NEEDED, IntentType.UNKNOWN]:
                assert intent_type in INTENT_PATTERNS, f"Missing patterns for {intent_type}"

    def test_all_intent_types_have_agents(self):
        """Test that all intent types have agent mappings."""
        for intent_type in IntentType:
            assert intent_type in INTENT_AGENTS, f"Missing agent mapping for {intent_type}"


class TestHighStakesKeywords:
    """Test suite for high-stakes keyword definitions."""

    def test_high_stakes_keywords_exist(self):
        """Test that high-stakes keywords are defined."""
        assert len(HIGH_STAKES_KEYWORDS) > 0

    def test_high_stakes_keywords_lowercase(self):
        """Test that high-stakes keywords are lowercase for matching."""
        for keyword in HIGH_STAKES_KEYWORDS:
            assert keyword == keyword.lower(), f"Keyword '{keyword}' should be lowercase"

    def test_expected_keywords_present(self):
        """Test that expected high-stakes keywords are present."""
        expected = ["delete", "send", "share", "production", "forecast", "predict"]

        for keyword in expected:
            assert keyword in HIGH_STAKES_KEYWORDS, f"Expected keyword '{keyword}' missing"
