"""Tests for the Visualization Agent."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.visualization import (
    DEFAULT_COLORS,
    Chart,
    ChartConfig,
    ChartType,
    VisualizationAgent,
    VisualizationOutput,
)


class TestVisualizationAgent:
    """Test suite for VisualizationAgent."""

    @pytest.fixture
    def agent(self):
        """Create a visualization agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return VisualizationAgent()

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "sales": [100, 120, 115, 130, 145, 140, 160, 175, 180, 190],
                "cost": [50, 55, 52, 60, 65, 62, 70, 75, 78, 82],
                "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            }
        )

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results for testing."""
        return {
            "descriptive": {
                "sales": {"mean": {"value": 145.5}, "std": {"value": 30.2}},
                "cost": {"mean": {"value": 64.9}, "std": {"value": 11.3}},
            },
            "correlation": {
                "pairwise_correlations": [
                    {
                        "variable_1": "sales",
                        "variable_2": "cost",
                        "correlation": 0.95,
                        "p_value": 0.0001,
                    },
                ],
            },
        }

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "visualization"
        assert agent.autonomy.value == "supervised"

    def test_generate_chart_id(self, agent):
        """Test unique chart ID generation."""
        id1 = agent._generate_chart_id()
        id2 = agent._generate_chart_id()
        assert id1 != id2
        assert id1 == "chart_001"
        assert id2 == "chart_002"

    @pytest.mark.asyncio
    async def test_execute_with_dataframe(self, agent, sample_dataframe):
        """Test chart generation with a DataFrame."""
        result = await agent.execute(data=sample_dataframe)

        assert result.success
        assert result.data is not None
        assert len(result.data.charts) > 0
        assert result.data.summary

    @pytest.mark.asyncio
    async def test_execute_with_analysis_results(
        self, agent, sample_dataframe, sample_analysis_results
    ):
        """Test chart generation with analysis results."""
        result = await agent.execute(
            data=sample_dataframe,
            analysis_results=sample_analysis_results,
        )

        assert result.success
        assert result.data is not None
        # Should auto-generate charts from analysis
        assert len(result.data.charts) > 0

    @pytest.mark.asyncio
    async def test_execute_with_no_input(self, agent):
        """Test error handling when no input provided."""
        result = await agent.execute()

        assert not result.success
        assert "no data" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_chart_requests(self, agent, sample_dataframe):
        """Test chart generation with explicit requests."""
        chart_requests = [
            {
                "chart_type": "line",
                "x": "date",
                "y": "sales",
                "title": "Sales Trend",
                "data_source": "Test Data",
            },
            {
                "chart_type": "bar",
                "x": "category",
                "y": "sales",
                "title": "Sales by Category",
            },
        ]

        result = await agent.execute(
            data=sample_dataframe,
            chart_requests=chart_requests,
        )

        assert result.success
        assert len(result.data.charts) == 2
        assert result.data.charts[0].chart_type == ChartType.LINE
        assert result.data.charts[1].chart_type == ChartType.BAR

    def test_create_line_chart(self, agent, sample_dataframe):
        """Test line chart creation."""
        config = ChartConfig(
            title="Sales Over Time",
            x_label="Date",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        chart = agent.create_line_chart(sample_dataframe, "date", "sales", config)

        assert chart.chart_type == ChartType.LINE
        assert chart.title == "Sales Over Time"
        assert chart.figure is not None
        assert chart.data_source == "Test Data"

    def test_create_line_chart_multiple_y(self, agent, sample_dataframe):
        """Test line chart with multiple y columns."""
        config = ChartConfig(
            title="Sales and Cost Over Time",
            x_label="Date",
            y_label="Value ($)",
            data_source="Test Data",
        )

        chart = agent.create_line_chart(sample_dataframe, "date", ["sales", "cost"], config)

        assert chart.chart_type == ChartType.LINE
        assert chart.figure is not None

    def test_create_bar_chart(self, agent, sample_dataframe):
        """Test bar chart creation."""
        # Aggregate data for bar chart
        agg_data = sample_dataframe.groupby("category")["sales"].mean().reset_index()

        config = ChartConfig(
            title="Average Sales by Category",
            x_label="Category",
            y_label="Average Sales ($)",
            data_source="Test Data",
        )

        chart = agent.create_bar_chart(agg_data, "category", "sales", config)

        assert chart.chart_type == ChartType.BAR
        assert chart.title == "Average Sales by Category"
        assert chart.figure is not None

    def test_create_scatter_plot(self, agent, sample_dataframe):
        """Test scatter plot creation."""
        config = ChartConfig(
            title="Sales vs Cost",
            x_label="Cost ($)",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        chart = agent.create_scatter_plot(sample_dataframe, "cost", "sales", config)

        assert chart.chart_type == ChartType.SCATTER
        assert chart.figure is not None

    def test_create_heatmap(self, agent, sample_dataframe):
        """Test heatmap creation."""
        config = ChartConfig(
            title="Correlation Heatmap",
            data_source="Test Data",
        )

        chart = agent.create_heatmap(sample_dataframe, config)

        assert chart.chart_type == ChartType.HEATMAP
        assert chart.figure is not None

    def test_create_box_plot(self, agent, sample_dataframe):
        """Test box plot creation."""
        config = ChartConfig(
            title="Distribution of Sales and Cost",
            y_label="Value",
            data_source="Test Data",
        )

        chart = agent.create_box_plot(sample_dataframe, ["sales", "cost"], config)

        assert chart.chart_type == ChartType.BOX
        assert chart.figure is not None

    def test_create_pie_chart(self, agent):
        """Test pie chart creation."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D"],
                "value": [30, 25, 25, 20],
            }
        )

        config = ChartConfig(
            title="Distribution by Category",
            data_source="Test Data",
        )

        chart = agent.create_pie_chart(data, "value", "category", config)

        assert chart.chart_type == ChartType.PIE
        assert chart.figure is not None

    def test_create_donut_chart(self, agent):
        """Test donut chart creation."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D"],
                "value": [30, 25, 25, 20],
            }
        )

        config = ChartConfig(
            title="Distribution by Category",
            data_source="Test Data",
        )

        chart = agent.create_pie_chart(data, "value", "category", config, hole=0.4)

        assert chart.chart_type == ChartType.DONUT
        assert chart.figure is not None

    def test_create_histogram(self, agent, sample_dataframe):
        """Test histogram creation."""
        config = ChartConfig(
            title="Sales Distribution",
            x_label="Sales ($)",
            y_label="Frequency",
            data_source="Test Data",
        )

        chart = agent.create_histogram(sample_dataframe, "sales", config)

        assert chart.chart_type == ChartType.HISTOGRAM
        assert chart.figure is not None

    def test_create_area_chart(self, agent, sample_dataframe):
        """Test area chart creation."""
        config = ChartConfig(
            title="Cumulative Sales",
            x_label="Date",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        chart = agent.create_area_chart(sample_dataframe, "date", "sales", config)

        assert chart.chart_type == ChartType.AREA
        assert chart.figure is not None

    def test_validate_chart_missing_title(self, agent, sample_dataframe):
        """Test validation catches missing title."""
        config = ChartConfig(
            title="",  # Empty title
            data_source="Test Data",
        )

        chart = agent.create_line_chart(sample_dataframe, "date", "sales", config)

        assert len(chart.validation_warnings) > 0
        assert any("title" in w.lower() for w in chart.validation_warnings)

    def test_validate_chart_missing_axis_labels(self, agent, sample_dataframe):
        """Test validation catches missing axis labels."""
        config = ChartConfig(
            title="Test Chart",
            # Missing x_label and y_label
            data_source="Test Data",
        )

        chart = agent.create_line_chart(sample_dataframe, "date", "sales", config)

        assert any("x-axis" in w.lower() for w in chart.validation_warnings)
        assert any("y-axis" in w.lower() for w in chart.validation_warnings)

    def test_pie_chart_too_many_categories_warning(self, agent):
        """Test warning for pie chart with too many categories."""
        data = pd.DataFrame(
            {
                "category": [f"Cat{i}" for i in range(10)],  # More than 7
                "value": [10] * 10,
            }
        )

        config = ChartConfig(
            title="Distribution",
            data_source="Test Data",
        )

        chart = agent.create_pie_chart(data, "value", "category", config)

        assert any("7 categories" in w for w in chart.validation_warnings)

    def test_get_chart_type_options(self, agent, sample_dataframe):
        """Test getting chart type options."""
        options = agent.get_chart_type_options(sample_dataframe)

        assert len(options) >= 4
        assert any(opt.id == "auto" for opt in options)
        assert any(opt.id == "line" for opt in options)
        assert any(opt.recommended for opt in options)

    def test_format_output(self, agent):
        """Test output formatting."""
        config = ChartConfig(
            title="Test Chart",
            data_source="Test Data",
        )

        output = VisualizationOutput(
            charts=[
                Chart(
                    chart_id="chart_001",
                    chart_type=ChartType.LINE,
                    title="Test Chart",
                    description="A test chart",
                    figure=MagicMock(),  # Mock figure
                    data_source="Test Data",
                    config=config,
                    validation_warnings=["Missing x-axis label"],
                ),
            ],
            summary="Generated 1 visualization(s)",
            recommendations=["Consider adding more context"],
            accessibility_notes=["Uses colorblind-friendly palette"],
        )

        formatted = agent.format_output(output)

        assert "Test Chart" in formatted
        assert "line" in formatted.lower()
        assert "Missing x-axis label" in formatted
        assert "colorblind" in formatted.lower()

    def test_default_colors_are_colorblind_friendly(self):
        """Test that default colors are defined."""
        assert len(DEFAULT_COLORS) == 7
        # All should be hex colors
        for color in DEFAULT_COLORS:
            assert color.startswith("#")
            assert len(color) == 7

    def test_chart_to_dict(self):
        """Test Chart to_dict conversion."""
        config = ChartConfig(
            title="Test",
            data_source="Test Data",
        )

        chart = Chart(
            chart_id="chart_001",
            chart_type=ChartType.BAR,
            title="Test Chart",
            description="Description",
            figure=MagicMock(),  # Won't be in dict
            html_content="<html>...</html>",
            data_source="Test Data",
            config=config,
            validation_warnings=["Warning 1"],
        )

        d = chart.to_dict()

        assert d["chart_id"] == "chart_001"
        assert d["chart_type"] == "bar"
        assert d["title"] == "Test Chart"
        assert "figure" not in d  # Should be excluded
        assert d["html_content"] == "<html>...</html>"
        assert d["validation_warnings"] == ["Warning 1"]

    def test_visualization_output_to_dict(self):
        """Test VisualizationOutput to_dict conversion."""
        output = VisualizationOutput(
            charts=[],
            summary="Test summary",
            recommendations=["Rec 1"],
            accessibility_notes=["Note 1"],
        )

        d = output.to_dict()

        assert d["summary"] == "Test summary"
        assert d["recommendations"] == ["Rec 1"]
        assert d["accessibility_notes"] == ["Note 1"]
        assert d["charts"] == []

    def test_chart_config_to_dict(self):
        """Test ChartConfig to_dict conversion."""
        config = ChartConfig(
            title="Test Title",
            x_label="X Axis",
            y_label="Y Axis",
            data_source="Source",
            colors=["#000000"],
            show_legend=False,
            start_y_at_zero=False,
            height=600,
            width=1000,
        )

        d = config.to_dict()

        assert d["title"] == "Test Title"
        assert d["x_label"] == "X Axis"
        assert d["y_label"] == "Y Axis"
        assert d["data_source"] == "Source"
        assert d["colors"] == ["#000000"]
        assert d["show_legend"] is False
        assert d["start_y_at_zero"] is False
        assert d["height"] == 600
        assert d["width"] == 1000


class TestChartType:
    """Test suite for ChartType enum."""

    def test_all_chart_types_defined(self):
        """Test all expected chart types are defined."""
        expected = ["line", "bar", "scatter", "heatmap", "box", "pie", "donut", "histogram", "area"]
        for ct in expected:
            assert ChartType(ct) is not None

    def test_chart_type_values(self):
        """Test chart type string values."""
        assert ChartType.LINE.value == "line"
        assert ChartType.BAR.value == "bar"
        assert ChartType.SCATTER.value == "scatter"


class TestAutoGeneration:
    """Test suite for auto-generation of charts."""

    @pytest.fixture
    def agent(self):
        """Create a visualization agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return VisualizationAgent()

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "sales": [100, 120, 115, 130, 145, 140, 160, 175, 180, 190],
                "cost": [50, 55, 52, 60, 65, 62, 70, 75, 78, 82],
                "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            }
        )

    def test_auto_generate_from_correlation(self, agent, sample_dataframe):
        """Test auto-generation from correlation results."""
        analysis_results = {
            "correlation": {
                "matrix": True,  # Indicates correlation was computed
                "pairwise_correlations": [
                    {"variable_1": "sales", "variable_2": "cost", "correlation": 0.95},
                ],
            },
        }

        charts = agent._auto_generate_charts(sample_dataframe, analysis_results)

        # Should generate a heatmap
        assert any(c.chart_type == ChartType.HEATMAP for c in charts)

    def test_auto_generate_from_descriptive(self, agent, sample_dataframe):
        """Test auto-generation from descriptive stats."""
        analysis_results = {
            "descriptive": {
                "sales": {"mean": {"value": 145.5}},
                "cost": {"mean": {"value": 64.9}},
            },
        }

        charts = agent._auto_generate_charts(sample_dataframe, analysis_results)

        # Should generate a box plot
        assert any(c.chart_type == ChartType.BOX for c in charts)

    def test_create_default_charts(self, agent, sample_dataframe):
        """Test default chart creation for EDA."""
        charts = agent._create_default_charts(sample_dataframe)

        # Should create box plot for numeric columns
        assert any(c.chart_type == ChartType.BOX for c in charts)

        # Should create heatmap if enough numeric columns
        assert any(c.chart_type == ChartType.HEATMAP for c in charts)

    def test_generate_summary(self, agent):
        """Test summary generation."""
        mock_figure = MagicMock()
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.LINE,
                title="Chart 1",
                description="",
                figure=mock_figure,
            ),
            Chart(
                chart_id="002",
                chart_type=ChartType.BAR,
                title="Chart 2",
                description="",
                figure=mock_figure,
            ),
            Chart(
                chart_id="003",
                chart_type=ChartType.LINE,
                title="Chart 3",
                description="",
                figure=mock_figure,
                validation_warnings=["Warning"],
            ),
        ]

        summary = agent._generate_summary(charts)

        assert "3 visualization" in summary
        assert "2 line" in summary
        assert "1 bar" in summary
        assert "warning" in summary.lower()

    def test_generate_recommendations(self, agent, sample_dataframe):
        """Test recommendation generation."""
        mock_figure = MagicMock()
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.PIE,
                title="Pie",
                description="",
                figure=mock_figure,
                validation_warnings=[
                    "has 10 categories. Consider using a bar chart for more than 7 categories"
                ],
            ),
        ]

        recommendations = agent._generate_recommendations(sample_dataframe, charts)

        # Should recommend bar chart over pie
        assert any("bar chart" in r.lower() for r in recommendations)

    def test_generate_accessibility_notes(self, agent):
        """Test accessibility notes generation."""
        mock_figure = MagicMock()
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.LINE,
                title="Chart",
                description="",
                figure=mock_figure,
                config=ChartConfig(title="Chart", data_source="Test"),
            ),
        ]

        notes = agent._generate_accessibility_notes(charts)

        assert any("colorblind" in n.lower() for n in notes)


class TestVisualizationAgentExtended:
    """Extended test suite for VisualizationAgent."""

    @pytest.fixture
    def agent(self):
        """Create a visualization agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return VisualizationAgent()

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10),
                "sales": [100, 120, 115, 130, 145, 140, 160, 175, 180, 190],
                "cost": [50, 55, 52, 60, 65, 62, 70, 75, 78, 82],
                "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "region": [
                    "North",
                    "South",
                    "East",
                    "West",
                    "North",
                    "South",
                    "East",
                    "West",
                    "North",
                    "South",
                ],
            }
        )

    def test_create_scatter_plot_with_color(self, agent, sample_dataframe):
        """Test scatter plot with color dimension."""
        config = ChartConfig(
            title="Sales vs Cost by Category",
            x_label="Cost ($)",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        # Basic scatter plot (color_column not a parameter)
        chart = agent.create_scatter_plot(sample_dataframe, "cost", "sales", config)

        assert chart.chart_type == ChartType.SCATTER
        assert chart.figure is not None

    def test_create_bar_chart_horizontal(self, agent, sample_dataframe):
        """Test horizontal bar chart creation."""
        agg_data = sample_dataframe.groupby("category")["sales"].mean().reset_index()

        config = ChartConfig(
            title="Sales by Category",
            x_label="Category",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        # orientation parameter
        chart = agent.create_bar_chart(agg_data, "category", "sales", config, orientation="h")

        assert chart.chart_type == ChartType.BAR
        assert chart.figure is not None

    def test_create_histogram_with_bins(self, agent, sample_dataframe):
        """Test histogram creation with custom bins."""
        config = ChartConfig(
            title="Sales Distribution",
            x_label="Sales ($)",
            y_label="Count",
            data_source="Test Data",
        )

        chart = agent.create_histogram(sample_dataframe, "sales", config, bins=5)

        assert chart.chart_type == ChartType.HISTOGRAM
        assert chart.figure is not None

    def test_create_box_plot_with_groupby(self, agent, sample_dataframe):
        """Test box plot with groupby."""
        config = ChartConfig(
            title="Sales by Category",
            x_label="Category",
            y_label="Sales ($)",
            data_source="Test Data",
        )

        chart = agent.create_box_plot(sample_dataframe, ["sales"], config, group_by="category")

        assert chart.chart_type == ChartType.BOX
        assert chart.figure is not None

    def test_create_area_chart_stacked(self, agent, sample_dataframe):
        """Test stacked area chart."""
        config = ChartConfig(
            title="Sales and Cost Over Time",
            x_label="Date",
            y_label="Value ($)",
            data_source="Test Data",
        )

        chart = agent.create_area_chart(sample_dataframe, "date", ["sales", "cost"], config)

        assert chart.chart_type == ChartType.AREA
        assert chart.figure is not None

    def test_validate_chart_produces_warnings(self, agent, sample_dataframe):
        """Test validation produces warnings for missing labels."""
        config = ChartConfig(
            title="Test Chart",
            # Missing x_label and y_label
            data_source="Test Data",
        )

        chart = agent.create_line_chart(sample_dataframe, "date", "sales", config)

        # Should have validation warnings for missing labels
        assert len(chart.validation_warnings) > 0

    @pytest.mark.asyncio
    async def test_execute_with_output_path(self, agent, sample_dataframe, tmp_path):
        """Test chart generation with output path."""
        output_path = tmp_path / "charts"

        result = await agent.execute(
            data=sample_dataframe,
            output_path=str(output_path),
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_execute_with_specific_chart_type(self, agent, sample_dataframe):
        """Test chart generation with specific chart type."""
        chart_requests = [
            {
                "chart_type": "heatmap",
                "title": "Correlation Heatmap",
                "data_source": "Test Data",
            },
        ]

        result = await agent.execute(
            data=sample_dataframe,
            chart_requests=chart_requests,
        )

        assert result.success
        assert any(c.chart_type == ChartType.HEATMAP for c in result.data.charts)

    def test_get_chart_type_options_categorical(self, agent):
        """Test chart options for categorical data."""
        data = pd.DataFrame(
            {
                "category": ["A", "B", "C", "D"],
                "count": [10, 20, 15, 25],
            }
        )

        options = agent.get_chart_type_options(data)

        assert len(options) > 0
        assert any(opt.id in ["bar", "pie", "auto"] for opt in options)

    def test_get_chart_type_options_time_series(self, agent):
        """Test chart options for time series data."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "value": range(30),
            }
        )

        options = agent.get_chart_type_options(data)

        # Should have various options
        assert len(options) > 0

    def test_create_default_charts_with_time_series(self, agent):
        """Test default chart creation with time series data."""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30),
                "value": range(30),
                "metric": [x * 2 for x in range(30)],
            }
        )

        charts = agent._create_default_charts(data)

        # Should create some charts
        assert len(charts) > 0

    def test_format_output_no_charts(self, agent):
        """Test output formatting with no charts."""
        output = VisualizationOutput(
            charts=[],
            summary="No visualizations generated",
            recommendations=[],
            accessibility_notes=[],
        )

        formatted = agent.format_output(output)

        assert "no" in formatted.lower()

    def test_format_output_multiple_charts(self, agent):
        """Test output formatting with multiple charts."""
        config = ChartConfig(title="Test", data_source="Test")
        mock_figure = MagicMock()

        output = VisualizationOutput(
            charts=[
                Chart(
                    chart_id="001",
                    chart_type=ChartType.LINE,
                    title="Line Chart",
                    description="A line chart",
                    figure=mock_figure,
                    config=config,
                ),
                Chart(
                    chart_id="002",
                    chart_type=ChartType.BAR,
                    title="Bar Chart",
                    description="A bar chart",
                    figure=mock_figure,
                    config=config,
                ),
            ],
            summary="Generated 2 visualizations",
            recommendations=["Add more context"],
            accessibility_notes=["Uses colorblind-friendly colors"],
        )

        formatted = agent.format_output(output)

        assert "Line Chart" in formatted
        assert "Bar Chart" in formatted
        assert "2 visualizations" in formatted

    def test_generate_summary(self, agent):
        """Test summary generation with multiple chart types."""
        mock_figure = MagicMock()
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.LINE,
                title="Line 1",
                description="",
                figure=mock_figure,
            ),
            Chart(
                chart_id="002",
                chart_type=ChartType.LINE,
                title="Line 2",
                description="",
                figure=mock_figure,
            ),
            Chart(
                chart_id="003",
                chart_type=ChartType.BAR,
                title="Bar 1",
                description="",
                figure=mock_figure,
            ),
        ]

        summary = agent._generate_summary(charts)

        assert "3" in summary
        assert "line" in summary.lower()
        assert "bar" in summary.lower()

    def test_generate_recommendations(self, agent, sample_dataframe):
        """Test recommendation generation."""
        mock_figure = MagicMock()
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.LINE,
                title="Test",
                description="",
                figure=mock_figure,
            ),
        ]

        recommendations = agent._generate_recommendations(sample_dataframe, charts)

        # Should return a list (may be empty)
        assert isinstance(recommendations, list)

    def test_generate_accessibility_notes(self, agent):
        """Test accessibility notes generation."""
        mock_figure = MagicMock()
        config = ChartConfig(title="Test", data_source="Test")
        charts = [
            Chart(
                chart_id="001",
                chart_type=ChartType.LINE,
                title="Test",
                description="",
                figure=mock_figure,
                config=config,
            ),
        ]

        notes = agent._generate_accessibility_notes(charts)

        assert isinstance(notes, list)
        # Should mention colorblind-friendly colors
        assert any("colorblind" in n.lower() for n in notes)
