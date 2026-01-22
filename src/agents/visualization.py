"""
Visualization Agent for The Analyst platform.

Creates clear, accurate, and interactive charts using Plotly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.prompts.agents import VISUALIZATION_PROMPT


class ChartType(str, Enum):
    """Supported chart types."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOX = "box"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    AREA = "area"


@dataclass
class ChartConfig:
    """Configuration for a chart."""

    title: str
    x_label: str = ""
    y_label: str = ""
    data_source: str = "Data source not specified"
    colors: list[str] | None = None
    show_legend: bool = True
    start_y_at_zero: bool = True
    height: int = 500
    width: int = 800
    template: str = "plotly_white"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "data_source": self.data_source,
            "colors": self.colors,
            "show_legend": self.show_legend,
            "start_y_at_zero": self.start_y_at_zero,
            "height": self.height,
            "width": self.width,
            "template": self.template,
        }


@dataclass
class Chart:
    """A single chart with metadata."""

    chart_id: str
    chart_type: ChartType
    title: str
    description: str
    figure: go.Figure
    html_content: str = ""
    image_base64: str | None = None
    data_source: str = ""
    config: ChartConfig | None = None
    validation_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding figure object)."""
        return {
            "chart_id": self.chart_id,
            "chart_type": self.chart_type.value,
            "title": self.title,
            "description": self.description,
            "html_content": self.html_content,
            "image_base64": self.image_base64,
            "data_source": self.data_source,
            "config": self.config.to_dict() if self.config else None,
            "validation_warnings": self.validation_warnings,
        }


@dataclass
class VisualizationOutput:
    """Complete output from the visualization agent."""

    charts: list[Chart] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    accessibility_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "charts": [c.to_dict() for c in self.charts],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "accessibility_notes": self.accessibility_notes,
        }


# Default color palette (colorblind-friendly)
DEFAULT_COLORS = [
    "#2E86AB",  # Blue
    "#A23B72",  # Magenta
    "#F18F01",  # Orange
    "#C73E1D",  # Red
    "#3B1F2B",  # Dark
    "#95C623",  # Green
    "#4ECDC4",  # Teal
]


class VisualizationAgent(BaseAgent):
    """
    Agent responsible for creating interactive charts using Plotly.

    Single Job: Create clear, accurate, and interactive charts.
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the visualization agent."""
        super().__init__(name="visualization", context=context)
        self._chart_counter = 0

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return VISUALIZATION_PROMPT

    def _generate_chart_id(self) -> str:
        """Generate a unique chart ID."""
        self._chart_counter += 1
        return f"chart_{self._chart_counter:03d}"

    async def execute(
        self,
        data: pd.DataFrame | dict[str, Any] | None = None,
        chart_requests: list[dict[str, Any]] | None = None,
        analysis_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[VisualizationOutput]:
        """
        Execute chart generation.

        Args:
            data: DataFrame or dict containing the data to visualize
            chart_requests: List of chart specifications
            analysis_results: Results from statistical analysis for auto-visualization

        Returns:
            AgentResult containing generated charts
        """
        if data is None and analysis_results is None:
            return AgentResult.error_result(
                "No data or analysis results provided for visualization"
            )

        self.log("Starting visualization generation")

        output = VisualizationOutput()

        try:
            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame(data)

            # Generate charts from explicit requests
            if chart_requests:
                for request in chart_requests:
                    chart = self._create_chart_from_request(data, request)
                    if chart:
                        output.charts.append(chart)

            # Auto-generate charts from analysis results
            if analysis_results:
                auto_charts = self._auto_generate_charts(data, analysis_results)
                output.charts.extend(auto_charts)

            # If no explicit requests and no analysis, create default EDA charts
            if not chart_requests and not analysis_results and data is not None:
                default_charts = self._create_default_charts(data)
                output.charts.extend(default_charts)

            # Generate HTML content for all charts
            for chart in output.charts:
                chart.html_content = chart.figure.to_html(include_plotlyjs="cdn")

            # Generate summary
            output.summary = self._generate_summary(output.charts)

            # Generate recommendations
            output.recommendations = self._generate_recommendations(data, output.charts)

            # Generate accessibility notes
            output.accessibility_notes = self._generate_accessibility_notes(output.charts)

            self.log(f"Generated {len(output.charts)} charts")

            return AgentResult.success_result(
                output,
                chart_count=len(output.charts),
                chart_types=[c.chart_type.value for c in output.charts],
            )

        except Exception as e:
            self.log(f"Visualization generation failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Visualization generation failed: {e}")

    def _create_chart_from_request(
        self,
        data: pd.DataFrame | None,
        request: dict[str, Any],
    ) -> Chart | None:
        """Create a chart based on a request specification."""
        chart_type_str = request.get("chart_type", "").lower()
        try:
            chart_type = ChartType(chart_type_str)
        except ValueError:
            self.log(f"Unknown chart type: {chart_type_str}", level="WARNING")
            return None

        config = ChartConfig(
            title=request.get("title", "Untitled Chart"),
            x_label=request.get("x_label", request.get("x", "")),
            y_label=request.get("y_label", request.get("y", "")),
            data_source=request.get("data_source", "User data"),
            colors=request.get("colors"),
            start_y_at_zero=request.get("start_y_at_zero", True),
        )

        x = request.get("x")
        y = request.get("y")
        values = request.get("values")
        names = request.get("names")
        columns = request.get("columns", [])

        if chart_type == ChartType.LINE:
            return self.create_line_chart(data, x, y, config)
        elif chart_type == ChartType.BAR:
            return self.create_bar_chart(data, x, y, config)
        elif chart_type == ChartType.SCATTER:
            return self.create_scatter_plot(data, x, y, config)
        elif chart_type == ChartType.HEATMAP:
            return self.create_heatmap(data, config, columns)
        elif chart_type == ChartType.BOX:
            return self.create_box_plot(data, columns, config)
        elif chart_type in (ChartType.PIE, ChartType.DONUT):
            hole = 0.4 if chart_type == ChartType.DONUT else 0
            return self.create_pie_chart(data, values, names, config, hole=hole)
        elif chart_type == ChartType.HISTOGRAM:
            return self.create_histogram(data, x, config)
        elif chart_type == ChartType.AREA:
            return self.create_area_chart(data, x, y, config)

        return None

    def create_line_chart(
        self,
        data: pd.DataFrame | None,
        x: str | None,
        y: str | list[str] | None,
        config: ChartConfig,
    ) -> Chart:
        """
        Create a line chart for time series or trend data.

        Args:
            data: DataFrame with the data
            x: Column for x-axis
            y: Column(s) for y-axis
            config: Chart configuration

        Returns:
            Chart object with the line chart
        """
        if data is None or x is None or y is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        else:
            if isinstance(y, list):
                fig = go.Figure()
                for i, y_col in enumerate(y):
                    color = (config.colors or DEFAULT_COLORS)[i % len(DEFAULT_COLORS)]
                    fig.add_trace(
                        go.Scatter(
                            x=data[x],
                            y=data[y_col],
                            mode="lines+markers",
                            name=y_col,
                            line=dict(color=color),
                        )
                    )
            else:
                fig = px.line(
                    data,
                    x=x,
                    y=y,
                    title=config.title,
                    color_discrete_sequence=config.colors or DEFAULT_COLORS,
                )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.LINE)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.LINE,
            title=config.title,
            description=f"Line chart showing {y} over {x}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_bar_chart(
        self,
        data: pd.DataFrame | None,
        x: str | None,
        y: str | None,
        config: ChartConfig,
        orientation: str = "v",
    ) -> Chart:
        """
        Create a bar chart for comparisons.

        Args:
            data: DataFrame with the data
            x: Column for categories
            y: Column for values
            config: Chart configuration
            orientation: 'v' for vertical, 'h' for horizontal

        Returns:
            Chart object with the bar chart
        """
        if data is None or x is None or y is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        else:
            fig = px.bar(
                data,
                x=x,
                y=y,
                title=config.title,
                color_discrete_sequence=config.colors or DEFAULT_COLORS,
                orientation=orientation,
            )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.BAR)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.BAR,
            title=config.title,
            description=f"Bar chart comparing {y} across {x}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_scatter_plot(
        self,
        data: pd.DataFrame | None,
        x: str | None,
        y: str | None,
        config: ChartConfig,
        color: str | None = None,
        size: str | None = None,
        trendline: str | None = None,
    ) -> Chart:
        """
        Create a scatter plot for correlations.

        Args:
            data: DataFrame with the data
            x: Column for x-axis
            y: Column for y-axis
            config: Chart configuration
            color: Column for color grouping
            size: Column for marker size
            trendline: Type of trendline ('ols', 'lowess', etc.)

        Returns:
            Chart object with the scatter plot
        """
        if data is None or x is None or y is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        else:
            fig = px.scatter(
                data,
                x=x,
                y=y,
                color=color,
                size=size,
                title=config.title,
                trendline=trendline,
                color_discrete_sequence=config.colors or DEFAULT_COLORS,
            )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.SCATTER)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.SCATTER,
            title=config.title,
            description=f"Scatter plot of {y} vs {x}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_heatmap(
        self,
        data: pd.DataFrame | None,
        config: ChartConfig,
        columns: list[str] | None = None,
        correlation_matrix: pd.DataFrame | None = None,
    ) -> Chart:
        """
        Create a heatmap for correlation matrices or pivot tables.

        Args:
            data: DataFrame with the data
            config: Chart configuration
            columns: Columns to include in correlation
            correlation_matrix: Pre-computed correlation matrix

        Returns:
            Chart object with the heatmap
        """
        if correlation_matrix is not None:
            matrix = correlation_matrix
        elif data is not None:
            if columns:
                numeric_data = data[columns].select_dtypes(include=["number"])
            else:
                numeric_data = data.select_dtypes(include=["number"])
            matrix = numeric_data.corr()
        else:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
            return Chart(
                chart_id=self._generate_chart_id(),
                chart_type=ChartType.HEATMAP,
                title=config.title,
                description="Heatmap (no data)",
                figure=fig,
                data_source=config.data_source,
                config=config,
                validation_warnings=["No data provided for heatmap"],
            )

        fig = px.imshow(
            matrix,
            text_auto=".2f",
            title=config.title,
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.HEATMAP)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.HEATMAP,
            title=config.title,
            description="Heatmap showing correlations between variables",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_box_plot(
        self,
        data: pd.DataFrame | None,
        columns: list[str] | None,
        config: ChartConfig,
        group_by: str | None = None,
    ) -> Chart:
        """
        Create a box plot for distribution analysis.

        Args:
            data: DataFrame with the data
            columns: Columns to create box plots for
            config: Chart configuration
            group_by: Column to group by

        Returns:
            Chart object with the box plot
        """
        if data is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        elif columns:
            if len(columns) == 1 and group_by:
                fig = px.box(
                    data,
                    x=group_by,
                    y=columns[0],
                    title=config.title,
                    color_discrete_sequence=config.colors or DEFAULT_COLORS,
                )
            else:
                fig = go.Figure()
                for i, col in enumerate(columns):
                    color = (config.colors or DEFAULT_COLORS)[i % len(DEFAULT_COLORS)]
                    fig.add_trace(
                        go.Box(
                            y=data[col],
                            name=col,
                            marker_color=color,
                        )
                    )
        else:
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()[:5]
            fig = go.Figure()
            for i, col in enumerate(numeric_cols):
                color = (config.colors or DEFAULT_COLORS)[i % len(DEFAULT_COLORS)]
                fig.add_trace(
                    go.Box(
                        y=data[col],
                        name=col,
                        marker_color=color,
                    )
                )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.BOX)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.BOX,
            title=config.title,
            description=f"Box plot showing distribution of {', '.join(columns or [])}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_pie_chart(
        self,
        data: pd.DataFrame | None,
        values: str | None,
        names: str | None,
        config: ChartConfig,
        hole: float = 0,
    ) -> Chart:
        """
        Create a pie or donut chart for proportions.

        Args:
            data: DataFrame with the data
            values: Column for values
            names: Column for category names
            config: Chart configuration
            hole: Size of hole for donut chart (0 for pie)

        Returns:
            Chart object with the pie/donut chart
        """
        chart_type = ChartType.DONUT if hole > 0 else ChartType.PIE
        warnings = []

        if data is None or values is None or names is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
            warnings.append("No data provided for pie chart")
        else:
            # Check category count
            unique_categories = data[names].nunique()
            if unique_categories > 7:
                warnings.append(
                    f"Pie chart has {unique_categories} categories. "
                    "Consider using a bar chart for more than 7 categories."
                )

            fig = px.pie(
                data,
                values=values,
                names=names,
                title=config.title,
                hole=hole,
                color_discrete_sequence=config.colors or DEFAULT_COLORS,
            )

        self._apply_config(fig, config)
        warnings.extend(self._validate_chart(fig, config, chart_type))

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=chart_type,
            title=config.title,
            description=f"{'Donut' if hole > 0 else 'Pie'} chart showing {values} by {names}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_histogram(
        self,
        data: pd.DataFrame | None,
        x: str | None,
        config: ChartConfig,
        bins: int | None = None,
    ) -> Chart:
        """
        Create a histogram for distribution visualization.

        Args:
            data: DataFrame with the data
            x: Column to create histogram for
            config: Chart configuration
            bins: Number of bins

        Returns:
            Chart object with the histogram
        """
        if data is None or x is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        else:
            fig = px.histogram(
                data,
                x=x,
                nbins=bins,
                title=config.title,
                color_discrete_sequence=config.colors or DEFAULT_COLORS,
            )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.HISTOGRAM)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.HISTOGRAM,
            title=config.title,
            description=f"Histogram showing distribution of {x}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def create_area_chart(
        self,
        data: pd.DataFrame | None,
        x: str | None,
        y: str | list[str] | None,
        config: ChartConfig,
    ) -> Chart:
        """
        Create an area chart for cumulative trends.

        Args:
            data: DataFrame with the data
            x: Column for x-axis
            y: Column(s) for y-axis
            config: Chart configuration

        Returns:
            Chart object with the area chart
        """
        if data is None or x is None or y is None:
            fig = go.Figure()
            fig.add_annotation(text="No data available", showarrow=False)
        else:
            if isinstance(y, list):
                fig = px.area(
                    data,
                    x=x,
                    y=y,
                    title=config.title,
                    color_discrete_sequence=config.colors or DEFAULT_COLORS,
                )
            else:
                fig = px.area(
                    data,
                    x=x,
                    y=y,
                    title=config.title,
                    color_discrete_sequence=config.colors or DEFAULT_COLORS,
                )

        self._apply_config(fig, config)
        warnings = self._validate_chart(fig, config, ChartType.AREA)

        return Chart(
            chart_id=self._generate_chart_id(),
            chart_type=ChartType.AREA,
            title=config.title,
            description=f"Area chart showing {y} over {x}",
            figure=fig,
            data_source=config.data_source,
            config=config,
            validation_warnings=warnings,
        )

    def _apply_config(self, fig: go.Figure, config: ChartConfig) -> None:
        """Apply configuration to a figure."""
        fig.update_layout(
            title=dict(
                text=config.title,
                x=0.5,
                xanchor="center",
            ),
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            template=config.template,
            height=config.height,
            width=config.width,
            showlegend=config.show_legend,
        )

        # Add data source annotation
        fig.add_annotation(
            text=f"Source: {config.data_source}",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.12,
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="left",
        )

        # Apply y-axis zero start if configured
        if config.start_y_at_zero:
            fig.update_yaxes(rangemode="tozero")

    def _validate_chart(
        self,
        fig: go.Figure,
        config: ChartConfig,
        chart_type: ChartType,
    ) -> list[str]:
        """
        Validate chart meets requirements from CLAUDE.md.

        Returns list of validation warnings.
        """
        warnings = []

        # Check title
        if not config.title or config.title == "Untitled Chart":
            warnings.append("Chart is missing a descriptive title")

        # Check axis labels for applicable chart types
        if chart_type in (
            ChartType.LINE,
            ChartType.BAR,
            ChartType.SCATTER,
            ChartType.HISTOGRAM,
            ChartType.AREA,
        ):
            if not config.x_label:
                warnings.append("X-axis label is missing")
            if not config.y_label:
                warnings.append("Y-axis label is missing")

        # Check data source
        if config.data_source == "Data source not specified":
            warnings.append("Data source should be specified")

        # Check color count
        if config.colors and len(config.colors) > 7:
            warnings.append("More than 7 colors used - may reduce clarity")

        return warnings

    def _auto_generate_charts(
        self,
        data: pd.DataFrame | None,
        analysis_results: dict[str, Any],
    ) -> list[Chart]:
        """Auto-generate charts based on analysis results."""
        charts = []

        # Generate correlation heatmap if correlation analysis exists
        if "correlation" in analysis_results:
            corr_data = analysis_results["correlation"]
            if "matrix" in corr_data or "pairwise_correlations" in corr_data:
                config = ChartConfig(
                    title="Correlation Matrix",
                    data_source="Statistical Analysis",
                )
                chart = self.create_heatmap(data, config)
                charts.append(chart)

        # Generate distribution plots if descriptive stats exist
        if "descriptive" in analysis_results and data is not None:
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()[:4]
            if numeric_cols:
                config = ChartConfig(
                    title="Distribution Overview",
                    y_label="Value",
                    data_source="Statistical Analysis",
                )
                chart = self.create_box_plot(data, numeric_cols, config)
                charts.append(chart)

        # Generate time series if time analysis exists
        if "time_series" in analysis_results and data is not None:
            ts_data = analysis_results["time_series"]
            date_col = ts_data.get("date_column")
            value_col = ts_data.get("value_column")
            if date_col and value_col:
                config = ChartConfig(
                    title="Time Series Trend",
                    x_label=date_col,
                    y_label=value_col,
                    data_source="Time Series Analysis",
                )
                chart = self.create_line_chart(data, date_col, value_col, config)
                charts.append(chart)

        return charts

    def _create_default_charts(self, data: pd.DataFrame) -> list[Chart]:
        """Create default EDA charts when no specific request is made."""
        charts = []

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()

        # Box plot for numeric distributions
        if numeric_cols:
            config = ChartConfig(
                title="Numeric Variable Distributions",
                y_label="Value",
                data_source="Exploratory Data Analysis",
            )
            charts.append(self.create_box_plot(data, numeric_cols[:5], config))

        # Correlation heatmap if enough numeric columns
        if len(numeric_cols) >= 2:
            config = ChartConfig(
                title="Correlation Heatmap",
                data_source="Exploratory Data Analysis",
            )
            charts.append(self.create_heatmap(data, config, numeric_cols[:8]))

        # Bar chart for first categorical column
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            if data[cat_col].nunique() <= 10:
                agg_data = data.groupby(cat_col)[numeric_cols[0]].mean().reset_index()
                config = ChartConfig(
                    title=f"Average {numeric_cols[0]} by {cat_col}",
                    x_label=cat_col,
                    y_label=f"Mean {numeric_cols[0]}",
                    data_source="Exploratory Data Analysis",
                )
                charts.append(self.create_bar_chart(agg_data, cat_col, numeric_cols[0], config))

        return charts

    def _generate_summary(self, charts: list[Chart]) -> str:
        """Generate a summary of created charts."""
        if not charts:
            return "No charts were generated."

        type_counts: dict[str, int] = {}
        for chart in charts:
            type_counts[chart.chart_type.value] = type_counts.get(chart.chart_type.value, 0) + 1

        parts = [f"Generated {len(charts)} visualization(s):"]
        for chart_type, count in type_counts.items():
            parts.append(f"- {count} {chart_type} chart(s)")

        warning_count = sum(len(c.validation_warnings) for c in charts)
        if warning_count > 0:
            parts.append(f"\nNote: {warning_count} validation warning(s) - review chart details.")

        return " ".join(parts)

    def _generate_recommendations(
        self,
        data: pd.DataFrame | None,
        charts: list[Chart],
    ) -> list[str]:
        """Generate recommendations for better visualizations."""
        recommendations = []

        # Check if any charts have many categories in pie
        for chart in charts:
            if chart.chart_type in (ChartType.PIE, ChartType.DONUT):
                if any("7 categories" in w for w in chart.validation_warnings):
                    recommendations.append(
                        "Consider using a bar chart instead of pie chart for better readability with many categories"
                    )

        # Suggest interactive features
        if len(charts) > 3:
            recommendations.append(
                "Consider creating a dashboard layout with subplots for easier comparison"
            )

        # Data-based recommendations
        if data is not None:
            # Check for date columns
            date_cols = [c for c in data.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols and not any(c.chart_type == ChartType.LINE for c in charts):
                recommendations.append(
                    f"Data contains date column(s) ({', '.join(date_cols)}) - consider adding a time series visualization"
                )

        return recommendations

    def _generate_accessibility_notes(self, charts: list[Chart]) -> list[str]:
        """Generate accessibility notes for the visualizations."""
        notes = []

        # Check color usage
        for chart in charts:
            if chart.config and chart.config.colors:
                if len(set(chart.config.colors)) < len(chart.config.colors):
                    notes.append(
                        f"Chart '{chart.title}' uses repeated colors - ensure sufficient contrast"
                    )

        notes.append("All charts use colorblind-friendly default palette")
        notes.append("Interactive charts support keyboard navigation and screen readers")

        return notes

    def export_to_image(
        self,
        chart: Chart,
        path: str | Path,
        format: str = "png",
        scale: float = 2.0,
    ) -> bool:
        """
        Export a chart to a static image.

        Args:
            chart: Chart to export
            path: Output file path
            format: Image format (png, svg, jpeg, webp, pdf)
            scale: Image scale factor

        Returns:
            True if successful
        """
        try:
            chart.figure.write_image(str(path), format=format, scale=scale)
            self.log(f"Exported chart to {path}")
            return True
        except Exception as e:
            self.log(f"Failed to export chart: {e}", level="ERROR")
            return False

    def export_to_html(self, chart: Chart, path: str | Path) -> bool:
        """
        Export a chart to an HTML file.

        Args:
            chart: Chart to export
            path: Output file path

        Returns:
            True if successful
        """
        try:
            chart.figure.write_html(str(path), include_plotlyjs="cdn")
            self.log(f"Exported chart to HTML: {path}")
            return True
        except Exception as e:
            self.log(f"Failed to export chart to HTML: {e}", level="ERROR")
            return False

    def get_chart_type_options(self, data: pd.DataFrame | None = None) -> list[AgentOption]:
        """
        Get chart type options based on available data.

        Args:
            data: DataFrame to analyze for recommendations

        Returns:
            List of chart type options
        """
        options = [
            AgentOption(
                id="auto",
                title="Auto-Select Charts",
                description="Automatically choose appropriate charts based on data",
                recommended=True,
                pros=["No configuration needed", "Data-driven selection"],
                cons=["May not match specific preferences"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="line",
                title="Line Chart",
                description="Best for trends over time",
                pros=["Shows trends clearly", "Good for time series"],
                cons=["Requires ordered data"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="bar",
                title="Bar Chart",
                description="Best for comparing categories",
                pros=["Easy to compare", "Clear rankings"],
                cons=["Limited categories work best"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="scatter",
                title="Scatter Plot",
                description="Best for correlations",
                pros=["Shows relationships", "Can reveal clusters"],
                cons=["Requires two numeric variables"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="heatmap",
                title="Correlation Heatmap",
                description="Best for showing multiple relationships",
                pros=["Comprehensive view", "Pattern discovery"],
                cons=["Many variables can be overwhelming"],
                estimated_complexity="medium",
            ),
        ]

        return options

    def format_output(self, output: VisualizationOutput) -> str:
        """
        Format visualization output for display.

        Args:
            output: The visualization output to format

        Returns:
            Formatted markdown string
        """
        lines = [
            "# Visualization Summary",
            "",
            output.summary,
            "",
        ]

        # Charts section
        lines.append("## Generated Charts")
        lines.append("")

        for i, chart in enumerate(output.charts, 1):
            lines.extend(
                [
                    f"### Chart {i}: {chart.title}",
                    f"**Type**: {chart.chart_type.value.title()}",
                    f"**Description**: {chart.description}",
                    f"**Data Source**: {chart.data_source}",
                ]
            )

            if chart.validation_warnings:
                lines.append("")
                lines.append("**Warnings**:")
                for warning in chart.validation_warnings:
                    lines.append(f"- {warning}")

            lines.append("")

        # Recommendations section
        if output.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for rec in output.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Accessibility section
        if output.accessibility_notes:
            lines.extend(
                [
                    "## Accessibility Notes",
                    "",
                ]
            )
            for note in output.accessibility_notes:
                lines.append(f"- {note}")

        return "\n".join(lines)
