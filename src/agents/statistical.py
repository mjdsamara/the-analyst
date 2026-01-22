"""
Statistical Agent for The Analyst platform.

Performs rigorous statistical analysis with full methodology transparency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.prompts.agents import STATISTICAL_PROMPT
from src.tools.statistics import StatisticsToolkit


class AnalysisType(str, Enum):
    """Types of statistical analysis available."""

    DESCRIPTIVE = "descriptive"
    DISTRIBUTION = "distribution"
    CORRELATION = "correlation"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    TIME_SERIES = "time_series"
    COMPREHENSIVE = "comprehensive"


@dataclass
class AnalysisResult:
    """Result from a statistical analysis."""

    analysis_type: AnalysisType
    methodology: str
    assumptions_checked: dict[str, bool]
    results: dict[str, Any]
    confidence_level: float
    interpretation: str
    limitations: list[str]
    reproducibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_type": self.analysis_type.value,
            "methodology": self.methodology,
            "assumptions_checked": self.assumptions_checked,
            "results": self.results,
            "confidence_level": self.confidence_level,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
            "reproducibility": self.reproducibility,
        }


@dataclass
class StatisticalAnalysisOutput:
    """Complete output from the statistical agent."""

    analyses: list[AnalysisResult] = field(default_factory=list)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyses": [a.to_dict() for a in self.analyses],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class StatisticalAgent(BaseAgent):
    """
    Agent responsible for statistical analysis and EDA.

    Single Job: Perform rigorous statistical analysis with full methodology transparency.
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the statistical agent."""
        super().__init__(name="statistical", context=context)
        self.toolkit = StatisticsToolkit()

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return STATISTICAL_PROMPT

    async def execute(
        self,
        data: pd.DataFrame | None = None,
        analysis_type: str | AnalysisType | None = None,
        columns: list[str] | None = None,
        target_column: str | None = None,
        group_column: str | None = None,
        **kwargs: Any,
    ) -> AgentResult[StatisticalAnalysisOutput]:
        """
        Execute statistical analysis.

        Args:
            data: DataFrame to analyze
            analysis_type: Type of analysis to perform
            columns: Specific columns to analyze (None = all numeric)
            target_column: Target column for hypothesis testing
            group_column: Column for grouping comparisons

        Returns:
            AgentResult containing the analysis output
        """
        if data is None:
            return AgentResult.error_result("No data provided for analysis")

        if data.empty:
            return AgentResult.error_result("Data is empty")

        # Convert string to enum if needed
        if isinstance(analysis_type, str):
            try:
                analysis_type = AnalysisType(analysis_type.lower())
            except ValueError:
                analysis_type = AnalysisType.COMPREHENSIVE

        if analysis_type is None:
            analysis_type = AnalysisType.COMPREHENSIVE

        self.log(f"Starting {analysis_type.value} analysis")

        # Get columns to analyze
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if columns:
            # Filter to only valid numeric columns
            columns = [c for c in columns if c in numeric_cols]
            if not columns:
                return AgentResult.error_result(
                    f"None of the specified columns are numeric. Available numeric columns: {numeric_cols}"
                )
        else:
            columns = numeric_cols

        if not columns:
            return AgentResult.error_result("No numeric columns found in data")

        # Perform analysis based on type
        output = StatisticalAnalysisOutput()

        try:
            if analysis_type == AnalysisType.DESCRIPTIVE:
                analysis = self._run_descriptive_analysis(data, columns)
                output.analyses.append(analysis)

            elif analysis_type == AnalysisType.DISTRIBUTION:
                analysis = self._run_distribution_analysis(data, columns)
                output.analyses.append(analysis)

            elif analysis_type == AnalysisType.CORRELATION:
                analysis = self._run_correlation_analysis(data, columns)
                output.analyses.append(analysis)

            elif analysis_type == AnalysisType.HYPOTHESIS_TESTING:
                if not target_column:
                    return AgentResult.error_result(
                        "target_column is required for hypothesis testing"
                    )
                analysis = self._run_hypothesis_testing(data, target_column, group_column)
                output.analyses.append(analysis)

            elif analysis_type == AnalysisType.TIME_SERIES:
                if not target_column:
                    return AgentResult.error_result(
                        "target_column is required for time series analysis"
                    )
                analysis = self._run_time_series_analysis(data, target_column)
                output.analyses.append(analysis)

            elif analysis_type == AnalysisType.COMPREHENSIVE:
                # Run multiple analyses
                output.analyses.append(self._run_descriptive_analysis(data, columns))
                output.analyses.append(self._run_distribution_analysis(data, columns))
                if len(columns) >= 2:
                    output.analyses.append(self._run_correlation_analysis(data, columns))

            # Generate summary
            output.summary = self._generate_summary(output.analyses)
            output.recommendations = self._generate_recommendations(output.analyses)

            self.log(f"Completed {len(output.analyses)} analyses")

            return AgentResult.success_result(
                output,
                analysis_type=analysis_type.value,
                columns_analyzed=columns,
                row_count=len(data),
            )

        except Exception as e:
            self.log(f"Analysis failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Analysis failed: {e}")

    def _run_descriptive_analysis(self, data: pd.DataFrame, columns: list[str]) -> AnalysisResult:
        """Run descriptive statistics analysis."""
        results: dict[str, Any] = {}
        assumptions = {"sufficient_sample_size": len(data) >= 30}

        for col in columns:
            stats_results = self.toolkit.descriptive_stats(data[col], name=col)
            results[col] = {
                name: {
                    "value": r.value,
                    "confidence_interval": r.confidence_interval,
                    "interpretation": r.interpretation,
                }
                for name, r in stats_results.items()
            }

        # Add overall statistics
        results["_overall"] = {
            "total_observations": len(data),
            "columns_analyzed": len(columns),
            "missing_values": data[columns].isna().sum().to_dict(),
        }

        interpretation = self._interpret_descriptive(results, columns)

        return AnalysisResult(
            analysis_type=AnalysisType.DESCRIPTIVE,
            methodology="Descriptive statistics with sample mean confidence intervals using t-distribution",
            assumptions_checked=assumptions,
            results=results,
            confidence_level=self.toolkit.confidence_level,
            interpretation=interpretation,
            limitations=self._get_descriptive_limitations(data, columns),
            reproducibility={
                "confidence_level": self.toolkit.confidence_level,
                "columns": columns,
                "n_observations": len(data),
            },
        )

    def _run_distribution_analysis(self, data: pd.DataFrame, columns: list[str]) -> AnalysisResult:
        """Run distribution analysis with normality tests."""
        results = {}
        assumptions = {"sample_size_adequate": len(data) >= 20}

        for col in columns:
            col_data = data[col].dropna()
            if len(col_data) < 8:  # Shapiro-Wilk requires at least 3, but 8+ is better
                continue

            normality = self.toolkit.normality_test(col_data)

            # Calculate percentiles
            percentiles = np.percentile(col_data, [5, 25, 50, 75, 95])

            results[col] = {
                "normality_test": {
                    "statistic": normality.value,
                    "p_value": normality.p_value,
                    "is_normal": (
                        normality.p_value > self.toolkit.alpha if normality.p_value else None
                    ),
                    "interpretation": normality.interpretation,
                },
                "percentiles": {
                    "p5": percentiles[0],
                    "p25": percentiles[1],
                    "p50": percentiles[2],
                    "p75": percentiles[3],
                    "p95": percentiles[4],
                },
                "range": {
                    "min": col_data.min(),
                    "max": col_data.max(),
                    "iqr": percentiles[3] - percentiles[1],
                },
            }

        interpretation = self._interpret_distribution(results)

        return AnalysisResult(
            analysis_type=AnalysisType.DISTRIBUTION,
            methodology="Distribution analysis using Shapiro-Wilk normality test and percentile analysis",
            assumptions_checked=assumptions,
            results=results,
            confidence_level=self.toolkit.confidence_level,
            interpretation=interpretation,
            limitations=[
                "Shapiro-Wilk test may be sensitive to small deviations in large samples",
                "Non-normality does not necessarily invalidate all statistical methods",
            ],
            reproducibility={
                "normality_test": "shapiro",
                "alpha": self.toolkit.alpha,
                "columns": columns,
            },
        )

    def _run_correlation_analysis(self, data: pd.DataFrame, columns: list[str]) -> AnalysisResult:
        """Run correlation analysis between numeric columns."""
        results: dict[str, Any] = {"pairwise_correlations": [], "correlation_matrix": {}}
        assumptions = {
            "sufficient_pairs": len(data) >= 10,
            "at_least_two_columns": len(columns) >= 2,
        }

        # Compute pairwise correlations
        for i, col1 in enumerate(columns):
            for col2 in columns[i + 1 :]:
                corr_result = self.toolkit.correlation(data[col1], data[col2])
                results["pairwise_correlations"].append(
                    {
                        "variable_1": col1,
                        "variable_2": col2,
                        "correlation": corr_result.value,
                        "confidence_interval": corr_result.confidence_interval,
                        "p_value": corr_result.p_value,
                        "effect_size_r_squared": corr_result.effect_size,
                        "interpretation": corr_result.interpretation,
                    }
                )

        # Correlation matrix
        corr_matrix = data[columns].corr()
        results["correlation_matrix"] = corr_matrix.to_dict()

        # Find strongest correlations
        strong_corrs = [c for c in results["pairwise_correlations"] if abs(c["correlation"]) >= 0.5]

        interpretation = self._interpret_correlation(results, strong_corrs)

        return AnalysisResult(
            analysis_type=AnalysisType.CORRELATION,
            methodology="Pearson correlation with Fisher z-transformation confidence intervals",
            assumptions_checked=assumptions,
            results=results,
            confidence_level=self.toolkit.confidence_level,
            interpretation=interpretation,
            limitations=[
                "Pearson correlation assumes linear relationships",
                "Correlation does not imply causation",
                "Outliers can significantly affect correlation coefficients",
            ],
            reproducibility={
                "method": "pearson",
                "confidence_level": self.toolkit.confidence_level,
                "columns": columns,
            },
        )

    def _run_hypothesis_testing(
        self,
        data: pd.DataFrame,
        target_column: str,
        group_column: str | None = None,
    ) -> AnalysisResult:
        """Run hypothesis testing."""
        results = {}
        assumptions = {"sample_size_adequate": len(data) >= 20}

        if group_column and group_column in data.columns:
            # Group comparison
            groups = data[group_column].unique()
            if len(groups) == 2:
                group1_data = data[data[group_column] == groups[0]][target_column]
                group2_data = data[data[group_column] == groups[1]][target_column]

                t_result = self.toolkit.t_test(group1_data, group2_data)

                results["t_test"] = {
                    "test_type": "independent_samples",
                    "group_1": str(groups[0]),
                    "group_2": str(groups[1]),
                    "statistic": t_result.value,
                    "p_value": t_result.p_value,
                    "effect_size_cohens_d": t_result.effect_size,
                    "interpretation": t_result.interpretation,
                }

                # Check normality assumption
                norm1 = self.toolkit.normality_test(group1_data.dropna())
                norm2 = self.toolkit.normality_test(group2_data.dropna())
                assumptions["group1_normal"] = (
                    norm1.p_value > self.toolkit.alpha if norm1.p_value else False
                )
                assumptions["group2_normal"] = (
                    norm2.p_value > self.toolkit.alpha if norm2.p_value else False
                )

            elif len(groups) > 2:
                # ANOVA-style comparison
                group_data = [data[data[group_column] == g][target_column].dropna() for g in groups]
                f_stat, p_value = stats.f_oneway(*group_data)

                # Effect size (eta-squared)
                ss_between = sum(
                    len(g) * (g.mean() - data[target_column].mean()) ** 2 for g in group_data
                )
                ss_total = sum((data[target_column] - data[target_column].mean()) ** 2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                results["anova"] = {
                    "test_type": "one_way_anova",
                    "groups": [str(g) for g in groups],
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "effect_size_eta_squared": eta_squared,
                    "interpretation": self._interpret_anova(p_value, eta_squared),
                }
        else:
            # One-sample t-test against mean
            t_result = self.toolkit.t_test(data[target_column])
            results["one_sample_t_test"] = {
                "test_type": "one_sample",
                "statistic": t_result.value,
                "p_value": t_result.p_value,
                "effect_size_cohens_d": t_result.effect_size,
                "interpretation": t_result.interpretation,
            }

        interpretation = self._interpret_hypothesis_test(results)

        return AnalysisResult(
            analysis_type=AnalysisType.HYPOTHESIS_TESTING,
            methodology="Hypothesis testing with appropriate test selection based on group structure",
            assumptions_checked=assumptions,
            results=results,
            confidence_level=self.toolkit.confidence_level,
            interpretation=interpretation,
            limitations=[
                "Statistical significance does not imply practical significance",
                "Effect sizes should be considered alongside p-values",
                "Multiple comparisons may require correction (e.g., Bonferroni)",
            ],
            reproducibility={
                "alpha": self.toolkit.alpha,
                "target_column": target_column,
                "group_column": group_column,
            },
        )

    def _run_time_series_analysis(self, data: pd.DataFrame, target_column: str) -> AnalysisResult:
        """Run basic time series analysis."""
        results = {}
        assumptions = {"ordered_data": True, "sufficient_observations": len(data) >= 10}

        series = data[target_column].dropna()

        # Basic time series statistics
        results["series_stats"] = {
            "n_observations": len(series),
            "start_value": series.iloc[0] if len(series) > 0 else None,
            "end_value": series.iloc[-1] if len(series) > 0 else None,
            "mean": series.mean(),
            "std": series.std(),
        }

        # Calculate changes
        diff = series.diff().dropna()
        results["changes"] = {
            "mean_change": diff.mean(),
            "std_change": diff.std(),
            "positive_changes": (diff > 0).sum(),
            "negative_changes": (diff < 0).sum(),
            "total_change": series.iloc[-1] - series.iloc[0] if len(series) > 1 else 0,
            "percent_change": (
                ((series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100)
                if len(series) > 1 and series.iloc[0] != 0
                else None
            ),
        }

        # Rolling statistics (if enough data)
        if len(series) >= 7:
            rolling_mean = series.rolling(window=7).mean()
            rolling_std = series.rolling(window=7).std()
            results["rolling_7"] = {
                "mean_of_means": rolling_mean.dropna().mean(),
                "mean_of_stds": rolling_std.dropna().mean(),
            }

        # Autocorrelation (lag-1)
        if len(series) >= 10:
            acf_1 = series.autocorr(lag=1)
            results["autocorrelation"] = {
                "lag_1": acf_1,
                "interpretation": (
                    "Strong persistence"
                    if abs(acf_1) > 0.7
                    else ("Moderate persistence" if abs(acf_1) > 0.3 else "Weak persistence")
                ),
            }

        interpretation = self._interpret_time_series(results)

        return AnalysisResult(
            analysis_type=AnalysisType.TIME_SERIES,
            methodology="Basic time series analysis with change statistics and autocorrelation",
            assumptions_checked=assumptions,
            results=results,
            confidence_level=self.toolkit.confidence_level,
            interpretation=interpretation,
            limitations=[
                "Basic analysis - for forecasting, use specialized models (Prophet, ARIMA)",
                "Assumes data is ordered chronologically",
                "Does not account for seasonality or trend decomposition",
            ],
            reproducibility={
                "target_column": target_column,
                "n_observations": len(series),
            },
        )

    def _interpret_descriptive(self, results: dict[str, Any], columns: list[str]) -> str:
        """Generate interpretation for descriptive statistics."""
        interpretations = []
        for col in columns:
            if col in results and col != "_overall":
                mean_info = results[col].get("mean", {})
                skew_info = results[col].get("skewness", {})

                mean_val = mean_info.get("value")
                ci = mean_info.get("confidence_interval")
                skew_interp = skew_info.get("interpretation", "")

                if mean_val is not None and ci:
                    interpretations.append(
                        f"{col}: Mean = {mean_val:.2f} (95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]). {skew_interp}."
                    )

        return " ".join(interpretations) if interpretations else "No interpretable results."

    def _interpret_distribution(self, results: dict[str, Any]) -> str:
        """Generate interpretation for distribution analysis."""
        normal_cols = []
        non_normal_cols = []

        for col, data in results.items():
            if "normality_test" in data:
                if data["normality_test"].get("is_normal"):
                    normal_cols.append(col)
                else:
                    non_normal_cols.append(col)

        parts = []
        if normal_cols:
            parts.append(f"Normally distributed: {', '.join(normal_cols)}")
        if non_normal_cols:
            parts.append(f"Non-normal distribution: {', '.join(non_normal_cols)}")

        return ". ".join(parts) if parts else "Distribution analysis complete."

    def _interpret_correlation(
        self, results: dict[str, Any], strong_corrs: list[dict[str, Any]]
    ) -> str:
        """Generate interpretation for correlation analysis."""
        if not strong_corrs:
            return "No strong correlations (|r| >= 0.5) found between variables."

        parts = []
        for corr in strong_corrs:
            direction = "positive" if corr["correlation"] > 0 else "negative"
            parts.append(
                f"{corr['variable_1']} and {corr['variable_2']} show a strong {direction} "
                f"correlation (r = {corr['correlation']:.3f}, p = {corr['p_value']:.4f})"
            )

        return ". ".join(parts)

    def _interpret_anova(self, p_value: float, eta_squared: float) -> str:
        """Interpret ANOVA results."""
        significant = p_value < self.toolkit.alpha

        if eta_squared < 0.01:
            effect = "negligible"
        elif eta_squared < 0.06:
            effect = "small"
        elif eta_squared < 0.14:
            effect = "medium"
        else:
            effect = "large"

        if significant:
            return f"Significant difference between groups with {effect} effect size"
        else:
            return f"No significant difference between groups (effect size: {effect})"

    def _interpret_hypothesis_test(self, results: dict[str, Any]) -> str:
        """Generate interpretation for hypothesis testing."""
        parts = []
        for test_name, test_data in results.items():
            if "interpretation" in test_data:
                parts.append(f"{test_name}: {test_data['interpretation']}")
        return ". ".join(parts) if parts else "Hypothesis testing complete."

    def _interpret_time_series(self, results: dict[str, Any]) -> str:
        """Generate interpretation for time series analysis."""
        parts = []

        changes = results.get("changes", {})
        if changes.get("percent_change") is not None:
            direction = "increased" if changes["percent_change"] > 0 else "decreased"
            parts.append(f"Series {direction} by {abs(changes['percent_change']):.1f}% overall")

        autocorr = results.get("autocorrelation", {})
        if "interpretation" in autocorr:
            parts.append(
                f"Shows {autocorr['interpretation'].lower()} (lag-1 ACF = {autocorr.get('lag_1', 0):.3f})"
            )

        return ". ".join(parts) if parts else "Time series analysis complete."

    def _get_descriptive_limitations(self, data: pd.DataFrame, columns: list[str]) -> list[str]:
        """Get limitations specific to this descriptive analysis."""
        limitations = []

        if len(data) < 30:
            limitations.append(
                f"Small sample size (n={len(data)}); confidence intervals may be unreliable"
            )

        missing_pct = data[columns].isna().sum().sum() / (len(data) * len(columns)) * 100
        if missing_pct > 5:
            limitations.append(
                f"High missing data rate ({missing_pct:.1f}%); results based on available data only"
            )

        return limitations

    def _generate_summary(self, analyses: list[AnalysisResult]) -> str:
        """Generate overall summary from all analyses."""
        parts = [f"Completed {len(analyses)} analysis type(s)."]

        for analysis in analyses:
            parts.append(f"- {analysis.analysis_type.value}: {analysis.interpretation}")

        return " ".join(parts)

    def _generate_recommendations(self, analyses: list[AnalysisResult]) -> list[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        for analysis in analyses:
            if analysis.analysis_type == AnalysisType.DISTRIBUTION:
                # Check for non-normal distributions
                for col, data in analysis.results.items():
                    if isinstance(data, dict) and "normality_test" in data:
                        if not data["normality_test"].get("is_normal"):
                            recommendations.append(
                                f"Consider non-parametric tests for '{col}' due to non-normal distribution"
                            )

            elif analysis.analysis_type == AnalysisType.CORRELATION:
                # Check for strong correlations
                for corr in analysis.results.get("pairwise_correlations", []):
                    if abs(corr.get("correlation", 0)) >= 0.8:
                        recommendations.append(
                            f"Investigate potential multicollinearity between "
                            f"'{corr['variable_1']}' and '{corr['variable_2']}'"
                        )

        return (
            recommendations
            if recommendations
            else ["No specific recommendations based on current analysis."]
        )

    def get_analysis_options(self, data: pd.DataFrame) -> list[AgentOption]:
        """
        Get available analysis options based on the data.

        Args:
            data: DataFrame to analyze

        Returns:
            List of analysis options for user selection
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        n_rows = len(data)

        options = [
            AgentOption(
                id="comprehensive",
                title="Comprehensive Analysis",
                description="Full EDA including descriptive stats, distributions, and correlations",
                recommended=True,
                pros=["Thorough coverage", "Identifies patterns", "Publication-ready"],
                cons=["More processing", "May include irrelevant findings"],
                estimated_complexity="medium",
            ),
            AgentOption(
                id="descriptive",
                title="Descriptive Statistics",
                description="Mean, median, std, percentiles with confidence intervals",
                pros=["Fast", "Easy to interpret", "Good starting point"],
                cons=["No relationship analysis", "Basic insights only"],
                estimated_complexity="low",
            ),
        ]

        if len(numeric_cols) >= 2:
            options.append(
                AgentOption(
                    id="correlation",
                    title="Correlation Analysis",
                    description="Pairwise correlations between all numeric variables",
                    pros=["Reveals relationships", "Identifies multicollinearity"],
                    cons=["Assumes linear relationships"],
                    estimated_complexity="low",
                )
            )

        if n_rows >= 20:
            options.append(
                AgentOption(
                    id="distribution",
                    title="Distribution Analysis",
                    description="Normality tests and percentile analysis",
                    pros=["Guides test selection", "Identifies outliers"],
                    cons=["Requires adequate sample size"],
                    estimated_complexity="low",
                )
            )

        return options

    def format_output(self, output: StatisticalAnalysisOutput) -> str:
        """
        Format analysis output for display.

        Args:
            output: The analysis output to format

        Returns:
            Formatted markdown string
        """
        lines = ["# Statistical Analysis Results", ""]

        for analysis in output.analyses:
            lines.extend(
                [
                    f"## {analysis.analysis_type.value.replace('_', ' ').title()}",
                    "",
                    "### Methodology",
                    analysis.methodology,
                    "",
                    "### Assumptions Checked",
                ]
            )

            for assumption, met in analysis.assumptions_checked.items():
                status = "Met" if met else "Violated"
                lines.append(f"- {assumption.replace('_', ' ').title()}: **{status}**")

            lines.extend(
                [
                    "",
                    "### Interpretation",
                    analysis.interpretation,
                    "",
                    f"### Confidence Level: {analysis.confidence_level:.0%}",
                    "",
                    "### Limitations",
                ]
            )

            for limitation in analysis.limitations:
                lines.append(f"- {limitation}")

            lines.append("")

        if output.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                ]
            )
            for rec in output.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.extend(
            [
                "## Summary",
                output.summary,
            ]
        )

        return "\n".join(lines)
