"""
Insights Agent for The Analyst platform.

Synthesizes findings from all analyses into actionable insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.prompts.agents import INSIGHTS_PROMPT


class ConfidenceLevel(str, Enum):
    """Confidence levels for insights."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightPriority(str, Enum):
    """Priority levels for insights."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """A single insight with supporting evidence."""

    title: str
    finding: str
    evidence: list[str]
    impact: str
    recommendation: str
    confidence: ConfidenceLevel
    priority: InsightPriority = InsightPriority.MEDIUM
    tags: list[str] = field(default_factory=list)
    related_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "finding": self.finding,
            "evidence": self.evidence,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "confidence": self.confidence.value,
            "priority": self.priority.value,
            "tags": self.tags,
            "related_columns": self.related_columns,
        }


@dataclass
class Anomaly:
    """A detected anomaly in the data or analysis."""

    description: str
    severity: str  # "high", "medium", "low"
    evidence: str
    potential_causes: list[str]
    recommended_investigation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
            "potential_causes": self.potential_causes,
            "recommended_investigation": self.recommended_investigation,
        }


@dataclass
class RecommendedAction:
    """A recommended action based on insights."""

    action: str
    expected_impact: str
    priority: int  # 1-5, 1 being highest
    effort: str  # "low", "medium", "high"
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "expected_impact": self.expected_impact,
            "priority": self.priority,
            "effort": self.effort,
            "dependencies": self.dependencies,
        }


@dataclass
class InsightsOutput:
    """Complete output from the insights agent."""

    insights: list[Insight] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    actions: list[RecommendedAction] = field(default_factory=list)
    questions_for_further_analysis: list[str] = field(default_factory=list)
    executive_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "insights": [i.to_dict() for i in self.insights],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "actions": [a.to_dict() for a in self.actions],
            "questions_for_further_analysis": self.questions_for_further_analysis,
            "executive_summary": self.executive_summary,
        }


class InsightsAgent(BaseAgent):
    """
    Agent responsible for synthesizing findings into actionable insights.

    Single Job: Synthesize findings from all analyses into actionable,
                stakeholder-ready insights.
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the insights agent."""
        super().__init__(name="insights", context=context)

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return INSIGHTS_PROMPT

    async def execute(
        self,
        analysis_results: dict[str, Any] | None = None,
        data_profile: dict[str, Any] | None = None,
        focus_areas: list[str] | None = None,
        business_context: str | None = None,
        **kwargs: Any,
    ) -> AgentResult[InsightsOutput]:
        """
        Execute insight synthesis.

        Args:
            analysis_results: Results from statistical/other analyses
            data_profile: Profile of the data being analyzed
            focus_areas: Specific areas to focus insights on
            business_context: Business context for relevance

        Returns:
            AgentResult containing synthesized insights
        """
        if not analysis_results and not data_profile:
            return AgentResult.error_result(
                "No analysis results or data profile provided for insight generation"
            )

        self.log("Starting insight synthesis")

        output = InsightsOutput()

        try:
            # Generate insights from analysis results
            if analysis_results:
                insights = self._extract_insights_from_analysis(analysis_results, focus_areas)
                output.insights.extend(insights)

                anomalies = self._detect_anomalies(analysis_results)
                output.anomalies.extend(anomalies)

            # Generate insights from data profile
            if data_profile:
                profile_insights = self._extract_insights_from_profile(data_profile)
                output.insights.extend(profile_insights)

            # Prioritize insights
            output.insights = self._prioritize_insights(output.insights, business_context)

            # Generate recommended actions
            output.actions = self._generate_actions(output.insights, output.anomalies)

            # Generate questions for further analysis
            output.questions_for_further_analysis = self._generate_questions(
                output.insights, output.anomalies, focus_areas
            )

            # Generate executive summary
            output.executive_summary = self._generate_executive_summary(output)

            self.log(
                f"Generated {len(output.insights)} insights, "
                f"{len(output.anomalies)} anomalies, "
                f"{len(output.actions)} recommended actions"
            )

            return AgentResult.success_result(
                output,
                insight_count=len(output.insights),
                anomaly_count=len(output.anomalies),
                action_count=len(output.actions),
            )

        except Exception as e:
            self.log(f"Insight generation failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Insight generation failed: {e}")

    def _extract_insights_from_analysis(
        self,
        analysis_results: dict[str, Any],
        focus_areas: list[str] | None = None,
    ) -> list[Insight]:
        """Extract insights from analysis results."""
        insights = []

        # Handle different analysis types
        for analysis_type, results in analysis_results.items():
            if not isinstance(results, dict):
                continue

            if analysis_type == "descriptive" or "descriptive" in str(analysis_type).lower():
                insights.extend(self._insights_from_descriptive(results))

            elif analysis_type == "correlation" or "correlation" in str(analysis_type).lower():
                insights.extend(self._insights_from_correlation(results))

            elif analysis_type == "distribution" or "distribution" in str(analysis_type).lower():
                insights.extend(self._insights_from_distribution(results))

            elif (
                analysis_type == "hypothesis_testing" or "hypothesis" in str(analysis_type).lower()
            ):
                insights.extend(self._insights_from_hypothesis(results))

            elif analysis_type == "time_series" or "time" in str(analysis_type).lower():
                insights.extend(self._insights_from_time_series(results))

        # Filter by focus areas if specified
        if focus_areas:
            insights = [
                i
                for i in insights
                if any(
                    fa.lower() in i.title.lower()
                    or fa.lower() in i.finding.lower()
                    or any(fa.lower() in col.lower() for col in i.related_columns)
                    for fa in focus_areas
                )
            ]

        return insights

    def _insights_from_descriptive(self, results: dict[str, Any]) -> list[Insight]:
        """Generate insights from descriptive statistics."""
        insights = []

        for col_name, col_stats in results.items():
            if col_name.startswith("_") or not isinstance(col_stats, dict):
                continue

            # Check for high variability
            mean_data = col_stats.get("mean", {})
            std_data = col_stats.get("std", {})

            if isinstance(mean_data, dict) and isinstance(std_data, dict):
                mean_val = mean_data.get("value")
                std_val = std_data.get("value")

                if mean_val and std_val and mean_val != 0:
                    cv = abs(std_val / mean_val)  # Coefficient of variation
                    if cv > 0.5:
                        insights.append(
                            Insight(
                                title=f"High Variability in {col_name}",
                                finding=f"The variable '{col_name}' shows high variability with a coefficient of variation of {cv:.2f}",
                                evidence=[
                                    f"Mean: {mean_val:.2f}",
                                    f"Standard deviation: {std_val:.2f}",
                                    f"CV: {cv:.2f} (>0.5 indicates high variability)",
                                ],
                                impact="High variability may indicate inconsistent processes or diverse subgroups requiring segmentation",
                                recommendation=f"Investigate the drivers of variability in '{col_name}' and consider segmentation analysis",
                                confidence=ConfidenceLevel.HIGH,
                                priority=InsightPriority.MEDIUM,
                                tags=["variability", "descriptive"],
                                related_columns=[col_name],
                            )
                        )

            # Check for skewness
            skew_data = col_stats.get("skewness", {})
            if isinstance(skew_data, dict):
                skew_interp = skew_data.get("interpretation", "")
                if "skewed" in skew_interp.lower():
                    insights.append(
                        Insight(
                            title=f"Skewed Distribution in {col_name}",
                            finding=f"'{col_name}' has a skewed distribution: {skew_interp}",
                            evidence=[f"Skewness interpretation: {skew_interp}"],
                            impact="Skewed distributions may affect the validity of parametric statistical tests",
                            recommendation="Consider using median instead of mean, or apply transformations for analysis",
                            confidence=ConfidenceLevel.HIGH,
                            priority=InsightPriority.LOW,
                            tags=["distribution", "skewness"],
                            related_columns=[col_name],
                        )
                    )

        return insights

    def _insights_from_correlation(self, results: dict[str, Any]) -> list[Insight]:
        """Generate insights from correlation analysis."""
        insights = []

        pairwise = results.get("pairwise_correlations", [])

        for corr in pairwise:
            if not isinstance(corr, dict):
                continue

            r = corr.get("correlation", 0)
            var1 = corr.get("variable_1", "")
            var2 = corr.get("variable_2", "")
            p_value = corr.get("p_value")

            # Strong positive correlation
            if r >= 0.7:
                insights.append(
                    Insight(
                        title=f"Strong Positive Correlation: {var1} & {var2}",
                        finding=f"Strong positive relationship between '{var1}' and '{var2}' (r={r:.3f})",
                        evidence=[
                            f"Correlation coefficient: {r:.3f}",
                            f"P-value: {p_value:.4f}" if p_value else "P-value: N/A",
                            f"R-squared: {r**2:.3f} ({r**2*100:.1f}% variance explained)",
                        ],
                        impact="These variables tend to increase together; changes in one may predict changes in the other",
                        recommendation=f"Investigate causal mechanisms between '{var1}' and '{var2}'. Consider multicollinearity in models.",
                        confidence=(
                            ConfidenceLevel.HIGH
                            if p_value and p_value < 0.01
                            else ConfidenceLevel.MEDIUM
                        ),
                        priority=InsightPriority.HIGH,
                        tags=["correlation", "positive", "strong"],
                        related_columns=[var1, var2],
                    )
                )

            # Strong negative correlation
            elif r <= -0.7:
                insights.append(
                    Insight(
                        title=f"Strong Negative Correlation: {var1} & {var2}",
                        finding=f"Strong negative relationship between '{var1}' and '{var2}' (r={r:.3f})",
                        evidence=[
                            f"Correlation coefficient: {r:.3f}",
                            f"P-value: {p_value:.4f}" if p_value else "P-value: N/A",
                            f"R-squared: {r**2:.3f} ({r**2*100:.1f}% variance explained)",
                        ],
                        impact="These variables move in opposite directions; as one increases, the other tends to decrease",
                        recommendation=f"Investigate trade-offs or inverse relationships between '{var1}' and '{var2}'",
                        confidence=(
                            ConfidenceLevel.HIGH
                            if p_value and p_value < 0.01
                            else ConfidenceLevel.MEDIUM
                        ),
                        priority=InsightPriority.HIGH,
                        tags=["correlation", "negative", "strong"],
                        related_columns=[var1, var2],
                    )
                )

            # Moderate correlation worth noting
            elif 0.5 <= abs(r) < 0.7:
                direction = "positive" if r > 0 else "negative"
                insights.append(
                    Insight(
                        title=f"Moderate {direction.title()} Correlation: {var1} & {var2}",
                        finding=f"Moderate {direction} relationship between '{var1}' and '{var2}' (r={r:.3f})",
                        evidence=[
                            f"Correlation coefficient: {r:.3f}",
                            f"P-value: {p_value:.4f}" if p_value else "P-value: N/A",
                        ],
                        impact=f"Moderate {direction} association may indicate partial relationship",
                        recommendation="Consider this relationship in analysis but note that other factors are also at play",
                        confidence=ConfidenceLevel.MEDIUM,
                        priority=InsightPriority.MEDIUM,
                        tags=["correlation", direction, "moderate"],
                        related_columns=[var1, var2],
                    )
                )

        return insights

    def _insights_from_distribution(self, results: dict[str, Any]) -> list[Insight]:
        """Generate insights from distribution analysis."""
        insights = []

        for col_name, col_data in results.items():
            if not isinstance(col_data, dict) or "normality_test" not in col_data:
                continue

            normality = col_data["normality_test"]
            is_normal = normality.get("is_normal", False)
            p_value = normality.get("p_value")

            if not is_normal:
                insights.append(
                    Insight(
                        title=f"Non-Normal Distribution: {col_name}",
                        finding=f"'{col_name}' does not follow a normal distribution",
                        evidence=[
                            (
                                f"Normality test p-value: {p_value:.4f}"
                                if p_value
                                else "Test performed"
                            ),
                            normality.get("interpretation", ""),
                        ],
                        impact="Standard parametric tests may not be appropriate; consider non-parametric alternatives",
                        recommendation=f"Use non-parametric tests (e.g., Mann-Whitney, Kruskal-Wallis) for '{col_name}'",
                        confidence=ConfidenceLevel.HIGH,
                        priority=InsightPriority.LOW,
                        tags=["distribution", "non-normal"],
                        related_columns=[col_name],
                    )
                )

        return insights

    def _insights_from_hypothesis(self, results: dict[str, Any]) -> list[Insight]:
        """Generate insights from hypothesis testing."""
        insights = []

        for test_name, test_data in results.items():
            if not isinstance(test_data, dict):
                continue

            p_value = test_data.get("p_value")
            effect_size = test_data.get("effect_size_cohens_d") or test_data.get(
                "effect_size_eta_squared"
            )
            interpretation = test_data.get("interpretation", "")

            if p_value is not None and p_value < 0.05:
                # Significant result
                if effect_size:
                    if abs(effect_size) >= 0.8:
                        effect_desc = "large"
                        priority = InsightPriority.CRITICAL
                    elif abs(effect_size) >= 0.5:
                        effect_desc = "medium"
                        priority = InsightPriority.HIGH
                    else:
                        effect_desc = "small"
                        priority = InsightPriority.MEDIUM

                    insights.append(
                        Insight(
                            title=f"Significant Difference Detected ({test_name})",
                            finding=f"Statistical test shows significant difference with {effect_desc} effect size",
                            evidence=[
                                f"P-value: {p_value:.4f}",
                                f"Effect size: {effect_size:.3f} ({effect_desc})",
                                interpretation,
                            ],
                            impact=f"The {effect_desc} effect size suggests this difference is practically meaningful",
                            recommendation="Investigate the factors driving this difference for actionable insights",
                            confidence=ConfidenceLevel.HIGH,
                            priority=priority,
                            tags=["hypothesis", "significant", effect_desc],
                            related_columns=(
                                list(test_data.get("groups", [])) if test_data.get("groups") else []
                            ),
                        )
                    )

        return insights

    def _insights_from_time_series(self, results: dict[str, Any]) -> list[Insight]:
        """Generate insights from time series analysis."""
        insights = []

        changes = results.get("changes", {})
        autocorr = results.get("autocorrelation", {})

        # Trend insight
        pct_change = changes.get("percent_change")
        if pct_change is not None:
            direction = "increasing" if pct_change > 0 else "decreasing"
            magnitude = abs(pct_change)

            if magnitude > 20:
                priority = InsightPriority.HIGH
                impact = f"Substantial {direction} trend requires attention"
            elif magnitude > 10:
                priority = InsightPriority.MEDIUM
                impact = f"Notable {direction} trend worth monitoring"
            else:
                priority = InsightPriority.LOW
                impact = f"Mild {direction} trend observed"

            insights.append(
                Insight(
                    title=f"Time Series Trend: {magnitude:.1f}% {direction.title()}",
                    finding=f"The series shows a {magnitude:.1f}% {direction} trend over the period",
                    evidence=[
                        f"Total change: {pct_change:.1f}%",
                        f"Mean period-over-period change: {changes.get('mean_change', 0):.2f}",
                    ],
                    impact=impact,
                    recommendation=f"{'Investigate drivers of growth' if pct_change > 0 else 'Identify causes of decline and corrective actions'}",
                    confidence=ConfidenceLevel.HIGH,
                    priority=priority,
                    tags=["time_series", "trend", direction],
                    related_columns=[],
                )
            )

        # Autocorrelation insight
        lag_1 = autocorr.get("lag_1")
        if lag_1 is not None and abs(lag_1) > 0.5:
            insights.append(
                Insight(
                    title="Strong Serial Dependence in Time Series",
                    finding=f"High autocorrelation (lag-1: {lag_1:.3f}) indicates strong persistence",
                    evidence=[
                        f"Lag-1 autocorrelation: {lag_1:.3f}",
                        autocorr.get("interpretation", ""),
                    ],
                    impact="Past values strongly predict future values; useful for forecasting",
                    recommendation="Consider ARIMA or similar models that account for autocorrelation",
                    confidence=ConfidenceLevel.HIGH,
                    priority=InsightPriority.MEDIUM,
                    tags=["time_series", "autocorrelation"],
                    related_columns=[],
                )
            )

        return insights

    def _extract_insights_from_profile(self, data_profile: dict[str, Any]) -> list[Insight]:
        """Extract insights from data profile."""
        insights = []

        # Missing data insight
        missing_summary = data_profile.get("missing_summary", {})
        row_count = data_profile.get("row_count", 1)

        high_missing_cols = [
            col
            for col, count in missing_summary.items()
            if count / row_count > 0.1  # More than 10% missing
        ]

        if high_missing_cols:
            insights.append(
                Insight(
                    title="Data Quality: High Missing Values",
                    finding=f"{len(high_missing_cols)} column(s) have more than 10% missing values",
                    evidence=[
                        f"Affected columns: {', '.join(high_missing_cols)}",
                        f"Missing rates: {', '.join([f'{col}: {missing_summary[col]/row_count*100:.1f}%' for col in high_missing_cols[:3]])}",
                    ],
                    impact="Missing data may bias analysis results and reduce statistical power",
                    recommendation="Consider imputation strategies or investigate data collection issues",
                    confidence=ConfidenceLevel.HIGH,
                    priority=InsightPriority.HIGH,
                    tags=["data_quality", "missing_values"],
                    related_columns=high_missing_cols,
                )
            )

        return insights

    def _detect_anomalies(self, analysis_results: dict[str, Any]) -> list[Anomaly]:
        """Detect anomalies in analysis results."""
        anomalies = []

        # Check for unexpected patterns
        for analysis_type, results in analysis_results.items():
            if not isinstance(results, dict):
                continue

            # Check correlation analysis for unexpected perfect correlations
            if "correlation" in str(analysis_type).lower():
                for corr in results.get("pairwise_correlations", []):
                    if isinstance(corr, dict) and abs(corr.get("correlation", 0)) > 0.99:
                        anomalies.append(
                            Anomaly(
                                description=f"Near-perfect correlation between {corr.get('variable_1')} and {corr.get('variable_2')}",
                                severity="high",
                                evidence=f"r = {corr.get('correlation', 0):.4f}",
                                potential_causes=[
                                    "Variables may be duplicates or derived from each other",
                                    "Data entry error",
                                    "Multicollinearity issue",
                                ],
                                recommended_investigation="Verify these are distinct variables and check for data issues",
                            )
                        )

        return anomalies

    def _prioritize_insights(
        self,
        insights: list[Insight],
        business_context: str | None = None,
    ) -> list[Insight]:
        """Prioritize insights by importance."""
        # Sort by priority and confidence
        priority_order = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3,
        }

        confidence_order = {
            ConfidenceLevel.HIGH: 0,
            ConfidenceLevel.MEDIUM: 1,
            ConfidenceLevel.LOW: 2,
        }

        return sorted(
            insights,
            key=lambda i: (
                priority_order.get(i.priority, 3),
                confidence_order.get(i.confidence, 2),
            ),
        )

    def _generate_actions(
        self,
        insights: list[Insight],
        anomalies: list[Anomaly],
    ) -> list[RecommendedAction]:
        """Generate recommended actions from insights and anomalies."""
        actions = []
        priority = 1

        # Actions from high-priority insights
        for insight in insights[:5]:  # Top 5 insights
            if insight.recommendation:
                effort = (
                    "low"
                    if insight.priority == InsightPriority.LOW
                    else ("high" if insight.priority == InsightPriority.CRITICAL else "medium")
                )
                actions.append(
                    RecommendedAction(
                        action=insight.recommendation,
                        expected_impact=insight.impact,
                        priority=priority,
                        effort=effort,
                    )
                )
                priority += 1

        # Actions from anomalies
        for anomaly in anomalies:
            if anomaly.severity == "high":
                actions.append(
                    RecommendedAction(
                        action=anomaly.recommended_investigation,
                        expected_impact=f"Resolve: {anomaly.description}",
                        priority=priority,
                        effort="medium",
                    )
                )
                priority += 1

        return actions

    def _generate_questions(
        self,
        insights: list[Insight],
        anomalies: list[Anomaly],
        focus_areas: list[str] | None = None,
    ) -> list[str]:
        """Generate questions for further analysis."""
        questions = []

        # Questions from correlation insights
        correlation_insights = [i for i in insights if "correlation" in i.tags]
        for insight in correlation_insights[:2]:
            if len(insight.related_columns) >= 2:
                questions.append(
                    f"What is the causal mechanism between {insight.related_columns[0]} "
                    f"and {insight.related_columns[1]}?"
                )

        # Questions from anomalies
        for anomaly in anomalies:
            questions.append(f"What explains the anomaly: {anomaly.description}?")

        # General questions
        if focus_areas:
            for area in focus_areas[:2]:
                questions.append(f"What additional factors influence {area}?")

        return questions[:5]  # Limit to 5 questions

    def _generate_executive_summary(self, output: InsightsOutput) -> str:
        """Generate executive summary from all insights."""
        parts = []

        # Count by priority
        critical_count = len([i for i in output.insights if i.priority == InsightPriority.CRITICAL])
        high_count = len([i for i in output.insights if i.priority == InsightPriority.HIGH])

        # Opening
        parts.append(
            f"Analysis identified {len(output.insights)} key insight(s) "
            f"and {len(output.anomalies)} anomaly/anomalies."
        )

        # Priority summary
        if critical_count > 0:
            parts.append(f"{critical_count} critical finding(s) require immediate attention.")
        if high_count > 0:
            parts.append(f"{high_count} high-priority finding(s) identified.")

        # Top insights
        if output.insights:
            parts.append("\nKey findings:")
            for insight in output.insights[:3]:
                parts.append(f"- {insight.title}: {insight.finding[:100]}...")

        # Top actions
        if output.actions:
            parts.append("\nRecommended actions:")
            for action in output.actions[:3]:
                parts.append(f"{action.priority}. {action.action}")

        return " ".join(parts)

    def get_focus_options(self, available_insights: list[str]) -> list[AgentOption]:
        """
        Get options for insight focus areas.

        Args:
            available_insights: Types of insights available

        Returns:
            List of focus area options
        """
        return [
            AgentOption(
                id="all",
                title="All Insights",
                description="Review all generated insights without filtering",
                recommended=True,
                pros=["Comprehensive view", "No missed findings"],
                cons=["May include less relevant insights"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="high_priority",
                title="High Priority Only",
                description="Focus on critical and high priority insights",
                pros=["Actionable", "Time-efficient"],
                cons=["May miss nuanced findings"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="correlations",
                title="Relationship Insights",
                description="Focus on correlations and relationships between variables",
                pros=["Reveals connections", "Good for strategy"],
                cons=["Misses standalone metrics"],
                estimated_complexity="low",
            ),
        ]

    def format_output(self, output: InsightsOutput) -> str:
        """
        Format insights output for display.

        Args:
            output: The insights output to format

        Returns:
            Formatted markdown string
        """
        lines = [
            "# Key Insights Summary",
            "",
            "## Executive Summary",
            output.executive_summary,
            "",
        ]

        # Insights section
        for i, insight in enumerate(output.insights, 1):
            lines.extend(
                [
                    f"## Insight {i}: {insight.title}",
                    "",
                    f"**Finding**: {insight.finding}",
                    "",
                    "**Evidence**:",
                ]
            )
            for evidence in insight.evidence:
                lines.append(f"- {evidence}")
            lines.extend(
                [
                    "",
                    f"**Impact**: {insight.impact}",
                    "",
                    f"**Recommendation**: {insight.recommendation}",
                    "",
                    f"**Confidence**: {insight.confidence.value.title()} | **Priority**: {insight.priority.value.title()}",
                    "",
                ]
            )

        # Anomalies section
        if output.anomalies:
            lines.extend(
                [
                    "## Anomalies Detected",
                    "",
                ]
            )
            for anomaly in output.anomalies:
                lines.extend(
                    [
                        f"- **{anomaly.description}** (Severity: {anomaly.severity})",
                        f"  - Evidence: {anomaly.evidence}",
                        f"  - Investigation: {anomaly.recommended_investigation}",
                        "",
                    ]
                )

        # Actions section
        if output.actions:
            lines.extend(
                [
                    "## Recommended Actions (Priority Order)",
                    "",
                ]
            )
            for action in output.actions:
                lines.append(f"{action.priority}. {action.action} (Effort: {action.effort})")
                lines.append(f"   - Expected impact: {action.expected_impact}")
                lines.append("")

        # Questions section
        if output.questions_for_further_analysis:
            lines.extend(
                [
                    "## Questions for Further Analysis",
                    "",
                ]
            )
            for question in output.questions_for_further_analysis:
                lines.append(f"- {question}")

        return "\n".join(lines)
