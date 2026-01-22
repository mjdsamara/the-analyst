"""
Report Agent for The Analyst platform.

Generates formatted outputs (PDF, PowerPoint, HTML, Markdown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.prompts.agents import REPORT_PROMPT


class ReportFormat(str, Enum):
    """Supported report formats."""

    PDF = "pdf"
    POWERPOINT = "pptx"
    HTML = "html"
    MARKDOWN = "markdown"


class AudienceType(str, Enum):
    """Target audience types."""

    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    GENERAL = "general"


@dataclass
class ReportSection:
    """A section in the report."""

    title: str
    content: str
    order: int
    section_type: str = "text"  # text, chart, table, summary
    chart_ids: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    subsections: list[ReportSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "order": self.order,
            "section_type": self.section_type,
            "chart_ids": self.chart_ids,
            "tables": self.tables,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class ReportMetadata:
    """Metadata for the report."""

    title: str
    subtitle: str = ""
    author: str = "The Analyst"
    created_at: datetime = field(default_factory=datetime.utcnow)
    audience: AudienceType = AudienceType.GENERAL
    format: ReportFormat = ReportFormat.PDF
    version: str = "1.0"
    confidentiality: str = "Internal"
    page_count: int | None = None
    word_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "audience": self.audience.value,
            "format": self.format.value,
            "version": self.version,
            "confidentiality": self.confidentiality,
            "page_count": self.page_count,
            "word_count": self.word_count,
        }


@dataclass
class ReportDraft:
    """A draft report structure for approval."""

    metadata: ReportMetadata
    sections: list[ReportSection]
    estimated_pages: int = 0
    charts_included: list[str] = field(default_factory=list)
    tables_included: int = 0
    approval_status: str = "pending"  # pending, approved, rejected
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "estimated_pages": self.estimated_pages,
            "charts_included": self.charts_included,
            "tables_included": self.tables_included,
            "approval_status": self.approval_status,
            "feedback": self.feedback,
        }


@dataclass
class ReportOutput:
    """Complete output from the report agent."""

    metadata: ReportMetadata
    sections: list[ReportSection]
    file_path: Path | None = None
    file_content: str = ""
    size_bytes: int = 0
    generation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "file_path": str(self.file_path) if self.file_path else None,
            "size_bytes": self.size_bytes,
            "generation_notes": self.generation_notes,
        }


class ReportAgent(BaseAgent):
    """
    Agent responsible for generating formatted reports.

    Single Job: Generate formatted outputs (PDF, PowerPoint, HTML, Markdown).
    """

    def __init__(self, context: AgentContext | None = None) -> None:
        """Initialize the report agent."""
        super().__init__(name="report", context=context)
        self._figure_counter = 0

    @property
    def system_prompt(self) -> str:
        """Get the agent's system prompt."""
        return REPORT_PROMPT

    def _generate_figure_reference(self) -> str:
        """Generate a figure reference number."""
        self._figure_counter += 1
        return f"Figure {self._figure_counter}"

    async def execute(
        self,
        insights: dict[str, Any] | None = None,
        visualizations: dict[str, Any] | None = None,
        analysis_results: dict[str, Any] | None = None,
        format: str | ReportFormat = ReportFormat.MARKDOWN,
        audience: str | AudienceType = AudienceType.GENERAL,
        title: str = "Analytics Report",
        output_path: str | Path | None = None,
        draft_only: bool = False,
        **kwargs: Any,
    ) -> AgentResult[ReportOutput | ReportDraft]:
        """
        Execute report generation.

        Args:
            insights: Output from insights agent
            visualizations: Output from visualization agent
            analysis_results: Raw analysis results
            format: Output format (pdf, pptx, html, markdown)
            audience: Target audience (executive, technical, general)
            title: Report title
            output_path: Path to save the report
            draft_only: If True, return draft structure for approval

        Returns:
            AgentResult containing report output or draft
        """
        if not insights and not visualizations and not analysis_results:
            return AgentResult.error_result(
                "No content provided for report generation. "
                "Please provide insights, visualizations, or analysis results."
            )

        self.log("Starting report generation")

        # Convert string enums to enum types
        if isinstance(format, str):
            try:
                format = ReportFormat(format.lower())
            except ValueError:
                format = ReportFormat.MARKDOWN

        if isinstance(audience, str):
            try:
                audience = AudienceType(audience.lower())
            except ValueError:
                audience = AudienceType.GENERAL

        try:
            # Create metadata
            metadata = ReportMetadata(
                title=title,
                audience=audience,
                format=format,
            )

            # Build sections
            sections = self._build_sections(
                insights=insights,
                visualizations=visualizations,
                analysis_results=analysis_results,
                audience=audience,
            )

            # If draft only, return for approval
            if draft_only:
                draft = self._create_draft(metadata, sections, visualizations)
                self.log("Created draft for approval")
                return AgentResult.success_result(
                    draft,
                    requires_approval=True,
                    draft_sections=len(sections),
                )

            # Generate the report
            output = self._generate_report(
                metadata=metadata,
                sections=sections,
                visualizations=visualizations,
                output_path=Path(output_path) if output_path else None,
            )

            self.log(f"Generated {format.value} report with {len(sections)} sections")

            return AgentResult.success_result(
                output,
                format=format.value,
                section_count=len(sections),
                has_file=output.file_path is not None,
            )

        except Exception as e:
            self.log(f"Report generation failed: {e}", level="ERROR")
            return AgentResult.error_result(f"Report generation failed: {e}")

    def _build_sections(
        self,
        insights: dict[str, Any] | None,
        visualizations: dict[str, Any] | None,
        analysis_results: dict[str, Any] | None,
        audience: AudienceType,
    ) -> list[ReportSection]:
        """Build report sections from available content."""
        sections = []
        order = 1

        # 1. Executive Summary (always first, max 1 page)
        exec_summary = self._generate_executive_summary(insights, analysis_results, audience)
        sections.append(
            ReportSection(
                title="Executive Summary",
                content=exec_summary,
                order=order,
                section_type="summary",
            )
        )
        order += 1

        # 2. Methodology section
        methodology = self._generate_methodology_section(analysis_results)
        sections.append(
            ReportSection(
                title="Methodology",
                content=methodology,
                order=order,
                section_type="text",
            )
        )
        order += 1

        # 3. Key Findings section
        if insights:
            findings_section = self._generate_findings_section(insights, visualizations)
            sections.append(
                ReportSection(
                    title="Key Findings",
                    content=findings_section["content"],
                    order=order,
                    section_type="findings",
                    chart_ids=findings_section.get("chart_ids", []),
                )
            )
            order += 1

        # 4. Detailed Analysis (if technical audience or comprehensive)
        if audience == AudienceType.TECHNICAL and analysis_results:
            detailed = self._generate_detailed_analysis(analysis_results)
            sections.append(
                ReportSection(
                    title="Detailed Analysis",
                    content=detailed,
                    order=order,
                    section_type="text",
                )
            )
            order += 1

        # 5. Visualizations section
        if visualizations:
            viz_section = self._generate_visualizations_section(visualizations)
            sections.append(
                ReportSection(
                    title="Visualizations",
                    content=viz_section["content"],
                    order=order,
                    section_type="charts",
                    chart_ids=viz_section.get("chart_ids", []),
                )
            )
            order += 1

        # 6. Recommendations section
        if insights:
            recommendations = self._generate_recommendations_section(insights)
            sections.append(
                ReportSection(
                    title="Recommendations",
                    content=recommendations,
                    order=order,
                    section_type="text",
                )
            )
            order += 1

        # 7. Appendix (technical details)
        if audience == AudienceType.TECHNICAL:
            appendix = self._generate_appendix(analysis_results, insights)
            if appendix:
                sections.append(
                    ReportSection(
                        title="Appendix",
                        content=appendix,
                        order=order,
                        section_type="appendix",
                    )
                )

        return sections

    def _generate_executive_summary(
        self,
        insights: dict[str, Any] | None,
        analysis_results: dict[str, Any] | None,
        audience: AudienceType,
    ) -> str:
        """Generate executive summary content."""
        parts = []

        # Overview
        parts.append("This report presents the findings from our analytics analysis.")
        parts.append("")

        # Key metrics if available
        if insights and "executive_summary" in insights:
            parts.append(insights["executive_summary"])
            parts.append("")

        # Top insights
        if insights and "insights" in insights:
            insight_list = insights["insights"]
            if insight_list:
                parts.append("**Key Highlights:**")
                for insight in insight_list[:3]:  # Top 3 insights
                    if isinstance(insight, dict):
                        title = insight.get("title", "Finding")
                        finding = insight.get("finding", "")
                        parts.append(f"- **{title}**: {finding[:150]}...")
                    else:
                        parts.append(f"- {str(insight)[:150]}...")
                parts.append("")

        # Actions required
        if insights and "actions" in insights:
            actions = insights["actions"]
            if actions:
                parts.append("**Recommended Actions:**")
                for action in actions[:3]:  # Top 3 actions
                    if isinstance(action, dict):
                        parts.append(f"- {action.get('action', str(action))}")
                    else:
                        parts.append(f"- {str(action)}")
                parts.append("")

        # Add confidence statement
        parts.append(
            "*Confidence levels and detailed methodology are provided in subsequent sections.*"
        )

        return "\n".join(parts)

    def _generate_methodology_section(
        self,
        analysis_results: dict[str, Any] | None,
    ) -> str:
        """Generate methodology section content."""
        parts = []

        parts.append("### Data Sources")
        parts.append("The analysis was conducted using data from the provided sources.")
        parts.append("")

        parts.append("### Analytical Approach")

        if analysis_results:
            # List analysis types performed
            analysis_types = []
            if "descriptive" in analysis_results:
                analysis_types.append("Descriptive statistics")
            if "correlation" in analysis_results:
                analysis_types.append("Correlation analysis")
            if "distribution" in analysis_results:
                analysis_types.append("Distribution analysis")
            if "hypothesis_testing" in analysis_results:
                analysis_types.append("Hypothesis testing")
            if "time_series" in analysis_results:
                analysis_types.append("Time series analysis")

            if analysis_types:
                parts.append("The following analytical methods were applied:")
                for at in analysis_types:
                    parts.append(f"- {at}")
            else:
                parts.append("Standard statistical analysis methods were applied.")
        else:
            parts.append("The analysis followed standard statistical practices.")

        parts.append("")
        parts.append("### Statistical Standards")
        parts.append("- Confidence level: 95%")
        parts.append("- All statistical tests include effect sizes where applicable")
        parts.append("- P-values are reported with appropriate context")

        return "\n".join(parts)

    def _generate_findings_section(
        self,
        insights: dict[str, Any],
        visualizations: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Generate key findings section content."""
        parts = []
        chart_ids = []

        if "insights" in insights:
            for i, insight in enumerate(insights["insights"], 1):
                if isinstance(insight, dict):
                    parts.append(f"### Finding {i}: {insight.get('title', 'Insight')}")
                    parts.append("")
                    parts.append(f"**Observation**: {insight.get('finding', '')}")
                    parts.append("")

                    # Evidence
                    evidence = insight.get("evidence", [])
                    if evidence:
                        parts.append("**Supporting Evidence:**")
                        for ev in evidence:
                            parts.append(f"- {ev}")
                        parts.append("")

                    # Impact
                    impact = insight.get("impact", "")
                    if impact:
                        parts.append(f"**Business Impact**: {impact}")
                        parts.append("")

                    # Confidence
                    confidence = insight.get("confidence", "")
                    priority = insight.get("priority", "")
                    if confidence or priority:
                        parts.append(
                            f"**Confidence**: {confidence.title() if confidence else 'N/A'} | "
                            f"**Priority**: {priority.title() if priority else 'N/A'}"
                        )
                        parts.append("")

        # Reference visualizations if available
        if visualizations and "charts" in visualizations:
            parts.append("*See Visualizations section for supporting charts.*")
            chart_ids = [c.get("chart_id", "") for c in visualizations.get("charts", [])]

        return {
            "content": "\n".join(parts),
            "chart_ids": chart_ids,
        }

    def _generate_detailed_analysis(
        self,
        analysis_results: dict[str, Any],
    ) -> str:
        """Generate detailed analysis for technical audience."""
        parts = []

        for analysis_type, results in analysis_results.items():
            if not isinstance(results, dict):
                continue

            parts.append(f"### {analysis_type.replace('_', ' ').title()}")
            parts.append("")

            # Format results based on type
            if analysis_type == "descriptive":
                for col, stats in results.items():
                    if isinstance(stats, dict) and not col.startswith("_"):
                        parts.append(f"**{col}**:")
                        for stat_name, stat_value in stats.items():
                            if isinstance(stat_value, dict):
                                val = stat_value.get("value", stat_value)
                                parts.append(f"  - {stat_name}: {val}")
                            else:
                                parts.append(f"  - {stat_name}: {stat_value}")
                        parts.append("")

            elif analysis_type == "correlation":
                pairs = results.get("pairwise_correlations", [])
                if pairs:
                    parts.append("| Variable 1 | Variable 2 | Correlation | P-value |")
                    parts.append("|------------|------------|-------------|---------|")
                    for pair in pairs[:10]:  # Limit to top 10
                        if isinstance(pair, dict):
                            parts.append(
                                f"| {pair.get('variable_1', '')} | "
                                f"{pair.get('variable_2', '')} | "
                                f"{pair.get('correlation', 0):.3f} | "
                                f"{pair.get('p_value', 'N/A')} |"
                            )
                    parts.append("")

            else:
                # Generic formatting
                parts.append(f"Results: {len(results)} items analyzed")
                parts.append("")

        return "\n".join(parts)

    def _generate_visualizations_section(
        self,
        visualizations: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate visualizations section content."""
        parts = []
        chart_ids = []

        charts = visualizations.get("charts", [])
        for i, chart in enumerate(charts, 1):
            if isinstance(chart, dict):
                fig_ref = self._generate_figure_reference()
                chart_id = chart.get("chart_id", f"chart_{i}")
                chart_ids.append(chart_id)

                parts.append(f"### {fig_ref}: {chart.get('title', 'Chart')}")
                parts.append("")
                parts.append(f"**Type**: {chart.get('chart_type', 'Unknown').title()}")
                parts.append(f"**Description**: {chart.get('description', '')}")
                parts.append(f"**Data Source**: {chart.get('data_source', 'N/A')}")
                parts.append("")

                # Add warnings if any
                warnings = chart.get("validation_warnings", [])
                if warnings:
                    parts.append("**Notes:**")
                    for warning in warnings:
                        parts.append(f"- {warning}")
                    parts.append("")

                parts.append(f"*[Chart {chart_id} embedded here]*")
                parts.append("")

        # Summary from visualization output
        summary = visualizations.get("summary", "")
        if summary:
            parts.append("### Visualization Summary")
            parts.append(summary)

        return {
            "content": "\n".join(parts),
            "chart_ids": chart_ids,
        }

    def _generate_recommendations_section(
        self,
        insights: dict[str, Any],
    ) -> str:
        """Generate recommendations section content."""
        parts = []

        actions = insights.get("actions", [])
        if actions:
            parts.append("Based on the analysis, we recommend the following actions:")
            parts.append("")

            for i, action in enumerate(actions, 1):
                if isinstance(action, dict):
                    parts.append(f"**{i}. {action.get('action', 'Action')}**")
                    impact = action.get("expected_impact", "")
                    if impact:
                        parts.append(f"   - Expected Impact: {impact}")
                    effort = action.get("effort", "")
                    if effort:
                        parts.append(f"   - Effort Level: {effort.title()}")
                    parts.append("")
                else:
                    parts.append(f"{i}. {action}")
                    parts.append("")
        else:
            parts.append("No specific actions identified from the analysis.")
            parts.append("Consider reviewing the findings for potential opportunities.")

        # Questions for further analysis
        questions = insights.get("questions_for_further_analysis", [])
        if questions:
            parts.append("")
            parts.append("### Areas for Further Investigation")
            for q in questions:
                parts.append(f"- {q}")

        return "\n".join(parts)

    def _generate_appendix(
        self,
        analysis_results: dict[str, Any] | None,
        insights: dict[str, Any] | None,
    ) -> str:
        """Generate appendix content."""
        parts = []

        parts.append("### Technical Details")
        parts.append("")

        # Data quality notes
        if insights and "anomalies" in insights:
            anomalies = insights["anomalies"]
            if anomalies:
                parts.append("#### Data Anomalies Detected")
                for anomaly in anomalies:
                    if isinstance(anomaly, dict):
                        parts.append(f"- {anomaly.get('description', str(anomaly))}")
                        parts.append(f"  - Severity: {anomaly.get('severity', 'N/A')}")
                parts.append("")

        # Statistical parameters
        parts.append("#### Statistical Parameters")
        parts.append("- Confidence Level: 95%")
        parts.append("- Alpha Level: 0.05")
        parts.append("- Random Seed: 42 (where applicable)")
        parts.append("")

        # Reproducibility
        parts.append("#### Reproducibility")
        parts.append("All analyses can be reproduced using the same data and parameters.")
        parts.append(f"Report generated: {datetime.utcnow().isoformat()}")

        return "\n".join(parts)

    def _create_draft(
        self,
        metadata: ReportMetadata,
        sections: list[ReportSection],
        visualizations: dict[str, Any] | None,
    ) -> ReportDraft:
        """Create a draft for approval."""
        # Estimate pages (rough: ~500 words per page)
        total_content = " ".join(s.content for s in sections)
        word_count = len(total_content.split())
        estimated_pages = max(1, word_count // 500)

        # Get chart IDs
        chart_ids = []
        for section in sections:
            chart_ids.extend(section.chart_ids)

        # Count tables
        table_count = sum(len(s.tables) for s in sections)

        return ReportDraft(
            metadata=metadata,
            sections=sections,
            estimated_pages=estimated_pages,
            charts_included=chart_ids,
            tables_included=table_count,
        )

    def _generate_report(
        self,
        metadata: ReportMetadata,
        sections: list[ReportSection],
        visualizations: dict[str, Any] | None,
        output_path: Path | None,
    ) -> ReportOutput:
        """Generate the final report."""
        # For now, generate Markdown (other formats would need additional libraries)
        content = self._render_markdown(metadata, sections)

        output = ReportOutput(
            metadata=metadata,
            sections=sections,
            file_content=content,
            size_bytes=len(content.encode("utf-8")),
        )

        # Word count
        metadata.word_count = len(content.split())
        metadata.page_count = max(1, metadata.word_count // 500)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if metadata.format == ReportFormat.MARKDOWN:
                output_path = output_path.with_suffix(".md")
                output_path.write_text(content, encoding="utf-8")
                output.file_path = output_path
                self.log(f"Saved Markdown report to {output_path}")

            elif metadata.format == ReportFormat.HTML:
                html_content = self._render_html(metadata, sections, content)
                output_path = output_path.with_suffix(".html")
                output_path.write_text(html_content, encoding="utf-8")
                output.file_path = output_path
                output.file_content = html_content
                output.size_bytes = len(html_content.encode("utf-8"))
                self.log(f"Saved HTML report to {output_path}")

            elif metadata.format == ReportFormat.PDF:
                # PDF generation would require weasyprint
                output.generation_notes.append(
                    "PDF generation requires WeasyPrint. Markdown version saved instead."
                )
                output_path = output_path.with_suffix(".md")
                output_path.write_text(content, encoding="utf-8")
                output.file_path = output_path
                self.log("Saved Markdown report (PDF generation not available)")

            elif metadata.format == ReportFormat.POWERPOINT:
                # PPTX generation would require python-pptx
                output.generation_notes.append(
                    "PowerPoint generation requires python-pptx. Markdown version saved instead."
                )
                output_path = output_path.with_suffix(".md")
                output_path.write_text(content, encoding="utf-8")
                output.file_path = output_path
                self.log("Saved Markdown report (PPTX generation not available)")

        return output

    def _render_markdown(
        self,
        metadata: ReportMetadata,
        sections: list[ReportSection],
    ) -> str:
        """Render report as Markdown."""
        lines = []

        # Title page
        lines.extend(
            [
                f"# {metadata.title}",
                "",
            ]
        )
        if metadata.subtitle:
            lines.append(f"## {metadata.subtitle}")
            lines.append("")

        lines.extend(
            [
                f"**Author**: {metadata.author}",
                f"**Date**: {metadata.created_at.strftime('%Y-%m-%d')}",
                f"**Audience**: {metadata.audience.value.title()}",
                f"**Confidentiality**: {metadata.confidentiality}",
                "",
                "---",
                "",
            ]
        )

        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for section in sections:
            lines.append(
                f"{section.order}. [{section.title}](#{section.title.lower().replace(' ', '-')})"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

        # Sections
        for section in sections:
            lines.append(f"## {section.order}. {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # Footer
        lines.extend(
            [
                "*This report was generated by The Analyst - AI-Powered Analytics Platform*",
                "",
                f"*Report ID: {metadata.version} | Generated: {metadata.created_at.isoformat()}*",
            ]
        )

        return "\n".join(lines)

    def _render_html(
        self,
        metadata: ReportMetadata,
        sections: list[ReportSection],
        markdown_content: str,
    ) -> str:
        """Render report as HTML."""
        # Simple HTML wrapper around markdown
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        h1 {{ color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 0.5rem; }}
        h2 {{ color: #1a5276; margin-top: 2rem; }}
        h3 {{ color: #2874a6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 1rem; overflow-x: auto; }}
        hr {{ border: none; border-top: 1px solid #ddd; margin: 2rem 0; }}
        .metadata {{ background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 2rem; }}
        .footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; font-size: 0.9rem; color: #666; }}
    </style>
</head>
<body>
    <div class="metadata">
        <h1>{metadata.title}</h1>
        <p><strong>Author:</strong> {metadata.author} |
           <strong>Date:</strong> {metadata.created_at.strftime('%Y-%m-%d')} |
           <strong>Audience:</strong> {metadata.audience.value.title()}</p>
    </div>

    <div class="content">
"""
        # Simple markdown to HTML conversion (basic)
        content_html = markdown_content
        content_html = content_html.replace("# ", "<h1>").replace("\n## ", "</h1>\n<h2>")
        content_html = content_html.replace("\n### ", "</h2>\n<h3>")
        content_html = content_html.replace("**", "<strong>").replace("**", "</strong>")
        content_html = content_html.replace("\n- ", "\n<li>")
        content_html = content_html.replace("\n---\n", "\n<hr>\n")
        content_html = content_html.replace("\n\n", "</p>\n<p>")

        html += f"""
        {content_html}
    </div>

    <div class="footer">
        <p>This report was generated by The Analyst - AI-Powered Analytics Platform</p>
        <p>Report Version: {metadata.version} | Generated: {metadata.created_at.isoformat()}</p>
    </div>
</body>
</html>
"""
        return html

    def create_draft_structure(
        self,
        insights: dict[str, Any] | None = None,
        visualizations: dict[str, Any] | None = None,
        analysis_results: dict[str, Any] | None = None,
        title: str = "Analytics Report",
        audience: AudienceType = AudienceType.GENERAL,
    ) -> ReportDraft:
        """
        Create a draft report structure for approval.

        Args:
            insights: Output from insights agent
            visualizations: Output from visualization agent
            analysis_results: Raw analysis results
            title: Report title
            audience: Target audience

        Returns:
            ReportDraft for approval
        """
        metadata = ReportMetadata(
            title=title,
            audience=audience,
        )

        sections = self._build_sections(
            insights=insights,
            visualizations=visualizations,
            analysis_results=analysis_results,
            audience=audience,
        )

        return self._create_draft(metadata, sections, visualizations)

    def get_format_options(self) -> list[AgentOption]:
        """
        Get available report format options.

        Returns:
            List of format options
        """
        return [
            AgentOption(
                id="pdf",
                title="PDF Report",
                description="Formal report in PDF format, suitable for sharing",
                recommended=True,
                pros=["Professional appearance", "Widely compatible", "Print-ready"],
                cons=["Requires WeasyPrint library", "Static content"],
                estimated_complexity="medium",
            ),
            AgentOption(
                id="pptx",
                title="PowerPoint Presentation",
                description="Slide deck for presentations",
                pros=["Presentation-ready", "Easy to customize"],
                cons=["Requires python-pptx library", "Limited detail"],
                estimated_complexity="medium",
            ),
            AgentOption(
                id="html",
                title="HTML Dashboard",
                description="Interactive web-based report",
                pros=["Interactive charts", "No dependencies", "Shareable link"],
                cons=["Requires web browser", "May need hosting"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="markdown",
                title="Markdown Document",
                description="Plain text with formatting, good for documentation",
                pros=["Version control friendly", "No dependencies", "Portable"],
                cons=["Less visual polish", "Charts as references"],
                estimated_complexity="low",
            ),
        ]

    def get_audience_options(self) -> list[AgentOption]:
        """
        Get available audience options.

        Returns:
            List of audience options
        """
        return [
            AgentOption(
                id="executive",
                title="Executive Summary",
                description="High-level overview for leadership",
                recommended=True,
                pros=["Concise", "Action-focused", "Business language"],
                cons=["Limited technical detail"],
                estimated_complexity="low",
            ),
            AgentOption(
                id="technical",
                title="Technical Report",
                description="Detailed analysis for data teams",
                pros=["Full methodology", "Statistical details", "Reproducible"],
                cons=["Longer document", "May overwhelm non-technical readers"],
                estimated_complexity="medium",
            ),
            AgentOption(
                id="general",
                title="General Audience",
                description="Balanced report for mixed audiences",
                pros=["Accessible", "Balanced detail", "Wide appeal"],
                cons=["May lack depth for specialists"],
                estimated_complexity="low",
            ),
        ]

    def format_output(self, output: ReportOutput | ReportDraft) -> str:
        """
        Format report output for display.

        Args:
            output: The report output or draft to format

        Returns:
            Formatted markdown string
        """
        lines = []

        if isinstance(output, ReportDraft):
            lines.extend(
                [
                    "# Report Draft - Awaiting Approval",
                    "",
                    f"**Title**: {output.metadata.title}",
                    f"**Format**: {output.metadata.format.value.upper()}",
                    f"**Audience**: {output.metadata.audience.value.title()}",
                    f"**Estimated Pages**: {output.estimated_pages}",
                    f"**Charts Included**: {len(output.charts_included)}",
                    "",
                    "## Proposed Sections",
                    "",
                ]
            )

            for section in output.sections:
                lines.append(f"{section.order}. **{section.title}** ({section.section_type})")

            lines.extend(
                [
                    "",
                    "---",
                    "",
                    "*Please review and approve this structure, or provide feedback for changes.*",
                ]
            )

        else:
            lines.extend(
                [
                    "# Report Generated Successfully",
                    "",
                    f"**Title**: {output.metadata.title}",
                    f"**Format**: {output.metadata.format.value.upper()}",
                    f"**Sections**: {len(output.sections)}",
                    f"**Word Count**: {output.metadata.word_count or 'N/A'}",
                    f"**Estimated Pages**: {output.metadata.page_count or 'N/A'}",
                    "",
                ]
            )

            if output.file_path:
                lines.append(f"**Saved to**: {output.file_path}")
                lines.append("")

            if output.generation_notes:
                lines.append("## Generation Notes")
                for note in output.generation_notes:
                    lines.append(f"- {note}")
                lines.append("")

            lines.extend(
                [
                    "## Sections",
                    "",
                ]
            )

            for section in output.sections:
                lines.append(f"{section.order}. {section.title}")

        return "\n".join(lines)
