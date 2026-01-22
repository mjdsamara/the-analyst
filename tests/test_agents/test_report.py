"""Tests for the Report Agent."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Set environment variables before importing modules that use Settings
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

from src.agents.report import (
    AudienceType,
    ReportAgent,
    ReportDraft,
    ReportFormat,
    ReportMetadata,
    ReportOutput,
    ReportSection,
)


class TestReportAgent:
    """Test suite for ReportAgent."""

    @pytest.fixture
    def agent(self):
        """Create a report agent for testing."""
        with patch("src.agents.base.AsyncAnthropic"):
            return ReportAgent()

    @pytest.fixture
    def sample_insights(self):
        """Create sample insights for testing."""
        return {
            "insights": [
                {
                    "title": "Strong Sales Growth",
                    "finding": "Sales increased by 25% over the quarter",
                    "evidence": ["Q1 vs Q2 comparison", "YoY growth data"],
                    "impact": "Revenue targets exceeded",
                    "recommendation": "Maintain current strategy",
                    "confidence": "high",
                    "priority": "high",
                },
                {
                    "title": "Cost Efficiency Improved",
                    "finding": "Operating costs reduced by 10%",
                    "evidence": ["Cost analysis report"],
                    "impact": "Improved margins",
                    "recommendation": "Scale cost reduction initiatives",
                    "confidence": "medium",
                    "priority": "medium",
                },
            ],
            "anomalies": [
                {
                    "description": "Unusual spike in returns",
                    "severity": "medium",
                    "evidence": "30% increase in Week 8",
                },
            ],
            "actions": [
                {
                    "action": "Expand marketing budget",
                    "expected_impact": "15% sales increase",
                    "effort": "medium",
                },
                {
                    "action": "Investigate returns spike",
                    "expected_impact": "Reduce returns by 20%",
                    "effort": "low",
                },
            ],
            "questions_for_further_analysis": [
                "What drove the returns spike?",
                "Which products have highest growth potential?",
            ],
            "executive_summary": "Strong quarter with 25% sales growth and improved cost efficiency.",
        }

    @pytest.fixture
    def sample_visualizations(self):
        """Create sample visualization output for testing."""
        return {
            "charts": [
                {
                    "chart_id": "chart_001",
                    "chart_type": "line",
                    "title": "Sales Trend",
                    "description": "Monthly sales over time",
                    "data_source": "Sales Database",
                    "validation_warnings": [],
                },
                {
                    "chart_id": "chart_002",
                    "chart_type": "bar",
                    "title": "Sales by Category",
                    "description": "Category comparison",
                    "data_source": "Sales Database",
                    "validation_warnings": ["Missing y-axis label"],
                },
            ],
            "summary": "Generated 2 visualizations",
            "recommendations": ["Consider adding trend lines"],
        }

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results for testing."""
        return {
            "descriptive": {
                "sales": {"mean": {"value": 145.5}, "std": {"value": 30.2}},
            },
            "correlation": {
                "pairwise_correlations": [
                    {
                        "variable_1": "sales",
                        "variable_2": "marketing",
                        "correlation": 0.85,
                        "p_value": 0.001,
                    },
                ],
            },
        }

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent.name == "report"
        assert agent.autonomy.value == "advisory"

    @pytest.mark.asyncio
    async def test_execute_with_insights(self, agent, sample_insights):
        """Test report generation with insights."""
        result = await agent.execute(
            insights=sample_insights,
            format="markdown",
            audience="general",
            title="Test Report",
        )

        assert result.success
        assert result.data is not None
        assert isinstance(result.data, ReportOutput)
        assert len(result.data.sections) > 0
        assert result.data.metadata.title == "Test Report"

    @pytest.mark.asyncio
    async def test_execute_with_visualizations(self, agent, sample_insights, sample_visualizations):
        """Test report generation with visualizations."""
        result = await agent.execute(
            insights=sample_insights,
            visualizations=sample_visualizations,
            format="html",
            audience="executive",
            title="Executive Report",
        )

        assert result.success
        assert result.data is not None
        # Should include visualization section
        section_titles = [s.title for s in result.data.sections]
        assert "Visualizations" in section_titles

    @pytest.mark.asyncio
    async def test_execute_with_analysis_results(self, agent, sample_analysis_results):
        """Test report generation with raw analysis results."""
        result = await agent.execute(
            analysis_results=sample_analysis_results,
            format="markdown",
            audience="technical",
            title="Technical Analysis Report",
        )

        assert result.success
        assert result.data is not None
        # Technical audience should have detailed analysis section
        section_titles = [s.title for s in result.data.sections]
        assert "Detailed Analysis" in section_titles

    @pytest.mark.asyncio
    async def test_execute_with_no_input(self, agent):
        """Test error handling when no input provided."""
        result = await agent.execute()

        assert not result.success
        assert "no content" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_draft_only(self, agent, sample_insights):
        """Test draft generation for approval."""
        result = await agent.execute(
            insights=sample_insights,
            format="pdf",
            audience="executive",
            title="Draft Report",
            draft_only=True,
        )

        assert result.success
        assert result.data is not None
        assert isinstance(result.data, ReportDraft)
        assert result.data.approval_status == "pending"

    @pytest.mark.asyncio
    async def test_execute_with_output_path(self, agent, sample_insights, tmp_path):
        """Test report generation with file output."""
        output_file = tmp_path / "test_report"

        result = await agent.execute(
            insights=sample_insights,
            format="markdown",
            title="File Output Test",
            output_path=str(output_file),
        )

        assert result.success
        assert result.data.file_path is not None
        assert result.data.file_path.exists()
        assert result.data.file_path.suffix == ".md"

    @pytest.mark.asyncio
    async def test_execute_html_format(self, agent, sample_insights, tmp_path):
        """Test HTML report generation."""
        output_file = tmp_path / "test_report"

        result = await agent.execute(
            insights=sample_insights,
            format="html",
            title="HTML Report",
            output_path=str(output_file),
        )

        assert result.success
        assert result.data.file_path.suffix == ".html"
        # Check HTML content
        content = result.data.file_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "HTML Report" in content

    def test_build_sections_general_audience(self, agent, sample_insights):
        """Test section building for general audience."""
        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=None,
            analysis_results=None,
            audience=AudienceType.GENERAL,
        )

        # Should have standard sections
        section_titles = [s.title for s in sections]
        assert "Executive Summary" in section_titles
        assert "Methodology" in section_titles
        assert "Key Findings" in section_titles
        assert "Recommendations" in section_titles

        # Should NOT have detailed analysis (that's for technical)
        assert "Detailed Analysis" not in section_titles

    def test_build_sections_technical_audience(
        self, agent, sample_insights, sample_analysis_results
    ):
        """Test section building for technical audience."""
        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=None,
            analysis_results=sample_analysis_results,
            audience=AudienceType.TECHNICAL,
        )

        # Should have detailed sections
        section_titles = [s.title for s in sections]
        assert "Detailed Analysis" in section_titles
        assert "Appendix" in section_titles

    def test_build_sections_executive_audience(self, agent, sample_insights):
        """Test section building for executive audience."""
        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=None,
            analysis_results=None,
            audience=AudienceType.EXECUTIVE,
        )

        # Should be concise
        section_titles = [s.title for s in sections]
        assert "Executive Summary" in section_titles
        # Executive reports don't need appendix
        assert "Appendix" not in section_titles

    def test_generate_executive_summary(self, agent, sample_insights):
        """Test executive summary generation."""
        summary = agent._generate_executive_summary(
            insights=sample_insights,
            analysis_results=None,
            audience=AudienceType.EXECUTIVE,
        )

        assert "Strong quarter" in summary
        assert "Key Highlights" in summary
        assert "Recommended Actions" in summary

    def test_generate_methodology_section(self, agent, sample_analysis_results):
        """Test methodology section generation."""
        methodology = agent._generate_methodology_section(sample_analysis_results)

        assert "Data Sources" in methodology
        assert "Analytical Approach" in methodology
        assert "Descriptive statistics" in methodology
        assert "Correlation analysis" in methodology

    def test_generate_findings_section(self, agent, sample_insights, sample_visualizations):
        """Test findings section generation."""
        result = agent._generate_findings_section(sample_insights, sample_visualizations)

        assert "Finding 1" in result["content"]
        assert "Strong Sales Growth" in result["content"]
        assert "Evidence" in result["content"]
        assert "Business Impact" in result["content"]
        assert len(result["chart_ids"]) == 2

    def test_generate_recommendations_section(self, agent, sample_insights):
        """Test recommendations section generation."""
        recommendations = agent._generate_recommendations_section(sample_insights)

        assert "Expand marketing budget" in recommendations
        assert "Expected Impact" in recommendations
        assert "Areas for Further Investigation" in recommendations

    def test_create_draft(self, agent, sample_insights, sample_visualizations):
        """Test draft creation."""
        metadata = ReportMetadata(
            title="Test Report",
            audience=AudienceType.GENERAL,
            format=ReportFormat.PDF,
        )

        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=sample_visualizations,
            analysis_results=None,
            audience=AudienceType.GENERAL,
        )

        draft = agent._create_draft(metadata, sections, sample_visualizations)

        assert draft.metadata.title == "Test Report"
        assert draft.estimated_pages >= 1
        # Charts are included from both Findings and Visualizations sections
        assert len(draft.charts_included) >= 2
        assert "chart_001" in draft.charts_included
        assert "chart_002" in draft.charts_included
        assert draft.approval_status == "pending"

    def test_render_markdown(self, agent, sample_insights):
        """Test markdown rendering."""
        metadata = ReportMetadata(
            title="Markdown Test",
            subtitle="Subtitle",
            audience=AudienceType.GENERAL,
            format=ReportFormat.MARKDOWN,
        )

        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=None,
            analysis_results=None,
            audience=AudienceType.GENERAL,
        )

        content = agent._render_markdown(metadata, sections)

        assert "# Markdown Test" in content
        assert "## Subtitle" in content
        assert "Table of Contents" in content
        assert "Executive Summary" in content
        assert "---" in content  # Section separators

    def test_render_html(self, agent, sample_insights):
        """Test HTML rendering."""
        metadata = ReportMetadata(
            title="HTML Test",
            audience=AudienceType.GENERAL,
            format=ReportFormat.HTML,
        )

        sections = agent._build_sections(
            insights=sample_insights,
            visualizations=None,
            analysis_results=None,
            audience=AudienceType.GENERAL,
        )

        markdown_content = agent._render_markdown(metadata, sections)
        html_content = agent._render_html(metadata, sections, markdown_content)

        assert "<!DOCTYPE html>" in html_content
        assert "<title>HTML Test</title>" in html_content
        assert "font-family" in html_content  # Has CSS

    def test_get_format_options(self, agent):
        """Test getting format options."""
        options = agent.get_format_options()

        assert len(options) == 4
        option_ids = [o.id for o in options]
        assert "pdf" in option_ids
        assert "pptx" in option_ids
        assert "html" in option_ids
        assert "markdown" in option_ids
        assert any(o.recommended for o in options)

    def test_get_audience_options(self, agent):
        """Test getting audience options."""
        options = agent.get_audience_options()

        assert len(options) == 3
        option_ids = [o.id for o in options]
        assert "executive" in option_ids
        assert "technical" in option_ids
        assert "general" in option_ids

    def test_format_output_report(self, agent, sample_insights):
        """Test output formatting for completed report."""
        metadata = ReportMetadata(
            title="Format Test",
            format=ReportFormat.MARKDOWN,
            word_count=1500,
            page_count=3,
        )

        output = ReportOutput(
            metadata=metadata,
            sections=[
                ReportSection(title="Summary", content="Test", order=1),
                ReportSection(title="Findings", content="Test", order=2),
            ],
            file_path=Path("/tmp/test.md"),
            file_content="# Report content",
            size_bytes=1000,
            generation_notes=["Note 1"],
        )

        formatted = agent.format_output(output)

        assert "Format Test" in formatted
        assert "MARKDOWN" in formatted
        assert "Word Count" in formatted
        assert "/tmp/test.md" in formatted
        assert "Note 1" in formatted

    def test_format_output_draft(self, agent, sample_insights):
        """Test output formatting for draft."""
        metadata = ReportMetadata(
            title="Draft Test",
            format=ReportFormat.PDF,
            audience=AudienceType.EXECUTIVE,
        )

        draft = ReportDraft(
            metadata=metadata,
            sections=[
                ReportSection(title="Summary", content="Test", order=1, section_type="summary"),
                ReportSection(title="Findings", content="Test", order=2, section_type="findings"),
            ],
            estimated_pages=5,
            charts_included=["chart_001", "chart_002"],
        )

        formatted = agent.format_output(draft)

        assert "Draft" in formatted
        assert "Awaiting Approval" in formatted
        assert "Draft Test" in formatted
        assert "Estimated Pages" in formatted and "5" in formatted
        assert "Charts Included" in formatted and "2" in formatted

    def test_create_draft_structure(self, agent, sample_insights, sample_visualizations):
        """Test create_draft_structure method."""
        draft = agent.create_draft_structure(
            insights=sample_insights,
            visualizations=sample_visualizations,
            title="Structure Test",
            audience=AudienceType.TECHNICAL,
        )

        assert draft.metadata.title == "Structure Test"
        assert draft.metadata.audience == AudienceType.TECHNICAL
        assert len(draft.sections) > 0
        assert draft.approval_status == "pending"


class TestReportSection:
    """Test suite for ReportSection dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        section = ReportSection(
            title="Test Section",
            content="Section content here",
            order=1,
            section_type="text",
            chart_ids=["chart_001"],
            tables=[{"name": "table1"}],
            subsections=[
                ReportSection(
                    title="Subsection",
                    content="Subsection content",
                    order=1,
                )
            ],
        )

        d = section.to_dict()

        assert d["title"] == "Test Section"
        assert d["content"] == "Section content here"
        assert d["order"] == 1
        assert d["section_type"] == "text"
        assert d["chart_ids"] == ["chart_001"]
        assert len(d["tables"]) == 1
        assert len(d["subsections"]) == 1
        assert d["subsections"][0]["title"] == "Subsection"


class TestReportMetadata:
    """Test suite for ReportMetadata dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ReportMetadata(
            title="Test Report",
            subtitle="A subtitle",
            author="Test Author",
            audience=AudienceType.EXECUTIVE,
            format=ReportFormat.PDF,
            version="2.0",
            confidentiality="Confidential",
            page_count=10,
            word_count=5000,
        )

        d = metadata.to_dict()

        assert d["title"] == "Test Report"
        assert d["subtitle"] == "A subtitle"
        assert d["author"] == "Test Author"
        assert d["audience"] == "executive"
        assert d["format"] == "pdf"
        assert d["version"] == "2.0"
        assert d["confidentiality"] == "Confidential"
        assert d["page_count"] == 10
        assert d["word_count"] == 5000
        assert "created_at" in d


class TestReportDraft:
    """Test suite for ReportDraft dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ReportMetadata(title="Draft", format=ReportFormat.HTML)
        draft = ReportDraft(
            metadata=metadata,
            sections=[ReportSection(title="Section 1", content="Content", order=1)],
            estimated_pages=3,
            charts_included=["chart_001"],
            tables_included=2,
            approval_status="pending",
            feedback="",
        )

        d = draft.to_dict()

        assert d["estimated_pages"] == 3
        assert d["charts_included"] == ["chart_001"]
        assert d["tables_included"] == 2
        assert d["approval_status"] == "pending"
        assert len(d["sections"]) == 1


class TestReportOutput:
    """Test suite for ReportOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ReportMetadata(title="Output", format=ReportFormat.MARKDOWN)
        output = ReportOutput(
            metadata=metadata,
            sections=[ReportSection(title="Section 1", content="Content", order=1)],
            file_path=Path("/tmp/report.md"),
            file_content="# Report",
            size_bytes=500,
            generation_notes=["Note 1", "Note 2"],
        )

        d = output.to_dict()

        assert d["file_path"] == "/tmp/report.md"
        assert d["size_bytes"] == 500
        assert d["generation_notes"] == ["Note 1", "Note 2"]
        assert len(d["sections"]) == 1


class TestReportFormat:
    """Test suite for ReportFormat enum."""

    def test_all_formats_defined(self):
        """Test all expected formats are defined."""
        expected = ["pdf", "pptx", "html", "markdown"]
        for fmt in expected:
            assert ReportFormat(fmt) is not None

    def test_format_values(self):
        """Test format string values."""
        assert ReportFormat.PDF.value == "pdf"
        assert ReportFormat.POWERPOINT.value == "pptx"
        assert ReportFormat.HTML.value == "html"
        assert ReportFormat.MARKDOWN.value == "markdown"


class TestAudienceType:
    """Test suite for AudienceType enum."""

    def test_all_audiences_defined(self):
        """Test all expected audiences are defined."""
        expected = ["executive", "technical", "general"]
        for aud in expected:
            assert AudienceType(aud) is not None

    def test_audience_values(self):
        """Test audience string values."""
        assert AudienceType.EXECUTIVE.value == "executive"
        assert AudienceType.TECHNICAL.value == "technical"
        assert AudienceType.GENERAL.value == "general"
