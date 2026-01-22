"""Tests for Obsidian integration."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.obsidian import ObsidianNote, ObsidianVault


class TestObsidianNote:
    """Tests for ObsidianNote dataclass."""

    def test_note_creation(self):
        """Test basic note creation."""
        note = ObsidianNote(
            title="Test Note",
            content="This is a test note.",
        )
        assert note.title == "Test Note"
        assert note.content == "This is a test note."
        assert note.tags == []
        assert note.links == []
        assert note.frontmatter == {}
        assert note.created_at is not None

    def test_note_with_tags(self):
        """Test note with tags."""
        note = ObsidianNote(
            title="Tagged Note",
            content="Content",
            tags=["analysis", "statistical"],
        )
        assert note.tags == ["analysis", "statistical"]

    def test_note_with_links(self):
        """Test note with links."""
        note = ObsidianNote(
            title="Linked Note",
            content="Content",
            links=["Related Analysis", "Previous Report"],
        )
        assert len(note.links) == 2

    def test_note_with_frontmatter(self):
        """Test note with custom frontmatter."""
        note = ObsidianNote(
            title="Custom Note",
            content="Content",
            frontmatter={"type": "analysis", "dataset": "sales"},
        )
        assert note.frontmatter["type"] == "analysis"
        assert note.frontmatter["dataset"] == "sales"

    def test_to_markdown_basic(self):
        """Test markdown generation for basic note."""
        note = ObsidianNote(
            title="Test Note",
            content="This is the content.",
        )
        md = note.to_markdown()

        assert "---" in md
        assert "# Test Note" in md
        assert "This is the content." in md
        assert "date:" in md

    def test_to_markdown_with_tags(self):
        """Test markdown generation with tags."""
        note = ObsidianNote(
            title="Tagged Note",
            content="Content",
            tags=["analysis", "report"],
        )
        md = note.to_markdown()

        assert "tags: [analysis, report]" in md

    def test_to_markdown_with_links(self):
        """Test markdown generation with links."""
        note = ObsidianNote(
            title="Linked Note",
            content="Content",
            links=["Related Analysis", "Previous Report"],
        )
        md = note.to_markdown()

        assert "## Related" in md
        assert "[[Related Analysis]]" in md
        assert "[[Previous Report]]" in md

    def test_to_markdown_with_wiki_links(self):
        """Test that existing wiki-style links are preserved."""
        note = ObsidianNote(
            title="Test",
            content="Content",
            links=["[[Already Linked]]", "Plain Link"],
        )
        md = note.to_markdown()

        assert "[[Already Linked]]" in md
        assert "[[Plain Link]]" in md

    def test_to_markdown_with_frontmatter(self):
        """Test markdown generation with frontmatter."""
        note = ObsidianNote(
            title="Custom Note",
            content="Content",
            frontmatter={"type": "analysis", "confidence": "high"},
        )
        md = note.to_markdown()

        assert "type: analysis" in md
        assert "confidence: high" in md


class TestObsidianVault:
    """Tests for ObsidianVault class."""

    @pytest.fixture
    def temp_vault(self, tmp_path):
        """Create a temporary vault directory."""
        vault_path = tmp_path / "test_vault"
        vault_path.mkdir()
        return vault_path

    @pytest.fixture
    def vault(self, temp_vault):
        """Create an ObsidianVault instance."""
        return ObsidianVault(vault_path=temp_vault)

    def test_vault_initialization(self, temp_vault):
        """Test vault initialization."""
        vault = ObsidianVault(vault_path=temp_vault)
        assert vault.vault_path == temp_vault
        assert vault.folder_name == "The Analyst"

    def test_vault_custom_folder(self, temp_vault):
        """Test vault with custom folder name."""
        vault = ObsidianVault(vault_path=temp_vault, folder_name="Custom")
        assert vault.folder_name == "Custom"
        assert vault.base_path == temp_vault / "Custom"

    def test_vault_from_env(self, temp_vault):
        """Test vault path from environment variable."""
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(temp_vault)}):
            vault = ObsidianVault()
            assert vault.vault_path == temp_vault

    def test_vault_default_path(self):
        """Test default vault path."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OBSIDIAN_VAULT_PATH", None)
            vault = ObsidianVault()
            assert vault.vault_path == Path.home() / "Documents" / "Obsidian"

    def test_base_path(self, vault, temp_vault):
        """Test base_path property."""
        assert vault.base_path == temp_vault / "The Analyst"

    def test_analyses_path(self, vault, temp_vault):
        """Test analyses_path property."""
        assert vault.analyses_path == temp_vault / "The Analyst" / "Analyses"

    def test_notifications_path(self, vault, temp_vault):
        """Test notifications_path property."""
        assert vault.notifications_path == temp_vault / "The Analyst" / "Notifications"

    def test_daily_notes_path(self, vault, temp_vault):
        """Test daily_notes_path property."""
        assert vault.daily_notes_path == temp_vault / "The Analyst" / "Daily"

    def test_is_available_true(self, vault, temp_vault):
        """Test is_available returns True for existing vault."""
        assert vault.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False for nonexistent vault."""
        vault = ObsidianVault(vault_path=Path("/nonexistent/path"))
        assert vault.is_available() is False

    def test_get_daily_note_path(self, vault):
        """Test daily note path generation."""
        date = datetime(2026, 1, 22)
        path = vault.get_daily_note_path(date)
        assert path.name == "2026-01-22.md"
        assert "Daily" in str(path)

    def test_get_daily_note_path_default_today(self, vault):
        """Test daily note path defaults to today."""
        path = vault.get_daily_note_path()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        assert today in path.name

    def test_create_note(self, vault, temp_vault):
        """Test creating a note."""
        note = ObsidianNote(
            title="Test Analysis",
            content="Analysis results here.",
            tags=["test"],
        )
        filepath = vault.create_note(note)

        assert filepath.exists()
        assert filepath.parent == temp_vault / "The Analyst" / "Analyses"

        content = filepath.read_text()
        assert "# Test Analysis" in content
        assert "Analysis results here." in content

    def test_create_note_custom_subfolder(self, vault, temp_vault):
        """Test creating a note in custom subfolder."""
        note = ObsidianNote(title="Custom", content="Content")
        filepath = vault.create_note(note, subfolder="Reports")

        assert filepath.parent == temp_vault / "The Analyst" / "Reports"

    def test_create_note_sanitizes_title(self, vault):
        """Test that note titles are sanitized for filenames."""
        note = ObsidianNote(
            title="Analysis: Test/Data <Report>",
            content="Content",
        )
        filepath = vault.create_note(note)

        # Should not contain special characters
        assert "/" not in filepath.name
        assert "<" not in filepath.name
        assert ">" not in filepath.name

    def test_create_analysis_note(self, vault, temp_vault):
        """Test creating an analysis note."""
        result = {
            "title": "Sales Analysis",
            "summary": "Revenue increased by 15%",
            "methodology": "Linear regression",
            "findings": ["Q4 strongest quarter", "Weekend sales higher"],
            "recommendations": ["Increase weekend staff", "Focus on Q4 marketing"],
            "confidence": "95%",
            "limitations": "Data from 2025 only",
            "analysis_type": "statistical",
        }
        filepath = vault.create_analysis_note(result)

        assert filepath.exists()
        content = filepath.read_text()

        assert "# Sales Analysis" in content
        assert "## Summary" in content
        assert "Revenue increased by 15%" in content
        assert "## Methodology" in content
        assert "Linear regression" in content
        assert "## Key Findings" in content
        assert "Q4 strongest quarter" in content
        assert "## Recommendations" in content
        assert "Increase weekend staff" in content
        assert "## Verification" in content
        assert "95%" in content

    def test_create_analysis_note_custom_title(self, vault):
        """Test creating analysis note with custom title."""
        result = {"summary": "Test summary"}
        filepath = vault.create_analysis_note(result, title="Custom Title")

        content = filepath.read_text()
        assert "# Custom Title" in content

    def test_create_analysis_note_minimal(self, vault):
        """Test creating analysis note with minimal data."""
        result = {"summary": "Basic summary only"}
        filepath = vault.create_analysis_note(result)

        assert filepath.exists()
        content = filepath.read_text()
        assert "Basic summary only" in content

    def test_link_related_analyses(self, vault):
        """Test linking related analyses."""
        # Create a note first
        note = ObsidianNote(
            title="Main Analysis",
            content="Main content",
        )
        main_path = vault.create_note(note)

        # Link related analyses
        vault.link_related_analyses(
            main_path,
            ["Previous Analysis", "Related Report"],
        )

        content = main_path.read_text()
        assert "## Related Analyses" in content
        assert "[[Previous Analysis]]" in content
        assert "[[Related Report]]" in content

    def test_link_related_analyses_existing_section(self, vault):
        """Test linking when Related Analyses section already exists."""
        note = ObsidianNote(
            title="Main Analysis",
            content="Content\n\n## Related Analyses\n\n- [[First Link]]",
        )
        main_path = vault.create_note(note)

        vault.link_related_analyses(main_path, ["New Link"])

        content = main_path.read_text()
        assert "[[First Link]]" in content
        assert "[[New Link]]" in content

    def test_link_related_analyses_with_path(self, vault):
        """Test linking with Path objects."""
        note = ObsidianNote(title="Main", content="Content")
        main_path = vault.create_note(note)

        # Create a related note
        related_note = ObsidianNote(title="Related", content="Related content")
        related_path = vault.create_note(related_note)

        vault.link_related_analyses(main_path, [related_path])

        content = main_path.read_text()
        # Should extract title from path
        assert "[[" in content and "]]" in content

    def test_append_to_daily_note_new(self, vault):
        """Test appending to a new daily note."""
        date = datetime(2026, 1, 22)
        path = vault.append_to_daily_note(
            content="Analysis completed at 10:00",
            section="Analyses",
            date=date,
        )

        assert path.exists()
        content = path.read_text()
        assert "## Analyses" in content
        assert "Analysis completed at 10:00" in content
        assert "2026-01-22" in content

    def test_append_to_daily_note_existing(self, vault):
        """Test appending to an existing daily note."""
        date = datetime(2026, 1, 22)

        # First append
        path = vault.append_to_daily_note(
            content="First entry",
            section="Analyses",
            date=date,
        )

        # Second append
        vault.append_to_daily_note(
            content="Second entry",
            section="Analyses",
            date=date,
        )

        content = path.read_text()
        assert "First entry" in content
        assert "Second entry" in content

    def test_append_to_daily_note_new_section(self, vault):
        """Test appending to a new section in existing daily note."""
        date = datetime(2026, 1, 22)

        vault.append_to_daily_note(
            content="Analysis entry",
            section="Analyses",
            date=date,
        )

        path = vault.append_to_daily_note(
            content="Report entry",
            section="Reports",
            date=date,
        )

        content = path.read_text()
        assert "## Analyses" in content
        assert "## Reports" in content
        assert "Analysis entry" in content
        assert "Report entry" in content

    def test_get_recent_analyses_empty(self, vault):
        """Test getting recent analyses from empty vault."""
        notes = vault.get_recent_analyses()
        assert notes == []

    def test_get_recent_analyses(self, vault):
        """Test getting recent analyses."""
        # Create some notes
        for i in range(5):
            note = ObsidianNote(title=f"Analysis {i}", content=f"Content {i}")
            vault.create_note(note)

        notes = vault.get_recent_analyses(limit=3)
        assert len(notes) == 3

    def test_get_recent_analyses_sorted_by_time(self, vault):
        """Test that recent analyses are sorted by modification time."""
        import time

        # Create notes with slight delay
        for i in range(3):
            note = ObsidianNote(title=f"Analysis {i}", content=f"Content {i}")
            vault.create_note(note)
            time.sleep(0.01)  # Small delay to ensure different mtime

        notes = vault.get_recent_analyses()
        # Most recent should be first
        mtimes = [n.stat().st_mtime for n in notes]
        assert mtimes == sorted(mtimes, reverse=True)

    def test_search_notes_no_match(self, vault):
        """Test search with no matches."""
        note = ObsidianNote(title="Test", content="Hello world")
        vault.create_note(note)

        matches = vault.search_notes("nonexistent")
        assert matches == []

    def test_search_notes_with_match(self, vault):
        """Test search with matches."""
        note1 = ObsidianNote(title="Sales Analysis", content="Revenue increased")
        note2 = ObsidianNote(title="Traffic Analysis", content="Page views up")
        vault.create_note(note1)
        vault.create_note(note2)

        matches = vault.search_notes("analysis")
        assert len(matches) == 2

    def test_search_notes_case_insensitive(self, vault):
        """Test that search is case insensitive."""
        note = ObsidianNote(title="Test", content="Revenue ANALYSIS")
        vault.create_note(note)

        matches = vault.search_notes("analysis")
        assert len(matches) == 1

    def test_search_notes_in_folder(self, vault):
        """Test search within specific folder."""
        note1 = ObsidianNote(title="Analysis", content="Content with keyword")
        note2 = ObsidianNote(title="Report", content="Content with keyword")
        vault.create_note(note1, subfolder="Analyses")
        vault.create_note(note2, subfolder="Reports")

        matches = vault.search_notes("keyword", folder="Analyses")
        assert len(matches) == 1

    def test_ensure_directories(self, vault, temp_vault):
        """Test that directories are created."""
        vault._ensure_directories()

        assert vault.analyses_path.exists()
        assert vault.notifications_path.exists()
        assert vault.daily_notes_path.exists()
