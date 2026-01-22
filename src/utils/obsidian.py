"""
Obsidian integration for The Analyst platform.

Provides functionality to create and manage notes in an Obsidian vault.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ObsidianNote:
    """Represents an Obsidian note with content and metadata."""

    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_markdown(self) -> str:
        """
        Convert the note to markdown with YAML frontmatter.

        Returns:
            Formatted markdown string
        """
        lines = ["---"]

        # Build frontmatter
        fm = {
            "date": self.created_at.strftime("%Y-%m-%d"),
            "created": self.created_at.isoformat(),
            **self.frontmatter,
        }

        # Add tags if present
        if self.tags:
            fm["tags"] = self.tags

        # Write frontmatter
        for key, value in fm.items():
            if isinstance(value, list):
                lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")

        lines.append("---")
        lines.append("")

        # Add title
        lines.append(f"# {self.title}")
        lines.append("")

        # Add content
        lines.append(self.content)

        # Add links section if present
        if self.links:
            lines.append("")
            lines.append("## Related")
            lines.append("")
            for link in self.links:
                # Convert to wiki-style link if not already
                if not link.startswith("[["):
                    link = f"[[{link}]]"
                lines.append(f"- {link}")

        return "\n".join(lines)


class ObsidianVault:
    """
    Manages interactions with an Obsidian vault.

    Provides methods to create analysis notes, link related analyses,
    and manage the vault structure for The Analyst.
    """

    def __init__(
        self,
        vault_path: Path | str | None = None,
        folder_name: str = "The Analyst",
    ) -> None:
        """
        Initialize the Obsidian vault manager.

        Args:
            vault_path: Path to the Obsidian vault. If not provided,
                       will try to load from OBSIDIAN_VAULT_PATH env var
                       or default to ~/Documents/Obsidian.
            folder_name: Folder name within the vault for The Analyst notes.
        """
        if vault_path is None:
            vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")

        if vault_path is None:
            # Default to ~/Documents/Obsidian
            vault_path = Path.home() / "Documents" / "Obsidian"
        elif isinstance(vault_path, str):
            vault_path = Path(vault_path)

        self.vault_path = vault_path
        self.folder_name = folder_name
        self._base_path = self.vault_path / self.folder_name

    @property
    def base_path(self) -> Path:
        """Get the base path for The Analyst notes."""
        return self._base_path

    @property
    def analyses_path(self) -> Path:
        """Get the path for analysis notes."""
        return self._base_path / "Analyses"

    @property
    def notifications_path(self) -> Path:
        """Get the path for notification notes."""
        return self._base_path / "Notifications"

    @property
    def daily_notes_path(self) -> Path:
        """Get the path for daily notes."""
        return self._base_path / "Daily"

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.analyses_path.mkdir(parents=True, exist_ok=True)
        self.notifications_path.mkdir(parents=True, exist_ok=True)
        self.daily_notes_path.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        """
        Check if the Obsidian vault is available.

        Returns:
            True if the vault path exists and is a directory
        """
        return self.vault_path.exists() and self.vault_path.is_dir()

    def get_daily_note_path(self, date: datetime | None = None) -> Path:
        """
        Get the path for a daily note.

        Args:
            date: Date for the daily note. Defaults to today.

        Returns:
            Path to the daily note file
        """
        if date is None:
            date = datetime.utcnow()

        filename = date.strftime("%Y-%m-%d") + ".md"
        return self.daily_notes_path / filename

    def create_note(self, note: ObsidianNote, subfolder: str = "Analyses") -> Path:
        """
        Create a note in the vault.

        Args:
            note: The ObsidianNote to create
            subfolder: Subfolder within The Analyst folder

        Returns:
            Path to the created note
        """
        self._ensure_directories()

        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in note.title)
        safe_title = safe_title.strip().replace(" ", "_")

        # Add timestamp to ensure uniqueness
        timestamp = note.created_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title}.md"

        target_dir = self._base_path / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

        filepath = target_dir / filename
        filepath.write_text(note.to_markdown())

        return filepath

    def create_analysis_note(
        self,
        analysis_result: dict[str, Any],
        title: str | None = None,
    ) -> Path:
        """
        Create an analysis note from analysis results.

        Args:
            analysis_result: Dictionary containing analysis results with keys like:
                - title: Analysis title
                - summary: Summary of findings
                - methodology: Methods used
                - findings: Detailed findings
                - recommendations: Recommendations
                - confidence: Confidence level
                - limitations: Limitations
            title: Optional override for the note title

        Returns:
            Path to the created note
        """
        note_title = title or analysis_result.get("title", "Analysis Results")

        # Build content sections
        content_parts = []

        # Summary
        if "summary" in analysis_result:
            content_parts.append("## Summary")
            content_parts.append("")
            content_parts.append(analysis_result["summary"])
            content_parts.append("")

        # Key Findings
        if "findings" in analysis_result:
            content_parts.append("## Key Findings")
            content_parts.append("")
            findings = analysis_result["findings"]
            if isinstance(findings, list):
                for finding in findings:
                    content_parts.append(f"- {finding}")
            else:
                content_parts.append(str(findings))
            content_parts.append("")

        # Methodology
        if "methodology" in analysis_result:
            content_parts.append("## Methodology")
            content_parts.append("")
            content_parts.append(analysis_result["methodology"])
            content_parts.append("")

        # Recommendations
        if "recommendations" in analysis_result:
            content_parts.append("## Recommendations")
            content_parts.append("")
            recommendations = analysis_result["recommendations"]
            if isinstance(recommendations, list):
                for rec in recommendations:
                    content_parts.append(f"- {rec}")
            else:
                content_parts.append(str(recommendations))
            content_parts.append("")

        # Confidence and Limitations
        if "confidence" in analysis_result or "limitations" in analysis_result:
            content_parts.append("## Verification")
            content_parts.append("")
            if "confidence" in analysis_result:
                content_parts.append(f"**Confidence Level**: {analysis_result['confidence']}")
            if "limitations" in analysis_result:
                content_parts.append(f"**Limitations**: {analysis_result['limitations']}")
            content_parts.append("")

        # Build tags
        tags = ["analysis", "the-analyst"]
        if "analysis_type" in analysis_result:
            tags.append(analysis_result["analysis_type"].lower().replace(" ", "-"))
        if "dataset" in analysis_result:
            tags.append(f"dataset-{analysis_result['dataset'].lower().replace(' ', '-')}")

        # Build frontmatter
        frontmatter = {
            "type": "analysis-summary",
        }
        if "analysis_type" in analysis_result:
            frontmatter["analysis_type"] = analysis_result["analysis_type"]
        if "dataset" in analysis_result:
            frontmatter["dataset"] = analysis_result["dataset"]
        if "confidence" in analysis_result:
            frontmatter["confidence"] = analysis_result["confidence"]

        # Create the note
        note = ObsidianNote(
            title=note_title,
            content="\n".join(content_parts),
            tags=tags,
            frontmatter=frontmatter,
        )

        return self.create_note(note, subfolder="Analyses")

    def link_related_analyses(
        self,
        note_path: Path,
        related_notes: list[str | Path],
    ) -> None:
        """
        Add links to related analyses in an existing note.

        Args:
            note_path: Path to the note to update
            related_notes: List of related note titles or paths
        """
        if not note_path.exists():
            return

        content = note_path.read_text()

        # Format links
        links_section = "\n\n## Related Analyses\n\n"
        for related in related_notes:
            if isinstance(related, Path):
                # Extract title from filename
                related = related.stem.split("_", 2)[-1].replace("_", " ")
            # Add as wiki-style link
            links_section += f"- [[{related}]]\n"

        # Check if Related Analyses section already exists
        if "## Related Analyses" in content:
            # Append to existing section
            parts = content.split("## Related Analyses")
            # Find the end of the existing section (next ## or end of file)
            rest = parts[1]
            next_section = rest.find("\n## ")
            if next_section > 0:
                content = (
                    parts[0]
                    + "## Related Analyses"
                    + rest[:next_section]
                    + links_section.split("## Related Analyses\n\n")[1]
                    + rest[next_section:]
                )
            else:
                content = (
                    parts[0]
                    + "## Related Analyses"
                    + rest
                    + "\n"
                    + links_section.split("## Related Analyses\n\n")[1]
                )
        else:
            # Add new section at the end
            content += links_section

        note_path.write_text(content)

    def append_to_daily_note(
        self,
        content: str,
        section: str = "Analyses",
        date: datetime | None = None,
    ) -> Path:
        """
        Append content to a daily note.

        Args:
            content: Content to append
            section: Section header to append under
            date: Date for the daily note. Defaults to today.

        Returns:
            Path to the daily note
        """
        self._ensure_directories()

        daily_path = self.get_daily_note_path(date)

        if daily_path.exists():
            existing = daily_path.read_text()
        else:
            # Create new daily note
            if date is None:
                date = datetime.utcnow()
            existing = f"""---
date: {date.strftime("%Y-%m-%d")}
tags: [daily, the-analyst]
---

# Daily Note - {date.strftime("%Y-%m-%d")}

"""

        # Check if section exists
        section_header = f"## {section}"
        if section_header in existing:
            # Append to existing section
            parts = existing.split(section_header)
            rest = parts[1]
            next_section = rest.find("\n## ")
            if next_section > 0:
                updated = (
                    parts[0]
                    + section_header
                    + rest[:next_section]
                    + "\n"
                    + content
                    + rest[next_section:]
                )
            else:
                updated = parts[0] + section_header + rest + "\n" + content
        else:
            # Add new section
            updated = existing + f"\n{section_header}\n\n{content}\n"

        daily_path.write_text(updated)
        return daily_path

    def get_recent_analyses(self, limit: int = 10) -> list[Path]:
        """
        Get paths to recent analysis notes.

        Args:
            limit: Maximum number of notes to return

        Returns:
            List of paths to recent analysis notes, sorted by modification time
        """
        if not self.analyses_path.exists():
            return []

        notes = list(self.analyses_path.glob("*.md"))
        notes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return notes[:limit]

    def search_notes(self, query: str, folder: str | None = None) -> list[Path]:
        """
        Search for notes containing a query string.

        Args:
            query: Text to search for
            folder: Optional folder to limit search to

        Returns:
            List of paths to matching notes
        """
        search_path = self._base_path / folder if folder else self._base_path

        if not search_path.exists():
            return []

        matches = []
        query_lower = query.lower()

        for note_path in search_path.rglob("*.md"):
            try:
                content = note_path.read_text().lower()
                if query_lower in content:
                    matches.append(note_path)
            except (OSError, UnicodeDecodeError):
                continue

        return matches
