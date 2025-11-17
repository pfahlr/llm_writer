from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Section(BaseModel):
  """Represents a chapter or section in the outline."""

  id: str = Field(..., description="Unique section ID matching TaskSpec.context.outline_id")
  title: str
  summary: str = Field(..., description="Summary injected into prompts")
  subsections: Optional[List[Section]] = None

  class Config:
    # Enable recursive types for nested subsections
    arbitrary_types_allowed = True


class Part(BaseModel):
  """Major division of the work (part, act, book)."""

  id: str
  title: str
  summary: Optional[str] = None
  sections: List[Section] = Field(default_factory=list)


class Outline(BaseModel):
  """Root outline structure for a multi-part work."""

  id: str = Field(..., description="Identifier for the work")
  title: str
  parts: List[Part] = Field(default_factory=list)

  def find_section(self, section_id: str) -> Optional[Section]:
    """Find a section by ID anywhere in the outline tree."""
    for part in self.parts:
      for section in part.sections:
        if section.id == section_id:
          return section
        # Check subsections recursively
        found = self._find_in_subsections(section, section_id)
        if found:
          return found
    return None

  def _find_in_subsections(self, section: Section, section_id: str) -> Optional[Section]:
    """Recursively search subsections."""
    if not section.subsections:
      return None
    for subsection in section.subsections:
      if subsection.id == section_id:
        return subsection
      found = self._find_in_subsections(subsection, section_id)
      if found:
        return found
    return None

  def get_context_for_section(
    self, section_id: str, include_siblings: bool = True
  ) -> Optional[Dict]:
    """
    Build context dictionary for a section including:
    - Part info (where this section lives)
    - Section summary
    - Sibling sections (previous/next chapters)
    - Parent section if nested
    """
    section = self.find_section(section_id)
    if not section:
      return None

    # Find the part containing this section
    containing_part = None
    siblings = []
    for part in self.parts:
      if section in part.sections:
        containing_part = part
        siblings = part.sections if include_siblings else []
        break

    # Build context
    context = {
      "section": {
        "id": section.id,
        "title": section.title,
        "summary": section.summary,
      },
      "part": {
        "id": containing_part.id if containing_part else None,
        "title": containing_part.title if containing_part else None,
        "summary": containing_part.summary if containing_part else None,
      },
    }

    if include_siblings and siblings:
      idx = siblings.index(section)
      context["previous_section"] = (
        {"id": siblings[idx - 1].id, "title": siblings[idx - 1].title} if idx > 0 else None
      )
      context["next_section"] = (
        {"id": siblings[idx + 1].id, "title": siblings[idx + 1].title}
        if idx < len(siblings) - 1
        else None
      )
      context["sibling_sections"] = [
        {"id": s.id, "title": s.title, "summary": s.summary} for s in siblings
      ]

    return context
