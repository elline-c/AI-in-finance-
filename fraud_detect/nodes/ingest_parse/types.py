"""Shared page-content types for the layout-tolerant Node 1 parser.

``Word`` mirrors the fields we use from pdfplumber's ``extract_words`` output
(``x0``/``x1`` are horizontal edges in PDF points, ``top`` is the distance
from the page top). ``Line`` is one line of ``extract_text``. ``PageContent``
bundles both views of one page.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float


@dataclass(frozen=True)
class Line:
    text: str
    top: float = 0.0


@dataclass
class PageContent:
    page: int  # 1-based page number within the file
    lines: list[Line] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)
