"""
Spy AI — Document Preprocessor

Separates source text into classified segments:
- CONTENT: actual article/source text to analyze
- INSTRUCTION: assignment instructions ("Translate the following...", "Write a commentary...")
- METADATA: document metadata ("Name and Surname:", "Student ID:", "Assignment 3")
- URL: URLs and source references
- HEADING: article headings/titles
- AUTHOR: author information ("By Elaina Zachos")
- DATE: publication dates ("Published November 10, 2017")

This ensures that assignment instructions and metadata are never analyzed as article terminology.
"""

import re
from typing import List
from models import DocumentSegment


# ---------------------------------------------------------------------------
# Patterns for detecting non-content segments
# ---------------------------------------------------------------------------

# Assignment instruction patterns
INSTRUCTION_PATTERNS = [
    # "Translate the following ... into Turkish"
    re.compile(
        r'^\d*\.?\s*Translate\s+the\s+(?:following|below)\b.*$',
        re.IGNORECASE | re.MULTILINE
    ),
    # "Write a commentary on your translation..."
    re.compile(
        r'^\d*\.?\s*Write\s+(?:a\s+)?(?:commentary|essay|report|summary)\b.*$',
        re.IGNORECASE | re.MULTILINE
    ),
    # "(min. 300 words)" or "(minimum 300 words)"
    re.compile(
        r'\(?\s*(?:min\.?|minimum|max\.?|maximum)\s*\.?\s*\d+\s*(?:words?|kelime)\s*\)?',
        re.IGNORECASE
    ),
    # "Assume that the translation will be published..."
    re.compile(
        r'^\(?\s*Assume\s+that\b.*$',
        re.IGNORECASE | re.MULTILINE
    ),
]

# Metadata patterns (lines that are form fields or headers)
METADATA_PATTERNS = [
    # "Name and Surname:", "Student ID:", "Name:", "Surname:"
    re.compile(
        r'^(?:Name\s*(?:and\s*Surname)?|Student\s*ID|Surname|Ad\s*(?:ve\s*)?Soyad|Öğrenci\s*No)\s*:\s*$',
        re.IGNORECASE | re.MULTILINE
    ),
    # "Assignment N" or "Ödev N"
    re.compile(
        r'^(?:Assignment|Ödev|Homework|HW)\s*\d+\s*$',
        re.IGNORECASE | re.MULTILINE
    ),
    # Course codes like "IMT 1109 Introduction to Translation I"
    re.compile(
        r'^[A-Z]{2,4}\s*\d{3,4}\s+.*$',
        re.MULTILINE
    ),
]

# URL pattern
URL_PATTERN = re.compile(
    r'(?:(?:Source|Kaynak)\s*:\s*)?'
    r'(https?://[^\s\)\]]+)',
    re.IGNORECASE
)

# Author line pattern: "By <Name>" at start of line
AUTHOR_PATTERN = re.compile(
    r'^By\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*$',
    re.MULTILINE
)

# Date line pattern: "Published Month DD, YYYY"
DATE_PATTERN = re.compile(
    r'^Published\s+'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{1,2},?\s+\d{4}\s*$',
    re.IGNORECASE | re.MULTILINE
)

# Source reference line: "Source: <url_or_text>"
SOURCE_REF_PATTERN = re.compile(
    r'^(?:Source|Kaynak)\s*:\s*(.+)$',
    re.IGNORECASE | re.MULTILINE
)


def segment_document(text: str) -> List[DocumentSegment]:
    """
    Classify each line/region of the input document as content or non-content.

    Returns a list of DocumentSegment objects covering the full text.
    Content segments are what should be analyzed by the NLP pipeline.
    Non-content segments (instructions, metadata, URLs) are preserved for
    display but excluded from term/entity analysis.
    """
    segments = []
    # Track which character ranges are non-content
    non_content_spans = []  # [(start, end, segment_type)]

    # --- Detect URLs ---
    for m in URL_PATTERN.finditer(text):
        url_start = m.start()
        url_end = m.end()
        # Extend to cover the full line containing the URL
        line_start = text.rfind('\n', 0, url_start) + 1
        line_end = text.find('\n', url_end)
        if line_end == -1:
            line_end = len(text)
        # Check if this line is ONLY a URL or "Source: URL"
        line_text = text[line_start:line_end].strip()
        if SOURCE_REF_PATTERN.match(line_text) or line_text == m.group(0).strip():
            non_content_spans.append((line_start, line_end, "url"))
        else:
            # URL is embedded in a longer line — just mark the URL portion
            non_content_spans.append((m.start(), m.end(), "url"))

    # --- Detect metadata lines ---
    for pattern in METADATA_PATTERNS:
        for m in pattern.finditer(text):
            non_content_spans.append((m.start(), m.end(), "metadata"))

    # --- Detect instruction lines ---
    for pattern in INSTRUCTION_PATTERNS:
        for m in pattern.finditer(text):
            non_content_spans.append((m.start(), m.end(), "instruction"))

    # --- Detect author lines ---
    for m in AUTHOR_PATTERN.finditer(text):
        non_content_spans.append((m.start(), m.end(), "author"))

    # --- Detect date lines ---
    for m in DATE_PATTERN.finditer(text):
        non_content_spans.append((m.start(), m.end(), "date"))

    # --- Detect source reference lines ---
    for m in SOURCE_REF_PATTERN.finditer(text):
        # Only if not already covered by URL detection
        already_covered = any(
            s <= m.start() and e >= m.end()
            for s, e, _ in non_content_spans
        )
        if not already_covered:
            non_content_spans.append((m.start(), m.end(), "url"))

    # Sort and merge overlapping spans
    non_content_spans.sort(key=lambda x: x[0])
    merged = []
    for start, end, seg_type in non_content_spans:
        if merged and start <= merged[-1][1]:
            # Merge overlapping spans, keep the broader type
            prev_start, prev_end, prev_type = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_type)
        else:
            merged.append((start, end, seg_type))

    # Build segments: fill gaps between non-content spans with "content"
    cursor = 0
    for start, end, seg_type in merged:
        if cursor < start:
            content_text = text[cursor:start]
            if content_text.strip():
                segments.append(DocumentSegment(
                    segment_type="content",
                    text=content_text,
                    start_offset=cursor,
                    end_offset=start
                ))
        seg_text = text[start:end]
        if seg_text.strip():
            segments.append(DocumentSegment(
                segment_type=seg_type,
                text=seg_text,
                start_offset=start,
                end_offset=end
            ))
        cursor = end

    # Remaining text after last non-content span
    if cursor < len(text):
        remaining = text[cursor:]
        if remaining.strip():
            segments.append(DocumentSegment(
                segment_type="content",
                text=remaining,
                start_offset=cursor,
                end_offset=len(text)
            ))

    # If no non-content was found, entire text is content
    if not segments:
        segments.append(DocumentSegment(
            segment_type="content",
            text=text,
            start_offset=0,
            end_offset=len(text)
        ))

    return segments


def get_content_text(segments: List[DocumentSegment]) -> str:
    """Extract only the content segments, joined together."""
    return "\n".join(s.text for s in segments if s.segment_type == "content")


def get_content_segments(segments: List[DocumentSegment]) -> List[DocumentSegment]:
    """Return only content segments."""
    return [s for s in segments if s.segment_type == "content"]


def extract_source_url(text: str) -> str | None:
    """Extract the source URL from the document if present."""
    m = URL_PATTERN.search(text)
    if m:
        return m.group(1)
    return None


def infer_domain_from_url(url: str) -> str | None:
    """Infer the document domain from a source URL."""
    if not url:
        return None
    url_lower = url.lower()
    domain_map = {
        "nationalgeographic.com": "science/wildlife/journalism",
        "bbc.com": "journalism/current_affairs",
        "bbc.co.uk": "journalism/current_affairs",
        "nhs.uk": "medicine/healthcare",
        "nih.gov": "medicine/biomedical",
        "nature.com": "science/research",
        "sciencedirect.com": "science/research",
        "who.int": "medicine/public_health",
        "reuters.com": "journalism/news",
        "theguardian.com": "journalism/news",
        "nytimes.com": "journalism/news",
        "washingtonpost.com": "journalism/news",
    }
    for domain, topic in domain_map.items():
        if domain in url_lower:
            return topic
    return None


def is_metadata_region(offset: int, segments: List[DocumentSegment]) -> bool:
    """Check if a character offset falls within a non-content segment."""
    for seg in segments:
        if seg.segment_type != "content" and seg.start_offset <= offset < seg.end_offset:
            return True
    return False
