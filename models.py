"""
Spy AI — Data Models for Occurrence-Centric Analysis

Every analyzed item is represented as an independent Occurrence or EntitySpan
with its own context, POS, morphology, confidence, and provenance.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Semantic Categories
# ---------------------------------------------------------------------------

class Category:
    """Classification categories for analyzed items."""
    GENERAL_WORD = "general_word"
    USEFUL_LEXICAL_ITEM = "useful_lexical_item"
    LEXICAL_TERM = "useful_lexical_item"  # backwards compatibility alias
    SPECIALIZED_TERM = "specialized_term"
    COLLOCATION = "collocation"
    IDIOM = "idiom"
    PHRASAL_VERB = "phrasal_verb"
    NAMED_ENTITY = "named_entity"
    MULTIWORD_TERM = "multiword_term"
    ORDINARY_PHRASE = "ordinary_phrase"
    BIOLOGICAL_SPECIES = "biological_species"
    SOURCE_METADATA = "source_metadata"
    GRAMMATICAL_UNIT = "grammatical_unit"
    URL = "url"


class EntityType:
    """Fine-grained entity type classification."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    INSTITUTION = "INSTITUTION"
    PUBLICATION = "PUBLICATION"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    PRODUCT = "PRODUCT"
    BIOLOGICAL_SPECIES = "BIOLOGICAL_SPECIES"
    DATE = "DATE"
    NORP = "NORP"  # Nationalities, religious/political groups


class ConfidenceLevel:
    """Human-readable confidence tiers for UI gating."""
    HIGH = "high"       # >= 0.75 — show normally
    MEDIUM = "medium"   # 0.45-0.75 — show with "contextual" indicator
    LOW = "low"         # < 0.45 — suppress fabricated content

    @staticmethod
    def from_score(score: float) -> str:
        if score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.45:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Document Segmentation
# ---------------------------------------------------------------------------

@dataclass
class DocumentSegment:
    """A classified segment of the input document."""
    segment_type: str       # "content", "instruction", "metadata", "url", "heading", "author", "date"
    text: str
    start_offset: int
    end_offset: int


# ---------------------------------------------------------------------------
# Sense Candidate
# ---------------------------------------------------------------------------

@dataclass
class SenseCandidate:
    """A candidate word sense with provenance."""
    definition: str
    source: str             # "WordNet", "Gemini", "Google Dictionary", "Contextual inference"
    score: float = 0.0
    is_primary: bool = False


# ---------------------------------------------------------------------------
# Translation Candidate
# ---------------------------------------------------------------------------

@dataclass
class TranslationCandidate:
    """A candidate translation with provenance."""
    text: str
    source: str             # "Google Translate", "Gemini", "DeepL", "Bilingual dictionary"
    score: float = 0.0


# ---------------------------------------------------------------------------
# Occurrence (core unit of analysis)
# ---------------------------------------------------------------------------

@dataclass
class Occurrence:
    """
    The fundamental unit of analysis. Every detected token or span in the
    source text gets its own Occurrence with independent context, POS,
    sense, and translation analysis.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Text
    surface: str = ""               # "motivated"
    lemma: str = ""                 # "motivate"
    normalized_form: str = ""       # lowercase surface

    # Position
    sentence_idx: int = 0           # index of sentence in document
    sentence_text: str = ""         # full sentence text
    context: str = ""               # prev + current + next sentence
    start_offset: int = 0           # char offset in source text
    end_offset: int = 0             # char offset end

    # Linguistics
    pos: str = ""                   # "VERB", "ADJ", "NOUN", "PROPN"
    pos_fine: str = ""              # Fine-grained POS (e.g., "VBN")
    morphology: dict = field(default_factory=dict)  # {"Tense": "Past", "VerbForm": "Part"}
    syntactic_role: str = ""        # "nsubj", "amod", "dobj", etc.

    # Classification & Ranking
    category: str = Category.GENERAL_WORD
    translation_value: float = 0.0  # 0.0 to 1.0 translation-research score
    extraction_reason: str = ""     # Human-readable justification for extraction
    entity_type: Optional[str] = None
    entity_span_id: Optional[str] = None

    # Senses
    candidate_senses: list = field(default_factory=list)  # [SenseCandidate, ...]
    selected_sense: Optional[dict] = None  # {"definition": ..., "source": ...}

    # Translations
    translations: list = field(default_factory=list)  # [TranslationCandidate, ...]
    selected_translation: str = ""

    # Confidence
    confidence: dict = field(default_factory=lambda: {
        "term": 0.0,
        "sense": 0.0,
        "translation": 0.0,
        "overall": 0.0
    })

    # Provenance
    evidence: list = field(default_factory=list)  # ["WordNet", "Google Translate", ...]

    # Metadata flag
    is_metadata: bool = False

    def confidence_level(self) -> str:
        """Return the human-readable confidence tier."""
        return ConfidenceLevel.from_score(self.confidence.get("overall", 0.0))

    def to_dict(self) -> dict:
        """Serialize for SSE streaming to frontend."""
        return {
            "id": self.id,
            "surface": self.surface,
            "lemma": self.lemma,
            "sentence_text": self.sentence_text,
            "context": self.context,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "pos": self.pos,
            "pos_fine": self.pos_fine,
            "morphology": self.morphology,
            "syntactic_role": self.syntactic_role,
            "category": self.category,
            "translation_value": round(self.translation_value, 2),
            "extraction_reason": self.extraction_reason,
            "entity_type": self.entity_type,
            "candidate_senses": [
                {"definition": s.definition, "source": s.source,
                 "score": round(s.score, 2), "is_primary": s.is_primary}
                for s in self.candidate_senses
            ] if self.candidate_senses else [],
            "selected_sense": self.selected_sense,
            "translations": [
                {"text": t.text, "source": t.source, "score": round(t.score, 2)}
                for t in self.translations
            ] if self.translations else [],
            "selected_translation": self.selected_translation,
            "confidence": {k: round(v, 2) for k, v in self.confidence.items()},
            "confidence_level": self.confidence_level(),
            "evidence": self.evidence,
            "is_metadata": self.is_metadata,
        }


# ---------------------------------------------------------------------------
# Entity Span
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
    """
    A named entity detected in the source text.
    Multi-word entities are represented as a single span.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    surface: str = ""               # "James Paget Hospital"
    entity_type: str = ""           # "ORGANIZATION"
    entity_type_display: str = ""   # "Organization"
    entity_subtype: str = ""        # "hospital"

    # Position
    start_offset: int = 0
    end_offset: int = 0
    sentence_text: str = ""
    context: str = ""

    # Research
    description: str = ""
    location: Optional[str] = None
    source: str = ""                # "Wikipedia", "DuckDuckGo", etc.
    confidence: float = 0.0

    def confidence_level(self) -> str:
        return ConfidenceLevel.from_score(self.confidence)

    def to_dict(self) -> dict:
        """Serialize for SSE streaming."""
        return {
            "id": self.id,
            "name": self.surface,
            "label": self.entity_type,
            "label_display": self.entity_type_display,
            "entity_subtype": self.entity_subtype,
            "summary": self.description,
            "location": self.location,
            "source": self.source,
            "confidence": round(self.confidence, 2),
            "confidence_level": self.confidence_level(),
        }
