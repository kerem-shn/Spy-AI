"""
Spy AI — Entity Resolver

Context-aware named entity recognition and linking.

Key improvements over the old system:
1. Multi-word entity span consolidation BEFORE external lookup
2. Context-aware Wikipedia disambiguation (uses entity type + context)
3. Never resolves entities from isolated tokens when a larger span exists
4. Validates entity descriptions against detected type
"""

import re
import logging
from typing import List, Set, Tuple, Optional

import wikipedia
from nltk.corpus import wordnet as wn

from models import EntitySpan, EntityType

logger = logging.getLogger("SpyAI.EntityResolver")

# ---------------------------------------------------------------------------
# Entity type display mapping
# ---------------------------------------------------------------------------

ENTITY_LABEL_DISPLAY = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "LOC": "Location",
    "EVENT": "Event",
    "WORK_OF_ART": "Work of Art",
    "NORP": "Group/Nationality",
    "FAC": "Facility",
    "PRODUCT": "Product",
    "DATE": "Date",
    "LAW": "Law/Treaty",
}

# Entity labels we care about
RELEVANT_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "NORP", "FAC"}

# Common abbreviations / acronyms that Wikipedia may not resolve directly
COMMON_ABBREVIATIONS = {
    "USA": "United States of America",
    "UK": "United Kingdom",
    "EU": "European Union",
    "UN": "United Nations",
    "WHO": "World Health Organization",
    "NHS": "National Health Service",
    "AAD": "American Academy of Dermatology",
    "AMA": "American Medical Association",
    "FDA": "Food and Drug Administration",
    "CDC": "Centers for Disease Control and Prevention",
    "NATO": "NATO",
    "NASA": "NASA",
    "MIT": "Massachusetts Institute of Technology",
    "UCLA": "University of California, Los Angeles",
    "BBC": "BBC",
    "CNN": "CNN",
    "IMF": "International Monetary Fund",
}


def detect_entities(
    doc,
    segments=None,
) -> Tuple[List[EntitySpan], Set[int]]:
    """
    Detect and consolidate named entities from a spaCy doc.

    Returns:
        - List of EntitySpan objects
        - Set of token indices belonging to entities (for term extraction exclusion)
    """
    entity_spans = []
    entity_token_indices = set()
    seen_surface = set()

    sentences = list(doc.sents)

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_ENTITY_LABELS:
            continue

        name = ent.text.strip()
        if len(name) < 2:
            continue

        # Validate entity
        if not _is_valid_entity(name, ent.label_):
            continue

        # Check if this entity falls in a metadata region
        if segments:
            from text_preprocessor import is_metadata_region
            if is_metadata_region(ent.start_char, segments):
                continue

        # Deduplicate by surface form
        if name in seen_surface:
            # Still mark token indices
            for i in range(ent.start, ent.end):
                entity_token_indices.add(i)
            continue
        seen_surface.add(name)

        # Mark token indices
        for i in range(ent.start, ent.end):
            entity_token_indices.add(i)

        # Build context
        sent_text = ""
        context = ""
        for i, sent in enumerate(sentences):
            if ent.start_char >= sent.start_char and ent.start_char < sent.end_char:
                sent_text = sent.text.strip()
                parts = []
                if i > 0:
                    parts.append(sentences[i - 1].text.strip())
                parts.append(sent_text)
                if i < len(sentences) - 1:
                    parts.append(sentences[i + 1].text.strip())
                context = " ".join(parts)
                break

        entity_span = EntitySpan(
            surface=name,
            entity_type=ent.label_,
            entity_type_display=ENTITY_LABEL_DISPLAY.get(ent.label_, ent.label_),
            start_offset=ent.start_char,
            end_offset=ent.end_char,
            sentence_text=sent_text,
            context=context,
        )

        # Infer subtype from context
        entity_span.entity_subtype = _infer_subtype(name, ent.label_, context)

        entity_spans.append(entity_span)

    # Propagate detected entity names to any other un-annotated occurrences in the document
    entity_names = {s.surface.lower() for s in entity_spans}
    for token in doc:
        if token.text.lower() in entity_names:
            entity_token_indices.add(token.i)

    return entity_spans, entity_token_indices


def _is_valid_entity(name: str, label: str) -> bool:
    """
    Filter NER false positives while keeping legitimate entities.
    """
    name_stripped = name.strip()

    if len(name_stripped) < 2:
        return False
    if name_stripped.isdigit():
        return False

    # PERSON and GPE are usually reliable
    if label in ("PERSON", "GPE"):
        return True

    # NORP — keep if capitalized
    if label == "NORP":
        return name_stripped[0].isupper()

    words = name_stripped.split()

    # Single lowercase word → likely false positive for ORG/EVENT
    if len(words) == 1 and name_stripped[0].islower():
        return False

    # Check for medical terms mislabeled as entities
    if len(words) == 1:
        synsets = wn.synsets(name_stripped.lower())
        if synsets:
            for s in synsets:
                defn = s.definition().lower()
                medical_kws = ["disease", "condition", "disorder", "inflammation",
                               "infection", "syndrome", "tissue", "rash"]
                if any(kw in defn for kw in medical_kws):
                    return False

    # All-lowercase multi-word → reject
    if all(w[0].islower() for w in words if w):
        return False

    return True


def _infer_subtype(name: str, label: str, context: str) -> str:
    """Infer a more specific subtype from context."""
    name_lower = name.lower()
    context_lower = context.lower()

    if label == "ORG":
        if any(kw in name_lower for kw in ["hospital", "clinic", "medical"]):
            return "hospital"
        if any(kw in name_lower for kw in ["university", "college", "school", "institute"]):
            return "educational_institution"
        if any(kw in name_lower for kw in ["journal", "reports", "review", "magazine"]):
            return "publication"
        if any(kw in context_lower for kw in ["hospital", "clinic", "medical", "patients"]):
            return "healthcare"
        if any(kw in context_lower for kw in ["university", "research", "study", "professor"]):
            return "research_institution"
        return "organization"

    if label == "GPE":
        if any(kw in context_lower for kw in ["republic", "country", "nation"]):
            return "country"
        if any(kw in context_lower for kw in ["city", "town", "village"]):
            return "city"
        return "place"

    if label == "PERSON":
        if any(kw in context_lower for kw in ["author", "wrote", "writer"]):
            return "author"
        if any(kw in context_lower for kw in ["researcher", "professor", "scientist", "study"]):
            return "researcher"
        if any(kw in context_lower for kw in ["nurse", "doctor", "dr."]):
            return "healthcare_professional"
        return "person"

    return ""


def research_entity(
    entity: EntitySpan,
    cache=None,
    ddgs_available: bool = False,
    ddgs_cls=None,
) -> EntitySpan:
    """
    Research a named entity using context-aware disambiguation.

    Key difference from old system: uses entity type + context to
    disambiguate Wikipedia results.
    """
    name = entity.surface

    # 0. Check cache
    if cache:
        cached = cache.get("entity_v2", name)
        if cached:
            entity.description = cached.get("summary", "")
            entity.source = cached.get("source", "")
            entity.location = cached.get("location")
            entity.confidence = cached.get("confidence", 0.7)
            return entity

    summary = ""
    source = ""
    location = None
    confidence = 0.0

    # 1. Context-aware Wikipedia search
    summary, source, confidence = _wikipedia_search_contextual(
        name, entity.entity_type, entity.entity_subtype, entity.context
    )

    # 2. DuckDuckGo fallback with context
    if not summary and ddgs_available and ddgs_cls:
        summary, source, confidence = _ddgs_search_contextual(
            name, entity.entity_type, entity.entity_subtype, entity.context, ddgs_cls
        )

    # 3. Final fallback
    if not summary:
        summary = "No information available."
        source = "N/A"
        confidence = 0.0

    # Extract location if applicable
    if entity.entity_type in ("ORG", "GPE", "FAC"):
        location = _extract_location_from_summary(summary, entity.context)

    entity.description = summary
    entity.source = source
    entity.location = location
    entity.confidence = confidence

    # Cache result
    if cache:
        cache.set("entity_v2", name, {
            "summary": summary,
            "source": source,
            "location": location,
            "confidence": confidence,
        })

    return entity


def _wikipedia_search_contextual(
    name: str, entity_type: str, subtype: str, context: str
) -> Tuple[str, str, float]:
    """
    Search Wikipedia with context-aware disambiguation.
    Tries expanded abbreviations and alternative queries.
    """
    # For short names / acronyms, try the expanded form first
    search_names = [name]
    name_upper = name.strip().upper()
    if name_upper in COMMON_ABBREVIATIONS:
        expanded = COMMON_ABBREVIATIONS[name_upper]
        if expanded != name:
            search_names.insert(0, expanded)  # Try expanded form first

    for search_name in search_names:
        try:
            wikipedia.set_lang("en")
            summary = wikipedia.summary(search_name, sentences=3)

            # Validate: does the summary match the entity type?
            if _validate_entity_summary(summary, entity_type, subtype, context):
                return summary, "Wikipedia", 0.9
            else:
                # Summary doesn't match — try to find a better one
                logger.info(f"Wikipedia summary for '{search_name}' doesn't match type {entity_type}, trying alternatives")
                better = _try_wikipedia_alternatives(search_name, entity_type, subtype, context)
                if better:
                    return better, "Wikipedia", 0.8
                # Fall back to the original even if it's not perfect
                return summary, "Wikipedia", 0.5

        except wikipedia.exceptions.DisambiguationError as e:
            # Context-aware disambiguation: pick the option that matches our entity type
            if e.options:
                best_option = _disambiguate_wikipedia(e.options, search_name, entity_type, subtype, context)
                if best_option:
                    try:
                        summary = wikipedia.summary(best_option, sentences=3)
                        return summary, "Wikipedia", 0.85
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Wikipedia lookup failed for '{search_name}': {e}")

    # If all search names failed, try with entity type hint appended
    if entity_type == "PERSON":
        for hint in ["politician", "president", "person"]:
            try:
                results = wikipedia.search(f"{name} {hint}", results=3)
                for r in results:
                    if name.split()[0].lower() in r.lower() or name.split()[-1].lower() in r.lower():
                        try:
                            summary = wikipedia.summary(r, sentences=3)
                            if _validate_entity_summary(summary, entity_type, subtype, context):
                                return summary, "Wikipedia", 0.75
                        except Exception:
                            continue
            except Exception:
                continue

    return "", "", 0.0


def _disambiguate_wikipedia(
    options: list, name: str, entity_type: str, subtype: str, context: str
) -> Optional[str]:
    """
    Pick the best Wikipedia disambiguation option based on entity type and context.

    For "James Paget Hospital" with options like ["James (band)", "James Paget Hospital", ...]:
    → Pick "James Paget Hospital" because it contains the full entity name
    → Reject "James (band)" because type is ORGANIZATION/hospital
    """
    name_lower = name.lower()
    context_lower = context.lower()

    # Type-specific keywords to look for in option titles
    type_keywords = {
        "ORG": ["hospital", "university", "company", "organization", "institution",
                "foundation", "association", "corporation", "clinic"],
        "PERSON": ["politician", "actor", "author", "scientist", "researcher"],
        "GPE": ["city", "town", "country", "province", "state", "district"],
    }

    subtype_keywords = {
        "hospital": ["hospital", "medical", "health"],
        "educational_institution": ["university", "college", "school", "institute"],
        "publication": ["journal", "magazine", "newspaper"],
        "research_institution": ["university", "research", "institute", "laboratory"],
    }

    scored_options = []
    for opt in options:
        score = 0
        opt_lower = opt.lower()

        # Exact or near-exact match with full name
        if name_lower in opt_lower:
            score += 10

        # Contains entity type keywords
        keywords = type_keywords.get(entity_type, [])
        for kw in keywords:
            if kw in opt_lower:
                score += 3

        # Contains subtype keywords
        sub_keywords = subtype_keywords.get(subtype, [])
        for kw in sub_keywords:
            if kw in opt_lower:
                score += 5

        # Penalize clearly wrong types
        wrong_type_indicators = {
            "ORG": ["(band)", "(singer)", "(album)", "(song)", "(film)", "(novel)"],
            "PERSON": ["(company)", "(organization)", "(city)", "(country)"],
            "GPE": ["(band)", "(company)", "(person)"],
        }
        for wrong in wrong_type_indicators.get(entity_type, []):
            if wrong in opt_lower:
                score -= 8

        scored_options.append((score, opt))

    scored_options.sort(key=lambda x: x[0], reverse=True)

    if scored_options and scored_options[0][0] > 0:
        return scored_options[0][1]

    # Default: try the first option that contains the full name
    for opt in options:
        if name_lower in opt.lower():
            return opt

    return options[0] if options else None


def _try_wikipedia_alternatives(
    name: str, entity_type: str, subtype: str, context: str
) -> Optional[str]:
    """Try alternative Wikipedia search queries when the direct search gives wrong results."""
    # Try with type qualifier
    type_qualifiers = {
        "ORG": ["organization", "institution", "company"],
        "PERSON": ["person"],
        "GPE": ["place", "location", "city", "country"],
    }

    qualifiers = type_qualifiers.get(entity_type, [])
    if subtype:
        qualifiers.insert(0, subtype)

    for qual in qualifiers:
        try:
            query = f"{name} ({qual})"
            results = wikipedia.search(query, results=3)
            for result in results:
                if name.lower() in result.lower():
                    try:
                        summary = wikipedia.summary(result, sentences=3)
                        if _validate_entity_summary(summary, entity_type, subtype, context):
                            return summary
                    except Exception:
                        continue
        except Exception:
            continue

    return None


def _validate_entity_summary(
    summary: str, entity_type: str, subtype: str, context: str
) -> bool:
    """
    Validate that a Wikipedia summary matches the expected entity type.

    For "James Paget Hospital" (ORG/hospital):
    - Summary about a hospital in Norfolk → VALID
    - Summary about the band James → INVALID
    """
    if not summary:
        return False

    summary_lower = summary.lower()

    # Type-specific validation
    type_validators = {
        "ORG": {
            "positive": ["organization", "company", "institution", "hospital",
                         "university", "foundation", "corporation", "charity",
                         "school", "college", "journal", "publication", "clinic",
                         "academy", "association", "society", "agency", "department"],
            "negative": ["singer", "band", "album", "song", "film", "actor",
                         "player", "athlete"],
        },
        "PERSON": {
            "positive": ["born", "is a", "was a", "politician", "author",
                         "scientist", "researcher", "nurse", "doctor",
                         "president", "minister", "leader", "prime minister",
                         "statesman", "served as", "elected"],
            "negative": ["company", "organization", "city", "country"],
        },
        "GPE": {
            "positive": ["city", "town", "village", "country", "state", "province",
                         "district", "municipality", "located", "population",
                         "county", "borough", "parish", "republic", "nation"],
            "negative": ["band", "singer", "album", "company"],
        },
    }

    validator = type_validators.get(entity_type)
    if not validator:
        return True  # No validator → accept

    # Check for negative indicators (strong rejection)
    negative_count = sum(1 for kw in validator["negative"] if kw in summary_lower)
    positive_count = sum(1 for kw in validator["positive"] if kw in summary_lower)

    # Subtype-specific validation
    if subtype:
        if subtype in summary_lower:
            positive_count += 3  # Strong positive signal

    # If more negative than positive, reject
    if negative_count > positive_count:
        return False

    return True


def _ddgs_search_contextual(
    name: str, entity_type: str, subtype: str, context: str, ddgs_cls
) -> Tuple[str, str, float]:
    """
    DuckDuckGo search with context-aware query construction.
    """
    try:
        # Build a type-qualified search query
        type_hint = subtype or entity_type.lower()
        query = f'"{name}" {type_hint}'

        with ddgs_cls() as ddgs:
            results = list(ddgs.text(query, region='en-us', max_results=3))

        if results:
            # Filter results: prefer ones that contain the full name
            name_lower = name.lower()
            relevant_snippets = []
            for r in results:
                body = r.get("body", "")
                if body and name_lower in body.lower():
                    relevant_snippets.append(body)

            if not relevant_snippets:
                relevant_snippets = [r.get("body", "") for r in results if r.get("body")]

            if relevant_snippets:
                summary = " ".join(relevant_snippets[:2])
                return summary, "Web Search", 0.6

    except Exception as e:
        logger.debug(f"DuckDuckGo search failed for '{name}': {e}")

    return "", "", 0.0


def _extract_location_from_summary(summary: str, context: str) -> Optional[str]:
    """Try to extract a location from an entity summary."""
    if not summary:
        return None

    # Common patterns: "located in X", "in X, England", "based in X"
    patterns = [
        r'(?:located|based|situated)\s+in\s+([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)',
        r'in\s+([A-Z][a-z]+(?:shire|land|ton|burgh|pool|ham|minster))',
        r'([A-Z][a-z]+,\s+(?:England|Scotland|Wales|Ireland|UK|USA|US))',
    ]
    for pat in patterns:
        m = re.search(pat, summary)
        if m:
            return m.group(1)

    return None
