"""
Spy AI — Translation Module

Context-aware translation candidate generation, candidate validation,
and semantic isolation.

Key architecture:
1. Candidates are per-occurrence (independent evaluation).
2. Clean candidate validation layer:
   - Rejects identical or normalized English source words.
   - Rejects English source tokens appearing as Turkish alternatives.
   - Rejects un-translated English content words copied from sentence.
   - Rejects meta-explanations.
3. Marker-drift detection for contextual translation without extraneous network calls.
4. Quality over quantity — displays 1-3 genuine alternatives, never pads.
"""

import re
import json
import logging
from typing import List, Optional, Callable, Set
from nltk.corpus import wordnet as wn

from models import Occurrence, TranslationCandidate

logger = logging.getLogger("SpyAI.Translator")

# Turkish-specific characters
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")

# Common Turkish meta-explanation words
EXPLANATION_MARKERS = {
    "kelimesi", "kelime", "anlamına", "terimi", "terim", "demektir",
    "olarak", "çevrilir", "çevirisi", "örneğin", "veya", "ya da"
}


def translate_occurrence(
    occurrence: Occurrence,
    translate_fn: Callable,
    gtx_client=None,
    gemini_client=None,
    gemini_available: bool = False,
    meaning_translate_fn: Optional[Callable] = None,
    neighbor_translations: Optional[Set[str]] = None,
) -> Occurrence:
    """
    Generate contextually ranked, validated Turkish translation candidates for an occurrence.

    Guarantees:
    - Occurrence-level isolation (no shared mutable state).
    - Source term cannot appear as a Turkish translation.
    - Quality over quantity: only valid Turkish alternatives returned.
    """
    word = occurrence.lemma or occurrence.surface.lower()
    surface = occurrence.surface
    sentence = occurrence.sentence_text or ""
    raw_candidates: List[TranslationCandidate] = []

    # 1. Gemini contextual translations (if available)
    if gemini_available and gemini_client:
        gemini_trans = _gemini_translations(word, sentence, occurrence.pos, gemini_client)
        raw_candidates.extend(gemini_trans)

    # 2. GTX bilingual dictionary synonyms (primary source for dictionary-quality equivalents)
    if gtx_client:
        gtx_trans = _gtx_translations(word, gtx_client)
        raw_candidates.extend(gtx_trans)
        if surface.lower() != word.lower():
            surf_gtx = _gtx_translations(surface.lower(), gtx_client)
            raw_candidates.extend(surf_gtx)

    # 3. Isolated translation fallback (ensures at least 1 reliable candidate and populates isolated_texts)
    isolated_texts: Set[str] = set(_normalize_for_comparison(c.text) for c in raw_candidates)
    if translate_fn and not isolated_texts:
        try:
            lookup = surface if len(surface.split()) > 1 else word
            fallback = translate_fn(lookup).strip()
            if fallback and _normalize_for_comparison(fallback) != _normalize_for_comparison(lookup):
                fb_cand = TranslationCandidate(
                    text=fallback.lower(),
                    source="Google Translate",
                    score=0.6,
                )
                raw_candidates.append(fb_cand)
                isolated_texts.add(_normalize_for_comparison(fb_cand.text))
        except Exception:
            pass

    # 4. Contextual sentence translation (marker trick guarded by isolated_texts)
    if sentence and translate_fn:
        ctx_trans = _contextual_translation(
            word=word,
            surface=surface,
            sentence=sentence,
            translate_fn=translate_fn,
            isolated_texts=isolated_texts,
        )
        if ctx_trans:
            raw_candidates.append(ctx_trans)

    # Split compound/slash entries (e.g. "çoban / herder" -> ["çoban", "herder"])
    expanded_candidates = _expand_candidate_entries(raw_candidates)

    # --- Strict Validation Layer ---
    valid_candidates = []
    for cand in expanded_candidates:
        if _is_valid_turkish_candidate(
            candidate_text=cand.text,
            source_surface=surface,
            source_lemma=word,
            sentence=sentence,
            isolated_texts=isolated_texts,
        ):
            valid_candidates.append(cand)

    # Deduplicate and rank
    deduped = _deduplicate_translations(valid_candidates)
    ranked = _rank_translations(deduped, sentence, occurrence.pos)

    # Quality over quantity: limit to top 3, no filler
    final = ranked[:3]
    if len(final) > 1 and final[0].score > 0.7:
        final = [c for c in final if c.score >= 0.35]

    # Update occurrence
    occurrence.translations = final
    if final:
        occurrence.selected_translation = final[0].text
        occurrence.confidence["translation"] = min(0.95, final[0].score + 0.1)
    else:
        # Absolute fallback: try translate_fn directly on word or surface
        if translate_fn:
            for fallback_query in [word, surface]:
                try:
                    solo = translate_fn(fallback_query).strip()
                    if solo and _is_valid_turkish_candidate(solo, surface, word, sentence):
                        c = TranslationCandidate(text=solo.lower(), source="Google Translate", score=0.5)
                        occurrence.translations = [c]
                        occurrence.selected_translation = c.text
                        occurrence.confidence["translation"] = 0.5
                        break
                except Exception:
                    pass

        if not occurrence.translations:
            occurrence.selected_translation = ""
            occurrence.confidence["translation"] = 0.0

    # Add evidence
    trans_sources = list(set(c.source for c in occurrence.translations if c.source))
    occurrence.evidence = list(set(occurrence.evidence + trans_sources))

    # Overall confidence
    term_conf = occurrence.confidence.get("term", 0.5)
    sense_conf = occurrence.confidence.get("sense", 0.5)
    trans_conf = occurrence.confidence.get("translation", 0.5)
    occurrence.confidence["overall"] = (term_conf + sense_conf + trans_conf) / 3.0

    return occurrence


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for equality checks: lowercase, strip punctuation & extra spaces."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\u00C0-\u017F]", "", t)
    return " ".join(t.split())


def _expand_candidate_entries(candidates: List[TranslationCandidate]) -> List[TranslationCandidate]:
    """Split composite strings like 'çoban / herder' or 'çoban, güdücü' into distinct candidates."""
    expanded = []
    for c in candidates:
        text = c.text
        # Clean parentheses e.g. "çoban (isim)" -> "çoban"
        text = re.sub(r'\(.*?\)', '', text).strip()

        # Check for slashes or commas
        if "/" in text or "," in text:
            parts = re.split(r'[/,]', text)
            for p in parts:
                p_clean = p.strip()
                if p_clean:
                    expanded.append(TranslationCandidate(
                        text=p_clean.lower(),
                        source=c.source,
                        score=c.score,
                    ))
        else:
            if text:
                expanded.append(TranslationCandidate(
                    text=text.lower(),
                    source=c.source,
                    score=c.score,
                ))
    return expanded


def _is_valid_turkish_candidate(
    candidate_text: str,
    source_surface: str,
    source_lemma: str,
    sentence: str = "",
    isolated_texts: Optional[Set[str]] = None,
) -> bool:
    """
    Semantic consistency & Turkish candidate validation layer.
    Rejects:
    1. Empty, too short, or non-alphabetic candidates.
    2. Exact matches with the English source surface or lemma (case-insensitive & normalized).
    3. Standalone English word unchanged as candidate.
    4. Unchanged English content words copied from the source sentence.
    5. Explanatory meta-phrases ("kelimesi", "anlamına gelir").
    """
    if not candidate_text:
        return False

    cand_norm = _normalize_for_comparison(candidate_text)
    if len(cand_norm) < 2:
        return False

    surf_norm = _normalize_for_comparison(source_surface)
    lemma_norm = _normalize_for_comparison(source_lemma)

    # 1. HARD RULE: Candidate must not be identical to source term or lemma
    if cand_norm == surf_norm or cand_norm == lemma_norm:
        return False

    # Also reject if single-word candidate matches any token of the source term
    surf_tokens = set(surf_norm.split())
    if cand_norm in surf_tokens:
        return False

    # 2. HARD RULE: Candidate must not contain the English source word as a whole word
    if surf_norm and len(surf_norm) >= 3 and re.search(r'\b' + re.escape(surf_norm) + r'\b', cand_norm):
        return False
    if lemma_norm and len(lemma_norm) >= 3 and re.search(r'\b' + re.escape(lemma_norm) + r'\b', cand_norm):
        return False

    # 3. Reject meta-explanations
    cand_words = set(cand_norm.split())
    if cand_words.intersection(EXPLANATION_MARKERS):
        return False

    # 4. Reject if candidate is an unchanged ENGLISH content word from the sentence
    # (e.g. "herder" or "hometown" appearing untranslated).
    # Only applies to words len >= 4 with no Turkish characters that exist in WordNet as English words.
    has_tr_char = any(ch in TURKISH_CHARS for ch in candidate_text)
    if not has_tr_char and sentence and len(cand_norm) >= 4:
        sentence_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
        if cand_norm in sentence_words:
            try:
                if wn.synsets(cand_norm):
                    return False
            except Exception:
                pass

    return True


def _gemini_translations(
    word: str, sentence: str, pos: str, gemini_client
) -> List[TranslationCandidate]:
    """Get contextual Turkish translations from Gemini."""
    try:
        prompt = (
            f"Provide 2-3 distinct, contextually accurate Turkish translations "
            f"for the English word/phrase '{word}' (POS: {pos}) as used in:\n"
            f'"{sentence}"\n\n'
            f"Rules:\n"
            f"1. Return ONLY a JSON array of strings.\n"
            f"2. Every item must be a valid Turkish translation matching this specific term.\n"
            f"3. Never return the English word itself or explanation phrases.\n"
            f"4. If only one translation is genuinely appropriate, return just one.\n"
            f"5. No explanations, no markdown."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"temperature": 0.1}
        )
        text_resp = response.text.strip()
        if text_resp.startswith("```"):
            text_resp = re.sub(r"^```[a-z]*\n?", "", text_resp)
            text_resp = re.sub(r"\n?```$", "", text_resp).strip()

        syns = json.loads(text_resp)
        if isinstance(syns, list):
            return [
                TranslationCandidate(
                    text=s.strip().lower(),
                    source="Gemini AI",
                    score=0.85 - (i * 0.1),
                )
                for i, s in enumerate(syns)
                if isinstance(s, str) and s.strip()
            ][:3]
    except Exception as e:
        logger.debug(f"Gemini translation failed for '{word}': {e}")

    return []


def _gtx_translations(word: str, gtx_client) -> List[TranslationCandidate]:
    """Get translations from Google Translate GTX bilingual dictionary."""
    candidates = []
    norm_word = _normalize_for_comparison(word)
    try:
        data = gtx_client.query_raw(word, sl="en", tl="tr")

        # Primary translation
        if data and data[0] and data[0][0] and data[0][0][0]:
            primary = data[0][0][0].strip().lower()
            if primary and _normalize_for_comparison(primary) != norm_word:
                candidates.append(TranslationCandidate(
                    text=primary,
                    source="Google Translate",
                    score=0.70,
                ))

        # Bilingual dictionary synonyms (dt=bd)
        if data and len(data) > 1 and data[1]:
            for entry in data[1]:
                if isinstance(entry, list) and len(entry) > 1:
                    for s in entry[1]:
                        if isinstance(s, str):
                            sc = s.strip().lower()
                            if sc and len(sc) > 1 and _normalize_for_comparison(sc) != norm_word:
                                candidates.append(TranslationCandidate(
                                    text=sc,
                                    source="Bilingual Dictionary",
                                    score=0.60,
                                Odd=False if not hasattr(TranslationCandidate, 'Odd') else None
                                ))
    except Exception as e:
        logger.debug(f"GTX translation failed for '{word}': {e}")

    # Remove any None-attribute artifact
    return [c for c in candidates if c.text]


def _contextual_translation(
    word: str,
    surface: str,
    sentence: str,
    translate_fn: Callable,
    isolated_texts: Optional[Set[str]] = None,
) -> Optional[TranslationCandidate]:
    """
    Translate the word in context by marking it in the sentence
    and extracting the marked portion from the translated sentence.

    Guarded against marker drift (brackets jumping to neighboring words or full sentence).
    """
    try:
        target = surface if surface in sentence else word
        marked = sentence.replace(target, f"[[{target}]]", 1)
        if "[[" not in marked:
            pattern = re.compile(re.escape(target), re.IGNORECASE)
            marked = pattern.sub(f"[[{target}]]", sentence, count=1)

        if "[[" not in marked:
            return None

        translated = translate_fn(marked)
        match = re.search(r"\[\[(.*?)\]\]", translated)
        if match:
            result = match.group(1).strip().lower()
            result_norm = _normalize_for_comparison(result)

            # Check if result is empty or unchanged English word
            if not result_norm or result_norm in (_normalize_for_comparison(word), _normalize_for_comparison(surface)):
                return None

            # Reject if marker captured an entire sentence (> 4 words)
            if len(result_norm.split()) > 4:
                return None

            # Guard against marker drift: if isolated translation exists and result shares NO stem with it,
            # verify whether the bracket drifted to another word in the sentence
            if isolated_texts:
                has_stem_match = any(
                    iso in result_norm or result_norm in iso
                    for iso in isolated_texts if len(iso) >= 2
                )
                if not has_stem_match:
                    other_words = [
                        w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', sentence)
                        if w.lower() not in (word.lower(), surface.lower())
                    ]
                    for ow in other_words[:4]:
                        try:
                            ow_tr = _normalize_for_comparison(translate_fn(ow))
                            if ow_tr and (ow_tr in result_norm or result_norm in ow_tr):
                                logger.info(f"Marker drift detected: '{result}' belongs to '{ow}', not '{word}'")
                                return None
                        except Exception:
                            pass

            return TranslationCandidate(
                text=result,
                source="Contextual Translation",
                score=0.75,
            )
    except Exception as e:
        logger.debug(f"Contextual translation failed for '{word}': {e}")

    return None


def _deduplicate_translations(candidates: List[TranslationCandidate]) -> List[TranslationCandidate]:
    """Remove duplicate translations based on normalized text, keeping the highest-scored version."""
    seen = {}
    for c in candidates:
        key = _normalize_for_comparison(c.text)
        if not key:
            continue
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    return list(seen.values())


def _rank_translations(
    candidates: List[TranslationCandidate],
    sentence: str,
    pos: str
) -> List[TranslationCandidate]:
    """
    Rank translations by quality and contextual relevance.
    Gemini > Contextual > GTX primary > GTX dictionary
    """
    source_boost = {
        "Gemini AI": 0.15,
        "Contextual Translation": 0.10,
        "Google Translate": 0.05,
        "Bilingual Dictionary": 0.0,
    }

    for c in candidates:
        c.score += source_boost.get(c.source, 0.0)

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    return candidates


def translate_senses_to_turkish(
    occurrence: Occurrence,
    translate_fn: Callable,
) -> list:
    """
    Translate the English sense definitions to Turkish for the TR tab.
    Returns a list of dicts matching the candidate_senses structure.
    """
    translated = []
    for sense in occurrence.candidate_senses:
        try:
            tr_def = translate_fn(sense.definition)
        except Exception:
            tr_def = sense.definition

        translated.append({
            "definition": tr_def if tr_def else sense.definition,
            "source": sense.source,
            "is_primary": sense.is_primary,
        })

    return translated
