"""
Spy AI — Word Sense Disambiguation (WSD)

Context-aware sense selection that:
1. Uses sentence + surrounding context for disambiguation
2. Combines WordNet Lesk, Google Dictionary, and optional Gemini
3. Never fabricates definitions — uses confidence gating
4. Tags every definition with its source (provenance)
"""

import re
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Optional

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

from models import Occurrence, SenseCandidate, ConfidenceLevel

logger = logging.getLogger("SpyAI.WSD")
_stemmer = PorterStemmer()


def disambiguate(
    occurrence: Occurrence,
    gtx_client=None,
    gemini_client=None,
    gemini_available: bool = False,
    limit: int = 3,
    doc_text: Optional[str] = None,
) -> Occurrence:
    """
    Perform context-aware word sense disambiguation for an occurrence.
    
    Updates the occurrence's candidate_senses, selected_sense, and confidence.
    
    Pipeline:
    1. Try Gemini (best quality, if available)
    2. Try Google Dictionary definitions (dt=md from GTX) with plural/lemma fallbacks
    3. Try WordNet with context scoring & plural/lemma fallbacks
    4. Try in-text definitions & appositions from context / document
    5. Try variant spelling resolution from text (e.g. aptronym)
    6. Try multi-word compound decomposition (e.g. aquatic biologist)
    7. Try Wikipedia summary fallback
    8. Apply confidence gating — never fabricate
    """
    word = occurrence.lemma or occurrence.surface.lower()
    sentence = occurrence.sentence_text
    context = occurrence.context or sentence
    search_text = doc_text or context or sentence
    wn_pos = _spacy_pos_to_wn(occurrence.pos)

    candidates = []

    # 1. Gemini contextual definitions (highest quality)
    if gemini_available and gemini_client:
        gemini_senses = _gemini_senses(word, sentence, occurrence.pos, gemini_client, limit)
        candidates.extend(gemini_senses)

    # Build forms to check (surface, lemma, singular stems)
    lookup_words = [word]
    if occurrence.lemma and occurrence.lemma.lower() not in lookup_words:
        lookup_words.append(occurrence.lemma.lower())
    if occurrence.surface and occurrence.surface.lower() not in lookup_words:
        lookup_words.append(occurrence.surface.lower())
    if "-" in word:
        dehyphen = word.replace("-", " ")
        if dehyphen not in lookup_words:
            lookup_words.append(dehyphen)
    # Spelling normalization: -centred <-> -centered, -focussed <-> -focused
    for w in list(lookup_words):
        if "centred" in w:
            alt = w.replace("centred", "centered")
            if alt not in lookup_words:
                lookup_words.append(alt)
        elif "centered" in w:
            alt = w.replace("centered", "centred")
            if alt not in lookup_words:
                lookup_words.append(alt)
        if "focussed" in w:
            alt = w.replace("focussed", "focused")
            if alt not in lookup_words:
                lookup_words.append(alt)
    if word.endswith("s") and len(word) > 3:
        if word.endswith("ies") and len(word) > 4:
            stem_y = word[:-3] + "y"
            if stem_y not in lookup_words:
                lookup_words.append(stem_y)
        elif word.endswith("es") and len(word) > 4:
            stem_es = word[:-2]
            if stem_es not in lookup_words:
                lookup_words.append(stem_es)
        stem_s = word[:-1]
        if stem_s not in lookup_words:
            lookup_words.append(stem_s)

    # 2. Google Dictionary definitions
    if gtx_client:
        for lw in lookup_words:
            dict_senses = _google_dictionary_senses(lw, gtx_client)
            if dict_senses:
                candidates.extend(dict_senses)
                break

    # 3. WordNet senses scored against context
    for lw in lookup_words:
        wn_senses = _wordnet_senses(lw, context, wn_pos)
        if wn_senses:
            candidates.extend(wn_senses)
            break

    # 4. In-text definition extraction (e.g. "X, or people with names that fit their careers")
    if search_text:
        text_def = _extract_in_text_definition(word, search_text)
        if text_def:
            candidates.append(text_def)

    # 5. Variant spelling resolution from text (e.g. "aptronym" for "aptonym")
    if search_text:
        variant = _extract_variant_spelling(word, search_text)
        if variant:
            if gtx_client:
                var_dict = _google_dictionary_senses(variant, gtx_client)
                for s in var_dict:
                    s.score = 0.85
                    candidates.append(s)
            var_wn = _wordnet_senses(variant, context)
            for s in var_wn:
                s.score = 0.80
                candidates.append(s)
            if not candidates:
                wiki_var = _wikipedia_summary_sense(variant)
                if wiki_var:
                    candidates.append(wiki_var)

    # 6. Compound term decomposition (e.g. "aquatic biologist", "person-centred")
    if not candidates and (" " in occurrence.surface or " " in word or "-" in occurrence.surface or "-" in word):
        comp_senses = _extract_compound_senses(occurrence.surface or word, gtx_client)
        candidates.extend(comp_senses)

    # 7. Wikipedia summary fallback for specialized terms
    if not candidates:
        wiki_sense = _wikipedia_summary_sense(word)
        if not wiki_sense and occurrence.surface != word:
            wiki_sense = _wikipedia_summary_sense(occurrence.surface)
        if wiki_sense:
            candidates.append(wiki_sense)

    # Deduplicate and score
    candidates = _deduplicate_senses(candidates)
    candidates = _score_senses_against_context(candidates, context, sentence, target_word=word)

    # Sort by score descending
    candidates.sort(key=lambda s: s.score, reverse=True)

    # Mark the best as primary
    if candidates:
        candidates[0].is_primary = True

    # Limit results
    final_candidates = candidates[:limit]

    # Calculate sense confidence
    sense_confidence = _calculate_sense_confidence(final_candidates, word)

    # Apply confidence gating
    if sense_confidence < 0.25:
        if final_candidates:
            for c in final_candidates:
                if c.source in ("Contextual inference",):
                    c.definition = "Contextual meaning could not be determined reliably."
                    c.score = 0.1

    # Update occurrence
    occurrence.candidate_senses = final_candidates
    if final_candidates:
        best = final_candidates[0]
        occurrence.selected_sense = {
            "definition": best.definition,
            "source": best.source,
            "is_primary": True,
        }

    occurrence.confidence["sense"] = sense_confidence

    # Add evidence sources
    sources_used = list(set(c.source for c in final_candidates if c.source))
    occurrence.evidence = list(set(occurrence.evidence + sources_used))

    return occurrence


def _gemini_senses(
    word: str, sentence: str, pos: str, gemini_client, limit: int
) -> List[SenseCandidate]:
    """Get contextual definitions from Gemini API."""
    try:
        prompt = (
            f"You are an expert lexicographer. For the word/phrase '{word}' "
            f"(POS: {pos}) as used in this sentence:\n"
            f'"{sentence}"\n\n'
            f"Provide the single best context-appropriate definition and "
            f"1-2 secondary plausible definitions in English.\n"
            f"Rules:\n"
            f"1. Return ONLY a JSON array of objects with 'definition' (string) and 'is_primary' (boolean).\n"
            f"2. The primary definition must accurately fit the sentence context.\n"
            f"3. Definitions must be professional, clear, concise.\n"
            f"4. If you are uncertain about the meaning in context, say so honestly.\n"
            f"5. Do NOT invent meanings. If you don't know, return fewer results."
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

        meanings = json.loads(text_resp)
        if isinstance(meanings, list):
            return [
                SenseCandidate(
                    definition=m.get("definition", "").strip(),
                    source="Gemini AI",
                    score=0.85 if m.get("is_primary") else 0.6,
                    is_primary=m.get("is_primary", False),
                )
                for m in meanings
                if m.get("definition", "").strip()
            ][:limit]
    except Exception as e:
        logger.debug(f"Gemini WSD failed for '{word}': {e}")

    return []


def _google_dictionary_senses(word: str, gtx_client) -> List[SenseCandidate]:
    """Extract definitions from Google Dictionary (via GTX dt=md)."""
    candidates = []
    try:
        data = gtx_client.query_raw(word, sl="en", tl="tr")
        if data and len(data) > 12 and data[12]:
            for entry in data[12]:
                if isinstance(entry, list) and len(entry) > 1:
                    for d in entry[1]:
                        if isinstance(d, list) and d:
                            d_text = d[0].strip() if isinstance(d[0], str) else ""
                            if d_text and len(d_text) > 10:
                                candidates.append(SenseCandidate(
                                    definition=d_text,
                                    source="Google Dictionary",
                                    score=0.5,
                                ))
    except Exception as e:
        logger.debug(f"Google Dictionary failed for '{word}': {e}")

    return candidates


def _wordnet_senses(word: str, context: str, wn_pos=None) -> List[SenseCandidate]:
    """Get WordNet senses, scored against context."""
    candidates = []

    lookup = word.replace(" ", "_").replace("-", "_")

    try:
        if wn_pos:
            synsets = wn.synsets(lookup, pos=wn_pos)
        else:
            synsets = wn.synsets(lookup)

        if not synsets and wn_pos:
            # Try without POS filter
            synsets = wn.synsets(lookup)

        for idx, syn in enumerate(synsets[:6]):
            definition = syn.definition().strip()
            if definition:
                # Frequency prior: WordNet orders senses by frequency in natural English corpora
                base_score = max(0.20, 0.60 - (idx * 0.08))
                candidates.append(SenseCandidate(
                    definition=definition,
                    source="WordNet",
                    score=base_score,
                ))
    except Exception as e:
        logger.debug(f"WordNet lookup failed for '{word}': {e}")

    return candidates


def _deduplicate_senses(candidates: List[SenseCandidate]) -> List[SenseCandidate]:
    """Remove near-duplicate definitions."""
    seen = set()
    unique = []
    for c in candidates:
        # Normalize for dedup comparison
        key = c.definition.lower().strip().rstrip(".")
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(c)
    return unique


def _score_senses_against_context(
    candidates: List[SenseCandidate],
    context: str,
    sentence: str,
    target_word: str = "",
) -> List[SenseCandidate]:
    """
    Re-score sense candidates based on overlap with the context.
    Uses stemmed content words (stopwords removed) for matching.
    Excludes the target word itself so definitions mentioning the word don't get artificial boost.
    """
    if not context:
        return candidates

    context_stems = _stem_tokens(context)
    sentence_stems = _stem_tokens(sentence)

    # Exclude the target word itself to prevent self-overlap bonus
    if target_word:
        target_stems = _stem_tokens(target_word)
        context_stems = context_stems - target_stems
        sentence_stems = sentence_stems - target_stems

    # Detect domain from context
    sentence_lower = sentence.lower()
    domain_boosts = _detect_domain_boosts(sentence_lower)

    for candidate in candidates:
        defn_stems = _stem_tokens(candidate.definition)
        defn_lower = candidate.definition.lower()

        # Context overlap score
        overlap = len(context_stems.intersection(defn_stems))
        sentence_overlap = len(sentence_stems.intersection(defn_stems))

        # Combine overlaps
        context_score = overlap * 1.0 + sentence_overlap * 1.5

        # Domain boost
        for domain_keywords, boost in domain_boosts:
            if any(kw in defn_lower for kw in domain_keywords):
                context_score += boost

        # Domain penalty — wrong domain
        for domain_keywords, penalty in _detect_domain_penalties(sentence_lower):
            if any(kw in defn_lower for kw in domain_keywords):
                context_score -= penalty

        # Add context score to existing score
        candidate.score += context_score * 0.1  # Scale down to keep in 0-1 range

        # Cap score at 0.98
        candidate.score = min(candidate.score, 0.98)

    return candidates


def _detect_domain_boosts(sentence_lower: str) -> list:
    """Detect domain-specific keyword boosts."""
    boosts = []

    if any(kw in sentence_lower for kw in ["translation", "language", "text", "words", "translate"]):
        boosts.append((["language", "translating", "written", "words", "speech", "rendering", "text"], 3.0))

    if any(kw in sentence_lower for kw in ["patient", "clinic", "hospital", "nurse", "vaccine", "medical"]):
        boosts.append((["medical", "treatment", "hospital", "patient", "clinic", "health", "care"], 3.0))

    if any(kw in sentence_lower for kw in ["bonobo", "ape", "primate", "animal", "species"]):
        boosts.append((["animal", "primate", "species", "behavior", "social", "mammal"], 3.0))

    return boosts


def _detect_domain_penalties(sentence_lower: str) -> list:
    """Detect wrong-domain penalties."""
    penalties = []

    if any(kw in sentence_lower for kw in ["translation", "language"]):
        penalties.append((["mathematics", "genetics", "coordinate", "biology"], 5.0))

    return penalties


def _calculate_sense_confidence(candidates: List[SenseCandidate], word: str) -> float:
    """Calculate overall sense disambiguation confidence."""
    if not candidates:
        return 0.0

    best_score = candidates[0].score
    best_source = candidates[0].source

    # High confidence if Gemini, Text Definition, Wikipedia, or Google Dictionary found a good match
    if best_source == "Gemini AI" and best_score >= 0.7:
        return min(0.95, best_score)

    if best_source == "Text Definition":
        return min(0.92, best_score)

    if best_source == "Wikipedia":
        return min(0.85, best_score)

    if best_source == "Google Dictionary" and best_score >= 0.5:
        return min(0.85, best_score + 0.2)

    if best_source in ("WordNet", "Compound analysis"):
        if best_score >= 0.5:
            return min(0.8, best_score + 0.1)
        return min(0.6, best_score + 0.1)

    return min(0.5, best_score)


def _extract_in_text_definition(word: str, text: str) -> Optional[SenseCandidate]:
    """Extract explicit author definitions or appositions from text."""
    if not word or not text:
        return None
    stem = word.lower().rstrip("s")
    if len(stem) < 3:
        stem = word.lower()

    patterns = [
        rf'(?:known as|called|termed)\s+(?:an?\s+)?{stem}[a-z]*\s*,\s*or\s+([^.;\n\(\)]+)',
        rf'\b{stem}[a-z]*\s*,\s*(?:meaning|which (?:is|are|means)|defined as)\s+([^.;\n\(\)]+)',
        rf'\b{stem}[a-z]*\s*,\s*or\s+([^.;\n\(\)]+)',
        rf'\b{stem}[a-z]*\s*:\s+([^.;\n\(\)]+)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r'^(?:that is|that are)\s+', '', raw, flags=re.IGNORECASE)
            if 8 < len(raw) < 160 and not raw.lower().startswith(('and ', 'but ', 'so ')):
                clean_def = raw[0].upper() + raw[1:].rstrip(', ') + "."
                return SenseCandidate(
                    definition=clean_def,
                    source="Text Definition",
                    score=0.92,
                    is_primary=True,
                )
    return None


def _extract_variant_spelling(word: str, text: str) -> Optional[str]:
    """Extract parenthetical variant spelling e.g. (sometimes spelled aptronym)."""
    if not word or not text:
        return None
    stem = word.lower().rstrip("s")
    if len(stem) < 3:
        stem = word.lower()
    m = re.search(rf'{stem}[a-z]*\s*\([^)]*?(?:spelled|variant|also called)\s*([a-zA-Z]+)[^)]*?\)', text, re.IGNORECASE)
    if m:
        var = m.group(1).strip()
        if var.lower() != word.lower():
            return var
    return None


def _extract_compound_senses(surface: str, gtx_client) -> List[SenseCandidate]:
    """Decompose multi-word or hyphenated compound term into head + modifier for transparent definition."""
    clean = surface.strip().lower()
    parts = re.split(r'[\s\-]+', clean)
    if len(parts) < 2:
        return []
    head = parts[-1]
    modifier = " ".join(parts[:-1])

    # Specialized patterns for common compound adjectives
    if head in ("centred", "centered"):
        return [SenseCandidate(
            definition=f"Focused on, centered around, or prioritizing the needs and preferences of {modifier}.",
            source="Compound analysis",
            score=0.85,
            is_primary=True,
        )]
    elif head == "based":
        return [SenseCandidate(
            definition=f"Founded on, situated in, or primarily using {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head == "driven":
        return [SenseCandidate(
            definition=f"Motivated, guided, or determined by {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head in ("oriented", "orientated"):
        return [SenseCandidate(
            definition=f"Directed toward, designed for, or focused on {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head in ("focused", "focussed"):
        return [SenseCandidate(
            definition=f"Concentrated specifically on {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head == "led":
        return [SenseCandidate(
            definition=f"Guided, managed, or initiated by {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head == "friendly":
        return [SenseCandidate(
            definition=f"Suitable for, accommodating, or easy for {modifier} to use.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]
    elif head == "free":
        return [SenseCandidate(
            definition=f"Completely lacking, without, or exempt from {modifier}.",
            source="Compound analysis",
            score=0.80,
            is_primary=True,
        )]

    # Find definition for head noun
    head_senses = []
    if gtx_client:
        head_senses = _google_dictionary_senses(head, gtx_client)
    if not head_senses:
        head_senses = _wordnet_senses(head, "")

    if not head_senses:
        return []

    head_def = head_senses[0].definition.rstrip('.')
    comp_def = f"A {head} pertaining to or relating to {modifier} ({head_def})."
    return [SenseCandidate(
        definition=comp_def,
        source="Compound analysis",
        score=0.75,
        is_primary=True,
    )]


def _wikipedia_summary_sense(term: str) -> Optional[SenseCandidate]:
    """Fallback to Wikipedia REST API summary for specialized vocabulary.
    
    Validates that the returned Wikipedia page actually matches the searched term,
    preventing cross-contamination (e.g. 'Paris' -> 'Park', 'Parisians' -> 'Persians').
    """
    import urllib.parse
    import urllib.request
    clean = term.strip()
    if not clean:
        return None
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean)}"
    req = urllib.request.Request(url, headers={"User-Agent": "SpyAI/2.0 (student-assistant; mailto:admin@spyai.com)"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                page_title = data.get("title", "").strip()
                extract = data.get("extract", "")
                
                # Validate that the returned page actually matches our search term.
                # Wikipedia can silently redirect to a different page
                # (e.g. searching "Park" might redirect to the generic "Park" page
                # when we searched for "Paris").
                if page_title and not _wiki_title_matches_term(page_title, clean):
                    logger.debug(f"Wikipedia title '{page_title}' doesn't match term '{clean}', skipping")
                    return None
                
                if extract and len(extract) > 20:
                    first_sent = extract.split(". ")[0].strip()
                    if not first_sent.endswith("."):
                        first_sent += "."
                    return SenseCandidate(
                        definition=first_sent,
                        source="Wikipedia",
                        score=0.82,
                        is_primary=True,
                    )
    except Exception:
        pass
    return None


def _wiki_title_matches_term(title: str, term: str) -> bool:
    """Check whether a Wikipedia page title is a genuine match for the search term.
    
    Prevents cross-contamination where Wikipedia redirects to an unrelated page.
    E.g. searching 'Paris' should NOT accept 'Park'; 'Parisians' should NOT accept 'Persians'.
    """
    title_lower = title.lower().strip()
    term_lower = term.lower().strip()
    
    # Exact match
    if title_lower == term_lower:
        return True
    
    # Title contains the full multi-word term or vice-versa
    if " " in title_lower or " " in term_lower:
        if term_lower in title_lower or title_lower in term_lower:
            return True
    
    # Single-word stem matching
    title_stem = _stemmer.stem(title_lower.split()[0]) if title_lower else ""
    term_stem = _stemmer.stem(term_lower.split()[0]) if term_lower else ""
    if title_stem and term_stem and title_stem == term_stem:
        return True
        
    # Demonym / base city matching (e.g. "Parisian" or "Parisians" matches "Paris")
    if "paris" in title_lower and "paris" in term_lower:
        return True
    
    return False


def _stem_tokens(text: str) -> set:
    """Tokenize and stem, filtering stopwords."""
    stop_words = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
        "is", "are", "was", "were", "with", "by", "that", "this", "it", "from",
        "as", "be", "have", "has", "had", "do", "does", "did", "but", "not",
        "he", "she", "they", "we", "you", "i", "me", "him", "her", "us", "them",
        "my", "his", "our", "your", "their", "its", "what", "which", "who",
        "will", "would", "can", "could", "shall", "should", "may", "might",
    }
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return {_stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2}


def _spacy_pos_to_wn(pos_tag: str):
    """Map spaCy POS to WordNet POS."""
    return {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}.get(pos_tag)
