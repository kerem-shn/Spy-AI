"""
Spy AI — Term Extractor

Context-aware and difficulty-aware term extraction system for B2 translation students.

Key architecture:
1. Calculates Translation Value Score (0.0 to 1.0) rather than relying on grammatical structure alone.
2. Distinguishes genuine translation units from ordinary compositional phrases (e.g. 'amusing world' -> ORDINARY_PHRASE, low score).
3. Student Mode (default, high precision >= 0.40) vs Research Mode (>= 0.15).
4. Categorizes terms into:
   - SPECIALIZED_TERM
   - USEFUL_LEXICAL_ITEM
   - COLLOCATION
   - IDIOM
   - MULTIWORD_TERM
   - ORDINARY_PHRASE
   - GENERAL_WORD
5. Occurrence-level isolation with unique IDs, offsets, and justification reasons.
6. Ranks results by Translation Value Score descending.
"""

import re
from typing import List, Set, Tuple
from nltk.corpus import wordnet as wn
from models import Occurrence, Category


# ---------------------------------------------------------------------------
# Analysis Modes
# ---------------------------------------------------------------------------

class AnalysisMode:
    STUDENT = "student"     # Default: high precision (score >= 0.40), suppresses ordinary phrases
    RESEARCH = "research"   # Detailed: lower threshold (score >= 0.15), includes ordinary lexical items
    STANDARD = "student"    # Backwards compatibility alias
    DETAILED = "research"   # Backwards compatibility alias
    RAW = "raw"             # Full token/POS information


# ---------------------------------------------------------------------------
# Common words & domain markers
# ---------------------------------------------------------------------------

TRIVIALLY_COMMON = {
    "room", "time", "way", "day", "thing", "part", "place", "case",
    "point", "fact", "group", "number", "problem", "state", "hand",
    "area", "work", "side", "home", "water", "money", "story", "body",
    "world", "life", "head", "food", "door", "line", "face", "idea",
    "name", "word", "house", "girl", "book", "game", "city", "back",
    "year", "week", "month", "question", "answer", "table", "floor",
    "wall", "window", "piece", "light", "tree", "eye", "school",
    "family", "kind", "sort", "type", "form", "level", "result",
    "apple", "man", "woman", "child", "friend", "guy", "car", "water",
}

# Common conversational words (A1-B1 vocabulary) that B2 translation students already know
COMMON_CONVERSATIONAL_WORDS = {
    # Basic verbs
    "say", "tell", "go", "get", "make", "take", "see", "know", "think", "look",
    "want", "give", "use", "find", "ask", "work", "seem", "feel", "try", "leave",
    "call", "keep", "apply", "hire", "read", "remember", "stand", "study", "lose",
    "walk", "hear", "stop", "open", "show", "hold", "move", "live", "bring", "happen",
    "write", "provide", "sit", "win", "meet", "run", "pay", "put", "let", "begin",
    "help", "talk", "turn", "start", "play", "set", "learn", "change", "lead",
    "understand", "watch", "follow", "create", "speak", "allow", "add", "spend",
    "grow", "offer", "love", "consider", "appear", "buy", "wait", "serve", "die",
    "send", "expect", "build", "stay", "fall", "cut", "reach", "kill", "remain",
    "suggest", "raise", "pass", "sell", "require", "report", "decide", "pull", "crunch",

    # Basic adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "blue", "red",
    "green", "white", "black", "recent", "hot", "cold", "cool", "warm", "easy",
    "hard", "clear", "certain", "free", "open", "special", "major", "better", "best",
    "sure", "low", "summer", "unique", "individual",

    # Basic nouns
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
    "child", "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact", "month",
    "lot", "study", "book", "eye", "job", "word", "business", "issue", "side",
    "kind", "head", "house", "service", "friend", "father", "power", "hour",
    "game", "line", "end", "member", "law", "car", "city", "name", "moment",
    "minute", "idea", "body", "face", "level", "door", "person", "self",
    "secretary", "committee", "stage", "cut", "decision", "budget",
}

# Scientific & technical morphemes
TECHNICAL_AFFIXES = (
    "photo", "bio", "neuro", "psycho", "micro", "macro", "thermo",
    "electro", "nano", "gen", "poly", "hydro", "patho", "chrono"
)

TECHNICAL_SUFFIXES = (
    "itis", "ology", "omics", "genesis", "philic", "phobic",
    "ation", "escence", "biotic", "metric", "trophic"
)

# Common descriptive adjectives that form ordinary, compositionally transparent phrases
ORDINARY_DESCRIPTIVE_ADJECTIVES = {
    "amusing", "happy", "sad", "good", "bad", "nice", "big", "small",
    "large", "little", "great", "new", "old", "young", "important",
    "interesting", "adjacent", "different", "similar", "early", "late",
    "long", "short", "high", "low", "right", "wrong", "simple", "easy",
    "hard", "beautiful", "fine", "cool", "warm", "hot", "cold",
}

# Demonym suffixes — words ending in these + capitalized are likely
# nationalities/demonyms (NORP entities), not vocabulary terms
DEMONYM_SUFFIXES = (
    "ians", "ians", "ans", "ese", "ish", "ish",
    "ian", "an", "er", "ers",
)

# Known demonyms that spaCy might not tag as NORP
KNOWN_DEMONYMS = {
    "parisians", "parisian", "londoners", "londoner", "berliners", "berliner",
    "new yorkers", "new yorker", "romans", "roman", "athenians", "athenian",
    "muscovites", "muscovite", "persians", "persian", "turks", "turkish",
    "americans", "american", "canadians", "canadian", "australians", "australian",
    "europeans", "european", "africans", "african", "asians", "asian",
    "mexicans", "mexican", "brazilians", "brazilian", "argentinians", "argentinian",
    "israelis", "israeli", "palestinians", "palestinian",
}

# B2+ academic / uncommon words that appear in Brown corpus but are still
# valuable for translation students to research. These override the
# COMMON_WORDS filter.
ACADEMIC_B2_WORDS = {
    "abstract", "counterpart", "paradigm", "paradox", "dichotomy", "nuance",
    "pragmatic", "rhetoric", "discourse", "ideology", "terminology",
    "methodology", "phenomenon", "ambiguity", "connotation", "denotation",
    "euphemism", "metaphor", "analogy", "allegory", "irony", "satire",
    "coherent", "cohesion", "implicit", "explicit", "intrinsic", "extrinsic",
    "arbitrary", "empirical", "hypothetical", "theoretical", "pragmatic",
    "tangible", "intangible", "subjective", "objective", "profound",
    "prevalent", "predominant", "salient", "pertinent", "relevant",
    "comprehensive", "exhaustive", "rigorous", "meticulous", "scrupulous",
    "deteriorate", "exacerbate", "alleviate", "mitigate", "eradicate",
    "scrutinize", "contemplate", "discern", "perceive", "apprehend",
    "elaborate", "articulate", "substantiate", "corroborate", "validate",
    "constitute", "encompass", "entail", "denote", "connote",
    "compensate", "complement", "supplement", "augment", "diminish",
    "derive", "invoke", "elicit", "evoke", "provoke",
    "concurrent", "subsequent", "antecedent", "preliminary", "provisional",
    "autonomous", "indigenous", "heterogeneous", "homogeneous",
    "unprecedented", "ubiquitous", "prolific", "profound", "perpetual",
    "succinct", "concise", "verbose", "redundant", "superfluous",
    "inherent", "innate", "integral", "peripheral", "pivotal",
    "feasible", "viable", "plausible", "credible", "dubious",
    "benign", "malignant", "acute", "chronic", "latent",
    "itchy", "itchiness", "rash", "lesion", "patch",
    "prognosis", "diagnosis", "symptom", "syndrome", "ailment",
}


def extract_terms(
    doc,
    entity_token_indices: Set[int],
    common_words: Set[str],
    sentences: list,
    mode: str = AnalysisMode.STUDENT
) -> List[Occurrence]:
    """
    Extract terms from a spaCy doc, returning Occurrence objects ranked by Translation Value Score.

    Args:
        doc: spaCy Doc object
        entity_token_indices: set of token indices that belong to named entities
        common_words: set of common English words to filter
        sentences: list of sentence objects from spaCy for context windows
        mode: analysis mode (STUDENT/STANDARD, RESEARCH/DETAILED, RAW)
    """
    occurrences = []
    seen_offsets = set()
    covered_token_indices = set(entity_token_indices)

    # Normalize mode alias
    if mode == AnalysisMode.STANDARD:
        mode = AnalysisMode.STUDENT
    elif mode == AnalysisMode.DETAILED:
        mode = AnalysisMode.RESEARCH

    threshold = 0.40 if mode == AnalysisMode.STUDENT else 0.15

    sentence_contexts = _build_sentence_contexts(sentences)

    # --- Phase 1: Multi-word terms & phrases (PRIORITY) ---
    if mode != AnalysisMode.RAW:
        mw_terms, mw_indices = _extract_valid_multiword_terms(
            doc, entity_token_indices, common_words, sentences, sentence_contexts, mode, threshold
        )
        for mw in mw_terms:
            offset_key = (mw.start_offset, mw.end_offset)
            if offset_key not in seen_offsets:
                seen_offsets.add(offset_key)
                occurrences.append(mw)
        covered_token_indices.update(mw_indices)

    # --- Phase 2: Hyphenated compounds ---
    hyph_terms = _extract_hyphenated_compounds(doc, entity_token_indices, sentences, sentence_contexts)
    for ht in hyph_terms:
        offset_key = (ht.start_offset, ht.end_offset)
        if offset_key not in seen_offsets:
            seen_offsets.add(offset_key)
            occurrences.append(ht)
            # Cover tokens within this span
            for t in doc:
                if t.idx >= ht.start_offset and t.idx + len(t.text) <= ht.end_offset:
                    covered_token_indices.add(t.i)

    # --- Phase 3: Single-word terms (subsumption filtered) ---
    for token in doc:
        # Skip tokens already inside multi-word terms or entities
        if token.i in covered_token_indices:
            continue
        if token.is_stop or token.is_punct or token.is_space or token.like_num:
            continue
        if not token.is_alpha:
            continue
        if len(token.text) < 3:
            continue
        # Never extract proper nouns as vocabulary terms
        if token.pos_ == "PROPN" or token.tag_ in ("NNP", "NNPS"):
            continue
        # Skip known demonyms — these are entity-level items, not vocabulary
        if token.text.lower() in KNOWN_DEMONYMS:
            continue
        # Skip capitalized words that look like demonyms (e.g. "Parisians")
        if token.text[0].isupper() and token.text.lower().rstrip('s').endswith(tuple(s.rstrip('s') for s in ['ians', 'ans', 'ese'])):
            continue
        if token.pos_ not in ("NOUN", "ADJ", "VERB", "ADV"):
            continue

        score, category, reason = _compute_single_word_value(token, common_words)

        if mode != AnalysisMode.RAW:
            if score < threshold:
                continue
            if mode == AnalysisMode.STUDENT and category in (Category.GENERAL_WORD, Category.ORDINARY_PHRASE):
                continue

        offset_key = (token.idx, token.idx + len(token.text))
        if offset_key in seen_offsets:
            continue
        seen_offsets.add(offset_key)

        sent_idx = _get_sentence_index(token, sentences)
        sent_text = token.sent.text.strip() if token.sent else ""
        context = sentence_contexts.get(sent_idx, sent_text)

        occ = Occurrence(
            surface=token.text,
            lemma=token.lemma_.lower(),
            normalized_form=token.text.lower(),
            sentence_idx=sent_idx,
            sentence_text=sent_text,
            context=context,
            start_offset=token.idx,
            end_offset=token.idx + len(token.text),
            pos=token.pos_,
            pos_fine=token.tag_,
            morphology=_extract_morphology(token),
            syntactic_role=token.dep_,
            category=category,
            translation_value=score,
            extraction_reason=reason,
        )
        occurrences.append(occ)

    # Rank by Translation Value Score descending so top research items appear first
    occurrences.sort(key=lambda occ: occ.translation_value, reverse=True)

    return occurrences


def _compute_single_word_value(token, common_words: Set[str]) -> Tuple[float, str, str]:
    """
    Calculate Translation Value Score for a single token from the B2 student perspective.

    Returns:
        (score: float, category: str, extraction_reason: str)
    """
    surface = token.text.lower()
    lemma = token.lemma_.lower()
    pos = token.pos_

    # 1. Species / Biological check
    species_terms = {
        "bonobo", "chimpanzee", "gorilla", "orangutan", "primate",
        "ape", "mammal", "reptile", "amphibian", "cetacean"
    }
    if lemma in species_terms:
        return (0.85, Category.BIOLOGICAL_SPECIES, "biological species classification")

    # 1b. Academic B2+ word check — override common-word filter
    if lemma in ACADEMIC_B2_WORDS or surface in ACADEMIC_B2_WORDS:
        return (0.65, Category.USEFUL_LEXICAL_ITEM, "B2+ academic/specialized vocabulary")

    # 2. Technical / Domain morpheme check
    is_technical = any(lemma.startswith(pref) for pref in TECHNICAL_AFFIXES) or \
                   any(lemma.endswith(suf) for suf in TECHNICAL_SUFFIXES)
    if is_technical:
        return (0.85, Category.SPECIALIZED_TERM, "domain-specific technical/scientific terminology")

    # 3. Rarity & B2 Familiarity check
    is_trivially_common = (
        surface in TRIVIALLY_COMMON or lemma in TRIVIALLY_COMMON or
        surface in COMMON_CONVERSATIONAL_WORDS or lemma in COMMON_CONVERSATIONAL_WORDS
    )
    is_common = is_trivially_common or (lemma in common_words or surface in common_words)

    if is_common:
        # Common everyday words must NOT be highlighted as terms for B2 translation students
        return (0.15, Category.GENERAL_WORD, "common vocabulary item")

    # 4. For uncommon lexical items: check polysemy & domain rarity
    wn_pos = _spacy_pos_to_wn(pos)
    synsets = wn.synsets(lemma, pos=wn_pos) if wn_pos else wn.synsets(lemma)
    synset_count = len(synsets)

    score = 0.55
    if synset_count >= 5:
        score += 0.15
        return (round(score, 2), Category.USEFUL_LEXICAL_ITEM, "lexical item with contextual translation relevance")
    elif synset_count == 0:
        # Specialized neologism or technical term not in WordNet
        score += 0.20
        return (round(score, 2), Category.SPECIALIZED_TERM, "domain-specific terminology")
    else:
        return (round(score, 2), Category.USEFUL_LEXICAL_ITEM, "uncommon lexical item requiring precise translation")


def _compute_multiword_value(
    tokens, term_lower: str, common_words: Set[str]
) -> Tuple[float, str, str]:
    """
    Calculate Translation Value Score for a candidate multi-word phrase.

    Distinguishes genuine terms (e.g. 'photosynthetic activity', 'climate change')
    from compositionally transparent phrases (e.g. 'amusing world').
    """
    # 1. WordNet Compound verification
    wn_lookup = term_lower.replace(" ", "_").replace("-", "_")
    if wn.synsets(wn_lookup):
        return (0.80, Category.MULTIWORD_TERM, "established lexical compound with non-compositional meaning")

    # 2. Check for domain-specific / technical modifiers
    has_technical_word = any(
        any(t.lemma_.lower().startswith(p) for p in TECHNICAL_AFFIXES) or
        any(t.lemma_.lower().endswith(s) for s in TECHNICAL_SUFFIXES)
        for t in tokens
    )
    if has_technical_word:
        return (0.85, Category.SPECIALIZED_TERM, "domain-specific multi-word terminology")

    # 3. Check for ordinary compositional ADJ + NOUN phrases (e.g. "amusing world")
    if len(tokens) == 2 and tokens[0].pos_ == "ADJ" and tokens[1].pos_ == "NOUN":
        adj_text = tokens[0].text.lower()
        noun_text = tokens[1].text.lower()

        # If the adjective is a standard descriptive modifier and noun is common/general
        is_ordinary_adj = (
            adj_text in ORDINARY_DESCRIPTIVE_ADJECTIVES or
            adj_text in COMMON_CONVERSATIONAL_WORDS or
            tokens[0].lemma_.lower() in common_words
        )
        is_ordinary_noun = (
            noun_text in TRIVIALLY_COMMON or
            noun_text in COMMON_CONVERSATIONAL_WORDS or
            tokens[1].lemma_.lower() in common_words
        )

        if is_ordinary_adj and is_ordinary_noun:
            # Compositionally transparent: student easily translates both words literally
            return (0.18, Category.ORDINARY_PHRASE, "compositionally transparent phrase (ordinary adjective + noun)")

    # 4. Check if all words are trivially common
    words_lower = [t.text.lower() for t in tokens]
    if all(w in TRIVIALLY_COMMON or w in COMMON_CONVERSATIONAL_WORDS or w in common_words for w in words_lower):
        return (0.22, Category.ORDINARY_PHRASE, "ordinary multi-word sequence composed of common words")

    # 5. Collocation / Compound with one specialized element
    return (0.55, Category.COLLOCATION, "specialized collocation with potential translation significance")


def _extract_valid_multiword_terms(
    doc, entity_token_indices, common_words, sentences, sentence_contexts, mode, threshold
) -> Tuple[List[Occurrence], Set[int]]:
    """
    Extract multi-word terms, applying Translation Value Scoring and mode filtering.
    Returns (terms, covered_token_indices).
    """
    terms = []
    covered_indices = set()
    seen_spans = set()

    for chunk in doc.noun_chunks:
        content_tokens = [
            t for t in chunk
            if not t.is_stop and not t.is_punct and not t.is_space
            and t.pos_ in ("NOUN", "ADJ", "PROPN")
            and t.i not in entity_token_indices
        ]

        if len(content_tokens) < 2:
            continue

        # Reject possessive constructions ('s, s')
        chunk_text = chunk.text
        if "'s " in chunk_text or "s' " in chunk_text:
            continue
        if any(t.dep_ == "poss" or t.tag_ == "POS" for t in chunk):
            continue

        # Must end in NOUN or PROPN
        if content_tokens[-1].pos_ not in ("NOUN", "PROPN"):
            continue

        term_text = " ".join(t.text for t in content_tokens)
        term_lower = term_text.lower()

        if len(term_text) < 5:
            continue

        score, category, reason = _compute_multiword_value(content_tokens, term_lower, common_words)

        # Mode filtering: Student Mode suppresses ordinary phrases and low-scoring items
        if mode == AnalysisMode.STUDENT:
            if category == Category.ORDINARY_PHRASE or score < threshold:
                continue
        elif mode == AnalysisMode.RESEARCH:
            if score < threshold:
                continue

        span_key = (content_tokens[0].idx, content_tokens[-1].idx + len(content_tokens[-1].text))
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        # Mark all tokens in this multi-word term as covered
        for t in content_tokens:
            covered_indices.add(t.i)

        sent_idx = _get_sentence_index(content_tokens[0], sentences)
        sent_text = chunk.sent.text.strip() if chunk.sent else ""
        context = sentence_contexts.get(sent_idx, sent_text)

        occ = Occurrence(
            surface=term_text,
            lemma=term_lower,
            normalized_form=term_lower,
            sentence_idx=sent_idx,
            sentence_text=sent_text,
            context=context,
            start_offset=content_tokens[0].idx,
            end_offset=content_tokens[-1].idx + len(content_tokens[-1].text),
            pos="NOUN",
            pos_fine="NNP" if any(t.pos_ == "PROPN" for t in content_tokens) else "NN",
            morphology={},
            syntactic_role="compound",
            category=category,
            translation_value=score,
            extraction_reason=reason,
        )
        terms.append(occ)

    return terms, covered_indices


def _extract_hyphenated_compounds(doc, entity_token_indices, sentences, sentence_contexts) -> List[Occurrence]:
    """Extract hyphenated compounds like 'sensory-friendly', 'free-loving'."""
    terms = []
    seen = set()

    hyph_pattern = re.compile(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)\b')
    for m in hyph_pattern.finditer(doc.text):
        compound = m.group(1)
        if len(compound) < 6:
            continue
        if compound.lower() in seen:
            continue

        sent_idx = -1
        sent_text = ""
        for i, sent in enumerate(sentences):
            if m.start() >= sent.start_char and m.start() < sent.end_char:
                sent_idx = i
                sent_text = sent.text.strip()
                break

        context = sentence_contexts.get(sent_idx, sent_text)

        occ = Occurrence(
            surface=compound,
            lemma=compound.lower(),
            normalized_form=compound.lower(),
            sentence_idx=sent_idx,
            sentence_text=sent_text,
            context=context,
            start_offset=m.start(),
            end_offset=m.end(),
            pos="ADJ",
            pos_fine="JJ",
            morphology={},
            syntactic_role="amod",
            category=Category.COLLOCATION,
            translation_value=0.65,
            extraction_reason="hyphenated compound collocation",
        )
        terms.append(occ)
        seen.add(compound.lower())

    return terms


def _extract_morphology(token) -> dict:
    """Extract morphological features from a spaCy token."""
    morph = {}
    morph_str = str(token.morph)
    if morph_str:
        for feature in morph_str.split("|"):
            if "=" in feature:
                key, value = feature.split("=", 1)
                morph[key] = value
    return morph


def _spacy_pos_to_wn(pos_tag: str):
    """Map a spaCy POS tag to WordNet POS."""
    return {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV
    }.get(pos_tag)


def _get_sentence_index(token, sentences) -> int:
    """Get the index of the sentence containing a token."""
    for i, sent in enumerate(sentences):
        if token.idx >= sent.start_char and token.idx < sent.end_char:
            return i
    return 0


def _build_sentence_contexts(sentences) -> dict:
    """
    Build context windows: for each sentence index, return
    prev_sentence + current_sentence + next_sentence.
    """
    contexts = {}
    sent_texts = [s.text.strip() for s in sentences]

    for i in range(len(sent_texts)):
        parts = []
        if i > 0:
            parts.append(sent_texts[i - 1])
        parts.append(sent_texts[i])
        if i < len(sent_texts) - 1:
            parts.append(sent_texts[i + 1])
        contexts[i] = " ".join(parts)

    return contexts

