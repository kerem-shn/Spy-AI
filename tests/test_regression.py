"""
Spy AI — Regression Tests

Tests covering all 10 failure cases identified from the screenshots.
These ensure the refactored pipeline solves the original architectural problems.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from text_preprocessor import segment_document, get_content_text
from models import Occurrence, Category


# ══════════════════════════════════════════════════════════════
# Test 1: "stranger's room" must NOT be extracted as a single term
# ══════════════════════════════════════════════════════════════

class TestStrangerRoom:
    """Regression: 'stranger room' was created from 'stranger's room' noun chunk."""

    def test_possessive_not_merged(self):
        """Possessive constructions should not produce compound terms."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms

        text = "The bonobos were willing to help a stranger access food in the stranger's room."
        doc = nlp(text)
        sentences = list(doc.sents)

        # No entity token indices for this test
        occurrences = extract_terms(doc, set(), set(), sentences)

        # Get all extracted surfaces
        surfaces = [occ.surface.lower() for occ in occurrences]
        lemmas = [occ.lemma.lower() for occ in occurrences]

        # "stranger room" should NOT appear as a term
        assert "stranger room" not in surfaces, \
            f"'stranger room' should not be extracted as a term. Got surfaces: {surfaces}"
        assert "stranger room" not in lemmas, \
            f"'stranger room' should not appear in lemmas. Got: {lemmas}"


# ══════════════════════════════════════════════════════════════
# Test 2: "source text" in assignment instructions must be metadata
# ══════════════════════════════════════════════════════════════

class TestMetadataDetection:
    """Regression: 'source text', 'min words' from instructions were analyzed as terms."""

    def test_instruction_text_filtered(self):
        """Assignment instructions should be classified as non-content."""
        text = """1. Translate the following source text into Turkish.
2. Write a commentary on your translation. (min. 300 words)

Name and Surname:
Student ID:

Bonobos are great apes known for their peaceful behavior."""

        segments = segment_document(text)
        content = get_content_text(segments)

        # "Translate the following" should NOT be in content
        assert "Translate the following" not in content, \
            f"Instruction text should be filtered. Content: {content}"

        # "min. 300 words" should NOT be in content
        assert "min." not in content, \
            f"'min. 300 words' should be filtered. Content: {content}"

        # "Name and Surname:" should NOT be in content
        assert "Name and Surname:" not in content, \
            f"Metadata should be filtered. Content: {content}"

        # Actual article content SHOULD remain
        assert "Bonobos are great apes" in content, \
            f"Article content should be preserved. Content: {content}"

    def test_source_text_not_a_term(self):
        """'source text' from metadata should never become an analyzed term."""
        text = "Translate the following source text into Turkish."
        segments = segment_document(text)

        # This whole line should be instruction type
        assert any(s.segment_type == "instruction" for s in segments), \
            f"Line should be 'instruction', got: {[(s.segment_type, s.text[:40]) for s in segments]}"


# ══════════════════════════════════════════════════════════════
# Test 3: Entity Detection — "James Paget Hospital"
# ══════════════════════════════════════════════════════════════

class TestEntityDetection:
    """Regression: James Paget Hospital was resolved to the band 'James'."""

    def test_hospital_entity_type(self):
        """'James Paget Hospital' should be detected as ORGANIZATION."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from entity_resolver import detect_entities

        text = "He was admitted to James Paget Hospital in Gorleston, Norfolk."
        doc = nlp(text)

        entity_spans, _ = detect_entities(doc)

        # Find the entity for James Paget Hospital
        hospital_entities = [e for e in entity_spans if "Paget" in e.surface or "James" in e.surface]

        if hospital_entities:
            ent = hospital_entities[0]
            # Should be ORG, not PERSON
            assert ent.entity_type in ("ORG", "FAC", "GPE"), \
                f"James Paget Hospital should be ORG/FAC, got: {ent.entity_type}"

    def test_hospital_disambiguation(self):
        """Hospital entity should resolve to a medical institution, not a band."""
        from entity_resolver import _disambiguate_wikipedia

        options = ["James (band)", "James Paget Hospital", "James (given name)", "James Paget"]
        result = _disambiguate_wikipedia(
            options, "James Paget Hospital", "ORG", "hospital",
            "He was admitted to James Paget Hospital in Gorleston."
        )

        assert result == "James Paget Hospital", \
            f"Should disambiguate to 'James Paget Hospital', got: {result}"

    def test_entity_validation_wrong_type(self):
        """Summary about a band should fail validation for ORG/hospital entity."""
        from entity_resolver import _validate_entity_summary

        band_summary = "James is a rock band formed in Manchester, England in 1982. The band's singer is Tim Booth."
        result = _validate_entity_summary(band_summary, "ORG", "hospital", "admitted to hospital")

        assert result is False, \
            "Band summary should fail validation for hospital entity"

    def test_entity_validation_correct_type(self):
        """Summary about a hospital should pass validation for ORG/hospital entity."""
        from entity_resolver import _validate_entity_summary

        hospital_summary = "James Paget University Hospital is a hospital in Gorleston, Norfolk, England."
        result = _validate_entity_summary(hospital_summary, "ORG", "hospital", "admitted to hospital")

        assert result is True, \
            "Hospital summary should pass validation for hospital entity"


# ══════════════════════════════════════════════════════════════
# Test 4: No hallucinated definitions
# ══════════════════════════════════════════════════════════════

class TestNoHallucination:
    """Regression: System fabricated definitions like 'recognized across a nation'."""

    def test_no_fabricated_template_definitions(self):
        """Definitions should not use template phrases like 'recognized and acknowledged across'."""
        from wsd import _wordnet_senses

        # Test with a word that exists in WordNet
        senses = _wordnet_senses("incentive", "Bonobos were motivated by incentives.", "n")

        for sense in senses:
            assert "recognized and acknowledged across" not in sense.definition, \
                f"Fabricated template definition found: {sense.definition}"
            assert "across an entire country" not in sense.definition, \
                f"Fabricated template definition found: {sense.definition}"


# ══════════════════════════════════════════════════════════════
# Test 5: Occurrence-level independence
# ══════════════════════════════════════════════════════════════

class TestOccurrenceIndependence:
    """Regression: Same word in different sentences got same analysis."""

    def test_same_word_different_contexts(self):
        """Each occurrence of the same word should carry its own context."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms

        text = "The patch appeared on his skin. She applied a software patch to fix the bug."
        doc = nlp(text)
        sentences = list(doc.sents)

        occurrences = extract_terms(doc, set(), set(), sentences, mode="detailed")

        # Find occurrences of "patch"
        patch_occs = [occ for occ in occurrences if occ.lemma == "patch"]

        if len(patch_occs) >= 2:
            # They should have different sentence contexts
            contexts = [occ.sentence_text for occ in patch_occs]
            assert len(set(contexts)) >= 2, \
                f"'patch' should have different contexts, got: {contexts}"


# ══════════════════════════════════════════════════════════════
# Test 6: Possessive morphology
# ══════════════════════════════════════════════════════════════

class TestPossessiveMorphology:
    """Regression: Possessives like stranger's caused incorrect term boundaries."""

    def test_possessive_boundary(self):
        """Possessive constructions must not merge into compound terms."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms

        text = "The researchers' findings were published. The stranger's room was empty."
        doc = nlp(text)
        sentences = list(doc.sents)

        occurrences = extract_terms(doc, set(), set(), sentences)
        surfaces = [occ.surface.lower() for occ in occurrences]

        # "researcher findings" and "stranger room" should NOT be compounds
        assert "researcher findings" not in surfaces
        assert "stranger room" not in surfaces
        assert "researchers findings" not in surfaces


# ══════════════════════════════════════════════════════════════
# Test 7: URL detection
# ══════════════════════════════════════════════════════════════

class TestURLDetection:
    """Regression: URLs in source text were treated as content."""

    def test_url_filtered(self):
        """URLs should be classified as non-content."""
        text = """Source: https://www.nationalgeographic.com/animals/article/bonobos-help-strangers

Bonobos are remarkable primates known for altruistic behavior."""

        segments = segment_document(text)
        content = get_content_text(segments)

        assert "nationalgeographic.com" not in content, \
            f"URL should be filtered from content. Got: {content}"
        assert "Bonobos are remarkable" in content, \
            f"Article content should remain. Got: {content}"


# ══════════════════════════════════════════════════════════════
# Test 8: Document Segment Types
# ══════════════════════════════════════════════════════════════

class TestDocumentSegments:
    """Test proper classification of document segments."""

    def test_assignment_header(self):
        text = "Assignment 3\nName and Surname:\nStudent ID:\n\nBonobos are peaceful apes."
        segments = segment_document(text)

        content_segments = [s for s in segments if s.segment_type == "content"]
        non_content = [s for s in segments if s.segment_type != "content"]

        assert len(content_segments) >= 1, "Should have at least one content segment"
        assert len(non_content) >= 1, "Should have non-content segments"
        assert any("Bonobos" in s.text for s in content_segments), \
            "Article text should be in content segments"

    def test_min_words_instruction(self):
        text = "(min. 300 words)\nThe bonobos showed remarkable behavior."
        segments = segment_document(text)

        # The "(min. 300 words)" should be instruction type
        instruction_segs = [s for s in segments if s.segment_type == "instruction"]
        assert len(instruction_segs) >= 1, \
            f"Should detect '(min. 300 words)' as instruction. Segments: {[(s.segment_type, s.text[:30]) for s in segments]}"


# ══════════════════════════════════════════════════════════════
# Test 9: Term worthiness
# ══════════════════════════════════════════════════════════════

class TestTermWorthiness:
    """Ensure trivially common words are not highlighted in STANDARD mode."""

    def test_trivially_common_words_filtered(self):
        """Words like 'room', 'apple', 'time' should not be terms in STANDARD mode."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms, AnalysisMode

        text = "The room had a table with an apple and a book on it."
        doc = nlp(text)
        sentences = list(doc.sents)

        occurrences = extract_terms(doc, set(), set(), sentences, mode=AnalysisMode.STANDARD)
        surfaces = [occ.surface.lower() for occ in occurrences]

        # These trivially common words should NOT be highlighted in STANDARD mode
        assert "room" not in surfaces, f"'room' should be filtered in STANDARD mode. Got: {surfaces}"
        assert "table" not in surfaces, f"'table' should be filtered in STANDARD mode. Got: {surfaces}"
        assert "apple" not in surfaces, f"'apple' should be filtered in STANDARD mode. Got: {surfaces}"
        assert "book" not in surfaces, f"'book' should be filtered in STANDARD mode. Got: {surfaces}"


# ══════════════════════════════════════════════════════════════
# Test 10: Confidence gating
# ══════════════════════════════════════════════════════════════

class TestConfidenceGating:
    """Ensure confidence levels are properly calculated."""

    def test_confidence_levels(self):
        """ConfidenceLevel.from_score should return correct tiers."""
        from models import ConfidenceLevel

        assert ConfidenceLevel.from_score(0.9) == "high"
        assert ConfidenceLevel.from_score(0.75) == "high"
        assert ConfidenceLevel.from_score(0.6) == "medium"
        assert ConfidenceLevel.from_score(0.45) == "medium"
        assert ConfidenceLevel.from_score(0.3) == "low"
        assert ConfidenceLevel.from_score(0.0) == "low"

    def test_occurrence_has_confidence(self):
        """Every occurrence should have a confidence dict."""
        occ = Occurrence(
            surface="test",
            lemma="test",
            pos="NOUN",
        )
        assert "term" in occ.confidence
        assert "sense" in occ.confidence
        assert "translation" in occ.confidence
        assert "overall" in occ.confidence


# ══════════════════════════════════════════════════════════════
# NEW REGRESSION TESTS (Second Iteration)
# ══════════════════════════════════════════════════════════════

class TestContextContamination:
    """
    REGRESSION TEST 1: Context contamination between neighboring words.
    Sentence: 'The herder returned to his hometown.'
    'herder' translation must NOT contain 'memleket'.
    'hometown' translation may contain 'memleket'.
    The two occurrences must have independent semantic analyses.
    """

    def test_herder_does_not_contain_memleket(self):
        from translator import translate_occurrence
        from models import Occurrence

        sentence = "The herder returned to his hometown."

        # Simulate Google Translate behavior where bracket marker shifted to 'memleketine'
        def mock_translate(text: str) -> str:
            if "[[herder]]" in text or "[[The herder]]" in text:
                # Simulated marker drift onto neighboring word
                return "Çoban [[memleketine]] döndü."
            elif text.lower() == "herder":
                return "çoban"
            elif text.lower() == "hometown":
                return "memleket"
            elif text.lower() == "returned":
                return "döndü"
            return text

        occ_herder = Occurrence(
            surface="herder",
            lemma="herder",
            sentence_text=sentence,
            pos="NOUN",
        )
        occ_hometown = Occurrence(
            surface="hometown",
            lemma="hometown",
            sentence_text=sentence,
            pos="NOUN",
        )

        # Translate both occurrences independently
        translate_occurrence(occ_herder, translate_fn=mock_translate)
        translate_occurrence(occ_hometown, translate_fn=mock_translate)

        herder_trans = [t.text.lower() for t in occ_herder.translations]
        hometown_trans = [t.text.lower() for t in occ_hometown.translations]

        # Herder must NOT contain memleket or memleketine
        assert not any("memleket" in t for t in herder_trans), \
            f"'herder' translation must NOT contain 'memleket'. Got: {herder_trans}"
        assert "memleket" not in occ_herder.selected_translation.lower()

        # Hometown may contain memleket
        assert any("memleket" in t for t in hometown_trans) or occ_hometown.selected_translation == "memleket", \
            f"'hometown' should be translated as 'memleket'. Got: {hometown_trans}"

        # Independent objects
        assert occ_herder.id != occ_hometown.id
        assert occ_herder.translations != occ_hometown.translations


class TestNoEnglishAsTurkishAlternative:
    """
    REGRESSION TEST 2: The first translation is correct, but the alternative is just the English word.
    Source: 'herder'
    Expected: first translation is valid Turkish (e.g. 'çoban'),
    alternative must NOT be 'herder'.
    No English source-language duplicate is allowed.
    """

    def test_no_english_duplicate_in_alternatives(self):
        from translator import translate_occurrence
        from models import Occurrence

        # Simulate GTX bilingual dictionary returning 'herder' as an entry alongside 'çoban'
        class MockGTXClient:
            def query_raw(self, word, sl="en", tl="tr"):
                # Returns 'çoban' as primary, and both 'çoban' and 'herder' in dictionary
                return [
                    [["çoban", word, None, None]],
                    [["noun", ["çoban", "herder", "sürü çobanı"]]]
                ]

        def mock_translate(text):
            return "çoban"

        occ = Occurrence(
            surface="herder",
            lemma="herder",
            pos="NOUN",
        )

        translate_occurrence(occ, translate_fn=mock_translate, gtx_client=MockGTXClient())

        trans_texts = [t.text.lower().strip() for t in occ.translations]

        # 'herder' can NEVER appear as a translation
        assert "herder" not in trans_texts, \
            f"'herder' must NOT appear as a Turkish translation. Got: {trans_texts}"

        # Selected translation should be Turkish
        assert occ.selected_translation.lower() in ("çoban", "sürü çobanı")

        # Every candidate must pass validation
        for t in trans_texts:
            assert t != "herder", f"Found unchanged source word in translations: {t}"


class TestAmusingWorldDeprioritized:
    """
    REGRESSION TEST 3: 'amusing world' should NOT automatically be extracted as a high-priority translation term.
    """

    def test_amusing_world_suppressed_in_student_mode(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms, AnalysisMode

        text = "The protagonist lived in an amusing world where anything could happen."
        doc = nlp(text)
        sentences = list(doc.sents)

        # In Student Mode (default)
        occurrences = extract_terms(doc, set(), set(), sentences, mode=AnalysisMode.STUDENT)
        surfaces = [occ.surface.lower() for occ in occurrences]

        # 'amusing world' should NOT be extracted in Student Mode
        assert "amusing world" not in surfaces, \
            f"'amusing world' should NOT be extracted in Student Mode. Extracted: {surfaces}"

    def test_amusing_world_has_low_translation_value(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms, AnalysisMode
        from models import Category

        text = "She lived in an amusing world."
        doc = nlp(text)
        sentences = list(doc.sents)

        # In Research Mode (lower threshold to inspect)
        occurrences = extract_terms(doc, set(), set(), sentences, mode=AnalysisMode.RESEARCH)
        amusing_occ = next((occ for occ in occurrences if occ.surface.lower() == "amusing world"), None)

        if amusing_occ:
            assert amusing_occ.translation_value < 0.35, \
                f"'amusing world' should have low translation value. Got: {amusing_occ.translation_value}"
            assert amusing_occ.category == Category.ORDINARY_PHRASE


class TestDifficultyAwarePrioritization:
    """
    REGRESSION TEST 4: The system should prioritize genuinely translation-relevant vocabulary
    over ordinary adjective+noun phrases.
    """

    def test_specialized_terms_prioritized_over_ordinary_phrases(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms, AnalysisMode

        text = "In an amusing world, the herder monitored photosynthetic activity."
        doc = nlp(text)
        sentences = list(doc.sents)

        occurrences = extract_terms(doc, set(), set(), sentences, mode=AnalysisMode.STUDENT)
        surfaces = [occ.surface.lower() for occ in occurrences]

        # Specialized / difficult terms should be extracted
        assert any("herder" in s for s in surfaces), f"'herder' should be extracted. Got: {surfaces}"
        assert any("photosynthetic" in s for s in surfaces), f"'photosynthetic activity' should be extracted. Got: {surfaces}"

        # 'amusing world' must not be in Student Mode
        assert "amusing world" not in surfaces

        # Occurrences are sorted by translation value descending
        for i in range(len(occurrences) - 1):
            assert occurrences[i].translation_value >= occurrences[i + 1].translation_value, \
                "Occurrences must be sorted by translation_value descending"


class TestOccurrenceIsolation:
    """
    REGRESSION TEST 5: Repeated words:
    'The herder saw the village. Later, another herder returned to his hometown.'
    Each occurrence must have its own analysis object.
    """

    def test_repeated_word_independent_occurrences(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            pytest.skip("spaCy not available")

        from term_extractor import extract_terms, AnalysisMode

        text = "The herder saw the village. Later, another herder returned to his hometown."
        doc = nlp(text)
        sentences = list(doc.sents)

        occurrences = extract_terms(doc, set(), set(), sentences, mode=AnalysisMode.STUDENT)
        herder_occs = [occ for occ in occurrences if occ.lemma == "herder"]

        # Exactly 2 occurrences of 'herder'
        assert len(herder_occs) == 2, f"Expected 2 occurrences of 'herder', got {len(herder_occs)}"

        occ1, occ2 = herder_occs[0], herder_occs[1]

        # Distinct IDs
        assert occ1.id != occ2.id

        # Distinct positions
        assert occ1.start_offset != occ2.start_offset
        assert occ1.end_offset != occ2.end_offset

        # Distinct sentences
        assert occ1.sentence_idx != occ2.sentence_idx
        assert occ1.sentence_text != occ2.sentence_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

