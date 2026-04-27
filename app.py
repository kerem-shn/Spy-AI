"""
Spy AI — Translation Pre-Research Assistant
Flask backend for analyzing source texts and providing context-aware
translations, meanings, and entity summaries.
"""

import os
import re
import logging
from flask import Flask, render_template, request, jsonify

import spacy
from deep_translator import GoogleTranslator
try:
    from deep_translator import DeeplTranslator
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

import wikipedia
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Optional imports for file parsing
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpyAI")

for resource in ["wordnet", "omw-1.4", "punkt", "punkt_tab",
                  "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

wikipedia.set_lang("en")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt"}
ENTITY_LABELS = {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART", "NORP"}
ENTITY_LABEL_DISPLAY = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "EVENT": "Event",
    "WORK_OF_ART": "Work of Art",
    "NORP": "Group/Nationality",
}

# ---------------------------------------------------------------------------
# Helpers — File Parsing
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_storage) -> str:
    """Extract plain text from an uploaded file (PDF, DOCX, or TXT)."""
    filename = file_storage.filename.lower()
    ext = filename.rsplit(".", 1)[1] if "." in filename else ""

    if ext == "pdf":
        if not PDF_AVAILABLE:
            raise ValueError("PDF support requires 'pypdf'. Install: pip install pypdf")
        reader = PdfReader(file_storage)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    elif ext in ("docx", "doc"):
        if not DOCX_AVAILABLE:
            raise ValueError("DOCX support requires 'python-docx'. Install: pip install python-docx")
        doc = DocxDocument(file_storage)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    elif ext == "txt":
        return file_storage.read().decode("utf-8", errors="replace").strip()

    else:
        raise ValueError(f"Unsupported file type: .{ext}")


# ---------------------------------------------------------------------------
# Helpers — Translation
# ---------------------------------------------------------------------------

def build_translator(direction: str, deepl_key: str | None = None):
    """Return a translator function that tries DeepL first, then Google."""
    src, tgt = ("en", "tr") if direction == "en-tr" else ("tr", "en")

    deepl_translator = None
    if deepl_key and DEEPL_AVAILABLE:
        try:
            deepl_translator = DeeplTranslator(
                api_key=deepl_key,
                source=src,
                target=tgt,
                use_free_api=deepl_key.strip().endswith(":fx"),
            )
            deepl_translator.translate("test")
            logger.info("DeepL translator initialized successfully.")
        except Exception as e:
            logger.warning(f"DeepL init failed ({e}); falling back to Google Translate.")
            deepl_translator = None

    google_translator = GoogleTranslator(source=src, target=tgt)

    def translate(text: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            if deepl_translator:
                return deepl_translator.translate(text)
        except Exception:
            pass
        try:
            return google_translator.translate(text)
        except Exception:
            return ""

    engine_name = "DeepL" if deepl_translator else "Google Translate"
    return translate, engine_name


# ---------------------------------------------------------------------------
# Helpers — Smarter Term Filtering
# ---------------------------------------------------------------------------

def is_term(token, entity_spans: set) -> bool:
    """Single-word noun term detection."""
    if token.pos_ != "NOUN":
        return False
    if len(token.text) < 5:
        return False
    if token.is_stop or token.is_punct or token.is_space or token.like_num:
        return False
    if not token.is_alpha:
        return False
    if token.i in entity_spans:
        return False
    lemma = token.lemma_.lower()
    if not wn.synsets(lemma):
        return False
    return True


def extract_multiword_terms(doc, entity_spans: set) -> list:
    """
    Extract multi-word terms using spaCy noun chunks, hyphenated compounds,
    and quoted terms (e.g. 'mother patch', 'herald patch').
    """
    terms = []
    seen = set()

    # 1. spaCy noun chunks (multi-word noun phrases)
    for chunk in doc.noun_chunks:
        # Skip chunks that are single common words or overlap entities
        tokens = [t for t in chunk if not t.is_stop and not t.is_punct
                  and not t.is_space and t.pos_ in ("NOUN", "ADJ", "PROPN")]
        if len(tokens) < 2:
            continue
        # Skip if any token overlaps with named entities
        if any(t.i in entity_spans for t in tokens):
            continue
        text = " ".join(t.text for t in tokens).strip()
        if len(text) < 5 or text.lower() in seen:
            continue
        seen.add(text.lower())
        sentence = chunk.sent.text.strip() if chunk.sent else ""
        terms.append({"text": text, "sentence": sentence})

    # 2. Hyphenated compounds
    hyphenated = re.findall(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)\b', doc.text)
    for h in set(hyphenated):
        if len(h) > 5 and h.lower() not in seen:
            seen.add(h.lower())
            for sent in doc.sents:
                if h in sent.text:
                    terms.append({"text": h, "sentence": sent.text.strip()})
                    break

    # 3. Quoted terms — detect patterns like 'mother' patch, 'herald' patch
    # or full quoted phrases like 'herald patch'
    quoted = re.findall(r"['‘’“”\"](\w+(?:\s+\w+)*)['‘’“”\"]", doc.text)
    for q in quoted:
        q_clean = q.strip()
        if len(q_clean) < 3 or q_clean.lower() in seen:
            continue
        # Try to find this quoted word + the next noun as a compound
        # e.g., 'mother' patch -> "mother patch"
        for sent in doc.sents:
            if q_clean in sent.text:
                # Look for pattern: 'quoted' + following_noun
                pattern = re.search(
                    r"['‘’“”\"]" + re.escape(q_clean) + r"['‘’“”\"]\s*(\w+)",
                    sent.text
                )
                if pattern:
                    following = pattern.group(1)
                    compound = f"{q_clean} {following}"
                    if compound.lower() not in seen:
                        seen.add(compound.lower())
                        terms.append({"text": compound, "sentence": sent.text.strip()})
                # Also add the quoted word itself if it's a noun
                if q_clean.lower() not in seen and len(q_clean) >= 5:
                    if wn.synsets(q_clean.lower()):
                        seen.add(q_clean.lower())
                        terms.append({"text": q_clean, "sentence": sent.text.strip()})
                break

    return terms


def get_sentence_for_token(token) -> str:
    """Return the sentence string that contains this token."""
    return token.sent.text.strip() if token.sent else ""


# ---------------------------------------------------------------------------
# Helpers — Better WSD for Compounds
# ---------------------------------------------------------------------------

def get_context_aware_meanings(word: str, sentence: str, limit: int = 3):
    """
    Use NLTK Lesk for WSD. For hyphenated compounds, try the full term
    first, then fall back to the head word (last component).
    """
    meanings = []
    lookup_word = word

    # For hyphenated compounds, try full compound first
    if "-" in word:
        compound_underscore = word.replace("-", "_")
        if wn.synsets(compound_underscore):
            lookup_word = compound_underscore
        else:
            # Use the head word (last component in English compounds)
            parts = word.split("-")
            # Try last part first, then first part
            for part in reversed(parts):
                if len(part) > 3 and wn.synsets(part):
                    lookup_word = part
                    break

    try:
        tokens = word_tokenize(sentence)
    except Exception:
        tokens = sentence.split()

    # Lesk disambiguation
    best_sense = None
    try:
        best_sense = lesk(tokens, lookup_word, "n")
        if not best_sense:
            best_sense = lesk(tokens, lookup_word, "v")
        if not best_sense:
            best_sense = lesk(tokens, lookup_word, "a")  # adjective
        if not best_sense:
            best_sense = lesk(tokens, lookup_word)
    except Exception:
        pass

    all_synsets = wn.synsets(lookup_word)

    if best_sense:
        meanings.append({
            "definition": best_sense.definition(),
            "is_primary": True,
        })
        for syn in all_synsets:
            if syn != best_sense and len(meanings) < limit:
                meanings.append({
                    "definition": syn.definition(),
                    "is_primary": False,
                })
    else:
        for syn in all_synsets[:limit]:
            meanings.append({
                "definition": syn.definition(),
                "is_primary": len(meanings) == 0,
            })

    if not meanings:
        meanings.append({"definition": "No definition available.", "is_primary": True})

    return meanings


def get_translations(word: str, translate_fn):
    """Get translations for a term (no sentence translation)."""
    translations = set()

    # 1. Translate the isolated word/compound
    try:
        direct = translate_fn(word)
        if direct:
            translations.add(direct)
    except Exception:
        pass

    # 2. WordNet synonyms for alternatives
    lookup = word.replace("-", "_") if "-" in word else word
    try:
        for syn in wn.synsets(lookup)[:3]:
            for lemma in syn.lemmas()[:2]:
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    try:
                        alt = translate_fn(name)
                        if alt:
                            translations.add(alt)
                    except Exception:
                        pass
    except Exception:
        pass

    result = list(translations)[:4]
    if not result:
        result = ["Translation unavailable"]

    return result


# ---------------------------------------------------------------------------
# Helpers — NER False Positive Filtering
# ---------------------------------------------------------------------------

def is_valid_entity(name: str, label: str) -> bool:
    """
    Filter NER false positives — but keep legitimate entities.
    """
    name_stripped = name.strip()
    if len(name_stripped) < 2:
        return False
    if name_stripped.isdigit():
        return False

    # PERSON and GPE are usually reliable — always keep them
    if label in ("PERSON", "GPE"):
        return True

    # NORP (nationalities/groups) — keep if capitalized
    if label == "NORP":
        return name_stripped[0].isupper()

    # For ORG, EVENT, WORK_OF_ART:
    # Only filter if it's clearly a common/medical term, NOT a proper noun
    words = name_stripped.split()

    # Single-word entity: filter only if it's a medical/scientific term
    if len(words) == 1:
        synsets = wn.synsets(name_stripped.lower())
        if synsets:
            for s in synsets:
                defn = s.definition().lower()
                medical_kws = ["disease", "condition", "disorder", "inflammation",
                               "infection", "syndrome", "skin", "rash",
                               "medicine", "therapy", "treatment", "tissue"]
                if any(kw in defn for kw in medical_kws):
                    return False
        return True

    # Multi-word: filter only if the combined phrase IS in WordNet
    # (meaning it's a dictionary term, not a proper noun)
    if len(words) >= 2:
        combined = "_".join(w.lower() for w in words)
        if wn.synsets(combined):
            # It's a real dictionary term
            return False

    return True


# ---------------------------------------------------------------------------
# Helpers — Improved Entity Research
# ---------------------------------------------------------------------------

def get_entity_summary(name: str, label: str) -> dict:
    """
    Get a comprehensive summary for a named entity.
    Tries Wikipedia (EN + TR), then DuckDuckGo with multiple strategies.
    """
    summary = ""
    source = ""

    # 1. Try English Wikipedia
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(name, sentences=3)
        source = "Wikipedia"
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            try:
                summary = wikipedia.summary(e.options[0], sentences=3)
                source = "Wikipedia"
            except Exception:
                pass
    except Exception:
        pass

    # 2. Try Turkish Wikipedia (for Turkish names like "Koru Tıp Merkezi")
    if not summary:
        try:
            wikipedia.set_lang("tr")
            summary = wikipedia.summary(name, sentences=3)
            source = "Wikipedia (TR)"
        except wikipedia.exceptions.DisambiguationError as e:
            if e.options:
                try:
                    summary = wikipedia.summary(e.options[0], sentences=3)
                    source = "Wikipedia (TR)"
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            wikipedia.set_lang("en")

    # 3. Fallback: DuckDuckGo with multiple query strategies
    if not summary and DDGS_AVAILABLE:
        queries = [
            f'"{name}"',  # exact match
            f"{name} {ENTITY_LABEL_DISPLAY.get(label, '')}".strip(),
            name,
        ]
        for query in queries:
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                if results:
                    snippets = [r.get("body", "") for r in results if r.get("body")]
                    if snippets:
                        summary = " ".join(snippets[:3])
                        source = "Web Search"
                        break
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed for '{name}': {e}")
                continue

    if not summary:
        summary = "No information available."
        source = "N/A"

    return {
        "label": label,
        "label_display": ENTITY_LABEL_DISPLAY.get(label, label),
        "summary": summary,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_text(text: str, direction: str, deepl_key: str | None = None) -> dict:
    """Full analysis pipeline: terms + entities."""
    if not nlp:
        return {"error": "spaCy model not loaded. Please install en_core_web_sm."}

    translate_fn, engine = build_translator(direction, deepl_key)
    # Build a meaning translator (always EN->TR for meanings)
    meaning_translate_fn, _ = build_translator("en-tr", deepl_key)
    doc = nlp(text)

    # Build set of token indices that belong to entities (to avoid overlap)
    entity_spans = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            entity_spans.add(i)

    # --- Term extraction (single-word nouns) ---
    terms = {}
    term_originals = {}  # lemma -> set of original surface forms
    for token in doc:
        if is_term(token, entity_spans):
            lemma = token.lemma_.lower()
            # Track original word forms for highlighting
            if lemma not in term_originals:
                term_originals[lemma] = set()
            term_originals[lemma].add(token.text)
            if lemma not in terms:
                sentence = get_sentence_for_token(token)
                translations = get_translations(lemma, translate_fn)
                meanings_en = get_context_aware_meanings(lemma, sentence)
                meanings_tr = []
                for m in meanings_en:
                    try:
                        tr_def = meaning_translate_fn(m["definition"])
                    except Exception:
                        tr_def = m["definition"]
                    meanings_tr.append({
                        "definition": tr_def if tr_def else m["definition"],
                        "is_primary": m["is_primary"],
                    })
                terms[lemma] = {
                    "context": sentence,
                    "translations": translations,
                    "meanings_en": meanings_en,
                    "meanings_tr": meanings_tr,
                    "originals": [],
                }
    # Attach original forms
    for lemma, forms in term_originals.items():
        if lemma in terms:
            terms[lemma]["originals"] = list(forms)

    # --- Multi-word terms (noun chunks, hyphenated, quoted) ---
    multiword = extract_multiword_terms(doc, entity_spans)
    for comp in multiword:
        word = comp["text"].lower()
        if word not in terms:
            sentence = comp["sentence"]
            translations = get_translations(word, translate_fn)
            meanings_en = get_context_aware_meanings(word, sentence)
            meanings_tr = []
            for m in meanings_en:
                try:
                    tr_def = meaning_translate_fn(m["definition"])
                except Exception:
                    tr_def = m["definition"]
                meanings_tr.append({
                    "definition": tr_def if tr_def else m["definition"],
                    "is_primary": m["is_primary"],
                })
            terms[word] = {
                "context": sentence,
                "translations": translations,
                "meanings_en": meanings_en,
                "meanings_tr": meanings_tr,
                "originals": [comp["text"]],
            }

    # --- Entity extraction with false-positive filtering ---
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            name = ent.text.strip()
            if len(name) < 2 or name in entities:
                continue
            # Filter false positives
            if not is_valid_entity(name, ent.label_):
                logger.info(f"Filtered false-positive entity: '{name}' ({ent.label_})")
                continue
            entities[name] = get_entity_summary(name, ent.label_)

    terms = dict(sorted(terms.items()))

    return {
        "source_text": text,
        "terms": terms,
        "entities": entities,
        "stats": {
            "total_terms": len(terms),
            "total_entities": len(entities),
            "translation_engine": engine,
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload PDF, DOCX, or TXT."}), 400

    direction = request.form.get("direction", "en-tr")
    deepl_key = request.form.get("deepl_key", "").strip() or None

    try:
        text = extract_text_from_file(file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return jsonify({"error": "Failed to read the file. Please try a different format."}), 500

    if not text:
        return jsonify({"error": "The uploaded file appears to be empty."}), 400

    try:
        result = analyze_text(text, direction, deepl_key)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": "An error occurred during analysis. Please try again."}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  SPY AI — Translation Pre-Research Assistant")
    logger.info("  Starting on http://localhost:5000")
    logger.info("=" * 50)
    app.run(debug=True, port=5000)
