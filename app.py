"""
Spy AI — Translation Pre-Research Assistant
Flask backend for analyzing source texts and providing context-aware
translations, meanings, and entity summaries.
"""

import os
import re
import logging
import sqlite3
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

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
from nltk.stem import PorterStemmer

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
                  "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "brown"]:
    nltk.download(resource, quiet=True)

from nltk.corpus import brown
logger.info("Initializing COMMON_WORDS filter...")
try:
    _brown_words = [w.lower() for w in brown.words() if w.isalpha()]
    _freq = nltk.FreqDist(_brown_words)
    # The Top 5000 most common English words to filter out
    COMMON_WORDS = set([w for w, f in _freq.most_common(5000)])
    logger.info(f"Filter active: {len(COMMON_WORDS)} common words excluded.")
except Exception as e:
    logger.warning(f"Brown filter init failed: {e}")
    COMMON_WORDS = set()

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

wikipedia.set_lang("en")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spy-ai-super-secret-key-123")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, identifier, name, role):
        self.id = id
        self.identifier = identifier
        self.name = name
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    u = cache.get_user_by_id(user_id)
    if u:
        return User(u[0], u[1], u[2], u[4])
    return None

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
# Domain-Specific Translation Overrides
# These bypass the translation engine for known medical/dermatology terms.
# ---------------------------------------------------------------------------
TRANSLATION_OVERRIDES = {
    "christmas tree rash": ["Madalyon Hastalığı", "Gül Hastalığı"],
    "mother patch":  ["birincil lezyon", "ilk lezyon", "madalyon plak", "primer plak"],
    "herald patch":  ["haberci plak", "öncü plak"],
    "daughter patch": ["ikincil lezyon", "artçı plak", "sekonder plak"],
    "patch":          ["plak", "lezyon", "yama"],
    "pityriasis rosea": ["Pitiriyazis Rozea"],
    "skin rash": ["deri döküntüsü", "cilt döküntüsü"],
    "abdomen": ["karın", "batın"],
    "scalp": ["kafa derisi", "saç derisi"],
    "headache": ["baş ağrısı"],
    "fatigue": ["yorgunluk", "tükenmişlik", "bitkinlik"], 
    "soles": ["ayak tabanları"],

}

# ---------------------------------------------------------------------------
# Domain-Specific Definition Overrides
# These bypass WordNet and Wikipedia for specific terms/entities.
# ---------------------------------------------------------------------------
DEFINITION_OVERRIDES = {
    "mother patch": "The initial, large, oval-shaped patch that appears during the first stage of pityriasis rosea, typically on the chest, back, or abdomen.",
    "herald patch": "Another name for the mother patch; the first clinical sign of pityriasis rosea, usually measuring 2 to 10 centimeters.",
    "daughter patch": "Smaller secondary lesions that appear in stages after the initial herald patch, often following skin cleavage lines.",
    "pityriasis rosea": "A common, self-limiting skin condition characterized by a herald patch followed by a widespread 'Christmas tree' distribution of smaller lesions.",
    "christmas tree rash": "A descriptive name for pityriasis rosea, referring to the characteristic pattern the secondary lesions form on the back, resembling the branches of a fir tree or a medallion.",
    "american academy of dermatology": "The American Academy of Dermatology (AAD) is a non-profit professional organization of dermatologists in the United States and Canada, based in Rosemont, Illinois, near Chicago. It was founded in 1938 and has more than 21,000 members. The academy grants fellowships and associate memberships, as well as fellowships for nonresidents of the United States or Canada.",
    "the american academy of dermatology": "The American Academy of Dermatology (AAD) is a non-profit professional organization of dermatologists in the United States and Canada, based in Rosemont, Illinois, near Chicago. It was founded in 1938 and has more than 21,000 members. The academy grants fellowships and associate memberships, as well as fellowships for nonresidents of the United States or Canada.",
    "Skin Dermatology": "Dermatology is the branch of medicine that focuses on the diagnosis, treatment, and prevention of diseases and conditions affecting the skin, hair, nails, and mucous membranes.",
    "soles": ["the underside of the foot from the heel to the toes", "the bottom of a shoe"],
    "headache": ["pain in the head caused by dilation of cerebral arteries or muscle contractions or a reaction to drugs", "something or someone that causes anxiety; a source of unhappiness."],
    "patch": ["a small area of skin that is different from the skin around it", "a piece of material used to mend or cover a hole"],
    "board-certified": ["kurul onaylı", "sertifikalı"]
}

class SpyAICache:
    _lock = threading.Lock()

    def __init__(self, db_path="spy_ai_cache.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, timeout=30)
        return self._local.conn

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, value TEXT, category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT UNIQUE,
                    name TEXT,
                    password_hash TEXT,
                    role TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    test_id TEXT,
                    score INTEGER,
                    total_questions INTEGER,
                    answers_json TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER UNIQUE,
                    test_id TEXT,
                    question_index INTEGER DEFAULT 0,
                    total_questions INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.commit()
            conn.close()

    def get_user_by_identifier(self, identifier):
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("SELECT * FROM users WHERE identifier=?", (identifier,))
            return cursor.fetchone()
        except: return None

    def get_user_by_id(self, user_id):
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
            return cursor.fetchone()
        except: return None

    def create_user(self, identifier, name, password_hash, role):
        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (identifier, name, password_hash, role) VALUES (?, ?, ?, ?)",
                               (identifier, name, password_hash, role))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                return None

    def save_quiz_result(self, user_id, test_id, score, total, answers_json):
        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO quiz_results (user_id, test_id, score, total_questions, answers_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, test_id, score, total, answers_json))
                conn.commit()
            except Exception as e:
                logger.error(f"Error saving quiz result: {e}")

    def get_all_results(self):
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("""
                SELECT users.identifier, users.name, quiz_results.test_id, 
                       quiz_results.score, quiz_results.total_questions, 
                       quiz_results.timestamp, quiz_results.answers_json
                FROM quiz_results
                JOIN users ON users.id = quiz_results.user_id
                ORDER BY quiz_results.timestamp DESC
            """)
            return cursor.fetchall()
        except: return []

    def get_user_result(self, user_id, test_id):
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("""
                SELECT id FROM quiz_results WHERE user_id=? AND test_id=?
            """, (user_id, test_id))
            return cursor.fetchone()
        except: return None

    def upsert_progress(self, user_id, test_id, question_index, total):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                conn.execute("""
                    INSERT INTO quiz_progress (user_id, test_id, question_index, total_questions, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                        test_id=excluded.test_id,
                        question_index=excluded.question_index,
                        total_questions=excluded.total_questions,
                        updated_at=CURRENT_TIMESTAMP
                """, (user_id, test_id, question_index, total))
                conn.commit()
                conn.close()
        except: pass

    def get_all_progress(self):
        try:
            cursor = self._get_conn().cursor()
            cursor.execute("""
                SELECT users.name, users.identifier, qp.test_id,
                       qp.question_index, qp.total_questions, qp.updated_at
                FROM quiz_progress qp
                JOIN users ON users.id = qp.user_id
                ORDER BY qp.updated_at DESC
            """)
            return cursor.fetchall()
        except: return []

    def delete_progress(self, user_id):
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                conn.execute("DELETE FROM quiz_progress WHERE user_id=?", (user_id,))
                conn.commit()
                conn.close()
        except: pass

    def get(self, category, key):
        try:
            cursor = self._get_conn().cursor()
            # FIX: Match the composite key stored in the DB
            composite_key = f"{category}:{key}"
            cursor.execute("SELECT value FROM cache WHERE key=?", (composite_key,))
            res = cursor.fetchone()
            return json.loads(res[0]) if res else None
        except: return None

    def set(self, category, key, value):
        with self._lock:
            try:
                conn = self._get_conn()
                cursor = conn.cursor()
                composite_key = f"{category}:{key}"
                cursor.execute("INSERT OR REPLACE INTO cache (key, value, category) VALUES (?, ?, ?)",
                               (composite_key, json.dumps(value), category))
                conn.commit()
            except: pass

cache = SpyAICache()

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
    """Single-word term detection (nouns and adjectives)."""
    if token.pos_ not in ("NOUN", "ADJ"):
        return False
    if len(token.text) < 4:
        return False
    if token.is_stop or token.is_punct or token.is_space or token.like_num:
        return False
    if not token.is_alpha:
        return False
    if token.i in entity_spans:
        return False

    # RESEARCH ENGINE OVERHAUL: Filter out the Top 5000 most common English words
    lemma = resolve_lemma(token)
    if lemma in COMMON_WORDS or token.text.lower() in COMMON_WORDS:
        return False

    wn_pos = spacy_pos_to_wn(token.pos_)
    if not wn.synsets(lemma, pos=wn_pos):
        return False
    return True


def extract_multiword_terms(doc, entity_spans: set) -> list:
    """
    Extract multi-word terms using spaCy noun chunks, hyphenated compounds,
    quoted terms, and WordNet-verified NOUN+NOUN or ADJ+NOUN compounds.
    """
    terms = []
    seen = set()

    # 1. Quoted terms (highest priority) - Use only double quotes to avoid contractions
    quoted = re.findall(r'[“”"](\w+(?:\s+\w+)*)[“”"]', doc.text)
    for q in quoted:
        if len(q) >= 3:
            # Find the sentence containing this text
            match = re.search(re.escape(q), doc.text)
            sentence = ""
            if match:
                start = match.start()
                # Find sentence boundary around start
                s_start = doc.text.rfind(".", 0, start) + 1
                s_end = doc.text.find(".", start) + 1
                sentence = doc.text[s_start:s_end].strip()
            terms.append({"text": q, "sentence": sentence})
            seen.add(q.lower())

    # 2. spaCy noun chunks (multi-word noun phrases)
    for chunk in doc.noun_chunks:
        # Skip chunks that are single common words or overlap entities
        tokens = [t for t in chunk if not t.is_stop and not t.is_punct
                  and not t.is_space and t.pos_ in ("NOUN", "ADJ", "PROPN")]
        if len(tokens) < 2:
            continue
        # Skip if any token overlaps with named entities
        if any(t.i in entity_spans for t in tokens):
            continue
        
        # Stricter research-engine check: must end in NOUN or PROPN
        if tokens[-1].pos_ not in ("NOUN", "PROPN"):
            continue

        # Limit length to avoid long noisy phrases like "skin condition experience itchiness"
        # Most valid multi-word terms are 2-3 words.
        if len(tokens) > 3:
            # Only keep if it's a valid WordNet compound
            full_text = "_".join(t.text.lower() for t in tokens)
            if not wn.synsets(full_text):
                continue

        text = " ".join(t.text for t in tokens).strip()
        if len(text) < 5 or text.lower() in seen:
            continue
        seen.add(text.lower())
        sentence = chunk.sent.text.strip() if chunk.sent else ""
        terms.append({"text": text, "sentence": sentence})

    # 3. Hyphenated compounds
    hyphenated = re.findall(r'\b([a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*)\b', doc.text)
    for h in set(hyphenated):
        if len(h) > 5 and h.lower() not in seen:
            seen.add(h.lower())
            for sent in doc.sents:
                if h in sent.text:
                    terms.append({"text": h, "sentence": sent.text.strip()})
                    break

    return terms


def get_sentence_for_token(token) -> str:
    """Safely get the text of the sentence containing a token."""
    return token.sent.text.strip() if token.sent else ""


# ---------------------------------------------------------------------------
# Helpers — Better WSD for Compounds
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers — POS Mapping & Stemmer
# ---------------------------------------------------------------------------

_stemmer = PorterStemmer()


def spacy_pos_to_wn(pos_tag: str):
    """Map a spaCy POS tag to the equivalent WordNet POS constant."""
    return {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}.get(pos_tag)


def resolve_lemma(token) -> str:
    """Return the best lemma for a token.
    If spaCy's lemma has fewer WordNet synsets than the original surface form,
    prefer the surface form (e.g. 'sole' instead of 'sol')."""
    spacy_lemma = token.lemma_.lower()
    surface_form = token.text.lower()
    if spacy_lemma == surface_form:
        return spacy_lemma

    wn_pos = spacy_pos_to_wn(token.pos_)
    lemma_synsets = wn.synsets(spacy_lemma, pos=wn_pos)
    surface_synsets = wn.synsets(surface_form, pos=wn_pos)

    if len(surface_synsets) > len(lemma_synsets):
        return surface_form
    return spacy_lemma


def _stem_tokens(text: str) -> set:
    """Tokenize and stem a string, returning a set of stems."""
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    return {_stemmer.stem(t) for t in tokens if t.isalnum()}


def get_context_aware_meanings(word: str, sentence: str, wn_pos=None, limit: int = 3):
    """
    Enhanced Lesk Algorithm:
    Disambiguates word senses by comparing stems of the context sentence
    against stems of definitions, examples, and hypernym definitions.
    Applies frequency bias for more accurate results.
    """
    # 1. Check for custom definition overrides FIRST (bypasses cache)
    override_key = word.lower().strip()
    if override_key in DEFINITION_OVERRIDES:
        val = DEFINITION_OVERRIDES[override_key]
        if isinstance(val, list):
            return [{"definition": d, "is_primary": i == 0} for i, d in enumerate(val)]
        return [{"definition": val, "is_primary": True}]

    # 2. Check cache
    cache_key = f"{word}:{sentence[:100]}:{wn_pos}"
    cached = cache.get("term_meanings", cache_key)
    if cached: return cached

    meanings = []
    lookup_word = word.replace(" ", "_").replace("-", "_")
    all_synsets = wn.synsets(lookup_word, pos=wn_pos)
    if not all_synsets:
        all_synsets = wn.synsets(lookup_word)

    if not all_synsets:
        return [{"definition": "No definition available.", "is_primary": True}]

    context_stems = _stem_tokens(sentence)
    scored_senses = []

    # Scoring with Medical/Scientific Bias
    # If the sentence contains medical keywords, boost medical senses
    medical_context = any(kw in sentence.lower() for kw in 
                         ["skin", "patient", "disease", "treatment", "medical", "clinical", "symptom", "rash", "pain"])

    for i, syn in enumerate(all_synsets):
        defn = syn.definition().lower()
        signature = _stem_tokens(defn)
        for ex in syn.examples():
            signature.update(_stem_tokens(ex))
        for hyper in syn.hypernyms():
            signature.update(_stem_tokens(hyper.definition()))

        overlap = len(context_stems.intersection(signature))
        freq_bias = 1.0 / (i + 1)
        
        # Medical Bias: prioritize senses whose definitions contain medical terms
        med_bias = 0
        if medical_context:
            med_kws = ["disease", "disorder", "medical", "condition", "inflammation", "anatomy", "tissue", "body", "pathological"]
            if any(kw in defn for kw in med_kws):
                med_bias = 2.0 # Significant boost

        score = overlap + freq_bias + med_bias
        scored_senses.append((score, syn))

    # Sort by score descending
    scored_senses.sort(key=lambda x: x[0], reverse=True)
    best_syns = [s[1] for s in scored_senses]

    for i, syn in enumerate(best_syns[:limit]):
        meanings.append({
            "definition": syn.definition(),
            "is_primary": i == 0,
        })

    cache.set("term_meanings", cache_key, meanings)
    return meanings


def get_contextual_translation(word: str, sentence: str, translate_fn):
    """
    Translates a word within its context sentence to ensure correct POS and sense.
    Uses markers [[word]] to identify the target in the translated output.
    """
    cache_key = f"{word}:{sentence[:100]}"
    cached = cache.get("context_trans", cache_key)
    if cached: return cached

    try:
        # Wrap the word in markers within the sentence
        # Example: "The patient has a [[rash]] on his back."
        marked_sentence = sentence.replace(word, f"[[{word}]]", 1)
        if "[[" not in marked_sentence:
            # Fallback if literal match failed (e.g. case difference)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            marked_sentence = pattern.sub(f"[[{word}]]", sentence, count=1)

        tr_sentence = translate_fn(marked_sentence)
        
        # Extract the marked word from the translation
        # Example: "Hastanın sırtında bir [[döküntü]] var."
        match = re.search(r"\[\[(.*?)\]\]", tr_sentence)
        if match:
            result = match.group(1).strip().lower()
            cache.set("context_trans", cache_key, result)
            return result
    except:
        pass

    # Final fallback: translate isolated word
    res = translate_fn(word).strip().lower()
    cache.set("context_trans", cache_key, res)
    return res


def get_translations(word: str, sentence: str, translate_fn):
    """Get high-quality translations using contextual disambiguation."""
    # Check domain-specific translation overrides first
    override_key = word.lower().strip()
    if override_key in TRANSLATION_OVERRIDES:
        return list(TRANSLATION_OVERRIDES[override_key])

    translations = []
    
    # 1. Primary: Contextual translation (the most accurate)
    primary = get_contextual_translation(word, sentence, translate_fn)
    if primary:
        translations.append(primary)

    # 2. Secondary: WordNet Synonyms are often too noisy for TR, 
    # so we only add the isolated translation if it differs
    isolated = translate_fn(word).strip().lower()
    if isolated and isolated not in translations:
        translations.append(isolated)

    # Dedup and limit
    seen = set()
    final = []
    for t in translations:
        if t not in seen and len(final) < 3:
            final.append(t)
            seen.add(t)

    return final if final else ["Translation unavailable"]


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
    # We want to keep entities even if they are dictionary words (e.g. Apple, Amazon)
    # BUT we want to filter out clear medical terms that spaCy mislabels as ORG/PRODUCT
    words = name_stripped.split()
    if len(words) == 1:
        # If it's a single word and it's lowercase in the text, it's likely a false positive
        if name_stripped[0].islower():
            return False
        
        # Check if it's a medical term mislabeled as ORG
        synsets = wn.synsets(name_stripped.lower())
        if synsets:
            for s in synsets:
                defn = s.definition().lower()
                medical_kws = ["disease", "condition", "disorder", "inflammation",
                               "infection", "syndrome", "tissue", "rash"]
                if any(kw in defn for kw in medical_kws):
                    # Only filter if it's NOT capitalized in a way that suggests a proper noun
                    return False
    
    # For multi-word ORGs, we generally trust them unless they are all lowercase
    if all(w[0].islower() for w in words):
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
    # 1. Check for manual definition overrides first (bypasses Wikipedia/Cache)
    override_key = name.lower().strip()
    if override_key in DEFINITION_OVERRIDES:
        val = DEFINITION_OVERRIDES[override_key]
        # Handle list of definitions if provided for an entity
        summary_text = val[0] if isinstance(val, list) else val
        return {
            "label": label,
            "label_display": ENTITY_LABEL_DISPLAY.get(label, label),
            "summary": summary_text,
            "source": "Custom Definition",
        }

    summary = ""
    source = ""

    # 2. Try English Wikipedia
    if not summary:
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

    # 2. Wikipedia (Localized Fallback) - ONLY if primary failed and entity seems Turkish
    if not summary:
        # Check if name contains Turkish characters
        is_turkish = any(c in name for c in "ıİğĞüÜşŞöÖçÇ")
        if is_turkish:
            try:
                wikipedia.set_lang("tr")
                summary = wikipedia.summary(name, sentences=2)
                source = "Wikipedia (TR)"
            except Exception:
                pass
            finally:
                wikipedia.set_lang("en")

    # 3. NIH (National Institutes of Health) Fallback for medical terms
    if not summary and DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                # Targeted search on NIH site
                results = list(ddgs.text(f"site:nih.gov {name}", max_results=3))
                if results:
                    snippets = [r.get("body", "") for r in results if r.get("body")]
                    if snippets:
                        summary = " ".join(snippets[:2])
                        source = "NIH (National Institutes of Health)"
        except Exception:
            pass

    # 4. Fallback: DuckDuckGo (force English region)
    if not summary and DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f'"{name}"', region='en-us', max_results=3))
            if results:
                snippets = [r.get("body", "") for r in results if r.get("body")]
                if snippets:
                    summary = " ".join(snippets[:3])
                    source = "Web Search"
        except Exception:
            pass

    # 5. Fallback: WordNet Dictionary
    if not summary:
        try:
            lookup = name.lower().replace(" ", "_")
            synsets = wn.synsets(lookup)
            if synsets:
                summary = synsets[0].definition()
                source = "WordNet Dictionary"
        except Exception:
            pass

    # 6. Final fallback: Deep Browser Research
    if not summary and DDGS_AVAILABLE:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(f'{name} official definition explanation', max_results=5))
                if results:
                    snippets = [r.get("body", "") for r in results if r.get("body")]
                    if snippets:
                        summary = " ".join(snippets[:3])
                        source = "Deep Research"
        except Exception:
            pass

    if not summary:
        summary = "No information available."
        source = "N/A"

    result = {
        "label": label,
        "label_display": ENTITY_LABEL_DISPLAY.get(label, label),
        "summary": summary,
        "source": source,
    }
    cache.set("entity", name, result)
    return result


# ---------------------------------------------------------------------------
# Streaming Analysis Pipeline
# ---------------------------------------------------------------------------

def stream_analysis(text: str, direction: str, deepl_key: str | None = None):
    def send(event_type, payload):
        return f"data: {json.dumps({'type': event_type, 'payload': payload})}\n\n"
    
    # Buffer-busting: send 4KB of whitespace in an SSE comment to force flushing through proxies
    yield f": {' ' * 4096}\n\n"

    if not nlp:
        yield send("error", "spaCy model not loaded. Please install en_core_web_sm.")
        return

    logger.info("Starting streaming analysis...")
    yield send("status", "Starting analysis...")

    translate_fn, engine = build_translator(direction, deepl_key)
    meaning_translate_fn, _ = build_translator("en-tr", deepl_key)
    doc = nlp(text)

    entity_spans = set()
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            entity_spans.add(i)

    yield send("status", "Extracting terms...")
    
    # RESEARCH ENGINE: Targeted term items
    term_items = []
    
    for token in doc:
        if is_term(token, entity_spans):
            lemma = resolve_lemma(token)
            term_items.append({
                "lemma": lemma,
                "sentence": get_sentence_for_token(token),
                "original": token.text,
                "wn_pos": spacy_pos_to_wn(token.pos_)
            })

    multiword = extract_multiword_terms(doc, entity_spans)
    for comp in multiword:
        term_items.append({
            "lemma": comp["text"].lower(),
            "sentence": comp["sentence"],
            "original": comp["text"],
            "wn_pos": None # Multi-word lookup usually defaults to None
        })

    # Deduplicate by lemma
    seen_lemmas = {}
    for item in term_items:
        l = item["lemma"]
        if l not in seen_lemmas:
            seen_lemmas[l] = item
            seen_lemmas[l]["originals"] = {item["original"]}
        else:
            seen_lemmas[l]["originals"].add(item["original"])

    # Inject forced override terms — ALWAYS present when their component
    # words exist anywhere in the source text.  Uses simple substring
    # matching on the full text (not regex) to avoid smart-quote /
    # sentence-segmentation edge cases.
    text_lower = text.lower()
    for override_key in TRANSLATION_OVERRIDES:
        if override_key not in seen_lemmas:
            words = override_key.split()
            # Check full text for ALL component words (simple, robust)
            if all(w in text_lower for w in words):
                # Try to find a sentence that contains ALL words (best context)
                found_sentence = ""
                for sent in doc.sents:
                    sl = sent.text.lower()
                    if all(w in sl for w in words):
                        found_sentence = sent.text.strip()
                        break
                # Fallback: any sentence that contains at least one word
                if not found_sentence:
                    for sent in doc.sents:
                        if any(w in sent.text.lower() for w in words):
                            found_sentence = sent.text.strip()
                            break
                # Always inject — even with empty context
                seen_lemmas[override_key] = {
                    "lemma": override_key,
                    "sentence": found_sentence,
                    "original": override_key,
                    "wn_pos": None,
                    "originals": {override_key},
                }

    # Suppress standalone terms that are mere components of compound
    # overrides.  e.g. "herald" alone must not appear because
    # "herald patch" already exists as a compound term.
    # Words that are themselves override keys (like "patch") are kept.
    _parts_to_suppress = set()
    for key in TRANSLATION_OVERRIDES:
        if " " in key and key in seen_lemmas:
            for w in key.split():
                if w not in TRANSLATION_OVERRIDES:
                    _parts_to_suppress.add(w)
    for w in _parts_to_suppress:
        seen_lemmas.pop(w, None)

    final_terms = list(seen_lemmas.values())

    entities_to_research = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            name = ent.text.strip()
            if len(name) >= 2 and is_valid_entity(name, ent.label_):
                if not any(e["name"] == name for e in entities_to_research):
                    entities_to_research.append({"name": name, "label": ent.label_})

    yield send("meta", {
        "source_text": text,
        "total_terms": len(final_terms),
        "total_entities": len(entities_to_research),
        "engine": engine
    })

    yield send("status", "Processing terms...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        def process_term(item):
            word = item["lemma"]
            sentence = item["sentence"]
            translations = get_translations(word, sentence, translate_fn)
            meanings_en = get_context_aware_meanings(word, sentence, item["wn_pos"])
            meanings_tr = []
            for m in meanings_en:
                try: tr_def = meaning_translate_fn(m["definition"])
                except Exception: tr_def = m["definition"]
                meanings_tr.append({
                    "definition": tr_def if tr_def else m["definition"],
                    "is_primary": m["is_primary"],
                })
            return {
                "lemma": word,
                "context": sentence,
                "translations": translations,
                "meanings_en": meanings_en,
                "meanings_tr": meanings_tr,
                "originals": list(item["originals"])
            }
        
        futures = [executor.submit(process_term, item) for item in final_terms]
        for future in futures:
            yield send("term", future.result())

    yield send("status", "Researching entities...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        def process_ent(ent):
            return {"name": ent["name"], "summary": get_entity_summary(ent["name"], ent["label"])}
        
        futures = [executor.submit(process_ent, ent) for ent in entities_to_research]
        for future in futures:
            yield send("entity", future.result())

    yield send("done", "Analysis complete.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Routes — Authentication
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        role = request.form.get("role")
        name = request.form.get("name")
        identifier = request.form.get("identifier") # ID for student, Name for teacher
        password = request.form.get("password")

        if role == "student":
            if not name or not identifier:
                flash("Name and Student ID are required.")
                return render_template("login.html")
            
            # Check if student exists, else create
            u = cache.get_user_by_identifier(identifier)
            if not u:
                user_id = cache.create_user(identifier, name, None, "student")
                u = cache.get_user_by_id(user_id)
            else:
                # Validate name matches — prevent account hijacking via shared student ID
                if u[2] and u[2].strip().lower() != name.strip().lower():
                    flash("This Student ID is already registered to a different name. Please check your ID.")
                    return render_template("login.html")
            
            user_obj = User(u[0], u[1], u[2], u[4])
            login_user(user_obj)
            return redirect(url_for("index"))
        
        else: # Teacher
            if not identifier or not password:
                flash("Name and Password are required.")
                return render_template("login.html")
            
            u = cache.get_user_by_identifier(identifier)
            if u and u[3] and check_password_hash(u[3], password):
                user_obj = User(u[0], u[1], u[2], u[4])
                login_user(user_obj)
                return redirect(url_for("index"))
            else:
                flash("Invalid credentials.")
    
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        password = request.form.get("password")
        if not name or not password:
            flash("Name and Password are required.")
            return render_template("register.html")
        
        hashed = generate_password_hash(password)
        if cache.create_user(name, name, hashed, "teacher"):
            flash("Teacher registered successfully! Please login.")
            return redirect(url_for("login"))
        else:
            flash("Username already exists.")
            
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------------------------------------------------------------------
# Routes — Core
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user)

@app.route("/teacher/dashboard")
@login_required
def dashboard():
    if current_user.role != "teacher":
        return redirect(url_for("index"))
    
    results = cache.get_all_results()
    return render_template("dashboard.html", results=results)

@app.route("/api/save_result", methods=["POST"])
@login_required
def save_result():
    data = request.json
    cache.save_quiz_result(
        current_user.id,
        data.get("test_id"),
        data.get("score"),
        data.get("total"),
        json.dumps(data.get("answers"))
    )
    return jsonify({"success": True})


@app.route("/api/has_taken/<test_id>", methods=["GET"])
@login_required
def has_taken(test_id):
    result = cache.get_user_result(current_user.id, test_id)
    return jsonify({"taken": result is not None})


@app.route("/api/update_progress", methods=["POST"])
@login_required
def update_progress():
    data = request.json
    cache.upsert_progress(
        current_user.id,
        data.get("test_id"),
        data.get("question_index", 0),
        data.get("total", 0)
    )
    return jsonify({"success": True})


@app.route("/api/student_progress", methods=["GET"])
@login_required
def student_progress():
    if current_user.role != "teacher":
        return jsonify({"error": "Forbidden"}), 403
    rows = cache.get_all_progress()
    return jsonify([{
        "name": r[0], "identifier": r[1], "test_id": r[2],
        "question_index": r[3], "total": r[4], "updated_at": r[5]
    } for r in rows])


@app.route("/api/clear_progress", methods=["POST"])
@login_required
def clear_progress():
    cache.delete_progress(current_user.id)
    return jsonify({"success": True})


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

    return Response(
        stream_with_context(stream_analysis(text, direction, deepl_key)),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  SPY AI — Translation Pre-Research Assistant")
    logger.info("  Starting on http://localhost:5000")
    logger.info("=" * 50)
    app.run(debug=True, port=5000)
