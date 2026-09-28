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
import time
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

import spacy
from deep_translator import GoogleTranslator
try:
    from deep_translator import DeeplTranslator
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False

# Optional: Gemini API for top-tier linguistic analysis
try:
    from google import genai
    _gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if _gemini_api_key:
        _gemini_client = genai.Client(api_key=_gemini_api_key)
        GEMINI_AVAILABLE = True
    else:
        _gemini_client = None
        GEMINI_AVAILABLE = False
except ImportError:
    _gemini_client = None
    GEMINI_AVAILABLE = False

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
except (OSError, ImportError):
    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        logger.info("spaCy model loaded via direct en_core_web_sm import.")
    except Exception:
        logger.warning("Attempting automatic download of en_core_web_sm...")
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            nlp = None

wikipedia.set_lang("en")
try:
    wikipedia.set_user_agent("SpyAI/2.0 (student-assistant; mailto:admin@spyai.com)")
except Exception:
    pass

app = Flask(__name__)
# Enable ProxyFix for Railway / reverse proxy SSL termination and headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spy-ai-super-secret-key-123")

# Secure cookies in production environments (Railway / Render / HTTPS), disable for local development
is_production = os.environ.get("RAILWAY_ENVIRONMENT") is not None or os.environ.get("RENDER") is not None or os.environ.get("FLASK_ENV") == "production"
app.config["SESSION_COOKIE_SECURE"] = is_production
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.session_protection = "strong" # Bind session to IP/UA
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id, identifier, name, role):
        self.id = id
        self.identifier = identifier
        self.name = name
        self.role = role

class GuestUser:
    """Lightweight guest user that doesn't require DB storage."""
    is_authenticated = True
    is_active = True
    is_anonymous = False
    id = "guest"
    identifier = "guest"
    name = "Guest"
    role = "guest"
    def get_id(self):
        return "guest"

@login_manager.user_loader
def load_user(user_id):
    if user_id == "guest":
        return GuestUser()
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
    "NORP": "Group/Nationality"
}

@app.after_request
def add_header(response):
    # Prevent caching of all responses to ensure sessions don't get mixed up by proxies
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    # Crucial: Tell CDNs that the response varies by the cookie
    response.headers["Vary"] = "Cookie"
    return response

# ---------------------------------------------------------------------------
# Domain-Specific Translation Overrides
# Removed hardcoded lists - system dynamically handles all domains.
# ---------------------------------------------------------------------------
TRANSLATION_OVERRIDES = {}

# ---------------------------------------------------------------------------
# Domain-Specific Definition Overrides
# Removed hardcoded lists - system dynamically handles all domains.
# ---------------------------------------------------------------------------
DEFINITION_OVERRIDES = {}


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
                ORDER BY quiz_results.score DESC, quiz_results.timestamp DESC
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
# Helpers — Robust Translation & Dictionary Client (Google GTX)
# ---------------------------------------------------------------------------

class GoogleGTXClient:
    """
    High-reliability client for contextual translation, bilingual dictionary synonyms,
    and Oxford English definitions. Uses connection pooling, rate-limiting, retries,
    and caching to prevent 'Translation unavailable' errors.
    """
    _lock = threading.Lock()
    _session = None
    _last_req_time = 0.0

    @classmethod
    def get_session(cls):
        if cls._session is None:
            with cls._lock:
                if cls._session is None:
                    cls._session = requests.Session()
                    cls._session.headers.update({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                        "Accept": "*/*",
                    })
        return cls._session

    @classmethod
    def _rate_limit(cls):
        with cls._lock:
            now = time.time()
            elapsed = now - cls._last_req_time
            if elapsed < 0.10:
                time.sleep(0.10 - elapsed)
            cls._last_req_time = time.time()

    @classmethod
    def query_raw(cls, text: str, sl: str = "en", tl: str = "tr"):
        if not text or not text.strip():
            return None
        clean_text = text.strip()
        cache_key = f"gtx_raw:{sl}:{tl}:{clean_text.lower()}"
        cached = cache.get("gtx_raw", cache_key)
        if cached:
            return cached

        session = cls.get_session()
        endpoints = [
            f"https://clients5.google.com/translate_a/single?client=dict-chrome-ex&sl={sl}&tl={tl}&dt=t&dt=bd&dt=md&dt=ss&dt=ex&q={requests.utils.quote(clean_text)}",
            f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&dt=bd&dt=md&dt=ss&dt=ex&q={requests.utils.quote(clean_text)}",
        ]
        for url in endpoints:
            for attempt in range(2):
                try:
                    cls._rate_limit()
                    resp = session.get(url, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        cache.set("gtx_raw", cache_key, data)
                        return data
                    elif resp.status_code == 429:
                        break  # Rate-limited on this endpoint, try next
                except Exception:
                    pass
        return None

    @classmethod
    def translate(cls, text: str, sl: str = "en", tl: str = "tr") -> str:
        if not text or not text.strip():
            return ""
        clean_text = text.strip()
        cache_key = f"trans:{sl}:{tl}:{clean_text[:120]}"
        cached = cache.get("translations", cache_key)
        if cached:
            return cached

        data = cls.query_raw(clean_text, sl=sl, tl=tl)
        if data and data[0]:
            pieces = [p[0] for p in data[0] if p and p[0]]
            res = "".join(pieces).strip()
            if res:
                cache.set("translations", cache_key, res)
                return res

        # Fallback to GoogleTranslator from deep_translator
        try:
            gt = GoogleTranslator(source=sl, target=tl)
            res = gt.translate(clean_text)
            if res:
                cache.set("translations", cache_key, res)
                return res
        except Exception:
            pass

        return clean_text


# ---------------------------------------------------------------------------
# Helpers — Translation
# ---------------------------------------------------------------------------

def build_translator(direction: str, deepl_key: str | None = None):
    """Return a translator function that tries DeepL first, then GoogleGTXClient."""
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
            logger.warning(f"DeepL init failed ({e}); falling back to Google GTX.")
            deepl_translator = None

    def translate(text: str) -> str:
        if not text or not text.strip():
            return ""
        try:
            if deepl_translator:
                res = deepl_translator.translate(text)
                if res: return res
        except Exception:
            pass
        return GoogleGTXClient.translate(text, sl=src, tl=tgt)

    engine_name = "DeepL" if deepl_translator else "Google Translate"
    return translate, engine_name


# ---------------------------------------------------------------------------
# NEW MODULAR PIPELINE — Imports
# ---------------------------------------------------------------------------

from text_preprocessor import segment_document, get_content_text, extract_source_url, infer_domain_from_url, is_metadata_region
from term_extractor import extract_terms, AnalysisMode
from entity_resolver import detect_entities, research_entity, ENTITY_LABEL_DISPLAY as NEW_ENTITY_LABEL_DISPLAY
from wsd import disambiguate as wsd_disambiguate
from translator import translate_occurrence, translate_senses_to_turkish
from models import Occurrence, EntitySpan, Category, ConfidenceLevel


# ---------------------------------------------------------------------------
# Streaming Analysis Pipeline (v2 — Occurrence-Centric)
# ---------------------------------------------------------------------------

def stream_analysis(text: str, direction: str, deepl_key: str | None = None, mode: str = AnalysisMode.STUDENT):
    """
    New analysis pipeline:
    1. Document segmentation (separate content from metadata/instructions)
    2. spaCy NLP on content only
    3. Named entity detection + span consolidation
    4. Term extraction with proper boundaries and Translation Value Scoring
    5. Per-occurrence WSD
    6. Per-occurrence translation with candidate validation & anti-contamination
    7. Confidence + provenance
    8. Entity research with context-aware disambiguation
    9. Stream results to frontend
    """
    def send(event_type, payload):
        return f"data: {json.dumps({'type': event_type, 'payload': payload})}\n\n"

    # Buffer-busting for proxy compatibility
    yield f": {' ' * 4096}\n\n"

    if not nlp:
        yield send("error", "spaCy model not loaded. Please install en_core_web_sm.")
        return

    logger.info(f"Starting streaming analysis (v2 pipeline, mode={mode})...")
    yield send("status", "Starting analysis...")

    # --- Phase 1: Document Segmentation ---
    yield send("status", "Segmenting document...")
    segments = segment_document(text)
    content_text = get_content_text(segments)
    source_url = extract_source_url(text)
    domain = infer_domain_from_url(source_url) if source_url else None

    if not content_text.strip():
        # If no content detected, use the full text as fallback
        content_text = text

    logger.info(f"Document segmented: {len(segments)} segments, "
                f"{sum(1 for s in segments if s.segment_type == 'content')} content, "
                f"domain={domain}")

    # --- Phase 2: spaCy NLP ---
    yield send("status", "Analyzing text...")
    doc = nlp(content_text)
    sentences = list(doc.sents)

    # --- Phase 3: Named Entity Detection ---
    yield send("status", "Detecting entities...")
    entity_spans, entity_token_indices = detect_entities(doc, segments)

    logger.info(f"Detected {len(entity_spans)} entities")

    # --- Phase 4: Term Extraction ---
    yield send("status", "Extracting terms...")
    occurrences = extract_terms(
        doc,
        entity_token_indices,
        COMMON_WORDS,
        sentences,
        mode=mode,
    )

    logger.info(f"Extracted {len(occurrences)} term occurrences")

    # --- Build translators ---
    translate_fn, engine = build_translator(direction, deepl_key)
    meaning_translate_fn, _ = build_translator("en-tr", deepl_key)

    # --- Send meta event ---
    yield send("meta", {
        "source_text": text,  # Send the FULL text (including metadata) for display
        "total_terms": len(occurrences),
        "total_entities": len(entity_spans),
        "engine": engine,
        "domain": domain,
        "mode": mode,
        "segments": [
            {"type": s.segment_type, "start": s.start_offset, "end": s.end_offset}
            for s in segments
        ],
    })

    # --- Phase 5-7: WSD + Translation (per-occurrence) ---
    yield send("status", "Processing terms...")

    with ThreadPoolExecutor(max_workers=4) as executor:
        def process_occurrence(occ: Occurrence):
            # WSD
            wsd_disambiguate(
                occ,
                gtx_client=GoogleGTXClient,
                gemini_client=_gemini_client if GEMINI_AVAILABLE else None,
                gemini_available=GEMINI_AVAILABLE,
                doc_text=text,
            )

            # Translation with validation layer & anti-contamination guard
            translate_occurrence(
                occ,
                translate_fn=translate_fn,
                gtx_client=GoogleGTXClient,
                gemini_client=_gemini_client if GEMINI_AVAILABLE else None,
                gemini_available=GEMINI_AVAILABLE,
                meaning_translate_fn=meaning_translate_fn,
            )

            # Translate senses to Turkish
            meanings_tr = translate_senses_to_turkish(occ, meaning_translate_fn)

            # Build the occurrence payload for the frontend
            return {
                # Legacy fields (backward compatible)
                "lemma": occ.lemma,
                "context": occ.sentence_text,
                "translations": [t.text for t in occ.translations] if occ.translations else [occ.selected_translation] if occ.selected_translation else [],
                "meanings_en": [
                    {"definition": s.definition, "is_primary": s.is_primary, "source": s.source}
                    for s in occ.candidate_senses
                ] if occ.candidate_senses else [],
                "meanings_tr": meanings_tr,
                "originals": [occ.surface],

                # Occurrence-level fields (v2)
                "id": occ.id,
                "surface": occ.surface,
                "pos": occ.pos,
                "pos_fine": occ.pos_fine,
                "morphology": occ.morphology,
                "syntactic_role": occ.syntactic_role,
                "category": occ.category,
                "translation_value": round(occ.translation_value, 2),
                "extraction_reason": occ.extraction_reason,
                "start_offset": occ.start_offset,
                "end_offset": occ.end_offset,
                "confidence": {k: round(v, 2) for k, v in occ.confidence.items()},
                "confidence_level": occ.confidence_level(),
                "evidence": occ.evidence,
            }

        futures = [(i, executor.submit(process_occurrence, occ)) for i, occ in enumerate(occurrences)]
        # Collect all results first, then yield in original order for deterministic output
        results_by_index = {}
        for i, future in futures:
            try:
                results_by_index[i] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Occurrence processing failed: {e}")
        for i in sorted(results_by_index.keys()):
            yield send("term", results_by_index[i])

    # --- Phase 8: Entity Research ---
    yield send("status", "Researching entities...")

    ddgs_cls = DDGS if DDGS_AVAILABLE else None

    with ThreadPoolExecutor(max_workers=5) as executor:
        def process_entity(entity_span: EntitySpan):
            try:
                researched = research_entity(
                    entity_span,
                    cache=cache,
                    ddgs_available=DDGS_AVAILABLE,
                    ddgs_cls=ddgs_cls,
                )
                return {
                    "name": researched.surface,
                    "summary": {
                        "label": researched.entity_type,
                        "label_display": researched.entity_type_display,
                        "summary": researched.description,
                        "source": researched.source,
                        "location": researched.location,
                        "entity_subtype": researched.entity_subtype,
                        "confidence": round(researched.confidence, 2),
                        "confidence_level": researched.confidence_level(),
                    }
                }
            except Exception as e:
                logger.error(f"Entity research failed for '{entity_span.surface}': {e}")
                return {
                    "name": entity_span.surface,
                    "summary": {
                        "label": entity_span.entity_type,
                        "label_display": entity_span.entity_type_display,
                        "summary": "Information could not be retrieved at this time.",
                        "source": "N/A",
                        "confidence": 0.0,
                        "confidence_level": "low",
                    }
                }

        futures = [(i, executor.submit(process_entity, ent)) for i, ent in enumerate(entity_spans)]
        # Collect all results first, then yield in original order for deterministic output
        entity_results_by_index = {}
        for i, future in futures:
            try:
                entity_results_by_index[i] = future.result(timeout=30)
            except Exception as e:
                logger.error(f"Entity future failed: {e}")
                entity_results_by_index[i] = {
                    "name": "Unknown Entity",
                    "summary": {
                        "label": "ORG",
                        "label_display": "Organization",
                        "summary": "Information could not be retrieved.",
                        "source": "N/A",
                        "confidence": 0.0,
                        "confidence_level": "low",
                    }
                }
        for i in sorted(entity_results_by_index.keys()):
            yield send("entity", entity_results_by_index[i])

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

        if role == "guest":
            login_user(GuestUser())
            return redirect(url_for("index"))

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
    in_progress = cache.get_all_progress()
    return render_template("dashboard.html", results=results, in_progress=in_progress)

@app.route("/api/teacher/data")
@login_required
def teacher_data():
    if current_user.role != "teacher":
        return jsonify({"error": "Forbidden"}), 403
    
    results = cache.get_all_results()
    in_progress = cache.get_all_progress()
    
    # Process results into a cleaner JSON list
    completed_list = []
    for r in results:
        # Score is already saved as pct (0-100) in my latest fix
        pct = r[3] if r[4] == 100 else round((r[3] / r[4]) * 100)
        completed_list.append({
            "id": r[0], "name": r[1], "test_id": r[2],
            "score": pct, "timestamp": r[5]
        })
    
    active_list = []
    for p in in_progress:
        active_list.append({
            "name": p[0], "identifier": p[1], "test_id": p[2],
            "question_index": p[3], "total": p[4], "updated_at": p[5]
        })
        
    return jsonify({
        "completed": completed_list,
        "in_progress": active_list
    })

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
    mode = request.form.get("mode", "student").strip().lower() or AnalysisMode.STUDENT

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
        stream_with_context(stream_analysis(text, direction, deepl_key, mode=mode)),
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
    port = int(os.environ.get("PORT", 5000))
    logger.info("=" * 50)
    logger.info("  SPY AI — Translation Pre-Research Assistant")
    logger.info(f"  Starting on http://0.0.0.0:{port}")
    logger.info("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
