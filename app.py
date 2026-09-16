# ==========================
# app.py - IEEE Research Paper Reader & Audio Scholar
# ==========================

from flask import Flask, request, jsonify, render_template, session, send_from_directory
from flask_cors import CORS
import os
import tempfile
import time
import fitz  # PyMuPDF
import hashlib
import re
import requests
from datetime import datetime
import json
import random
import logging
from werkzeug.utils import secure_filename

try:
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("⚠️ BeautifulSoup not installed, URL scraping might be limited")

from dotenv import load_dotenv
load_dotenv()

# Import Groq with version check
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed, some features will be disabled")

# Import LangChain with error handling
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not installed, some features will be disabled")

# Import sklearn with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed, some features will be disabled")

# ==========================
# LOGGING SETUP
# ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# APP SETUP
# ==========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-ieee-scholar-98765")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app, supports_credentials=True)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(tempfile.gettempdir(), "pdf_uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ==========================
# ENVIRONMENT VARIABLES
# ==========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL_NAME", os.environ.get("GROQ_MODEL", "qwen2.5vl:latest"))

# Initialize Groq client
groq_client = None
active_groq_model = None
DISCOVERED_GROQ_MODELS = []

# Keywords to exclude (non-chat, safety classifiers, audio, embeddings, gated preview models)
EXCLUDED_MODEL_KEYWORDS = [
    "guard", "prompt-guard", "canopylabs", "whisper", "tts", 
    "embed", "safetensors", "orpheus", "vision", "bge"
]

if GROQ_API_KEY and GROQ_AVAILABLE:
    try:
        import httpx
        http_client = httpx.Client(timeout=60.0)
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            http_client=http_client
        )
        logger.info("✅ Groq client initialized successfully")
        
        # Dynamically discover all valid active chat models from Groq account
        try:
            models_res = groq_client.models.list()
            raw_models = [m.id for m in models_res.data if getattr(m, 'active', True)]
            
            # Filter out classifiers, audio, and gated models
            for mid in raw_models:
                mid_lower = mid.lower()
                if not any(k in mid_lower for k in EXCLUDED_MODEL_KEYWORDS):
                    DISCOVERED_GROQ_MODELS.append(mid)
            
            # Sort with best models first
            def model_priority(m_id):
                m_low = m_id.lower()
                if "70b" in m_low or "deepseek" in m_low or "qwen" in m_low:
                    return 0
                if "llama-3.3" in m_low or "llama-3.2" in m_low:
                    return 1
                if "mistral" in m_low or "saba" in m_low or "gemma" in m_low:
                    return 2
                return 3
            
            DISCOVERED_GROQ_MODELS.sort(key=model_priority)
            logger.info(f"Discovered valid Groq chat models: {DISCOVERED_GROQ_MODELS}")
            
            if DISCOVERED_GROQ_MODELS:
                active_groq_model = DISCOVERED_GROQ_MODELS[0]
        except Exception as me:
            logger.warning(f"Could not list Groq models: {me}")
            DISCOVERED_GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            active_groq_model = DISCOVERED_GROQ_MODELS[0]
            
        logger.info(f"Selected primary Groq model: {active_groq_model}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq client: {e}")
        groq_client = None
else:
    logger.info("ℹ️ Using local Ollama / non-Groq model configuration")

def query_llm(prompt, temperature=0.3):
    """Query Groq API with automatic model failover and local Ollama fallback"""
    global active_groq_model

    # 1. Try Groq with dynamically discovered models
    if groq_client and GROQ_API_KEY:
        models_to_try = []
        if active_groq_model:
            models_to_try.append(active_groq_model)
        for m in DISCOVERED_GROQ_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        for model_name in models_to_try:
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                active_groq_model = model_name
                return completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq model '{model_name}' failed ({e}). Trying next available model...")
                continue

    # 2. Try Local Ollama
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=90
        )
        res.raise_for_status()
        return res.json().get("response", "")
    except Exception as e:
        logger.warning(f"Ollama request failed: {e}")

    raise Exception(
        "No available LLM backend could fulfill the request. "
        "Please check your GROQ_API_KEY / Groq account or start local Ollama ('ollama run qwen2.5vl' or 'ollama serve')."
    )

    # 2. Try Local Ollama
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
            timeout=90
        )
        res.raise_for_status()
        return res.json().get("response", "")
    except Exception as e:
        logger.warning(f"Ollama request failed: {e}")

    raise Exception(
        "No available LLM backend could fulfill the request. "
        "Please verify your GROQ_API_KEY / Groq model or start Ollama locally ('ollama run qwen2.5vl' or 'ollama serve')."
    )

# ==========================
# USER DATABASE & AUTH
# ==========================
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
OTP_STORAGE = {}

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_users(users):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save users: {e}")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def validate_phone(phone):
    phone_clean = re.sub(r'[\s\-\(\)\+]', '', phone)
    return re.match(r'^[0-9]{7,15}$', phone_clean) is not None

def normalize_identifier(identifier):
    identifier = identifier.strip().lower()
    if re.search(r'^[0-9\+\-\(\)\s]+$', identifier) and len(re.sub(r'\D', '', identifier)) >= 7:
        return re.sub(r'\D', '', identifier)
    return identifier

def is_email(identifier):
    return '@' in identifier and '.' in identifier

def get_user_by_identifier(identifier):
    users = load_users()
    normalized = normalize_identifier(identifier)
    for user_id, user_data in users.items():
        if user_id.lower() == normalized or user_id.lower() == identifier.lower():
            return user_id, user_data
        if user_data.get("identifier", "").lower() == normalized or user_data.get("identifier", "").lower() == identifier.lower():
            return user_id, user_data
        if user_data.get("name", "").lower() == identifier.lower():
            return user_id, user_data
    return None, None

def store_otp(identifier, otp):
    normalized = normalize_identifier(identifier)
    OTP_STORAGE[normalized] = {
        "otp": otp,
        "expires": time.time() + 60,
        "attempts": 0
    }

def verify_otp(identifier, otp):
    normalized = normalize_identifier(identifier)
    if normalized in OTP_STORAGE:
        stored = OTP_STORAGE[normalized]
        if time.time() > stored["expires"]:
            del OTP_STORAGE[normalized]
            return "expired"
        if stored["otp"] == otp:
            del OTP_STORAGE[normalized]
            return "valid"
        stored["attempts"] += 1
        if stored["attempts"] >= 3:
            del OTP_STORAGE[normalized]
            return "resend"
        return "invalid"
    return "invalid"

def generate_otp():
    return f"{random.randint(100000, 999999)}"

# ==========================
# IEEE 2-COLUMN & SECTION PARSER
# ==========================
def extract_ieee_pdf(file_path):
    """
    Intelligently extracts text from IEEE double-column papers.
    Handles title/abstract full-width banner, then separates Left/Right columns.
    Extracts structured sections, equations, and metadata.
    """
    doc = fitz.open(file_path)
    full_text_pages = []
    structured_sections = []
    equations_found = []
    
    total_pages = len(doc)

    for page_idx, page in enumerate(doc):
        page_rect = page.rect
        mid_x = page_rect.width / 2.0
        
        # Extract raw blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        
        # Filter text blocks only (block_type == 0)
        text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip() and b[6] == 0]
        
        if not text_blocks:
            continue

        if page_idx == 0:
            # First page: Top banner (Title, Authors, Abstract) is often full-width
            # Detect where 2-column layout starts (look for I. INTRODUCTION or blocks below mid_x boundary)
            banner_blocks = []
            col_left = []
            col_right = []
            
            # Find candidate split line for header banner
            split_y = 0
            for b in text_blocks:
                b_text = b[4]
                if re.search(r'\b(Abstract|Index Terms|Keywords)\b', b_text, re.IGNORECASE):
                    # Banner often ends after abstract or keywords
                    split_y = max(split_y, b[3])
            
            # If no explicit abstract marker found, use top 28% of page as banner
            if split_y == 0:
                split_y = page_rect.height * 0.28
            else:
                split_y += 10  # safety margin

            for b in text_blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                if y1 <= split_y or (x0 < mid_x * 0.6 and x1 > mid_x * 1.4):
                    banner_blocks.append((y0, text))
                elif (x0 + x1) / 2.0 < mid_x:
                    col_left.append((y0, text))
                else:
                    col_right.append((y0, text))

            banner_blocks.sort(key=lambda item: item[0])
            col_left.sort(key=lambda item: item[0])
            col_right.sort(key=lambda item: item[0])

            page_text = "\n".join([t for _, t in banner_blocks]) + "\n"
            page_text += "\n".join([t for _, t in col_left]) + "\n"
            page_text += "\n".join([t for _, t in col_right])

        else:
            # Subsequent pages: IEEE standard 2-column layout
            col_left = []
            col_right = []
            full_width = []

            for b in text_blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                # Check if block spans across both columns (e.g. large table or figure caption)
                if x0 < mid_x * 0.6 and x1 > mid_x * 1.4:
                    full_width.append((y0, text))
                elif (x0 + x1) / 2.0 < mid_x:
                    col_left.append((y0, text))
                else:
                    col_right.append((y0, text))

            col_left.sort(key=lambda item: item[0])
            col_right.sort(key=lambda item: item[0])
            full_width.sort(key=lambda item: item[0])

            # Merge left column, then right column, plus full-width elements
            page_text = "\n".join([t for _, t in col_left]) + "\n"
            page_text += "\n".join([t for _, t in col_right]) + "\n"
            if full_width:
                page_text += "\n".join([t for _, t in full_width])

        # Clean de-hyphenation across line breaks (e.g. "approxi-\nmation" -> "approximation")
        clean_page_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', page_text)
        full_text_pages.append(clean_page_text)

    doc.close()
    full_text = "\n\n".join(full_text_pages)

    # ----------------------------------------------------
    # Parse Structured IEEE Sections
    # ----------------------------------------------------
    section_patterns = [
        (r'(?i)\babstract\b[:\.\s]?', 'Abstract'),
        (r'(?i)\b(index terms|keywords)\b[:\.\s]?', 'Keywords & Index Terms'),
        (r'(?i)\bI\.\s+INTRODUCTION\b', 'I. Introduction'),
        (r'(?i)\bII\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bIII\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bIV\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bV\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bVI\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bVII\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\bVIII\.\s+([A-Z\s\-]{3,40})\b', None),
        (r'(?i)\b(CONCLUSION|CONCLUSIONS|CONCLUDING REMARKS)\b', 'Conclusion'),
        (r'(?i)\b(REFERENCES|BIBLIOGRAPHY)\b', 'References')
    ]

    # Find section split points
    splits = []
    for pattern, default_name in section_patterns:
        for match in re.finditer(pattern, full_text):
            title = default_name if default_name else match.group(0).strip()
            splits.append({
                "start": int(match.start()),
                "title": str(title)
            })

    splits.sort(key=lambda s: int(s["start"]))

    # Build section text slices
    if splits:
        # Pre-section content (Title / Authors)
        first_start = int(splits[0]["start"])
        if first_start > 40:
            header_content = full_text[:first_start].strip()
            structured_sections.append({
                "id": "sec_0",
                "title": "Title & Header",
                "content": header_content[:1500],
                "word_count": len(header_content.split()),
                "est_minutes": max(1, round(len(header_content.split()) / 140.0, 1))
            })

        for i in range(len(splits)):
            curr_start = int(splits[i]["start"])
            curr_end = int(splits[i+1]["start"]) if i+1 < len(splits) else len(full_text)
            sec_text = full_text[curr_start:curr_end].strip()
            
            # Estimated listening duration (~140 words per minute)
            word_count = len(sec_text.split())
            est_minutes = max(1, round(word_count / 140.0, 1))

            structured_sections.append({
                "id": f"sec_{len(structured_sections)}",
                "title": splits[i]["title"],
                "content": sec_text,
                "word_count": word_count,
                "est_minutes": est_minutes
            })
    else:
        # Fallback if no roman numeral headers detected
        structured_sections.append({
            "id": "sec_0",
            "title": "Full Document",
            "content": full_text,
            "word_count": len(full_text.split()),
            "est_minutes": max(1, round(len(full_text.split()) / 140.0, 1))
        })

    # ----------------------------------------------------
    # Extract Mathematical Expressions & Equations
    # ----------------------------------------------------
    math_patterns = [
        r'\$\$([\s\S]*?)\$\$',
        r'\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}',
        r'\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}',
        r'(\b[a-zA-Z0-9_\^\+\-\*/\(\)\{\}\\\s]{2,}\s*=\s*[a-zA-Z0-9_\^\+\-\*/\(\)\\\{\}\s\sum\int\prod\partial\nabla\in\le\ge]{3,}\s*\(\d+\))',
        r'(\b\w+\s*\(.*?\)\s*=\s*[\w\+\-\*/\(\)\\\{\}\sum\int_^\d\.\s]{4,}\b)',
        r'(\\mathcal\{[A-Z]\}\s*=\s*[^;\n]{4,})',
        r'(\b\mathcal\{L\}[_\w]*\s*=\s*[^;\n]{4,})',
        r'(\b\min_{[^}]+}\s*[^;\n]{4,})',
        r'(\b\max_{[^}]+}\s*[^;\n]{4,})'
    ]
    
    eq_id_counter = 1
    for p in math_patterns:
        for match in re.finditer(p, full_text):
            raw_eq = match.group(0).strip()
            # Clean up newlines within formula
            clean_eq = re.sub(r'\s+', ' ', raw_eq).strip()
            if len(clean_eq) > 5 and not any(e["equation"] == clean_eq for e in equations_found) and len(equations_found) < 25:
                # Detect equation number if present like (1), (2)
                num_match = re.search(r'\((\d+)\)$', clean_eq)
                eq_label = f"Equation ({num_match.group(1)})" if num_match else f"Formula #{eq_id_counter}"
                equations_found.append({
                    "id": f"eq_{int(time.time()*1000)}_{eq_id_counter}",
                    "name": eq_label,
                    "equation": clean_eq,
                    "latex": clean_eq if clean_eq.startswith("$") else f"$${clean_eq}$$",
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                eq_id_counter += 1

    # Extract paper title candidate (first non-empty line)
    title_match = re.search(r'^[^\n]{10,120}', full_text.strip())
    paper_title = title_match.group(0).strip() if title_match else "IEEE Research Paper"

    return {
        "title": paper_title,
        "full_text": full_text,
        "total_pages": total_pages,
        "sections": structured_sections,
        "equations": equations_found
    }

# ==========================
# RAG DATABASE & STORAGE
# ==========================
user_data = {}

def get_user_data():
    user_id = session.get("user_id")
    if not user_id:
        return None
    if user_id not in user_data:
        user_data[user_id] = {
            "vectorizer": None,
            "matrix": None,
            "stored_chunks": [],
            "source_name": None,
            "source_type": None,
            "paper_info": None,
            "podcast_script": None,
            "saved_notes": []
        }
    return user_data[user_id]

def build_db(text, user_data_obj):
    if not LANGCHAIN_AVAILABLE or not SKLEARN_AVAILABLE:
        raise Exception("Required libraries (LangChain/scikit-learn) not available.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=120)
    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c.strip()) > 60][:120]
    if not chunks:
        raise Exception("No readable text found in paper.")
    user_data_obj["stored_chunks"] = chunks
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    user_data_obj["vectorizer"] = vectorizer
    user_data_obj["matrix"] = matrix
    return len(chunks)

def build_scholar_prompt(question, context, paper_title="IEEE Paper"):
    return f"""
You are an expert IEEE Academic Fellow and Senior Research Scientist.
Answer the question with academic precision, citing key concepts, formulas, and findings from "{paper_title}".

Formatting & Rigor Rules:
- Render mathematical variables and equations cleanly using LaTeX format (e.g. $E = mc^2$ or $$\\mathcal{{L}}_{{total}} = \\lambda_1 \\mathcal{{L}}_{{recon}} + \\lambda_2 \\mathcal{{L}}_{{reg}}$$).
- Use concise bullet points for experimental metrics, comparisons, and algorithmic steps.
- If referencing limitations or tradeoffs, provide objective technical reasoning.
- If the specific answer is not directly in the provided context, state what is available in the paper and what is unstated.

PAPER CONTEXT:
{context}

USER QUESTION:
{question}

SCHOLARLY ANSWER:
"""

# ==========================
# AUTHENTICATION ROUTES
# ==========================
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json() or {}
        identifier = data.get("identifier", "").strip()
        name = data.get("name", "").strip() or identifier.split("@")[0]
        password = data.get("password", "").strip()
        
        if not identifier or not password:
            return jsonify({"error": "Identifier and password required"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        if len(identifier) < 3:
            return jsonify({"error": "Username/Email must be at least 3 characters"}), 400

        is_email_identifier = is_email(identifier)
        if is_email_identifier and not validate_email(identifier):
            return jsonify({"error": "Invalid email format"}), 400
        
        normalized = normalize_identifier(identifier)
        users = load_users()
        
        existing_uid, _ = get_user_by_identifier(identifier)
        if existing_uid:
            return jsonify({"error": "An account already exists with this username/email"}), 400
            
        users[normalized] = {
            "name": name,
            "password": hash_password(password), 
            "created_at": datetime.now().isoformat(),
            "identifier": normalized,
            "is_email": is_email_identifier,
            "type": "email" if is_email_identifier else ("phone" if identifier.isdigit() else "username")
        }
        save_users(users)
        session["user"] = name
        session["user_id"] = normalized
        session.permanent = True
        return jsonify({"success": True, "message": "Signup successful", "user": name, "user_id": normalized})
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    identifier = data.get("identifier", "").strip()
    password = data.get("password", "").strip()

    if not identifier or not password:
        return jsonify({"error": "Identifier and password required"}), 400

    user_info = get_user_by_identifier(identifier)
    if not user_info:
        return jsonify({"error": "No account found. Please register first or use Guest Access."}), 404

    uid, user = user_info
    if not verify_password(password, user.get("password", "")):
        return jsonify({"error": "Invalid password"}), 401

    session["user"] = user.get("name") or identifier
    session["user_id"] = uid
    session.permanent = True

    return jsonify({
        "success": True,
        "message": "Login successful",
        "redirect": "/",
        "user": session["user"],
        "user_id": uid
    })

@app.route("/guest_login", methods=["POST", "GET"])
def guest_login():
    guest_id = f"guest_{random.randint(1000, 9999)}"
    guest_name = "Guest Scholar"
    session["user"] = guest_name
    session["user_id"] = guest_id
    session.permanent = True
    return jsonify({
        "success": True,
        "redirect": "/",
        "user": guest_name,
        "user_id": guest_id
    })

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

@app.route("/check_auth", methods=["GET"])
def check_auth():
    user = session.get("user")
    user_id = session.get("user_id")
    if user and user_id:
        return jsonify({"authenticated": True, "user": user, "user_id": user_id})
    return jsonify({"authenticated": False})

@app.route("/get_otp", methods=["POST"])
def get_otp_route():
    data = request.get_json() or {}
    identifier = data.get("identifier", "").strip()
    user_id, user_data = get_user_by_identifier(identifier)
    if not user_id:
        return jsonify({"success": False, "error": "Account not registered"}), 404
    otp = generate_otp()
    store_otp(identifier, otp)
    return jsonify({"success": True, "otp": otp})

@app.route("/resend_otp", methods=["POST"])
def resend_otp_route():
    data = request.get_json() or {}
    identifier = data.get("identifier", "").strip()
    user_id, user_data = get_user_by_identifier(identifier)
    if not user_id:
        return jsonify({"success": False, "error": "Account not registered"}), 404
    otp = generate_otp()
    store_otp(identifier, otp)
    return jsonify({"success": True, "otp": otp})

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    data = request.get_json() or {}
    identifier = data.get("identifier", "").strip()
    otp = data.get("otp", "").strip()
    result = verify_otp(identifier, otp)
    if result == "valid":
        session["reset_user"] = identifier
        return jsonify({"success": True})
    elif result == "expired":
        return jsonify({"success": False, "error": "OTP expired"})
    elif result == "resend":
        return jsonify({"success": False, "error": "Too many attempts"})
    return jsonify({"success": False, "error": "Invalid OTP"})

@app.route("/reset_password", methods=["POST"])
def reset_password():
    identifier = session.get("reset_user")
    if not identifier:
        return jsonify({"success": False, "error": "Session expired"})
    data = request.get_json() or {}
    new_pw = data.get("new_password", "").strip()
    if len(new_pw) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters"})
    users = load_users()
    uid, _ = get_user_by_identifier(identifier)
    if uid and uid in users:
        users[uid]["password"] = hash_password(new_pw)
        save_users(users)
        session.pop("reset_user", None)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "User not found"})

# ==========================
# PAGE ROUTES & STATIC SERVING
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login-page")
def login_page():
    return render_template("login.html")

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ==========================
# IEEE PAPER UPLOAD & PARSING
# ==========================
@app.route("/upload", methods=["POST"])
@app.route("/upload_paper", methods=["POST"])
def upload_paper():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized. Please login first."}), 401
        
    if "file" not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only IEEE / Academic PDF files are supported"}), 400

    try:
        raw_name = secure_filename(file.filename)
        saved_filename = f"{int(time.time())}_{raw_name}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
        file.save(save_path)
        
        # Parse IEEE Paper with 2-Column de-noiser & Section Extractor
        parsed_data = extract_ieee_pdf(save_path)
        
        if not parsed_data["full_text"].strip():
            return jsonify({"error": "Could not extract text from this paper. It might be a scanned image."}), 400

        # Build index and vector database for the active user
        user_data_obj = get_user_data()
        user_data_obj["source_name"] = parsed_data["title"] or raw_name
        user_data_obj["source_type"] = "pdf"
        user_data_obj["saved_file_path"] = f"/uploads/{saved_filename}"
        user_data_obj["paper_info"] = parsed_data
        user_data_obj["podcast_script"] = None
        
        chunk_count = build_db(parsed_data["full_text"], user_data_obj)

        # Generate comprehensive executive summary
        summary_prompt = f"""
Provide an executive academic overview of this IEEE research paper:
Title: {parsed_data['title']}

Content Excerpt:
{parsed_data['full_text'][:4500]}

Format with clear markdown sections:
1. 🎯 **Research Problem & Motivation**
2. 💡 **Core Technical Innovation**
3. 🔬 **Key Methodology / Architecture**
4. 📊 **Experimental Highlights & Performance Gain**
5. ⚠️ **Main Limitations & Open Questions**
"""
        initial_explanation = query_llm(summary_prompt, temperature=0.2)

        return jsonify({
            "success": True,
            "title": parsed_data["title"],
            "source_name": parsed_data["title"] or raw_name,
            "file_url": f"/uploads/{saved_filename}",
            "total_pages": parsed_data["total_pages"],
            "total_sections": len(parsed_data["sections"]),
            "sections": parsed_data["sections"],
            "equations": parsed_data["equations"],
            "chunks": chunk_count,
            "overview_summary": initial_explanation
        })
    except Exception as e:
        logger.error(f"Error processing IEEE PDF: {e}")
        return jsonify({"error": f"Failed to parse paper: {str(e)}"}), 500

@app.route("/process_url", methods=["POST"])
def process_url_endpoint():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Paper URL or ArXiv link required"}), 400

    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url

    # Convert ArXiv abstract URL to direct PDF if applicable
    if "arxiv.org/abs/" in url:
        url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            saved_filename = f"{int(time.time())}_paper.pdf"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
            with open(save_path, "wb") as f:
                f.write(response.content)
            
            parsed_data = extract_ieee_pdf(save_path)
            file_url = f"/uploads/{saved_filename}"
        else:
            if not WEB_SCRAPING_AVAILABLE:
                raise Exception("Web scraping libraries not available")
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.extract()
            title = soup.title.string.strip() if soup.title and soup.title.string else "Online Paper"
            text = " ".join(soup.stripped_strings)
            parsed_data = {
                "title": title,
                "full_text": text,
                "total_pages": 1,
                "sections": [{"id": "sec_0", "title": "Web Document", "content": text, "word_count": len(text.split()), "est_minutes": max(1, round(len(text.split())/140.0, 1))}],
                "equations": []
            }
            file_url = url

        user_data_obj = get_user_data()
        paper_title = str(parsed_data.get("title", "Online Paper"))
        full_paper_text = str(parsed_data.get("full_text", ""))
        
        user_data_obj["source_name"] = paper_title
        user_data_obj["source_type"] = "url"
        user_data_obj["saved_file_path"] = file_url
        user_data_obj["paper_info"] = parsed_data
        user_data_obj["podcast_script"] = None

        chunk_count = build_db(full_paper_text, user_data_obj)

        summary_prompt = f"Provide a structured academic summary of '{paper_title}':\n\n{full_paper_text[:4000]}"
        initial_explanation = query_llm(summary_prompt)

        return jsonify({
            "success": True,
            "title": parsed_data["title"],
            "source_name": parsed_data["title"],
            "file_url": file_url,
            "total_pages": parsed_data["total_pages"],
            "sections": parsed_data["sections"],
            "equations": parsed_data["equations"],
            "chunks": chunk_count,
            "overview_summary": initial_explanation
        })
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return jsonify({"error": f"Failed to fetch paper: {str(e)}"}), 500

# ==========================
# PAPER ANALYSIS & SECTIONS API
# ==========================
@app.route("/api/paper/details", methods=["GET"])
def get_paper_details():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"loaded": False})
    
    info = user_data_obj["paper_info"]
    return jsonify({
        "loaded": True,
        "title": info["title"],
        "source_name": user_data_obj.get("source_name", "Research Paper"),
        "total_pages": info.get("total_pages", 1),
        "file_url": user_data_obj.get("saved_file_path", ""),
        "sections": info.get("sections", []),
        "equations": info.get("equations", []),
        "has_podcast": user_data_obj.get("podcast_script") is not None
    })

@app.route("/api/paper/equations", methods=["GET"])
def get_equations():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"equations": []})
    return jsonify({"equations": user_data_obj["paper_info"].get("equations", [])})

@app.route("/api/paper/add_equation", methods=["POST"])
def add_equation():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json() or {}
    equation_text = data.get("equation", "").strip()
    name = data.get("name", "").strip() or "Custom Formula"
    
    if not equation_text:
        return jsonify({"error": "Equation expression is required"}), 400

    user_data_obj = get_user_data()
    if not user_data_obj:
        return jsonify({"error": "No user session"}), 400
    if not user_data_obj.get("paper_info"):
        user_data_obj["paper_info"] = {"title": "Active Study", "sections": [], "equations": []}
    
    if "equations" not in user_data_obj["paper_info"]:
        user_data_obj["paper_info"]["equations"] = []

    eq_obj = {
        "id": f"eq_{int(time.time()*1000)}_{random.randint(100,999)}",
        "name": name,
        "equation": equation_text,
        "latex": equation_text if equation_text.startswith("$") else f"$${equation_text}$$",
        "timestamp": datetime.now().strftime("%H:%M")
    }
    user_data_obj["paper_info"]["equations"].append(eq_obj)
    return jsonify({"success": True, "equation": eq_obj, "total": len(user_data_obj["paper_info"]["equations"])})

@app.route("/api/paper/delete_equation/<eq_id>", methods=["DELETE"])
def delete_equation(eq_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if user_data_obj and user_data_obj.get("paper_info"):
        eqs = user_data_obj["paper_info"].get("equations", [])
        # Support string items or dictionary items
        updated = []
        for e in eqs:
            if isinstance(e, dict) and e.get("id") == eq_id:
                continue
            elif isinstance(e, str) and e == eq_id:
                continue
            updated.append(e)
        user_data_obj["paper_info"]["equations"] = updated
    return jsonify({"success": True})

@app.route("/api/paper/auto_extract_equations", methods=["POST"])
def auto_extract_equations():
    """Uses LLM to extract all mathematical formulations, loss functions, and optimization problems from the paper into LaTeX."""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"error": "Please load a paper first"}), 400

    paper = user_data_obj["paper_info"]
    prompt = f"""
Identify and extract all core mathematical equations, loss functions, objective functions, Bayesian formulations, neural layer equations, and parameter update formulas from this IEEE research paper:
Title: {paper['title']}

Paper Content:
{paper['full_text'][:5000]}

Rules:
1. Extract 5 to 10 key equations.
2. Provide clean LaTeX markup for each equation without surrounding markdown code blocks.
3. Include the equation name/description (e.g., "Loss Function", "Attention Mechanism", "State Transition Probability", "Objective Function").
4. Output STRICTLY a JSON array of objects with keys 'name', 'latex', and 'description':
[
  {{
    "name": "Composite Loss Function (Eq. 4)",
    "latex": "\\mathcal{{L}}_{{total}} = \\alpha \\mathcal{{L}}_{{recon}} + \\beta \\mathcal{{L}}_{{reg}}",
    "description": "Balances reconstruction fidelity against latent regularization."
  }}
]
"""
    try:
        raw_res = query_llm(prompt, temperature=0.2)
        clean_json = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.MULTILINE)
        
        extracted_list = []
        if "equations" not in paper:
            paper["equations"] = []

        try:
            items = json.loads(clean_json)
            for item in items:
                if isinstance(item, dict) and (item.get("latex") or item.get("equation")):
                    latex_str = item.get("latex") or item.get("equation")
                    eq_obj = {
                        "id": f"eq_{int(time.time()*1000)}_{random.randint(100,999)}",
                        "name": item.get("name") or "Extracted Formula",
                        "equation": latex_str,
                        "latex": latex_str if latex_str.startswith("$") else f"$${latex_str}$$",
                        "description": item.get("description", ""),
                        "timestamp": datetime.now().strftime("%H:%M")
                    }
                    paper["equations"].append(eq_obj)
                    extracted_list.append(eq_obj)
        except Exception:
            # Fallback regex search on LLM text
            for line in raw_res.split("\n"):
                if "=" in line and len(line) > 6:
                    clean_line = re.sub(r'^[\-\*\d\.\s]+', '', line).strip()
                    eq_obj = {
                        "id": f"eq_{int(time.time()*1000)}_{random.randint(100,999)}",
                        "name": "Formulation",
                        "equation": clean_line,
                        "latex": f"$${clean_line}$$",
                        "description": "",
                        "timestamp": datetime.now().strftime("%H:%M")
                    }
                    paper["equations"].append(eq_obj)
                    extracted_list.append(eq_obj)

        return jsonify({"success": True, "added": len(extracted_list), "equations": paper["equations"]})
    except Exception as e:
        logger.error(f"Auto equation extraction error: {e}")
        return jsonify({"error": f"Failed to extract equations: {str(e)}"}), 500

@app.route("/api/paper/explain_equation", methods=["POST"])
def explain_equation():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    data = request.get_json() or {}
    equation = data.get("equation", "").strip()
    if not equation:
        return jsonify({"error": "No equation provided"}), 400

    user_data_obj = get_user_data()
    paper_title = user_data_obj.get("source_name", "IEEE Paper") if user_data_obj else "IEEE Paper"
    
    prompt = f"""
You are an expert theoretical mathematician and computer science professor.
Analyze and explain the following mathematical equation from the paper "{paper_title}":

EQUATION:
{equation}

Please provide a crystal-clear, pedagogical breakdown:
1. 🧮 **LaTeX Notation**: Standardized format of the equation.
2. 🔍 **Variable & Parameter Deconstruction**: Define every variable, constant, index, and operator.
3. 💡 **Intuitive Meaning**: What physical or algorithmic concept is this equation calculating or optimizing?
4. ⚙️ **Role in the System/Model**: Why is this equation necessary for the paper's proposed architecture/objective function?
"""
    try:
        explanation = query_llm(prompt, temperature=0.2)
        return jsonify({"success": True, "equation": equation, "explanation": explanation})
    except Exception as e:
        logger.error(f"Equation explanation error: {e}")
        return jsonify({"error": f"Failed to analyze equation: {str(e)}"}), 500

@app.route("/api/paper/critique", methods=["POST"])
def critique_paper():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"error": "Please load an IEEE paper first"}), 400

    paper = user_data_obj["paper_info"]
    prompt = f"""
Conduct an in-depth, peer-review level critical appraisal of this IEEE research paper:
Title: {paper['title']}

Text Excerpt:
{paper['full_text'][:5000]}

Provide rigorous feedback using the following structure:
1. 🌟 **Primary Contributions & Originality** (What makes this work distinct from prior art?)
2. 📐 **Methodological Soundness** (Are the mathematical formulations and assumptions valid?)
3. 📊 **Experimental Rigor & Benchmarking** (Are baseline comparisons fair? Are ablation studies adequate?)
4. ⚠️ **Threats to Validity & Limitations** (What scenarios would cause this method to fail or degrade?)
5. 🚀 **Actionable Future Research Directions** (How can future researchers extend this work?)
"""
    try:
        critique = query_llm(prompt, temperature=0.3)
        return jsonify({"success": True, "critique": critique})
    except Exception as e:
        logger.error(f"Critique error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================
# AI PODCAST & AUDIO STUDIO
# ==========================
@app.route("/api/paper/podcast_script", methods=["POST"])
def generate_podcast_script():
    """
    Generates a 2-host conversational podcast script (NotebookLM style)
    breaking down the research paper into an entertaining, high-yield audio dialogue.
    """
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"error": "Please upload a research paper first."}), 400

    # If cached, return existing podcast
    if user_data_obj.get("podcast_script"):
        return jsonify({"success": True, "script": user_data_obj["podcast_script"]})

    paper = user_data_obj["paper_info"]
    prompt = f"""
You are creating an engaging, intellectually stimulating research podcast episode between two hosts:
- **Alex (Host 1)**: Energetic, curious tech enthusiast who asks great clarifying questions and breaks down intuition.
- **Dr. Sam (Host 2)**: Expert Research Scientist who understands the deep technical architecture, math, benchmarks, and real-world implications.

Topic Paper: "{paper['title']}"

Paper Content:
{paper['full_text'][:5000]}

Episode Guidelines:
- Write a lively 8 to 12 turn dialogue script.
- Start with a catchy hook explaining why this problem matters.
- Break down the core technical breakthrough without confusing jargon.
- Discuss surprising benchmark results and realistic limitations.
- Output ONLY valid JSON formatted as a list of dialogue turns:
[
  {{"speaker": "Alex", "text": "Welcome back to Research Unpacked! Today we are diving into..."}},
  {{"speaker": "Dr. Sam", "text": "Thanks Alex! This paper tackles a huge bottleneck in..."}}
]
"""
    try:
        raw_response = query_llm(prompt, temperature=0.5)
        # Clean JSON markdown blocks
        clean_json = re.sub(r'^```json\s*|\s*```$', '', raw_response.strip(), flags=re.MULTILINE)
        try:
            dialogue = json.loads(clean_json)
        except Exception:
            # Fallback if json parsing fails: create structured turns from text
            dialogue = [
                {"speaker": "Alex", "text": f"Welcome back! Today we are breaking down the paper: {paper['title']}."},
                {"speaker": "Dr. Sam", "text": "This paper presents groundbreaking methods. Here is why it matters and what the experiments reveal."}
            ]
            for line in raw_response.split("\n"):
                if line.startswith("Alex:") or line.startswith("**Alex**"):
                    dialogue.append({"speaker": "Alex", "text": re.sub(r'^\*?\*?Alex\*?\*?:\s*', '', line)})
                elif line.startswith("Dr. Sam:") or line.startswith("**Dr. Sam**"):
                    dialogue.append({"speaker": "Dr. Sam", "text": re.sub(r'^\*?\*?Dr\. Sam\*?\*?:\s*', '', line)})

        user_data_obj["podcast_script"] = dialogue
        return jsonify({"success": True, "script": dialogue})
    except Exception as e:
        logger.error(f"Podcast generation error: {e}")
        return jsonify({"error": f"Failed to generate podcast: {str(e)}"}), 500

# ==========================
# SCHOLAR Q&A (RAG)
# ==========================
@app.route("/ask", methods=["POST"])
def ask():
    if "user_id" not in session:
        return jsonify({"answer": "Please login first"}), 401
    
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a question."})
    
    user_data_obj = get_user_data()
    if not user_data_obj or user_data_obj.get("vectorizer") is None:
        return jsonify({"answer": "Please upload an IEEE research paper or enter a paper link first."})
        
    try:
        q_vec = user_data_obj["vectorizer"].transform([question])
        scores = (user_data_obj["matrix"] @ q_vec.T).toarray().ravel()
        top_idx = scores.argsort()[-5:][::-1]
        context = "\n\n".join([user_data_obj["stored_chunks"][i] for i in top_idx])
        
        prompt = build_scholar_prompt(question, context, user_data_obj.get("source_name", "Paper"))
        answer = query_llm(prompt)
        
        return jsonify({"answer": answer, "type": "document"})
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

# ==========================
# NOTES & BOOKMARKS API
# ==========================
@app.route("/api/notes/add", methods=["POST"])
def add_note():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json() or {}
    note_text = data.get("text", "").strip()
    section_title = data.get("section", "General")
    tag = data.get("tag", "Key Point")
    if not note_text:
        return jsonify({"error": "Note text is required"}), 400
        
    user_data_obj = get_user_data()
    note_obj = {
        "id": f"note_{int(time.time()*1000)}_{random.randint(100,999)}",
        "text": note_text,
        "section": section_title,
        "tag": tag,
        "timestamp": datetime.now().strftime("%b %d, %H:%M")
    }
    user_data_obj["saved_notes"].append(note_obj)
    return jsonify({"success": True, "note": note_obj, "total_notes": len(user_data_obj["saved_notes"])})

@app.route("/api/notes/get", methods=["GET"])
def get_notes():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    return jsonify({"notes": user_data_obj.get("saved_notes", []) if user_data_obj else []})

@app.route("/api/notes/delete/<note_id>", methods=["DELETE"])
def delete_note(note_id):
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if user_data_obj:
        user_data_obj["saved_notes"] = [n for n in user_data_obj.get("saved_notes", []) if n.get("id") != note_id]
    return jsonify({"success": True, "notes": user_data_obj.get("saved_notes", [])})

@app.route("/api/notes/clear", methods=["POST"])
def clear_notes():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if user_data_obj:
        user_data_obj["saved_notes"] = []
    return jsonify({"success": True})

@app.route("/api/notes/auto_extract", methods=["POST"])
def auto_extract_notes():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_data_obj = get_user_data()
    if not user_data_obj or not user_data_obj.get("paper_info"):
        return jsonify({"error": "Please load a paper first"}), 400

    paper = user_data_obj["paper_info"]
    prompt = f"""
Extract 5 to 7 high-yield, indispensable research notes and key study bullet points from this IEEE paper:
Title: {paper['title']}

Content:
{paper['full_text'][:4500]}

Format as a valid JSON list of items with 'text' and 'tag' (choose tag from: Novelty, Methodology, Benchmark, Limitation, Key Finding):
[
  {{"text": "Proposed X method achieves Y% lower latency by optimizing Z.", "tag": "Novelty"}},
  {{"text": "Evaluated on Dataset A against Baselines B and C with 10-fold CV.", "tag": "Benchmark"}}
]
"""
    try:
        raw_res = query_llm(prompt, temperature=0.3)
        clean_json = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.MULTILINE)
        extracted = []
        try:
            items = json.loads(clean_json)
            for item in items:
                if isinstance(item, dict) and item.get("text"):
                    note_obj = {
                        "id": f"note_{int(time.time()*1000)}_{random.randint(100,999)}",
                        "text": item.get("text").strip(),
                        "section": "Auto Extracted",
                        "tag": item.get("tag", "Key Finding"),
                        "timestamp": datetime.now().strftime("%b %d, %H:%M")
                    }
                    user_data_obj["saved_notes"].append(note_obj)
                    extracted.append(note_obj)
        except Exception:
            for line in raw_res.split("\n"):
                line_clean = re.sub(r'^[\-\*\d\.\s]+', '', line).strip()
                if len(line_clean) > 20:
                    note_obj = {
                        "id": f"note_{int(time.time()*1000)}_{random.randint(100,999)}",
                        "text": line_clean,
                        "section": "Auto Extracted",
                        "tag": "Key Finding",
                        "timestamp": datetime.now().strftime("%b %d, %H:%M")
                    }
                    user_data_obj["saved_notes"].append(note_obj)
                    extracted.append(note_obj)

        return jsonify({"success": True, "added": len(extracted), "notes": user_data_obj["saved_notes"]})
    except Exception as e:
        logger.error(f"Auto note extraction error: {e}")
        return jsonify({"error": f"Failed to extract notes: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "groq_available": groq_client is not None,
        "langchain_available": LANGCHAIN_AVAILABLE,
        "sklearn_available": SKLEARN_AVAILABLE
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"🚀 Starting IEEE Research Scholar on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)