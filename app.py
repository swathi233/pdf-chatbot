# ==========================
# app.py - Complete Flask Application with Railway Compatibility
# ==========================

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import tempfile
import time
import fitz  # PyMuPDF
import hashlib
import re
from datetime import datetime
import json
import random
import logging
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

# Import Groq with version check
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed, some features will be disabled")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
# LOGGING SETUP
# ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# APP SETUP
# ==========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-for-railway")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ==========================
# ENVIRONMENT VARIABLES
# ==========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize Groq client
groq_client = None
MODEL = "llama-3.1-8b-instant"

if GROQ_API_KEY and GROQ_AVAILABLE:
    try:
        import httpx
        http_client = httpx.Client(timeout=60.0)
        groq_client = Groq(
            api_key=GROQ_API_KEY,
            http_client=http_client
        )
        logger.info("✅ Groq client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Groq client: {e}")
        groq_client = None
else:
    logger.warning("⚠️ Groq client not available")

# ==========================
# USER DATABASE
# ==========================
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")
OTP_STORAGE = {}

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
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
    # Simple phone validation - accepts various formats
    phone = re.sub(r'[\s\-\(\)\+]', '', phone)
    return re.match(r'^[0-9]{7,15}$', phone) is not None

def normalize_identifier(identifier):
    """Normalize identifier to a standard format"""
    identifier = identifier.strip().lower()
    # If it's a phone number, remove special characters
    if re.search(r'[0-9]', identifier) and not re.search(r'@', identifier):
        # It's a phone number
        phone = re.sub(r'[\s\-\(\)\+]', '', identifier)
        return phone
    return identifier

def is_email(identifier):
    return '@' in identifier and '.' in identifier

def get_user_by_identifier(identifier):
    """Find user by email or phone number"""
    users = load_users()
    normalized = normalize_identifier(identifier)
    
    for user_id, user_data in users.items():
        if user_id == normalized:
            return user_id, user_data
        # Also check if user_id matches the raw identifier
        if user_id == identifier:
            return user_id, user_data
    
    return None, None

# ==========================
# OTP STORAGE (5 seconds expiry)
# ==========================
def store_otp(identifier, otp):
    normalized = normalize_identifier(identifier)
    OTP_STORAGE[normalized] = {
        "otp": otp,
        "expires": time.time() + 5,  # 5 seconds expiry
        "attempts": 0
    }
    logger.info(f"OTP stored for {normalized}: {otp} (expires in 5 seconds)")

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
# CASUAL GREETING HANDLER
# ==========================
def is_casual_greeting(text):
    """Check if the user message is a casual greeting"""
    text = text.lower().strip()
    # Common casual greetings
    casual_greetings = [
        'hi', 'hello', 'hey', 'hi there', 'hello there', 'hey there',
        'good morning', 'good afternoon', 'good evening', 'good night',
        'greetings', 'howdy', 'yo', 'sup', 'what\'s up', 'whats up',
        'hey hey', 'hi hi', 'hello hello', 'morning', 'evening'
    ]
    
    # Check for exact matches or simple variations
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    
    for greeting in casual_greetings:
        if text_clean == greeting or text_clean.startswith(greeting + ' '):
            return True
    
    # Check for patterns like "hello world" or "hi everyone"
    if any(g in text_clean for g in ['hi', 'hello', 'hey']) and len(text_clean.split()) <= 3:
        return True
    
    return False

def get_casual_response(text):
    """Generate a friendly response to casual greetings"""
    text_lower = text.lower().strip()
    
    # Time-based greetings
    current_hour = datetime.now().hour
    
    responses = []
    
    if 'good morning' in text_lower or 'morning' in text_lower:
        responses = [
            "🌅 Good morning to you too! How can I help you today?",
            "☀️ Good morning! Ready to dive into your documents?",
            "🌄 Good morning! I'm here to help with your research."
        ]
    elif 'good afternoon' in text_lower or 'afternoon' in text_lower:
        responses = [
            "🌤️ Good afternoon! What can I assist you with?",
            "☀️ Good afternoon! Your documents are ready for questions.",
            "🌞 Good afternoon! How can I help you explore your PDFs?"
        ]
    elif 'good evening' in text_lower or 'evening' in text_lower:
        responses = [
            "🌙 Good evening! I'm here to help you with your research.",
            "🌟 Good evening! What would you like to know about your documents?",
            "🌆 Good evening! Ready to answer your questions."
        ]
    elif 'good night' in text_lower or 'night' in text_lower:
        responses = [
            "🌙 Good night! I'll be here when you need me.",
            "💤 Good night! Sweet dreams and see you tomorrow.",
            "🌃 Good night! Remember, I'm always available."
        ]
    elif 'hi' in text_lower or 'hello' in text_lower or 'hey' in text_lower:
        responses = [
            "👋 Hey there! How can I help you with your documents today?",
            "👋 Hello! I'm ready to answer any questions about your PDF.",
            "😊 Hi! What would you like to know?",
            "👋 Hey! Feel free to ask me anything about your uploaded documents.",
            "Hello! 👋 I'm your research assistant. What can I do for you?",
            "Hi there! 😊 I'm here to help you understand your PDFs better."
        ]
    else:
        responses = [
            "👋 Hello! How can I assist you today?",
            "😊 Hi there! Ready to help with your research questions.",
            "👋 Hey! What would you like to explore?"
        ]
    
    return random.choice(responses)

# ==========================
# AUTHENTICATION ROUTES
# ==========================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        identifier = data.get("identifier", "").strip()
        password = data.get("password", "").strip()
        
        if not identifier or not password:
            return jsonify({"error": "All fields required"}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        # Validate identifier (email or phone)
        is_email_identifier = is_email(identifier)
        if is_email_identifier and not validate_email(identifier):
            return jsonify({"error": "Invalid email format"}), 400
        elif not is_email_identifier and not validate_phone(identifier):
            return jsonify({"error": "Invalid phone number"}), 400
        
        normalized = normalize_identifier(identifier)
        users = load_users()
        
        if normalized in users:
            return jsonify({"error": "Account already exists"}), 400
            
        users[normalized] = {
            "password": hash_password(password), 
            "created_at": datetime.now().isoformat(),
            "identifier": normalized,
            "is_email": is_email_identifier
        }
        save_users(users)
        session["user_id"] = normalized
        return jsonify({"success": True, "message": "Signup successful", "user": normalized})
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        identifier = data.get("identifier", "").strip()
        password = data.get("password", "").strip()
        
        if not identifier or not password:
            return jsonify({"error": "All fields required"}), 400
        
        user_id, user_data = get_user_by_identifier(identifier)
        
        if not user_data:
            return jsonify({"error": "Account not registered. Please sign up first."}), 401
            
        if not verify_password(password, user_data["password"]):
            return jsonify({"error": "Invalid credentials"}), 401
            
        session["user_id"] = user_id
        return jsonify({"success": True, "message": "Login successful", "user": user_id})
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_otp", methods=["POST"])
def get_otp():
    """Endpoint for client to get OTP for a user"""
    try:
        data = request.get_json()
        identifier = data.get("identifier", "").strip()
        
        if not identifier:
            return jsonify({"error": "Email or phone number required"}), 400
        
        normalized = normalize_identifier(identifier)
        users = load_users()
        
        # Check if user exists
        if normalized not in users:
            # Try to find by raw identifier
            found = False
            for user_id in users:
                if user_id == identifier:
                    normalized = user_id
                    found = True
                    break
            if not found:
                return jsonify({"error": "No account found with this email/phone"}), 404
        
        otp = generate_otp()
        store_otp(normalized, otp)
        
        logger.info(f"📧 OTP for {normalized}: {otp} (expires in 5 seconds)")
        
        return jsonify({
            "success": True,
            "otp": otp,
            "identifier": normalized
        })
    except Exception as e:
        logger.error(f"Get OTP error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    try:
        data = request.get_json()
        identifier = data.get("identifier", "").strip()
        
        if not identifier:
            return jsonify({"error": "Identifier required"}), 400
            
        normalized = normalize_identifier(identifier)
        users = load_users()
        
        if normalized not in users:
            return jsonify({"error": "No account found"}), 404
            
        otp = generate_otp()
        store_otp(normalized, otp)
        
        logger.info(f"📧 New OTP for {normalized}: {otp} (expires in 5 seconds)")
        
        return jsonify({
            "success": True,
            "message": "New OTP generated",
            "otp": otp
        })
    except Exception as e:
        logger.error(f"Resend OTP error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    try:
        data = request.get_json()
        identifier = data.get("identifier", "").strip()
        otp = data.get("otp", "").strip()

        if not identifier or not otp:
            return jsonify({"error": "All fields required"}), 400

        normalized = normalize_identifier(identifier)
        result = verify_otp(normalized, otp)
        
        if result == "valid":
            session["reset_verified"] = normalized
            return jsonify({"success": True, "message": "OTP verified"})
        elif result == "expired":
            return jsonify({"error": "OTP expired. Request a new one."}), 401
        elif result == "resend":
            new_otp = generate_otp()
            store_otp(normalized, new_otp)
            return jsonify({
                "error": "Too many attempts. A new OTP has been generated.",
                "new_otp": new_otp
            }), 401
        else:
            return jsonify({"error": "Invalid OTP"}), 401
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/reset_password", methods=["POST"])
def reset_password():
    try:
        if "reset_verified" not in session:
            return jsonify({"error": "Unauthorized. Please verify OTP first"}), 401
            
        data = request.get_json()
        new_password = data.get("new_password", "").strip()
        
        if len(new_password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
            
        user_id = session["reset_verified"]
        users = load_users()
        
        if user_id not in users:
            return jsonify({"error": "User not found"}), 404
            
        users[user_id]["password"] = hash_password(new_password)
        save_users(users)
        session.pop("reset_verified", None)
        return jsonify({"success": True, "message": "Password reset successful"})
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

@app.route("/check_auth", methods=["GET"])
def check_auth():
    if "user_id" in session:
        return jsonify({"authenticated": True, "user": session["user_id"]})
    return jsonify({"authenticated": False})

# ==========================
# PDF & CHAT ROUTES
# ==========================

user_data = {}
user_greetings = {}  # Store greeting status per user

def get_user_data():
    user_id = session.get("user_id")
    if not user_id:
        return None
    if user_id not in user_data:
        user_data[user_id] = {
            "vectorizer": None,
            "matrix": None,
            "stored_chunks": []
        }
    return user_data[user_id]

def extract_pdf(path):
    doc = fitz.open(path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def build_db(text, user_data_obj):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c.strip()) > 80][:60]
    if not chunks:
        raise Exception("No readable text found")
    user_data_obj["stored_chunks"] = chunks
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)
    user_data_obj["vectorizer"] = vectorizer
    user_data_obj["matrix"] = matrix
    return len(chunks)

def build_prompt(question, context):
    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If answer is not found, say "Not found in document".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def generate_greeting(filename, chunks_count):
    """Generate a personalized greeting based on the uploaded document"""
    greetings = [
        f"🎉 Awesome! I've just finished reading '{filename}'. It's broken down into {chunks_count} sections. Feel free to ask me anything about it!",
        f"📚 Great! '{filename}' is now ready for queries. I've split it into {chunks_count} parts for better understanding. What would you like to know?",
        f"✨ Perfect! Your document '{filename}' has been successfully processed into {chunks_count} chunks. I'm here to help you explore its contents!",
        f"📖 Excellent! '{filename}' is loaded and ready. With {chunks_count} sections to reference, I can answer your questions in detail. Ask away!",
        f"🚀 Success! I've indexed '{filename}' with {chunks_count} chunks. Now you can ask me anything about the document content!",
        f"💡 All set! '{filename}' has been processed into {chunks_count} chunks. I'm ready to assist you with any questions!",
        f"📑 Done! '{filename}' is now in my knowledge base with {chunks_count} sections. What would you like to explore?",
        f"🌟 Fantastic! I've analyzed '{filename}' and created {chunks_count} reference points. How can I help you today?",
        f"📊 Ready to go! '{filename}' has been chunked into {chunks_count} pieces for precise answers. Your questions are welcome!",
        f"🎯 Perfect upload! '{filename}' is fully processed with {chunks_count} chunks. I'm excited to help you dive into this document!"
    ]
    return random.choice(greetings)

@app.route("/")
def home():
    """Main application page"""
    return render_template("index.html")

@app.route("/login-page")
def login_page():
    """Separate login page"""
    return render_template("login.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session:
        return jsonify({"error": "Please login first"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    filename = file.filename
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        path = tmp.name
    text = extract_pdf(path)
    os.remove(path)
    user_data_obj = get_user_data()
    if not user_data_obj:
        return jsonify({"error": "Session error"}), 401
    chunks = build_db(text, user_data_obj)
    
    # Generate and store greeting for this user
    user_id = session["user_id"]
    greeting = generate_greeting(filename, chunks)
    user_greetings[user_id] = greeting
    
    return jsonify({
        "ready": True, 
        "chunks": chunks,
        "greeting": greeting,
        "filename": filename
    })

@app.route("/get_greeting", methods=["GET"])
def get_greeting():
    """Get the stored greeting for the current user"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    greeting = user_greetings.get(user_id)
    if greeting:
        return jsonify({"greeting": greeting})
    return jsonify({"greeting": None})

@app.route("/ask", methods=["POST"])
def ask():
    if "user_id" not in session:
        return jsonify({"answer": "Please login first"}), 401
    
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Empty question"})
    
    # Check if it's a casual greeting
    if is_casual_greeting(question):
        response = get_casual_response(question)
        return jsonify({"answer": response, "type": "greeting"})
    
    if groq_client is None:
        return jsonify({"answer": "Groq API not configured. Please check your API key."}), 500
        
    user_data_obj = get_user_data()
    if not user_data_obj or user_data_obj["vectorizer"] is None:
        return jsonify({"answer": "Please upload a PDF first to ask questions about it."})
        
    try:
        q_vec = user_data_obj["vectorizer"].transform([question])
        scores = (user_data_obj["matrix"] @ q_vec.T).toarray().ravel()
        top_idx = scores.argsort()[-5:][::-1]
        context = "\n\n".join([user_data_obj["stored_chunks"][i] for i in top_idx])
        
        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": build_prompt(question, context)}],
            temperature=0.2
        )
        answer = completion.choices[0].message.content
        
        # If the answer is "Not found in document" and it's a question, provide a helpful response
        if "not found" in answer.lower() and len(question) > 10:
            answer = "I couldn't find that specific information in the document. Could you rephrase your question or ask about something else from the PDF?"
        
        return jsonify({"answer": answer, "type": "document"})
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route("/health")
def health():
    status = {
        "status": "running",
        "groq_available": groq_client is not None,
        "users_count": len(load_users()),
        "otp_storage": len(OTP_STORAGE)
    }
    return jsonify(status)

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    logger.info(f"🚀 Starting server on port {port}")
    logger.info(f"🔑 GROQ configured: {bool(GROQ_API_KEY)}")
    
    app.run(host="0.0.0.0", port=port, debug=True)