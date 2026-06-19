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
import smtplib
import socket
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
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
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ==========================
# ENVIRONMENT VARIABLES
# ==========================
GMAIL_SENDER = os.environ.get("GMAIL_SENDER")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.1-8b-instant"

# ==========================
# USER DATABASE
# ==========================
USERS_FILE = "users.json"
OTP_STORAGE = {}

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def generate_otp():
    return f"{random.randint(100000, 999999)}"

# ==========================
# FIXED: EMAIL SENDING WITH TIMEOUT
# ==========================
def send_otp_via_gmail(to_email, otp):
    """Send OTP via Gmail with timeout protection"""
    try:
        logger.info(f"Attempting to send OTP to {to_email}")
        
        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = "Password Reset OTP - PDF Assistant"

        body = f"""
Hello,

Your OTP for verification is:

{otp}

This OTP is valid for 10 minutes.

Regards,
PDF Assistant
"""

        msg.attach(MIMEText(body, "plain"))

        # Try multiple SMTP servers with timeout
        smtp_servers = [
            ("smtp.gmail.com", 587),
            ("smtp.gmail.com", 465),  # SSL
        ]
        
        for host, port in smtp_servers:
            try:
                logger.info(f"Trying {host}:{port}")
                
                if port == 465:
                    # SSL connection
                    server = smtplib.SMTP_SSL(host, port, timeout=10)
                else:
                    # TLS connection
                    server = smtplib.SMTP(host, port, timeout=10)
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                
                server.login(GMAIL_SENDER, GMAIL_PASSWORD)
                server.sendmail(GMAIL_SENDER, to_email, msg.as_string())
                server.quit()
                
                logger.info(f"✅ OTP sent successfully to {to_email} via {host}:{port}")
                return True
                
            except smtplib.SMTPAuthenticationError:
                logger.error("❌ SMTP Authentication failed - check GMAIL_SENDER and GMAIL_PASSWORD")
                return False
            except (smtplib.SMTPException, socket.timeout, ConnectionError) as e:
                logger.warning(f"Failed with {host}:{port}: {e}")
                continue
        
        # If all attempts fail, log the OTP for development
        logger.warning(f"⚠️ All SMTP attempts failed. OTP for {to_email}: {otp}")
        return False

    except Exception as e:
        logger.error(f"❌ Gmail Error: {e}")
        return False

def send_otp(email, otp):
    """Send OTP with fallback to console logging"""
    logger.info(f"🔑 OTP for {email}: {otp}")
    
    # Try to send via email
    success = send_otp_via_gmail(email, otp)
    
    # If email fails but we're in development, still allow OTP
    if not success and os.environ.get("ENV") != "production":
        logger.warning(f"⚠️ Email failed but OTP is: {otp} (check logs)")
        return True
    
    return success

def store_otp(email, otp):
    OTP_STORAGE[email] = {"otp": otp, "expires": time.time() + 600, "attempts": 0}

def verify_otp(email, otp):
    if email in OTP_STORAGE:
        stored = OTP_STORAGE[email]

        if time.time() > stored["expires"]:
            return "expired"

        if stored["otp"] == otp:
            del OTP_STORAGE[email]
            return "valid"

        stored["attempts"] += 1

        if stored["attempts"] >= 3:
            return "resend"

        return "invalid"

    return "invalid"

# ==========================
# AUTHENTICATION ROUTES
# ==========================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    if not email or not password:
        return jsonify({"error": "All fields required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if not validate_email(email):
        return jsonify({"error": "Invalid email"}), 400
    users = load_users()
    if email in users:
        return jsonify({"error": "Email already registered"}), 400
    users[email] = {"password": hash_password(password), "created_at": datetime.now().isoformat()}
    save_users(users)
    session["user_id"] = email
    return jsonify({"success": True, "message": "Signup successful", "user": email})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    if not email or not password:
        return jsonify({"error": "All fields required"}), 400
    users = load_users()
    if email not in users or not verify_password(password, users[email]["password"]):
        return jsonify({"error": "Invalid credentials"}), 401
    session["user_id"] = email
    return jsonify({"success": True, "message": "Login successful", "user": email})

@app.route("/forgot_password", methods=["POST"])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        
        if not email:
            return jsonify({"error": "Email required"}), 400
        
        users = load_users()
        if email not in users:
            return jsonify({"error": "No account found with this email"}), 404
        
        otp = generate_otp()
        
        # Send OTP with timeout
        success = send_otp(email, otp)
        
        if success:
            store_otp(email, otp)
            return jsonify({"success": True, "message": "OTP sent to your email"})
        else:
            # Still store OTP for testing but return error
            store_otp(email, otp)
            return jsonify({"error": "Failed to send OTP. Check your email settings."}), 500
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    otp = data.get("otp", "").strip()

    if not email or not otp:
        return jsonify({"error": "All fields required"}), 400

    result = verify_otp(email, otp)
    
    if result == "valid":
        session["reset_verified"] = email
        return jsonify({"success": True, "message": "OTP verified"})
    elif result == "expired":
        return jsonify({"error": "OTP expired. Request a new one."}), 401
    else:
        # Wrong OTP → generate and resend a new OTP
        new_otp = generate_otp()
        
        if send_otp(email, new_otp):
            store_otp(email, new_otp)
            return jsonify({
                "error": "Invalid OTP. A new OTP has been sent to your email."
            }), 401
        
        return jsonify({
            "error": "Invalid OTP and failed to resend a new OTP."
        }), 500

@app.route("/reset_password", methods=["POST"])
def reset_password():
    if "reset_verified" not in session:
        return jsonify({"error": "Unauthorized. Please verify OTP first"}), 401
    data = request.get_json()
    new_password = data.get("new_password", "").strip()
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    email = session["reset_verified"]
    users = load_users()
    if email not in users:
        return jsonify({"error": "User not found"}), 404
    users[email]["password"] = hash_password(new_password)
    save_users(users)
    session.pop("reset_verified", None)
    return jsonify({"success": True, "message": "Password reset successful"})

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session:
        return jsonify({"error": "Please login first"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        path = tmp.name
    text = extract_pdf(path)
    os.remove(path)
    user_data_obj = get_user_data()
    if not user_data_obj:
        return jsonify({"error": "Session error"}), 401
    chunks = build_db(text, user_data_obj)
    return jsonify({"ready": True, "chunks": chunks})

@app.route("/ask", methods=["POST"])
def ask():
    if "user_id" not in session:
        return jsonify({"answer": "Please login first"}), 401
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Empty question"})
    user_data_obj = get_user_data()
    if not user_data_obj or user_data_obj["vectorizer"] is None:
        return jsonify({"answer": "Upload PDF first"})
    q_vec = user_data_obj["vectorizer"].transform([question])
    scores = (user_data_obj["matrix"] @ q_vec.T).toarray().ravel()
    top_idx = scores.argsort()[-5:][::-1]
    context = "\n\n".join([user_data_obj["stored_chunks"][i] for i in top_idx])
    completion = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(question, context)}],
        temperature=0.2
    )
    return jsonify({"answer": completion.choices[0].message.content})

@app.route("/health")
def health():
    return jsonify({"status": "running"})

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)