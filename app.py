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
import requests  # needed for DNS fallback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

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
# DNS FALLBACK: get IP of smtp.gmail.com
# ==========================
def get_smtp_ip():
    """Fetch current IP of smtp.gmail.com using Google's DNS-over-HTTPS."""
    try:
        resp = requests.get(
            "https://dns.google/resolve?name=smtp.gmail.com&type=A",
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("Answer"):
                for ans in data["Answer"]:
                    if ans["type"] == 1:  # A record
                        return ans["data"]
    except Exception:
        pass
    return None

# ==========================
# OTP SENDING (with IP fallback)
# ==========================
def send_otp_via_gmail(to_email, otp):
    if not GMAIL_SENDER or not GMAIL_PASSWORD:
        print("❌ Gmail not configured")
        return False

    # Build the email
    msg = MIMEMultipart()
    msg['From'] = GMAIL_SENDER
    msg['To'] = to_email
    msg['Subject'] = "Password Reset OTP - PDF Assistant"
    body = f"""
    <html>
    <body style="font-family: Arial; max-width:600px; margin:auto;">
        <h2 style="color:#667eea;">Password Reset OTP</h2>
        <p>Your OTP is:</p>
        <h1 style="font-size:32px; letter-spacing:5px; background:#f0f0f0; padding:15px; border-radius:8px;">{otp}</h1>
        <p><i>Valid for 10 minutes.</i></p>
        <p>If you didn't request this, ignore this email.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, 'html'))

    smtp_host = "smtp.gmail.com"
    smtp_ports = [587, 465]  # TLS then SSL

    for port in smtp_ports:
        try:
            # Try with hostname first
            if port == 587:
                server = smtplib.SMTP(smtp_host, port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(smtp_host, port, timeout=10)

            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"✅ OTP sent via Gmail to {to_email} (hostname)")
            return True

        except (socket.gaierror, socket.error) as dns_err:
            # DNS resolution failed – try IP fallback
            print(f"⚠️ Hostname failed: {dns_err}. Trying IP fallback...")
            ip = get_smtp_ip()
            if not ip:
                print("❌ Could not fetch IP for smtp.gmail.com")
                continue
            try:
                if port == 587:
                    server = smtplib.SMTP(ip, port, timeout=10)
                    server.ehlo(smtp_host)   # EHLO with proper hostname
                    server.starttls()
                    server.ehlo(smtp_host)
                else:
                    server = smtplib.SMTP_SSL(ip, port, timeout=10)
                    server.ehlo(smtp_host)

                server.login(GMAIL_SENDER, GMAIL_PASSWORD)
                server.send_message(msg)
                server.quit()
                print(f"✅ OTP sent via Gmail to {to_email} (IP fallback: {ip})")
                return True
            except Exception as e:
                print(f"❌ IP fallback failed: {e}")
                continue
        except Exception as e:
            print(f"❌ SMTP error on port {port}: {e}")
            continue

    print("❌ All connection attempts failed.")
    return False

def send_otp(email, otp):
    # Debug – you can remove this line if you don't want it
    print(f"🔑 OTP for {email}: {otp}")
    return send_otp_via_gmail(email, otp)

def store_otp(email, otp):
    OTP_STORAGE[email] = {"otp": otp, "expires": time.time() + 600}

def verify_otp(email, otp):
    if email in OTP_STORAGE:
        stored = OTP_STORAGE[email]
        if time.time() < stored["expires"] and stored["otp"] == otp:
            del OTP_STORAGE[email]
            return True
    return False

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
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    if not email:
        return jsonify({"error": "Email required"}), 400
    users = load_users()
    if email not in users:
        return jsonify({"error": "No account found with this email"}), 404
    otp = generate_otp()
    if send_otp(email, otp):
        store_otp(email, otp)
        return jsonify({"success": True, "message": "OTP sent to your email"})
    else:
        return jsonify({"error": "Failed to send OTP. Check network or Gmail credentials."}), 500

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    otp = data.get("otp", "").strip()
    if not email or not otp:
        return jsonify({"error": "All fields required"}), 400
    if verify_otp(email, otp):
        session["reset_verified"] = email
        return jsonify({"success": True, "message": "OTP verified"})
    else:
        return jsonify({"error": "Invalid or expired OTP"}), 401

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)