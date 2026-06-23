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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import ssl
import sys
import requests
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================
# APP SETUP
# ==========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload

# ==========================
# ENVIRONMENT VARIABLES
# ==========================
GMAIL_SENDER = os.environ.get("GMAIL_SENDER")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.environ.get("RESEND_FROM_EMAIL")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL")

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
USERS_FILE = "users.json"
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

def generate_otp():
    return f"{random.randint(100000, 999999)}"

# ==========================
# EMAIL PROVIDERS (No Mailgun)
# ==========================

def send_otp_via_resend(to_email, otp):
    """
    PRIMARY: Send OTP via Resend API (HTTPS - most reliable)
    Get API key from: https://resend.com
    """
    if not RESEND_API_KEY or not RESEND_FROM_EMAIL:
        logger.warning("⚠️ Resend not configured")
        return False
    
    try:
        logger.info(f"📧 Attempting to send via Resend to {to_email}")
        
        url = "https://api.resend.com/emails"
        headers = {
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "from": RESEND_FROM_EMAIL,
            "to": [to_email],
            "subject": "🔑 Password Reset OTP - PDF Assistant",
            "text": f"""Your OTP for password reset is: {otp}

This OTP is valid for 10 minutes.

If you didn't request this, please ignore this email.

Best regards,
PDF Assistant""",
            "html": f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }}
        .otp {{ 
            font-size: 36px; 
            font-weight: bold; 
            letter-spacing: 8px;
            background: #f0f4ff;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            color: #2563eb;
        }}
        .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <h2>🔑 Password Reset OTP</h2>
    <p>Your OTP for password reset is:</p>
    <div class="otp">{otp}</div>
    <p><strong>Valid for 10 minutes</strong></p>
    <p>If you didn't request this, please ignore this email.</p>
    <div class="footer">PDF Assistant - Secure Authentication</div>
</body>
</html>"""
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=15)
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"✅ OTP sent via Resend to {to_email}")
            return True
        else:
            logger.error(f"❌ Resend error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Resend error: {e}")
        return False

def send_otp_via_sendgrid(to_email, otp):
    """SECONDARY: Send OTP via SendGrid API (HTTPS)"""
    if not SENDGRID_API_KEY or not SENDGRID_FROM_EMAIL:
        logger.warning("⚠️ SendGrid not configured")
        return False
    
    try:
        logger.info(f"📧 Attempting to send via SendGrid to {to_email}")
        
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {
            "Authorization": f"Bearer {SENDGRID_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "personalizations": [{"to": [{"email": to_email}]}],
            "from": {"email": SENDGRID_FROM_EMAIL},
            "subject": "Password Reset OTP - PDF Assistant",
            "content": [{
                "type": "text/plain",
                "value": f"Your OTP for password reset is: {otp}\n\nThis OTP is valid for 10 minutes."
            }]
        }
        response = requests.post(url, json=data, headers=headers, timeout=10)
        if response.status_code == 202:
            logger.info(f"✅ OTP sent via SendGrid to {to_email}")
            return True
        else:
            logger.error(f"❌ SendGrid error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ SendGrid error: {e}")
        return False

def send_otp_via_gmail(to_email, otp):
    """FALLBACK: Send OTP via Gmail SMTP (with network fixes)"""
    if not GMAIL_SENDER or not GMAIL_PASSWORD:
        logger.warning("⚠️ Gmail credentials not configured")
        return False
    
    try:
        logger.info(f"📧 Attempting to send via Gmail to {to_email}")
        
        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = "🔑 Password Reset OTP - PDF Assistant"
        
        body = f"""
Hello,

Your OTP for password reset is:

🔑 {otp}

This OTP is valid for 10 minutes.

If you didn't request this, please ignore this email.

Best regards,
PDF Assistant
"""

        msg.attach(MIMEText(body, "plain"))
        
        # Try multiple configurations for network resilience
        smtp_configs = [
            # Standard TLS
            {"host": "smtp.gmail.com", "port": 587, "use_tls": True, "timeout": 30},
            # SSL
            {"host": "smtp.gmail.com", "port": 465, "use_ssl": True, "timeout": 30},
            # Force IPv4
            {"host": "smtp.gmail.com", "port": 587, "use_tls": True, "timeout": 45, "force_ipv4": True},
        ]
        
        for config in smtp_configs:
            try:
                if config.get("force_ipv4"):
                    # Create IPv4 socket explicitly
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(config.get("timeout", 30))
                    sock.connect((config["host"], config["port"]))
                    
                    if config.get("use_ssl", False):
                        context = ssl.create_default_context()
                        server = smtplib.SMTP_SSL()
                        server.sock = context.wrap_socket(sock, server_hostname=config["host"])
                    else:
                        server = smtplib.SMTP()
                        server.sock = sock
                else:
                    if config.get("use_ssl", False):
                        context = ssl.create_default_context()
                        server = smtplib.SMTP_SSL(
                            config["host"], 
                            config["port"], 
                            timeout=config.get("timeout", 30),
                            context=context
                        )
                    else:
                        server = smtplib.SMTP(
                            config["host"], 
                            config["port"], 
                            timeout=config.get("timeout", 30)
                        )
                
                if config.get("use_tls"):
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                
                server.login(GMAIL_SENDER, GMAIL_PASSWORD)
                server.sendmail(GMAIL_SENDER, to_email, msg.as_string())
                server.quit()
                
                logger.info(f"✅ OTP sent via Gmail to {to_email}")
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Gmail config {config.get('port')} failed: {e}")
                continue
        
        return False
                
    except Exception as e:
        logger.error(f"❌ Gmail error: {e}")
        return False

def send_otp_email(to_email, otp):
    """
    Send OTP using multiple providers with intelligent fallback
    Priority: Resend > SendGrid > Gmail SMTP
    """
    logger.info(f"🔑 OTP for {to_email}: {otp}")
    
    # Always store OTP first
    store_otp(to_email, otp)
    
    # Try providers in order of reliability (HTTPS APIs first)
    providers = [
        ("Resend", send_otp_via_resend),
        ("SendGrid", send_otp_via_sendgrid),
        ("Gmail", send_otp_via_gmail),
    ]
    
    for provider_name, provider_func in providers:
        try:
            logger.info(f"🔄 Trying {provider_name}...")
            success = provider_func(to_email, otp)
            if success:
                logger.info(f"✅ Successfully sent via {provider_name}")
                return True
        except Exception as e:
            logger.error(f"❌ {provider_name} failed with error: {e}")
            continue
    
    logger.warning(f"⚠️ All email providers failed for {to_email}")
    return False

def store_otp(email, otp):
    OTP_STORAGE[email] = {
        "otp": otp, 
        "expires": time.time() + 600,  # 10 minutes
        "attempts": 0,
        "created_at": datetime.now().isoformat()
    }
    logger.info(f"✅ OTP stored for {email}")

def verify_otp(email, otp):
    if email in OTP_STORAGE:
        stored = OTP_STORAGE[email]
        
        if time.time() > stored["expires"]:
            del OTP_STORAGE[email]
            logger.warning(f"⏰ OTP expired for {email}")
            return "expired"
        
        if stored["otp"] == otp:
            del OTP_STORAGE[email]
            logger.info(f"✅ OTP verified for {email}")
            return "valid"
        
        stored["attempts"] += 1
        logger.warning(f"❌ Invalid OTP attempt {stored['attempts']}/3 for {email}")
        
        if stored["attempts"] >= 3:
            # Auto-generate new OTP
            new_otp = generate_otp()
            store_otp(email, new_otp)
            send_otp_email(email, new_otp)
            return "resend"
        
        return "invalid"
    
    logger.warning(f"❌ No OTP found for {email}")
    return "invalid"

def get_otp_from_logs(email):
    """Helper function to get OTP from storage (for debugging)"""
    if email in OTP_STORAGE:
        return OTP_STORAGE[email]["otp"]
    return None

# ==========================
# AUTHENTICATION ROUTES
# ==========================
@app.route("/signup", methods=["POST"])
def signup():
    try:
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
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
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
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Internal server error"}), 500

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
        
        # Send OTP via email with automatic fallback
        success = send_otp_email(email, otp)
        
        # Always return success for security (don't reveal if email exists)
        message = "OTP sent to your email" + (" (check spam folder)" if not success else "")
        
        # In development, include OTP in response for testing
        if os.environ.get("FLASK_ENV") == "development":
            message += f" [DEV: {otp}]"
        
        return jsonify({
            "success": True, 
            "message": message
        })
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    try:
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
            return jsonify({
                "error": "OTP expired. A new OTP has been sent."
            }), 401
        elif result == "resend":
            return jsonify({
                "error": "Too many attempts. A new OTP has been sent."
            }), 401
        else:
            return jsonify({
                "error": "Invalid OTP. Please try again."
            }), 401
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        
        if not email:
            return jsonify({"error": "Email required"}), 400
            
        users = load_users()
        if email not in users:
            return jsonify({"error": "No account found with this email"}), 404
        
        # Clear existing OTP
        if email in OTP_STORAGE:
            del OTP_STORAGE[email]
        
        # Generate and send new OTP
        otp = generate_otp()
        success = send_otp_email(email, otp)
        
        return jsonify({
            "success": True,
            "message": "New OTP sent" + (" (check spam folder)" if not success else "")
        })
            
    except Exception as e:
        logger.error(f"Resend OTP error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_otp_for_testing", methods=["POST"])
def get_otp_for_testing():
    """Development endpoint to retrieve OTP (remove in production)"""
    if os.environ.get("FLASK_ENV") != "development":
        return jsonify({"error": "Not available in production"}), 403
    
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    
    if email in OTP_STORAGE:
        return jsonify({
            "otp": OTP_STORAGE[email]["otp"],
            "expires": OTP_STORAGE[email]["expires"]
        })
    return jsonify({"error": "No OTP found"}), 404

@app.route("/reset_password", methods=["POST"])
def reset_password():
    try:
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
    
    if groq_client is None:
        return jsonify({"answer": "Groq API not configured. Please check your API key."}), 500
        
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Empty question"})
        
    user_data_obj = get_user_data()
    if not user_data_obj or user_data_obj["vectorizer"] is None:
        return jsonify({"answer": "Upload PDF first"})
        
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
        return jsonify({"answer": completion.choices[0].message.content})
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500

@app.route("/health")
def health():
    status = {
        "status": "running",
        "groq_available": groq_client is not None,
        "users_count": len(load_users()),
        "otp_storage": len(OTP_STORAGE),
        "email_providers": {
            "resend": bool(RESEND_API_KEY and RESEND_FROM_EMAIL),
            "sendgrid": bool(SENDGRID_API_KEY and SENDGRID_FROM_EMAIL),
            "gmail": bool(GMAIL_SENDER and GMAIL_PASSWORD)
        }
    }
    return jsonify(status)

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)