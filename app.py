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
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

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
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Thread pool for background email sending
email_executor = ThreadPoolExecutor(max_workers=2)

# ==========================
# ENVIRONMENT VARIABLES
# ==========================
GMAIL_SENDER = os.environ.get("GMAIL_SENDER")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Optional: SendGrid for backup
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.environ.get("SENDGRID_FROM_EMAIL")

# Initialize Groq client with error handling
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
# Use persistent volume in Railway
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

def generate_otp():
    return f"{random.randint(100000, 999999)}"

# ==========================
# EMAIL SENDING - RAILWAY OPTIMIZED
# ==========================

def send_otp_via_gmail_async(to_email, otp):
    """Send OTP via Gmail in background thread - Optimized for Railway"""
    if not GMAIL_SENDER or not GMAIL_PASSWORD:
        logger.error("❌ Gmail credentials not configured")
        return False
        
    try:
        # Create message
        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER
        msg["To"] = to_email
        msg["Subject"] = "Password Reset OTP - PDF Assistant"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: #f4f6f9;">
            <div style="background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h2 style="color: #1a237e; margin-bottom: 20px;">🔐 Password Reset OTP</h2>
                <p style="color: #333; font-size: 16px;">Hello,</p>
                <p style="color: #333; font-size: 16px;">You requested to reset your password. Use the OTP below to verify your identity:</p>
                <div style="background-color: #e8eaf6; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                    <span style="font-size: 32px; font-weight: bold; color: #1a237e; letter-spacing: 5px;">{otp}</span>
                </div>
                <p style="color: #666; font-size: 14px;">This OTP is valid for <strong>10 minutes</strong>.</p>
                <p style="color: #666; font-size: 14px;">If you didn't request this, please ignore this email.</p>
                <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 20px 0;">
                <p style="color: #999; font-size: 12px; text-align: center;">PDF Research Assistant</p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))
        
        text_body = f"""
        Password Reset OTP
        
        Your OTP is: {otp}
        
        This OTP is valid for 10 minutes.
        If you didn't request this, please ignore this email.
        
        PDF Research Assistant
        """
        msg.attach(MIMEText(text_body, "plain"))

        # Try SMTP with Railway optimized settings
        try:
            logger.info(f"Attempting to send OTP to {to_email} via Gmail...")
            # Use socket timeout
            socket.setdefaulttimeout(15)
            
            # Try TLS first
            context = ssl.create_default_context()
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
            server.set_debuglevel(0)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, to_email, msg.as_string())
            server.quit()
            logger.info(f"✅ OTP sent successfully via Gmail TLS to {to_email}")
            return True
        except Exception as e1:
            logger.warning(f"TLS method failed: {e1}")
            
            # Try SSL as fallback
            try:
                logger.info(f"Trying Gmail SSL for {to_email}...")
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15, context=context)
                server.set_debuglevel(0)
                server.ehlo()
                server.login(GMAIL_SENDER, GMAIL_PASSWORD)
                server.sendmail(GMAIL_SENDER, to_email, msg.as_string())
                server.quit()
                logger.info(f"✅ OTP sent successfully via Gmail SSL to {to_email}")
                return True
            except Exception as e2:
                logger.error(f"SSL method also failed: {e2}")
                raise e2

    except Exception as e:
        logger.error(f"❌ Gmail Error for {to_email}: {e}")
        return False

def send_otp_via_sendgrid_async(to_email, otp):
    """Send OTP via SendGrid as backup"""
    if not SENDGRID_API_KEY:
        return False
    
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
        message = Mail(
            from_email=SENDGRID_FROM_EMAIL or GMAIL_SENDER,
            to_emails=to_email,
            subject="Password Reset OTP - PDF Assistant",
            html_content=f"""
            <html>
            <body>
                <h2>🔐 Password Reset OTP</h2>
                <p>Your OTP is: <strong style="font-size: 24px;">{otp}</strong></p>
                <p>This OTP is valid for 10 minutes.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
            </html>
            """
        )
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code in [200, 202]:
            logger.info(f"✅ OTP sent via SendGrid to {to_email}")
            return True
        else:
            logger.error(f"SendGrid error: {response.status_code}")
            return False
    except ImportError:
        logger.warning("SendGrid not installed")
        return False
    except Exception as e:
        logger.error(f"SendGrid error: {e}")
        return False

def send_otp_background(email, otp):
    """Send OTP in background thread - returns immediately"""
    logger.info(f"📧 Queuing OTP for {email}: {otp}")
    
    def send_worker():
        success = False
        methods_tried = []
        
        # Method 1: Try Gmail
        try:
            if send_otp_via_gmail_async(email, otp):
                success = True
                methods_tried.append("✅ Gmail")
            else:
                methods_tried.append("❌ Gmail")
        except Exception as e:
            methods_tried.append(f"❌ Gmail: {str(e)[:30]}")
        
        # Method 2: Try SendGrid if configured
        if not success and SENDGRID_API_KEY:
            try:
                if send_otp_via_sendgrid_async(email, otp):
                    success = True
                    methods_tried.append("✅ SendGrid")
                else:
                    methods_tried.append("❌ SendGrid")
            except Exception as e:
                methods_tried.append(f"❌ SendGrid: {str(e)[:30]}")
        
        logger.info(f"Email methods for {email}: {', '.join(methods_tried)}")
        
        if not success:
            logger.warning(f"⚠️ All email methods failed. OTP for {email}: {otp}")
            logger.warning(f"📧 Please use this OTP from server logs: {otp}")
    
    # Submit to thread pool
    email_executor.submit(send_worker)
    return True

def send_otp(email, otp):
    """Send OTP - simplified version that returns immediately"""
    logger.info(f"🔑 OTP generated for {email}: {otp}")
    return send_otp_background(email, otp)

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
        
        # Store OTP
        store_otp(email, otp)
        
        # Send OTP asynchronously - returns immediately
        send_otp(email, otp)
        
        # Return success immediately without waiting for email
        return jsonify({
            "success": True, 
            "message": "OTP sent to your email. Check spam folder."
        })
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/resend_otp", methods=["POST"])
def resend_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        
        if not email:
            return jsonify({"error": "Email required"}), 400
            
        # Generate new OTP
        otp = generate_otp()
        store_otp(email, otp)
        
        # Send OTP asynchronously
        send_otp(email, otp)
        
        return jsonify({
            "success": True,
            "message": "New OTP sent"
        })
    except Exception as e:
        logger.error(f"Resend OTP error: {e}")
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
            return jsonify({"error": "OTP expired. Request a new one."}), 401
        elif result == "resend":
            # Generate new OTP
            new_otp = generate_otp()
            store_otp(email, new_otp)
            send_otp(email, new_otp)
            return jsonify({
                "error": "Too many attempts. A new OTP has been sent."
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
        "gmail_configured": bool(GMAIL_SENDER and GMAIL_PASSWORD),
        "sendgrid_configured": bool(SENDGRID_API_KEY)
    }
    return jsonify(status)

# ==========================
# RUN - RAILWAY COMPATIBLE
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)