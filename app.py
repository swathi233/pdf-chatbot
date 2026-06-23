from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import json
import logging
import sys
import random
import time
from dotenv import load_dotenv
from email_utils import send_otp_email

load_dotenv()

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
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

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

def generate_otp():
    return f"{random.randint(100000, 999999)}"

def store_otp(email, otp):
    OTP_STORAGE[email] = {
        "otp": otp,
        "expires": time.time() + 600
    }

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return """
    <h1>PDF Assistant API</h1>
    <p>Server is running!</p>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/health">/health</a> - Check server status</li>
        <li>/signup - Create account</li>
        <li>/login - Login</li>
        <li>/forgot_password - Request OTP</li>
        <li>/verify_reset_otp - Verify OTP</li>
        <li>/reset_password - Reset password</li>
    </ul>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "message": "Server is healthy",
        "users_count": len(load_users()),
        "email_configured": bool(os.environ.get("RESEND_API_KEY"))
    })

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
            
        users = load_users()
        if email in users:
            return jsonify({"error": "Email already registered"}), 400
            
        users[email] = {"password": password}
        save_users(users)
        session["user_id"] = email
        return jsonify({"success": True, "message": "Signup successful", "user": email})
    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        
        if not email or not password:
            return jsonify({"error": "All fields required"}), 400
            
        users = load_users()
        if email not in users or users[email]["password"] != password:
            return jsonify({"error": "Invalid credentials"}), 401
            
        session["user_id"] = email
        return jsonify({"success": True, "message": "Login successful", "user": email})
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": str(e)}), 500

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
        store_otp(email, otp)
        
        # Try to send email
        success = send_otp_email(email, otp)
        
        if success:
            return jsonify({"success": True, "message": "OTP sent to your email"})
        else:
            # In development, return OTP so user can test
            if os.environ.get("FLASK_ENV") == "development":
                return jsonify({"success": True, "message": f"OTP: {otp} (check logs for details)"})
            else:
                return jsonify({"success": True, "message": "OTP sent to your email (check spam folder)"})
            
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/verify_reset_otp", methods=["POST"])
def verify_reset_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        otp = data.get("otp", "").strip()

        if not email or not otp:
            return jsonify({"error": "All fields required"}), 400

        if email not in OTP_STORAGE:
            return jsonify({"error": "No OTP found"}), 401
            
        stored = OTP_STORAGE[email]
        
        if time.time() > stored["expires"]:
            del OTP_STORAGE[email]
            return jsonify({"error": "OTP expired"}), 401
        
        if stored["otp"] == otp:
            del OTP_STORAGE[email]
            session["reset_verified"] = email
            return jsonify({"success": True, "message": "OTP verified"})
        else:
            return jsonify({"error": "Invalid OTP"}), 401
            
    except Exception as e:
        logger.error(f"Verify OTP error: {e}")
        return jsonify({"error": str(e)}), 500

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
            
        users[email]["password"] = new_password
        save_users(users)
        session.pop("reset_verified", None)
        return jsonify({"success": True, "message": "Password reset successful"})
    except Exception as e:
        logger.error(f"Reset password error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/check_auth", methods=["GET"])
def check_auth():
    if "user_id" in session:
        return jsonify({"authenticated": True, "user": session["user_id"]})
    return jsonify({"authenticated": False})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out"})

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)