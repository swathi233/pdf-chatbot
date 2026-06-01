from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import time
import fitz  # PyMuPDF

from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
# APP SETUP
# ==========================

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# ==========================
# GROQ (FIXED MODEL)
# ==========================

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ✅ FIXED MODEL (no longer decommissioned)
MODEL = "llama-3.1-8b-instant"

# ==========================
# GLOBAL STORAGE
# ==========================

vectorizer = None
matrix = None
stored_chunks = []

# ==========================
# PDF EXTRACTION
# ==========================

def extract_pdf(path):
    try:
        doc = fitz.open(path)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        return text

    except Exception as e:
        print("PDF ERROR:", e)
        return ""

# ==========================
# BUILD INDEX (REAL RAG)
# ==========================

def build_db(text):

    global vectorizer, matrix, stored_chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    chunks = [c for c in chunks if len(c.strip()) > 80]

    # safety limit for Railway
    chunks = chunks[:60]

    if not chunks:
        raise Exception("No readable text found in PDF")

    stored_chunks = chunks

    # REAL vectorization (FAST + NO CRASH)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(chunks)

    print("Chunks indexed:", len(chunks))

    return len(chunks)

# ==========================
# PROMPT
# ==========================

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

# ==========================
# HOME
# ==========================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# UPLOAD (FIXED)
# ==========================

@app.route("/upload", methods=["POST"])
def upload():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp.name)
            path = tmp.name

        print("PDF saved")

        text = extract_pdf(path)

        os.remove(path)

        print("Text length:", len(text))

        start = time.time()

        chunks = build_db(text)

        elapsed = round(time.time() - start, 2)

        return jsonify({
            "ready": True,
            "chunks": chunks,
            "embed_time": elapsed
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({"error": str(e)}), 500

# ==========================
# ASK (REAL PDF CHAT FIXED)
# ==========================

@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Empty question"})

        if vectorizer is None:
            return jsonify({"answer": "Upload PDF first"})

        # TF-IDF SEARCH (REAL RAG)
        q_vec = vectorizer.transform([question])
        scores = (matrix @ q_vec.T).toarray().ravel()

        top_idx = scores.argsort()[-5:][::-1]
        context = "\n\n".join([stored_chunks[i] for i in top_idx])

        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": build_prompt(question, context)
            }],
            temperature=0.2
        )

        return jsonify({
            "answer": completion.choices[0].message.content
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({"answer": str(e)})

# ==========================
# HEALTH CHECK
# ==========================

@app.route("/health")
def health():
    return jsonify({"status": "running"})

# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)