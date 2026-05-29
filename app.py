
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import os
import re
import time
import tempfile
import chromadb
import fitz

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)
CORS(app)

# =========================================
# GROQ SETUP
# =========================================

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama3-70b-8192"

# =========================================
# CHROMADB
# =========================================

collection = None

client = chromadb.PersistentClient(path="./chroma_db")

# =========================================
# EMBEDDING MODEL
# =========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")

# =========================================
# GREETINGS
# =========================================

GREETINGS = {
    "hi": "👋 Hello!",
    "hello": "👋 Hello!",
    "hey": "👋 Hi!",
    "good morning": "☀️ Good morning"
}

# =========================================
# CLEAN TEXT
# =========================================

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# =========================================
# PDF EXTRACTION
# =========================================

def extract_pdf(path):
    try:
        doc = fitz.open(path)

        text = []

        for page in doc:
            t = page.get_text()

            if t:
                text.append(t)

        doc.close()

        return "\n".join(text)

    except Exception as e:
        print("PDF ERROR:", e)
        return ""

# =========================================
# BUILD VECTOR DB
# =========================================

def build_db(text):
    global collection

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    chunks = splitter.split_text(text)

    chunks = [c for c in chunks if len(c) > 80]

    if not chunks:
        raise Exception("No text extracted from PDF")

    embeddings = embedding_model.encode(chunks)

    try:
        client.delete_collection("docs")
    except:
        pass

    collection = client.get_or_create_collection("docs")

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings]
    )

    return len(chunks)

# =========================================
# PROMPT
# =========================================

def build_prompt(question, context):
    return f"""
You are a strict research assistant.

RULES:
- Answer ONLY using provided context
- If answer not found, say:
  "I could not find this in the document"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# =========================================
# HOME
# =========================================

@app.route("/")
def home():
    return render_template("index.html")

# =========================================
# UPLOAD PDF
# =========================================

@app.route("/upload", methods=["POST"])
def upload():

    try:
        file = request.files["file"]

        if not file:
            return jsonify({
                "error": "No file uploaded"
            })

        path = os.path.join(
            tempfile.gettempdir(),
            file.filename
        )

        file.save(path)

        text = extract_pdf(path)

        start = time.time()

        chunks = build_db(text)

        elapsed = round(time.time() - start, 2)

        return jsonify({
            "ready": True,
            "chunks": chunks,
            "embed_time": elapsed
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# =========================================
# ASK QUESTION
# =========================================

@app.route("/ask", methods=["POST"])
def ask():

    global collection

    try:
        q = request.json.get(
            "question",
            ""
        ).strip()

        if q.lower() in GREETINGS:
            return jsonify({
                "answer": GREETINGS[q.lower()]
            })

        if collection is None:
            return jsonify({
                "answer": "❌ Please upload a PDF first."
            })

        # ==========================
        # QUERY EMBEDDING
        # ==========================

        query_emb = embedding_model.encode([q])

        # ==========================
        # VECTOR SEARCH
        # ==========================

        results = collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=6
        )

        context = "\n\n".join(
            results["documents"][0]
        )[:8000]

        # ==========================
        # LLM CALL
        # ==========================

        start = time.time()

        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(q, context)
                }
            ],
            temperature=0.2
        )

        answer = completion.choices[0].message.content

        elapsed = round(
            time.time() - start,
            2
        )

        return jsonify({
            "answer": answer,
            "time": elapsed
        })

    except Exception as e:
        return jsonify({
            "answer": str(e)
        })

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
