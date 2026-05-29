
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

# =====================================================
# FLASK APP
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# GROQ SETUP
# =====================================================

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

MODEL = "llama3-8b-8192"

# =====================================================
# CHROMADB
# =====================================================

collection = None

client = chromadb.PersistentClient(
    path="./chroma_db"
)

# =====================================================
# EMBEDDING MODEL (LAZY LOADING)
# =====================================================

embedding_model = None

def get_model():

    global embedding_model

    if embedding_model is None:

        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )

        print("Embedding model loaded.")

    return embedding_model

# =====================================================
# GREETINGS
# =====================================================

GREETINGS = {
    "hi": "👋 Hello!",
    "hello": "👋 Hello!",
    "hey": "👋 Hi!",
    "good morning": "☀️ Good morning"
}

# =====================================================
# CLEAN TEXT
# =====================================================

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

# =====================================================
# PDF EXTRACTION
# =====================================================

def extract_pdf(path):

    try:

        doc = fitz.open(path)

        text = []

        for page in doc:

            page_text = page.get_text()

            if page_text:
                text.append(page_text)

        doc.close()

        return "\n".join(text)

    except Exception as e:

        print("PDF ERROR:", e)

        return ""

# =====================================================
# BUILD VECTOR DATABASE
# =====================================================

def build_db(text):

    global collection

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )

    chunks = splitter.split_text(text)

    chunks = [
        c for c in chunks
        if len(c) > 80
    ]

    if not chunks:
        raise Exception("No text extracted from PDF")

    # ============================================
    # EMBEDDINGS
    # ============================================

    embeddings = get_model().encode(chunks)

    # ============================================
    # RESET COLLECTION
    # ============================================

    try:
        client.delete_collection("docs")
    except:
        pass

    collection = client.get_or_create_collection(
        "docs"
    )

    # ============================================
    # STORE EMBEDDINGS
    # ============================================

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings]
    )

    return len(chunks)

# =====================================================
# PROMPT TEMPLATE
# =====================================================

def build_prompt(question, context):

    return f"""
You are a strict research assistant.

RULES:
- Answer ONLY from provided context
- Do NOT hallucinate
- If answer not found say:
"I could not find this in the document"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =====================================================
# PDF UPLOAD
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    global collection

    try:

        file = request.files["file"]

        if not file:

            return jsonify({
                "error": "No file uploaded"
            })

        # ============================================
        # SAVE TEMP FILE
        # ============================================

        temp_path = os.path.join(
            tempfile.gettempdir(),
            file.filename
        )

        file.save(temp_path)

        # ============================================
        # EXTRACT TEXT
        # ============================================

        text = extract_pdf(temp_path)

        if not text.strip():

            return jsonify({
                "error": "No readable text found in PDF"
            })

        # ============================================
        # BUILD VECTOR DB
        # ============================================

        start = time.time()

        chunks = build_db(text)

        elapsed = round(
            time.time() - start,
            2
        )

        return jsonify({
            "ready": True,
            "chunks": chunks,
            "embed_time": elapsed
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =====================================================
# ASK QUESTION
# =====================================================

@app.route("/ask", methods=["POST"])
def ask():

    global collection

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        # ============================================
        # GREETINGS
        # ============================================

        if question.lower() in GREETINGS:

            return jsonify({
                "answer": GREETINGS[
                    question.lower()
                ]
            })

        # ============================================
        # CHECK PDF
        # ============================================

        if collection is None:

            return jsonify({
                "answer":
                "❌ Please upload a PDF first."
            })

        # ============================================
        # QUERY EMBEDDING
        # ============================================

        query_embedding = get_model().encode(
            [question]
        )

        # ============================================
        # VECTOR SEARCH
        # ============================================

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=6
        )

        documents = results["documents"][0]

        context = "\n\n".join(
            documents
        )[:8000]

        # ============================================
        # GROQ API CALL
        # ============================================

        start = time.time()

        completion = groq_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(
                        question,
                        context
                    )
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
            "response_time": elapsed
        })

    except Exception as e:

        return jsonify({
            "answer": str(e)
        })

# =====================================================
# HEALTH CHECK
# =====================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "running"
    })

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

