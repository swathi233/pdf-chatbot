from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import os
import time
import tempfile
import fitz
import chromadb

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# APP SETUP
# ==============================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB limit

# ==============================
# GROQ
# ==============================

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama3-8b-8192"

# ==============================
# CHROMA (IN MEMORY SAFE MODE)
# ==============================

client = chromadb.Client()
collection = client.get_or_create_collection("docs")

# ==============================
# EMBEDDING MODEL (LAZY LOAD)
# ==============================

embedding_model = None

def get_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu"
        )
        print("Model loaded")
    return embedding_model


def safe_encode(texts):
    model = get_model()
    return model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True
    )

# ==============================
# PDF EXTRACTION
# ==============================

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

# ==============================
# VECTOR DB BUILDER (FIXED)
# ==============================

def build_db(text):

    global collection

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    chunks = [c for c in chunks if len(c.strip()) > 80]

    if not chunks:
        raise Exception("No readable text found in PDF")

    # LIMIT FOR RAILWAY MEMORY SAFETY
    chunks = chunks[:40]

    print("Chunks:", len(chunks))

    embeddings = safe_encode(chunks)

    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except:
        pass

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return len(chunks)

# ==============================
# PROMPT
# ==============================

def build_prompt(question, context):
    return f"""
You are a strict assistant.

Only answer using the context below.
If answer not found, say: "Not found in document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# ==============================
# HOME
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# UPLOAD FIXED (MAIN FIX)
# ==============================

@app.route("/upload", methods=["POST"])
def upload():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # TEMP FILE SAFE
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

        return jsonify({
            "error": str(e)
        }), 500

# ==============================
# ASK
# ==============================

@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Empty question"})

        # CHECK EMPTY DB
        if collection.count() == 0:
            return jsonify({"answer": "Upload PDF first"})

        query_embedding = safe_encode([question])

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )

        docs = results["documents"][0]
        context = "\n\n".join(docs)[:6000]

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

# ==============================
# HEALTH
# ==============================

@app.route("/health")
def health():
    return jsonify({"status": "running"})

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)