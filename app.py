from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import os
import re
import time
import tempfile
import fitz
import chromadb

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# =====================================================
# GROQ
# =====================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY not found")

groq_client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama3-8b-8192"

# =====================================================
# CHROMADB
# =====================================================

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="docs"
)

# =====================================================
# EMBEDDING MODEL
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

        pages = []

        for page in doc:
            txt = page.get_text()

            if txt:
                pages.append(txt)

        doc.close()

        return "\n".join(pages)

    except Exception as e:
        print("PDF ERROR:", e)
        return ""


# =====================================================
# VECTOR DATABASE
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
        if len(c.strip()) > 80
    ]

    if not chunks:
        raise Exception("No usable text found")

    embeddings = get_model().encode(
        chunks,
        show_progress_bar=False
    )

    try:
        existing = collection.get()

        if existing["ids"]:
            collection.delete(
                ids=existing["ids"]
            )
    except Exception as e:
        print("Delete warning:", e)

    collection.add(
        ids=[
            str(i)
            for i in range(len(chunks))
        ],
        documents=chunks,
        embeddings=[
            emb.tolist()
            for emb in embeddings
        ]
    )

    return len(chunks)


# =====================================================
# PROMPT
# =====================================================

def build_prompt(question, context):

    return f"""
You are a strict research assistant.

RULES:
- Answer ONLY using the provided context.
- Do not invent information.
- If the answer does not exist in the document say:
"I could not find this in the document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# UPLOAD PDF
# =====================================================

@app.route("/upload", methods=["POST"])
def upload():

    try:

        if "file" not in request.files:
            return jsonify({
                "error": "No file field found"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({
                "error": "Only PDF files are supported"
            }), 400

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            file.save(temp_file.name)

            temp_path = temp_file.name

        text = extract_pdf(temp_path)

        try:
            os.remove(temp_path)
        except:
            pass

        if not text.strip():
            return jsonify({
                "error": "No readable text found in PDF"
            }), 400

        start = time.time()

        chunk_count = build_db(text)

        elapsed = round(
            time.time() - start,
            2
        )

        return jsonify({
            "ready": True,
            "chunks": chunk_count,
            "embed_time": elapsed
        })

    except Exception as e:

        print("UPLOAD ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# =====================================================
# ASK QUESTION
# =====================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:
            return jsonify({
                "answer": "Question is empty."
            })

        if question.lower() in GREETINGS:
            return jsonify({
                "answer": GREETINGS[
                    question.lower()
                ]
            })

        existing = collection.get()

        if len(existing["ids"]) == 0:
            return jsonify({
                "answer":
                "❌ Please upload a PDF first."
            })

        query_embedding = get_model().encode(
            [question]
        )

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=6
        )

        documents = results["documents"][0]

        context = "\n\n".join(
            documents
        )[:8000]

        start = time.time()

        completion = groq_client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": build_prompt(
                        question,
                        context
                    )
                }
            ]
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

        print("ASK ERROR:", str(e))

        return jsonify({
            "answer": str(e)
        })


# =====================================================
# HEALTH
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
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )