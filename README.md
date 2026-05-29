# 📄 PDF Research Assistant (Offline RAG Chatbot)

A modern AI-powered PDF chatbot built using:

* Flask
* ChromaDB
* Sentence Transformers
* Groq LLM API
* PyMuPDF
* LangChain Text Splitters

Upload PDFs, research papers, reports, and interact with them using Retrieval-Augmented Generation (RAG).

---

# 🚀 Features

* 📄 Upload PDF documents
* 🔍 Semantic search using embeddings
* 🤖 AI-powered question answering
* ⚡ Fast inference using Groq API
* 🧠 Local vector database using ChromaDB
* 📱 Fully responsive mobile UI
* 💬 Always-visible mobile chatbox
* ☁️ Deployable on Render / Railway / Replit
* 🌙 Modern dark UI

---

# 📱 Mobile Responsive Improvements

This project is optimized for:

* Mobile phones
* Tablets
* Desktop browsers

### Mobile Features

* Fixed bottom chat input
* Responsive layout
* Full-screen chat experience
* Auto-scroll messages
* Hidden side panel on smaller screens
* Touch-friendly buttons

---

# 🖼️ Demo

## Upload PDF

* Upload any PDF
* Automatic text extraction
* Embedding generation
* ChromaDB indexing

## Ask Questions

Example prompts:

```txt
Summarize this paper.

Explain the methodology.

What datasets were used?

What are the key findings?
```

---

# 🏗️ Tech Stack

| Technology            | Purpose         |
| --------------------- | --------------- |
| Flask                 | Backend API     |
| ChromaDB              | Vector Database |
| Sentence Transformers | Embeddings      |
| Groq API              | LLM Inference   |
| PyMuPDF               | PDF Parsing     |
| LangChain             | Text Chunking   |
| HTML/CSS/JS           | Frontend        |

---

# 📂 Project Structure

```bash
pdf-chatbot/
│
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
│
├── templates/
│   └── index.html
│
└── chroma_db/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPO_URL

cd pdf-chatbot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Setup Groq API Key

Generate a free API key from:

```txt
https://console.groq.com
```

---

## Windows CMD

```bash
set GROQ_API_KEY=your_api_key
```

## Windows PowerShell

```bash
$env:GROQ_API_KEY="your_api_key"
```

## Linux / Mac

```bash
export GROQ_API_KEY="your_api_key"
```

---

# ▶️ Run Application

```bash
python app.py
```

Open browser:

```txt
http://127.0.0.1:5000
```

---

# ☁️ Free Deployment Options

## 1. Render (Recommended)

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

### Add Environment Variable

```txt
GROQ_API_KEY=your_api_key
```

---

## 2. Railway

### Start Command

```bash
gunicorn app:app
```

---

## 3. Replit

### Run Command

```bash
python app.py
```

---

# 📱 Mobile UI Fixes Included

The latest frontend improvements include:

* Responsive chat layout
* Fixed bottom input box
* Better textarea scaling
* Scrollable chat window
* Mobile-safe viewport handling
* No hidden chatbox on small screens
* Improved touch support

---

# 🧠 How It Works

## Step 1 — Upload PDF

PDF text extracted using:

```python
PyMuPDF
```

---

## Step 2 — Chunking

Text split using:

```python
RecursiveCharacterTextSplitter
```

---

## Step 3 — Embeddings

Embeddings generated using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

---

## Step 4 — Vector Search

Relevant chunks retrieved using:

```python
ChromaDB similarity search
```

---

## Step 5 — LLM Response

Context passed to Groq API for final answer generation.

---

# 📌 Example Questions

```txt
Summarize this PDF.

Explain the architecture.

What future work is proposed?

What conclusions were made?
```

---

# 🔧 Future Improvements

* Multi-PDF support
* OCR support
* Streaming responses
* Citation highlighting
* Authentication
* Docker support
* Chat history
* Voice input
* PDF image extraction

---

# 🐳 Docker Support

Future deployment can include:

* Flask container
* ChromaDB persistence
* Nginx reverse proxy

---

# 📜 License

MIT License

---

# 🙌 Acknowledgements

* Flask
* ChromaDB
* Sentence Transformers
* Groq
* LangChain
* PyMuPDF

---

# ⭐ Star This Repository

If this project helped you, please consider starring the repository.
