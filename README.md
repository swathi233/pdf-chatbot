# 📄 PDF Research Assistant (Offline RAG Chatbot)

A powerful AI-powered PDF chatbot built using:

* Flask
* ChromaDB
* Sentence Transformers
* Groq LLM API
* PyMuPDF
* LangChain Text Splitters

Upload research papers, reports, or PDFs and ask questions directly from the document using Retrieval-Augmented Generation (RAG).

---

# 🚀 Features

* 📄 Upload PDF documents
* 🔍 Semantic search using embeddings
* 🤖 AI-powered question answering
* ⚡ Fast inference using Groq API
* 🧠 Local vector database using ChromaDB
* 📱 Mobile-friendly UI
* ☁️ Deployable on Render/Railway

---

# 🖼️ Demo

## Upload PDF

* Upload any PDF document
* Text gets extracted and embedded
* Vector database created automatically

## Ask Questions

Examples:

```txt
What is the proposed methodology?

Summarize the conclusion.

What datasets were used?

Explain the architecture.
```

---

# 🏗️ Tech Stack

| Technology            | Purpose         |
| --------------------- | --------------- |
| Flask                 | Backend API     |
| ChromaDB              | Vector database |
| Sentence Transformers | Embeddings      |
| Groq API              | LLM inference   |
| PyMuPDF               | PDF extraction  |
| LangChain             | Text chunking   |
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
├── .gitignore
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

# 🔑 Setup Groq API

## Create API Key

Go to:

https://console.groq.com

Generate a free API key.

---

## Windows

### CMD

```bash
set GROQ_API_KEY=your_api_key
```

### PowerShell

```bash
$env:GROQ_API_KEY="your_api_key"
```

---

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

# ☁️ Deploy on Render

## 1. Push Code to GitHub

```bash
git init

git add .

git commit -m "Initial commit"

git branch -M main

git remote add origin YOUR_REPO_URL

git push -u origin main
```

---

## 2. Create Render Web Service

Go to:

https://render.com

Create:

* New Web Service
* Connect GitHub repository

---

## 3. Render Settings

| Field         | Value                             |
| ------------- | --------------------------------- |
| Runtime       | Python                            |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app`                |

---

## 4. Add Environment Variable

```txt
GROQ_API_KEY = your_api_key
```

---

## 5. Deploy

Render will generate:

```txt
https://your-app-name.onrender.com
```

---

# 🧠 How It Works

## Step 1 — PDF Upload

PDF text is extracted using:

```python
PyMuPDF
```

---

## Step 2 — Text Chunking

Text is split into chunks using:

```python
RecursiveCharacterTextSplitter
```

---

## Step 3 — Embedding Generation

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

## Step 5 — AI Answering

Context sent to Groq LLM for final answer generation.

---

# 📌 Example Queries

```txt
Summarize the paper.

What problem does this paper solve?

Explain the methodology.

What are the key findings?

What future work is proposed?
```

---

# 🔧 Future Improvements

* Multi-PDF support
* Streaming responses
* Citation highlighting
* OCR support
* Image extraction
* Reranking
* Authentication
* Chat history
* Docker deployment

---

# 🐳 Docker Support (Optional)

Future deployment can include:

* Flask container
* ChromaDB persistence
* Nginx reverse proxy

---

# 📱 Mobile Support

The UI is responsive and works on:

* Mobile
* Tablet
* Desktop

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

If you found this project useful, please consider starring the repository.
