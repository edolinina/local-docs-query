# 📚 DocChat - Local RAG App

A simple local document Q&A system using:
- Qdrant (vector DB)
- Ollama (LLM)
- SentenceTransformers (embeddings)
- Gradio (UI)

---

## 🚀 Features

- Index PDF, DOCX, PPT files from a folder
- Store embeddings in Qdrant
- Ask questions over your documents
- Metadata-aware retrieval

---

## 🛠 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install cli
```bash
pip install -e .
pyinstaller --onefile cli.py
```

### 3. Start Ollama
```bash
ollama run mistral
```

### 4. Index the target folder
```bash
docsquery index ./docs
```

### 5. Ask a question
```bash
docsquery ask 'What is the meaning of model overfitting?'
```

### 6. Start Q&A bot
```bash
docchat chat
```
