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

### 1. Install cli
```bash
pip install -e .
```

### 2. Index the target folder
```bash
docsquery index ./docs
```

### 3. Ask a question
```bash
docsquery ask 'What is the meaning of model overfitting?'
```
