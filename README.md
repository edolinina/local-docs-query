# 📚 DocsQuery – Local RAG CLI

A simple local document Q&A system with pluggable LLM backends.

Supports:

* Hugging Face models (local Transformers)
* Ollama (local or remote)
* Chroma (vector database)
* SentenceTransformers (embeddings)

---

## 🚀 Features

* 📂 Index PDF, DOCX, PPT, TXT files
* 🧠 Local embeddings (MiniLM)
* 🔎 Semantic search with Chroma
* 🤖 Multiple LLM providers:

  * Hugging Face
  * Ollama (local or remote)
* 📎 Optional source display
* 🎯 Metadata-aware filtering (folder, file, etc.)

---

## 🛠 Setup

### 1. Install CLI

```bash
python3 -m venv .venv # or pip install virtualenv & virtualenv .venv
source .venv/bin/activate

# minimal - Ollama only
pip install -e ".[ollama]"

# hugging face support only
pip install -e ".[hf]"

# full setup
pip install  -e .

# CPU-only install (recommended for HF)
pip install "docsquery[hf]" \
  --index-url https://download.pytorch.org/whl/cpu
```

---

### 2. Configure LLM

```bash
docsquery setup
```

You will be prompted:

#### Example (Hugging Face)

```
Provider (hf/ollama): hf
Model [google/gemma-3-4b-it]:
```

#### Example (Ollama)

```
Provider (hf/ollama): ollama
Model [llama3]:
Ollama URL [http://localhost:11434]:
```

---

### 🔐 Hugging Face (only if using HF LLM)

Set your token:

```bash
export HF_TOKEN=<your_token>
```

Required for gated models like:

* google/gemma
* llama models

---

### 🧠 Ollama Setup (only if using Ollama)

Install Ollama:

https://ollama.com

Run it:

```bash
ollama serve
```

The CLI will automatically:

* pull the model (via API)
* work with local OR remote Ollama

---

### 🌐 Remote Ollama example

```bash
docsquery setup \
  --provider ollama \
  --model llama3 \
  --url http://192.168.1.50:11434
```

---

## 📂 Index Documents

```bash
# the whole folder
docsquery index ./docs

# single file
docsquery index ./docs/book.pdf
```

* Recursively scans folder
* Skips already indexed files (via hash)
* Stores data in local Chroma DB

---

## ❓ Ask Questions

```bash
docsquery ask "what is heatmap"
```

---

## 📎 Show Sources

```bash
docsquery ask "What are the possible reasons for model overfitting" --sources
```

---

## 🎯 Filter by Folder

```bash
docsquery ask "What is qlearning" --folder ./docs/rl
```

---

## ⚙️ Configuration

Stored at:

```
~/.docsquery/config.json
```

Example:

```json
{
  "provider": "ollama",
  "model": "llama3",
  "base_url": "http://localhost:11434"
}
```

---

## 🧠 Embeddings

Uses:

* SentenceTransformers
* Model: `all-MiniLM-L6-v2`

---

## 🗄 Vector Store

Uses:

* Chroma

Stored locally:

```
./chroma_db
```

---

## ⚡ Performance Notes

### Hugging Face

* Slower on Mac (no CUDA)
* Use smaller models for speed:

---

### Ollama

* Faster (Metal optimized on Mac)
* Recommended models:

  * `llama3`
  * `gemma`

---

## 🧩 Architecture

```
Docs → Chunking → Embeddings → Chroma
                                     ↓
                               Retriever
                                     ↓
                          HF / Ollama LLM
                                     ↓
                                  Answer
```
