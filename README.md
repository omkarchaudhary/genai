# 🧠 RAG Knowledge Assistant

A production-ready **RAG (Retrieval-Augmented Generation)** application built with LangChain, Groq LLM, and ChromaDB.

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it like an **open-book exam** for AI:

- Without RAG: AI only knows what it was trained on
- With RAG: AI can look up information in YOUR documents first, then answer!

```
┌─────────────────────────────────────────────────────────────────┐
│                     WITH RAG (Open Book)                         │
│                                                                  │
│   User: "What's in my company policy?"                          │
│   AI: *searches your uploaded documents*                         │
│   AI: "Based on your policy document, employees get..." ✅       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API Key (free at https://console.groq.com)

### Installation

```bash
# 1. Navigate to project root
cd D:\Projects\AI\genai-series

# 2. Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1  # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with your API key
# Copy .env.example to .env and add your GROQ_API_KEY
```

### Run the Application

```bash
# Navigate to app folder
cd apps/rag_app

# Run Streamlit
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG APPLICATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   config.py  │    │   utils.py   │    │   app.py     │               │
│  │   (Settings) │    │  (Helpers)   │    │ (Streamlit)  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                 │                        │
│         ┌───────────────────────────────────────┼───────────────┐       │
│         │                                       │               │       │
│         ▼                                       ▼               ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  document_   │───▶│    text_     │───▶│   vector_    │              │
│  │  loader.py   │    │ processor.py │    │   store.py   │              │
│  │  (Load Docs) │    │  (Chunking)  │    │  (ChromaDB)  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                 │                        │
│                                                 ▼                        │
│                                          ┌──────────────┐               │
│                                          │  rag_chain.py│               │
│                                          │  (LLM + RAG) │               │
│                                          └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Module Breakdown

### 1. `config.py` - Configuration

**What:** Single source of truth for all settings.

**Key Settings:**
| Setting | Value | Purpose |
|---------|-------|---------|
| `GROQ_MODEL_NAME` | llama-3.3-70b-versatile | AI model to use |
| `LLM_TEMPERATURE` | 0.1 | Creativity (0=focused, 1=creative) |
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |

---

### 2. `utils.py` - Helper Functions

**What:** Reusable utility functions.

**Functions:**

- `is_valid_url()` - Validate URL format
- `is_valid_file_extension()` - Check allowed file types
- `clean_text()` - Normalize whitespace

---

### 3. `document_loader.py` - Load Documents

**What:** Reads content from different sources and converts to common format.

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│   PDF File  │────▶│   load_pdf()    │────▶│                     │
└─────────────┘     └─────────────────┘     │                     │
                                            │    LangChain        │
┌─────────────┐     ┌─────────────────┐     │    Document         │
│  TXT File   │────▶│ load_text_file()│────▶│    Object           │
└─────────────┘     └─────────────────┘     │                     │
                                            │  {                  │
┌─────────────┐     ┌─────────────────┐     │    page_content,    │
│    URL      │────▶│ load_from_url() │────▶│    metadata         │
└─────────────┘     └─────────────────┘     │  }                  │
                                            └─────────────────────┘
```

**Supported Sources:**

- 📄 PDF files (using `pypdf`)
- 📝 TXT files
- 🌐 Web URLs (using `BeautifulSoup`)

---

### 4. `text_processor.py` - Text Chunking

**What:** Splits large documents into smaller, searchable pieces.

**Why Chunking?**

- LLMs have token limits
- Smaller chunks = more precise search results
- Overlap preserves context between chunks

```
Original Document (5000 chars):
┌─────────────────────────────────────────────────────────────────┐
│ Lorem ipsum dolor sit amet, consectetur adipiscing elit...      │
└─────────────────────────────────────────────────────────────────┘

After Chunking (CHUNK_SIZE=1000, OVERLAP=200):
┌────────────────┐
│    Chunk 1     │ (chars 0-1000)
└───────┬────────┘
        │ 200 char overlap
┌───────┴────────┐
│    Chunk 2     │ (chars 800-1800)
└───────┬────────┘
        │ 200 char overlap
┌───────┴────────┐
│    Chunk 3     │ (chars 1600-2600)
└────────────────┘
```

---

### 5. `vector_store.py` - Vector Database (ChromaDB)

**What:** Stores document chunks as embeddings for semantic search.

**How Embeddings Work:**

```
"cat"  → [0.2, 0.8, 0.1, 0.5, ...]  (768 numbers)
"dog"  → [0.3, 0.7, 0.2, 0.4, ...]  (similar to cat!)
"car"  → [0.9, 0.1, 0.8, 0.2, ...]  (very different)

Similar meanings = Similar vectors = Found in search!
```

**Why ChromaDB?**
| Feature | Benefit |
|---------|---------|
| Free | No cost |
| Local | Data stays private |
| Persistent | Survives app restart |
| Simple | Easy setup |

**Why HuggingFace Embeddings?**

- Free (no API key needed)
- Runs locally
- Model: `all-MiniLM-L6-v2`

---

### 6. `rag_chain.py` - RAG Chain with Groq LLM

**What:** Combines retrieval + LLM to answer questions.

**Flow:**

```
Question → Retrieve Relevant Chunks → Build Prompt → LLM → Answer
```

**Why Groq?**
| Feature | Benefit |
|---------|---------|
| ⚡ Speed | 10x faster than competitors |
| 💰 Free tier | No cost to start |
| 🧠 Llama 3 | Top open-source model |

**LCEL (LangChain Expression Language):**

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

---

### 7. `app.py` - Streamlit UI

**What:** Web interface tying everything together.

**Features:**

- 📁 File upload (PDF, TXT)
- 🌐 URL scraping
- 💬 Chat interface
- 📊 Knowledge base stats
- 📚 Source citations

---

## 🔄 Complete Data Flow

### Phase 1: Adding Knowledge

```
Upload PDF
    │
    ▼
document_loader.py (extract text)
    │
    ▼
text_processor.py (split into chunks)
    │
    ▼
vector_store.py (convert to embeddings, store in ChromaDB)
    │
    ▼
✅ "50 chunks added to knowledge base!"
```

### Phase 2: Asking Questions

```
User: "What is the vacation policy?"
    │
    ▼
vector_store.py (find similar chunks)
    │
    ▼
rag_chain.py (build prompt + call LLM)
    │
    ▼
💬 "Based on employee handbook, you get 15 days..."
   📚 Source: employee_handbook.pdf (page 12)
```

---

## 🛠️ Technology Stack

| Component      | Technology                     | Why?                           |
| -------------- | ------------------------------ | ------------------------------ |
| **LLM**        | Groq (Llama 3.3 70B)           | Fast, free tier, powerful      |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free, local, no API key        |
| **Vector DB**  | ChromaDB                       | Free, local, persistent        |
| **UI**         | Streamlit                      | Simple, Python-only            |
| **Framework**  | LangChain                      | Industry standard for LLM apps |

---

## 📋 Requirements

```txt
# Core LangChain
langchain
langchain-community
langchain-groq

# Vector Database
chromadb

# Embeddings
sentence-transformers
langchain-huggingface

# Document Processing
pypdf
beautifulsoup4
requests

# Utilities
python-dotenv
streamlit
tiktoken
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at: https://console.groq.com

---

## 📁 Project Structure

```
apps/rag_app/
├── __init__.py          # Package initializer
├── config.py            # All settings in one place
├── utils.py             # Helper functions
├── document_loader.py   # Load PDF, TXT, URLs
├── text_processor.py    # Chunk text for embeddings
├── vector_store.py      # ChromaDB operations
├── rag_chain.py         # LangChain + Groq RAG
├── app.py               # Main Streamlit UI
├── README.md            # This file
└── chroma_db/           # Vector database storage (auto-created)
```

---

## 🎯 Key Concepts to Remember

1. **RAG = Search + Generate**

   - Traditional AI: "I only know what I was trained on"
   - RAG AI: "Let me look that up in your documents first!"

2. **Embeddings = Meaning as Numbers**

   - Similar meaning = Similar numbers = Found in search

3. **Chunking = Breaking Big → Small**

   - Big document → Small searchable pieces → Better results

4. **Prompt Engineering = Controlling AI**

   - Good prompt = AI follows instructions
   - Bad prompt = AI makes stuff up

5. **Modular Design = Easy to Change**
   - Want different LLM? → Change only rag_chain.py
   - Want different DB? → Change only vector_store.py

---

## 🚀 Future Improvements

| Feature                 | Module to Modify   |
| ----------------------- | ------------------ |
| Add Word/Excel support  | document_loader.py |
| Add authentication      | app.py             |
| Add conversation memory | rag_chain.py       |
| Multiple collections    | vector_store.py    |
| REST API                | New: api.py        |
| Docker deployment       | New: Dockerfile    |

---

## 📝 License

MIT License - Feel free to use and modify!
