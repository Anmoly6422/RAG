# 📚 PDF RAG System (Retrieval-Augmented Generation)

A modular, production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with **Python**, **LangChain**, **SentenceTransformers**, **ChromaDB**, and **Google Gemini AI**.

This project enables intelligent, context-aware Question Answering (QA) over local PDF documents with precise source citations.

---

## 🌟 Key Features

- **📄 Automated PDF Loading & Chunking**: Extracts text from PDFs and recursively splits documents into optimal context chunks.
- **⚡ Fast Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for high-dimensional vector embeddings.
- **💾 Persistent Vector Database**: Uses **ChromaDB** to store document vectors with metadata. Avoids re-embedding on subsequent runs.
- **🎯 Similarity Search Retriever**: Queries relevant document chunks based on vector distance and calculates relevance scores.
- **🤖 Gemini LLM Integration**: Generates grounded answers using Google Gemini AI (`gemini-3.8-flash`).
- **📍 Detailed Source Citations**: Returns exact source filenames, page numbers, and relevance scores for every answer.
- **💻 Flexible CLI Interface**: Pass custom questions directly via command-line arguments.

---

## 🏗️ Architecture Flow

```text
[ PDF Documents ] ──► [ Document Loader ] ──► [ Text Splitter (Chunks) ]
                                                     │
                                                     ▼
                                          [ SentenceTransformers ]
                                                     │
                                                     ▼
                                            [ ChromaDB Vector Store ]
                                                     │
 [ User Question ] ──► [ Vector Query ] ─────────────┤
                                                     ▼
                                          [ Relevant Context Chunks ]
                                                     │
                                                     ▼
                                          [ Gemini LLM Prompt ] ──► [ Answer & Sources ]
```

---

## 🛠️ Tech Stack

- **Language**: Python >= 3.11
- **LLM**: Google Gemini (`langchain-google-genai`)
- **Vector DB**: ChromaDB
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Document Loading & Chunking**: LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Anmoly6422/RAG.git
cd RAG
```

### 2. Install Dependencies
Using `pip`:
```bash
pip install -r requirement.txt
```

Or using `uv`:
```bash
uv sync
```

### 3. Configure API Key
Create a `.env` file in the root directory (or copy `.env.example`):
```bash
cp .env.example .env
```

Add your Google Gemini API Key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 📖 Usage Guide

### Place your PDFs
Place your PDF files inside the `data/pdf/` directory.

### Run a Custom Query
Ask any question against your PDF documents:
```bash
python main.py "What are the technical skills listed in the resume?"
```

```bash
python main.py "What projects are built with React Native?"
```

### Run Default Sample Query
Running `main.py` without arguments asks the default query:
```bash
python main.py
```

### Options & Flags

| Flag | Description | Default | Example |
|---|---|---|---|
| `query` | The question you want to ask | `"Explain about ExpoMind?"` | `python main.py "Your question"` |
| `--top_k` | Number of document chunks to retrieve | `3` | `python main.py "Question" --top_k 5` |
| `--reindex` | Force clearing and re-indexing of documents | `False` | `python main.py --reindex` |
| `--pdf_dir` | Directory containing PDF files | `"data/pdf"` | `python main.py --pdf_dir "data/my_pdfs"` |

---

## 📁 Project Structure

```text
RAG/
├── data/
│   ├── pdf/                 # Place your PDF documents here
│   └── vector_store/        # Persistent ChromaDB storage (auto-generated)
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── data_loader.py    # PDF loading & text chunking
│       ├── embedding.py      # SentenceTransformer embeddings
│       ├── vectorstore.py    # ChromaDB vector store wrapper
│       ├── search.py         # RAG retriever & similarity search
│       ├── llm.py            # Gemini LLM integration
│       └── rag.py            # Complete RAG pipeline orchestration
├── .env.example             # Environment variables template
├── .gitignore
├── main.py                  # CLI entry point
├── pyproject.toml           # Project metadata & dependencies
└── requirement.txt          # Python dependencies
```

---

## 📜 License

MIT License
