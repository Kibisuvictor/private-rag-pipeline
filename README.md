# Local RAG Pipeline — Chat with Your Documents

![CI](https://github.com/YOUR_USERNAME/rag-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A fully local Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions over your own PDF documents. No API keys, no data leaves your machine, no cloud costs.

Built with **Ollama**, **ChromaDB**, **LangChain**, and **Streamlit**.

---

![Demo screenshot placeholder — replace with a GIF of the Streamlit UI](docs/demo.gif)

> *"What adaptation strategies are recommended for smallholder farmers in arid zones?"*
> → Answers instantly with relevant excerpts and source citations from your documents.

---

## What is RAG?

Large Language Models (LLMs) are trained on general knowledge but know nothing about your private documents. **Retrieval-Augmented Generation** solves this by:

1. Converting your documents into vector embeddings and storing them in a vector database (ChromaDB)
2. When a question arrives, retrieving the most semantically similar document chunks
3. Passing those chunks as context to the LLM, which generates a grounded answer

This means the model can only answer from your actual documents — it won't hallucinate facts that aren't there.

---

## Features

- **Fully offline** — embeddings and LLM inference run locally via Ollama
- **Source citations** — every answer shows which document chunks were used
- **Persistent vector store** — ingest once, query many times (ChromaDB persists to disk)
- **Multi-document support** — ingest an entire folder of PDFs at once
- **Chat history** — conversational interface remembers context within a session
- **CI pipeline** — GitHub Actions runs linting and tests on every push

---

## Architecture

```
Ingestion pipeline                     Query pipeline
─────────────────                      ──────────────
PDF / text files                       User question (Streamlit)
      │                                      │
      ▼                                      ▼
  Chunking                             Embed question
  (RecursiveCharacterTextSplitter)     (all-MiniLM-L6-v2)
      │                                      │
      ▼                                      ▼
  Embedding model  ──────────────►  Vector similarity search
  (all-MiniLM-L6-v2)                   (Top-4 chunks from ChromaDB)
      │                                      │
      ▼                                      ▼
  ChromaDB  ◄────────────────────    Prompt assembly
  (persistent vector store)           (context + question)
                                            │
                                            ▼
                                       Ollama LLM
                                       (Mistral / LLaMA 3)
                                            │
                                            ▼
                                    Answer + source citations
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM inference | [Ollama](https://ollama.com) | Runs open-source LLMs locally with no API key |
| Vector store | [ChromaDB](https://www.trychroma.com) | Lightweight, file-based, no server required |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Fast, high-quality, runs on CPU |
| Orchestration | [LangChain](https://python.langchain.com) | LCEL chain composition with `RunnablePassthrough` |
| UI | [Streamlit](https://streamlit.io) | Rapid prototyping for data apps |
| CI | GitHub Actions | Lint + test on every push |

---

## Project Structure

```
rag-pipeline/
├── ingest.py               # Load, chunk, embed, and store documents
├── rag.py                  # Query chain (retriever + LLM)
├── app.py                  # Streamlit chat UI
├── data/                   # Place your PDF/text files here
├── chroma_db/              # ChromaDB persists here (git-ignored)
├── tests/
│   └── test_chunking.py    # Unit tests for ingestion pipeline
├── docs/
│   └── demo.gif            # UI demo (add your own screenshot)
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Pull a local LLM

```bash
ollama pull mistral        # ~4GB, recommended
# or
ollama pull llama3.2:3b    # ~2GB if RAM is limited
```

### 3. Add your documents

Drop PDF or text files into the `data/` folder:

```bash
cp your_report.pdf data/
```

### 4. Ingest documents

```bash
python ingest.py
```

This chunks your documents, generates embeddings, and stores them in ChromaDB. Run this once per batch of new documents.

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start asking questions.

---

## Configuration

Key parameters are in `ingest.py` and `rag.py`. The values below were chosen after testing on research PDFs:

| Parameter | Value | Reasoning |
|---|---|---|
| `chunk_size` | 800 chars | Balances context richness with retrieval precision. Larger chunks give more context but reduce recall diversity. |
| `chunk_overlap` | 150 chars | Prevents information loss at chunk boundaries. Critical for sentences that span split points. |
| `k` (top-k retrieval) | 4 chunks | Enough context for most factual questions without overloading the LLM context window. |
| `temperature` | 0.1 | Near-deterministic output — important for factual Q&A where consistency matters. |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim vectors, fast CPU inference, strong semantic similarity performance on English text. |

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- Chunk size boundaries (no chunk exceeds 800 chars)
- Overlap correctness (adjacent chunks share content)
- Metadata preservation across splits (source, page number)
- Edge cases (empty docs, special characters, whitespace-only content)
- Content integrity (no words dropped during splitting)

Tests are designed to run in CI without Ollama or ChromaDB — they test pure ingestion logic.

---

## What I Would Add Next

- **Re-ranking** — use a cross-encoder (e.g. `ms-marco-MiniLM`) to re-rank retrieved chunks before passing to the LLM, improving answer quality
- **RAGAS evaluation** — automated RAG pipeline evaluation using faithfulness, answer relevancy, and context precision metrics
- **Pinecone / Qdrant** — swap ChromaDB for a managed vector store and deploy the query pipeline to Cloud Run
- **Hybrid search** — combine dense vector search with BM25 keyword search for better recall on named entities
- **Document metadata filtering** — filter retrieval by document source, date, or category before similarity search

---

## Dataset

This project was tested using [World Bank climate reports on East Africa](https://openknowledge.worldbank.org) and CIFOR-ICRAF research publications on agroforestry and land use in Kenya. These are publicly available under open access licences.

Using domain-specific documents significantly improves answer quality compared to generic test PDFs — the embedding model is better able to retrieve precise chunks when the vocabulary is consistent.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Victor** — Data Scientist | [LinkedIn](https://linkedin.com/in/victor-kibisu) | [GitHub](https://github.com/Kibisuvictor)
