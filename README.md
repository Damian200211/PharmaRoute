# PharmaRoute: Local Metadata-Routed RAG Pipeline

A privacy-preserving, on-device Retrieval-Augmented Generation (RAG) system engineered to segment, classify, and query bundled pharmaceutical PDFs and regulatory dossiers. Powered by quantized local LLM execution (**Mistral-7B-Instruct-v0.2** via `llama-cpp-python`) and **LlamaIndex**, the pipeline automates document boundary detection and uses metadata-filtered vector retrieval to eliminate cross-document hallucination.

---

## Technical Overview

Pharmaceutical submissions often arrive as single, monolithic PDF bundles containing dozens of disparate document types (e.g., Certificates of Analysis, BSE/TSE Declarations, Packaging Specs, Chain of Custody). Standard naive RAG pipelines index these arbitrarily, yielding noisy chunk retrieval across unrelated documents.

This pipeline resolves this through a two-stage LLM-assisted workflow:
1. **Dynamic Page-Boundary Classification:** Iterates through raw PDF pages, prompting Mistral-7B with zero-shot classification to detect logical boundaries between distinct document types without manual human splitting.
2. **Metadata-Filtered Query Routing:** When a user poses a question, an intent-routing prompt identifies the specific target document type and applies strict `MetadataFilters` during vector search. This restricts nearest-neighbor search exclusively to valid source sub-documents.
3. **Local, Air-Gapped Inference:** Operates completely on local GPU hardware via GGUF 4-bit quantization, meeting strict compliance and data-privacy standards required for proprietary enterprise data.

---

## Architecture Flow

```text
Monolithic PDF
      │
      ▼
[ PyPDF2 Page Extraction ]
      │
      ▼
[ Mistral 7B Boundary Classifier ] ──► Detects doc transitions & tags metadata
      │
      ▼
[ LangChain Recursive Splitter ]  ──► 512-token chunks (100 overlap) + rich metadata
      │
      ▼
[ BAAI/bge-small-en-v1.5 ]        ──► Dense embeddings into LlamaIndex VectorStore
      │
      ├────────────────────────────────────────┐
      ▼                                        ▼
User Query ──► [ Query Intent Classifier ] ──► [ Metadata-Filtered Retriever ]
                                                       │
                                                       ▼
                                            [ Compact Synthesizer ]
                                                       │
                                                       ▼
                                              Structured Answer + Audit Context
```

---

## Key Features

- **Automated Logical Document Segmentation:** Splits bundled PDF files into discrete logical units using sequential page context comparisons (`is_same_document`).
- **Precision Metadata Filtering:** Bypasses vector noise by applying deterministic metadata filters (`FilterOperator.EQ`) at retrieval time based on query intent.
- **Zero Cloud Leakage:** Fully local execution running quantized Mistral 7B (Q4_K_M) alongside local HuggingFace embeddings (`bge-small-en-v1.5`).
- **Interactive Inspection UI:** Gradio interface featuring real-time index feedback, predicted routing paths, and retrieved source chunks with exact page ranges for auditability.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM Engine** | Mistral-7B-Instruct-v0.2 (GGUF Q4_K_M) | Local inference via `llama-cpp-python` with full GPU layer offloading |
| **Orchestration** | LlamaIndex (`llama-index-core`) | Vector store management, index construction, and structured synthesis |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Dense vector representation optimized for semantic search |
| **Chunking** | `langchain-text-splitters` | Document-level chunking with boundary retention |
| **Ingestion** | `PyPDF2` | In-memory text extraction per PDF page |
| **User Interface** | Gradio | Reactive dashboard for file processing, querying, and provenance checks |

---

## Setup & Local Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ support (recommended for full layer offload)

### 1. Clone & Environment Setup
```bash
git clone [https://github.com/your-username/pharma-doc-router-rag.git](https://github.com/your-username/pharma-doc-router-rag.git)
cd pharma-doc-router-rag
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install pre-compiled CUDA wheels for `llama-cpp-python` to ensure GPU acceleration:

```bash
# CUDA 12.2 / 12.x configuration
pip install --no-cache-dir --only-binary llama-cpp-python llama-cpp-python \
  --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu122](https://abetlen.github.io/llama-cpp-python/whl/cu122)

# Supporting libraries
pip install PyPDF2 langchain-text-splitters llama-index llama-index-embeddings-huggingface llama-index-llms-llama-cpp gradio "uvicorn<0.30.0"
```

### 3. Run the Application
```bash
python app.py
```
Access the Gradio web interface at `http://127.0.0.1:7860` (or via the generated public link).

---

## Design Decisions & Trade-Offs

- **Hierarchical Classification vs. Single-Pass Indexing:** Running classification per page boundary incurs upfront indexing latency, but significantly cuts down hallucination and retrieval latency during query evaluation.
- **Compact Synthesizer:** Configured `ResponseMode.COMPACT` inside LlamaIndex to maximize context window utility within Mistral's 4,096 context token limit while maintaining strict adherence to retrieved text.
- **GGUF Q4_K_M Quantization:** Selected to enable full offload of all 33 model layers to standard consumer/enterprise GPUs (~6 GB VRAM consumption) without noticeable loss in classification accuracy.
