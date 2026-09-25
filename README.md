# PharmaRoute: Multimodal Document Intelligence & RAG Chatbot

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) system designed to ingest, parse, segment, and query complex pharmaceutical dossiers and regulatory filings. Built with an **OCR-fallback pipeline (Tesseract + OpenCV)**, **embedding-based document segmentation**, and an on-device quantized LLM (**Mistral-7B-Instruct-v0.2** via `llama-cpp-python`), this application delivers grounded answers with strict citation tracking, confidence scoring, and automated fallback routing.

---

## Demo


https://github.com/user-attachments/assets/d0f9148a-2099-476d-85a6-f9149ddddc7c


---

## Technical Overview

Pharmaceutical supply chains and regulatory operations rely on massive multi-document PDFs that combine pristine digital text with low-resolution scanned forms, certificates, and declarations. Naive RAG setups fail on these bundles due to image-only pages, ambiguous document transitions, and irrelevant context contamination across sections.

This system addresses these challenges with an end-to-end local architecture:
1. **Hybrid Ingestion (Digital + Adaptive OCR):** Extracts digital text via `PyPDF2`, automatically falling back to `pdf2image` and `pytesseract` with Gaussian adaptive binarization for scanned pages.
2. **Embedding-Based Semantic Segmentation:** Uses dense embedding cosine distance (`bge-small-en-v1.5`) across sequential pages to detect document transitions, isolating logical documents (e.g., separating an SDS from a Certificate of Quality).
3. **Zero-Shot Document Classification:** Prompts quantized Mistral-7B to categorize each segmented section across 9 domain-specific classes (Packaging Specs, BSE/TSE, Chain of Custody, SDS, etc.).
4. **Metadata-Routed Vector Search with Global Fallback:** Predicts the target document type based on user query intent, retrieving chunks under a strict `MetadataFilter`. If no direct matches exist, the retriever triggers an automatic fallback across the entire document corpus.
5. **Grounded Synthesis with Source Provenance:** Uses a constrained citation prompt template forcing `[Chunk X, Pages Y-Z]` source attribution, confidence metrics, and defensive stop tokens to block LLM dialogue leakage.

---

## Architecture Flow

```text
Bundled PDF File (Digital or Scanned)
                │
                ▼
   [ Text vs. Scanned Page Check ]
     ├── Digital ──► PyPDF2 Direct Extraction
     └── Scanned ──► OpenCV Preprocessing (Adaptive Threshold) + Tesseract OCR
                │
                ▼
   [ Semantic Document Segmentation ]
     └── Cosine Similarity of Page Embeddings (Threshold: 0.75)
                │
                ▼
   [ Mistral-7B Zero-Shot Classifier ]
     └── Labels logical documents (SDS, COQ, BSE/TSE, Packaging, etc.)
                │
                ▼
   [ Recursive Character Text Splitter ]
     └── 512-token chunks (100 overlap) + Page/Type Metadata
                │
                ▼
   [ BGE-Small Vector Store Index ]
                │
 ┌──────────────┴────────────────────────────────┐
 │ Query Flow                                    │
 ▼                                               ▼
User Query ──► Intent Classifier ──► Filtered Retriever (doc_type == Target)
                                                 │
                                       (No Chunks Found?)
                                         ├── Yes ──► Global Fallback Retriever
                                         └── No  ──► Retain Filtered Nodes
                                                 │
                                                 ▼
                                     [ Compact QA Synthesizer ]
                                                 │
                                                 ▼
                                    Gradio Multi-Turn Chatbot
                             (Answer + [Chunk, Page] + Confidence %)
```

---

## Key Features

- **Adaptive OCR Preprocessing:** Automatically applies grayscale conversion and Gaussian adaptive thresholding (`cv2.adaptiveThreshold`) to resolve noisy scans, fax artifacts, and low-contrast text.
- **Semantic Boundary Detection:** Replaces rigid regex/page rules with cosine similarity transitions on dense vector embeddings to segment combined PDF batches dynamically.
- **Fail-Safe Retrieval Routing:** Mitigates routing misclassifications by defaulting to an unfiltered similarity search if filtered metadata queries yield empty nodes.
- **Strict Provenance & Guardrails:** Enforces `[Chunk X, Pages Y-Z]` inline citations and truncates extraneous generation loops via LLM stop tokens (`["Question:", "Q:"]`).
- **Interactive Multi-Turn Chatbot:** Features an updated Gradio interface supporting continuous chat history, session clearing, and retrieval confidence statistics per response.

---

## Tech Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Local LLM** | Mistral-7B-Instruct-v0.2 (GGUF Q4_K_M) | 4-bit quantized local generation via `llama-cpp-python` with full GPU layer offload (`n_gpu_layers: -1`) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Lightweight, high-accuracy semantic embeddings for boundary detection and vector search |
| **Framework** | LlamaIndex (`llama-index-core`) | Orchestration for chunk metadata schema, vector indexing, and compact response synthesis |
| **Vision & OCR** | OpenCV, Tesseract OCR, `pdf2image` | Optical character recognition pipeline with image binarization for scanned filings |
| **PDF Extraction** | `PyPDF2` | Rapid digital-layer text extraction |
| **Interface** | Gradio | Conversational web UI with real-time indexing logs and multi-turn state management |

---

## Setup & Local Installation

### Prerequisites
- Linux / Ubuntu / Google Colab (with NVIDIA GPU runtime)
- Python 3.10+
- System packages: `poppler-utils` and `tesseract-ocr`

### 1. System Dependencies
```bash
sudo apt-get update -qq
sudo apt-get install -y -qq poppler-utils tesseract-ocr
```

### 2. Environment Setup
```bash
git clone [https://github.com/your-username/pharma-doc-intelligence-rag.git](https://github.com/your-username/pharma-doc-intelligence-rag.git)
cd pharma-doc-intelligence-rag
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
Install pre-compiled CUDA wheels for `llama-cpp-python` to ensure GPU offloading:

```bash
# Pre-built CUDA 12.2 / 12.x wheels
pip install --no-cache-dir --only-binary llama-cpp-python llama-cpp-python \
  --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu122](https://abetlen.github.io/llama-cpp-python/whl/cu122)

# Core libraries
pip install PyPDF2 langchain-text-splitters llama-index \
  llama-index-embeddings-huggingface llama-index-llms-llama-cpp \
  gradio pdf2image pytesseract Pillow opencv-python-headless "uvicorn<0.30.0"
```

### 4. Run the Pipeline
```bash
python app.py
```
The script will check for the quantized GGUF model locally, download it if missing (~4.1 GB), initialize CUDA offloading, and launch the Gradio server.

---

## Core Engineering Decisions

- **Hybrid OCR Pipeline:** Running OCR on every page degrades performance. The conditional check (`extract_text_from_page`) runs fast digital extraction first and triggers OpenCV/Tesseract only when character yields fall to zero.
- **Adaptive Document Boundaries:** Bundled PDF page lengths fluctuate across vendors. Calculating semantic drift between adjacent page prefixes (`similarity < 0.75`) dynamically isolates logical sub-documents without manual rule authoring.
- **Filtered-to-Global Query Fallback:** Metadata filtering eliminates cross-document hallucinations, but strict filters can cause false negatives if the user's intent prediction is slightly off. The automated fallback guarantees the retriever still answers questions from context even when document-type predictions misfire.
