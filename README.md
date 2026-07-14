# voice_rag
Multimodal RAG system with voice interaction, hybrid retrieval (Vector + BM25 + RRF + MMR), Groq/Ollama routing, OCR, and offline fallback.


# 🎙️ Hybrid Voice-Enabled Multimodal RAG Assistant

A **Hybrid Retrieval-Augmented Generation (RAG)** system that enables users to ask questions about their documents using **voice or text**. The assistant combines **semantic search**, **keyword search**, **OCR**, **table extraction**, and **hybrid retrieval** to generate accurate answers from uploaded documents.

The system automatically switches between **Groq (online)** and **Ollama (offline)**, ensuring uninterrupted document question answering even without an internet connection.

---

## 🚀 Features

- 📄 Supports multiple document formats
  - PDF
  - DOCX
  - PPTX
  - TXT
  - Images (PNG, JPG, JPEG, BMP, TIFF, WEBP)

- 🔍 Hybrid Retrieval
  - Chroma Vector Search
  - BM25 Keyword Search
  - Reciprocal Rank Fusion (RRF)
  - Maximum Marginal Relevance (MMR)

- 🎯 Cross-Encoder Reranking

- 🖼 OCR-based Image Processing using Tesseract

- 📊 Table Extraction from documents

- 🎤 Voice-Based Question Answering
  - Speech-to-Text (Faster Whisper)
  - Text-to-Speech (Piper TTS)

- 🤖 Automatic Model Routing
  - Uses **Groq** when internet is available
  - Automatically falls back to **Ollama** for offline inference

- 📚 Reference Tracking
  - Maintains image and table references
  - Provides source citations with page numbers

- ⚡ Duplicate document detection using file hashing

- 💾 Persistent Chroma Vector Database

---

# 🏗 System Architecture

```
                    Documents
(PDF / DOCX / PPTX / TXT / Images)
                    │
                    ▼
             Document Partitioning
                    │
                    ▼
          Image & Table Extraction
                    │
                    ▼
               OCR Processing
                    │
                    ▼
              Intelligent Chunking
                    │
                    ▼
             Chunk Enrichment
                    │
                    ▼
        Embedding Generation (BGE)
                    │
                    ▼
        Chroma DB + BM25 Index
                    │
                    ▼
           Hybrid Retrieval
     (Vector + BM25 + RRF + MMR)
                    │
                    ▼
        Cross Encoder Reranker
                    │
                    ▼
            Context Assembly
                    │
                    ▼
             Model Router
           ┌────────┴────────┐
           │                 │
           ▼                 ▼
      Groq (Online)     Ollama (Offline)
           │                 │
           └────────┬────────┘
                    ▼
            Answer Generation
                    │
                    ▼
        Voice Response (Piper TTS)
```

---

# 📂 Project Structure

```
Hybrid-Voice-RAG/
│
├── data/
│   ├── documents/
│   └── images/
│
├── db/
│   ├── chroma_db/
│   └── bm25.pkl
│
├── logs/
│
├── metadata/
│   ├── documents.json
│   ├── chunks.json
│   ├── ingestion_log.json
│   └── reference_store.json
│
├── src/
│   ├── answer_generator.py
│   ├── chunking.py
│   ├── config.py
│   ├── enrichment.py
│   ├── image_processor.py
│   ├── model_router.py
│   ├── partitioning.py
│   ├── pipeline.py
│   ├── reference_store.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── table_processor.py
│   └── vectorstore.py
│
├── voices/
│
├── ingest.py
├── interrupt_service.py
├── speech_to_text.py
├── voice_chat.py
├── voice_service.py
│
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

# ⚙️ Technologies Used

### Programming Language

- Python

### Retrieval

- ChromaDB
- BM25
- LangChain

### Embeddings

- BAAI BGE Small Embedding Model

### Reranking

- Cross Encoder (MiniLM)

### LLMs

- Groq API
- Ollama

### OCR

- Tesseract OCR

### Speech

- Faster Whisper
- Piper TTS
- WebRTC Voice Activity Detection

### Libraries

- LangChain
- HuggingFace
- Unstructured
- ChromaDB
- NumPy
- SoundDevice
- Pillow

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/Hybrid-Voice-RAG.git

cd Hybrid-Voice-RAG
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside the project directory.

```env
GROQ_API_KEY=your_groq_api_key

TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

# ▶️ Usage

## Step 1 — Ingest Documents

Place your documents inside

```
data/documents/
```

Run

```bash
python ingest.py
```

---

## Step 2 — Start Voice Assistant

```bash
python voice_chat.py
```

Speak your questions naturally, and the assistant will retrieve relevant information from the indexed documents and answer using voice.

---

# 🔄 Retrieval Pipeline

1. Document Partitioning
2. Chunk Creation
3. OCR & Table Processing
4. Chunk Enrichment
5. Embedding Generation
6. Chroma Vector Search
7. BM25 Keyword Search
8. Reciprocal Rank Fusion (RRF)
9. Maximum Marginal Relevance (MMR)
10. Cross-Encoder Reranking
11. Context Assembly
12. Model Routing (Groq/Ollama)
13. Answer Generation
14. Voice Response

---

# 💡 Future Improvements

- Web Interface
- Multi-user Support
- Chat History
- Streaming Responses
- Docker Deployment
- Cloud Storage Integration
- Multi-language Support
- Source Highlighting in Responses

---

# 📜 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Madhav Kotha**

If you found this project useful, feel free to ⭐ the repository.
