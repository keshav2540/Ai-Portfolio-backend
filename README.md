# AI Portfolio — Backend API (FastAPI)

Independent backend service for the AI-powered portfolio chatbot and resume matcher.

## Features
- **FastAPI REST API**: High-performance asynchronous endpoints for chat, streaming chat, and JD matching.
- **Groq LLM Integration**: Ultra-fast LLM generation and streaming via LangChain (`ChatGroq`) with automatic fallback (`llama-3.3-70b-versatile` -> `llama-3.1-8b-instant`).
- **Multimodal Role Fit Matching**: Supports Job Descriptions as PDF (native & scanned), DOCX (paragraphs & tables), Images (PNG/JPG/WEBP via Groq Vision OCR), or raw text.
- **Qdrant Vector Database**: Cloud vector similarity search for semantic resume retrieval (768-dim embeddings).
- **MongoDB Atlas**: Persistent storage for session histories and match queries.
- **Standalone Architecture**: Fully independent service that communicates with any frontend (Vercel, Netlify, localhost) via CORS.

## Project Structure

```text
├── Procfile              # Process definition for Render / PaaS deployment
├── README.md             # Project documentation & setup guide
├── render.yaml           # Declarative blueprint for Render deployment
├── pyproject.toml        # Python project configuration & dependencies
├── requirements.txt      # Locked production dependencies
├── uv.lock               # Lockfile for reproducible builds
├── .python-version       # Pinned Python version (3.11.10)
├── .env.example          # Template for required environment variables
├── .gitignore            # Clean git ignore rules (secrets, venv, cache)
├── main.py               # FastAPI application entrypoint & route handlers
├── llm.py                # Groq LLM adapter (LangChain ChatGroq, fallback, streaming)
├── embeddings.py         # Gemini 768-dim embeddings for Qdrant Cloud
├── rag.py                # Qdrant Cloud vector search & adaptive RAG
├── matcher.py            # Resume-to-job matching engine with multi-JD support
├── jd_parser.py          # PDF, DOCX, and Image parser (Groq Vision OCR)
├── store.py              # MongoDB Atlas session history & match store
├── ingest.py             # Resume ingestion CLI tool for Qdrant
├── test_suite.py         # Unit test suite (21 unit tests)
└── resume.txt            # Local resume text fallback (git-ignored)
```

---

## Local Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Fill in your `.env` variables:
   - `GROQ_API_KEY`: Groq Cloud API key (from [console.groq.com](https://console.groq.com))
   - `GROQ_MODEL`: Primary Groq model (default: `llama-3.3-70b-versatile`)
   - `GROQ_FALLBACK_MODEL`: Fallback Groq model (default: `llama-3.1-8b-instant`)
   - `GROQ_VISION_MODEL`: Groq Vision model for image/scanned OCR (default: `qwen/qwen3.8-27b`)
   - `gemini_api_key`: Google API key (only used for Qdrant 768-dim embeddings)
   - `EMBEDDING_MODEL`: Embedding model (`models/gemini-embedding-001`)
   - `EMBEDDING_DIM`: Embedding vector dimension (`768`)
   - `Qdrant_url`: Qdrant Cloud cluster endpoint
   - `Qdrant_api_key`: Qdrant Cloud API key
   - `MONGO_URI`: MongoDB Atlas connection URI
   - `DB_NAME`: Database name (`ai_portfolio`)
   - `FRONTEND_ORIGIN`: Allowed frontend origin (e.g. `http://localhost:5173` or your Vercel frontend URL)
   - `INGEST_ADMIN_KEY`: Secret key for resume ingestion endpoint

3. **Ingest Resume**:
   ```bash
   python ingest.py
   ```

4. **Start the API Server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   - Interactive Swagger API Docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/`

---

## Deploy to Render (Web Service)

This backend repository is production-ready for deployment as a Web Service on **Render** using the included `render.yaml`.

### 1-Click Blueprint Deploy
1. Push this repository to GitHub.
2. In the Render Dashboard, click **New + > Blueprint**.
3. Connect this repository (`Ai-Portfolio-backend`).
4. Render will read `render.yaml` and configure:
   - **Runtime**: Python 3.11.10
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/`
5. Fill in your environment variables in the Render dashboard prompts:
   - `GROQ_API_KEY`
   - `gemini_api_key` (for embeddings)
   - `Qdrant_url`
   - `Qdrant_api_key`
   - `MONGO_URI`
   - `INGEST_ADMIN_KEY`
   - `FRONTEND_ORIGIN` (Set to your independent frontend URL, e.g. `https://your-frontend.vercel.app`)
6. Click **Apply**.

---

## Connecting with Frontend

Your frontend (hosted independently on Vercel or elsewhere) connects to this backend via:
1. **Frontend `VITE_API_URL`**: Set to your Render backend URL (e.g. `https://ai-portfolio-backend.onrender.com`).
2. **Backend `FRONTEND_ORIGIN`**: Set to your frontend URL (e.g. `https://your-frontend.vercel.app`) so CORS requests are allowed.
