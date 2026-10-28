# AI Portfolio — Backend API (FastAPI)

Independent backend service for the AI-powered portfolio chatbot and resume matcher.

## Features
- **FastAPI REST API**: High-performance asynchronous endpoints for chat, streaming chat, and JD matching.
- **Google Gemini Integration**: Fast LLM reasoning and Gemini embeddings via HTTPX REST.
- **Qdrant Vector Database**: Cloud vector similarity search for resume retrieval.
- **MongoDB Atlas**: Persistent storage for session histories and match queries.
- **Standalone Architecture**: Fully independent service that communicates with any frontend (Vercel, Netlify, localhost) via CORS.

---

## Local Development Setup

1. **Install dependencies**:
   ```bash
   uv sync
   # or: pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Fill in your `.env` variables:
   - `gemini_api_key`: Google AI Studio API key
   - `MODEL`: Gemini model name (`gemini-3.8-flash`)
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
   uv run python ingest.py
   ```

4. **Start the API Server**:
   ```bash
   uv run uvicorn main:app --reload --port 8000
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
   - `gemini_api_key`
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
