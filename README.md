# Analyst-Ready SEC Filings RAG System

AI Engineering project for Fordham University.

## Team
- Joel Ebukatobi (Co-author)
- [Classmate Name] (Co-author)

## Course
- Course: AI Engineering
- Professor: Apostolos Filippas
- Institution: Fordham University

## What This Does
This project ingests SEC 10-K filings from `eloukas/edgar-corpus`, parses key sections, builds section-aware chunks, applies hybrid retrieval (FAISS + BM25), compares filing years with GPT-4o, and generates analyst-ready structured JSON outputs.

## Repository Layout
- app.py
- notebook.ipynb
- requirements.txt
- .env.example
- data/chunks.pkl (generated)
- data/embeddings.pkl (generated)
- src/ingest.py
- src/chunk.py
- src/embed.py
- src/retrieve.py
- src/generate.py
- src/evaluate.py

## Setup
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set `OPENAI_API_KEY`.

## Run Streamlit
```bash
streamlit run app.py
```

## Notebook
Open `notebook.ipynb` and run sections in order. The notebook is organized into 10 labeled sections that progressively build the full system.

## Deployment
Deploy `app.py` to Streamlit Community Cloud and set `OPENAI_API_KEY` in app secrets.
