# Analyst-Ready SEC Filings RAG System

AI Engineering project for Fordham University.

## Team

- Joel Onwuanaku (Co-author)
- Claudia Gisemba (Co-author)

## Course

- Course: AI Engineering
- Professor: Apostolos Filippas
- Institution: Fordham University

## What This Does

The **Streamlit app** pulls **live SEC 10-K text** via [`edgartools`](https://pypi.org/project/edgartools/) (`Company`, `get_filings`), selects a canonical filing per fiscal year (base **10-K** by default; optional **10-K/A**), extracts Items (1, 1A, 3, 7, 8), **chunks** and **embeds** excerpts, runs **hybrid retrieval** (FAISS + BM25), then uses **two LLM steps** (`compare` → structured report) with selectable **Views** (Board, Risk Analyst, Research, Legal & Compliance, Regulatory / Auditor). Outputs include provenance (accession, extraction tier, etc.), optional view-specific sections, and **disk caches** under `data/filings/` and `data/reports/` (gitignored).

**Extras:** [`src/ingest.py`](src/ingest.py) still supports alternate ingestion from **`eloukas/edgar-corpus`** for notebooks or offline experiments. **`run_benchmark.py`** plus [`src/evaluate.py`](src/evaluate.py) benchmark report quality against fixed test cases.

Do **not** commit `.env` or generated `data/**` pickles; see [.gitignore](.gitignore).

## Requirements

- **Python ≥ 3.13** (see [`pyproject.toml`](pyproject.toml)).

## Repository Layout (high level)

| Path | Role |
|------|------|
| [`app.py`](app.py) | Streamlit UI, filing fetch + cache, report rendering |
| [`run_benchmark.py`](run_benchmark.py) | CLI benchmark harness |
| [`src/generate.py`](src/generate.py), [`src/view_schemas.py`](src/view_schemas.py) | LLM prompts, views, report cache |
| [`src/filing_select.py`](src/filing_select.py) | Deterministic base vs amended 10-K selection |
| [`src/retrieve.py`](src/retrieve.py) | HybridRetriever |
| [`src/chunk.py`](src/chunk.py), [`src/embed.py`](src/embed.py) | Chunking / embeddings |
| [`src/evaluate.py`](src/evaluate.py) | Faithfulness / gatekeeper-style metrics |
| [`src/ingest.py`](src/ingest.py) | Optional HF / live ingest helpers |
| [`scripts/`](scripts/), [`tests/test_cases.jsonl`](tests/test_cases.jsonl) | Benchmark helpers & fixtures |
| [`.streamlit/config.toml`](.streamlit/config.toml) | Theme |

## Setup (local)

### 1. Clone and enter the repo

```bash
git clone git@github.com:joelebukatobi/rag-project.git
cd rag-project
```

### 2. Environment variables

Copy the example env file and set your OpenAI key:

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

### 3. Install dependencies

**Option A — `uv` (matches [`uv.lock`](uv.lock)):**

```bash
uv sync
uv run streamlit run app.py
```

**Option B — `venv` + pip:**

```bash
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 4. SEC identity

`edgartools` requires a real **User-Agent identity** string. The app calls `set_identity(...)` in [`app.py`](app.py). For your own deployment, replace it with **your** contact email per [SEC fair-access guidance](https://www.sec.gov/os/accessing-edgar-data).

### 5. Optional: benchmarks

With `.env` configured and caches available as needed:

```bash
python run_benchmark.py
```

(Exact flags and test data are in [`tests/test_cases.jsonl`](tests/test_cases.jsonl) and [`run_benchmark.py`](run_benchmark.py).)

## Notebook

Open [`notebook.ipynb`](notebook.ipynb) and run sections in order if you are using the notebook track.

## Deployment (Streamlit Community Cloud)

Point the app entry to **`app.py`**. Set **`OPENAI_API_KEY`** (and any other secrets) in **Streamlit app secrets** (not in the repo). Pushes to the connected branch usually trigger a redeploy; confirm the deployed commit in the Cloud UI. **Private GitHub repos** work once Streamlit has been authorized to read private repositories. The public `*.streamlit.app` URL may still be reachable by anyone with the link unless you restrict access in Streamlit’s sharing settings for your plan.
