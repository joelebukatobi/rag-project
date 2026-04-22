# Implementation plan (reproducible underwriting RAG)

This document tracks what we are fixing, how we will fix it, and the agreed product behavior (UI + `generate.py` + upstream filing selection).

## What we are fixing (problems to solve)

1. **Run-to-run and user-to-user inconsistency**  
   The same ticker, years, and section can produce different materiality, findings, or evidence because:
   - **Multiple 10-K-family filings** exist per fiscal year (10-K vs 10-K/A, multiple accessions). If the app picks a different filing, extracted Item text, chunks, and LLM context change.
   - **Retrieval and chunking** can differ slightly (ordering, tie-breaks, query text), which changes the evidence set passed to the model.
   - The **LLM report step** (`generate.py`) can amplify small evidence differences, especially in a two-step (delta → structure) flow.

2. **Weak audit trail**  
   Underwriting and risk need to answer: *which exact SEC document and which chunks* drove each conclusion. Today that is not always explicit in the product output.

3. **Prompt/view behavior was not first-class**  
   We need **aligned view profiles** (Board, Risk Analyst, etc.) that each have a defined **primary focus**, **UI surface**, and **LLM output shape**—not ad hoc persona text or custom roles.

**North star:** same inputs + same evidence ⇒ same (or defensibly equivalent) report JSON; every run is **attributable** to a canonical filing and chunk IDs.

---

## A. Reproducibility and auditability (upstream; not `generate.py` alone)

### A1. Canonical filing: base 10-K only; no silent 10-K/A

- **Default:** use the **base 10-K** for the selected fiscal year (non-amended).
- **10-K/A:** not selected automatically. Use a **10-K/A** only if the user **explicitly opts in** (e.g. a clear UI or documented intent). No silent fallback to amended filings when base extraction is short or noisy—prefer a **loud error** with filing metadata so the user can opt in.
- **Persist the choice:** store and reuse `accession_no`, `period_of_report`, and `filing_date` for `(ticker, fiscal_year)` so every repeat run hits the same document.

**Deliverable:** a stable index (e.g. `data/filings/filing_index.json` or metadata alongside cached chunks) keyed by `(ticker, fiscal_year)`.

### A2. Evidence fingerprint (stable evidence set)

Per run, log and hash:

- selected `accession_no` and dates  
- extraction path (attribute vs `Item` indexer vs `section()`) and section length  
- `top_k`, retriever settings, embedding model  
- retrieved `chunk_id` sets for year A and year B  

**Fingerprint** example: `hash(accession_a + accession_b + section_id + chunk_ids + prompt_version + model + view + underwriting_focus)`.

### A3. Deterministic retrieval

- Fixed chunking parameters; stable `chunk_id`s; deterministic sort and tie-breaks in hybrid retrieval.  
- Same filter inputs on every run for the same UI state.

### A4. Report cache (optional)

- Key cached reports on `evidence_fingerprint` + `prompt_version` + `model` + **view** + inputs.  
- Avoids cost drift; supports “same in, same out” when evidence is stable.

### A5. Provenance in JSON and UI

- Top-level `provenance` (or equivalent): accession(s), dates, extraction notes, chunk IDs used, `prompt_version`, `model`, **view** name, underwriting focus text.

---

## B. Aligned view profiles (no custom personas)

We are **not** supporting freeform “custom role” or open-ended persona text. The product uses **only** these five aligned views, each with a fixed prompt/schema contract and UI mapping.

| View                 | Primary focus                         | UI components to show                                      | LLM output emphasis (day 1: new fields per view)        |
| -------------------- | ------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| **Board**            | Strategic impact and BLUF             | Executive summary + Strategic outlook card                 | Short exec narrative; clear net posture; few findings  |
| **Risk Analyst**     | Material changes and evidence         | Operational and risk delta cards + source snippets         | Findings with citations; materiality; chunk-grounded     |
| **Research**         | Long-term trends                      | Multi-year meta-signals + disclosure growth %              | Trend fields; meta-signal #18 (disclosure size) + narrative |
| **Legal & Compliance** | Accuracy and hallucination check    | Judge scores + faithfulness metrics                        | Scores/flags: citation coverage, quote support, risk    |
| **Regulatory / Auditor** | Disclosure gaps                 | Comparative omission analysis (what was removed)           | Removed/changed topic lists; year-over-year diff framing |

**Implementation notes:**

- **`app.py`:** single **View** dropdown (these five only); **underwriting focus** remains the topical query.  
- **`src/generate.py`:** `view` selects which **JSON schema** and **second-stage prompt** to use (compare → structured output, with view-specific `schema_hint` and instructions).  
- **Strict enums and required fields** per view to reduce drift and to match the UI (each view may expose different top-level or nested fields).

### B1. User preferences (optional, secondary)

- If we keep a short **constraints** text area, it should apply **only** within non-negotiable rules (citations, no invented numbers, schema). No substitute for the five views.

### B2. Prompt assembly (conceptual)

`base_rules + view_rules + section_specific_gatekeepers + schema_for_view + evidence_context`  
Increment **`PROMPT_VERSION`** when any view schema or rule changes.

---

## C. Verification / acceptance

- **Repeatability:** same ticker, years, section, view, and underwriting focus run multiple times → same accession, same chunk IDs, same report JSON (or materially identical findings and materiality with same sources).  
- **No silent 10-K/A** unless user explicitly requested amended filing.  
- **View contract:** each view renders the agreed UI components without missing required model fields.  
- **Audit:** provenance block is always present and accurate.

---

## D. File ownership (rough)

| Area                         | Where it lives (conceptually)        |
| ---------------------------- | ------------------------------------ |
| Filing pick + accession pin  | `app.py` (`get_filing_data` and helpers) |
| Chunking / embeddings         | `src/chunk.py`, `src/embed.py`        |
| Retrieval                     | `src/retrieve.py`                     |
| View-specific LLM + schema   | `src/generate.py`                     |
| View + underwriting focus UI | `app.py` form + report render         |

---

## E. What changed recently (for collaborators)

- **`src/generate.py`:** two-step flow (`compare` → `generate_structured_output`) with **five aligned views** (schemas in `src/view_schemas.py`), **report disk cache** keyed by evidence fingerprint + view + `REPORT_PROMPT_VERSION`, and merged **provenance** on the report.
- **`app.py`:** **View** dropdown; **Use amended 10-K/A** checkbox (default off); upstream **deterministic filing pick** (no “longest of 12” tournament); cache files named with **accession slug** + `*_provenance.json`; **Provenance** expander; extra JSON blocks for Research / Legal / Regulatory views when present.
- **`src/filing_select.py`:** helpers for base vs amended 10-K and deterministic sort.

---

## F. Performance (must-haves stay; response time is a first-class goal)

**Targets (product language):**

- **Cache hit** (filings + chunks + vectors + report already on disk): **sub-second to a few seconds** for UI update; no redundant LLM calls when the evidence fingerprint and view match a stored report.
- **First-time / cold path** (no disk cache for that ticker/year/section): as fast as practical; dominant costs are usually **embedding** and **LLM** (two calls in the current `generate.py` flow). We do not remove correctness features to win speed; we **parallelize, cache, and warm** instead.

**Strategies (implementation):**

1. **Filing + vector cache (already partially there):** Keep `data/filings/*_chunks.pkl` + `*_vecs.pkl` per `(ticker, year, section)` so repeat runs skip SEC + embed. **Accession pinning** must key the same paths or metadata so we never rebuild embeddings for the wrong document.

2. **Report cache on evidence fingerprint:** After the first successful `generate_report` for `(evidence_fingerprint, view, prompt_version, model)`, store the final JSON (e.g. `data/reports/<hash>.pkl`). Second run with same inputs returns **without** `compare` / `generate_structured_output` — this is the biggest win for “feels instant” on repeat analysis.

3. **Warm heavy resources on app start:** `@st.cache_resource` for `SentenceTransformer` (and any large models) so the **first user** after deploy pays less cold-start on embed; optionally trigger a one-time warm-up import in `app.py` startup.

4. **Avoid duplicate work across years:** When loading year A and year B, **parallelize** independent I/O (thread pool or async where safe) for `get_filing_data` if the framework allows, so wall time is closer to `max(A, B)` than `A + B`.

5. **LLM latency (two-step flow):** Must-haves include view-specific structured output; options to reduce latency without dropping quality:
   - use **one** structured call only for views that allow it (optional later optimization per view);
   - keep `temperature=0`, trim context to top_k chunks only;
   - consider a **faster** model for draft step only if the team accepts it (policy decision).

6. **UI perception:** Use Streamlit spinners/messages that distinguish “fetching filing,” “embedding,” and “generating report” so waits feel predictable; optional streaming later.

**Non-negotiable:** Fingerprint and provenance logging must not be skipped for speed; cache keys must include view + evidence identity so we never serve a mismatched report.
