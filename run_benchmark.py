import os
import json
import pickle
import uuid
from datetime import datetime, timezone
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Import your actual components
from src.retrieve import HybridRetriever
from src.generate import generate_report
from src.evaluate import CreditEvaluator

MIN_FAITHFULNESS = 0.85
MIN_LOGIC = 0.80
MIN_OVERALL = 0.75
MIN_MUST_MENTION_COVERAGE = 0.70


def _flatten_report_text(report: dict) -> str:
    findings = report.get("findings", [])
    findings_text = " ".join(
        [
            f"{f.get('title', '')} {f.get('evidence', '')} {f.get('summary', '')}"
            for f in findings
            if isinstance(f, dict)
        ]
    )
    return f"{report.get('executive_summary', '')} {findings_text}".lower()


def _extract_materiality_labels(report: dict) -> list[str]:
    labels = []
    for f in report.get("findings", []):
        if isinstance(f, dict):
            labels.append(str(f.get("materiality", "")).strip().upper())
    return [x for x in labels if x]


def _expected_verdict_match(expected: str, predicted_labels: list[str]) -> bool:
    exp = str(expected or "ANY").upper()
    if exp == "ANY":
        return True
    return exp in predicted_labels


def _must_mention_coverage(report: dict, must_mention: list[str]) -> float:
    terms = [str(t).lower() for t in (must_mention or []) if str(t).strip()]
    if not terms:
        return 1.0
    text = _flatten_report_text(report)
    covered = sum(1 for t in terms if t in text)
    return covered / len(terms)


def _logic_gate_check(logic_gate: str, score_card: dict) -> bool:
    gate = str(logic_gate or "general_delta").lower()
    if gate in {"5x_rule", "10pct_rule", "boilerplate_filter"}:
        return float(score_card.get("metrics", {}).get("gatekeeper_compliance", 0.0)) >= MIN_LOGIC
    return True


def _load_previous_results(output_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(output_csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(output_csv_path)
    except Exception:
        return pd.DataFrame()

def _normalize_vectors(obj: object) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        return obj.astype("float32")
    if isinstance(obj, dict):
        vectors = obj.get("vectors")
        if isinstance(vectors, np.ndarray):
            return vectors.astype("float32")
    return None


def _normalize_metadata(obj: object) -> list[dict] | None:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("metadata", "chunks"):
            val = obj.get(key)
            if isinstance(val, list):
                return val
    return None


def _load_retrieval_artifacts(base_dir: str) -> tuple[np.ndarray, list[dict]]:
    """
    Loads retrieval artifacts with backward compatibility:
    1) Preferred legacy format: data/faiss_index.bin + data/metadata.pkl
    2) Fallback format: data/filings/*_vecs.pkl + *_chunks.pkl
    """
    index_path = os.path.join(base_dir, "data", "faiss_index.bin")
    metadata_path = os.path.join(base_dir, "data", "metadata.pkl")
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        vectors = index.reconstruct_n(0, index.ntotal).astype("float32")
        with open(metadata_path, "rb") as f:
            metadata_obj = pickle.load(f)
        metadata = _normalize_metadata(metadata_obj)
        if metadata is None:
            raise ValueError("metadata.pkl is not a supported list/dict metadata format.")
        return vectors, metadata

    filings_dir = os.path.join(base_dir, "data", "filings")
    if not os.path.isdir(filings_dir):
        data_dir = os.path.join(base_dir, "data")
        found = os.listdir(data_dir) if os.path.exists(data_dir) else "Directory missing"
        raise FileNotFoundError(f"Missing data/filings directory. Found in data/: {found}")

    all_vecs: list[np.ndarray] = []
    all_meta: list[dict] = []
    v_files = [f for f in os.listdir(filings_dir) if f.endswith("_vecs.pkl")]
    print(f"Found {len(v_files)} vector files. Pairing with chunks...")

    for v_file in sorted(v_files):
        v_path = os.path.join(filings_dir, v_file)
        c_file = v_file.replace("_vecs.pkl", "_chunks.pkl")
        c_path = os.path.join(filings_dir, c_file)
        if not os.path.exists(c_path):
            continue

        with open(v_path, "rb") as vf:
            vec_obj = pickle.load(vf)
        with open(c_path, "rb") as cf:
            meta_obj = pickle.load(cf)

        vec_data = _normalize_vectors(vec_obj)
        meta_data = _normalize_metadata(meta_obj)
        if vec_data is None or meta_data is None:
            print(f"Skipping {v_file}: unsupported artifact structure.")
            continue

        if len(meta_data) != vec_data.shape[0]:
            print(
                f"Skipping {v_file}: Mismatch between vectors ({vec_data.shape[0]}) and chunks ({len(meta_data)})"
            )
            continue

        all_vecs.append(vec_data)
        all_meta.extend(meta_data)

    if not all_vecs:
        raise FileNotFoundError(f"No valid pairs found in {filings_dir}. Check file naming.")

    final_vectors = np.vstack(all_vecs)
    print(f"Total Database Size: {len(all_meta)} chunks.")
    return final_vectors, all_meta

def main():
    # 1. PATH SETUP (Aligned with Project Structure)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(base_dir, ".env"))
    if not os.getenv("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY. Ensure it is set in environment or .env file.")
        return

    test_cases_path = os.path.join(base_dir, "tests", "test_cases.jsonl")
    output_csv_path = os.path.join(base_dir, "benchmark_results.csv")
    run_artifacts_dir = os.path.join(base_dir, "benchmark_runs")
    os.makedirs(run_artifacts_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    run_utc = datetime.now(timezone.utc).isoformat()
    prev_df = _load_previous_results(output_csv_path)

    # 2. LOAD DATA ARTIFACTS
    print("Loading retrieval artifacts...")
    try:
        vectors, metadata = _load_retrieval_artifacts(base_dir)
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    # 3. INITIALIZE HYBRID ENGINE
    # This matches your dataclass requiring (vectors, metadata)
    retriever = HybridRetriever(vectors=vectors, metadata=metadata)
    evaluator = CreditEvaluator(judge_model="gpt-4o")
    results = []

    # 4. LOAD EVALUATION SUITE
    if not os.path.exists(test_cases_path):
        print(f"Test cases not found at {test_cases_path}")
        return

    with open(test_cases_path, "r") as f:
        test_cases = [json.loads(line) for line in f]

    # 5. EXECUTION LOOP (Mapping to your ingest.py sections)
    print(f"Running Benchmark against Signal Taxonomy (Sections 1, 1A, 3, 7, 8)...")
    print(f"Run ID: {run_id}")
    for case in tqdm(test_cases, desc="Benchmarking"):
        try:
            # Matches your generate_report signature
            report = generate_report(
                retriever=retriever,
                ticker=case['ticker'],
                section_type=case['section_type'], # e.g., 'section_7'
                year_a=case['year_a'],
                year_b=case['year_b'],
                query=case['query']
            )

            # Retrieve evidence from BOTH years used by generation to avoid judge/context mismatch.
            chunks_a = retriever.retrieve(
                query=case['query'], 
                ticker=case['ticker'], 
                section_type=case['section_type'],
                year=case['year_a'],
                top_k=10
            )
            chunks_b = retriever.retrieve(
                query=case['query'],
                ticker=case['ticker'],
                section_type=case['section_type'],
                year=case['year_b'],
                top_k=10
            )
            chunks = chunks_a + chunks_b

            # Evaluate against your credit-specific metrics
            score_card = evaluator.evaluate_report(report, chunks)
            faithfulness = float(score_card['metrics'].get('faithfulness', 0))
            logic = float(score_card['metrics'].get('gatekeeper_compliance', 0))
            overall = float(score_card.get('overall_score', 0))

            expected_verdict = case.get("expected_verdict", "ANY")
            must_mention = case.get("must_mention", [])
            logic_gate = case.get("logic_gate", "general_delta")

            predicted_labels = _extract_materiality_labels(report)
            verdict_match = _expected_verdict_match(expected_verdict, predicted_labels)
            mention_coverage = _must_mention_coverage(report, must_mention)
            gate_ok = _logic_gate_check(logic_gate, score_card)
            quality_gate_pass = (
                faithfulness >= MIN_FAITHFULNESS
                and logic >= MIN_LOGIC
                and overall >= MIN_OVERALL
                and mention_coverage >= MIN_MUST_MENTION_COVERAGE
                and verdict_match
                and gate_ok
            )

            results.append({
                "case_id": case['case_id'],
                "ticker": case['ticker'],
                "section": case['section_type'],
                "year_a": case['year_a'],
                "year_b": case['year_b'],
                "run_id": run_id,
                "run_utc": run_utc,
                "judge_model": evaluator.judge_model,
                "overall_score": overall,
                "faithfulness": faithfulness,
                "logic_compliance": logic,
                "evidence_density": float(score_card['metrics'].get('evidence_density', 0)),
                "judge_status": score_card['status'],
                "expected_verdict": expected_verdict,
                "predicted_materialities": "|".join(predicted_labels) if predicted_labels else "",
                "verdict_match": verdict_match,
                "must_mention_coverage": round(mention_coverage, 3),
                "logic_gate": logic_gate,
                "logic_gate_pass": gate_ok,
                "status": "PASS" if quality_gate_pass else "FAIL"
            })

        except Exception as e:
            print(f"\nSkipping {case.get('case_id')} | Error: {e}")
            results.append({
                "case_id": case.get('case_id'),
                "run_id": run_id,
                "run_utc": run_utc,
                "status": "ERROR",
                "error_msg": str(e)
            })

    # 6. EXPORT RESULTS
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    json_report_path = os.path.join(run_artifacts_dir, f"{run_id}.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*40)
    print("FINANCIAL RAG BENCHMARK SUMMARY")
    print(f"Total Cases: {len(df)}")
    print(f"Mean Score: {df['overall_score'].dropna().mean():.2f}")
    print(f"Pass Rate: {(df['status'] == 'PASS').mean()*100:.1f}%")
    print(f"Faithfulness Mean: {df['faithfulness'].dropna().mean():.2f}")
    print(f"Logic Mean: {df['logic_compliance'].dropna().mean():.2f}")
    print(f"Verdict Match Rate: {df['verdict_match'].fillna(False).mean()*100:.1f}%")
    print(f"Must-Mention Coverage Mean: {df['must_mention_coverage'].dropna().mean():.2f}")

    if not prev_df.empty and "overall_score" in prev_df.columns:
        prev_score = prev_df["overall_score"].dropna().mean()
        curr_score = df["overall_score"].dropna().mean()
        print(f"Score Delta vs Previous Run: {curr_score - prev_score:+.3f}")

    if not prev_df.empty and "status" in prev_df.columns:
        prev_pass = (prev_df["status"] == "PASS").mean() * 100
        curr_pass = (df["status"] == "PASS").mean() * 100
        print(f"Pass Rate Delta vs Previous Run: {curr_pass - prev_pass:+.2f}pp")

    print(f"Report: benchmark_results.csv")
    print(f"Run Artifact: benchmark_runs/{run_id}.json")
    print("="*40)

if __name__ == "__main__":
    main()