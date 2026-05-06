import json
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.generate import generate_report

class CreditEvaluator:
    def __init__(self, judge_model: str = "gpt-4o"):
        self.client = OpenAI()
        self.judge_model = judge_model

    def evaluate_report(self, report: Dict[str, Any], context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs a 3-point inspection: Faithfulness, Gatekeeper Adherence, and Evidence Density.
        """
        # 1. Check Faithfulness (Did the model invent numbers?)
        faithfulness = self._check_faithfulness(report, context_chunks)
        
        # 2. Check Gatekeeper Compliance (5x/10% Rules)
        gatekeeper = self._check_gatekeeper_logic(report)
        
        # 3. Evidence Density (Verifiability)
        evidence_score = self._calculate_evidence_density(report)

        return {
            "overall_score": round((faithfulness + gatekeeper + evidence_score) / 3, 2),
            "metrics": {
                "faithfulness": faithfulness,
                "gatekeeper_compliance": gatekeeper,
                "evidence_density": evidence_score
            },
            "status": "PASS" if faithfulness > 0.8 else "FAIL"
        }

    def _check_faithfulness(self, report: Dict[str, Any], chunks: List[Dict[str, Any]]) -> float:
        """Uses LLM-as-a-Judge to verify claims against raw chunks."""
        context_text = "\n".join([f"ID {c['chunk_id']}: {c['text'][:500]}" for c in chunks])
        findings_text = json.dumps(report.get("findings", []))
        
        prompt = f"""
        COMPARE THE FINDINGS TO THE SOURCE DATA.
        SOURCE DATA: {context_text}
        GENERATED FINDINGS: {findings_text}
        
        Identify any claims, dollar amounts, or percentages in the FINDINGS that do not exist in the SOURCE DATA.
        Output a JSON: {{"hallucination_count": int, "total_claims": int}}
        """
        
        res = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(res.choices[0].message.content)
        return (data['total_claims'] - data['hallucination_count']) / max(data['total_claims'], 1)

    def _check_gatekeeper_logic(self, report: Dict[str, Any]) -> float:
        """Deterministic check for the 5x/10% underwriting rules."""
        findings = report.get("findings", [])
        if not findings: return 1.0
        
        score = 0
        for f in findings:
            # Rule: If materiality is HIGH, evidence must mention specific stress/thresholds
            if f['materiality'] == "HIGH":
                if any(x in f['evidence'].lower() for x in ["%", "$", "exceeds", "breach"]):
                    score += 1
            else:
                score += 1
        return score / len(findings)

    def _calculate_evidence_density(self, report: Dict[str, Any]) -> float:
        """Scores the 'verifiability' of the report based on ID citations."""
        findings = report.get("findings", [])
        if not findings: return 0.0
        
        cited = 0
        for f in findings:
            if f.get("source") and len(str(f["source"])) > 0:
                cited += 1
        return cited / len(findings)

def run_benchmark(retriever, test_cases: List[Dict[str, Any]]):
    evaluator = CreditEvaluator()
    results = []
    
    for case in test_cases:
        report = generate_report(
            retriever, case['ticker'], case['section_type'], 
            case['year_a'], case['year_b'], case['query']
        )
        # We need the chunks used for generation to evaluate faithfulness
        chunks = retriever.retrieve(case['query'], ticker=case['ticker'], top_k=10)
        
        score = evaluator.evaluate_report(report, chunks)
        results.append({"ticker": case['ticker'], "result": score})
        
    return results