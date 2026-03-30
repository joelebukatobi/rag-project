from __future__ import annotations

import json
from typing import Dict, List, Tuple
from openai import OpenAI
from src.retrieve import HybridRetriever

def _build_context(chunks: List[Dict[str, object]], label: str) -> str:
    """Builds a structured context window for the model."""
    lines = [f"### {label} DATA"]
    for i, c in enumerate(chunks, start=1):
        # We now include the meta_section_len to help with Signal #18
        meta = f" (Section Length: {c.get('meta_section_len', 'N/A')} chars)"
        lines.append(
            f"[{i}] {c['ticker']} {c['year']} | {c['section_type']} | ID: {c['chunk_id']}{meta}"
        )
        lines.append(str(c["text"])[:2000])
    return "\n".join(lines)


# --- THE CREDIT SCORING RUBRIC (The "Law of the Land") ---
_UNDERWRITING_MATERIALITY_GUIDE = """
ANALYSIS GUIDELINES (NO SCORING):

1. LIQUIDITY CONTEXT (The 5x Rule):
   - If [Total Liquidity] > 5x the exposure of a specific risk (e.g., a lawsuit or vendor loss), 
     classify that risk as 'LOW MATERIALITY'. 
   - Reason: The company's balance sheet provides a sufficient safety buffer.

2. DISCLOSURE COMPLEXITY (Meta-Signal #18):
   - If YoY Disclosure Size increases by >20%, classify as 'MEDIUM MATERIALITY'.
   - Tag as: 'Increased Reporting Complexity'. 

3. FINANCIAL NOISE VS. STRESS:
   - Mark-to-market debt value changes (Fair Value) are 'MARKET MATERIALITY'.
   - Do NOT flag as 'High Materiality' unless paired with 'Refinancing' or 'Covenant' warnings.

4. LEGAL THRESHOLDS:
   - Classify lawsuits as 'LOW MATERIALITY' unless the contingency exceeds 10% of cash reserves.
   - If it exceeds 10%, classify as 'HIGH MATERIALITY'.

5. STRATEGIC SHIFTS (Item 1):
   - Any brand sunset, market exit, or segment consolidation is 'MEDIUM MATERIALITY'.
"""

# --- TAXONOMY-DRIVEN PROMPTS ---
_TAXONOMY_INSTRUCTIONS = """
Analyze the provided SEC excerpts using the Credit Signal Taxonomy (v1). 
Focus on identifying the following signal clusters:
1. BUSINESS FRAGILITY: Customer concentration, supply chain shifts, competitive pressure.
2. RISK INTENSITY: New risk factors, expanded risk language, regulatory escalation.
3. LIQUIDITY STRESS: Phrases like 'may not be sufficient', refinancing risk, capital allocation shifts.
4. MANAGEMENT TONE: Increased hedging language, specificity shifts (vague to specific), or sentiment decay.
5. CROSS-SECTION CONTRADICTIONS: Mismatches between management optimism and risk disclosures.

STRICT CREDIT ANALYSIS RULES (FAIR VALUE GATEKEEPER):
- If you detect a decrease in the 'Fair Value' of debt in Item 8:
  1. CROSS-CHECK Item 7 (MD&A) and Item 1A (Risk) for terms like 'liquidity', 'refinancing', 'covenant', or 'facility access'.
  2. IF AND ONLY IF those terms indicate actual stress, flag it as a 'Credit Stress Signal'.
  3. OTHERWISE, categorize it as 'Market Risk (Interest Rate-driven)' and explicitly state that no firm-specific liquidity stress was found.

STRICT LEGAL ANALYSIS RULES (ITEM 3):
- Detect expansion in legal proceedings (more cases, higher complexity, or new jurisdictions).
- CROSS-CHECK Cash & Cash Equivalents in Item 8.
- GATEKEEPER: Do NOT flag legal proceedings as 'Liquidity Stress' unless:
  1. Estimated loss contingencies are quantified and exceed 10% of current cash/ST investments.
  2. Management explicitly mentions legal settlements as a risk to debt covenants.
  3. There is an 'Emphasis of Matter' regarding legal going-concern risks.
- OTHERWISE: Label as 'Increased Regulatory/Legal Headwinds' or 'Business Fragility' without implying a credit liquidity event.

STRICT CREDIT ANALYSIS RULES:
{_UNDERWRITING_MATERIALITY_GUIDE}
"""

def compare(
    retriever: HybridRetriever,
    ticker: str,
    section_type: str,
    year_a: int,
    year_b: int,
    query: str,
    top_k: int = 10, # Increased k to capture more taxonomy signals
    model: str = "gpt-4o-mini",
) -> Dict[str, object]:
    client = OpenAI()

    # Retrieve comparative chunks
    chunks_a = retriever.retrieve(query=query, ticker=ticker, year=year_a, section_type=section_type, top_k=top_k)
    chunks_b = retriever.retrieve(query=query, ticker=ticker, year=year_b, section_type=section_type, top_k=top_k)

    prompt = f"""
        ROLE: Senior Credit Underwriter
        TASK: Delta Analysis of SEC filings for {ticker} ({year_a} vs {year_b}).
        INSTRUCTIONS:
        {_TAXONOMY_INSTRUCTIONS}

        EVIDENCE REQUIREMENTS:
        - Cite specific chunk IDs for every signal detected.
        - If a section length (Meta Signal #18) has changed significantly, flag it.
        - Distinguish between 'Boilerplate' and 'Material' changes.
        - Distinguish between 'Market Risk' and 'Credit Stress' per the Rules above

    {_build_context(chunks_a, f'BASE YEAR {year_a}')}

    {_build_context(chunks_b, f'TARGET YEAR {year_b}')}
    """.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a specialized Credit Analytics Engine (CreditDelta)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return {
        "ticker": ticker,
        "section_type": section_type,
        "year_a": year_a,
        "year_b": year_b,
        "raw_diff": response.choices[0].message.content,
    }

def generate_structured_output(
    diff_result: Dict[str, object],
    model: str = "gpt-4o-mini",
) -> Dict[str, object]:
    """
    Maps the raw taxonomy analysis into a professional Underwriting Report.
    Uses the 5x and 10% Gatekeepers to determine Materiality levels.
    """
    client = OpenAI()

    # Define the target JSON structure for the UI
    schema_hint = {
        "executive_summary": "High-conviction BLUF of strategic/financial shifts.",
        "findings": [
            {
                "category": "STRATEGIC | REGULATORY | FINANCIAL | GOVERNANCE",
                "materiality": "HIGH | MEDIUM | LOW | MARKET",
                "title": "Finding Headline",
                "evidence": "Detailed excerpt from text",
                "verdict": "Underwriting conclusion citing the Gatekeeper rule applied",
                "source": "Chunk ID"
            }
        ],
        "strategic_outlook": {
            "primary_driver": "The most significant delta found",
            "net_posture": "STABLE | CAUTIONARY | IMPROVING",
            "liquidity_buffer": "Statement on 5x/10% rule status"
        }
    }

    # Construct the final transformation prompt
    prompt = f"""
    ROLE: Senior Credit Committee Member
    TASK: Convert Delta Analysis into an Underwriting Report.
    
    UNDERWRITING GATEKEEPERS:
    {_UNDERWRITING_MATERIALITY_GUIDE}

    INSTRUCTIONS:
    1. For every finding, apply the 5x Liquidity and 10% Legal thresholds.
    2. If a risk is mitigated by liquidity (5x Rule), mark as 'LOW MATERIALITY'.
    3. Categorize Interest-Rate debt fluctuations as 'MARKET MATERIALITY'.
    4. Provide the 'net_posture' based on the cumulative materiality of findings.

    ANALYSIS DATA TO CONVERT:
    {diff_result.get('raw_diff', '')}

    OUTPUT FORMAT (Strict JSON):
    {json.dumps(schema_hint)}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a specialized Credit Analytics Engine that outputs strictly valid JSON for executive review."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    
    return json.loads(response.choices[0].message.content)