import json
import os

def generate_synthetic_suite():
    """
    Generates a structured test suite in JSONL format.
    Includes 'Edge Cases' specifically designed to trigger the Underwriting Gatekeepers.
    """
    os.makedirs("tests", exist_ok=True)
    
    # 1. High-Value Manual Seed Cases (The "Stress Tests")
    # These focus on the 5x Liquidity and 10% Legal rules.
    test_cases = [
        {
            "case_id": "STRESS-001-LIQUIDITY",
            "ticker": "DAL",
            "year_a": 2022, "year_b": 2023,
            "section_type": "section_7",
            "query": "Assess liquidity buffer and debt maturity profile.",
            "expected_verdict": "MEDIUM",
            "must_mention": ["revolving credit", "liquidity", "$"],
            "logic_gate": "5x_rule"
        },
        {
            "case_id": "STRESS-002-LEGAL",
            "ticker": "TSLA",
            "year_a": 2022, "year_b": 2023,
            "section_type": "section_3",
            "query": "Current status of DOJ/SEC investigations into Autopilot.",
            "expected_verdict": "HIGH",
            "must_mention": ["subpoena", "DOJ", "investigation"],
            "logic_gate": "10pct_rule"
        },
        {
            "case_id": "STRESS-003-BOILERPLATE",
            "ticker": "AAPL",
            "year_a": 2022, "year_b": 2023,
            "section_type": "section_1a",
            "query": "General market risk and competition.",
            "expected_verdict": "LOW",
            "must_mention": ["competition", "iPhone"],
            "logic_gate": "boilerplate_filter"
        }
    ]

    # 2. Programmatic diverse coverage (27 additional cases)
    tickers = [
        "MSFT", "AMZN", "NVDA", "META", "GOOGL", "NFLX", "JPM", "GS", "BA", "F", 
        "GM", "XOM", "CVX", "WMT", "TGT", "HD", "LOW", "DIS", "SBUX", "PFE", 
        "MRK", "JNJ", "UNH", "V", "MA", "ORCL", "COST"
    ]
    
    sections = ["section_1", "section_1a", "section_7"]

    for i, ticker in enumerate(tickers):
        sect = sections[i % len(sections)]
        test_cases.append({
            "case_id": f"GEN-{i+4:03d}-{ticker}",
            "ticker": ticker,
            "year_a": 2022,
            "year_b": 2023,
            "section_type": sect,
            "query": f"Analyze material changes in {ticker} strategic positioning.",
            "expected_verdict": "ANY",
            "must_mention": [],
            "logic_gate": "general_delta"
        })

    # Save to JSONL for streaming evaluation
    output_path = "tests/test_cases.jsonl"
    with open(output_path, "w") as f:
        for case in test_cases:
            f.write(json.dumps(case) + "\n")

    print(f"✅ Success: Generated {len(test_cases)} test cases at {output_path}")

if __name__ == "__main__":
    generate_synthetic_suite()