import argparse
import os
import pandas as pd


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    text = str(val).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def analyze(csv_path: str, top_n: int = 10) -> int:
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No rows in benchmark CSV.")
        return 1

    if "status" not in df.columns:
        print("Missing required column: status")
        return 1

    pass_rate = (df["status"] == "PASS").mean() * 100
    _print_header("Benchmark Snapshot")
    print(f"Rows: {len(df)}")
    print(f"Pass rate: {pass_rate:.1f}%")
    if "overall_score" in df.columns:
        print(f"Mean overall score: {df['overall_score'].dropna().mean():.3f}")
    if "faithfulness" in df.columns:
        print(f"Mean faithfulness: {df['faithfulness'].dropna().mean():.3f}")

    fail_df = df[df["status"] != "PASS"].copy()
    _print_header("Failure Counts")
    print(f"Fail/Error cases: {len(fail_df)}")
    if fail_df.empty:
        print("No failures found.")
        return 0

    # Thresholds mirror run_benchmark defaults.
    if "faithfulness" in fail_df.columns:
        print(f"faithfulness_below_0_85: {(fail_df['faithfulness'] < 0.85).sum()}")
    if "logic_compliance" in fail_df.columns:
        print(f"logic_below_0_80: {(fail_df['logic_compliance'] < 0.80).sum()}")
    if "overall_score" in fail_df.columns:
        print(f"overall_below_0_75: {(fail_df['overall_score'] < 0.75).sum()}")
    if "must_mention_coverage" in fail_df.columns:
        print(f"must_mention_below_0_70: {(fail_df['must_mention_coverage'] < 0.70).sum()}")
    if "verdict_match" in fail_df.columns:
        print(
            f"verdict_mismatch: {(~fail_df['verdict_match'].apply(_to_bool)).sum()}"
        )
    if "logic_gate_pass" in fail_df.columns:
        print(
            f"logic_gate_fail: {(~fail_df['logic_gate_pass'].apply(_to_bool)).sum()}"
        )

    if "section" in fail_df.columns:
        _print_header("Failure Breakdown by Section")
        sec = fail_df.groupby("section").size().sort_values(ascending=False)
        for k, v in sec.items():
            print(f"{k}: {v}")

    if "ticker" in fail_df.columns:
        _print_header("Top Tickers by Failures")
        tick = fail_df.groupby("ticker").size().sort_values(ascending=False).head(10)
        for k, v in tick.items():
            print(f"{k}: {v}")

    _print_header(f"Worst {top_n} Cases")
    sort_col = "overall_score" if "overall_score" in fail_df.columns else "case_id"
    worst = fail_df.sort_values(sort_col, ascending=True).head(top_n)
    cols = [
        c
        for c in [
            "case_id",
            "ticker",
            "section",
            "overall_score",
            "faithfulness",
            "logic_compliance",
            "must_mention_coverage",
            "verdict_match",
            "logic_gate_pass",
            "status",
            "error_msg",
        ]
        if c in worst.columns
    ]
    print(worst[cols].to_string(index=False))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark failure reasons.")
    parser.add_argument(
        "--csv",
        default="benchmark_results.csv",
        help="Path to benchmark results CSV (default: benchmark_results.csv)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst cases to print (default: 10)",
    )
    args = parser.parse_args()
    raise SystemExit(analyze(args.csv, top_n=args.top_n))


if __name__ == "__main__":
    main()
