from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    benchmark_dir = project_root / "data" / "outputs" / "benchmarks"

    summary_files = sorted(benchmark_dir.glob("benchmark_summary_*.csv"))
    if not summary_files:
        print("No benchmark summary CSV found.")
        return

    latest = summary_files[-1]
    print(f"Reading: {latest}\n")

    with latest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"Algorithm: {row['algorithm']}")
            print(f"  runs: {row['runs']}")
            print(f"  total_score_mean: {row['total_score_mean']}")
            print(f"  total_score_std: {row['total_score_std']}")
            print(f"  assignments_selected_mean: {row['assignments_selected_mean']}")
            print(f"  cumulative_reward_mean: {row['cumulative_reward_mean']}")
            print(f"  invalid_actions_mean: {row['invalid_actions_mean']}")
            print()


if __name__ == "__main__":
    main()