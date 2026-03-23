from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx


def read_semicolon_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def find_latest_multiseed_file_for_algorithm(
    root: Path,
    algorithm_name: str,
) -> tuple[Path, list[dict]]:
    candidates: list[tuple[float, Path, list[dict]]] = []

    for path in root.rglob("multiseed_test_detailed_*.csv"):
        rows = read_semicolon_csv(path)
        if any(row.get("algorithm") == algorithm_name for row in rows):
            candidates.append((path.stat().st_mtime, path, rows))

    if not candidates:
        raise FileNotFoundError(
            f"No multiseed detailed CSV found for algorithm '{algorithm_name}'."
        )

    candidates.sort(key=lambda item: item[0])
    _mtime, path, rows = candidates[-1]
    return path, rows


def summarize_by_seed(rows: list[dict], algorithm_name: str) -> list[dict]:
    algo_rows = [row for row in rows if row.get("algorithm") == algorithm_name]

    grouped: dict[int, list[dict]] = {}
    for row in algo_rows:
        seed = to_int(row.get("seed"))
        grouped.setdefault(seed, []).append(row)

    summary_rows: list[dict] = []
    for seed in sorted(grouped.keys()):
        seed_rows = grouped[seed]
        summary_rows.append(
            {
                "algorithm": algorithm_name,
                "seed": seed,
                "mean_total_score": mean([to_float(r["total_score"]) for r in seed_rows]),
                "mean_assignments_selected": mean(
                    [to_float(r["assignments_selected"]) for r in seed_rows]
                ),
                "mean_invalid_actions": mean(
                    [to_float(r["invalid_actions"]) for r in seed_rows]
                ),
                "mean_total_match_score": mean(
                    [to_float(r["total_match_score"]) for r in seed_rows]
                ),
                "num_test_scenarios": len(seed_rows),
            }
        )
    return summary_rows


def summarize_greedy(rows: list[dict]) -> dict:
    greedy_rows = [row for row in rows if row.get("algorithm") == "greedy"]
    if not greedy_rows:
        raise ValueError("No greedy rows found in the provided detailed CSV.")

    return {
        "algorithm": "greedy",
        "mean_total_score": mean([to_float(r["total_score"]) for r in greedy_rows]),
        "mean_assignments_selected": mean(
            [to_float(r["assignments_selected"]) for r in greedy_rows]
        ),
        "mean_invalid_actions": mean(
            [to_float(r["invalid_actions"]) for r in greedy_rows]
        ),
        "mean_total_match_score": mean(
            [to_float(r["total_match_score"]) for r in greedy_rows]
        ),
        "num_test_scenarios": len(greedy_rows),
    }


def plot_mean_total_score(
    dqn_seed_rows: list[dict],
    qrdqn_seed_rows: list[dict],
    greedy_summary: dict,
    path: Path,
) -> None:
    dqn_x = [to_int(row["seed"]) for row in dqn_seed_rows]
    dqn_y = [to_float(row["mean_total_score"]) for row in dqn_seed_rows]

    qrdqn_x = [to_int(row["seed"]) for row in qrdqn_seed_rows]
    qrdqn_y = [to_float(row["mean_total_score"]) for row in qrdqn_seed_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(dqn_x, dqn_y, marker="o", label="DQN")
    plt.plot(qrdqn_x, qrdqn_y, marker="o", label="QR-DQN")
    plt.axhline(
        to_float(greedy_summary["mean_total_score"]),
        linestyle="--",
        label="Greedy mean",
    )
    plt.xlabel("Seed")
    plt.ylabel("Mean Total Score on Test Set")
    plt.title("Multi-Seed Comparison - Mean Total Score")
    plt.xticks(sorted(set(dqn_x + qrdqn_x)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_assignments(
    dqn_seed_rows: list[dict],
    qrdqn_seed_rows: list[dict],
    greedy_summary: dict,
    path: Path,
) -> None:
    dqn_x = [to_int(row["seed"]) for row in dqn_seed_rows]
    dqn_y = [to_float(row["mean_assignments_selected"]) for row in dqn_seed_rows]

    qrdqn_x = [to_int(row["seed"]) for row in qrdqn_seed_rows]
    qrdqn_y = [to_float(row["mean_assignments_selected"]) for row in qrdqn_seed_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(dqn_x, dqn_y, marker="o", label="DQN")
    plt.plot(qrdqn_x, qrdqn_y, marker="o", label="QR-DQN")
    plt.axhline(
        to_float(greedy_summary["mean_assignments_selected"]),
        linestyle="--",
        label="Greedy mean",
    )
    plt.xlabel("Seed")
    plt.ylabel("Mean Assignments Selected")
    plt.title("Multi-Seed Comparison - Mean Assignments")
    plt.xticks(sorted(set(dqn_x + qrdqn_x)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_invalid_actions(
    dqn_seed_rows: list[dict],
    qrdqn_seed_rows: list[dict],
    greedy_summary: dict,
    path: Path,
) -> None:
    dqn_x = [to_int(row["seed"]) for row in dqn_seed_rows]
    dqn_y = [to_float(row["mean_invalid_actions"]) for row in dqn_seed_rows]

    qrdqn_x = [to_int(row["seed"]) for row in qrdqn_seed_rows]
    qrdqn_y = [to_float(row["mean_invalid_actions"]) for row in qrdqn_seed_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(dqn_x, dqn_y, marker="o", label="DQN")
    plt.plot(qrdqn_x, qrdqn_y, marker="o", label="QR-DQN")
    plt.axhline(
        to_float(greedy_summary["mean_invalid_actions"]),
        linestyle="--",
        label="Greedy mean",
    )
    plt.xlabel("Seed")
    plt.ylabel("Mean Invalid Actions")
    plt.title("Multi-Seed Comparison - Mean Invalid Actions")
    plt.xticks(sorted(set(dqn_x + qrdqn_x)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_score_boxplot(
    dqn_rows: list[dict],
    qrdqn_rows: list[dict],
    greedy_rows: list[dict],
    path: Path,
) -> None:
    greedy_scores = [to_float(row["total_score"]) for row in greedy_rows if row.get("algorithm") == "greedy"]
    dqn_scores = [to_float(row["total_score"]) for row in dqn_rows if row.get("algorithm") == "dqn"]
    qrdqn_scores = [to_float(row["total_score"]) for row in qrdqn_rows if row.get("algorithm") == "qrdqn"]

    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [greedy_scores, dqn_scores, qrdqn_scores],
        labels=["Greedy", "DQN", "QR-DQN"],
    )
    plt.ylabel("Total Score")
    plt.title("Distribution of Test-Scenario Scores")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_root = project_root / "data" / "outputs"
    plot_dir = output_root / "plots"
    summary_dir = output_root / "comparison_summaries"

    plot_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    dqn_path, dqn_rows = find_latest_multiseed_file_for_algorithm(output_root, "dqn")
    qrdqn_path, qrdqn_rows = find_latest_multiseed_file_for_algorithm(output_root, "qrdqn")

    greedy_source_rows = dqn_rows if any(r.get("algorithm") == "greedy" for r in dqn_rows) else qrdqn_rows

    dqn_seed_summary = summarize_by_seed(dqn_rows, "dqn")
    qrdqn_seed_summary = summarize_by_seed(qrdqn_rows, "qrdqn")
    greedy_summary = summarize_greedy(greedy_source_rows)

    summary_rows = []
    summary_rows.extend(dqn_seed_summary)
    summary_rows.extend(qrdqn_seed_summary)
    summary_rows.append(greedy_summary)

    csv_path = summary_dir / "multiseed_dqn_qrdqn_summary.csv"
    xlsx_path = summary_dir / "multiseed_dqn_qrdqn_summary.xlsx"

    write_excel_friendly_csv(csv_path, summary_rows)
    write_simple_xlsx(xlsx_path, summary_rows, sheet_name="Summary")

    plot_mean_total_score(
        dqn_seed_rows=dqn_seed_summary,
        qrdqn_seed_rows=qrdqn_seed_summary,
        greedy_summary=greedy_summary,
        path=plot_dir / "multiseed_mean_total_score_dqn_qrdqn.png",
    )

    plot_mean_assignments(
        dqn_seed_rows=dqn_seed_summary,
        qrdqn_seed_rows=qrdqn_seed_summary,
        greedy_summary=greedy_summary,
        path=plot_dir / "multiseed_mean_assignments_dqn_qrdqn.png",
    )

    plot_mean_invalid_actions(
        dqn_seed_rows=dqn_seed_summary,
        qrdqn_seed_rows=qrdqn_seed_summary,
        greedy_summary=greedy_summary,
        path=plot_dir / "multiseed_mean_invalid_actions_dqn_qrdqn.png",
    )

    plot_score_boxplot(
        dqn_rows=dqn_rows,
        qrdqn_rows=qrdqn_rows,
        greedy_rows=greedy_source_rows,
        path=plot_dir / "multiseed_score_boxplot_greedy_dqn_qrdqn.png",
    )

    print(f"Loaded DQN detailed CSV:    {dqn_path}")
    print(f"Loaded QR-DQN detailed CSV: {qrdqn_path}")
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()