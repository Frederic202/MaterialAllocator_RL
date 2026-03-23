from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_semicolon_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return list(reader)


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


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


def aggregate_by_scenario(rows: list[dict], algorithm_name: str) -> dict[str, dict]:
    algo_rows = [row for row in rows if row.get("algorithm") == algorithm_name]

    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_assignments: dict[str, list[float]] = defaultdict(list)

    for row in algo_rows:
        scenario_id = row["scenario_id"]
        grouped_scores[scenario_id].append(to_float(row["total_score"]))
        grouped_assignments[scenario_id].append(to_float(row["assignments_selected"]))

    aggregated: dict[str, dict] = {}
    for scenario_id in grouped_scores:
        aggregated[scenario_id] = {
            "mean_total_score": mean(grouped_scores[scenario_id]),
            "std_total_score": std(grouped_scores[scenario_id]),
            "mean_assignments": mean(grouped_assignments[scenario_id]),
            "std_assignments": std(grouped_assignments[scenario_id]),
        }

    return aggregated


def plot_grouped_bars(
    scenario_ids: list[str],
    greedy_values: list[float],
    dqn_values: list[float],
    qrdqn_values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    x = list(range(len(scenario_ids)))
    width = 0.25

    fig_width = max(12, len(scenario_ids) * 0.8)
    plt.figure(figsize=(fig_width, 6))

    plt.bar([i - width for i in x], greedy_values, width=width, label="Greedy")
    plt.bar(x, dqn_values, width=width, label="DQN")
    plt.bar([i + width for i in x], qrdqn_values, width=width, label="QR-DQN")

    plt.xticks(x, scenario_ids, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    # Legende außerhalb
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=(0, 0, 0.84, 1))

    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_grouped_bars_with_errorbars(
    scenario_ids: list[str],
    greedy_means: list[float],
    greedy_stds: list[float],
    dqn_means: list[float],
    dqn_stds: list[float],
    qrdqn_means: list[float],
    qrdqn_stds: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    x = list(range(len(scenario_ids)))
    width = 0.25

    fig_width = max(12, len(scenario_ids) * 0.8)
    plt.figure(figsize=(fig_width, 6))

    plt.bar(
        [i - width for i in x],
        greedy_means,
        width=width,
        yerr=greedy_stds,
        capsize=4,
        label="Greedy",
    )
    plt.bar(
        x,
        dqn_means,
        width=width,
        yerr=dqn_stds,
        capsize=4,
        label="DQN",
    )
    plt.bar(
        [i + width for i in x],
        qrdqn_means,
        width=width,
        yerr=qrdqn_stds,
        capsize=4,
        label="QR-DQN",
    )

    plt.xticks(x, scenario_ids, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)

    # Legende außerhalb
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout(rect=(0, 0, 0.84, 1))

    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    output_root = project_root / "data" / "outputs"
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dqn_path, dqn_rows = find_latest_multiseed_file_for_algorithm(output_root, "dqn")
    qrdqn_path, qrdqn_rows = find_latest_multiseed_file_for_algorithm(output_root, "qrdqn")

    # Greedy ist in den Detailed-Dateien mit drin
    greedy_source_rows = dqn_rows if any(r.get("algorithm") == "greedy" for r in dqn_rows) else qrdqn_rows

    greedy_agg = aggregate_by_scenario(greedy_source_rows, "greedy")
    dqn_agg = aggregate_by_scenario(dqn_rows, "dqn")
    qrdqn_agg = aggregate_by_scenario(qrdqn_rows, "qrdqn")

    scenario_ids = sorted(
        set(greedy_agg.keys()) & set(dqn_agg.keys()) & set(qrdqn_agg.keys())
    )

    # Score-Daten
    greedy_score_means = [greedy_agg[s]["mean_total_score"] for s in scenario_ids]
    greedy_score_stds = [greedy_agg[s]["std_total_score"] for s in scenario_ids]

    dqn_score_means = [dqn_agg[s]["mean_total_score"] for s in scenario_ids]
    dqn_score_stds = [dqn_agg[s]["std_total_score"] for s in scenario_ids]

    qrdqn_score_means = [qrdqn_agg[s]["mean_total_score"] for s in scenario_ids]
    qrdqn_score_stds = [qrdqn_agg[s]["std_total_score"] for s in scenario_ids]

    # Assignment-Daten
    greedy_assignment_means = [greedy_agg[s]["mean_assignments"] for s in scenario_ids]
    greedy_assignment_stds = [greedy_agg[s]["std_assignments"] for s in scenario_ids]

    dqn_assignment_means = [dqn_agg[s]["mean_assignments"] for s in scenario_ids]
    dqn_assignment_stds = [dqn_agg[s]["std_assignments"] for s in scenario_ids]

    qrdqn_assignment_means = [qrdqn_agg[s]["mean_assignments"] for s in scenario_ids]
    qrdqn_assignment_stds = [qrdqn_agg[s]["std_assignments"] for s in scenario_ids]

    # Plot 1: Mean Total Score pro Szenario
    plot_grouped_bars_with_errorbars(
        scenario_ids=scenario_ids,
        greedy_means=greedy_score_means,
        greedy_stds=greedy_score_stds,
        dqn_means=dqn_score_means,
        dqn_stds=dqn_score_stds,
        qrdqn_means=qrdqn_score_means,
        qrdqn_stds=qrdqn_score_stds,
        ylabel="Mean Total Score",
        title="Greedy vs. DQN vs. QR-DQN - Mean Total Score per Scenario",
        output_path=plot_dir / "greedy_vs_dqn_vs_qrdqn_multiseed_total_score.png",
    )

    # Plot 2: Mean Assignments pro Szenario
    plot_grouped_bars_with_errorbars(
        scenario_ids=scenario_ids,
        greedy_means=greedy_assignment_means,
        greedy_stds=greedy_assignment_stds,
        dqn_means=dqn_assignment_means,
        dqn_stds=dqn_assignment_stds,
        qrdqn_means=qrdqn_assignment_means,
        qrdqn_stds=qrdqn_assignment_stds,
        ylabel="Mean Assignments Selected",
        title="Greedy vs. DQN vs. QR-DQN - Mean Assignments per Scenario",
        output_path=plot_dir / "greedy_vs_dqn_vs_qrdqn_multiseed_assignments.png",
    )

    # Plot 3: Ohne Errorbars, falls du es cleaner willst
    plot_grouped_bars(
        scenario_ids=scenario_ids,
        greedy_values=greedy_score_means,
        dqn_values=dqn_score_means,
        qrdqn_values=qrdqn_score_means,
        ylabel="Mean Total Score",
        title="Greedy vs. DQN vs. QR-DQN - Mean Total Score per Scenario",
        output_path=plot_dir / "greedy_vs_dqn_vs_qrdqn_multiseed_total_score_clean.png",
    )

    print(f"Loaded DQN detailed CSV:    {dqn_path}")
    print(f"Loaded QR-DQN detailed CSV: {qrdqn_path}")
    print(f"Scenarios plotted: {len(scenario_ids)}")
    print(f"Saved to: {plot_dir}")


if __name__ == "__main__":
    main()