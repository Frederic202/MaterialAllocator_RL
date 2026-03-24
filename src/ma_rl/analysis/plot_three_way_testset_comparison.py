from __future__ import annotations

import csv
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


def to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def get_latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return files[-1]


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    eval_dir = project_root / "data" / "outputs" / "testset_eval"
    plot_dir = project_root / "data" / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    greedy_file = get_latest_file(eval_dir, "greedy_testset_*.csv")
    dqn_file = get_latest_file(eval_dir, "dqn_testset_*.csv")
    qrdqn_file = get_latest_file(eval_dir, "qrdqn_testset_*.csv")

    greedy_rows = read_semicolon_csv(greedy_file)
    dqn_rows = read_semicolon_csv(dqn_file)
    qrdqn_rows = read_semicolon_csv(qrdqn_file)

    greedy_by_scenario = {row["scenario_id"]: row for row in greedy_rows}
    dqn_by_scenario = {row["scenario_id"]: row for row in dqn_rows}
    qrdqn_by_scenario = {row["scenario_id"]: row for row in qrdqn_rows}

    scenario_ids = sorted(
        set(greedy_by_scenario.keys())
        & set(dqn_by_scenario.keys())
        & set(qrdqn_by_scenario.keys())
    )

    greedy_scores = [to_float(greedy_by_scenario[s]["total_score"]) for s in scenario_ids]
    dqn_scores = [to_float(dqn_by_scenario[s]["total_score"]) for s in scenario_ids]
    qrdqn_scores = [to_float(qrdqn_by_scenario[s]["total_score"]) for s in scenario_ids]

    greedy_assignments = [to_int(greedy_by_scenario[s]["assignments_selected"]) for s in scenario_ids]
    dqn_assignments = [to_int(dqn_by_scenario[s]["assignments_selected"]) for s in scenario_ids]
    qrdqn_assignments = [to_int(qrdqn_by_scenario[s]["assignments_selected"]) for s in scenario_ids]

    x = list(range(len(scenario_ids)))
    width = 0.25

    # Plot 1: Total Score pro Szenario
    plt.figure(figsize=(11, 5.5))
    plt.bar([i - width for i in x], greedy_scores, width=width, label="Greedy")
    plt.bar(x, dqn_scores, width=width, label="DQN")
    plt.bar([i + width for i in x], qrdqn_scores, width=width, label="QR-DQN")
    plt.xticks(x, scenario_ids, rotation=45)
    plt.ylabel("Total Score")
    plt.title("Greedy vs. DQN vs. QR-DQN - Total Score per Scenario")
    plt.legend()
    plt.tight_layout()
    score_plot_path = plot_dir / "greedy_vs_dqn_vs_qrdqn_total_score.png"
    plt.savefig(score_plot_path, dpi=150)
    plt.close()

    # Plot 2: Assignments pro Szenario
    plt.figure(figsize=(11, 5.5))
    plt.bar([i - width for i in x], greedy_assignments, width=width, label="Greedy")
    plt.bar(x, dqn_assignments, width=width, label="DQN")
    plt.bar([i + width for i in x], qrdqn_assignments, width=width, label="QR-DQN")
    plt.xticks(x, scenario_ids, rotation=45)
    plt.ylabel("Assignments Selected")
    plt.title("Greedy vs. DQN vs. QR-DQN - Assignments per Scenario")
    plt.legend()
    plt.tight_layout()
    assignments_plot_path = plot_dir / "greedy_vs_dqn_vs_qrdqn_assignments.png"
    plt.savefig(assignments_plot_path, dpi=150)
    plt.close()

    # Plot 3: Mittelwert Total Score
    mean_scores = [
        mean(greedy_scores),
        mean(dqn_scores),
        mean(qrdqn_scores),
    ]
    labels = ["Greedy", "DQN", "QR-DQN"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, mean_scores)
    plt.ylabel("Mean Total Score")
    plt.title("Mean Total Score on Test Set")
    plt.tight_layout()
    mean_score_plot_path = plot_dir / "mean_total_score_greedy_dqn_qrdqn.png"
    plt.savefig(mean_score_plot_path, dpi=150)
    plt.close()

    # Plot 4: Mittelwert Assignments
    mean_assignments = [
        mean([float(v) for v in greedy_assignments]),
        mean([float(v) for v in dqn_assignments]),
        mean([float(v) for v in qrdqn_assignments]),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, mean_assignments)
    plt.ylabel("Mean Assignments Selected")
    plt.title("Mean Assignments on Test Set")
    plt.tight_layout()
    mean_assignments_plot_path = plot_dir / "mean_assignments_greedy_dqn_qrdqn.png"
    plt.savefig(mean_assignments_plot_path, dpi=150)
    plt.close()

    print(f"Loaded Greedy CSV:  {greedy_file}")
    print(f"Loaded DQN CSV:     {dqn_file}")
    print(f"Loaded QR-DQN CSV:  {qrdqn_file}")
    print()
    print(f"Saved: {score_plot_path}")
    print(f"Saved: {assignments_plot_path}")
    print(f"Saved: {mean_score_plot_path}")
    print(f"Saved: {mean_assignments_plot_path}")


if __name__ == "__main__":
    main()