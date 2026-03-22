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


def to_float(value: str) -> float:
    return float(value)


def to_int(value: str) -> int:
    return int(float(value))


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    eval_dir = project_root / "data" / "outputs" / "testset_eval"
    plot_dir = project_root / "data" / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    greedy_files = sorted(eval_dir.glob("greedy_testset_*.csv"))
    dqn_files = sorted(eval_dir.glob("dqn_testset_*.csv"))

    if not greedy_files or not dqn_files:
        print("Greedy or DQN CSV file not found.")
        return

    greedy_rows = read_semicolon_csv(greedy_files[-1])
    dqn_rows = read_semicolon_csv(dqn_files[-1])

    greedy_by_scenario = {row["scenario_id"]: row for row in greedy_rows}
    dqn_by_scenario = {row["scenario_id"]: row for row in dqn_rows}

    scenario_ids = sorted(set(greedy_by_scenario.keys()) & set(dqn_by_scenario.keys()))

    greedy_scores = [to_float(greedy_by_scenario[s]["total_score"]) for s in scenario_ids]
    dqn_scores = [to_float(dqn_by_scenario[s]["total_score"]) for s in scenario_ids]

    greedy_assignments = [to_int(greedy_by_scenario[s]["assignments_selected"]) for s in scenario_ids]
    dqn_assignments = [to_int(dqn_by_scenario[s]["assignments_selected"]) for s in scenario_ids]

    x = list(range(len(scenario_ids)))
    width = 0.35

    # Plot 1: Total Score
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], greedy_scores, width=width, label="Greedy")
    plt.bar([i + width / 2 for i in x], dqn_scores, width=width, label="DQN")
    plt.xticks(x, scenario_ids, rotation=45)
    plt.ylabel("Total Score")
    plt.title("Greedy vs. DQN - Total Score per Scenario")
    plt.legend()
    plt.tight_layout()
    score_plot_path = plot_dir / "greedy_vs_dqn_total_score.png"
    plt.savefig(score_plot_path, dpi=150)
    plt.close()

    # Plot 2: Assignments Selected
    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], greedy_assignments, width=width, label="Greedy")
    plt.bar([i + width / 2 for i in x], dqn_assignments, width=width, label="DQN")
    plt.xticks(x, scenario_ids, rotation=45)
    plt.ylabel("Assignments Selected")
    plt.title("Greedy vs. DQN - Assignments per Scenario")
    plt.legend()
    plt.tight_layout()
    assign_plot_path = plot_dir / "greedy_vs_dqn_assignments.png"
    plt.savefig(assign_plot_path, dpi=150)
    plt.close()

    print(f"Saved: {score_plot_path}")
    print(f"Saved: {assign_plot_path}")


if __name__ == "__main__":
    main()