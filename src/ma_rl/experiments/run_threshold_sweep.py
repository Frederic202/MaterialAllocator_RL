from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path

from ma_rl.baselines import solve_greedy
from ma_rl.data import load_scenario_from_json
from ma_rl.domain import (
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
    apply_assignment_set_score,
)
from ma_rl.matching import generate_feasible_matches


THRESHOLDS: list[float | None] = [0.0, -0.25, -0.5, None]


def build_common_configs():
    hard_rule_config = HardRuleConfig(
        allowed_type_pairs={("BR", "HR")},
        enforce_dimension_rules=True,
        allow_missing_dimensions=True,
    )

    score_weights = ScoreWeights(
        material_category_weight=0.0,
        order_category_weight=0.0,
        due_date_weight=1.0,
        production_date_weight=0.5,
        assignment_cost_weight=1.0,
        pile_penalty_weight=0.0,
        order_completion_bonus=10.0,
        homogeneity_penalty=5.0,
        unassigned_order_step_penalty=2.0,
    )

    feasible_match_config = FeasibleMatchConfig(
        rule_set_name="threshold_sweep_v1",
        include_non_allocatable_debug_matches=True,
    )

    return hard_rule_config, score_weights, feasible_match_config


def threshold_to_label(threshold: float | None) -> str:
    return "no_threshold" if threshold is None else str(threshold)


def filter_matches_by_threshold(feasible_matches, threshold: float | None):
    return [
        match
        for match in feasible_matches
        if match.allocatable
        and match.score is not None
        and (threshold is None or match.score >= threshold)
    ]


def analyze_single_scenario_for_threshold(
    scenario_path: Path,
    split_name: str,
    threshold: float | None,
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
) -> dict:
    scenario = load_scenario_from_json(scenario_path)

    feasible_matches = generate_feasible_matches(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    allocatable_matches = [m for m in feasible_matches if m.allocatable and m.score is not None]
    threshold_matches = filter_matches_by_threshold(feasible_matches, threshold)

    unique_assignable_order_steps = {
        m.order_step_id for m in threshold_matches
    }
    unique_assignable_materials = {
        m.material_id for m in threshold_matches
    }

    greedy_result = solve_greedy(
        feasible_matches=feasible_matches,
        penalty_threshold=threshold,
    )

    apply_assignment_set_score(
        assignment_set=greedy_result.assignment_set,
        scenario=scenario,
        weights=score_weights,
    )

    return {
        "split": split_name,
        "scenario_id": scenario.scenario_id,
        "scenario_file": scenario_path.name,
        "threshold": threshold_to_label(threshold),
        "materials": len(scenario.materials),
        "order_steps": len(scenario.order_steps),
        "feasible_matches_total": len(feasible_matches),
        "allocatable_matches": len(allocatable_matches),
        "threshold_matches": len(threshold_matches),
        "unique_assignable_order_steps": len(unique_assignable_order_steps),
        "unique_assignable_materials": len(unique_assignable_materials),
        "greedy_assignments_selected": len(greedy_result.assignment_set.assignments),
        "greedy_total_match_score": greedy_result.assignment_set.total_match_score,
        "greedy_total_penalty": greedy_result.assignment_set.total_penalty,
        "greedy_total_bonus": greedy_result.assignment_set.total_bonus,
        "greedy_total_score": greedy_result.assignment_set.total_score,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["threshold"], []).append(row)

    summary_rows: list[dict] = []

    for threshold_label, group_rows in grouped.items():
        assignable_steps = [r["unique_assignable_order_steps"] for r in group_rows]
        threshold_matches = [r["threshold_matches"] for r in group_rows]
        greedy_assignments = [r["greedy_assignments_selected"] for r in group_rows]
        greedy_scores = [float(r["greedy_total_score"]) for r in group_rows]

        dead_scenarios = sum(1 for r in group_rows if r["unique_assignable_order_steps"] == 0)
        zero_assignment_scenarios = sum(1 for r in group_rows if r["greedy_assignments_selected"] == 0)

        summary_rows.append(
            {
                "threshold": threshold_label,
                "scenarios": len(group_rows),
                "dead_scenarios": dead_scenarios,
                "zero_assignment_scenarios": zero_assignment_scenarios,
                "assignable_steps_mean": statistics.mean(assignable_steps),
                "assignable_steps_min": min(assignable_steps),
                "assignable_steps_max": max(assignable_steps),
                "threshold_matches_mean": statistics.mean(threshold_matches),
                "threshold_matches_min": min(threshold_matches),
                "threshold_matches_max": max(threshold_matches),
                "greedy_assignments_mean": statistics.mean(greedy_assignments),
                "greedy_assignments_min": min(greedy_assignments),
                "greedy_assignments_max": max(greedy_assignments),
                "greedy_total_score_mean": statistics.mean(greedy_scores),
                "greedy_total_score_min": min(greedy_scores),
                "greedy_total_score_max": max(greedy_scores),
            }
        )

    return summary_rows


def print_summary(summary_rows: list[dict]) -> None:
    print("=" * 90)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 90)

    for row in summary_rows:
        print(
            f"threshold={row['threshold']} | "
            f"scenarios={row['scenarios']} | "
            f"dead={row['dead_scenarios']} | "
            f"zero_assignments={row['zero_assignment_scenarios']} | "
            f"assignable_steps_mean={row['assignable_steps_mean']:.2f} | "
            f"threshold_matches_mean={row['threshold_matches_mean']:.2f} | "
            f"greedy_assignments_mean={row['greedy_assignments_mean']:.2f} | "
            f"greedy_score_mean={row['greedy_total_score_mean']:.2f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    generated_root = project_root / "data" / "scenarios" / "generated"
    output_root = project_root / "data" / "outputs" / "threshold_sweep"

    hard_rule_config, score_weights, feasible_match_config = build_common_configs()

    detailed_rows: list[dict] = []

    for split_name in ["train", "val", "test"]:
        split_dir = generated_root / split_name
        scenario_files = sorted(split_dir.glob("*.json"))

        for scenario_file in scenario_files:
            for threshold in THRESHOLDS:
                row = analyze_single_scenario_for_threshold(
                    scenario_path=scenario_file,
                    split_name=split_name,
                    threshold=threshold,
                    hard_rule_config=hard_rule_config,
                    score_weights=score_weights,
                    feasible_match_config=feasible_match_config,
                )
                detailed_rows.append(row)

    summary_rows = build_summary_rows(detailed_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_csv = output_root / f"threshold_sweep_detailed_{timestamp}.csv"
    summary_csv = output_root / f"threshold_sweep_summary_{timestamp}.csv"

    write_csv(detailed_csv, detailed_rows)
    write_csv(summary_csv, summary_rows)

    print_summary(summary_rows)
    print()
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()