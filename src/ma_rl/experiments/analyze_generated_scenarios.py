from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ma_rl.analysis import summarize_feasible_matches_by_type_pair, summarize_input_types
from ma_rl.baselines import solve_greedy
from ma_rl.data import load_scenario_from_json
from ma_rl.domain import (
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
    apply_assignment_set_score,
)
from ma_rl.matching import generate_feasible_matches


PENALTY_THRESHOLD = 0.0


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
        rule_set_name="analysis_v1",
        include_non_allocatable_debug_matches=True,
    )

    return hard_rule_config, score_weights, feasible_match_config


def _json_dumps_sorted(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def analyze_single_scenario(
    scenario_path: Path,
    split_name: str,
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
) -> tuple[dict, list[dict]]:
    scenario = load_scenario_from_json(scenario_path)

    input_summary = summarize_input_types(scenario)

    feasible_matches = generate_feasible_matches(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    type_pair_rows = summarize_feasible_matches_by_type_pair(
        feasible_matches=feasible_matches,
        scenario=scenario,
        penalty_threshold=PENALTY_THRESHOLD,
    )

    allocatable_matches = [m for m in feasible_matches if m.allocatable]
    scored_matches = [m for m in feasible_matches if m.score is not None]
    above_threshold_matches = [
        m for m in feasible_matches
        if m.score is not None and m.score >= PENALTY_THRESHOLD
    ]

    unique_assignable_order_steps = sorted(
        {
            m.order_step_id
            for m in feasible_matches
            if m.allocatable and m.score is not None and m.score >= PENALTY_THRESHOLD
        }
    )
    unique_assignable_materials = sorted(
        {
            m.material_id
            for m in feasible_matches
            if m.allocatable and m.score is not None and m.score >= PENALTY_THRESHOLD
        }
    )

    failed_rule_counts = Counter()
    for match in feasible_matches:
        for failed_rule_name in match.failed_rule_names:
            failed_rule_counts[failed_rule_name] += 1

    greedy_result = solve_greedy(
        feasible_matches=feasible_matches,
        penalty_threshold=PENALTY_THRESHOLD,
    )

    apply_assignment_set_score(
        assignment_set=greedy_result.assignment_set,
        scenario=scenario,
        weights=score_weights,
    )

    scenario_row = {
        "split": split_name,
        "scenario_id": scenario.scenario_id,
        "scenario_file": scenario_path.name,
        "materials": len(scenario.materials),
        "order_steps": len(scenario.order_steps),
        "material_type_counts": _json_dumps_sorted(dict(input_summary["material_type_counts"])),
        "order_step_type_counts": _json_dumps_sorted(dict(input_summary["order_step_type_counts"])),
        "feasible_matches_total": len(feasible_matches),
        "allocatable_matches": len(allocatable_matches),
        "non_allocatable_matches": len(feasible_matches) - len(allocatable_matches),
        "scored_matches": len(scored_matches),
        "above_threshold_matches": len(above_threshold_matches),
        "unique_assignable_order_steps": len(unique_assignable_order_steps),
        "unique_assignable_materials": len(unique_assignable_materials),
        "failed_rule_counts": _json_dumps_sorted(dict(failed_rule_counts)),
        "greedy_assignments_selected": len(greedy_result.assignment_set.assignments),
        "greedy_total_match_score": greedy_result.assignment_set.total_match_score,
        "greedy_total_penalty": greedy_result.assignment_set.total_penalty,
        "greedy_total_bonus": greedy_result.assignment_set.total_bonus,
        "greedy_total_score": greedy_result.assignment_set.total_score,
    }

    scenario_type_pair_rows = []
    for row in type_pair_rows:
        scenario_type_pair_rows.append(
            {
                "split": split_name,
                "scenario_id": scenario.scenario_id,
                "scenario_file": scenario_path.name,
                "mat_type_code": row["mat_type_code"],
                "prod_step_type_code": row["prod_step_type_code"],
                "total_matches": row["total_matches"],
                "allocatable_matches": row["allocatable_matches"],
                "non_allocatable_matches": row["non_allocatable_matches"],
                "scored_matches": row["scored_matches"],
                "threshold_matches": row["threshold_matches"],
                "avg_score": row["avg_score"],
                "min_score": row["min_score"],
                "max_score": row["max_score"],
                "failed_rule_names": _json_dumps_sorted(row["failed_rule_names"]),
            }
        )

    return scenario_row, scenario_type_pair_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_terminal_summary(rows: list[dict]) -> None:
    print("=" * 80)
    print("SCENARIO ANALYSIS SUMMARY")
    print("=" * 80)

    for row in rows:
        print(
            f"[{row['split']}] {row['scenario_id']} | "
            f"materials={row['materials']} | "
            f"order_steps={row['order_steps']} | "
            f"feasible={row['feasible_matches_total']} | "
            f"allocatable={row['allocatable_matches']} | "
            f"above_threshold={row['above_threshold_matches']} | "
            f"assignable_steps={row['unique_assignable_order_steps']} | "
            f"greedy_assignments={row['greedy_assignments_selected']} | "
            f"greedy_score={row['greedy_total_score']:.4f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    generated_root = project_root / "data" / "scenarios" / "generated"
    output_root = project_root / "data" / "outputs" / "scenario_analysis"

    hard_rule_config, score_weights, feasible_match_config = build_common_configs()

    all_scenario_rows: list[dict] = []
    all_type_pair_rows: list[dict] = []

    for split_name in ["train", "val", "test"]:
        split_dir = generated_root / split_name
        scenario_files = sorted(split_dir.glob("*.json"))

        for scenario_file in scenario_files:
            scenario_row, type_pair_rows = analyze_single_scenario(
                scenario_path=scenario_file,
                split_name=split_name,
                hard_rule_config=hard_rule_config,
                score_weights=score_weights,
                feasible_match_config=feasible_match_config,
            )
            all_scenario_rows.append(scenario_row)
            all_type_pair_rows.extend(type_pair_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv = output_root / f"scenario_summary_{timestamp}.csv"
    type_pair_csv = output_root / f"type_pair_summary_{timestamp}.csv"

    write_csv(summary_csv, all_scenario_rows)
    write_csv(type_pair_csv, all_type_pair_rows)

    print_terminal_summary(all_scenario_rows)
    print()
    print(f"Scenario summary CSV: {summary_csv}")
    print(f"Type-pair summary CSV: {type_pair_csv}")


if __name__ == "__main__":
    main()