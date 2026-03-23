from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from ma_rl.baselines import solve_greedy
from ma_rl.data import load_scenarios_from_folder
from ma_rl.domain import (
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
    apply_assignment_set_score,
)
from ma_rl.matching import generate_feasible_matches
from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx

DATASET_NAME = "generated_v2"

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    test_dir = project_root / "data" / "scenarios" / DATASET_NAME / "test"
    output_dir = project_root / "data" / "outputs" / "testset_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios_from_folder(test_dir)

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
        rule_set_name="testset_greedy_v1",
        include_non_allocatable_debug_matches=False,
    )

    rows = []

    for scenario in scenarios:
        feasible_matches = generate_feasible_matches(
            scenario=scenario,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
        )

        result = solve_greedy(
            feasible_matches=feasible_matches,
            penalty_threshold=None,
        )

        apply_assignment_set_score(
            assignment_set=result.assignment_set,
            scenario=scenario,
            weights=score_weights,
        )

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "num_materials": len(scenario.materials),
                "num_order_steps": len(scenario.order_steps),
                "assignments_selected": len(result.assignment_set.assignments),
                "total_match_score": result.assignment_set.total_match_score,
                "total_penalty": result.assignment_set.total_penalty,
                "total_bonus": result.assignment_set.total_bonus,
                "total_score": result.assignment_set.total_score,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"greedy_testset_{timestamp}.csv"
    xlsx_path = output_dir / f"greedy_testset_{timestamp}.xlsx"

    write_excel_friendly_csv(csv_path, rows)
    write_simple_xlsx(xlsx_path, rows, sheet_name="Greedy_Testset")

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()