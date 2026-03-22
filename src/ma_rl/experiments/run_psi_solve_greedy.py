from __future__ import annotations

from pathlib import Path

from ma_rl.baselines import solve_greedy
from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_psi_json,
)
from ma_rl.matching import generate_feasible_matches
from ma_rl.domain import (
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
    apply_assignment_set_score,
    build_default_allowed_type_pairs_for_psi_v1,
)
from ma_rl.analysis import (
    print_feasible_match_type_pair_summary,
    print_input_type_summary,
)

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    mat_type_sql = project_root / "data" / "raw" / "master_data_mat_type 2.sql"
    prod_step_sql = project_root / "data" / "raw" / "prodsteptype 2.sql"
    psi_json = project_root / "data" / "raw" / "data_unallocated_orders_fixed_outerDiameter.json"

    valid_material_type_codes = load_material_type_codes(mat_type_sql)
    valid_prod_step_type_codes = load_prod_step_type_codes(prod_step_sql)

    scenario = load_scenario_from_psi_json(
        path=psi_json,
        selected_step_types={"HR"},
        valid_material_type_codes=valid_material_type_codes,
        valid_prod_step_type_codes=valid_prod_step_type_codes,
        max_materials=300,
        max_order_steps=50,
        only_productive_materials=True,
    )

    print_input_type_summary(scenario)

    hard_rule_config = HardRuleConfig(
        allowed_type_pairs=build_default_allowed_type_pairs_for_psi_v1(),
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
        rule_set_name="psi_combined_v1",
        include_non_allocatable_debug_matches=True,
    )

    feasible_matches = generate_feasible_matches(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    print_feasible_match_type_pair_summary(
        feasible_matches=feasible_matches,
        scenario=scenario,
        penalty_threshold=0.0,
    )

    result = solve_greedy(
        feasible_matches=feasible_matches,
        penalty_threshold=0.0,
    )

    apply_assignment_set_score(
        assignment_set=result.assignment_set,
        scenario=scenario,
        weights=score_weights,
    )

    print("=" * 70)
    print("PSI COMBINED SCENARIO - SOLVE GREEDY")
    print("=" * 70)
    print(f"Scenario ID: {scenario.scenario_id}")
    print(f"Materials: {len(scenario.materials)}")
    print(f"OrderSteps: {len(scenario.order_steps)}")
    print(f"Feasible matches: {len(feasible_matches)}")
    print(f"Greedy candidates: {result.filtered_count}")
    print(f"Assignments selected: {len(result.assignment_set.assignments)}")
    print(f"Total match score: {result.assignment_set.total_match_score:.4f}")
    print(f"Total penalty: {result.assignment_set.total_penalty:.4f}")
    print(f"Total bonus: {result.assignment_set.total_bonus:.4f}")
    print(f"Final total score: {result.assignment_set.total_score:.4f}")
    print()

    print("First 10 assignments:")
    for assignment in result.assignment_set.assignments[:10]:
        print(
            f"  {assignment.material_id} -> {assignment.order_step_id} "
            f"(score={assignment.score:.4f})"
        )


if __name__ == "__main__":
    main()