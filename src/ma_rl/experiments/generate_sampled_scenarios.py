from __future__ import annotations

import random
from pathlib import Path

from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_psi_json,
    sample_subscenario_from_feasible_matches,
    write_scenario_to_json,
)
from ma_rl.domain import FeasibleMatchConfig, HardRuleConfig, ScoreWeights
from ma_rl.matching import generate_feasible_matches


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    mat_type_sql = project_root / "data" / "raw" / "master_data_mat_type 2.sql"
    prod_step_sql = project_root / "data" / "raw" / "prodsteptype 2.sql"
    psi_json = project_root / "data" / "raw" / "data_unallocated_orders_fixed_outerDiameter.json"

    valid_material_type_codes = load_material_type_codes(mat_type_sql)
    valid_prod_step_type_codes = load_prod_step_type_codes(prod_step_sql)

    full_scenario = load_scenario_from_psi_json(
        path=psi_json,
        selected_step_types={"HR"},
        valid_material_type_codes=valid_material_type_codes,
        valid_prod_step_type_codes=valid_prod_step_type_codes,
        max_materials=None,
        max_order_steps=None,
        only_productive_materials=True,
    )

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
        rule_set_name="psi_sampling_v1",
        include_non_allocatable_debug_matches=False,
    )

    feasible_matches = generate_feasible_matches(
        scenario=full_scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    output_root = project_root / "data" / "scenarios" / "generated"
    splits = {
        "train": 20,
        "val": 5,
        "test": 5,
    }

    base_seed = 42

    for split_name, count in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            rng = random.Random(base_seed + i + hash(split_name) % 10_000)

            scenario = sample_subscenario_from_feasible_matches(
                full_scenario=full_scenario,
                feasible_matches=feasible_matches,
                scenario_id=f"{split_name}_{i:03d}",
                rng=rng,
                target_order_steps=8,
                min_matches_per_step=2,
                max_matches_per_step=4,
                extra_distractor_materials=4,
            )

            write_scenario_to_json(
                scenario=scenario,
                path=split_dir / f"{scenario.scenario_id}.json",
            )

    print("Scenario generation finished.")
    print(f"Output folder: {output_root}")


if __name__ == "__main__":
    main()