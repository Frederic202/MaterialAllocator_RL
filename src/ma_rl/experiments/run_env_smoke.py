from pathlib import Path

import numpy as np

from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_psi_json,
)
from ma_rl.domain import EnvConfig, FeasibleMatchConfig, HardRuleConfig, ScoreWeights
from ma_rl.envs import MaterialAllocatorEnv


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
        max_materials=200,
        max_order_steps=30,
        only_productive_materials=True,
    )

    env = MaterialAllocatorEnv(
        scenario=scenario,
        hard_rule_config=HardRuleConfig(
            allowed_type_pairs={("BR", "HR")},
            enforce_dimension_rules=True,
            allow_missing_dimensions=True,
        ),
        score_weights=ScoreWeights(
            material_category_weight=0.0,
            order_category_weight=0.0,
            due_date_weight=1.0,
            production_date_weight=0.5,
            assignment_cost_weight=1.0,
            pile_penalty_weight=0.0,
            order_completion_bonus=10.0,
            homogeneity_penalty=5.0,
            unassigned_order_step_penalty=2.0,
        ),
        feasible_match_config=FeasibleMatchConfig(
            rule_set_name="psi_env_v1",
            include_non_allocatable_debug_matches=False,
        ),
        env_config=EnvConfig(max_steps_per_episode=500),
        penalty_threshold=0.0,
    )

    obs, info = env.reset()

    print("=" * 70)
    print("ENV SMOKE TEST")
    print("=" * 70)
    print(f"Materials: {len(scenario.materials)}")
    print(f"OrderSteps: {len(scenario.order_steps)}")
    print(f"Initial valid actions: {int(np.sum(obs['action_mask']))}")

    done = False
    truncated = False
    cumulative_reward = 0.0

    while not done and not truncated:
        valid_actions = np.where(env.get_action_mask() > 0.0)[0]
        if len(valid_actions) == 0:
            break

        action = int(valid_actions[0])
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward

    print(f"Assignments selected: {info['assignments_selected']}")
    print(f"Total score: {info['total_score']:.4f}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")


if __name__ == "__main__":
    main()