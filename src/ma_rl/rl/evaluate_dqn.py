from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_psi_json,
)
from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
)
from ma_rl.envs import MaterialAllocatorEnv


def build_env():
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
            rule_set_name="psi_dqn_v1",
            include_non_allocatable_debug_matches=False,
        ),
        env_config=EnvConfig(max_steps_per_episode=500),
        penalty_threshold=0.0,
        invalid_action_penalty=-1.0,
    )

    return env


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "data" / "outputs" / "models" / "dqn_material_allocator_v1.zip"

    env = build_env()
    model = DQN.load(str(model_path))

    obs, info = env.reset()
    done = False
    truncated = False
    cumulative_reward = 0.0
    invalid_actions = 0

    while not done and not truncated:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        cumulative_reward += reward

        if info.get("invalid_action", False):
            invalid_actions += 1

    print("=" * 70)
    print("DQN EVALUATION")
    print("=" * 70)
    print(f"Assignments selected: {info['assignments_selected']}")
    print(f"Total match score: {info['total_match_score']:.4f}")
    print(f"Total penalty: {info['total_penalty']:.4f}")
    print(f"Total bonus: {info['total_bonus']:.4f}")
    print(f"Final total score: {info['total_score']:.4f}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")
    print(f"Invalid actions: {invalid_actions}")
    print(f"Remaining valid actions: {int(np.sum(info['action_mask']))}")


if __name__ == "__main__":
    main()