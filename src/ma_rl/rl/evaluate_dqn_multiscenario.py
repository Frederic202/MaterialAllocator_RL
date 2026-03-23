from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from stable_baselines3 import DQN

from ma_rl.data import (
    compute_dataset_shape_config,
    load_scenarios_from_folder,
)
from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
)
from ma_rl.envs import MultiScenarioMaterialAllocatorEnv
from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx
from ma_rl.rl.masked_action_selection import select_masked_greedy_action


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
        rule_set_name="dqn_multiscenario_v1",
        include_non_allocatable_debug_matches=False,
    )

    env_config = EnvConfig(
        max_steps_per_episode=None,
        use_dynamic_max_steps=True,
        dynamic_max_steps_factor=2,
        min_steps_per_episode=20,
    )

    return hard_rule_config, score_weights, feasible_match_config, env_config

DATASET_NAME = "generated_v2"

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_dir = project_root / "data" / "scenarios" / DATASET_NAME / "train"
    val_dir = project_root / "data" / "scenarios" / DATASET_NAME / "val"
    test_dir = project_root / "data" / "scenarios" / DATASET_NAME / "test"

    output_dir = project_root / "data" / "outputs" / "testset_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_scenarios = load_scenarios_from_folder(train_dir)
    val_scenarios = load_scenarios_from_folder(val_dir)
    test_scenarios = load_scenarios_from_folder(test_dir)

    hard_rule_config, score_weights, feasible_match_config, env_config = build_common_configs()

    shape_config = compute_dataset_shape_config(
        scenarios=train_scenarios + val_scenarios + test_scenarios,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        penalty_threshold=None,
    )

    model_path = project_root / "data" / "outputs" / "models" / "dqn_multiscenario_v1.zip"
    model = DQN.load(str(model_path))

    rows = []

    for scenario in test_scenarios:
        env = MultiScenarioMaterialAllocatorEnv(
            scenarios=[scenario],
            shape_config=shape_config,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
            penalty_threshold=None,
            invalid_action_penalty=-1.0,
            scenario_seed=42,
        )

        obs, info = env.reset()
        done = False
        truncated = False
        cumulative_reward = 0.0
        invalid_actions = 0

        while not done and not truncated:
            action = select_masked_greedy_action(
                model=model,
                obs=obs,
                action_mask=info["action_mask"],
            )
            obs, reward, done, truncated, info = env.step(int(action))
            cumulative_reward += reward
            if info.get("invalid_action", False):
                invalid_actions += 1

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "num_materials": info["num_materials"],
                "num_order_steps": info["num_order_steps"],
                "num_actions": info["num_actions"],
                "episode_step_limit": info["episode_step_limit"],
                "assignments_selected": info["assignments_selected"],
                "total_match_score": info["total_match_score"],
                "total_penalty": info["total_penalty"],
                "total_bonus": info["total_bonus"],
                "total_score": info["total_score"],
                "cumulative_reward": cumulative_reward,
                "invalid_actions": invalid_actions,
                "final_valid_actions": int(info["action_mask"].sum()),
                "terminated": done,
                "truncated": truncated,
                "invalid_action_ratio": (
                        invalid_actions / max(1, info["step_count"])
                ),
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"dqn_testset_{timestamp}.csv"
    xlsx_path = output_dir / f"dqn_testset_{timestamp}.xlsx"

    write_excel_friendly_csv(csv_path, rows)
    write_simple_xlsx(xlsx_path, rows, sheet_name="DQN_Testset")

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()