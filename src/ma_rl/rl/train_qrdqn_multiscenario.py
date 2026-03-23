from __future__ import annotations

from pathlib import Path

from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor

from ma_rl.data import compute_dataset_shape_config, load_scenarios_from_folder
from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
)
from ma_rl.envs import MultiScenarioMaterialAllocatorEnv


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
        rule_set_name="qrdqn_multiscenario_v1",
        include_non_allocatable_debug_matches=False,
    )

    env_config = EnvConfig(
        max_steps_per_episode=None,
        allow_delay_action=False,
        use_dynamic_max_steps=True,
        dynamic_max_steps_factor=2,
        min_steps_per_episode=20,
    )

    return hard_rule_config, score_weights, feasible_match_config, env_config


DATASET_NAME = "generated_v2"

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    train_dir = project_root / "data" / "scenarios" / DATASET_NAME/ "train"
    val_dir = project_root / "data" / "scenarios" / DATASET_NAME / "val"
    output_model_dir = project_root / "data" / "outputs" / "models"
    output_tb_dir = project_root / "data" / "outputs" / "tensorboard"

    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_tb_dir.mkdir(parents=True, exist_ok=True)

    train_scenarios = load_scenarios_from_folder(train_dir)
    val_scenarios = load_scenarios_from_folder(val_dir)

    hard_rule_config, score_weights, feasible_match_config, env_config = build_common_configs()

    shape_config = compute_dataset_shape_config(
        scenarios=train_scenarios + val_scenarios,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        penalty_threshold=None,
    )

    env = MultiScenarioMaterialAllocatorEnv(
        scenarios=train_scenarios,
        shape_config=shape_config,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        env_config=env_config,
        penalty_threshold=None,
        invalid_action_penalty=-1.0,
        scenario_seed=42,
    )
    env = Monitor(env)

    model = QRDQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.30,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            n_quantiles=50,
            net_arch=[256, 256],
        ),
        verbose=1,
        tensorboard_log=str(output_tb_dir),
        seed=42,
    )

    model.learn(
        total_timesteps=100_000,
        log_interval=10,
        progress_bar=True,
        tb_log_name="qrdqn_multiscenario_v1",
    )

    model_path = output_model_dir / "qrdqn_multiscenario_v1"
    model.save(str(model_path))
    print(f"Saved model: {model_path}.zip")


if __name__ == "__main__":
    main()