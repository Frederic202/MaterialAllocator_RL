from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx
from ma_rl.data import compute_dataset_shape_config, load_scenarios_from_folder
from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
)
from ma_rl.envs import MultiScenarioMaterialAllocatorEnv
from ma_rl.rl.evaluation_utils import (
    evaluate_greedy_on_scenarios,
    evaluate_model_on_scenarios,
    summarize_eval_rows,
)
from ma_rl.rl.validation_callback import ValidationEvalCallback


TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 10_000
SEEDS = [1, 2, 3, 4, 5]


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
        rule_set_name="dqn_multiscenario_v2",
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


def plot_multiseed_test_scores(rows: list[dict], path: Path) -> None:
    dqn_rows = [row for row in rows if row["algorithm"] == "dqn"]
    greedy_rows = [row for row in rows if row["algorithm"] == "greedy"]

    dqn_by_seed: dict[int, list[float]] = {}
    for row in dqn_rows:
        dqn_by_seed.setdefault(int(row["seed"]), []).append(float(row["total_score"]))

    greedy_scores = [float(row["total_score"]) for row in greedy_rows]
    greedy_mean = sum(greedy_scores) / len(greedy_scores)

    seeds = sorted(dqn_by_seed.keys())
    dqn_means = [sum(dqn_by_seed[s]) / len(dqn_by_seed[s]) for s in seeds]

    plt.figure(figsize=(8, 4.5))
    plt.plot(seeds, dqn_means, marker="o", label="DQN mean test score")
    plt.axhline(greedy_mean, linestyle="--", label="Greedy mean test score")
    plt.xlabel("Seed")
    plt.ylabel("Mean Total Score on Test Set")
    plt.title("Multi-Seed DQN vs. Greedy")
    plt.xticks(seeds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    train_dir = project_root / "data" / "scenarios" / "generated" / "train"
    val_dir = project_root / "data" / "scenarios" / "generated" / "val"
    test_dir = project_root / "data" / "scenarios" / "generated" / "test"

    output_root = project_root / "data" / "outputs" / "multiseed_benchmark"
    output_root.mkdir(parents=True, exist_ok=True)

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

    all_test_rows: list[dict] = []

    greedy_rows = evaluate_greedy_on_scenarios(
        scenarios=test_scenarios,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        penalty_threshold=None,
    )

    for row in greedy_rows:
        row["algorithm"] = "greedy"
        row["seed"] = 0
        all_test_rows.append(row)

    for seed in SEEDS:
        print("=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        train_env = MultiScenarioMaterialAllocatorEnv(
            scenarios=train_scenarios,
            shape_config=shape_config,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
            penalty_threshold=None,
            invalid_action_penalty=-1.0,
            scenario_seed=seed,
        )
        train_env = Monitor(train_env)

        seed_output_dir = output_root / f"seed_{seed}"
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        callback = ValidationEvalCallback(
            val_scenarios=val_scenarios,
            shape_config=shape_config,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
            output_dir=seed_output_dir,
            eval_freq=EVAL_FREQ,
            penalty_threshold=None,
        )

        model = DQN(
            policy="MultiInputPolicy",
            env=train_env,
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
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            tensorboard_log=str(output_root / "tensorboard"),
            seed=seed,
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            log_interval=10,
            progress_bar=True,
            tb_log_name=f"dqn_multiseed_seed_{seed}",
        )

        final_model_path = seed_output_dir / "final_model"
        model.save(str(final_model_path))

        best_model_path = seed_output_dir / "best_model.zip"
        eval_model = DQN.load(str(best_model_path if best_model_path.exists() else final_model_path.with_suffix(".zip")))

        dqn_rows = evaluate_model_on_scenarios(
            model=eval_model,
            scenarios=test_scenarios,
            shape_config=shape_config,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
            penalty_threshold=None,
        )

        for row in dqn_rows:
            row["algorithm"] = "dqn"
            row["seed"] = seed
            all_test_rows.append(row)

    summary_rows = []
    for algorithm in ["greedy", "dqn"]:
        algo_rows = [row for row in all_test_rows if row["algorithm"] == algorithm]

        if algorithm == "greedy":
            grouped = {0: algo_rows}
        else:
            grouped = {}
            for row in algo_rows:
                grouped.setdefault(int(row["seed"]), []).append(row)

        for seed, rows in grouped.items():
            summary = summarize_eval_rows(rows)
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    **summary,
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detailed_csv = output_root / f"multiseed_test_detailed_{timestamp}.csv"
    detailed_xlsx = output_root / f"multiseed_test_detailed_{timestamp}.xlsx"
    summary_csv = output_root / f"multiseed_test_summary_{timestamp}.csv"
    summary_xlsx = output_root / f"multiseed_test_summary_{timestamp}.xlsx"
    plot_path = output_root / f"multiseed_test_scores_{timestamp}.png"

    write_excel_friendly_csv(detailed_csv, all_test_rows)
    write_simple_xlsx(detailed_xlsx, all_test_rows, sheet_name="DetailedTestResults")
    write_excel_friendly_csv(summary_csv, summary_rows)
    write_simple_xlsx(summary_xlsx, summary_rows, sheet_name="Summary")
    plot_multiseed_test_scores(all_test_rows, plot_path)

    print(f"Saved: {detailed_csv}")
    print(f"Saved: {detailed_xlsx}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_xlsx}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()