from __future__ import annotations

import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from ma_rl.analysis.metrics import BenchmarkRow, benchmark_row_to_dict
from ma_rl.baselines import solve_greedy
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
    apply_assignment_set_score,
)
from ma_rl.envs import MaterialAllocatorEnv
from ma_rl.matching import generate_feasible_matches


TOTAL_TIMESTEPS = 50_000
SEEDS = [1, 2, 3, 4, 5]


def build_common_objects():
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
        rule_set_name="psi_dqn_v1",
        include_non_allocatable_debug_matches=False,
    )

    env_config = EnvConfig(max_steps_per_episode=500)

    return project_root, scenario, hard_rule_config, score_weights, feasible_match_config, env_config


def build_env(
    scenario,
    hard_rule_config,
    score_weights,
    feasible_match_config,
    env_config,
):
    env = MaterialAllocatorEnv(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        env_config=env_config,
        penalty_threshold=0.0,
        invalid_action_penalty=-1.0,
    )
    return Monitor(env)


def evaluate_dqn(model, env) -> dict:
    obs, info = env.reset()
    done = False
    truncated = False
    cumulative_reward = 0.0
    invalid_actions = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        cumulative_reward += reward

        if info.get("invalid_action", False):
            invalid_actions += 1

    return {
        "assignments_selected": info["assignments_selected"],
        "total_match_score": info["total_match_score"],
        "total_penalty": info["total_penalty"],
        "total_bonus": info["total_bonus"],
        "total_score": info["total_score"],
        "cumulative_reward": cumulative_reward,
        "invalid_actions": invalid_actions,
        "remaining_valid_actions": int(np.sum(info["action_mask"])),
    }


def evaluate_greedy(
    scenario,
    hard_rule_config,
    score_weights,
    feasible_match_config,
) -> dict:
    feasible_matches = generate_feasible_matches(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
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

    return {
        "assignments_selected": len(result.assignment_set.assignments),
        "total_match_score": result.assignment_set.total_match_score,
        "total_penalty": result.assignment_set.total_penalty,
        "total_bonus": result.assignment_set.total_bonus,
        "total_score": result.assignment_set.total_score,
        "cumulative_reward": result.assignment_set.total_score,
        "invalid_actions": 0,
        "remaining_valid_actions": 0,
    }


def write_rows_to_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(benchmark_row_to_dict(rows[0]).keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(benchmark_row_to_dict(row))


def write_summary_csv(path: Path, rows: list[BenchmarkRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault(row.algorithm, []).append(row)

    summary_rows: list[dict] = []

    metrics = [
        "assignments_selected",
        "total_match_score",
        "total_penalty",
        "total_bonus",
        "total_score",
        "cumulative_reward",
        "invalid_actions",
    ]

    for algorithm, algo_rows in grouped.items():
        summary = {
            "algorithm": algorithm,
            "runs": len(algo_rows),
        }

        for metric in metrics:
            values = [float(getattr(r, metric)) for r in algo_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)

        summary_rows.append(summary)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main() -> None:
    (
        project_root,
        scenario,
        hard_rule_config,
        score_weights,
        feasible_match_config,
        env_config,
    ) = build_common_objects()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = project_root / "data" / "outputs" / "models" / f"multi_seed_{timestamp}"
    benchmark_dir = project_root / "data" / "outputs" / "benchmarks"
    model_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[BenchmarkRow] = []

    greedy_metrics = evaluate_greedy(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    for seed in SEEDS:
        print("=" * 70)
        print(f"RUNNING SEED {seed}")
        print("=" * 70)

        train_env = build_env(
            scenario=scenario,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
        )

        model = DQN(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1_000,
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
            seed=seed,
        )

        start_time = time.perf_counter()
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,
            progress_bar=True,
        )
        runtime_seconds = time.perf_counter() - start_time

        model_path = model_dir / f"dqn_seed_{seed}"
        model.save(str(model_path))

        eval_env = build_env(
            scenario=scenario,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
        )

        dqn_metrics = evaluate_dqn(model, eval_env)

        all_rows.append(
            BenchmarkRow(
                algorithm="greedy",
                seed=seed,
                scenario_id=scenario.scenario_id,
                assignments_selected=greedy_metrics["assignments_selected"],
                total_match_score=greedy_metrics["total_match_score"],
                total_penalty=greedy_metrics["total_penalty"],
                total_bonus=greedy_metrics["total_bonus"],
                total_score=greedy_metrics["total_score"],
                cumulative_reward=greedy_metrics["cumulative_reward"],
                invalid_actions=greedy_metrics["invalid_actions"],
                remaining_valid_actions=greedy_metrics["remaining_valid_actions"],
                training_timesteps=0,
                note="baseline repeated per seed for easier comparison",
            )
        )

        all_rows.append(
            BenchmarkRow(
                algorithm="dqn",
                seed=seed,
                scenario_id=scenario.scenario_id,
                assignments_selected=dqn_metrics["assignments_selected"],
                total_match_score=dqn_metrics["total_match_score"],
                total_penalty=dqn_metrics["total_penalty"],
                total_bonus=dqn_metrics["total_bonus"],
                total_score=dqn_metrics["total_score"],
                cumulative_reward=dqn_metrics["cumulative_reward"],
                invalid_actions=dqn_metrics["invalid_actions"],
                remaining_valid_actions=dqn_metrics["remaining_valid_actions"],
                training_timesteps=TOTAL_TIMESTEPS,
                note=f"runtime_seconds={runtime_seconds:.2f}; model={model_path.name}.zip",
            )
        )

    results_csv = benchmark_dir / f"benchmark_results_{timestamp}.csv"
    summary_csv = benchmark_dir / f"benchmark_summary_{timestamp}.csv"

    write_rows_to_csv(results_csv, all_rows)
    write_summary_csv(summary_csv, all_rows)

    print("\n" + "=" * 70)
    print("MULTI-SEED BENCHMARK FINISHED")
    print("=" * 70)
    print(f"Detailed results: {results_csv}")
    print(f"Summary results:  {summary_csv}")


if __name__ == "__main__":
    main()