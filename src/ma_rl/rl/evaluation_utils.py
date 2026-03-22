from __future__ import annotations

from statistics import mean

from stable_baselines3 import DQN

from ma_rl.baselines import solve_greedy
from ma_rl.domain import (
    DatasetShapeConfig,
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    Scenario,
    ScoreWeights,
    apply_assignment_set_score,
)
from ma_rl.envs import MultiScenarioMaterialAllocatorEnv
from ma_rl.matching import generate_feasible_matches
from ma_rl.rl.masked_action_selection import select_masked_greedy_action


def evaluate_model_on_scenarios(
    model: DQN,
    scenarios: list[Scenario],
    shape_config: DatasetShapeConfig,
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
    env_config: EnvConfig,
    penalty_threshold: float | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for scenario in scenarios:
        env = MultiScenarioMaterialAllocatorEnv(
            scenarios=[scenario],
            shape_config=shape_config,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
            env_config=env_config,
            penalty_threshold=penalty_threshold,
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
                "invalid_action_ratio": invalid_actions / max(1, info["step_count"]),
            }
        )

    return rows


def evaluate_greedy_on_scenarios(
    scenarios: list[Scenario],
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
    penalty_threshold: float | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for scenario in scenarios:
        feasible_matches = generate_feasible_matches(
            scenario=scenario,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
        )

        result = solve_greedy(
            feasible_matches=feasible_matches,
            penalty_threshold=penalty_threshold,
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
                "num_actions": len(
                    [
                        m for m in feasible_matches
                        if m.allocatable
                        and m.score is not None
                        and (penalty_threshold is None or m.score >= penalty_threshold)
                    ]
                ),
                "episode_step_limit": None,
                "assignments_selected": len(result.assignment_set.assignments),
                "total_match_score": result.assignment_set.total_match_score,
                "total_penalty": result.assignment_set.total_penalty,
                "total_bonus": result.assignment_set.total_bonus,
                "total_score": result.assignment_set.total_score,
                "cumulative_reward": result.assignment_set.total_score,
                "invalid_actions": 0,
                "final_valid_actions": 0,
                "terminated": True,
                "truncated": False,
                "invalid_action_ratio": 0.0,
            }
        )

    return rows


def summarize_eval_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("rows must not be empty")

    return {
        "num_scenarios": len(rows),
        "mean_assignments_selected": mean(float(r["assignments_selected"]) for r in rows),
        "mean_total_match_score": mean(float(r["total_match_score"]) for r in rows),
        "mean_total_penalty": mean(float(r["total_penalty"]) for r in rows),
        "mean_total_bonus": mean(float(r["total_bonus"]) for r in rows),
        "mean_total_score": mean(float(r["total_score"]) for r in rows),
        "mean_cumulative_reward": mean(float(r["cumulative_reward"]) for r in rows),
        "mean_invalid_actions": mean(float(r["invalid_actions"]) for r in rows),
        "mean_invalid_action_ratio": mean(float(r["invalid_action_ratio"]) for r in rows),
    }