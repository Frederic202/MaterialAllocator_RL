from __future__ import annotations

from pathlib import Path

from stable_baselines3 import DQN

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

    return project_root, scenario, hard_rule_config, score_weights, feasible_match_config


def run_greedy(scenario, hard_rule_config, score_weights, feasible_match_config):
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

    return result.assignment_set


def run_dqn(project_root, scenario, hard_rule_config, score_weights, feasible_match_config):
    model_path = project_root / "data" / "outputs" / "models" / "dqn_material_allocator_v1.zip"

    env = MaterialAllocatorEnv(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        env_config=EnvConfig(max_steps_per_episode=500),
        penalty_threshold=0.0,
        invalid_action_penalty=-1.0,
    )

    model = DQN.load(str(model_path))

    obs, info = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))

    return info


def main() -> None:
    (
        project_root,
        scenario,
        hard_rule_config,
        score_weights,
        feasible_match_config,
    ) = build_common_objects()

    greedy_assignment_set = run_greedy(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    dqn_info = run_dqn(
        project_root=project_root,
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    print("=" * 70)
    print("BENCHMARK: GREEDY vs DQN")
    print("=" * 70)
    print(
        f"Greedy  -> assignments={len(greedy_assignment_set.assignments)}, "
        f"score={greedy_assignment_set.total_score:.4f}"
    )
    print(
        f"DQN     -> assignments={dqn_info['assignments_selected']}, "
        f"score={dqn_info['total_score']:.4f}"
    )


if __name__ == "__main__":
    main()