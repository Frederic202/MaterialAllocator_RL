from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sb3_contrib import QRDQN

from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx
from ma_rl.data import compute_dataset_shape_config, load_scenarios_from_folder
from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    ScoreWeights,
)
from ma_rl.rl.evaluation_utils import evaluate_model_on_scenarios


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

    model_path = project_root / "data" / "outputs" / "models" / "qrdqn_multiscenario_v1.zip"
    model = QRDQN.load(str(model_path))

    rows = evaluate_model_on_scenarios(
        model=model,
        scenarios=test_scenarios,
        shape_config=shape_config,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
        env_config=env_config,
        penalty_threshold=None,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"qrdqn_testset_{timestamp}.csv"
    xlsx_path = output_dir / f"qrdqn_testset_{timestamp}.xlsx"

    write_excel_friendly_csv(csv_path, rows)
    write_simple_xlsx(xlsx_path, rows, sheet_name="QRDQN_Testset")

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()