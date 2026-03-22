from __future__ import annotations

from pathlib import Path

from ma_rl.data import load_scenario_from_json
from ma_rl.domain import (
    DatasetShapeConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    Scenario,
    ScoreWeights,
)
from ma_rl.matching import generate_feasible_matches


def load_scenarios_from_folder(folder: str | Path) -> list[Scenario]:
    folder = Path(folder)
    scenario_files = sorted(folder.glob("*.json"))
    return [load_scenario_from_json(path) for path in scenario_files]


def compute_dataset_shape_config(
    scenarios: list[Scenario],
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
    penalty_threshold: float | None = None,
) -> DatasetShapeConfig:
    max_materials = 0
    max_order_steps = 0
    max_actions = 0

    for scenario in scenarios:
        max_materials = max(max_materials, len(scenario.materials))
        max_order_steps = max(max_order_steps, len(scenario.order_steps))

        feasible_matches = generate_feasible_matches(
            scenario=scenario,
            hard_rule_config=hard_rule_config,
            score_weights=score_weights,
            feasible_match_config=feasible_match_config,
        )

        valid_actions = [
            m for m in feasible_matches
            if m.allocatable
            and m.score is not None
            and (penalty_threshold is None or m.score >= penalty_threshold)
        ]

        max_actions = max(max_actions, len(valid_actions))

    if max_actions == 0:
        raise ValueError("No valid actions found in dataset. Check rules and threshold.")

    return DatasetShapeConfig(
        max_materials=max_materials,
        max_order_steps=max_order_steps,
        max_actions=max_actions,
    )