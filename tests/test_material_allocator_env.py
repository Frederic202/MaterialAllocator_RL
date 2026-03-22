from datetime import date

import numpy as np

from ma_rl.domain import (
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    Material,
    OrderStep,
    Scenario,
    ScoreWeights,
)
from ma_rl.envs import MaterialAllocatorEnv


def test_material_allocator_env_smoke():
    scenario = Scenario(
        scenario_id="env_demo",
        today=date(2026, 3, 21),
        materials=[
            Material(
                material_id="M1",
                mat_type_code="BR",
                width=1300.0,
                thickness=220.0,
                length=12400.0,
                production_date=date(2026, 3, 1),
                pile_position=1,
                homogeneity_class="G1",
            ),
            Material(
                material_id="M2",
                mat_type_code="BR",
                width=1320.0,
                thickness=225.0,
                length=12400.0,
                production_date=date(2026, 2, 25),
                pile_position=1,
                homogeneity_class="G1",
            ),
        ],
        order_steps=[
            OrderStep(
                order_step_id="O1:S1",
                order_id="O1",
                prod_step_type_code="HR",
                required_width_min=1248.0,
                required_width_max=1368.0,
                required_thickness_min=210.0,
                required_thickness_max=230.0,
                required_length_min=12400.0,
                required_length_max=12400.0,
                due_date=date(2026, 3, 25),
                required_homogeneity_class="G1",
            ),
            OrderStep(
                order_step_id="O2:S1",
                order_id="O2",
                prod_step_type_code="HR",
                required_width_min=1248.0,
                required_width_max=1368.0,
                required_thickness_min=210.0,
                required_thickness_max=230.0,
                required_length_min=12400.0,
                required_length_max=12400.0,
                due_date=date(2026, 3, 28),
                required_homogeneity_class="G1",
            ),
        ],
    )

    env = MaterialAllocatorEnv(
        scenario=scenario,
        hard_rule_config=HardRuleConfig(
            allowed_type_pairs={("BR", "HR")},
            enforce_dimension_rules=True,
            allow_missing_dimensions=True,
        ),
        score_weights=ScoreWeights(),
        feasible_match_config=FeasibleMatchConfig(),
        env_config=EnvConfig(max_steps_per_episode=10),
        penalty_threshold=0.0,
    )

    obs, info = env.reset()

    assert "action_mask" in obs
    assert "candidate_scores" in obs
    assert "current_metrics" in obs
    assert np.sum(obs["action_mask"]) > 0

    done = False
    truncated = False

    while not done and not truncated:
        valid_actions = np.where(env.get_action_mask() > 0.0)[0]
        assert len(valid_actions) > 0

        action = int(valid_actions[0])
        obs, reward, done, truncated, info = env.step(action)

    assert info["assignments_selected"] > 0