from datetime import date

from ma_rl.domain import (
    Material,
    OrderStep,
    Scenario,
    HardRuleConfig,
    ScoreWeights,
    FeasibleMatchConfig,
)
from ma_rl.matching import generate_feasible_matches


def test_generate_feasible_matches_smoke():
    scenario = Scenario(
        scenario_id="demo",
        today=date(2026, 3, 21),
        materials=[
            Material(
                material_id="M1",
                mat_type_code="RW",
                width=5.0,
                thickness=2.0,
                length=10.0,
                yard="Y1",
                production_date=date(2026, 3, 1),
                pile_position=1,
                category_score=2.0,
            )
        ],
        order_steps=[
            OrderStep(
                order_step_id="S1",
                order_id="O1",
                prod_step_type_code="CC",
                required_width_min=4.0,
                required_width_max=6.0,
                required_thickness_min=1.0,
                required_thickness_max=3.0,
                required_length_min=9.0,
                required_length_max=11.0,
                due_date=date(2026, 3, 25),
                category_score=3.0,
            )
        ],
    )

    hard_rule_config = HardRuleConfig(
        allowed_type_pairs={("RW", "CC")}
    )
    score_weights = ScoreWeights()
    feasible_match_config = FeasibleMatchConfig()

    matches = generate_feasible_matches(
        scenario=scenario,
        hard_rule_config=hard_rule_config,
        score_weights=score_weights,
        feasible_match_config=feasible_match_config,
    )

    assert len(matches) == 1
    assert matches[0].material_id == "M1"
    assert matches[0].order_step_id == "S1"
    assert matches[0].score is not None