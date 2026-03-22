from datetime import date

from ma_rl.domain import (
    Assignment,
    AssignmentSet,
    Material,
    OrderStep,
    Scenario,
    ScoreWeights,
    calculate_assignment_set_score,
    apply_assignment_set_score,
)


def test_assignment_set_scoring_with_completion_bonus_and_unassigned_penalty():
    scenario = Scenario(
        scenario_id="demo",
        today=date(2026, 3, 21),
        materials=[
            Material(material_id="M1", mat_type_code="RW", homogeneity_class="H1"),
            Material(material_id="M2", mat_type_code="RW", homogeneity_class="H1"),
        ],
        order_steps=[
            OrderStep(order_step_id="S1", order_id="O1", prod_step_type_code="CC"),
            OrderStep(order_step_id="S2", order_id="O1", prod_step_type_code="CC"),
            OrderStep(order_step_id="S3", order_id="O2", prod_step_type_code="CC"),
        ],
    )

    assignment_set = AssignmentSet(
        assignments=[
            Assignment(material_id="M1", order_step_id="S1", score=5.0),
            Assignment(material_id="M2", order_step_id="S2", score=4.0),
        ]
    )

    weights = ScoreWeights(
        order_completion_bonus=10.0,
        homogeneity_penalty=5.0,
        unassigned_order_step_penalty=2.0,
    )

    breakdown = calculate_assignment_set_score(
        assignment_set=assignment_set,
        scenario=scenario,
        weights=weights,
    )

    assert breakdown.total_match_score == 9.0
    assert breakdown.total_bonus == 10.0
    assert breakdown.total_penalty == 2.0
    assert breakdown.total_score == 17.0
    assert breakdown.completed_order_ids == {"O1"}
    assert breakdown.unassigned_order_step_ids == {"S3"}


def test_assignment_set_scoring_with_homogeneity_penalty():
    scenario = Scenario(
        scenario_id="demo",
        today=date(2026, 3, 21),
        materials=[
            Material(material_id="M1", mat_type_code="RW", homogeneity_class="H1"),
            Material(material_id="M2", mat_type_code="RW", homogeneity_class="H2"),
        ],
        order_steps=[
            OrderStep(order_step_id="S1", order_id="O1", prod_step_type_code="CC"),
            OrderStep(order_step_id="S2", order_id="O1", prod_step_type_code="CC"),
        ],
    )

    assignment_set = AssignmentSet(
        assignments=[
            Assignment(material_id="M1", order_step_id="S1", score=5.0),
            Assignment(material_id="M2", order_step_id="S2", score=4.0),
        ]
    )

    weights = ScoreWeights(
        order_completion_bonus=10.0,
        homogeneity_penalty=5.0,
        unassigned_order_step_penalty=2.0,
    )

    breakdown = calculate_assignment_set_score(
        assignment_set=assignment_set,
        scenario=scenario,
        weights=weights,
    )

    assert breakdown.total_match_score == 9.0
    assert breakdown.total_bonus == 10.0
    assert breakdown.total_penalty == 5.0
    assert breakdown.total_score == 14.0
    assert breakdown.completed_order_ids == {"O1"}
    assert breakdown.inconsistent_homogeneity_order_ids == {"O1"}


def test_apply_assignment_set_score_updates_assignment_set():
    scenario = Scenario(
        scenario_id="demo",
        today=date(2026, 3, 21),
        materials=[
            Material(material_id="M1", mat_type_code="RW", homogeneity_class="H1"),
        ],
        order_steps=[
            OrderStep(order_step_id="S1", order_id="O1", prod_step_type_code="CC"),
        ],
    )

    assignment_set = AssignmentSet(
        assignments=[
            Assignment(material_id="M1", order_step_id="S1", score=3.0),
        ]
    )

    weights = ScoreWeights(
        order_completion_bonus=10.0,
        homogeneity_penalty=5.0,
        unassigned_order_step_penalty=2.0,
    )

    updated_assignment_set = apply_assignment_set_score(
        assignment_set=assignment_set,
        scenario=scenario,
        weights=weights,
    )

    assert updated_assignment_set.total_match_score == 3.0
    assert updated_assignment_set.total_bonus == 10.0
    assert updated_assignment_set.total_penalty == 0.0
    assert updated_assignment_set.total_score == 13.0