from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ma_rl.domain.config import ScoreWeights
from ma_rl.domain.models import AssignmentSet, Material, OrderStep, Scenario


@dataclass(slots=True)
class AssignmentSetScoreBreakdown:
    total_match_score: float
    total_penalty: float
    total_bonus: float
    total_score: float

    completed_order_ids: set[str]
    inconsistent_homogeneity_order_ids: set[str]
    unassigned_order_step_ids: set[str]


def _safe_ratio(value: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return value / denominator


def _range_center_cost(
    value: float | None,
    lower: float | None,
    upper: float | None,
) -> float:
    if value is None or lower is None or upper is None:
        return 0.0

    center = (lower + upper) / 2.0
    half_range = max((upper - lower) / 2.0, 1e-9)
    return abs(value - center) / half_range


def calculate_assignment_cost(material: Material, order_step: OrderStep) -> float:
    width_cost = _range_center_cost(
        material.width,
        order_step.required_width_min,
        order_step.required_width_max,
    )
    thickness_cost = _range_center_cost(
        material.thickness,
        order_step.required_thickness_min,
        order_step.required_thickness_max,
    )
    length_cost = _range_center_cost(
        material.length,
        order_step.required_length_min,
        order_step.required_length_max,
    )

    return width_cost + thickness_cost + length_cost


def calculate_due_date_score(order_step: OrderStep, today: date | None) -> float:
    if today is None or order_step.due_date is None:
        return 0.0

    days_until_due = (order_step.due_date - today).days

    if days_until_due <= 0:
        return 1.0
    if days_until_due >= 30:
        return 0.0

    return 1.0 - _safe_ratio(days_until_due, 30.0)


def calculate_production_date_score(material: Material, today: date | None) -> float:
    if today is None or material.production_date is None:
        return 0.0

    age_days = (today - material.production_date).days

    if age_days <= 0:
        return 0.0
    if age_days >= 30:
        return 1.0

    return _safe_ratio(age_days, 30.0)


def calculate_pile_penalty(material: Material) -> float:
    if material.pile_position is None:
        return 0.0
    return max(float(material.pile_position - 1), 0.0)


def calculate_feasible_match_score_components(
    material: Material,
    order_step: OrderStep,
    weights: ScoreWeights,
    today: date | None,
) -> dict[str, float]:
    assignment_cost = calculate_assignment_cost(material, order_step)
    due_date_score = calculate_due_date_score(order_step, today)
    production_date_score = calculate_production_date_score(material, today)
    pile_penalty = calculate_pile_penalty(material)

    order_category_score = order_step.category_score
    material_category_score = material.category_score

    total_without_categories = (
        weights.due_date_weight * due_date_score
        + weights.production_date_weight * production_date_score
        - weights.assignment_cost_weight * assignment_cost
        - weights.pile_penalty_weight * pile_penalty
    )

    total_score = (
        total_without_categories
        + weights.order_category_weight * order_category_score
        + weights.material_category_weight * material_category_score
    )

    return {
        "assignment_cost": assignment_cost,
        "due_date_score": due_date_score,
        "production_date_score": production_date_score,
        "pile_penalty": pile_penalty,
        "order_category_score": order_category_score,
        "material_category_score": material_category_score,
        "total_without_categories": total_without_categories,
        "total_score": total_score,
    }


def calculate_assignment_set_score(
    assignment_set: AssignmentSet,
    scenario: Scenario,
    weights: ScoreWeights,
) -> AssignmentSetScoreBreakdown:
    material_by_id = {material.material_id: material for material in scenario.materials}
    order_step_by_id = {step.order_step_id: step for step in scenario.order_steps}

    total_match_score = sum(assignment.score for assignment in assignment_set.assignments)

    assigned_order_step_ids = {
        assignment.order_step_id for assignment in assignment_set.assignments
    }
    unassigned_order_step_ids = {
        step.order_step_id
        for step in scenario.order_steps
        if step.order_step_id not in assigned_order_step_ids
    }

    # Order-Vollständigkeit
    all_step_ids_by_order: dict[str, set[str]] = {}
    assigned_step_ids_by_order: dict[str, set[str]] = {}

    for step in scenario.order_steps:
        all_step_ids_by_order.setdefault(step.order_id, set()).add(step.order_step_id)

    for assignment in assignment_set.assignments:
        step = order_step_by_id[assignment.order_step_id]
        assigned_step_ids_by_order.setdefault(step.order_id, set()).add(step.order_step_id)

    completed_order_ids = {
        order_id
        for order_id, all_step_ids in all_step_ids_by_order.items()
        if assigned_step_ids_by_order.get(order_id, set()) == all_step_ids
    }

    completion_bonus_total = len(completed_order_ids) * weights.order_completion_bonus

    # Homogenität pro Order
    inconsistent_homogeneity_order_ids: set[str] = set()

    material_classes_by_order: dict[str, set[str]] = {}
    for assignment in assignment_set.assignments:
        step = order_step_by_id[assignment.order_step_id]
        material = material_by_id[assignment.material_id]

        if material.homogeneity_class is None:
            continue

        material_classes_by_order.setdefault(step.order_id, set()).add(
            material.homogeneity_class
        )

    for order_id, homogeneity_classes in material_classes_by_order.items():
        if len(homogeneity_classes) > 1:
            inconsistent_homogeneity_order_ids.add(order_id)

    homogeneity_penalty_total = (
        len(inconsistent_homogeneity_order_ids) * weights.homogeneity_penalty
    )

    unassigned_penalty_total = (
        len(unassigned_order_step_ids) * weights.unassigned_order_step_penalty
    )

    total_penalty = homogeneity_penalty_total + unassigned_penalty_total
    total_bonus = completion_bonus_total
    total_score = total_match_score - total_penalty + total_bonus

    return AssignmentSetScoreBreakdown(
        total_match_score=total_match_score,
        total_penalty=total_penalty,
        total_bonus=total_bonus,
        total_score=total_score,
        completed_order_ids=completed_order_ids,
        inconsistent_homogeneity_order_ids=inconsistent_homogeneity_order_ids,
        unassigned_order_step_ids=unassigned_order_step_ids,
    )


def apply_assignment_set_score(
    assignment_set: AssignmentSet,
    scenario: Scenario,
    weights: ScoreWeights,
) -> AssignmentSet:
    breakdown = calculate_assignment_set_score(
        assignment_set=assignment_set,
        scenario=scenario,
        weights=weights,
    )

    assignment_set.total_match_score = breakdown.total_match_score
    assignment_set.total_penalty = breakdown.total_penalty
    assignment_set.total_bonus = breakdown.total_bonus
    assignment_set.total_score = breakdown.total_score

    return assignment_set