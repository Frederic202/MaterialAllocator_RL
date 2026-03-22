from __future__ import annotations

from ma_rl.domain.config import HardRuleConfig
from ma_rl.domain.models import Material, OrderStep


def is_order_category_allowed(order_step: OrderStep, config: HardRuleConfig) -> bool:
    return order_step.category_name not in config.forbidden_order_categories


def is_type_pair_allowed(
    material: Material,
    order_step: OrderStep,
    config: HardRuleConfig,
) -> bool:
    if not config.allowed_type_pairs:
        return True
    return (material.mat_type_code, order_step.prod_step_type_code) in config.allowed_type_pairs


def _value_matches_range(
    value: float | None,
    lower: float | None,
    upper: float | None,
    allow_missing_dimensions: bool,
) -> bool:
    if value is None or lower is None or upper is None:
        return allow_missing_dimensions

    return lower <= value <= upper


def dimensions_match(
    material: Material,
    order_step: OrderStep,
    config: HardRuleConfig,
) -> bool:
    if not config.enforce_dimension_rules:
        return True

    width_ok = _value_matches_range(
        material.width,
        order_step.required_width_min,
        order_step.required_width_max,
        config.allow_missing_dimensions,
    )
    thickness_ok = _value_matches_range(
        material.thickness,
        order_step.required_thickness_min,
        order_step.required_thickness_max,
        config.allow_missing_dimensions,
    )
    length_ok = _value_matches_range(
        material.length,
        order_step.required_length_min,
        order_step.required_length_max,
        config.allow_missing_dimensions,
    )

    return width_ok and thickness_ok and length_ok


def evaluate_hard_rules(
    material: Material,
    order_step: OrderStep,
    config: HardRuleConfig,
) -> tuple[bool, list[str]]:
    failed_rules: list[str] = []

    if not is_order_category_allowed(order_step, config):
        failed_rules.append("order_category_forbidden")

    if not is_type_pair_allowed(material, order_step, config):
        failed_rules.append("type_pair_not_allowed")

    if not dimensions_match(material, order_step, config):
        failed_rules.append("dimension_mismatch")

    return len(failed_rules) == 0, failed_rules