from __future__ import annotations

from ma_rl.domain.config import FeasibleMatchConfig, HardRuleConfig, ScoreWeights
from ma_rl.domain.models import FeasibleMatch, Material, OrderStep, Scenario
from ma_rl.domain.rules import evaluate_hard_rules
from ma_rl.domain.scoring import calculate_feasible_match_score_components


def generate_feasible_matches(
    scenario: Scenario,
    hard_rule_config: HardRuleConfig,
    score_weights: ScoreWeights,
    feasible_match_config: FeasibleMatchConfig,
) -> list[FeasibleMatch]:
    feasible_matches: list[FeasibleMatch] = []

    for material in scenario.materials:
        for order_step in scenario.order_steps:
            is_valid, failed_rule_names = evaluate_hard_rules(
                material=material,
                order_step=order_step,
                config=hard_rule_config,
            )

            if not is_valid:
                if feasible_match_config.include_non_allocatable_debug_matches:
                    feasible_matches.append(
                        FeasibleMatch(
                            material_id=material.material_id,
                            order_step_id=order_step.order_step_id,
                            allocatable=False,
                            rule_set_name=feasible_match_config.rule_set_name,
                            check_rule_set_names=[feasible_match_config.rule_set_name],
                            failed_rule_names=failed_rule_names,
                            score=None,
                            debug_messages=["Hard rules failed."],
                        )
                    )
                continue

            score_components = calculate_feasible_match_score_components(
                material=material,
                order_step=order_step,
                weights=score_weights,
                today=scenario.today,
            )

            # PSI-nah: Forbidden-Kategorien erst nach der Scoring-Phase aussortieren.
            if order_step.category_name in hard_rule_config.forbidden_order_categories:
                continue

            feasible_matches.append(
                FeasibleMatch(
                    material_id=material.material_id,
                    order_step_id=order_step.order_step_id,
                    allocatable=True,
                    rule_set_name=feasible_match_config.rule_set_name,
                    check_rule_set_names=[feasible_match_config.rule_set_name],
                    failed_rule_names=[],
                    score=score_components["total_score"],
                    assignment_cost=score_components["assignment_cost"],
                    due_date_score=score_components["due_date_score"],
                    production_date_score=score_components["production_date_score"],
                    pile_penalty=score_components["pile_penalty"],
                    order_category_score=score_components["order_category_score"],
                    material_category_score=score_components["material_category_score"],
                    debug_messages=[],
                )
            )

    return feasible_matches