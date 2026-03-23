from __future__ import annotations

import json
import random
from pathlib import Path

from ma_rl.domain import FeasibleMatch, Material, OrderStep, Scenario


def _material_to_dict(material: Material) -> dict:
    return {
        "material_id": material.material_id,
        "mat_type_code": material.mat_type_code,
        "width": material.width,
        "thickness": material.thickness,
        "length": material.length,
        "weight": material.weight,
        "yard": material.yard,
        "production_date": material.production_date.isoformat() if material.production_date else None,
        "pile_position": material.pile_position,
        "category_name": material.category_name,
        "category_score": material.category_score,
        "homogeneity_class": material.homogeneity_class,
    }


def _order_step_to_dict(order_step: OrderStep) -> dict:
    return {
        "order_step_id": order_step.order_step_id,
        "order_id": order_step.order_id,
        "prod_step_type_code": order_step.prod_step_type_code,
        "required_width_min": order_step.required_width_min,
        "required_width_max": order_step.required_width_max,
        "required_thickness_min": order_step.required_thickness_min,
        "required_thickness_max": order_step.required_thickness_max,
        "required_length_min": order_step.required_length_min,
        "required_length_max": order_step.required_length_max,
        "due_date": order_step.due_date.isoformat() if order_step.due_date else None,
        "category_name": order_step.category_name,
        "category_score": order_step.category_score,
        "required_homogeneity_class": order_step.required_homogeneity_class,
    }


def write_scenario_to_json(scenario: Scenario, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scenario_id": scenario.scenario_id,
        "today": scenario.today.isoformat() if scenario.today else None,
        "materials": [_material_to_dict(m) for m in scenario.materials],
        "order_steps": [_order_step_to_dict(s) for s in scenario.order_steps],
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _scenario_signature(scenario: Scenario) -> tuple[set[str], set[str]]:
    step_ids = {step.order_step_id for step in scenario.order_steps}
    material_ids = {material.material_id for material in scenario.materials}
    return step_ids, material_ids


def _is_too_similar(
    selected_step_ids: set[str],
    selected_material_ids: set[str],
    existing_scenarios: list[Scenario],
    max_step_overlap_ratio: float,
    max_material_overlap_ratio: float,
) -> bool:
    for scenario in existing_scenarios:
        existing_step_ids, existing_material_ids = _scenario_signature(scenario)

        step_overlap = _jaccard_overlap(selected_step_ids, existing_step_ids)
        material_overlap = _jaccard_overlap(selected_material_ids, existing_material_ids)

        if step_overlap > max_step_overlap_ratio:
            return True
        if material_overlap > max_material_overlap_ratio:
            return True

    return False


def sample_subscenario_from_feasible_matches(
    full_scenario: Scenario,
    feasible_matches: list[FeasibleMatch],
    scenario_id: str,
    rng: random.Random,
    target_order_steps: int = 8,
    min_matches_per_step: int = 2,
    max_matches_per_step: int = 4,
    extra_distractor_materials: int = 4,
    existing_scenarios: list[Scenario] | None = None,
    max_step_overlap_ratio: float = 0.5,
    max_material_overlap_ratio: float = 0.5,
    min_unique_assignable_order_steps: int = 8,
    penalty_threshold: float | None = None,
    max_attempts: int = 300,
) -> Scenario:
    material_by_id = {m.material_id: m for m in full_scenario.materials}
    order_step_by_id = {s.order_step_id: s for s in full_scenario.order_steps}

    valid_matches = [
        m for m in feasible_matches
        if m.allocatable
        and m.score is not None
        and (penalty_threshold is None or m.score >= penalty_threshold)
    ]

    matches_by_step: dict[str, list[FeasibleMatch]] = {}
    for match in valid_matches:
        matches_by_step.setdefault(match.order_step_id, []).append(match)

    eligible_step_ids = [
        step_id
        for step_id, matches in matches_by_step.items()
        if len(matches) >= min_matches_per_step
    ]

    if len(eligible_step_ids) < target_order_steps:
        raise ValueError(
            f"Not enough eligible order steps for sampling. "
            f"Needed {target_order_steps}, found {len(eligible_step_ids)}."
        )

    existing_scenarios = existing_scenarios or []

    last_reason = "unknown"

    for _ in range(max_attempts):
        sampled_step_ids = set(rng.sample(eligible_step_ids, target_order_steps))
        selected_material_ids: set[str] = set()

        for step_id in sampled_step_ids:
            step_matches = list(matches_by_step[step_id])
            rng.shuffle(step_matches)

            n_matches = min(len(step_matches), max_matches_per_step)
            n_matches = max(min_matches_per_step, n_matches)

            chosen_matches = step_matches[:n_matches]
            for match in chosen_matches:
                selected_material_ids.add(match.material_id)

        remaining_material_ids = [
            m.material_id
            for m in full_scenario.materials
            if m.material_id not in selected_material_ids
        ]
        rng.shuffle(remaining_material_ids)

        for material_id in remaining_material_ids[:extra_distractor_materials]:
            selected_material_ids.add(material_id)

        if _is_too_similar(
            selected_step_ids=sampled_step_ids,
            selected_material_ids=selected_material_ids,
            existing_scenarios=existing_scenarios,
            max_step_overlap_ratio=max_step_overlap_ratio,
            max_material_overlap_ratio=max_material_overlap_ratio,
        ):
            last_reason = "too_similar"
            continue

        threshold_matches = [
            match
            for match in valid_matches
            if match.order_step_id in sampled_step_ids
            and match.material_id in selected_material_ids
        ]

        unique_assignable_order_steps = {
            match.order_step_id for match in threshold_matches
        }

        if len(unique_assignable_order_steps) < min_unique_assignable_order_steps:
            last_reason = "too_few_assignable_steps"
            continue

        sampled_materials = [
            material_by_id[mid]
            for mid in selected_material_ids
            if mid in material_by_id
        ]
        sampled_order_steps = [
            order_step_by_id[sid]
            for sid in sampled_step_ids
            if sid in order_step_by_id
        ]

        return Scenario(
            scenario_id=scenario_id,
            today=full_scenario.today,
            materials=sampled_materials,
            order_steps=sampled_order_steps,
        )

    raise RuntimeError(
        f"Could not generate scenario '{scenario_id}' after {max_attempts} attempts. "
        f"Last reason: {last_reason}"
    )