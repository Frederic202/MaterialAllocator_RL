from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from ma_rl.domain import Material, OrderStep, Scenario


def _parse_datetime_to_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value).date()


def _extract_min_max_from_spec(
    spec: dict | None,
    field_name: str,
) -> tuple[float | None, float | None]:
    if spec is None:
        return None, None

    tgt_geometry = spec.get("tgtGeometry", {}) or {}
    min_geometry = spec.get("minGeometry", {}) or {}
    max_geometry = spec.get("maxGeometry", {}) or {}

    lower = min_geometry.get(field_name)
    upper = max_geometry.get(field_name)
    target = tgt_geometry.get(field_name)

    if lower is None:
        lower = target
    if upper is None:
        upper = target

    return lower, upper


def load_scenario_from_psi_json(
    path: str | Path,
    selected_step_types: Iterable[str] | None = None,
    valid_material_type_codes: set[str] | None = None,
    valid_prod_step_type_codes: set[str] | None = None,
    max_materials: int | None = None,
    max_order_steps: int | None = None,
    only_productive_materials: bool = True,
) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    selected_step_types_set = set(selected_step_types) if selected_step_types is not None else None

    # Materials
    materials: list[Material] = []

    for raw_mat in data.get("pieceMatDto", []):
        status = raw_mat.get("status")
        if only_productive_materials and status != "PRODUCTIVE":
            continue

        material_id = raw_mat["matId"]["businessCode"]
        mat_type_code = raw_mat["matTypeId"]["code"]

        if (
            valid_material_type_codes is not None
            and mat_type_code not in valid_material_type_codes
        ):
            continue

        geometry = raw_mat.get("pieceGeometry", {}) or {}
        grade = (raw_mat.get("gradeSpec", {}) or {}).get("grade")
        weight = (raw_mat.get("pieceWeight", {}) or {}).get("valueKg")
        relocation_time = (raw_mat.get("location", {}) or {}).get("relocationTime")

        materials.append(
            Material(
                material_id=material_id,
                mat_type_code=mat_type_code,
                width=geometry.get("width"),
                thickness=geometry.get("thickness"),
                length=geometry.get("length"),
                weight=weight,
                yard="PSI_DEFAULT",
                production_date=_parse_datetime_to_date(relocation_time),
                pile_position=1,
                category_name="default",
                category_score=0.0,
                homogeneity_class=grade,
            )
        )

        if max_materials is not None and len(materials) >= max_materials:
            break

    # OrderSteps
    order_steps: list[OrderStep] = []
    all_due_dates: list[date] = []

    for prod_order in data.get("prodOrders", []):
        order_main = prod_order.get("mainData", {}) or {}
        order_id = (order_main.get("id", {}) or {}).get("businessCode")
        due_date = _parse_datetime_to_date(order_main.get("tgtDateEnd"))
        order_grade = ((order_main.get("matSpecIn", {}) or {}).get("gradeSpec", {}) or {}).get("grade")

        if due_date is not None:
            all_due_dates.append(due_date)

        for step in prod_order.get("steps", []):
            step_main = step.get("mainData", {}) or {}
            step_code = step_main.get("stepCode")
            step_type_code = ((step_main.get("type", {}) or {}).get("code"))

            if step_type_code is None:
                continue

            if (
                valid_prod_step_type_codes is not None
                and step_type_code not in valid_prod_step_type_codes
            ):
                continue

            if (
                selected_step_types_set is not None
                and step_type_code not in selected_step_types_set
            ):
                continue

            input_spec = step_main.get("matSpecIn", {}) or {}

            width_min, width_max = _extract_min_max_from_spec(input_spec, "width")
            thickness_min, thickness_max = _extract_min_max_from_spec(input_spec, "thickness")
            length_min, length_max = _extract_min_max_from_spec(input_spec, "length")

            order_steps.append(
                OrderStep(
                    order_step_id=f"{order_id}:{step_code}",
                    order_id=order_id,
                    prod_step_type_code=step_type_code,
                    required_width_min=width_min,
                    required_width_max=width_max,
                    required_thickness_min=thickness_min,
                    required_thickness_max=thickness_max,
                    required_length_min=length_min,
                    required_length_max=length_max,
                    due_date=due_date,
                    category_name="default",
                    category_score=0.0,
                    required_homogeneity_class=order_grade,
                )
            )

            if max_order_steps is not None and len(order_steps) >= max_order_steps:
                break

        if max_order_steps is not None and len(order_steps) >= max_order_steps:
            break

    scenario_today = min(all_due_dates) if all_due_dates else None

    return Scenario(
        scenario_id="psi_combined_scenario_v1",
        today=scenario_today,
        materials=materials,
        order_steps=order_steps,
    )