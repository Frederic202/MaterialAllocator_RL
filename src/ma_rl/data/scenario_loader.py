from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ma_rl.domain import Material, OrderStep, Scenario


def _parse_date(value: str | None) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(value)


def load_scenario_from_json(
    path: str | Path,
    valid_material_type_codes: set[str] | None = None,
    valid_prod_step_type_codes: set[str] | None = None,
) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    scenario_id = data["scenario_id"]
    today = _parse_date(data.get("today"))

    materials: list[Material] = []
    for raw in data["materials"]:
        mat_type_code = raw["mat_type_code"]

        if (
            valid_material_type_codes is not None
            and mat_type_code not in valid_material_type_codes
        ):
            raise ValueError(
                f"Unknown material type code '{mat_type_code}' "
                f"in material '{raw['material_id']}'."
            )

        materials.append(
            Material(
                material_id=raw["material_id"],
                mat_type_code=mat_type_code,
                width=raw.get("width"),
                thickness=raw.get("thickness"),
                length=raw.get("length"),
                weight=raw.get("weight"),
                yard=raw.get("yard"),
                production_date=_parse_date(raw.get("production_date")),
                pile_position=raw.get("pile_position"),
                category_name=raw.get("category_name", "default"),
                category_score=raw.get("category_score", 0.0),
                homogeneity_class=raw.get("homogeneity_class"),
            )
        )

    order_steps: list[OrderStep] = []
    for raw in data["order_steps"]:
        prod_step_type_code = raw["prod_step_type_code"]

        if (
            valid_prod_step_type_codes is not None
            and prod_step_type_code not in valid_prod_step_type_codes
        ):
            raise ValueError(
                f"Unknown prod step type code '{prod_step_type_code}' "
                f"in order step '{raw['order_step_id']}'."
            )

        order_steps.append(
            OrderStep(
                order_step_id=raw["order_step_id"],
                order_id=raw["order_id"],
                prod_step_type_code=prod_step_type_code,
                required_width_min=raw.get("required_width_min"),
                required_width_max=raw.get("required_width_max"),
                required_thickness_min=raw.get("required_thickness_min"),
                required_thickness_max=raw.get("required_thickness_max"),
                required_length_min=raw.get("required_length_min"),
                required_length_max=raw.get("required_length_max"),
                due_date=_parse_date(raw.get("due_date")),
                category_name=raw.get("category_name", "default"),
                category_score=raw.get("category_score", 0.0),
                required_homogeneity_class=raw.get("required_homogeneity_class"),
            )
        )

    return Scenario(
        scenario_id=scenario_id,
        today=today,
        materials=materials,
        order_steps=order_steps,
    )