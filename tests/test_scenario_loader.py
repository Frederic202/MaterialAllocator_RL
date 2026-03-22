from pathlib import Path

from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_json,
)


def test_load_demo_scenario():
    base_dir = Path(__file__).resolve().parents[1]

    mat_codes = load_material_type_codes(
        base_dir / "data" / "raw" / "master_data_mat_type 2.sql"
    )
    prod_step_codes = load_prod_step_type_codes(
        base_dir / "data" / "raw" / "prodsteptype 2.sql"
    )

    scenario = load_scenario_from_json(
        base_dir / "data" / "scenarios" / "demo_scenario.json",
        valid_material_type_codes=mat_codes,
        valid_prod_step_type_codes=prod_step_codes,
    )

    assert scenario.scenario_id == "demo_v1"
    assert len(scenario.materials) == 3
    assert len(scenario.order_steps) == 3
    assert scenario.materials[0].mat_type_code in mat_codes
    assert scenario.order_steps[0].prod_step_type_code in prod_step_codes