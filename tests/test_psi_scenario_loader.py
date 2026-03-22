from pathlib import Path

from ma_rl.data import (
    load_material_type_codes,
    load_prod_step_type_codes,
    load_scenario_from_psi_json,
)


def test_load_combined_psi_scenario():
    project_root = Path(__file__).resolve().parents[1]

    mat_codes = load_material_type_codes(
        project_root / "data" / "raw" / "master_data_mat_type 2.sql"
    )
    prod_step_codes = load_prod_step_type_codes(
        project_root / "data" / "raw" / "prodsteptype 2.sql"
    )

    scenario = load_scenario_from_psi_json(
        path=project_root / "data" / "raw" / "data_unallocated_orders_fixed_outerDiameter.json",
        selected_step_types={"HR"},
        valid_material_type_codes=mat_codes,
        valid_prod_step_type_codes=prod_step_codes,
        max_materials=50,
        max_order_steps=20,
        only_productive_materials=True,
    )

    assert scenario.scenario_id == "psi_combined_scenario_v1"
    assert len(scenario.materials) > 0
    assert len(scenario.order_steps) > 0

    assert scenario.materials[0].mat_type_code in mat_codes
    assert scenario.order_steps[0].prod_step_type_code in prod_step_codes