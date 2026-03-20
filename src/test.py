from ma_rl.domain import Material

def test_material():
    m = Material("iron", 10, 5)
    assert m.name == "iron"
    assert m.value == 10
    assert m.weight == 5