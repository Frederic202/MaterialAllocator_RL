from __future__ import annotations


def build_default_allowed_type_pairs_for_psi_v1() -> set[tuple[str, str]]:
    """
    V1-Annahme für die aktuellen PSI-Daten:
    slab-/brammenartige BR-Materialien dürfen auf slab-input-lastige Steps.
    """
    return {
        ("BR", "PXHSM"),
        ("BR", "REH"),
        ("BR", "HR"),
    }