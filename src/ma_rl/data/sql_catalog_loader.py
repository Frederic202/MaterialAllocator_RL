from __future__ import annotations

import re
from pathlib import Path


_MATTYPE_PATTERN = re.compile(
    r"INSERT INTO public\.mattype\s*\(.*?\)\s*VALUES\s*\('([^']+)'",
    re.IGNORECASE,
)

_PRODSTEP_PATTERN = re.compile(
    r"INSERT INTO public\.prodsteptype\s*\(.*?\)\s*VALUES\s*\('([^']+)'",
    re.IGNORECASE,
)


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_material_type_codes(path: str | Path) -> set[str]:
    text = _read_text(path)
    return set(_MATTYPE_PATTERN.findall(text))


def load_prod_step_type_codes(path: str | Path) -> set[str]:
    text = _read_text(path)
    return set(_PRODSTEP_PATTERN.findall(text))