from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(slots=True)
class Material:
    material_id: str
    mat_type_code: str

    width: float
    thickness: float
    length: float
    weight: float

    yard: str
    production_date: date
    pile_position: int

    category_name: str = "default"
    category_score: float = 0.0
    homogeneity_class: str = "default"

    assigned: bool = False


@dataclass(slots=True)
class OrderStep:
    order_step_id: str
    order_id: str
    prod_step_type_code: str

    required_width_min: float
    required_width_max: float
    required_thickness_min: float
    required_thickness_max: float
    required_length_min: float
    required_length_max: float

    due_date: date

    category_name: str = "default"
    category_score: float = 0.0
    required_homogeneity_class: Optional[str] = None

    assigned: bool = False


@dataclass(slots=True, frozen=True)
class MatchKey:
    material_id: str
    order_step_id: str


@dataclass(slots=True)
class Match:
    material_id: str
    order_step_id: str
    valid: bool

    assignment_cost: float = 0.0
    due_date_score: float = 0.0
    production_date_score: float = 0.0
    pile_penalty: float = 0.0

    local_score: float = 0.0


@dataclass(slots=True)
class Assignment:
    material_id: str
    order_step_id: str
    match_score: float
    assignment_cost: float
    due_date_score: float
    production_date_score: float
    pile_penalty: float


@dataclass(slots=True)
class AssignmentSet:
    assignments: list[Assignment] = field(default_factory=list)
    total_score: float = 0.0
    total_match_score: float = 0.0
    total_penalty: float = 0.0
    total_bonus: float = 0.0

    def add_assignment(self, assignment: Assignment) -> None:
        self.assignments.append(assignment)

    @property
    def assigned_material_ids(self) -> set[str]:
        return {a.material_id for a in self.assignments}

    @property
    def assigned_order_step_ids(self) -> set[str]:
        return {a.order_step_id for a in self.assignments}

    @property
    def assigned_order_ids(self) -> set[str]:
        # Wird später sinnvoll, wenn wir Completion-Bonus berechnen
        return set()


@dataclass(slots=True)
class Scenario:
    scenario_id: str
    materials: list[Material]
    order_steps: list[OrderStep]


@dataclass(slots=True)
class StepResult:
    reward: float
    terminated: bool
    truncated: bool
    info: dict