from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(slots=True)
class Material:
    material_id: str
    mat_type_code: str

    width: float | None = None
    thickness: float | None = None
    length: float | None = None
    weight: float | None = None

    yard: str | None = None
    production_date: date | None = None
    pile_position: int | None = None

    category_name: str = "default"
    category_score: float = 0.0
    homogeneity_class: str | None = None

    assigned: bool = False


@dataclass(slots=True)
class OrderStep:
    order_step_id: str
    order_id: str
    prod_step_type_code: str

    required_width_min: float | None = None
    required_width_max: float | None = None
    required_thickness_min: float | None = None
    required_thickness_max: float | None = None
    required_length_min: float | None = None
    required_length_max: float | None = None

    due_date: date | None = None

    category_name: str = "default"
    category_score: float = 0.0
    required_homogeneity_class: str | None = None

    assigned: bool = False


@dataclass(slots=True, frozen=True)
class MatchKey:
    material_id: str
    order_step_id: str


@dataclass(slots=True)
class FeasibleMatch:
    material_id: str
    order_step_id: str

    allocatable: bool = True
    rule_set_name: str | None = None
    check_rule_set_names: list[str] = field(default_factory=list)
    failed_rule_names: list[str] = field(default_factory=list)

    score: float | None = None

    assignment_cost: float = 0.0
    due_date_score: float = 0.0
    production_date_score: float = 0.0
    pile_penalty: float = 0.0
    order_category_score: float = 0.0
    material_category_score: float = 0.0

    debug_messages: list[str] = field(default_factory=list)

    @property
    def key(self) -> MatchKey:
        return MatchKey(self.material_id, self.order_step_id)


@dataclass(slots=True)
class Assignment:
    material_id: str
    order_step_id: str

    score: float
    source_rule_set_name: str | None = None


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


@dataclass(slots=True)
class Scenario:
    scenario_id: str
    materials: list[Material]
    order_steps: list[OrderStep]
    today: date | None = None


@dataclass(slots=True)
class StepResult:
    reward: float
    terminated: bool
    truncated: bool
    info: dict