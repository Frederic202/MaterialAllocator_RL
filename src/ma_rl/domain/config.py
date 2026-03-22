from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HardRuleConfig:
    allowed_type_pairs: set[tuple[str, str]] = field(default_factory=set)
    forbidden_order_categories: set[str] = field(default_factory=lambda: {"Forbidden"})

    enforce_dimension_rules: bool = True
    allow_missing_dimensions: bool = True


@dataclass(slots=True)
class ScoreWeights:
    material_category_weight: float = 1.0
    order_category_weight: float = 1.0
    due_date_weight: float = 1.0
    production_date_weight: float = 1.0

    assignment_cost_weight: float = 1.0
    pile_penalty_weight: float = 1.0

    order_completion_bonus: float = 10.0
    homogeneity_penalty: float = 5.0
    unassigned_order_step_penalty: float = 2.0


@dataclass(slots=True)
class FeasibleMatchConfig:
    rule_set_name: str = "default"
    include_non_allocatable_debug_matches: bool = False


@dataclass(slots=True)
class EnvConfig:
    max_steps_per_episode: int | None = None
    allow_delay_action: bool = False

    use_dynamic_max_steps: bool = True
    dynamic_max_steps_factor: int = 2
    min_steps_per_episode: int = 20

@dataclass(slots=True)
class DatasetShapeConfig:
    max_materials: int
    max_order_steps: int
    max_actions: int