from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HardRuleConfig:
    allowed_type_pairs: set[tuple[str, str]] = field(default_factory=set)
    enforce_dimension_rules: bool = True
    enforce_single_use_material: bool = True


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
class EnvConfig:
    max_steps_per_episode: int = 500
    allow_delay_action: bool = False