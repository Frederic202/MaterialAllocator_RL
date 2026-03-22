from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BenchmarkRow:
    algorithm: str
    seed: int
    scenario_id: str

    assignments_selected: int
    total_match_score: float
    total_penalty: float
    total_bonus: float
    total_score: float

    cumulative_reward: float
    invalid_actions: int
    remaining_valid_actions: int

    training_timesteps: int
    note: str = ""


def benchmark_row_to_dict(row: BenchmarkRow) -> dict:
    return asdict(row)