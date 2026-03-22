from __future__ import annotations

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ma_rl.domain import (
    Assignment,
    AssignmentSet,
    DatasetShapeConfig,
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    Scenario,
    ScoreWeights,
    apply_assignment_set_score,
)
from ma_rl.matching import generate_feasible_matches


@dataclass(slots=True)
class CandidateMatch:
    action_id: int
    material_id: str
    order_step_id: str
    score: float
    rule_set_name: str | None


class MultiScenarioMaterialAllocatorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenarios: list[Scenario],
        shape_config: DatasetShapeConfig,
        hard_rule_config: HardRuleConfig,
        score_weights: ScoreWeights,
        feasible_match_config: FeasibleMatchConfig,
        env_config: EnvConfig,
        penalty_threshold: float | None = None,
        invalid_action_penalty: float = -1.0,
        scenario_seed: int = 42,
    ) -> None:
        super().__init__()

        if not scenarios:
            raise ValueError("scenarios must not be empty")

        self.scenarios = scenarios
        self.shape_config = shape_config
        self.hard_rule_config = hard_rule_config
        self.score_weights = score_weights
        self.feasible_match_config = feasible_match_config
        self.env_config = env_config
        self.penalty_threshold = penalty_threshold
        self.invalid_action_penalty = invalid_action_penalty
        self.current_episode_step_limit = 0

        self._scenario_rng = random.Random(scenario_seed)

        self.max_materials = shape_config.max_materials
        self.max_order_steps = shape_config.max_order_steps
        self.max_actions = shape_config.max_actions

        self.action_space = spaces.Discrete(self.max_actions)

        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_actions,),
                    dtype=np.float32,
                ),
                "candidate_scores": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(self.max_actions,),
                    dtype=np.float32,
                ),
                "candidate_material_indices": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_actions,),
                    dtype=np.float32,
                ),
                "candidate_order_step_indices": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_actions,),
                    dtype=np.float32,
                ),
                "material_used": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_materials,),
                    dtype=np.float32,
                ),
                "order_step_used": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_order_steps,),
                    dtype=np.float32,
                ),
                "current_metrics": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(4,),
                    dtype=np.float32,
                ),
            }
        )

        self.current_scenario: Scenario | None = None
        self.materials = []
        self.order_steps = []
        self.material_id_to_index: dict[str, int] = {}
        self.order_step_id_to_index: dict[str, int] = {}
        self.candidates: list[CandidateMatch] = []

        self._candidate_scores = np.zeros(self.max_actions, dtype=np.float32)
        self._candidate_material_indices = np.zeros(self.max_actions, dtype=np.float32)
        self._candidate_order_step_indices = np.zeros(self.max_actions, dtype=np.float32)

        self.current_num_actions = 0
        self.current_num_materials = 0
        self.current_num_order_steps = 0

        self.assignment_set: AssignmentSet | None = None
        self.used_material_ids: set[str] = set()
        self.used_order_step_ids: set[str] = set()
        self.step_count = 0

    @staticmethod
    def _normalize_index(index: int, size: int) -> float:
        if size <= 1:
            return 0.0
        return float(index) / float(size - 1)

    def _select_scenario(self) -> Scenario:
        return self._scenario_rng.choice(self.scenarios)

    def _compute_episode_step_limit(self) -> int:
        if self.env_config.use_dynamic_max_steps:
            dynamic_limit = max(
                self.env_config.min_steps_per_episode,
                self.env_config.dynamic_max_steps_factor * self.current_num_order_steps,
            )

            if self.env_config.max_steps_per_episode is not None:
                return min(dynamic_limit, self.env_config.max_steps_per_episode)

            return dynamic_limit

        if self.env_config.max_steps_per_episode is not None:
            return self.env_config.max_steps_per_episode

        return max(
            self.env_config.min_steps_per_episode,
            self.current_num_order_steps,
        )

    def _prepare_current_scenario(self, scenario: Scenario) -> None:
        self.current_scenario = scenario
        self.materials = list(scenario.materials)
        self.order_steps = list(scenario.order_steps)

        self.current_num_materials = len(self.materials)
        self.current_num_order_steps = len(self.order_steps)

        self.material_id_to_index = {
            material.material_id: idx for idx, material in enumerate(self.materials)
        }
        self.order_step_id_to_index = {
            step.order_step_id: idx for idx, step in enumerate(self.order_steps)
        }

        all_matches = generate_feasible_matches(
            scenario=scenario,
            hard_rule_config=self.hard_rule_config,
            score_weights=self.score_weights,
            feasible_match_config=self.feasible_match_config,
        )

        filtered_matches = [
            match
            for match in all_matches
            if match.allocatable
            and match.score is not None
            and (
                self.penalty_threshold is None
                or match.score >= self.penalty_threshold
            )
        ]

        filtered_matches.sort(key=lambda match: float(match.score), reverse=True)

        if len(filtered_matches) > self.max_actions:
            raise ValueError(
                f"Scenario {scenario.scenario_id} has {len(filtered_matches)} actions, "
                f"but shape_config.max_actions={self.max_actions}."
            )

        self.candidates = [
            CandidateMatch(
                action_id=idx,
                material_id=match.material_id,
                order_step_id=match.order_step_id,
                score=float(match.score),
                rule_set_name=match.rule_set_name,
            )
            for idx, match in enumerate(filtered_matches)
        ]
        self.current_num_actions = len(self.candidates)

        self._candidate_scores.fill(0.0)
        self._candidate_material_indices.fill(0.0)
        self._candidate_order_step_indices.fill(0.0)

        for candidate in self.candidates:
            self._candidate_scores[candidate.action_id] = candidate.score
            self._candidate_material_indices[candidate.action_id] = self._normalize_index(
                self.material_id_to_index[candidate.material_id],
                self.current_num_materials,
            )
            self._candidate_order_step_indices[candidate.action_id] = self._normalize_index(
                self.order_step_id_to_index[candidate.order_step_id],
                self.current_num_order_steps,
            )

    def _build_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.float32)

        for candidate in self.candidates:
            valid = True

            if candidate.material_id in self.used_material_ids:
                valid = False
            if candidate.order_step_id in self.used_order_step_ids:
                valid = False

            if valid:
                mask[candidate.action_id] = 1.0

        return mask

    def get_action_mask(self) -> np.ndarray:
        return self._build_action_mask()

    def _get_observation(self) -> dict[str, np.ndarray]:
        material_used = np.zeros(self.max_materials, dtype=np.float32)
        for material_id in self.used_material_ids:
            idx = self.material_id_to_index[material_id]
            material_used[idx] = 1.0

        order_step_used = np.zeros(self.max_order_steps, dtype=np.float32)
        for order_step_id in self.used_order_step_ids:
            idx = self.order_step_id_to_index[order_step_id]
            order_step_used[idx] = 1.0

        assert self.assignment_set is not None

        current_metrics = np.array(
            [
                float(len(self.assignment_set.assignments)),
                float(self.assignment_set.total_match_score),
                float(self.assignment_set.total_penalty),
                float(self.assignment_set.total_bonus),
            ],
            dtype=np.float32,
        )

        return {
            "action_mask": self._build_action_mask(),
            "candidate_scores": self._candidate_scores.copy(),
            "candidate_material_indices": self._candidate_material_indices.copy(),
            "candidate_order_step_indices": self._candidate_order_step_indices.copy(),
            "material_used": material_used,
            "order_step_used": order_step_used,
            "current_metrics": current_metrics,
        }

    def _get_info(self) -> dict:
        assert self.assignment_set is not None
        assert self.current_scenario is not None

        return {
            "scenario_id": self.current_scenario.scenario_id,
            "action_mask": self._build_action_mask(),
            "step_count": self.step_count,
            "assignments_selected": len(self.assignment_set.assignments),
            "total_match_score": self.assignment_set.total_match_score,
            "total_penalty": self.assignment_set.total_penalty,
            "total_bonus": self.assignment_set.total_bonus,
            "total_score": self.assignment_set.total_score,
            "num_actions": self.current_num_actions,
            "num_materials": self.current_num_materials,
            "num_order_steps": self.current_num_order_steps,
            "episode_step_limit": self.current_episode_step_limit,
        }

    def _is_terminated(self) -> bool:
        if len(self.used_order_step_ids) == self.current_num_order_steps:
            return True

        if np.sum(self._build_action_mask()) == 0:
            return True

        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        scenario = self._select_scenario()
        self._prepare_current_scenario(scenario)
        self.current_episode_step_limit = self._compute_episode_step_limit()

        self.assignment_set = AssignmentSet()
        apply_assignment_set_score(
            assignment_set=self.assignment_set,
            scenario=scenario,
            weights=self.score_weights,
        )

        self.used_material_ids = set()
        self.used_order_step_ids = set()
        self.step_count = 0

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        assert self.assignment_set is not None
        assert self.current_scenario is not None

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space.")

        previous_total_score = self.assignment_set.total_score
        terminated = False
        truncated = False

        action_mask = self._build_action_mask()
        is_valid_action = bool(action_mask[action] > 0.0)

        if not is_valid_action:
            reward = float(self.invalid_action_penalty)
            self.step_count += 1

            if self.step_count >= self.current_episode_step_limit:
                truncated = True
            else:
                terminated = self._is_terminated()

            observation = self._get_observation()
            info = self._get_info()
            info["invalid_action"] = True
            return observation, reward, terminated, truncated, info

        candidate = self.candidates[action]

        self.assignment_set.add_assignment(
            Assignment(
                material_id=candidate.material_id,
                order_step_id=candidate.order_step_id,
                score=candidate.score,
                source_rule_set_name=candidate.rule_set_name,
            )
        )

        self.used_material_ids.add(candidate.material_id)
        self.used_order_step_ids.add(candidate.order_step_id)

        apply_assignment_set_score(
            assignment_set=self.assignment_set,
            scenario=self.current_scenario,
            weights=self.score_weights,
        )

        reward = float(self.assignment_set.total_score - previous_total_score)
        self.step_count += 1

        if self.step_count >= self.current_episode_step_limit:
            truncated = True
        else:
            terminated = self._is_terminated()

        observation = self._get_observation()
        info = self._get_info()
        info["invalid_action"] = False

        return observation, reward, terminated, truncated, info