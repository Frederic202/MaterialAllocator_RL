from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import DQN


def select_masked_greedy_action(model: DQN, obs: dict, action_mask: np.ndarray) -> int:
    if np.sum(action_mask) <= 0:
        raise ValueError("No valid actions available for masked action selection.")

    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]

    masked_q_values = q_values.copy()
    masked_q_values[action_mask <= 0.0] = -1e12

    return int(np.argmax(masked_q_values))