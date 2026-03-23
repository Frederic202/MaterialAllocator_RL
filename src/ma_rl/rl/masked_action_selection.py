from __future__ import annotations

import numpy as np
import torch


def select_masked_greedy_action(model, obs: dict, action_mask: np.ndarray) -> int:
    if np.sum(action_mask) <= 0:
        raise ValueError("No valid actions available for masked action selection.")

    obs_tensor, _ = model.policy.obs_to_tensor(obs)

    with torch.no_grad():
        # QR-DQN: quantile_net -> mean over quantiles -> Q-values per action
        if hasattr(model.policy, "quantile_net"):
            q_values = model.policy.quantile_net(obs_tensor).mean(dim=1).cpu().numpy()[0]
        # DQN: q_net -> direct Q-values per action
        elif hasattr(model, "q_net"):
            q_values = model.q_net(obs_tensor).cpu().numpy()[0]
        else:
            raise TypeError("Unsupported model type for masked greedy action selection.")

    masked_q_values = q_values.copy()
    masked_q_values[action_mask <= 0.0] = -1e12

    return int(np.argmax(masked_q_values))