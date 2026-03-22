from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

from ma_rl.analysis import write_excel_friendly_csv, write_simple_xlsx
from ma_rl.domain import (
    DatasetShapeConfig,
    EnvConfig,
    FeasibleMatchConfig,
    HardRuleConfig,
    Scenario,
    ScoreWeights,
)
from ma_rl.rl.evaluation_utils import evaluate_model_on_scenarios, summarize_eval_rows


class ValidationEvalCallback(BaseCallback):
    def __init__(
        self,
        val_scenarios: list[Scenario],
        shape_config: DatasetShapeConfig,
        hard_rule_config: HardRuleConfig,
        score_weights: ScoreWeights,
        feasible_match_config: FeasibleMatchConfig,
        env_config: EnvConfig,
        output_dir: str | Path,
        eval_freq: int = 10_000,
        penalty_threshold: float | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.val_scenarios = val_scenarios
        self.shape_config = shape_config
        self.hard_rule_config = hard_rule_config
        self.score_weights = score_weights
        self.feasible_match_config = feasible_match_config
        self.env_config = env_config
        self.output_dir = Path(output_dir)
        self.eval_freq = eval_freq
        self.penalty_threshold = penalty_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[dict] = []
        self.best_mean_total_score = float("-inf")

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if self.num_timesteps % self.eval_freq != 0:
            return True

        rows = evaluate_model_on_scenarios(
            model=self.model,
            scenarios=self.val_scenarios,
            shape_config=self.shape_config,
            hard_rule_config=self.hard_rule_config,
            score_weights=self.score_weights,
            feasible_match_config=self.feasible_match_config,
            env_config=self.env_config,
            penalty_threshold=self.penalty_threshold,
        )

        summary = summarize_eval_rows(rows)
        history_row = {
            "timesteps": self.num_timesteps,
            **summary,
        }
        self.history.append(history_row)

        csv_path = self.output_dir / "validation_history.csv"
        xlsx_path = self.output_dir / "validation_history.xlsx"

        write_excel_friendly_csv(csv_path, self.history)
        write_simple_xlsx(xlsx_path, self.history, sheet_name="ValidationHistory")

        if summary["mean_total_score"] > self.best_mean_total_score:
            self.best_mean_total_score = summary["mean_total_score"]
            best_model_path = self.output_dir / "best_model"
            self.model.save(str(best_model_path))

        return True

    def _write_plot(self, path: Path) -> None:
        if not self.history:
            return

        x = [row["timesteps"] for row in self.history]
        y = [row["mean_total_score"] for row in self.history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(x, y, marker="o")
        plt.xlabel("Timesteps")
        plt.ylabel("Validation Mean Total Score")
        plt.title("Validation Score over Training Time")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()