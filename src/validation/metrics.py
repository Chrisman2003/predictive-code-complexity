"""
Evaluation metrics for story-point estimation.

All functions work on raw story points (not ranks), so numbers are comparable
with the GPT2SP / Deep-SE literature. `y_pred` is expected to be the decoded
scale value (1, 2, 3, 5, 8, 13, 20); `y_true` should be the *raw* point value.
"""
from typing import Dict, Optional, Sequence

import numpy as np


def _arr(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def mdae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.median(np.abs(_arr(y_true) - _arr(y_pred))))


def macro_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean of the per-true-value MAEs. A constant predictor scores badly here
    because rare classes count as much as common ones: the best collapse check."""
    y, p = _arr(y_true), _arr(y_pred)
    return float(np.mean([np.abs(y[y == v] - p[y == v]).mean() for v in np.unique(y)]))


def random_guess_mae(
    y_true: Sequence[float],
    reference_points: Sequence[float],
    n_runs: int = 1000,
    seed: int = 0,
) -> float:
    """Expected MAE when each guess is drawn at random from the reference
    (training) points. Used as the denominator of standardized accuracy."""
    rng = np.random.default_rng(seed)
    y, ref = _arr(y_true), _arr(reference_points)
    guesses = rng.choice(ref, size=(n_runs, len(y)))
    return float(np.abs(guesses - y).mean())


def standardized_accuracy(mae_model: float, mae_random: float) -> float:
    """SA = (1 - MAE_model / MAE_random_guessing) * 100."""
    return (1.0 - mae_model / mae_random) * 100.0


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    reference_points: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Headline metrics. If `reference_points` (normally the training points) is
    given, also returns the median/mean baselines, the improvement over the
    median baseline, and standardized accuracy.
    """
    y, p = _arr(y_true), _arr(y_pred)
    out: Dict[str, float] = {
        "n": int(len(y)),
        "mae": mae(y, p),
        "mdae": mdae(y, p),
        "macro_mae": macro_mae(y, p),
        "pred_std": float(p.std()),
    }
    if reference_points is not None and len(reference_points) > 0:
        ref = _arr(reference_points)
        out["baseline_median_mae"] = mae(y, np.full_like(y, np.median(ref)))
        out["baseline_mean_mae"] = mae(y, np.full_like(y, ref.mean()))
        base = out["baseline_median_mae"]
        out["improvement_vs_median_pct"] = (1.0 - out["mae"] / base) * 100.0 if base > 0 else float("nan")
        out["sa"] = standardized_accuracy(out["mae"], random_guess_mae(y, ref))
    return out


def paired_wilcoxon(
    y_true: Sequence[float],
    pred_model: Sequence[float],
    pred_baseline: Sequence[float],
) -> Dict[str, float]:
    """
    One-sided Wilcoxon signed-rank test on paired absolute errors
    (H1: the model's errors are smaller than the baseline's), plus the
    Vargha-Delaney A12 effect size (probability that a model error is smaller
    than a baseline error; 0.5 = no effect). Requires scipy.
    """
    from scipy.stats import wilcoxon

    y = _arr(y_true)
    e_model = np.abs(y - _arr(pred_model))
    e_base = np.abs(y - _arr(pred_baseline))
    if np.allclose(e_model, e_base):
        return {"p_value": 1.0, "a12": 0.5}
    p_value = float(wilcoxon(e_model, e_base, alternative="less").pvalue)
    a12 = (np.sum(e_model[:, None] < e_base[None, :]) + 0.5 * np.sum(e_model[:, None] == e_base[None, :])) / (
        len(e_model) * len(e_base)
    )
    return {"p_value": p_value, "a12": float(a12)}