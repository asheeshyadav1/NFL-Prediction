"""Metrics and baselines.

Every number the README quotes is produced here, on the held-out test season the
model never saw during training or selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Start/sit pairs closer than this in actual points are coin flips that mostly
# measure noise, so they are excluded and the exclusion is reported.
TIE_MARGIN = 1.0


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def start_sit_accuracy(
    frame: pd.DataFrame, y: np.ndarray, pred: np.ndarray, seed: int = 0,
    pairs_per_week: int = 400, same_position: bool = False,
) -> tuple[float, int]:
    """How often the projection ranks two players in the right order.

    Pairs are drawn within a single week, which is the decision a fantasy manager
    actually faces: given these two, who do I start?
    """
    rng = np.random.default_rng(seed)
    df = frame.reset_index(drop=True).assign(_y=y, _pred=pred)

    correct = total = 0
    groups = ["season", "week", "position"] if same_position else ["season", "week"]
    for _, grp in df.groupby(groups, sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < 2:
            continue
        a = rng.choice(idx, size=min(pairs_per_week, len(idx) * 4))
        b = rng.choice(idx, size=len(a))
        keep = a != b
        a, b = a[keep], b[keep]

        ya, yb = df["_y"].to_numpy()[a], df["_y"].to_numpy()[b]
        decisive = np.abs(ya - yb) >= TIE_MARGIN
        a, b, ya, yb = a[decisive], b[decisive], ya[decisive], yb[decisive]

        pa, pb = df["_pred"].to_numpy()[a], df["_pred"].to_numpy()[b]
        correct += int(np.sum((pa > pb) == (ya > yb)))
        total += len(a)

    return (correct / total if total else float("nan")), total


def compare(frame: pd.DataFrame, y: np.ndarray, model_pred: np.ndarray,
            baseline_pred: np.ndarray) -> dict:
    """Full model-vs-baseline report on one split."""
    b_mae, m_mae = mae(y, baseline_pred), mae(y, model_pred)
    b_acc, n_pairs = start_sit_accuracy(frame, y, baseline_pred)
    m_acc, _ = start_sit_accuracy(frame, y, model_pred)
    b_acc_pos, n_pos = start_sit_accuracy(frame, y, baseline_pred, same_position=True)
    m_acc_pos, _ = start_sit_accuracy(frame, y, model_pred, same_position=True)

    return {
        "n_player_weeks": int(len(y)),
        "baseline_mae": b_mae,
        "model_mae": m_mae,
        "mae_improvement_pct": 100.0 * (b_mae - m_mae) / b_mae,
        "baseline_rmse": rmse(y, baseline_pred),
        "model_rmse": rmse(y, model_pred),
        "start_sit_pairs": n_pairs,
        "baseline_start_sit_acc": b_acc,
        "model_start_sit_acc": m_acc,
        "start_sit_pairs_same_position": n_pos,
        "baseline_start_sit_acc_same_position": b_acc_pos,
        "model_start_sit_acc_same_position": m_acc_pos,
    }


def per_position(frame: pd.DataFrame, y: np.ndarray, model_pred: np.ndarray,
                 baseline_pred: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    pos = frame["position"].to_numpy()
    for p in ("QB", "RB", "WR", "TE"):
        m = pos == p
        if m.sum() == 0:
            continue
        out[p] = {
            "n": int(m.sum()),
            "baseline_mae": mae(y[m], baseline_pred[m]),
            "model_mae": mae(y[m], model_pred[m]),
        }
    return out


def render(report: dict) -> str:
    t = report["test"]
    lines = [
        "",
        f"HELD-OUT TEST SEASON: {report['test_season']}   ({t['n_player_weeks']:,} player-weeks)",
        "=" * 62,
        f"{'':<28}{'baseline':>12}{'model':>12}{'delta':>10}",
        "-" * 62,
        f"{'MAE (PPR points)':<28}{t['baseline_mae']:>12.3f}{t['model_mae']:>12.3f}"
        f"{t['baseline_mae'] - t['model_mae']:>+10.3f}",
        f"{'RMSE':<28}{t['baseline_rmse']:>12.3f}{t['model_rmse']:>12.3f}"
        f"{t['baseline_rmse'] - t['model_rmse']:>+10.3f}",
        f"{'Start/sit acc (any pos)':<28}{t['baseline_start_sit_acc']:>11.1%}"
        f"{t['model_start_sit_acc']:>12.1%}"
        f"{t['model_start_sit_acc'] - t['baseline_start_sit_acc']:>+9.1%}",
        f"{'Start/sit acc (same pos)':<28}"
        f"{t['baseline_start_sit_acc_same_position']:>11.1%}"
        f"{t['model_start_sit_acc_same_position']:>12.1%}"
        f"{t['model_start_sit_acc_same_position'] - t['baseline_start_sit_acc_same_position']:>+9.1%}",
        "-" * 62,
        f"MAE improvement over naive last-3 baseline: {t['mae_improvement_pct']:.1f}%",
        "",
        "By position (MAE):",
    ]
    for pos, s in report["per_position"].items():
        delta = 100.0 * (s["baseline_mae"] - s["model_mae"]) / s["baseline_mae"]
        lines.append(
            f"  {pos:<4} n={s['n']:>5,}  baseline {s['baseline_mae']:>6.3f}"
            f"   model {s['model_mae']:>6.3f}   ({delta:+.1f}%)"
        )
    return "\n".join(lines)


def main() -> None:
    path = ARTIFACTS / "results.json"
    if not path.exists():
        raise SystemExit("no results.json -- run `python model/train.py` first")
    print(render(json.loads(path.read_text())))


if __name__ == "__main__":
    main()
