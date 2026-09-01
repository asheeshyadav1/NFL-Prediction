"""Sequence windowing and the temporal split.

Two invariants carry the project's credibility:

1. A window for player-week *t* holds games *t-N .. t-1* only -- never its own.
2. The split is by time, never random; a random split leaks future weeks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import CONTEXT_FEATURES, SEQ_STATS, TARGET

SEQ_LEN = 6
# Also keeps the naive 3-game baseline well defined, so the comparison is fair.
MIN_HISTORY = 3


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def line(name: str, d: pd.DataFrame) -> str:
            return (
                f"  {name:<5} {len(d):>6,} rows  "
                f"seasons {d['season'].min()}-{d['season'].max()}"
            )

        return "\n".join(
            ["temporal split (no random shuffling):", line("train", self.train),
             line("val", self.val), line("test", self.test)]
        )


def temporal_split(df: pd.DataFrame, val_season: int, test_season: int) -> Split:
    if not test_season > val_season:
        raise ValueError("test season must come after the validation season")
    return Split(
        train=df[df["season"] < val_season].reset_index(drop=True),
        val=df[df["season"] == val_season].reset_index(drop=True),
        test=df[df["season"] == test_season].reset_index(drop=True),
    )


def build_windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Turn the modelling frame into fixed-length per-player sequences.

    Arrays align row-for-row with the kept rows of `df`; rows with fewer than
    MIN_HISTORY prior games are dropped.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    seq_mat = df[SEQ_STATS].to_numpy(dtype=np.float32)

    keep, seqs = [], []
    for _, idx in df.groupby("player_id", sort=False).indices.items():
        idx = np.sort(idx)
        for pos, row in enumerate(idx):
            if pos < MIN_HISTORY:
                continue
            hist = idx[max(0, pos - SEQ_LEN) : pos]  # strictly prior games
            window = seq_mat[hist]
            if len(window) < SEQ_LEN:  # left-pad short histories with zeros
                pad = np.zeros((SEQ_LEN - len(window), window.shape[1]), np.float32)
                window = np.vstack([pad, window])
            seqs.append(window)
            keep.append(row)

    keep = np.asarray(keep, dtype=np.int64)
    kept = df.iloc[keep]
    return {
        "seq": np.stack(seqs).astype(np.float32),
        "ctx": kept[CONTEXT_FEATURES].to_numpy(dtype=np.float32),
        "y": kept[TARGET].to_numpy(dtype=np.float32),
        "baseline": kept["roll3_ppr"].to_numpy(dtype=np.float32),
        "index": keep,
        "frame": kept.reset_index(drop=True),
    }


class Standardizer:
    """Mean/std scaling fit on the training split only; fitting on everything
    would leak the test-set distribution into training."""

    def __init__(self) -> None:
        self.seq_mean = self.seq_std = self.ctx_mean = self.ctx_std = None

    def fit(self, seq: np.ndarray, ctx: np.ndarray) -> "Standardizer":
        flat = seq.reshape(-1, seq.shape[-1])
        self.seq_mean, self.seq_std = flat.mean(0), flat.std(0) + 1e-6
        self.ctx_mean, self.ctx_std = ctx.mean(0), ctx.std(0) + 1e-6
        return self

    def transform(self, seq: np.ndarray, ctx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (seq - self.seq_mean) / self.seq_std, (ctx - self.ctx_mean) / self.ctx_std

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "seq_mean": self.seq_mean.tolist(), "seq_std": self.seq_std.tolist(),
            "ctx_mean": self.ctx_mean.tolist(), "ctx_std": self.ctx_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "Standardizer":
        s = cls()
        for k, v in state.items():
            setattr(s, k, np.asarray(v, dtype=np.float32))
        return s


class PlayerWeeks(Dataset):
    def __init__(self, seq: np.ndarray, ctx: np.ndarray, y: np.ndarray) -> None:
        self.seq, self.ctx, self.y = (torch.from_numpy(a) for a in (seq, ctx, y))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.seq[i], self.ctx[i], self.y[i]
