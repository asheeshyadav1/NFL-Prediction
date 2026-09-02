"""Load the trained weights and serve projections. This module owns the number."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from dataset import Standardizer  # noqa: E402
from net import ProjectionNet  # noqa: E402

log = logging.getLogger(__name__)
CHECKPOINT = MODEL_DIR / "artifacts" / "projection_net.pt"


class Projector:
    def __init__(self, checkpoint: Path = CHECKPOINT) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"no trained weights at {checkpoint} -- run `python model/train.py` first"
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.scaler = Standardizer.from_state_dict(ckpt["scaler"])
        self.model = ProjectionNet(
            n_seq_features=len(ckpt["seq_features"]),
            n_ctx_features=len(ckpt["ctx_features"]),
            hidden=ckpt["hidden"],
            # Older checkpoints predate the floor and were all non-negative.
            floor=ckpt.get("floor", 0.0),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.val_mae = float(ckpt["val_mae"])
        log.info("loaded projection model (val MAE %.3f)", self.val_mae)

    def project(self, seq: np.ndarray, ctx: np.ndarray) -> float:
        """Project one player-week. Input arrays are raw (unscaled)."""
        return float(self.project_batch(seq[None, ...], ctx[None, ...])[0])

    def project_batch(self, seq: np.ndarray, ctx: np.ndarray) -> np.ndarray:
        s, c = self.scaler.transform(seq.astype(np.float32), ctx.astype(np.float32))
        with torch.no_grad():
            return self.model(torch.from_numpy(s), torch.from_numpy(c)).numpy()
