"""Training entry point.

    python model/train.py

Trains on seasons before --val-season, selects the checkpoint on --val-season,
and reports on --test-season, which is touched exactly once at the end.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from data import load_schedules, load_weekly
from dataset import (
    MIN_HISTORY, SEQ_LEN, PlayerWeeks, Standardizer, build_windows, temporal_split,
)
from features import CONTEXT_FEATURES, SEQ_STATS, build
from net import ProjectionNet

log = logging.getLogger("train")
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def predict(model: nn.Module, seq: np.ndarray, ctx: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            s = torch.from_numpy(seq[i : i + batch])
            c = torch.from_numpy(ctx[i : i + batch])
            out.append(model(s, c).numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-season", type=int, default=2016)
    ap.add_argument("--val-season", type=int, default=2023)
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    years = list(range(args.first_season, args.test_season + 1))
    log.info("loading seasons %s-%s ...", years[0], years[-1])
    frame = build(load_weekly(years), load_schedules(years))
    log.info("modelling frame: %s rows", f"{len(frame):,}")

    split = temporal_split(frame, args.val_season, args.test_season)
    log.info("%s", split.describe())

    train_w = build_windows(split.train)
    val_w = build_windows(split.val)
    test_w = build_windows(split.test)
    log.info(
        "windows (seq_len=%d, min_history=%d): train %s / val %s / test %s",
        SEQ_LEN, MIN_HISTORY,
        f"{len(train_w['y']):,}", f"{len(val_w['y']):,}", f"{len(test_w['y']):,}",
    )

    scaler = Standardizer().fit(train_w["seq"], train_w["ctx"])
    tr_seq, tr_ctx = scaler.transform(train_w["seq"], train_w["ctx"])
    va_seq, va_ctx = scaler.transform(val_w["seq"], val_w["ctx"])
    te_seq, te_ctx = scaler.transform(test_w["seq"], test_w["ctx"])

    loader = DataLoader(
        PlayerWeeks(tr_seq, tr_ctx, train_w["y"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    model = ProjectionNet(len(SEQ_STATS), len(CONTEXT_FEATURES), hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # L1 matches the headline metric (MAE) and is less bullied by the long right
    # tail of fantasy scoring than MSE.
    loss_fn = nn.L1Loss()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACTS / "projection_net.pt"
    best_val, best_epoch, since_best = float("inf"), -1, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for seq, ctx, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(seq, ctx), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(y)

        val_mae = evaluate.mae(val_w["y"], predict(model, va_seq, va_ctx))
        flag = ""
        if val_mae < best_val - 1e-4:
            best_val, best_epoch, since_best, flag = val_mae, epoch, 0, "  *"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "seq_features": SEQ_STATS,
                    "ctx_features": CONTEXT_FEATURES,
                    "seq_len": SEQ_LEN,
                    "hidden": args.hidden,
                    "val_season": args.val_season,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )
        else:
            since_best += 1

        log.info(
            "epoch %2d  train L1 %.3f  val MAE %.3f%s",
            epoch, running / len(loader.dataset), val_mae, flag,
        )
        if since_best >= args.patience:
            log.info("no val improvement in %d epochs -- stopping", args.patience)
            break

    log.info("best epoch %d (val MAE %.3f); restoring that checkpoint", best_epoch, best_val)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])

    # The test season is scored once, here, after model selection is finished.
    test_pred = predict(model, te_seq, te_ctx)
    val_pred = predict(model, va_seq, va_ctx)
    report = {
        "test_season": args.test_season,
        "val_season": args.val_season,
        "train_seasons": [args.first_season, args.val_season - 1],
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val": evaluate.compare(val_w["frame"], val_w["y"], val_pred, val_w["baseline"]),
        "test": evaluate.compare(test_w["frame"], test_w["y"], test_pred, test_w["baseline"]),
        "per_position": evaluate.per_position(
            test_w["frame"], test_w["y"], test_pred, test_w["baseline"]
        ),
    }
    (ARTIFACTS / "results.json").write_text(json.dumps(report, indent=2))
    print(evaluate.render(report))


if __name__ == "__main__":
    main()
