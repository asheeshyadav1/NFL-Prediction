"""Precompute everything the site needs, so it can be hosted as static files.

    python scripts/export_static.py

The app normally asks a Python service for a projection. That service wants
torch, ~2GB of memory and a machine that never sleeps, which is a running cost.
Every answer it gives is deterministic, though: the model is fixed, the seasons
are complete, and retrieval has no randomness. So run all of it once here and
ship the results.

The model still produces every number. It just does so at export time rather
than per request. Baking the model itself into the page instead would mean
shipping 18.8 MB of sequence windows; the answers are 1.0 MB.

Writes into frontend/public/data:

    meta.json               model card, seasons covered
    weeks.json              every week, playable or upcoming
    players.json            one row per player, with career shape
    latest.json             each player's most recent week, for compare mode
    slates/<season>-<week>.json   one week's projections, fetched on demand
    snippets.json           the retrieved context, keyed by id

Retrieval is precomputed too, per player-week, using the same store the service
uses. That keeps citations identical to the served version and avoids
reimplementing a hashing vectoriser in JavaScript.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from api import rag  # noqa: E402
from api.feature_store import FeatureStore  # noqa: E402
from api.inference import Projector  # noqa: E402

log = logging.getLogger("export-static")
OUT = ROOT / "frontend" / "public" / "data"
TOP_K = 2


def _round(x: float, places: int = 1) -> float:
    return round(float(x), places)


def precompute_snippets(store, frame, projector) -> dict:
    """Top-k snippets per player-week, scoped exactly as the service scopes them.

    Grouped by (season, week) because every row in a week shares one eligibility
    mask, which turns 51k masked searches into 190 of them.
    """
    picked: dict[str, list[str]] = {}
    used: set[str] = set()

    groups = frame.groupby(["season", "week"], sort=True)
    for i, ((season, week), rows) in enumerate(groups, 1):
        for _, r in rows.iterrows():
            hits = store.search(
                f"{r['player_display_name']} {r['recent_team']} injury status outlook",
                k=TOP_K, season=int(season), week=int(week),
            )
            if hits:
                key = f"{r['player_id']}|{int(season)}|{int(week)}"
                picked[key] = [h.id for h in hits]
                used.update(h.id for h in hits)
        if i % 40 == 0:
            log.info("  retrieval %d/%d week groups", i, groups.ngroups)
    return picked, used


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("**/*.json"):
        stale.unlink()

    log.info("loading model and features ...")
    projector = Projector()
    store = FeatureStore()
    frame = store._frame
    seq, ctx = store._seq, store._ctx

    log.info("projecting %s player-weeks ...", f"{len(frame):,}")
    projections = projector.project_batch(seq, ctx)

    frame = frame.reset_index(drop=True).copy()
    frame["projection"] = np.round(projections, 1)
    frame["baseline_r"] = np.round(store._baseline, 1)
    frame["actual_r"] = np.round(store._y, 1)

    log.info("precomputing retrieval ...")
    rag_store = rag.build_store()
    cites, used_ids = precompute_snippets(rag_store, frame, projector)

    # --- snippets actually referenced, keyed by id ---
    snippets = {
        s.id: {"player": s.player, "published": s.published, "source": s.source, "text": s.text}
        for s in rag_store._snippets
        if s.id in used_ids
    }
    (OUT / "snippets.json").write_text(json.dumps(snippets, separators=(",", ":")))

    # --- weeks, including the upcoming ones ---
    (OUT / "weeks.json").write_text(json.dumps(store.weeks(), separators=(",", ":")))

    # --- per-week slates, fetched on demand ---
    slates = OUT / "slates"
    slates.mkdir(exist_ok=True)
    for (season, week), rows in frame.groupby(["season", "week"], sort=True):
        payload = [
            {
                "player_id": r["player_id"],
                "name": r["player_display_name"],
                "position": r["position"],
                "team": r["recent_team"],
                "opponent": r["opponent_team"],
                "season": int(season),
                "week": int(week),
                "projection": _round(r["projection"]),
                "baseline": _round(r["baseline_r"]),
                "actual": _round(r["actual_r"]),
                "cites": cites.get(f"{r['player_id']}|{int(season)}|{int(week)}", []),
            }
            for _, r in rows.iterrows()
        ]
        (slates / f"{int(season)}-{int(week)}.json").write_text(
            json.dumps(payload, separators=(",", ":"))
        )

    # --- players and their career shape, for the compare view ---
    players, latest = [], {}
    for player_id, rows in frame.groupby("player_id", sort=False):
        rows = rows.sort_values(["season", "week"])
        last = rows.iloc[-1]
        pts = rows["actual_r"]
        players.append({
            "player_id": player_id,
            "name": last["player_display_name"],
            "position": last["position"],
            "team": last["recent_team"],
            "career_games": int(len(pts)),
            "career_avg": _round(pts.mean()),
        })
        latest[player_id] = {
            "player_id": player_id,
            "name": last["player_display_name"],
            "position": last["position"],
            "team": last["recent_team"],
            "opponent": last["opponent_team"],
            "season": int(last["season"]),
            "week": int(last["week"]),
            "projection": _round(last["projection"]),
            "baseline": _round(last["baseline_r"]),
            "actual": _round(last["actual_r"]),
            "career_games": int(len(pts)),
            "career_avg": _round(pts.mean()),
            "last6_avg": _round(pts.tail(6).mean()),
            "cites": cites.get(f"{player_id}|{int(last['season'])}|{int(last['week'])}", []),
        }
    players.sort(key=lambda p: p["name"])
    (OUT / "players.json").write_text(json.dumps(players, separators=(",", ":")))
    (OUT / "latest.json").write_text(json.dumps(latest, separators=(",", ":")))

    (OUT / "meta.json").write_text(json.dumps({
        "val_mae": projector.val_mae,
        "seasons": [int(frame["season"].min()), int(frame["season"].max())],
        "player_weeks": int(len(frame)),
        "players": len(players),
        "snippets": len(snippets),
        "note": (
            "Projections precomputed by model/artifacts/projection_net.pt. "
            "Regenerate with scripts/export_static.py after retraining."
        ),
    }, indent=2))

    total = sum(f.stat().st_size for f in OUT.glob("**/*.json"))
    log.info("")
    log.info("wrote %s files, %.1f MB total", f"{len(list(OUT.glob('**/*.json'))):,}", total / 1e6)
    log.info("  %s player-weeks, %s players, %s snippets",
             f"{len(frame):,}", f"{len(players):,}", f"{len(snippets):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
