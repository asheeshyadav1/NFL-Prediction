"""Bring every feed up to date. Safe to re-run; safe to schedule.

    python scripts/refresh.py                 # data + corpora
    python scripts/refresh.py --retrain       # ...and retrain on what arrived

The raw loaders cache by season and skip a file that already exists, which is
what makes the pipeline reproducible offline -- and also what makes an
in-progress season go stale. This drops the cache for the seasons still in
motion, re-pulls them, and rebuilds both corpora.

Nothing here fails the run when a feed has not been published yet: before week
1 there are no box scores and no injury report, and that is the expected state,
not an error. The summary at the end says what actually landed.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

log = logging.getLogger("refresh")
RAW = ROOT / "data" / "raw"
FIRST_SEASON = 2016


def current_season(today: date | None = None) -> int:
    """The NFL season now in progress.

    A season is named for the calendar year it starts in and runs into
    February, so anything before March still belongs to the previous year.
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def drop_cache(season: int) -> list[str]:
    """Delete the cached pulls for a season still being played.

    Completed seasons are left alone: they never change, and re-downloading a
    decade of parquet on every run would be pure waste.
    """
    dropped = []
    for name in (f"weekly_{season}.parquet", f"injuries_{season}.parquet",
                 f"depth_charts_{season}.parquet", f"rosters_{season}.parquet",
                 "games.csv"):
        path = RAW / name
        if path.exists():
            path.unlink()
            dropped.append(name)
    return dropped


def run(label: str, args: list[str]) -> bool:
    """Run one ingest. A non-zero exit is reported, not raised -- a feed that
    does not exist yet must not take the rest of the refresh down."""
    log.info("--- %s", label)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        log.warning("%s reported nothing to do (exit %d)", label, result.returncode)
    return result.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=None,
                    help="season to refresh (default: the one in progress)")
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the projection model afterwards")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    season = args.season or current_season()
    log.info("refreshing season %s", season)

    dropped = drop_cache(season)
    log.info("dropped %d cached file(s): %s", len(dropped), ", ".join(dropped) or "none")

    # Injuries are rewritten for the whole range, not just this season: the
    # ingest writes one corpus file, so a single-season run would drop history.
    seasons = [str(y) for y in range(FIRST_SEASON, season + 1)]
    injuries = run("injury report", ["scripts/ingest_injuries.py", "--seasons", *seasons])
    preseason = run("depth charts", ["scripts/ingest_preseason.py", "--season", str(season)])

    # What can actually be projected now.
    from data import load_weekly  # noqa: E402  (after sys.path setup)

    try:
        weekly = load_weekly([season])
        played = sorted(int(w) for w in weekly["week"].unique())
    except Exception:
        played = []

    log.info("")
    log.info("=== %s ===", season)
    log.info("  weeks with box scores : %s", f"1-{max(played)}" if played else "none yet")
    log.info("  injury report corpus  : %s", "updated" if injuries else "not published yet")
    log.info("  depth chart corpus    : %s", "updated" if preseason else "unavailable")
    if not played:
        log.info("  -> no projectable player-weeks until week 1 box scores land")

    if args.retrain:
        if not played:
            log.warning("skipping retrain: %s has no box scores yet", season)
        else:
            run("retrain", ["model/train.py", "--test-season", str(season)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
