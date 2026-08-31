# GridironIQ — Fantasy Football Projections That Explain Themselves

> A weekly NFL fantasy-points projection engine where a **trained model** makes the
> predictions and an **LLM only explains them** — grounded in injury and news data
> via retrieval. The projections aren't the LLM guessing; they come from a sequence
> model that beats a naive baseline on a leakage-safe evaluation.

---

## Results

Trained on 2016–2022, model selected on 2023, and scored **once** on a held-out
2024 season the model never saw during training or selection.

| | Naive baseline | Model | Delta |
|---|---:|---:|---:|
| MAE (PPR points) | 4.912 | **4.519** | **−8.0%** |
| RMSE | 6.811 | **6.469** | −5.0% |
| Start/sit accuracy (any position) | 74.9% | **76.9%** | +2.1 pts |
| Start/sit accuracy (same position) | 72.0% | **74.1%** | +2.1 pts |

*4,213 held-out player-weeks. Baseline = "project = last-3-game average".
Start/sit accuracy is measured on within-week player pairs, excluding pairs
decided by less than 1.0 actual points, which are coin flips that mostly measure
noise (5,399 pairs any-position, 13,873 same-position).*

By position (MAE), held-out 2024:

| Position | n | Baseline | Model | Improvement |
|---|---:|---:|---:|---:|
| QB | 452 | 6.745 | 6.289 | +6.8% |
| RB | 1,106 | 4.796 | 4.426 | +7.7% |
| WR | 1,775 | 4.964 | 4.533 | +8.7% |
| TE | 880 | 4.010 | 3.700 | +7.7% |

The gain is consistent across positions rather than driven by one of them, which
is what you want to see — a single-position spike usually means a quirk rather
than a signal. Regenerate any of this with `python model/evaluate.py`; the raw
report is committed at `model/artifacts/results.json`.

**What the number means, honestly.** An 8% MAE improvement over a last-3-game
average is a real but modest edge, and that is roughly the right size for this
problem: weekly fantasy scoring is dominated by variance (touchdowns, game
script, injuries mid-game) that no amount of modelling removes. The start/sit
gain of ~2 points of accuracy is the more decision-relevant number, since
picking between two players is the task a manager actually faces. Anyone
claiming a much larger improvement on this target is usually leaking future
information — which is why the leakage guarantees below are tested rather than
asserted.

---

## Why this exists

Most "AI fantasy assistant" projects are a thin wrapper around a general LLM: you ask it who to
start, it confidently makes up an answer. This project inverts that. The number comes from a model
trained on historical performance; the LLM's only job is to turn that number — plus retrieved
current context (injuries, news, matchup) — into a readable recommendation.

That separation is the whole point:

- **Predictive accuracy** is owned by a model trained and evaluated honestly.
- **Explanation quality** is owned by an LLM grounded in retrieved evidence.
- Neither pretends to be the other.

The separation is enforced structurally, not just by convention:

1. `/recommend` computes both projections **before** retrieval or narration run.
2. The retriever returns only text — it has no access to the model.
3. The narration is checked against the projections before it is returned. If the
   LLM quotes a number we didn't give it, the response is flagged
   (`narration_grounded: false`) and the UI warns on it.

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# 1. Train (downloads nflverse data on first run, ~1 min of training on CPU)
.venv/bin/python model/train.py

# 2. Serve — model in-process, no database or API key needed
.venv/bin/uvicorn api.main:app --port 8000

# 3. Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

Run the two services split, the way they run in Kubernetes:

```bash
.venv/bin/uvicorn model_service.main:app --port 8001          # model service
MODEL_SERVICE_URL=http://localhost:8001 \
  .venv/bin/uvicorn api.main:app --port 8000                  # gateway
```

Or the whole stack in containers, including Postgres+pgvector:

```bash
docker compose -f infra/docker/compose.yaml up --build
```

Tests: `.venv/bin/python -m pytest tests/ -q`

### Configuration

Everything is optional — the service runs fully without any of it.

| Variable | Unset behaviour |
|---|---|
| `MODEL_SERVICE_URL` | Gateway loads the model in-process instead of calling the model service |
| `ANTHROPIC_API_KEY` | Narration falls back to a deterministic template |
| `DATABASE_URL` | Retrieval falls back to an in-process vector index |

---

## Architecture

### Runtime request path

```
User
  │
  ▼
Next.js frontend  ──►  FastAPI gateway
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       Projection model               RAG retriever
       (trained PyTorch)              (pgvector: news + injuries)
              │  ▲                            │  ▲
              │  │ features                   │  │ embedded news
        Postgres feature store          pgvector news store
              │                               │
              └───────────────┬───────────────┘
                              ▼
                        LLM narrator
                     (explains the pick)
                              │
                              ▼
                     response → frontend
```

The gateway fans out to two independent services. The **projection model** reads engineered
features and returns projected fantasy points. The **RAG retriever** pulls relevant
recent news/injury snippets from the vector store. Both results are handed to the **LLM narrator**,
which writes the recommendation. The model produces the number; the LLM never does.

### Deployment / CI-CD

```
git push ──► GitHub Actions ──► Container registry
             (test, train,         │  deploy
              build, push)         ▼
                         ┌──────────────────────────────────────────┐
                         │            Kubernetes cluster            │
                         │  Ingress ─► Frontend ─► API ─► Model svc │
                         └──────────────────────────────────────────┘
```

The **model service is its own Kubernetes deployment** — it loads weights, wants more memory, and
scales independently of web traffic. That isolation is a deliberate engineering choice, not a
buzzword: inference and the frontend scale on different axes.

The split has a measurable payoff rather than being architectural theatre.
Dependencies are declared per service (`requirements-model.txt` /
`requirements-api.txt`), so the gateway installs no `torch`, `pandas`, or
`pyarrow`, and the model service installs CPU-only torch wheels instead of
dragging in several GB of unused CUDA runtime:

| Image | Size | Starts in |
|---|---:|---|
| `frontend` | 342 MB | seconds |
| `api` (no torch) | 623 MB | seconds |
| `model` (CPU torch + weights) | 1.88 GB | ~1–2 min (weights + feature build) |

The gateway can therefore scale out cheaply on web traffic while the expensive,
slow-starting model service scales on projection volume.

---

## Tech stack

- **Model:** Python, PyTorch, scikit-learn (metrics), nflverse data
- **Serving:** FastAPI, Uvicorn
- **Data:** PostgreSQL with the `pgvector` extension (features + vector store in one instance)
- **RAG + LLM:** hashing-vectorizer embeddings + Claude (`claude-opus-5`) for narration
- **Frontend:** Next.js / TypeScript (one start/sit comparison view)
- **Infra:** Docker, Kubernetes (`kind` for local), GitHub Actions

---

## The model

### Data
Weekly player data from the nflverse release files, 2016–2024, regular season,
QB/RB/WR/TE. Target is PPR fantasy points. Data is fetched over HTTPS and cached
to parquet — `nfl_data_py` pins `pandas<2` and can't coexist with a current
torch/pandas stack, so `model/data.py` reads the same upstream files directly.

### Features
- Rolling last-N-game averages (points, targets, carries, target share)
- Usage: target share (`wopr`-style share metrics from the weekly feed)
- Opponent defensive strength vs. the player's position
- Home/away, rest days, week number, position

**Not included, and worth being straight about:** snap share and red-zone touches
are in the README's original feature wish-list but are not implemented. Snap
counts live in a separate nflverse feed keyed by a different player ID, and
red-zone touches require play-by-play data — both are real work rather than a
one-line addition, and claiming them without implementing them would be exactly
the kind of thing this project is built to avoid.

### Architecture
An LSTM reads the player's last 6 game lines; the final hidden state is
concatenated with pre-kickoff context (matchup, rest, role) and passed through a
small MLP head, with a softplus output since fantasy points are floored near
zero. Deliberately small (~60k parameters) — 34k training player-weeks does not
support anything larger, and an oversized model would only overfit and make the
honest evaluation look worse. Loss is L1, matching the headline MAE metric and
less dominated by the long right tail than MSE.

### Evaluation
- **Split by time, not randomly.** Train ≤2022, validate 2023, test 2024.
- **Baseline:** "project = last-3-game average".
- **Metrics:** MAE / RMSE, plus start/sit accuracy on within-week pairs.
- The test season is scored exactly once, after model selection is finished.

Leakage guarantees are **tested**, not asserted — `tests/test_leakage.py` pins:

| Guarantee | Test |
|---|---|
| A week's own box score is never in its input window | `test_target_never_appears_in_its_own_window` |
| Every window value comes from a strictly earlier game | `test_window_contains_only_strictly_prior_games` |
| Rolling features exclude the current week | `test_rolling_features_exclude_the_current_week` |
| Opponent ratings exclude the week being predicted | `test_opponent_strength_excludes_the_current_week` |
| The split is temporal and non-overlapping | `test_split_is_temporal_not_random` |
| The baseline really is "last 3 games" | `test_baseline_is_the_documented_naive_rule` |

These were verified to fail when a leak is deliberately injected — a leakage test
that can't fail is worse than none, because it buys false confidence.

Feature scaling is fit on the training split only, and CI fails the build if the
model stops beating the baseline.

---

## RAG layer

Injury/news snippets are embedded with a hashing vectorizer (deterministic,
fixed-dimension, no fitting step and no API round-trip) and retrieved by cosine
similarity. `DATABASE_URL` switches the backend to pgvector; otherwise an
in-process index serves the same interface.

Two details that matter:

- The player's **name and team are part of the embedded document**, not just the
  snippet body. They're the highest-signal terms for a start/sit query, and
  leaving them out retrieves plausible-looking but wrong-player news.
- Retrieval runs **one query per player** rather than one merged query. A
  combined "Player A Player B injury" query dilutes both names and reliably
  returns neither player's news.

Retrieval feeds *context* to the narrator; it never changes the projection.

> ⚠️ **`data/news/seed_news.json` is a synthetic demo fixture, not real
> reporting.** Every snippet was hand-written to exercise the retrieval path and
> is labelled `SYNTHETIC-DEMO` so a fabricated snippet can't be mistaken for a
> real report. Real player names are used only so retrieval matches player IDs
> in the feature store. Point the ingest at an actual news feed to run for real.

## LLM narration

Input: the model's projections + retrieved snippets + the two players. Output: a
short "start A over B because…" explanation. The system prompt forbids inventing
or adjusting a number, forbids adding injury/matchup claims not present in the
retrieved snippets, and requires surfacing tension when a snippet cuts against
the projection. The response is then verified to quote the projections we
supplied; a mismatch sets `narration_grounded: false`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status; `503` + `degraded` when the model service is unreachable |
| `POST /project` | The model's number. No retrieval, no LLM. |
| `POST /recommend` | Projection + retrieval + narration |

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"player_a": "Mark Andrews", "player_b": "Brock Bowers"}'
```

## Frontend

One screen: a two-player start/sit comparison showing both projections against
the naive baseline, the margin and a confidence band, the retrieved snippets with
similarity scores, and the LLM's reasoning. Because the demo runs on a completed
week, the **actual** result is shown next to each projection — including when the
model was wrong.

---

## Infra

- **Docker:** one image per service, each with its own requirements file; all
  three run as a non-root user with `readOnlyRootFilesystem` in Kubernetes.
  `infra/docker/compose.yaml` brings the stack up locally with Postgres.
- **Kubernetes:** `infra/k8s/` — Deployment + Service per component, plus an
  Ingress. The model service is its own Deployment with larger memory requests
  and a `startupProbe` sized for slow weight loading (so a genuinely wedged pod
  is still restarted promptly by a tight liveness probe). The model service is
  not exposed through the Ingress — it's cluster-internal only.
- **CI/CD:** `.github/workflows/ci.yml` — test → validate manifests → train →
  build/push images → deploy. Note this lives at the repo root, not under
  `infra/`: GitHub only reads workflows from `.github/workflows/` at the root.

Local cluster:

```bash
kind create cluster --config infra/k8s/kind-cluster.yaml
kubectl apply -f infra/k8s/          # after pushing images and setting the tags
```

---

## Repo structure

```
├── model/
│   ├── data.py            # nflverse pull + caching
│   ├── features.py        # feature engineering (every rolling stat shifted)
│   ├── dataset.py         # sequence windowing, temporal split, scaling
│   ├── net.py             # the LSTM + head
│   ├── train.py           # training loop, early stopping, writes results.json
│   ├── evaluate.py        # baselines + metrics (the credibility)
│   └── artifacts/         # trained weights + committed results.json
├── model_service/         # projection-only service (its own deployment)
├── api/                   # gateway: retrieval, narration, projection client
├── frontend/              # Next.js start/sit comparison view
├── tests/                 # leakage suite + service contract tests
├── scripts/load_features.py  # load features + news into Postgres/pgvector
├── data/news/             # synthetic news fixture for the RAG demo
└── infra/
    ├── docker/            # one Dockerfile per service + compose stack
    ├── k8s/               # Deployments, Services, Ingress, kind config
    └── sql/               # Postgres + pgvector schema
```

---

## Known gaps

Stated plainly rather than left for someone to discover:

- **Snap share and red-zone touches are not implemented** (see Features above).
- **The demo serves completed weeks.** Projecting a genuinely upcoming week needs
  the current week's schedule row and a live injury feed; the feature and serving
  code paths are the same, but that wiring isn't done.
- **Postgres is schema-and-loader-ready, not the default path.** The API runs on
  cached parquet plus an in-memory index unless `DATABASE_URL` is set.
- **The Kubernetes manifests are schema-validated, not cluster-tested.** They
  pass strict validation against the Kubernetes 1.31 schema, and all three
  images were built and run together over a Docker network (gateway → model
  service → frontend, verified end to end), but no cluster was available here
  to run an actual rollout.
- **The CI pipeline has not been executed.** It is written against this repo's
  layout but has never run on GitHub Actions, so treat the badge as aspirational
  until the first green build.
- **The live LLM narration path has not been executed.** No `ANTHROPIC_API_KEY`
  was available, so every run so far used the deterministic template fallback.
  The Anthropic call in `api/llm.py` is written against the current SDK
  (including refusal handling and server-side fallbacks) but is unverified
  against the real API; the grounding check around it *is* unit-tested.
- **A single trained seed.** Reported numbers come from one run (seed 17). A
  seed sweep would put error bars on the 8%.

---

## A note to my future self

The infra is the frame, not the painting. The first interview question for an AI/ML role will be
*"how did you evaluate the model,"* not *"why Kubernetes."* The model eval is the most polished
thing in the repo and the README leads with it. The k8s and CI/CD make you also look like someone
who ships — they can't substitute for the ML depth that's the entire reason this exists.
