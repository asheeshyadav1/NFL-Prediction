# Fantasy Football Toolkit — Projections That Explain Themselves

## What this is

A weekly NFL fantasy-points projection engine where a **trained sequence model** makes
the predictions and an **LLM only explains them**, grounded in retrieved injury and news
context. On a held-out 2024 season the model beats a naive last-3-game average by 8.0%
MAE (4.519 vs 4.912) under a leakage-safe, time-based evaluation whose guarantees are
unit-tested rather than asserted. The separation is structural: the projection is
computed before retrieval or narration run, the retriever has no access to the model,
and the narration is verified to quote the numbers it was given.

---

## Quick Start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# 1. Train (downloads nflverse data on first run, ~1 min on CPU)
.venv/bin/python model/train.py

# 2. Serve — model in-process, no database or API key needed
.venv/bin/uvicorn api.main:app --port 8000

# 3. Frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

Tests: `.venv/bin/python -m pytest tests/ -q`

Run the two services split, the way they run in Kubernetes:

```bash
.venv/bin/uvicorn model_service.main:app --port 8001          # model service
MODEL_SERVICE_URL=http://localhost:8001 \
  .venv/bin/uvicorn api.main:app --port 8000                  # gateway
```

Or the whole stack in containers, including Postgres + pgvector:

```bash
docker compose -f infra/docker/compose.yaml up --build
```

Try it from the command line:

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"player_a": "Mark Andrews", "player_b": "Brock Bowers"}'
```

### Configuration

Everything is optional — the service runs fully without any of it.

| Variable | Unset behaviour |
|---|---|
| `MODEL_SERVICE_URL` | Gateway loads the model in-process instead of calling the model service |
| `ANTHROPIC_API_KEY` | Narration falls back to a deterministic template |
| `DATABASE_URL` | Retrieval falls back to an in-process vector index |
| `API_URL` | Frontend proxies `/api/*` to `http://localhost:8000` |
| `ALLOWED_ORIGINS` | Gateway allows CORS from `http://localhost:3000` |

---

## System Architecture

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

The gateway fans out to two independent services. The **projection model** reads
engineered features and returns projected fantasy points. The **RAG retriever** pulls
recent news/injury snippets from the vector store. Both results are handed to the **LLM
narrator**, which writes the recommendation. The model produces the number; the LLM
never does.

That separation is enforced structurally, not by convention:

1. `/recommend` computes both projections **before** retrieval or narration run.
2. The retriever returns only text — it has no access to the model.
3. The narration is checked against the projections before it is returned. If the LLM
   quotes a number we didn't give it, the response is flagged
   (`narration_grounded: false`) and the UI warns on it.

### Services

| Service | Role | Notes |
|---|---|---|
| `frontend` | Next.js start/sit comparison view | Calls same-origin `/api/*` |
| `api` | Gateway: retrieval, narration, projection client | No torch — scales cheaply on web traffic |
| `model_service` | Projections only | Owns the weights and feature store |

The browser always calls same-origin `/api/*`. In Kubernetes the Ingress routes that
prefix straight to the gateway; elsewhere `frontend/app/api/[...path]/route.ts` proxies
it, reading `API_URL` **per request**. This is deliberately a route handler rather than
a `next.config.ts` rewrite or `env` block — Next resolves both of those at build time,
which bakes an address into the image and silently ignores what the deployment sets.

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status; `503` + `degraded` when the model service is unreachable |
| `POST /project` | The model's number. No retrieval, no LLM. |
| `POST /recommend` | Projection + retrieval + narration |

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

The **model service is its own Kubernetes deployment** — it loads weights, wants more
memory, and scales independently of web traffic. Dependencies are declared per service
(`requirements-model.txt` / `requirements-api.txt`), so the gateway installs no `torch`,
`pandas`, or `pyarrow`, and the model service installs CPU-only torch wheels instead of
several GB of unused CUDA runtime:

| Image | Size | Starts in |
|---|---:|---|
| `frontend` | 342 MB | seconds |
| `api` (no torch) | 623 MB | seconds |
| `model` (CPU torch + weights) | 1.88 GB | ~1–2 min (weights + feature build) |

All three run as a non-root user with `readOnlyRootFilesystem`. The model service is
**not** exposed through the Ingress — it is cluster-internal only. CI lives at
`.github/workflows/ci.yml` (test → validate manifests → train → build/push → deploy) and
fails the build if the model stops beating the naive baseline.

Local cluster:

```bash
kind create cluster --config infra/k8s/kind-cluster.yaml
kubectl apply -f infra/k8s/          # after pushing images and setting the tags
```

### Tech stack

- **Model:** Python, PyTorch, scikit-learn, nflverse data
- **Serving:** FastAPI, Uvicorn
- **Data:** PostgreSQL with `pgvector` (features + vector store in one instance)
- **RAG + LLM:** hashing-vectorizer embeddings + Claude (`claude-opus-5`) for narration
- **Frontend:** Next.js / TypeScript
- **Infra:** Docker, Kubernetes (`kind` for local), GitHub Actions

### Repo structure

```
├── model/
│   ├── data.py            # nflverse pull + caching
│   ├── features.py        # feature engineering (every rolling stat shifted)
│   ├── dataset.py         # sequence windowing, temporal split, scaling
│   ├── net.py             # the LSTM + head
│   ├── train.py           # training loop, early stopping, writes results.json
│   ├── evaluate.py        # baselines + metrics
│   └── artifacts/         # trained weights + committed results.json
├── model_service/         # projection-only service (its own deployment)
├── api/                   # gateway: retrieval, narration, projection client
├── frontend/              # Next.js start/sit view + /api/* proxy route
├── tests/                 # leakage suite + service contract tests
├── scripts/load_features.py  # load features + news into Postgres/pgvector
├── data/news/             # synthetic news fixture for the RAG demo
└── infra/
    ├── docker/            # one Dockerfile per service + compose stack
    ├── k8s/               # Deployments, Services, Ingress, kind config
    └── sql/               # Postgres + pgvector schema
```

> ⚠️ `data/news/seed_news.json` is a **synthetic demo fixture**, not real reporting.
> Every snippet is hand-written and labelled `SYNTHETIC-DEMO` so it cannot be mistaken
> for a real report. Point the ingest at an actual news feed to run for real.
