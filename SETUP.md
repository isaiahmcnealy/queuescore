# Setup

Everything below is verified working on **Python 3.14.2** (macOS). 3.11–3.13 also
work; if a package fails to install on your machine, drop to 3.11 or 3.12.

## Quickstart

```bash
git clone <repo-url> && cd queuescore
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

The first run pulls both live sources (ERCOT via gridstatus, TCEQ via their
public API) and caches snapshots under `data/snapshots/` — after that the app
runs **fully offline**. No API key is needed for the map, stitched links,
stages, or model scores (the trained bundle in `models/` and the match table
are committed); Claude-powered features skip gracefully without one.

`./run.sh start|stop|restart|status|logs` wraps the same server on port 8501.

Run the tests to confirm the install:

```bash
pytest        # expect: 29 passed
```

## Anthropic API key (briefs, Q&A, match adjudication)

The Claude-powered features need a key (`DRY_RUN=true` in `.env` disables all
Anthropic calls for fully offline work).

```bash
cp .env.example .env      # then paste your real ANTHROPIC_API_KEY
```

Nothing auto-loads `.env` unless `python-dotenv` is installed (it is in
`requirements.txt`). `src/config.py` calls `load_dotenv` on import, so a repo-root
`.env` is enough for local runs and for the arya systemd service once
`EnvironmentFile=` is enabled in [`deploy/queuescore.service`](deploy/queuescore.service).

## Production

Hosted on **arya** behind Cloudflare Tunnel at `queuescore.tech`. See
[DEPLOY.md](DEPLOY.md) (systemd, self-hosted runner, DNS).

## Snapshots & refreshing data

Source snapshots seed automatically on the app's first run and refresh when
you click the timestamp pill in the hero bar (each source falls back to its
snapshot if the live pull fails). To seed the model-side ERCOT queue snapshot
from the command line:

```bash
python -c "from src.ingest import fetch_ercot_queue; print(fetch_ercot_queue().shape)"
```

## Training data (only needed to retrain the model)

The trained model bundle is committed in `models/`, so normal runs need
nothing extra. To retrain, download the LBNL **"Queued Up"** dataset
(CC BY 4.0) and drop the xlsx into `data/raw/` — it is **not** in the repo
(`data/raw/*.xlsx` is gitignored; shared out-of-band).

## What's safe in git

Committed: source, docs, `requirements.txt`, `.env.example`, `deploy/`,
`scripts/`, workflows, `conftest.py`, `.gitignore`, the trained model bundle
(`models/`), and `data/snapshots/matches.parquet` (the cross-source match
table — committed deliberately so stitched links work out of the box).

Never committed (gitignored): `.env`, `data/raw/*.xlsx`, all other
`data/snapshots/` files (source parquets, verdict/brief caches), `.venv/`,
`.idea/`, `__pycache__/`.
