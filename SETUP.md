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

The app runs immediately — **no network, no API key**. It scores demo projects
with `DummyScorer`, and all Anthropic calls are gated behind `DRY_RUN` (on by
default in `src/config.py`).

Run the tests to confirm the install:

```bash
pytest        # expect: 8 passed
```

## Optional: live explanations (day-of)

The app only needs a key when you flip `DRY_RUN = False` in `src/config.py`.

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

## Optional: offline live-queue demo

The "Pull live queue" button falls back to a cached snapshot when offline. Seed
that cache once while you have network:

```bash
python -c "from src.ingest import fetch_ercot_queue; print(fetch_ercot_queue().shape)"
```

The snapshot lands in `data/snapshots/` (gitignored). After this, the button
works with no network.

## Day-of: training data

Download the LBNL **"Queued Up"** dataset (CC BY 4.0) and drop the xlsx into
`data/raw/`. It is **not** in the repo — it's shared out-of-band. `data/raw/*.xlsx`
is gitignored so it never gets committed.

## What's safe in git

Committed: source, `README.md`, `SETUP.md`, `DEPLOY.md`, `DATA.md`,
`requirements.txt`, `.env.example`, `deploy/`, `scripts/`, workflows,
`.gitkeep`s, `conftest.py`, `.gitignore`.

Never committed (gitignored): `.env`, `data/raw/*.xlsx`, `data/snapshots/*.parquet`,
`.venv/`, `.idea/`, `__pycache__/`.
