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

Nothing auto-loads `.env`, so export it before running (or add `python-dotenv`):

```bash
export $(grep -v '^#' .env | xargs)
```

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

Committed: source, `README.md`, `SETUP.md`, `requirements.txt`, `.env.example`,
`.gitkeep`s, `conftest.py`, `.gitignore`.

Never committed (gitignored): `.env`, `data/raw/*.xlsx`, `data/snapshots/*.parquet`,
`.venv/`, `.idea/`, `__pycache__/`.
