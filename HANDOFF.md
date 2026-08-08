# HANDOFF — session context for QueueScore

Read this first when starting a new working session. Last updated: Aug 8, 2026
(hackathon day). Deeper docs: [CHARTER.md](CHARTER.md) (goal/users),
[ROADMAP.md](ROADMAP.md) (milestones), [SOURCES.md](SOURCES.md) (data recon),
[DATA.md](DATA.md) (schemas).

## What this is
**QueueScore** — Candid Intelligence hackathon entry (Track 1, "Project Radar").
Live origination intelligence for Texas power projects: ERCOT interconnection
queue + TCEQ air permits, stitched into one map/table so deal originators spot
real projects early. Judged on: liveness, addictive presentation, signal
quality, depth on the hard parts (entity matching, stage inference).

## State: WORKING END-TO-END
- **Ingestion** (`src/sources/`): 3,723 live records (1,797 ERCOT via
  gridstatus + 1,926 TCEQ power-gen air permits via their reverse-engineered
  JSON API). Snapshot-cached, offline-safe, per-source freshness stamps.
- **Matching** (`src/resolve.py`): county gate → name similarity → Claude
  (Haiku) adjudication of ambiguous pairs, batched + disk-cached
  (`match_verdicts.json`). 71 confirmed links; ERCOT rows inherit permit
  coordinates. Run: `python -m src.resolve`.
- **Stages** (`src/stage.py`): rules ladder (Early planning → Engineering
  studies → Studies complete → Grid agreement signed; permits: filed → issued)
  with confidence + evidence; matched records get corroboration bumps.
- **App** (`src/app.py`, Streamlit): records table (click-to-select, status
  filter, gas-to-power focus, ℹ️ schema popover), Carto map colored by status
  + satellite site view, detail pane with stage/evidence/linked record,
  **Generate brief** button (7-section origination brief: Verdict/Why now/
  Snapshot/Who/Angle/Evidence/Gaps — disk-cached in `briefs.json`), record Q&A.
- **Tests**: 27 passing, all offline (`pytest`). CI on GitHub Actions (py3.14).

## Conventions & commands
- Run app: `./run.sh start|stop|restart` (port 8501). Venv: `.venv` (py3.14).
- Config via `.env` / `src/config.py`: `DRY_RUN` (false = live API),
  `ANTHROPIC_MODEL` (Opus, verdicts/briefs), `MATCH_MODEL` (Haiku,
  adjudication), `MATCH_MAX_CLAUDE_PAIRS` (cap, 60).
- Git: two people push in parallel (PRs + direct). ALWAYS: check clean tree,
  `git pull --rebase origin main` before pushing; keep `develop-isaiah` = main.
- Design: earthy palette in `src/app.py` PALETTE + `.streamlit/config.toml`
  (cream/sage/olive/terracotta). Plain language everywhere; state honest gaps.

## In flight / next
1. **External model integration**: Camille is training a success model
   (P(reach IA), LBNL data) OUTSIDE the repo. Agreed easiest path: she exports
   one `joblib` Pipeline trained in the repo venv → `models/` → fill
   `XGBScorer.load/score` in `src/scorer.py` (contract: `score(features) ->
   ScoreResult`), add `get_scorer()` ladder (XGB → Baseline → Dummy). Her
   features must be computable live: capacity, type, county, queue year/age
   (`config.MODEL_FEATURES`); needs generation-type normalizer (live ERCOT
   verbose names → LBNL `type_1` vocab).
2. **M7**: README rewrite for judges + 2-minute demo script + "what we'd build
   next" (agentic enrichment: hiring/news signals).
3. Optional polish: second-opinion pass on the 20 Claude matches (Opus,
   ~$0.05); confidence-tier tuning in stage.py (everything reads "high" today).

## Gotchas
- LBNL xlsx + snapshots are gitignored/shared out-of-band (`data/raw/`,
  seeded snapshots exist locally). `matches.parquet` IS committed (demo needs it).
- API account had a $0-credit incident — pipeline degrades gracefully, but if
  Claude calls fail, check console.anthropic.com billing first.
- TCEQ API needs browser User-Agent + `AppName: psp-frontend` header (SOURCES.md).
- The embedded preview browser can't paint WebGL maps reliably — verify maps in
  a real browser.
