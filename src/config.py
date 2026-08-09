"""Central configuration for QueueScore.

Every constant, path, and column mapping lives here so the rest of the codebase
never hardcodes a path or a magic column name. Two mapping dicts translate the
two raw data sources into the single internal feature schema defined in
``features.py``.

Day-of TODO: fill the LBNL mapping values once the "Queued Up" codebook is open
(the LBNL column names differ by data-release year, so they cannot be guessed).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# Load a local .env (gitignored) so ANTHROPIC_API_KEY is picked up automatically.
# Optional: if python-dotenv isn't installed or there's no .env, this is a no-op.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
RAW_DIR: Path = DATA_DIR / "raw"
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Filename used by ingest.save_snapshot / load_snapshot for the cached queue.
ERCOT_SNAPSHOT_NAME: str = "ercot_queue_latest.parquet"

# Camille's shared-feature ERCOT completion model (P(reach signed IA)).
MODEL_BUNDLE_NAME: str = "queuescore_ercot_shared.joblib"
MODEL_BUNDLE_PATH: Path = MODELS_DIR / MODEL_BUNDLE_NAME

# --------------------------------------------------------------------------- #
# Internal feature schema
# --------------------------------------------------------------------------- #
# The canonical model-facing column names. Both raw sources are mapped onto
# these before anything downstream touches the data. Keep this list and the
# TypedDict in features.py in lock-step.
FEATURE_COLUMNS: list[str] = [
    "queue_id",
    "capacity_mw",
    "queue_date",
    "county",
    "state",
    "generation_type",
    "proposed_completion_date",
    "status",
]

# Columns the model is allowed to train on (derived + raw). Everything not here
# is either an identifier, a raw date we derive from, or leakage.
MODEL_FEATURES: list[str] = [
    "capacity_mw",
    "generation_type",
    "county",
    "queue_year",
    "queue_age_days",
    "size_bucket",
]

# --------------------------------------------------------------------------- #
# Column mappings: raw source -> internal schema
# --------------------------------------------------------------------------- #
# gridstatus Ercot().get_interconnection_queue() returns these column names.
# Values are the internal schema names above. These keys are the live, verified
# gridstatus ERCOT columns.
GRIDSTATUS_TO_MODEL: dict[str, str] = {
    "Queue ID": "queue_id",
    "Capacity (MW)": "capacity_mw",
    "Queue Date": "queue_date",
    "County": "county",
    "State": "state",
    "Generation Type": "generation_type",
    "Proposed Completion Date": "proposed_completion_date",
    "Status": "status",
}

# LBNL "Queued Up" historical dataset -> internal schema.
# Resolved from the 2026 Data File codebook (sheet "04. Data Codebook"); see
# DATA.md. Read sheet "03. Complete Queue Data" with header=1 (row 0 is a
# banner). Note: q_id is unique only combined with `entity`, and `generation_type`
# comes from `type_1` (verbose live-ERCOT types must be normalized to match).
LBNL_TO_MODEL: dict[str, str] = {
    "q_id": "queue_id",
    "mw_1": "capacity_mw",
    "q_date": "queue_date",
    "county": "county",
    "state": "state",
    "type_1": "generation_type",
    "prop_date": "proposed_completion_date",
    "q_status": "status",
}

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
# The label the model predicts. A project is a positive example if it reached a
# signed Interconnection Agreement (IA) in the LBNL historical record. Projects
# still active (no terminal outcome) are excluded from training, not labeled 0.
TARGET_DEFINITION: str = (
    "binary: 1 if the project reached a signed Interconnection Agreement (IA) "
    "in the LBNL historical record, else 0. Still-active projects with no "
    "terminal outcome are dropped from the training set, not labeled negative."
)

# --------------------------------------------------------------------------- #
# Feature engineering constants
# --------------------------------------------------------------------------- #
# MW thresholds for size_bucket. (label, upper_bound_exclusive_mw)
SIZE_BUCKETS: list[tuple[str, float]] = [
    ("small", 20.0),
    ("medium", 100.0),
    ("large", 500.0),
    ("xlarge", float("inf")),
]

# --------------------------------------------------------------------------- #
# Runtime flags
# --------------------------------------------------------------------------- #
# When True, explain.py returns canned text instead of calling the Anthropic
# API. Set in `.env` as DRY_RUN=true|false (default true — no key/network needed).
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DRY_RUN: bool = _env_bool("DRY_RUN", False)

# Models are configurable via .env without code edits; defaults below.
# ANTHROPIC_MODEL: user-facing verdicts/Q&A (quality matters, low volume).
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

# MATCH_MODEL: high-volume match adjudication in resolve.py — simple yes/no
# classification, so Haiku ($1/$5 per MTok vs $5/$25) is plenty.
MATCH_MODEL: str = os.getenv("MATCH_MODEL", "claude-haiku-4-5")

# Cap on ambiguous pairs sent to Claude per run (runaway guardrail; verdicts
# are disk-cached so reruns cost nothing). The full backlog is ~800 pairs ≈
# 50 Haiku batches ≈ well under $0.10, so the default covers all of it.
# Override via .env to throttle: MATCH_MAX_CLAUDE_PAIRS=60
MATCH_MAX_CLAUDE_PAIRS: int = int(os.getenv("MATCH_MAX_CLAUDE_PAIRS", "1000"))

# Seed for DummyScorer so leaderboard output is stable across reruns.
RANDOM_SEED: int = 42
