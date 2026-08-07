"""Central configuration for QueueScore.

Every constant, path, and column mapping lives here so the rest of the codebase
never hardcodes a path or a magic column name. Two mapping dicts translate the
two raw data sources into the single internal feature schema defined in
``features.py``.

Day-of TODO: fill the LBNL mapping values once the "Queued Up" codebook is open
(the LBNL column names differ by data-release year, so they cannot be guessed).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"

# Filename used by ingest.save_snapshot / load_snapshot for the cached queue.
ERCOT_SNAPSHOT_NAME: str = "ercot_queue_latest.parquet"

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
# TODO(day-of): fill keys once the codebook is open. LBNL column headers vary
# by release year; do not guess them. Values (RHS) are stable — only the raw
# LBNL header strings on the LHS are unknown.
LBNL_TO_MODEL: dict[str, str] = {
    "TODO_lbnl_queue_id_col": "queue_id",
    "TODO_lbnl_capacity_mw_col": "capacity_mw",
    "TODO_lbnl_queue_date_col": "queue_date",
    "TODO_lbnl_county_col": "county",
    "TODO_lbnl_state_col": "state",
    "TODO_lbnl_generation_type_col": "generation_type",
    "TODO_lbnl_proposed_cod_col": "proposed_completion_date",
    "TODO_lbnl_status_col": "status",
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
# API. Defaults on so the app runs with no key and no network.
DRY_RUN: bool = True

# Model used by explain.py when DRY_RUN is False.
ANTHROPIC_MODEL: str = "claude-opus-4-8"

# Seed for DummyScorer so leaderboard output is stable across reruns.
RANDOM_SEED: int = 42
