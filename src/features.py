"""Feature engineering.

Turns a raw, source-mapped queue frame (columns == ``config.FEATURE_COLUMNS``)
into the model-ready feature matrix. The output schema is pinned by
``FeatureSchema`` so the scorer and tests agree on exactly what a row looks like.

Day-of implementation:
  * derive ``queue_year`` and ``queue_age_days`` from ``queue_date``
  * bucket ``capacity_mw`` into ``size_bucket`` via ``config.SIZE_BUCKETS``
  * categorical encoding (leave raw strings here; XGBScorer owns encoding)
  * NA handling for missing counties / capacities
Today this is a typed stub that returns an empty frame with the right columns.
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #


class FeatureSchema(TypedDict):
    """One model-ready row. Keys are the canonical feature names.

    This is the single source of truth for the shape the scorer consumes.
    Identifier columns (``queue_id``) travel alongside for joining results back
    to projects but are never fed to the model.
    """

    queue_id: str
    capacity_mw: float
    generation_type: str
    county: str
    queue_year: int
    queue_age_days: int
    size_bucket: str


# Columns present in a built feature frame: model features + the id for joins.
FEATURE_FRAME_COLUMNS: list[str] = ["queue_id", *config.MODEL_FEATURES]

# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
# These columns encode the outcome (directly or by proxy) and MUST NOT appear in
# any feature frame. build_features drops them defensively even if a raw source
# leaks one in. Enforced by tests.
LEAKAGE_BANNED_COLUMNS: list[str] = [
    "withdrawal_date",       # only set for projects that failed
    "ia_date",               # signing date == the target itself
    "actual_cod",            # commercial operation implies success
    "terminal_status",       # "Withdrawn"/"Operational" is the label
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model-ready feature frame from a source-mapped queue frame.

    Args:
        df: rows with columns == ``config.FEATURE_COLUMNS`` (output of the
            ingest/mapping step).

    Returns:
        A frame with columns == ``FEATURE_FRAME_COLUMNS``, one row per project,
        with no leakage columns present.

    Stub behavior: returns an empty, correctly-typed frame. Day-of, this fills
    in the derivations listed in the module docstring.
    """
    # TODO(day-of): real derivations. For now return the empty typed skeleton so
    # everything downstream can be wired and tested against the right columns.
    empty = pd.DataFrame(columns=FEATURE_FRAME_COLUMNS)
    return _drop_leakage(empty)


def _drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Defensively remove any leakage-banned column that snuck into ``df``."""
    present = [c for c in LEAKAGE_BANNED_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df
