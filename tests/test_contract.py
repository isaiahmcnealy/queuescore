"""Contract tests: the guarantees the rest of the app relies on.

Covers the scorer output contract, the snapshot cache round-trip, and that the
column mappings actually cover the required internal schema. These run today
against the wired stubs and should keep passing as real implementations land.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, ingest
from src.features import FEATURE_FRAME_COLUMNS
from src.scorer import BaselineScorer, DummyScorer, ScoreResult


@pytest.fixture
def sample_features() -> pd.DataFrame:
    rows = [
        ("ERC-1", 150.0, "Solar", "Harris", 2022, 640, "large"),
        ("ERC-2", 20.0, "Battery", "Travis", 2023, 275, "small"),
        ("ERC-3", 600.0, "Gas", "Bexar", 2020, 1400, "xlarge"),
    ]
    return pd.DataFrame(rows, columns=FEATURE_FRAME_COLUMNS)


# --------------------------------------------------------------------------- #
# Scorer contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scorer_cls", [DummyScorer, BaselineScorer])
def test_score_shape_and_range(scorer_cls, sample_features):
    result = scorer_cls().score(sample_features)
    assert isinstance(result, ScoreResult)

    n = len(sample_features)
    assert result.probabilities.shape == (n,)
    assert np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))

    assert len(result.attributions) == n
    assert all(isinstance(a, dict) for a in result.attributions)


def test_dummy_scorer_is_seeded(sample_features):
    a = DummyScorer(seed=7).score(sample_features).probabilities
    b = DummyScorer(seed=7).score(sample_features).probabilities
    np.testing.assert_array_equal(a, b)


def test_score_result_length_mismatch_rejected():
    with pytest.raises(ValueError):
        ScoreResult(probabilities=np.array([0.1, 0.2]), attributions=[{}])


# --------------------------------------------------------------------------- #
# Snapshot cache round-trip
# --------------------------------------------------------------------------- #
def test_snapshot_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    df = pd.DataFrame({"queue_id": ["A", "B"], "capacity_mw": [1.0, 2.0]})

    assert not ingest.snapshot_exists()
    ingest.save_snapshot(df)
    assert ingest.snapshot_exists()

    loaded = ingest.load_snapshot()
    pd.testing.assert_frame_equal(loaded, df)


def test_load_snapshot_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        ingest.load_snapshot()


# --------------------------------------------------------------------------- #
# Mapping coverage
# --------------------------------------------------------------------------- #
def test_gridstatus_mapping_covers_schema():
    mapped_targets = set(config.GRIDSTATUS_TO_MODEL.values())
    assert set(config.FEATURE_COLUMNS).issubset(mapped_targets)


def test_lbnl_mapping_covers_schema():
    mapped_targets = set(config.LBNL_TO_MODEL.values())
    assert set(config.FEATURE_COLUMNS).issubset(mapped_targets)
