"""Contract tests for the Project Radar ingestion layer (all offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sources import RECORD_COLUMNS
from src.sources import ercot, tceq


# --------------------------------------------------------------------------- #
# TCEQ normalization
# --------------------------------------------------------------------------- #
_TCEQ_RECORD = {
    "aiIdNumber": "90867",
    "aiPermittedName": "PECOS POWER GENERATION",
    "reName": "PECOS POWER",
    "aiIssueToName": "Pecos Power Generation Company, LLC",
    "aiCnty": "REEVES",
    "aiLatDecCoord": 31.39695,
    "aiLongDecCoord": -103.621825,
    "pspStatusCd": "NEW APPLICATION",
    "aiNaicsDesc": "Fossil Fuel Electric Power Generation",
    "aiIssueToBeginDt": "2025-05-05T00:00:00.000",
    "reRegNumber": "RN105809008",
}


def test_tceq_normalize_maps_unified_schema():
    df = tceq._normalize([_TCEQ_RECORD])
    assert list(df.columns) == RECORD_COLUMNS
    row = df.iloc[0]
    assert row["source"] == "tceq"
    assert row["source_id"] == "90867"
    assert row["company"] == "Pecos Power Generation Company, LLC"
    assert row["county"] == "Reeves"
    assert row["county_key"] == "REEVES"
    assert row["lat"] == pytest.approx(31.39695)
    assert row["status"] == "NEW APPLICATION"
    assert row["link_id"] == "RN105809008"


def test_tceq_normalize_dedupes_permits():
    df = tceq._normalize([_TCEQ_RECORD, dict(_TCEQ_RECORD)])
    assert len(df) == 1


# --------------------------------------------------------------------------- #
# ERCOT normalization
# --------------------------------------------------------------------------- #
def _ercot_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Queue ID": ["18INR0009"],
            "Project Name": ["Eagle Pines Gas"],
            "Interconnecting Entity": ["FGE Power"],
            "County": ["Cherokee"],
            "Generation Type": ["Gas - Combined-Cycle"],
            "Capacity (MW)": [1173.5],
            "Queue Date": ["2015-03-30"],
            "Status": ["Active"],
            "GIM Study Phase": ["SS Completed, FIS Started, No IA"],
            "IA Signed": [pd.NaT],
        }
    )


def test_ercot_normalize_maps_unified_schema():
    df = ercot._normalize(_ercot_raw())
    assert list(df.columns) == RECORD_COLUMNS
    row = df.iloc[0]
    assert row["source"] == "ercot"
    assert row["company"] == "FGE Power"
    assert row["county_key"] == "CHEROKEE"
    assert pd.isna(row["lat"])  # no coords until matched downstream
    assert row["capacity_mw"] == pytest.approx(1173.5)
    assert "No IA" in row["stage_signal"]


def test_ercot_stage_signal_includes_ia_date():
    raw = _ercot_raw()
    raw["IA Signed"] = [pd.Timestamp("2020-08-07")]
    row = ercot._normalize(raw).iloc[0]
    assert "IA signed 2020-08-07" in row["stage_signal"]


# --------------------------------------------------------------------------- #
# Snapshot + freshness round-trip (per source, isolated dir)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", [ercot, tceq])
def test_snapshot_and_freshness_round_trip(mod, tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    df = pd.DataFrame([["x"] * len(RECORD_COLUMNS)], columns=RECORD_COLUMNS)

    assert mod.last_updated() is None
    mod.save_snapshot(df)
    assert mod.snapshot_path().exists()
    assert mod.last_updated() is not None
    loaded = mod.load_snapshot()
    assert list(loaded.columns) == RECORD_COLUMNS
