"""Tests for stage inference (all offline, pure rules)."""

from __future__ import annotations

import pandas as pd

from src import stage
from src.sources import RECORD_COLUMNS


def _rec(source, source_id, status="", signal="", match_id=""):
    row = {c: "" for c in RECORD_COLUMNS}
    row.update(
        source=source, source_id=source_id, status=status, stage_signal=signal,
        county="Wise", county_key="WISE", lat=None, lon=None,
        capacity_mw=None, record_date=pd.NaT,
    )
    row["match_id"] = match_id
    row["match_reason"] = ""
    return row


def _annotate(rows):
    return stage.annotate_stages(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# ERCOT ladder
# --------------------------------------------------------------------------- #
def test_ercot_ia_signed_is_top_stage():
    df = _annotate([_rec("ercot", "E1", signal="SS Completed, FIS Completed, IA (IA signed 2020-08-07)")])
    r = df.iloc[0]
    assert r["stage"] == "Grid agreement signed"
    assert r["stage_rank"] == 3
    assert r["stage_confidence"] == "high"
    assert "IA signed" in r["stage_evidence"]


def test_ercot_ladder_ordering():
    df = _annotate([
        _rec("ercot", "A", signal="SS Completed, FIS Completed, No IA"),
        _rec("ercot", "B", signal="SS Completed, FIS Started, No IA"),
        _rec("ercot", "C", signal="SS Completed, No IA"),
        _rec("ercot", "D", signal=""),
    ])
    assert df["stage_rank"].tolist() == [2, 1, 1, 0]
    assert df.iloc[3]["stage"] == "Early planning"
    assert df.iloc[3]["stage_confidence"] == "low"


# --------------------------------------------------------------------------- #
# TCEQ ladder
# --------------------------------------------------------------------------- #
def test_tceq_permit_stages():
    df = _annotate([
        _rec("tceq", "T1", status="NEW APPLICATION"),
        _rec("tceq", "T2", status="ISSUED PERMIT"),
    ])
    assert df.iloc[0]["stage"] == "Permit application filed"
    assert df.iloc[1]["stage"] == "Permit issued"
    assert (df["stage_confidence"] == "high").all()


# --------------------------------------------------------------------------- #
# Cross-source corroboration
# --------------------------------------------------------------------------- #
def test_matched_project_gets_confidence_bump_and_joint_evidence():
    df = _annotate([
        _rec("ercot", "E1", signal="SS Completed, FIS Started, No IA", match_id="T1"),
        _rec("tceq", "T1", status="ISSUED PERMIT", match_id="E1"),
    ])
    e = df[df.source_id == "E1"].iloc[0]
    assert e["stage"] == "Engineering studies"     # permit never changes grid stage
    assert e["stage_confidence"] == "high"          # bumped from high stays... started=high
    assert "linked TCEQ record T1" in e["stage_evidence"]


def test_confidence_bump_from_medium():
    df = _annotate([
        _rec("ercot", "E1", signal="SS Completed, No IA", match_id="T1"),
        _rec("tceq", "T1", status="ISSUED PERMIT", match_id="E1"),
    ])
    assert df[df.source_id == "E1"].iloc[0]["stage_confidence"] == "high"  # medium -> high


def test_unmatched_record_no_bump():
    df = _annotate([_rec("ercot", "E1", signal="SS Completed, No IA")])
    assert df.iloc[0]["stage_confidence"] == "medium"
    assert "linked" not in df.iloc[0]["stage_evidence"]
