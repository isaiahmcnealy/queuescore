"""Tests for the cross-source matching funnel (all offline)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, resolve
from src.sources import RECORD_COLUMNS


def _record(source, source_id, company, project, county, lat=None, lon=None,
            kind="Gas"):
    return {
        "source": source, "source_id": source_id, "project_name": project,
        "company": company, "county": county.title(), "county_key": county.upper(),
        "lat": lat, "lon": lon, "status": "Active", "stage_signal": "",
        "capacity_mw": 100.0, "kind": kind, "record_date": pd.Timestamp("2024-01-01"),
        "link_id": source_id,
    }


@pytest.fixture
def records():
    return pd.DataFrame(
        [
            _record("ercot", "E1", "Wise County Power Company LLC", "Wise Repower", "Wise"),
            _record("ercot", "E2", "Wolf Hollow II Power LLC", "Wolf Hollow II", "Hood"),
            _record("ercot", "E3", "Totally Different Name", "Mystery Project", "Wise"),
            _record("tceq", "T1", "Wise County Power Company, LLC", "WISE COUNTY PLANT", "Wise", 33.0, -97.7),
            _record("tceq", "T2", "Wolf Hollow I Power, LLC", "WOLF HOLLOW I", "Hood", 32.4, -97.8),
            _record("tceq", "T3", "Far Away Corp", "FAR AWAY SITE", "Webb", 27.5, -99.5),
        ],
        columns=RECORD_COLUMNS,
    )


# --------------------------------------------------------------------------- #
# Name normalization / similarity
# --------------------------------------------------------------------------- #
def test_normalize_strips_legal_suffixes():
    assert resolve.normalize_name("Wise County Power Company, LLC") == "wise county power"
    assert resolve.normalize_name("Vistra Corp.") == "vistra"


def test_normalize_keeps_unit_numerals():
    # Roman numerals are identity for power plants — must NOT be stripped.
    a = resolve.normalize_name("Wolf Hollow II Power LLC")
    b = resolve.normalize_name("Wolf Hollow I Power LLC")
    assert a != b
    assert "ii" in a.split() and "i" in b.split()


def test_similarity_same_company_across_punctuation():
    s = resolve._similarity("Wise County Power Company LLC", "Wise County Power Company, LLC")
    assert s >= resolve.AUTO_MATCH


# --------------------------------------------------------------------------- #
# County gate + funnel
# --------------------------------------------------------------------------- #
def test_candidates_are_county_gated(records):
    cands = resolve.candidate_pairs(records)
    # T3 is in Webb; no ERCOT record there -> never a candidate
    assert not (cands["tceq_id"] == "T3").any()


def test_auto_match_found_and_wolf_hollow_units_not_confused(records):
    # exercise the funnel pieces directly (no snapshot writes, no API)
    cands = resolve.candidate_pairs(records)
    auto = cands[cands["score"] >= resolve.AUTO_MATCH]
    pairs = set(zip(auto["ercot_id"], auto["tceq_id"]))
    assert ("E1", "T1") in pairs            # same company, punctuation differs
    assert ("E2", "T2") not in pairs        # unit II vs unit I must not auto-match


def test_cross_county_rescue_for_orphan_gas_rows():
    # E9 is gas in Ector; there is NO TCEQ record in Ector, but a
    # near-identical name exists in Midland -> rescued, flagged, not auto.
    # E8 is solar in the same spot -> not rescued (no air permit expected).
    recs = pd.DataFrame(
        [
            _record("ercot", "E9", "Desert Power LLC", "Desert Power CC", "Ector"),
            _record("ercot", "E8", "Sunny Solar LLC", "Sunny Solar", "Ector", kind="Solar"),
            _record("tceq", "T9", "Desert Power LLC", "DESERT POWER PLANT", "Midland", 32.0, -102.1),
        ],
        columns=RECORD_COLUMNS,
    )
    cands = resolve.candidate_pairs(recs)
    rescued = cands[(cands["ercot_id"] == "E9") & (cands["tceq_id"] == "T9")]
    assert len(rescued) == 1
    assert bool(rescued.iloc[0]["cross_county"])
    assert not (cands["ercot_id"] == "E8").any()
    # Even at score 1.0 a cross-county pair must go to Claude, never auto.
    auto = cands[(cands["score"] >= resolve.AUTO_MATCH) & ~cands["cross_county"]]
    assert not (auto["ercot_id"] == "E9").any()


def test_candidates_carry_adjudication_signals(records):
    cands = resolve.candidate_pairs(records)
    assert {"ercot_mw", "tceq_mw", "ercot_year", "tceq_year"} <= set(cands.columns)
    row = cands.iloc[0]
    assert row["ercot_mw"] == pytest.approx(100.0)
    assert row["ercot_year"] == "2024"


def test_dry_run_adjudication_skips_api(records, monkeypatch):
    monkeypatch.setattr(config, "DRY_RUN", True)
    cands = resolve.candidate_pairs(records)
    ambiguous = cands[cands["score"] < resolve.AUTO_MATCH]
    judged = resolve.adjudicate(ambiguous)
    assert len(judged) == len(ambiguous)
    assert not judged["verdict"].any()      # conservative: nothing confirmed
    assert (~judged["adjudicated"]).all()


# --------------------------------------------------------------------------- #
# Linking back into the record table
# --------------------------------------------------------------------------- #
def test_link_records_inherits_permit_coordinates(records):
    matches = pd.DataFrame(
        [{"ercot_id": "E1", "tceq_id": "T1", "score": 1.0, "county": "Wise",
          "ercot_company": "", "ercot_project": "", "ercot_kind": "",
          "tceq_company": "", "tceq_project": "", "tceq_kind": "",
          "method": "auto", "reason": "test"}]
    )
    linked = resolve.link_records(records, matches)
    e1 = linked[(linked.source == "ercot") & (linked.source_id == "E1")].iloc[0]
    assert e1["lat"] == pytest.approx(33.0)
    assert e1["match_id"] == "T1"
    t1 = linked[(linked.source == "tceq") & (linked.source_id == "T1")].iloc[0]
    assert t1["match_id"] == "E1"
    # unmatched rows untouched
    e3 = linked[(linked.source == "ercot") & (linked.source_id == "E3")].iloc[0]
    assert e3["match_id"] == ""
