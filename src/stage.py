"""Stage inference (M4): how far along each project is.

Places every record on the development funnel, with a confidence level and the
evidence (source filings) that justify the call. Pure rules over signals the
sources already carry — no API calls:

ERCOT queue entries (the project spine), from the GIM study phase + IA date:
    Early planning -> Engineering studies -> Studies complete -> Grid agreement signed

TCEQ air permits, from the permit status:
    Permit application filed -> Permit issued

Cross-source matches raise confidence: an ERCOT project whose linked air permit
is issued has independent corroboration that it's really moving.
"""

from __future__ import annotations

import pandas as pd

# Ordered funnel. rank sorts the leaderboard; label is what users read.
STAGES: dict[int, str] = {
    0: "Early planning",
    1: "Engineering studies",
    2: "Studies complete",
    3: "Grid agreement signed",
}
PERMIT_STAGES: dict[str, tuple[int, str]] = {
    # TCEQ pspStatusCd -> (rank, label). Permits are early-stage signals.
    "NEW APPLICATION": (1, "Permit application filed"),
    "ISSUED PERMIT": (2, "Permit issued"),
    "RENEWAL/AMENDMENT": (2, "Permit issued"),
}

_CONF_ORDER = ["low", "medium", "high"]


def _bump(confidence: str) -> str:
    """One step up the confidence ladder (high stays high)."""
    i = _CONF_ORDER.index(confidence)
    return _CONF_ORDER[min(i + 1, len(_CONF_ORDER) - 1)]


def _stage_ercot(signal: str) -> tuple[int, str, str, str]:
    """(rank, label, confidence, evidence) from the ERCOT GIM study phase text."""
    s = (signal or "").lower()
    ev = f"ERCOT study phase: {signal!r}" if signal else "ERCOT queue entry (no study phase reported)"
    if "ia signed" in s:
        return 3, STAGES[3], "high", ev
    if "fis completed" in s:
        return 2, STAGES[2], "high", ev
    if "fis started" in s:
        return 1, STAGES[1], "high", ev
    if "ss completed" in s:
        return 1, STAGES[1], "medium", ev
    if "ss started" in s:
        return 0, STAGES[0], "medium", ev
    return 0, STAGES[0], "low", ev


def _stage_tceq(status: str) -> tuple[int, str, str, str]:
    """(rank, label, confidence, evidence) from the TCEQ permit status."""
    key = (status or "").strip().upper()
    ev = f"TCEQ permit status: {status or 'unknown'}"
    if key in PERMIT_STAGES:
        rank, label = PERMIT_STAGES[key]
        return rank, label, "high", ev
    return 0, "Permitting (status unclear)", "low", ev


def annotate_stages(records: pd.DataFrame) -> pd.DataFrame:
    """Add stage, stage_rank, stage_confidence, stage_evidence to every record.

    Uses ``match_id`` (from resolve.link_records) when present: a matched
    record's evidence cites both filings, and its confidence is bumped one
    level — two independent public sources agreeing beats one.
    """
    out = records.copy()
    has_matches = "match_id" in out.columns
    by_id = out.set_index(["source", "source_id"]) if has_matches else None

    ranks, labels, confs, evs = [], [], [], []
    for _, row in out.iterrows():
        if row["source"] == "ercot":
            rank, label, conf, ev = _stage_ercot(row.get("stage_signal", ""))
        else:
            rank, label, conf, ev = _stage_tceq(row.get("status", ""))

        if has_matches and row.get("match_id"):
            other_source = "tceq" if row["source"] == "ercot" else "ercot"
            try:
                other = by_id.loc[(other_source, row["match_id"])]
                if isinstance(other, pd.DataFrame):  # duplicate ids: take first
                    other = other.iloc[0]
                other_ev = (
                    f"linked {other_source.upper()} record {row['match_id']}: "
                    f"{other.get('status', '') or 'on file'}"
                )
                ev = f"{ev} · {other_ev}"
                conf = _bump(conf)
            except KeyError:
                pass

        ranks.append(rank)
        labels.append(label)
        confs.append(conf)
        evs.append(ev)

    out["stage_rank"] = ranks
    out["stage"] = labels
    out["stage_confidence"] = confs
    out["stage_evidence"] = evs
    return out
