"""Cross-source matching (the judged hard part of QueueScore).

Links ERCOT queue entries to TCEQ air permits that belong to the same real
project, in a three-stage funnel — cheap to expensive:

  1. County gate (free): only compare records in the same county.
  2. Name similarity (local): normalized company/project-name scoring.
     Very similar -> auto-match; clearly different -> auto-reject.
  3. Claude adjudication (capped, cached): only the ambiguous middle band.
     Uses the cheap model (config.MATCH_MODEL, Haiku) in batches, and every
     verdict is cached on disk keyed by the pair — reruns cost zero tokens.

Precision over volume by design: ambiguous pairs Claude rejects stay unlinked.

Output: a match table  ercot_id · tceq_id · score · method · verdict · reason.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
AUTO_MATCH = 0.88   # >= this similarity -> matched without Claude
AUTO_REJECT = 0.55  # <  this similarity -> discarded without Claude
BATCH_SIZE = 15     # ambiguous pairs per Claude request (shared prompt overhead)

MATCHES_NAME = "matches.parquet"
VERDICT_CACHE_NAME = "match_verdicts.json"

# Legal/structural suffixes that carry no identity signal. NOTE: roman
# numerals are deliberately NOT stripped — "Wolf Hollow II" vs "Wolf Hollow I"
# are different units, and stripping them caused false 1.0 matches.
_SUFFIXES = {
    "llc", "l.l.c", "inc", "incorporated", "lp", "l.p", "llp", "ltd",
    "limited", "corp", "corporation", "co", "company", "holdings",
    "partners", "partnership",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and legal suffixes ("LLC", "Inc", ...)."""
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


_UNIT_TOKENS = {"i", "ii", "iii", "iv", "v", "vi", "1", "2", "3", "4", "5", "6"}


def _units(normalized: str) -> set[str]:
    """Unit-number tokens (roman or digit) — identity signal for power plants."""
    return {t for t in normalized.split() if t in _UNIT_TOKENS}


def _similarity(a: str, b: str) -> float:
    """Blend of sequence similarity and token overlap on normalized names.

    Names that differ only in unit number ("Wolf Hollow II" vs "Wolf Hollow I")
    score deceptively high on characters, so mismatched units cap the score
    into the ambiguous band — Claude decides those, never the auto-matcher.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb)
    score = max(seq, jaccard)
    if _units(na) != _units(nb):
        score = min(score, AUTO_MATCH - 0.03)
    return score


def pair_score(ercot_row: pd.Series, tceq_row: pd.Series) -> float:
    """Best similarity across company<->company and project/site name fields."""
    return max(
        _similarity(ercot_row["company"], tceq_row["company"]),
        _similarity(ercot_row["project_name"], tceq_row["project_name"]),
        _similarity(ercot_row["company"], tceq_row["project_name"]),
        _similarity(ercot_row["project_name"], tceq_row["company"]),
    )


# --------------------------------------------------------------------------- #
# Stage 1 + 2: county gate and scored candidates
# --------------------------------------------------------------------------- #
def candidate_pairs(records: pd.DataFrame) -> pd.DataFrame:
    """Score every same-county ERCOT x TCEQ pair above the reject floor.

    Returns columns: ercot_id, tceq_id, score, plus display fields for
    adjudication/evidence.
    """
    ercot = records[records["source"] == "ercot"]
    tceq = records[records["source"] == "tceq"]
    rows = []
    for county, egroup in ercot.groupby("county_key"):
        tgroup = tceq[tceq["county_key"] == county]
        if not len(tgroup):
            continue
        for _, e in egroup.iterrows():
            for _, t in tgroup.iterrows():
                score = pair_score(e, t)
                if score < AUTO_REJECT:
                    continue
                rows.append(
                    {
                        "ercot_id": e["source_id"],
                        "tceq_id": t["source_id"],
                        "score": round(score, 3),
                        "county": e["county"],
                        "ercot_company": e["company"],
                        "ercot_project": e["project_name"],
                        "ercot_kind": e["kind"],
                        "tceq_company": t["company"],
                        "tceq_project": t["project_name"],
                        "tceq_kind": t["kind"],
                    }
                )
    cols = ["ercot_id", "tceq_id", "score", "county", "ercot_company",
            "ercot_project", "ercot_kind", "tceq_company", "tceq_project", "tceq_kind"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage 3: Claude adjudication of the ambiguous band (batched + disk-cached)
# --------------------------------------------------------------------------- #
_ADJUDICATE_SYSTEM = (
    "You judge whether two public records describe the SAME power project in "
    "Texas. Record A is from the ERCOT interconnection queue; record B is a "
    "TCEQ air permit. Same county is already verified. Companies often file "
    "under different LLC or holding-company names, so judge by the totality: "
    "name overlap, project vs company naming patterns, and whether the "
    "generation type is consistent (a wind farm does not need an air permit; "
    "gas plants do). Be conservative: when genuinely unsure, answer false.\n"
    "Reply with ONLY a JSON array, one object per pair, in the same order: "
    '[{"index": 0, "same_project": true, "reason": "<one short sentence>"}]'
)


def _cache_path():
    return config.SNAPSHOT_DIR / VERDICT_CACHE_NAME


def _load_verdict_cache() -> dict:
    if _cache_path().exists():
        return json.loads(_cache_path().read_text())
    return {}


def _save_verdict_cache(cache: dict) -> None:
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, indent=0))


def _pair_key(row: pd.Series) -> str:
    return f"{row['ercot_id']}|{row['tceq_id']}"


def _adjudicate_batch(batch: pd.DataFrame) -> list[dict]:
    """One Claude call for up to BATCH_SIZE pairs. Returns verdict dicts."""
    import anthropic

    pairs = []
    for i, (_, r) in enumerate(batch.iterrows()):
        pairs.append(
            {
                "index": i,
                "A_ercot": {"company": r["ercot_company"], "project": r["ercot_project"],
                             "type": r["ercot_kind"]},
                "B_tceq": {"company": r["tceq_company"], "site": r["tceq_project"],
                            "type": r["tceq_kind"]},
                "county": r["county"],
            }
        )
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.MATCH_MODEL,
        max_tokens=2000,
        system=_ADJUDICATE_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(pairs)}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.DOTALL)
    verdicts = json.loads(m.group(0) if m else text)
    out = [{"same_project": False, "reason": "no verdict returned"} for _ in range(len(batch))]
    for v in verdicts:
        i = int(v.get("index", -1))
        if 0 <= i < len(out):
            out[i] = {
                "same_project": bool(v.get("same_project", False)),
                "reason": str(v.get("reason", ""))[:200],
            }
    return out


def adjudicate(ambiguous: pd.DataFrame) -> pd.DataFrame:
    """Run the ambiguous band through Claude (cache-first, capped, batched).

    Returns ``ambiguous`` with added columns: verdict (bool), reason (str),
    adjudicated (bool — False for pairs skipped by DRY_RUN or the cap).
    """
    cache = _load_verdict_cache()
    verdicts: list[dict] = []
    to_ask_idx = []
    for idx, row in ambiguous.iterrows():
        key = _pair_key(row)
        if key in cache:
            verdicts.append({**cache[key], "adjudicated": True})
        else:
            verdicts.append({"same_project": False, "reason": "pending", "adjudicated": False})
            to_ask_idx.append(idx)

    if config.DRY_RUN:
        for i, idx in enumerate(to_ask_idx):
            pos = list(ambiguous.index).index(idx)
            verdicts[pos] = {
                "same_project": False,
                "reason": "[DRY_RUN] adjudication skipped",
                "adjudicated": False,
            }
    else:
        to_ask = ambiguous.loc[to_ask_idx].head(config.MATCH_MAX_CLAUDE_PAIRS)
        for start in range(0, len(to_ask), BATCH_SIZE):
            batch = to_ask.iloc[start: start + BATCH_SIZE]
            try:
                results = _adjudicate_batch(batch)
            except Exception as exc:  # noqa: BLE001 - degrade, don't crash the demo
                # API unavailable (no credits, network, rate limit): leave the
                # rest un-adjudicated; auto-matches still stand.
                print(f"[resolve] adjudication unavailable ({exc}); continuing without it")
                break
            for (idx, row), res in zip(batch.iterrows(), results):
                cache[_pair_key(row)] = res
                pos = list(ambiguous.index).index(idx)
                verdicts[pos] = {**res, "adjudicated": True}
        _save_verdict_cache(cache)

    out = ambiguous.copy()
    out["verdict"] = [v["same_project"] for v in verdicts]
    out["reason"] = [v["reason"] for v in verdicts]
    out["adjudicated"] = [v["adjudicated"] for v in verdicts]
    return out


# --------------------------------------------------------------------------- #
# The full funnel
# --------------------------------------------------------------------------- #
def resolve(records: pd.DataFrame) -> pd.DataFrame:
    """Run the whole matching funnel over the unified record table.

    Returns the match table (one row per confirmed ERCOT<->TCEQ link):
    ercot_id, tceq_id, score, method ("auto" | "claude"), reason, plus the
    display fields from candidate_pairs. Saved to a snapshot for the app.
    """
    cands = candidate_pairs(records)
    if not len(cands):
        return cands.assign(method="", reason="")

    auto = cands[cands["score"] >= AUTO_MATCH].copy()
    auto["method"] = "auto"
    auto["reason"] = "near-identical name in same county"

    ambiguous = cands[(cands["score"] < AUTO_MATCH)].copy()
    judged = adjudicate(ambiguous)
    confirmed = judged[judged["verdict"]].copy()
    confirmed["method"] = "claude"

    matches = pd.concat(
        [auto, confirmed[auto.columns.tolist()]], ignore_index=True
    )
    # One permit can serve several queue entries (repowers), but keep the
    # single best ERCOT link per permit and vice versa to stay precise.
    matches = (
        matches.sort_values("score", ascending=False)
        .drop_duplicates(subset=["ercot_id"])
        .drop_duplicates(subset=["tceq_id"])
        .reset_index(drop=True)
    )
    save_matches(matches)
    return matches


def link_records(records: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Fold match results back into the record table.

    ERCOT rows gain: coordinates inherited from their matched permit, plus
    match_id / match_reason columns (both sides). Unmatched rows are unchanged.
    """
    out = records.copy()
    out["match_id"] = ""
    out["match_reason"] = ""
    if not len(matches):
        return out

    tceq_by_id = out[out["source"] == "tceq"].set_index("source_id")
    for _, m in matches.iterrows():
        e_mask = (out["source"] == "ercot") & (out["source_id"] == m["ercot_id"])
        t_mask = (out["source"] == "tceq") & (out["source_id"] == m["tceq_id"])
        out.loc[e_mask, "match_id"] = m["tceq_id"]
        out.loc[t_mask, "match_id"] = m["ercot_id"]
        out.loc[e_mask, "match_reason"] = m["reason"]
        out.loc[t_mask, "match_reason"] = m["reason"]
        if m["tceq_id"] in tceq_by_id.index:
            t = tceq_by_id.loc[m["tceq_id"]]
            out.loc[e_mask, "lat"] = t["lat"]
            out.loc[e_mask, "lon"] = t["lon"]
    return out


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #
def matches_path():
    return config.SNAPSHOT_DIR / MATCHES_NAME


def save_matches(df: pd.DataFrame) -> None:
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(matches_path(), index=False)


def load_matches() -> pd.DataFrame | None:
    if matches_path().exists():
        return pd.read_parquet(matches_path())
    return None


if __name__ == "__main__":
    # Run the whole funnel from the command line:  python -m src.resolve
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.sources import load_all

    records, _ = load_all(refresh=False)
    result = resolve(records)
    print(f"{len(result)} confirmed matches "
          f"({result['method'].value_counts().to_dict() if len(result) else {}})")
    print(f"saved to {matches_path()}")
