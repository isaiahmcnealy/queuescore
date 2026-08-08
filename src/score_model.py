"""Score the live ERCOT queue with P(reach signed IA).

Camille's shared-feature XGBoost bundle. One import, three calls:

    from src.score_model import load_bundle, score_queue, explain_drivers

    bundle = load_bundle()
    scored = score_queue(ercot_radar_df, bundle)
    drivers = explain_drivers(scored.iloc[[0]], bundle)

``completion_probability`` = P(project reaches a signed Interconnection Agreement),
trained on LBNL Queued Up (ERCOT only) using only features that also exist in the
live radar parquet — no train/serve skew.

Notes:
  * Rank Active rows. Completed already signed an IA (ground truth; see
    ``validation_gap``).
  * Probabilities are raw (no post-hoc calibrator). Do not re-run
    ``predict_proba`` on the model directly.
  * The Active band is high and narrow — present as **tiers**, not a 1-to-N rank.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config

DEFAULT_BUNDLE_PATH: Path = config.MODEL_BUNDLE_PATH

# Human labels for SHAP drivers in the UI.
FEATURE_LABELS: dict[str, str] = {
    "type_1": "Generation type",
    "county": "County",
    "q_year": "Queue year",
    "log_mw_1": "Project size (log MW)",
    "cohort_n_county": "County cohort size",
    "cohort_mw_county": "County cohort MW",
    "cohort_n_year": "Year cohort size",
    "cohort_mw_year": "Year cohort MW",
}


def load_bundle(path: str | Path | None = None) -> dict:
    """Load the scoring bundle (model + feature schema + categories).

    Prefers the ``.json`` booster export next to the joblib (XGBoost-native,
    version-stable) and keeps schema/metadata from the joblib sidecar.
    """
    p = Path(path) if path else DEFAULT_BUNDLE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}")
    bundle = joblib.load(p)
    json_path = p.with_suffix(".json")
    if json_path.exists():
        from xgboost import XGBClassifier

        clf = XGBClassifier()
        clf.load_model(str(json_path))
        bundle["model"] = clf
    return bundle


# Fuel mapping: parquet 'kind' ("Fuel – Technology") -> training type_1 vocab.
_FUEL_MAP = {
    "Solar": "Solar",
    "Wind": "Wind",
    "Gas": "Gas",
    "Nuclear": "Nuclear",
    "Water": "Hydro",
    "Fuel Oil": "Oil",
    "Hydrogen": "Hydrogen",
}


def kind_to_type(k) -> object:
    if pd.isna(k):
        return np.nan
    fuel = str(k).replace("\u2013", "-").split("-")[0].strip()
    if fuel in _FUEL_MAP:
        return _FUEL_MAP[fuel]
    return "Battery" if ("Battery" in str(k) or "Storage" in str(k)) else "Other"


def map_live_parquet(live: pd.DataFrame) -> pd.DataFrame:
    """Map live radar columns onto the shared training schema."""
    d = pd.DataFrame(index=live.index)
    d["q_id"] = live["source_id"]
    # ERCOT INR numbers are year-stamped: "15INR0064b" -> 2015
    d["q_year"] = pd.to_numeric(live["source_id"].astype(str).str[:2], errors="coerce") + 2000
    d["mw_1"] = pd.to_numeric(live["capacity_mw"], errors="coerce")
    d["log_mw_1"] = np.log1p(d["mw_1"].where(d["mw_1"] > 0))
    d["county"] = live["county"].astype("string").str.upper().str.strip()
    d["type_1"] = live["kind"].map(kind_to_type)
    d["status"] = live["status"]
    for src, dst in [
        ("project_name", "project_name"),
        ("company", "company"),
        ("lat", "lat"),
        ("lon", "lon"),
        ("stage_signal", "stage_signal"),
        ("link_id", "link_id"),
    ]:
        if src in live.columns:
            d[dst] = live[src].values
    return d


def add_congestion(d: pd.DataFrame) -> pd.DataFrame:
    """Congestion on the snapshot — survivors-only definition (matches training)."""
    key = pd.Series(list(zip(d["county"], d["q_year"])), index=d.index)
    d["cohort_n_county"] = key.map(d.groupby(["county", "q_year"]).size())
    d["cohort_mw_county"] = key.map(d.groupby(["county", "q_year"])["mw_1"].sum())
    d["cohort_n_year"] = d["q_year"].map(d.groupby("q_year").size())
    d["cohort_mw_year"] = d["q_year"].map(d.groupby("q_year")["mw_1"].sum())
    return d


def _prepare_X(d: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X = d.copy()
    for c, cats in bundle["cat_categories"].items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cats)
    for f in bundle["features"]:
        if f not in X.columns:
            X[f] = np.nan
    return X[bundle["features"]]


def predict(d: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Completion probability (raw; calibrator applied only if present)."""
    raw = bundle["model"].predict_proba(_prepare_X(d, bundle))[:, 1]
    cal = bundle.get("calibrator")
    return cal.transform(raw) if cal is not None else raw


def score_queue(live: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Map → congestion → score. All rows, sorted high→low by probability."""
    d = map_live_parquet(live)
    d = add_congestion(d)
    d["completion_probability"] = predict(d, bundle)
    d["ia_tier"] = d["completion_probability"].map(ia_tier)
    return d.sort_values("completion_probability", ascending=False).reset_index(drop=True)


def ia_tier(p: float) -> str:
    """Compress a narrow probability band into presentation tiers."""
    if pd.isna(p):
        return ""
    if p >= 0.90:
        return "Top"
    if p >= 0.85:
        return "Likely"
    if p >= 0.70:
        return "Watch"
    return "At-risk"


def explain_drivers(d: pd.DataFrame, bundle: dict, top_k: int = 5) -> list[dict]:
    """One dict per row: ``{'top_drivers': [(feature, signed_shap), ...]}``.

    Positive → pushed toward signing an IA; negative → away.
    """
    import shap

    X = _prepare_X(d, bundle)
    explainer = bundle.get("_shap_explainer")
    if explainer is None:
        explainer = shap.TreeExplainer(bundle["model"])
        bundle["_shap_explainer"] = explainer
    sv = explainer(X)
    out: list[dict] = []
    for i in range(len(X)):
        vals = sv.values[i]
        order = np.argsort(-np.abs(vals))[:top_k]
        out.append(
            {
                "top_drivers": [
                    (bundle["features"][j], float(vals[j])) for j in order
                ]
            }
        )
    return out


# Back-compat alias used in Camille's handoff snippet.
explain = explain_drivers


def validation_gap(scored: pd.DataFrame) -> dict:
    """Do Completed (already signed) projects score higher than Active?"""
    comp = scored.loc[scored["status"] == "Completed", "completion_probability"]
    act = scored.loc[scored["status"] == "Active", "completion_probability"]
    return {
        "completed_mean": round(float(comp.mean()), 3),
        "active_mean": round(float(act.mean()), 3),
        "gap": round(float(comp.mean() - act.mean()), 3),
    }


def attach_scores(records: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Join ``completion_probability`` / ``ia_tier`` onto unified radar records.

    Only ERCOT rows are scored; TCEQ (and others) keep NaN / empty tier.
    """
    out = records.copy()
    out["completion_probability"] = np.nan
    out["ia_tier"] = ""
    ercot = out[out["source"] == "ercot"]
    if ercot.empty:
        return out
    scored = score_queue(ercot, bundle)
    by_id = scored.set_index("q_id")[["completion_probability", "ia_tier"]]
    # Align back to original row order / index for ERCOT rows.
    mapped = ercot["source_id"].astype(str).map(by_id["completion_probability"])
    tiers = ercot["source_id"].astype(str).map(by_id["ia_tier"]).fillna("")
    out.loc[ercot.index, "completion_probability"] = mapped.values
    out.loc[ercot.index, "ia_tier"] = tiers.values
    return out
