"""The scoring contract.

Everything downstream (app, explanations, tests) depends only on the ``Scorer``
interface and the ``ScoreResult`` shape defined here — never on a concrete
implementation. Swap ``DummyScorer`` for ``XGBScorer`` day-of and nothing else
changes.

Implementations:
  * ``DummyScorer``   — seeded random probabilities + fake attributions. Wired.
  * ``BaselineScorer``— historical completion rates by tech x size. Wired.
  * ``XGBScorer``     — real XGBoost + SHAP. Skeleton with TODO methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class ScoreResult:
    """The output of any Scorer.

    Attributes:
        probabilities: shape ``(n_rows,)``, each in ``[0, 1]``. Completion
            probability aligned row-for-row with the input feature frame.
        attributions: length ``n_rows``. Each entry maps feature name ->
            signed contribution to that row's score (SHAP-style; sign = push
            toward/away from completion).
    """

    probabilities: np.ndarray
    attributions: list[dict[str, float]]

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(self.attributions):
            raise ValueError(
                f"probabilities ({len(self.probabilities)}) and attributions "
                f"({len(self.attributions)}) length mismatch"
            )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class Scorer(ABC):
    """Scores a feature frame into completion probabilities + attributions."""

    @abstractmethod
    def score(self, features: pd.DataFrame) -> ScoreResult:
        """Score every row of ``features``.

        Args:
            features: model-ready frame (columns per ``features.FEATURE_FRAME_COLUMNS``).

        Returns:
            A ``ScoreResult`` aligned row-for-row with ``features``.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# DummyScorer — fully wired, deterministic
# --------------------------------------------------------------------------- #
class DummyScorer(Scorer):
    """Seeded random scores so the whole app runs before any model exists."""

    def __init__(self, seed: int = config.RANDOM_SEED) -> None:
        self._rng = np.random.default_rng(seed)

    def score(self, features: pd.DataFrame) -> ScoreResult:
        n = len(features)
        probs = self._rng.uniform(0.05, 0.95, size=n)
        feat_names = [c for c in config.MODEL_FEATURES]
        attributions: list[dict[str, float]] = []
        for _ in range(n):
            raw = self._rng.normal(0, 0.1, size=len(feat_names))
            attributions.append({name: float(v) for name, v in zip(feat_names, raw)})
        return ScoreResult(probabilities=probs, attributions=attributions)


# --------------------------------------------------------------------------- #
# BaselineScorer — fully wired, non-ML baseline
# --------------------------------------------------------------------------- #
# Placeholder historical completion rates by (generation_type, size_bucket).
# Day-of these get recomputed from the LBNL data; the numbers here are plausible
# stand-ins so the baseline is a real, deterministic function of the features.
_BASE_RATE: float = 0.20
_TECH_RATE: dict[str, float] = {
    "Solar": 0.24,
    "Wind": 0.22,
    "Battery": 0.30,
    "Gas": 0.35,
    "Nuclear": 0.15,
}
_SIZE_MULTIPLIER: dict[str, float] = {
    "small": 1.15,
    "medium": 1.0,
    "large": 0.85,
    "xlarge": 0.7,
}


class BaselineScorer(Scorer):
    """Completion probability = historical rate by tech, adjusted by size.

    A transparent, non-ML baseline the XGBoost model must beat. Attributions are
    the two multiplicative factors expressed additively around the base rate.
    """

    def score(self, features: pd.DataFrame) -> ScoreResult:
        probs: list[float] = []
        attributions: list[dict[str, float]] = []
        for _, row in features.iterrows():
            tech = str(row.get("generation_type", ""))
            size = str(row.get("size_bucket", "medium"))
            tech_rate = _TECH_RATE.get(tech, _BASE_RATE)
            size_mult = _SIZE_MULTIPLIER.get(size, 1.0)
            p = float(np.clip(tech_rate * size_mult, 0.0, 1.0))
            probs.append(p)
            attributions.append(
                {
                    "generation_type": tech_rate - _BASE_RATE,
                    "size_bucket": tech_rate * (size_mult - 1.0),
                }
            )
        return ScoreResult(
            probabilities=np.asarray(probs, dtype=float),
            attributions=attributions,
        )


# --------------------------------------------------------------------------- #
# XGBScorer — skeleton only
# --------------------------------------------------------------------------- #
class XGBScorer(Scorer):
    """Real model: XGBoost classifier + SHAP attributions.

    Skeleton only. Day-of implementation:
      * ``fit`` — train XGBClassifier on LBNL features/target, persist booster
      * ``load`` — restore a persisted booster + SHAP explainer
      * ``score`` — predict_proba + shap_values -> ScoreResult
    """

    def __init__(self) -> None:
        self._model = None       # xgboost.XGBClassifier once trained
        self._explainer = None   # shap.TreeExplainer once fit

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "XGBScorer":
        """Train on LBNL data and build the SHAP explainer. TODO(day-of)."""
        raise NotImplementedError("XGBScorer.fit is implemented day-of.")

    def load(self, model_path: str) -> "XGBScorer":
        """Restore a persisted booster + explainer. TODO(day-of)."""
        raise NotImplementedError("XGBScorer.load is implemented day-of.")

    def score(self, features: pd.DataFrame) -> ScoreResult:
        """predict_proba + SHAP -> ScoreResult. TODO(day-of)."""
        raise NotImplementedError("XGBScorer.score is implemented day-of.")
