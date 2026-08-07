# QueueScore

Score every project in the live ERCOT interconnection queue with a **completion
probability** — the likelihood it reaches a signed Interconnection Agreement (IA)
— trained on LBNL's *Queued Up* historical dataset.

Stack: Python · XGBoost · SHAP · Streamlit · plotly · gridstatus · Anthropic API.

> **This repo is a hackathon skeleton.** Every module is structured, typed, and
> runnable end-to-end on dummy data. Real model, real LBNL parsing, and real
> explanations get filled in day-of at the `TODO(day-of)` markers.

## Setup

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Runs immediately — **no network, no API key**. The app scores baked-in demo
projects with `DummyScorer` and gates all Anthropic calls behind
`config.DRY_RUN` (defaults on).

Full setup (env vars, offline cache, training data, teammate onboarding):
[SETUP.md](SETUP.md). Dataset shapes, the LBNL codebook, and the source→model
column mapping: [DATA.md](DATA.md).

Run the tests:

```bash
pytest
```

## The feature contract (source of truth)

Everything keys off one internal schema. Both raw sources are mapped onto it in
[`src/config.py`](src/config.py); the scorer and tests depend only on it.

### Internal schema (`config.FEATURE_COLUMNS`)

| column                     | type   | source                          |
| -------------------------- | ------ | ------------------------------- |
| `queue_id`                 | str    | identifier (not a model input)  |
| `capacity_mw`              | float  | raw                             |
| `queue_date`               | date   | raw (derive-from)               |
| `county`                   | str    | raw                             |
| `state`                    | str    | raw                             |
| `generation_type`          | str    | raw                             |
| `proposed_completion_date` | date   | raw                             |
| `status`                   | str    | raw                             |

### Model features (`config.MODEL_FEATURES`)

`capacity_mw`, `generation_type`, `county`, `queue_year`, `queue_age_days`,
`size_bucket` — produced by `features.build_features` and pinned by the
`FeatureSchema` TypedDict.

### Target (`config.TARGET_DEFINITION`)

Binary: **1** if the project reached a signed Interconnection Agreement in the
LBNL record, else **0**. Still-active projects (no terminal outcome) are
**dropped** from training — not labeled negative.

### Leakage-banned columns (`features.LEAKAGE_BANNED_COLUMNS`)

Never features — each encodes the outcome: `withdrawal_date`, `ia_date`,
`actual_cod`, `terminal_status`. `build_features` drops them defensively;
enforced by tests.

### The scoring contract (`src/scorer.py`)

```python
class Scorer(ABC):
    def score(self, features: pd.DataFrame) -> ScoreResult: ...

@dataclass
class ScoreResult:
    probabilities: np.ndarray            # (n_rows,), each in [0, 1]
    attributions: list[dict[str, float]] # per-row {feature: signed contribution}
```

- `DummyScorer` — seeded random probs + fake attributions. **Wired.**
- `BaselineScorer` — historical completion rate by tech × size. **Wired.**
- `XGBScorer` — XGBoost + SHAP. **Skeleton**, implemented day-of.

## Layout

```
queuescore/
├── src/
│   ├── config.py     constants, column mappings, target, flags
│   ├── ingest.py     load_lbnl (stub) + fetch_ercot_queue (real offline cache)
│   ├── features.py   build_features (stub) + FeatureSchema + leakage guard
│   ├── scorer.py     Scorer contract + Dummy/Baseline (wired) + XGB (skeleton)
│   ├── explain.py    Claude verdict + Q&A, gated by DRY_RUN
│   └── app.py        Streamlit: pull · leaderboard · map · detail
├── tests/test_contract.py
└── data/{raw,snapshots}/
```

## What is real vs. stubbed today

| Real now                                  | Stubbed for day-of                        |
| ----------------------------------------- | ----------------------------------------- |
| Scorer contract + Dummy/Baseline scorers  | `XGBScorer` (train + SHAP)                |
| ERCOT snapshot cache (offline round-trip) | Live→features wiring in the app           |
| gridstatus column mapping                 | LBNL column mapping (needs codebook)      |
| Streamlit app on all four panels          | `build_features` derivations              |
| DRY_RUN explanation layer                 | Live Anthropic calls                      |

---

Training data: **LBNL Queued Up (CC BY 4.0)**.
