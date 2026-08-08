# QueueScore

**Which power projects will actually get built, and why.**

QueueScore scores every project in Texas's live ERCOT interconnection queue with a calibrated *completion probability*: the likelihood it reaches a signed Interconnection Agreement (IA) instead of quietly dying in the queue. Pick any project on the map, see how likely it is to connect, and see exactly which factors are driving that score.

## The problem

The interconnection queue is where the U.S. energy buildout gets stuck. Before any power plant, battery, or data-center-adjacent generator can connect to the grid, it has to survive a multi-year gauntlet of studies and agreements, and **most projects never make it out.** Across the country, the large majority of capacity that enters interconnection queues is eventually withdrawn; only a small fraction ever reaches commercial operation. The queue is enormous, slow, and opaque: thousands of active ERCOT projects representing hundreds of gigawatts, and no easy way to tell the real projects from the speculative ones.

That opacity is expensive for everyone trying to build or power the grid. "AI needs power, power needs AI": the demand for new generation and load has never been higher, but the bottleneck isn't ambition, it's knowing *which* of these projects are actually going to happen.

## Who it's for

QueueScore is built for the people making bets on the grid:

- **Project developers** deciding which sites to pursue and where there's real headroom versus a jammed queue.
- **Large-load and data-center siting teams** who need power *soon* and can't afford to anchor a plan to a project that will never connect.
- **Investors and lenders** pricing the risk of a project reaching operation.
- **Grid planners and analysts** who need a fast, explainable read on the pipeline.

For all of them the core question is the same (*is this project real, and how do I know?*), and today that answer takes hours of manual queue-reading. QueueScore answers it in seconds, with the reasoning attached.

## What it does

- **Explore every project in Texas** on an interactive map. Toggle between an overview and a satellite view of the actual site to inspect the surrounding area.
- **Get a completion-probability score** for any ERCOT project: a calibrated estimate of its chance of reaching a signed IA.
- **See *why*.** Every score comes with a breakdown of the factors driving it (project size, queue vintage, resource type, and above all *local queue congestion*), so it's a transparent argument, not a black box.
- **Track development stage.** An enrichment layer surfaces real-world progress signals (status unclear, application filed, permit issued) that corroborate whether a project is genuinely moving.
- **Filter and slice.** Narrow the queue by project status, by resource type, and by the specific features stitched into each project, and choose which data source you're viewing.

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
`DRY_RUN` in `.env` (defaults to true if unset).

Full setup (env vars, offline cache, training data, teammate onboarding):
[SETUP.md](SETUP.md). Dataset shapes, the LBNL codebook, and the source→model
column mapping: [DATA.md](DATA.md). Production (arya / Cloudflare): [DEPLOY.md](DEPLOY.md).

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
├── src/                 app + scoring pipeline
├── tests/
├── deploy/              systemd unit for arya
├── scripts/             run_prod, restart, setup_runner
├── .github/workflows/   ci.yml (ubuntu) + deploy.yml (self-hosted)
└── data/{raw,snapshots}/
```

Production hosting (arya, Cloudflare Tunnel, deploy-on-push): [DEPLOY.md](DEPLOY.md).

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
