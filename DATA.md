# Data Reference

Day-of reference for the two datasets QueueScore uses. Numbers below were pulled
live on **2026-08-07** — refresh if the ERCOT queue or LBNL edition changes.

| | Training (labels) | Scoring (live) |
|---|---|---|
| Source | LBNL "Queued Up" 2026 Data File | ERCOT queue via `gridstatus` |
| Local path | `data/raw/LBNL_Ix_Queue_Data_File_thru2025.xlsx` (15 MB, gitignored) | `data/snapshots/ercot_queue_latest.parquet` (39 KB) |
| Grain | 1 row / interconnection request | 1 row / active-or-completed project |
| Rows | 38,201 national · **3,757 ERCOT** | **1,797** (ERCOT only) |
| Has withdrawn projects? | ✅ yes (the negatives) | ❌ **no** — survivorship bias |
| License | CC BY 4.0 (attribute LBNL + GridTracker) | ERCOT public data |

**The core reason for two datasets:** the live ERCOT queue only shows projects
that are still Active or already Completed — withdrawn/cancelled projects are
dropped. You cannot learn "what fails" from it. LBNL keeps the full history
including the ~31% that withdrew, so it's the only viable training source. We
train on LBNL, then score the live ERCOT queue.

---

## 1. LBNL "Queued Up" (training data)

- **Download:** https://emp.lbl.gov/queues → "Queued Up 2026 Data File XLSX"
  - Direct: `https://emp.lbl.gov/sites/default/files/2026-05/LBNL_Ix_Queue_Data_File_thru2025.xlsx`
  - Behind Cloudflare — a scripted `curl` with a browser User-Agent works; a
    plain `curl` gets a challenge page.
- **43 sheets.** Only two matter for modeling:
  - **`03. Complete Queue Data`** — the project-level dataset. Header is on the
    **2nd row** (row 1 is a "RETURN TO CONTENTS" banner), so read with `header=1`.
  - **`04. Data Codebook`** — the data dictionary (below).
- Handy pre-aggregated reference sheets: `23. Completion Rate Trend`,
  `24. Comp. Rate Gen Type`, `25. Comp. Rate Region`, `31. IR to IA - type`.

### Codebook (30 columns)

| Field | Meaning | Notes |
|---|---|---|
| `q_id` | queue position / ID | **Unique only when combined with `entity`** |
| `q_status` | current status | `active` · `withdrawn` · `suspended` · `operational` — **the label source** |
| `q_date` | interconnection request date | project entered queue |
| `prop_date` | proposed online date | revised over time |
| `on_date` | became operational | ⚠️ leakage |
| `wd_date` | withdrawn date | ⚠️ leakage |
| `ia_date` | signed IA date | ⚠️ leakage (this *is* the target event) |
| `IA_phase_raw` / `IA_phase_clean` | study phase | `clean` standardized incl. "IA Executed" · ⚠️ leakage-adjacent |
| `county` | county | first only if multi-county |
| `state` | state | |
| `fips_code` | 5-digit county FIPS | **county-map join key** |
| `poi_name` | point of interconnection | |
| `region` | ISO/region | filter `== "ERCOT"` |
| `project_name` | name | mostly missing |
| `utility` / `entity` | transmission provider | `entity` = one of 57 balancing areas |
| `developer` | developer | mostly missing |
| `cluster` | queue cluster | |
| `service` | NRIS / ERIS / NRIS-ERIS / Other | |
| `project_type` | Generation / Surplus / Upgrade / Replacement | filter to Generation |
| `type_1` | resource type | **→ generation_type.** Solar, Wind, Battery, Gas, Hydro, Coal, Offshore Wind, Nuclear, Geothermal, Diesel, Oil, Hydrogen, Other Storage, Other |
| `type_2` / `type_3` | extra types | populated ⇒ hybrid / co-located |
| `type_clean` | standardized combined type | e.g. "Solar+Battery" |
| `mw_1` | capacity of type_1 (MW) | **→ capacity_mw** |
| `mw_2` / `mw_3` | capacity of type_2/3 | imputed storage MW excluded in this release |
| `q_year` / `prop_year` | derived years | from `q_date` / `prop_date` |

### ERCOT subset — label distribution (3,757 rows)

| `q_status` | count | share |
|---|---|---|
| active | 1,796 | 47.8% |
| withdrawn | 1,158 | 30.8% |
| operational | 613 | 16.3% |
| suspended | 190 | 5.1% |

- `ia_date` populated: 1,207 (32.1%) · `q_year`: 2001–2025 · `mw_1` median 180, mean 214.
- `type_1` (ERCOT): Solar 1,257 · Battery 1,149 · Wind 701 · Gas 305 · Other 287 · Coal 37 · rest tiny.

**Trainable population** = terminal outcomes only (drop `active`):
`operational` (613) vs `withdrawn` + `suspended` (1,348) → **~31% positive base rate**.
See [config.py `TARGET_DEFINITION`](src/config.py).

### Baseline reference (national, from sheet `24. Comp. Rate Gen Type`, 2000–2020 requests, count-based completion = operational / terminal)

| Type | ≈ completion rate |
|---|---|
| Nuclear | ~60% (tiny n) |
| Gas | ~32% |
| Solar | ~15% |
| Solar+Battery | ~11% |
| Battery | ~11% |

Use these (recomputed for ERCOT) to seed `BaselineScorer._TECH_RATE` day-of.

### Gotchas
- Unique key is **`q_id` + `entity`**, not `q_id` alone.
- Hybrids: `type_2`/`type_3` set ⇒ co-located; `mw_2`/`mw_3` partly excluded here.
- Multi-county projects list only the first county.
- Don't train on `active`/`suspended` as negatives — they have no terminal outcome.

---

## 2. ERCOT live queue (scoring data)

- **Fetch:** `gridstatus.Ercot().get_interconnection_queue()` (wrapped by
  [`src/ingest.py`](src/ingest.py) `fetch_ercot_queue`, which maps to the internal
  schema and caches a snapshot).
- **Seed the offline cache once online:**
  ```bash
  .venv/bin/python -c "from src.ingest import fetch_ercot_queue; print(fetch_ercot_queue().shape)"
  ```
- **Shape:** 1,797 rows × 35 raw columns → 8 mapped internal columns.

### Raw columns (35)
`Queue ID`, `Project Name`, `Interconnecting Entity`, `County`, `State`,
`Interconnection Location`, `Transmission Owner`, `Generation Type`,
`Capacity (MW)`, `Summer Capacity (MW)`, `Winter Capacity (MW)`, `Queue Date`,
`Status`, `Proposed Completion Date`, `Withdrawn Date`, `Withdrawal Comment`,
`Actual Completion Date`, `Fuel`, `Technology`, `GIM Study Phase`, study-date
columns, `IA Signed`, permit columns, `Approved for Energization/Synchronization`,
`Comment`.

### Key stats / gotchas
- **`Status` has only 2 values:** Active (1,234) · Completed (563). **No withdrawn
  rows** — `Withdrawn Date` is 0% populated. This is the survivorship bias above.
- `IA Signed` populated for 563 (== all Completed). `Actual Completion Date` for 116.
- `Generation Type` is verbose ("Other - Battery Energy Storage",
  "Solar - Photovoltaic Solar"); `Fuel`/`Technology` are cleaner. **Normalize to
  LBNL's `type_1` vocabulary before scoring.**
- `Capacity (MW)`: min **−53.3** (bad rows to clip), median 200.9, max 1,981.
- `Summer/Winter Capacity (MW)` come back as object/strings and are empty here.
- All rows `State == Texas`.
- `Queue Date`: 2013→2026 (live queue is younger than LBNL's 2001→2025 history).

---

## 3. Unified schema mapping (source of truth is `config.py`)

| Internal (`FEATURE_COLUMNS`) | LBNL column | Live ERCOT column |
|---|---|---|
| `queue_id` | `q_id` (+ `entity`) | `Queue ID` |
| `capacity_mw` | `mw_1` | `Capacity (MW)` |
| `queue_date` | `q_date` | `Queue Date` |
| `county` | `county` | `County` |
| `state` | `state` | `State` |
| `generation_type` | `type_1` | `Generation Type` (needs normalize) |
| `proposed_completion_date` | `prop_date` | `Proposed Completion Date` |
| `status` | `q_status` | `Status` |
| *(label)* | `q_status`/`ia_date` | `IA Signed` (live is unlabeled negatives-free) |

### Leakage — never features (both sources)
| Internal concept | LBNL | Live ERCOT |
|---|---|---|
| withdrawal date | `wd_date` | `Withdrawn Date` |
| IA signed date | `ia_date` | `IA Signed` |
| actual COD | `on_date` | `Actual Completion Date` |
| terminal status / phase | `q_status`, `IA_phase_clean` | `Status` |

See [`features.LEAKAGE_BANNED_COLUMNS`](src/features.py).

---

## Refresh checklist (day-of)
- [ ] Re-download LBNL xlsx if a newer edition is out (check emp.lbl.gov/queues date).
- [ ] Re-run the seed command to refresh the ERCOT snapshot.
- [ ] Fill `config.LBNL_TO_MODEL` from the mapping table above (codebook is resolved).
- [ ] Recompute `BaselineScorer._TECH_RATE` from ERCOT terminal outcomes.
- [ ] Confirm target = `q_status == operational` over the terminal population.
