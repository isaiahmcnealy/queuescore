# Sources — Project Radar (Track 1)

Recon for the Candid hackathon (Sat 8/8/2026). Two public sources form the spine;
we join them by entity resolution and infer each project's funnel stage.

## Source 1 — ERCOT interconnection queue  ✅ already wired
- Pulled live via `gridstatus.Ercot().get_interconnection_queue()` (see `src/ingest.py`).
- 1,797 rows × 35 cols. Fields we use: `Queue ID`, `Project Name`,
  `Interconnecting Entity`, `County`, `Generation Type`, `Capacity (MW)`,
  `Queue Date`, `Status`, `GIM Study Phase`, `IA Signed`,
  `Approved for Energization/Synchronization`, `Proposed Completion Date`.
- Liveness + offline snapshot cache already built. See DATA.md for full schema.

## Source 2 — TCEQ air permits (Permit Search API)  ✅ de-risked, public JSON
Portal: https://permit-search.tceq.texas.gov/  (React app over a public REST API)

**Base:** `https://permit-search.tceq.texas.gov/psp-webservices/v1`

**Auth (public, no login):**
```
GET /token?appName=psp-frontend   ->  returns a JWT string (use as Bearer)
```

**Search:**
```
POST /search/permitsWithPages?page=0&size=50
  headers: Authorization: Bearer <token>
           AppName: psp-frontend           # REQUIRED (400 without it)
           Content-Type: application/json
  body:    { ...filter... }                # SCHEMA TBD — includes a "keyword" field.
                                            # Empty {} -> 500. Reverse-engineer AM (below).
  -> { totalElements, totalPages, content: [ {record}, ... ] }
```
15,748 total records across all programs (filter to air + power).

**Fields per record (everything we need is here):**
| field | use |
|---|---|
| `aiPgmCd` = `AIRNSR` | filter to **Air New Source Review** (gas plants need these) |
| `aiIssueToName` | company — **join key to ERCOT `Interconnecting Entity`** |
| `aiPermittedName` / `reName` | site / project name (entity resolution) |
| `aiCnty` | county — **join key to ERCOT `County`** |
| `aiLatDecCoord` / `aiLongDecCoord` | **real coords for the live map** |
| `pspStatusCd` | `NEW APPLICATION` / `ISSUED PERMIT` — **stage signal** |
| `aiNaicsCd` / `aiNaicsDesc` | industry — filter to power gen (NAICS 2211xx) |
| `aiIdNumber`, `reRegNumber` | permit # / RN###### (link back to source filing) |
| `aiIssueToBeginDt` | date (timeline) |

**First task tomorrow (≈20 min):** nail the POST body schema. Options:
1. Reload the portal with a network capture, run a search, copy the request payload.
2. Fallback with zero reverse-engineering: **TCEQ Central Registry** bulk
   Excel/ascii download (regulated entities: name, county, coords).

## The join (the judged "hard part")
- **Entity resolution:** ERCOT `Interconnecting Entity` ↔ TCEQ `aiIssueToName`,
  gated on matching `County`. Fuzzy string match to propose, Claude to adjudicate
  the hard cases ("same project, different LLC?"). Keep the source records as evidence.
- **Stage inference:** combine signals into a funnel stage + confidence:
  - ERCOT: `GIM Study Phase`, `IA Signed`, `Approved for Energization/Synchronization`, `Status`
  - TCEQ: air-permit `pspStatusCd` (`NEW APPLICATION` → `ISSUED PERMIT`)
  - Stages: concept → feasibility → study → IA → construction → COD, each with
    the filings that justify the call.

## On-thesis lens (bonus)
Candid cares most about **early-stage gas-to-power and behind-the-meter data-center
power**. Filter ERCOT `Generation Type` = Gas + TCEQ `AIRNSR` presence to surface
exactly those.
