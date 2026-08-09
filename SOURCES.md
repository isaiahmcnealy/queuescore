# Sources — QueueScore (Track 1)

Recon for the Candid hackathon (Sat 8/8/2026). Two public sources form the spine;
we join them by entity resolution and infer each project's funnel stage.

## Source 1 — ERCOT interconnection queue  ✅ wired
- Pulled live via `gridstatus.Ercot().get_interconnection_queue()` — radar
  records in `src/sources/ercot.py`; model-side queue fetch in `src/ingest.py`.
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

**Search (schema CRACKED — implemented in `src/sources/tceq.py`):**
```
POST /search/permitsWithPages?page=0&size=500
  headers: Authorization: Bearer <token>
           AppName: psp-frontend           # REQUIRED (400 without it)
           Content-Type: application/json
  body:
  {
    "keyword": "electric power generation",   # server-side activity/name search
    "normalizedAddress": "",
    "radius": null, "latitude": null, "longitude": null,   # nulls = all of Texas
    "permitStatus": ["NEW APPLICATION", "ISSUED PERMIT", "RENEWAL/AMENDMENT"],
    "mediaPrograms": [{"mediaCd": "AIR", "programDesc": ["Air New Source"]}],
    "pageable": {"paged": true, "pageNumber": 0, "pageSize": 500}
  }
  -> { totalElements, totalPages, content: [ {record}, ... ] }
```
Verified live: 270,308 records unfiltered · 96,091 AIRNSR · **1,926 with the
power-generation keyword** (~4 requests at 500/page). 1,902/1,926 carry real
coordinates; 102 are NEW APPLICATIONs (the early-stage signal).

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
