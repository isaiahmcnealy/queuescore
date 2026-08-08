"""TCEQ air-permit ingestion via the public Permit Search API.

The portal at https://permit-search.tceq.texas.gov is a React app over a public
JSON API (no login). Flow: GET a token, then POST a search. Schema was
reverse-engineered from the app bundle — see SOURCES.md for the full recon.

We pull Air New Source Review (AIRNSR) permits filtered server-side by the
keyword "electric power generation" (~2k records ≈ 4 requests), which is
exactly the power-plant slice QueueScore cares about. A NEW APPLICATION
air permit for a gas plant is one of the earliest public signals a project
is real.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import requests

from src import config
from src.sources import RECORD_COLUMNS

BASE = "https://permit-search.tceq.texas.gov/psp-webservices/v1"
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)

# The power-generation slice: server-side keyword search over business
# name/activity. Yields ~1,900 records vs 96k unfiltered AIRNSR.
POWER_KEYWORD = "electric power generation"
PAGE_SIZE = 500

SNAPSHOT_NAME = "tceq_air_permits.parquet"
META_NAME = "tceq_air_permits.meta.json"


def _token(session: requests.Session) -> str:
    r = session.get(f"{BASE}/token", params={"appName": "psp-frontend"}, timeout=30)
    r.raise_for_status()
    return r.text.strip()


def _search_page(session: requests.Session, token: str, page: int) -> dict:
    payload = {
        "keyword": POWER_KEYWORD,
        "normalizedAddress": "",
        "radius": None,       # null lat/lon/radius = all of Texas
        "latitude": None,
        "longitude": None,
        "permitStatus": ["NEW APPLICATION", "ISSUED PERMIT", "RENEWAL/AMENDMENT"],
        "mediaPrograms": [{"mediaCd": "AIR", "programDesc": ["Air New Source"]}],
        "pageable": {"paged": True, "pageNumber": page, "pageSize": PAGE_SIZE},
    }
    r = session.post(
        f"{BASE}/search/permitsWithPages",
        params={"page": page, "size": PAGE_SIZE},
        headers={
            "Authorization": f"Bearer {token}",
            "AppName": "psp-frontend",  # required — 400 without it
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _normalize(records: list[dict]) -> pd.DataFrame:
    """Map raw TCEQ records onto the unified record schema."""
    rows = []
    for r in records:
        county_raw = (r.get("aiCnty") or "").strip()
        rows.append(
            {
                "source": "tceq",
                "source_id": str(r.get("aiIdNumber") or "").strip(),
                "project_name": (r.get("aiPermittedName") or r.get("reName") or "").strip(),
                "company": (r.get("aiIssueToName") or "").strip(),
                "county": county_raw.title(),
                "county_key": county_raw.upper(),
                "lat": r.get("aiLatDecCoord"),
                "lon": r.get("aiLongDecCoord"),
                "status": (r.get("pspStatusCd") or "").strip(),
                "stage_signal": (r.get("pspStatusCd") or "").strip(),
                "capacity_mw": None,
                "kind": (r.get("aiNaicsDesc") or "").strip(),
                "record_date": r.get("aiIssueToBeginDt"),
                "link_id": (r.get("reRegNumber") or "").strip(),
            }
        )
    df = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    # One company can hold several permits for one site; keep unique permits.
    return df.drop_duplicates(subset=["source_id"]).reset_index(drop=True)


def fetch_air_permits(use_cache_on_error: bool = True) -> pd.DataFrame:
    """Pull power-generation air permits live; fall back to the snapshot offline.

    Returns a frame in the unified record schema and saves it (plus a
    fetched-at stamp) so the app runs with no network.
    """
    try:
        with requests.Session() as s:
            s.headers["User-Agent"] = _UA
            token = _token(s)
            first = _search_page(s, token, 0)
            content = list(first.get("content", []))
            total_pages = int(first.get("totalPages") or 1)
            # totalPages is reported against the server's own page size; guard
            # with a hard cap so a schema surprise can't turn into 3k requests.
            for page in range(1, min(total_pages, 10)):
                nxt = _search_page(s, token, page)
                batch = nxt.get("content", [])
                if not batch:
                    break
                content.extend(batch)
        df = _normalize(content)
        save_snapshot(df)
        return df
    except Exception as exc:  # noqa: BLE001 - any failure -> offline fallback
        if use_cache_on_error and snapshot_path().exists():
            return load_snapshot()
        raise RuntimeError(
            "TCEQ live pull failed and no snapshot exists. "
            "Run once online to seed the cache."
        ) from exc


# --------------------------------------------------------------------------- #
# Snapshot cache + freshness
# --------------------------------------------------------------------------- #
def snapshot_path():
    return config.SNAPSHOT_DIR / SNAPSHOT_NAME


def _meta_path():
    return config.SNAPSHOT_DIR / META_NAME


def save_snapshot(df: pd.DataFrame) -> None:
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(snapshot_path(), index=False)
    _meta_path().write_text(
        json.dumps({"fetched_at": dt.datetime.now().isoformat(), "rows": len(df)})
    )


def load_snapshot() -> pd.DataFrame:
    return pd.read_parquet(snapshot_path())


def last_updated() -> dt.datetime | None:
    """When the snapshot was last refreshed (None if never)."""
    if not _meta_path().exists():
        return None
    return dt.datetime.fromisoformat(json.loads(_meta_path().read_text())["fetched_at"])
