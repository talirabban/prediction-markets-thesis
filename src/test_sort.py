"""
test_sort.py
============
Probe which Gamma sort/filter parameters are actually honored.
Confirmed broken: order=volume (silently ignored).
Untested (and worth trying): order=closed_time, order=end_date, date-range filters.

If any of these returns markets in descending endDate order — that's our winner.

Run: python src/test_sort.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

GAMMA = "https://gamma-api.polymarket.com/markets"
COMMON = {"closed": "true", "limit": 10}


def try_params(label: str, extra: dict) -> None:
    params = {**COMMON, **extra}
    print(f"\n=== {label} ===")
    print(f"params: {params}")
    try:
        r = requests.get(GAMMA, params=params, timeout=30)
    except Exception as e:
        print(f"  REQUEST FAILED: {e}")
        return
    print(f"status: {r.status_code}")
    if r.status_code != 200:
        print(f"  body: {r.text[:200]}")
        return
    data = r.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        print(f"  unexpected response shape: {type(data).__name__}")
        return
    print(f"got {len(data)} markets. First 10 endDates (look for DESCENDING dates):")
    for m in data:
        end = m.get("endDate", "?")
        v = m.get("volumeNum", 0)
        q = (m.get("question") or "")[:55]
        try:
            v_str = f"${v:>10,.0f}" if v else "$        ?"
        except Exception:
            v_str = "?"
        print(f"  endDate={end}  vol={v_str}  {q}")


if __name__ == "__main__":
    print("Today is:", datetime.now(timezone.utc).isoformat())
    print("If a sort works, the first markets in that block should have")
    print("endDates within the last few days. If not — the param is ignored.\n")

    # Baseline — what we already use
    try_params("BASELINE (no sort, no filter)", {})

    # Try sort params from Gamma docs
    try_params("order=closed_time, ascending=false", {"order": "closed_time", "ascending": "false"})
    try_params("order=end_date, ascending=false",   {"order": "end_date",   "ascending": "false"})
    try_params("order=endDate, ascending=false",    {"order": "endDate",    "ascending": "false"})
    try_params("order=closedTime, ascending=false", {"order": "closedTime", "ascending": "false"})

    # Alternative param naming conventions some APIs use
    try_params("sort=-endDate",       {"sort": "-endDate"})
    try_params("orderBy=endDate&orderDir=desc", {"orderBy": "endDate", "orderDir": "desc"})

    # Date-range filters (also useful even if sort doesn't work)
    try_params("end_date_min=2026-04-15", {"end_date_min": "2026-04-15"})
    try_params("endDateMin=2026-04-15",   {"endDateMin": "2026-04-15"})