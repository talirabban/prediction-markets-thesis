"""
collect_markets.py  (PATCHED v3)
================================
Pull resolved Polymarket markets from the Gamma API, ORDERED BY MOST RECENT
RESOLUTION FIRST. This is the version that gets us markets whose prices
still exist in CLOB.

Changes from v2:
  - Added `order=closedTime&ascending=false`. Verified working via test_sort.py.
    The previous default order returned mostly old markets whose price history
    has been purged by the CLOB.
  - `closedTime` is the *actual resolution time*, which is what aligns with
    CLOB's ~30-day retention window — exactly the concept we want.

Output: data/raw/markets/page_NNNN.json

Usage:
    python src/collect_markets.py                  # pull everything (recommended)
    python src/collect_markets.py --max-markets 50 # hard cap (smoke test)

NOTE: Before re-running this after v2, delete the old cache:
    rm -rf data/raw/markets/
The old cache was pulled in the wrong order and is dominated by stale markets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "markets"
RAW_DIR.mkdir(parents=True, exist_ok=True)

PAGE_SIZE = 500
SLEEP_BETWEEN = 0.5
REQUEST_TIMEOUT = 30


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "columbia-thesis-research/0.1 (academic use)",
        "Accept": "application/json",
    })
    return s


def fetch_page(session: requests.Session, offset: int, limit: int) -> list[dict]:
    """One page of resolved markets, sorted by most-recent close first."""
    params = {
        "closed": "true",
        "limit": limit,
        "offset": offset,
        "order": "closedTime",      # VERIFIED to work via test_sort.py
        "ascending": "false",       # most recently closed first
    }
    r = session.get(f"{GAMMA_BASE}/markets", params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def main(max_markets: int | None) -> None:
    session = make_session()
    total = 0
    page_num = 0
    offset = 0
    empty_in_a_row = 0

    if max_markets is None:
        print("Pulling ALL resolved Polymarket markets, most-recent-close first...")
    else:
        print(f"Pulling resolved markets, most-recent first (hard cap: {max_markets:,})...")
    print(f"Cache directory: {RAW_DIR}")

    while True:
        if max_markets is not None:
            remaining = max_markets - total
            if remaining <= 0:
                break
            page_limit = min(PAGE_SIZE, remaining)
        else:
            page_limit = PAGE_SIZE

        page_file = RAW_DIR / f"page_{page_num:04d}.json"

        markets: list[dict] | None = None
        if page_file.exists():
            try:
                markets = json.loads(page_file.read_text())
                source = "cached"
            except json.JSONDecodeError:
                page_file.unlink()
                markets = None

        if markets is None:
            try:
                markets = fetch_page(session, offset, page_limit)
                page_file.write_text(json.dumps(markets))
                source = "fetched"
                time.sleep(SLEEP_BETWEEN)
            except requests.HTTPError as e:
                print(f"  ! HTTP error on page {page_num}: {e}. Sleep 5s, retry once.")
                time.sleep(5)
                try:
                    markets = fetch_page(session, offset, page_limit)
                    page_file.write_text(json.dumps(markets))
                    source = "fetched (after retry)"
                except Exception as e2:
                    # Hitting the 100K offset cap is expected — not a real error.
                    if "422" in str(e2):
                        print(f"  Hit Gamma's offset ceiling at {offset:,}. Stopping cleanly.")
                    else:
                        print(f"  ! Giving up on page {page_num}: {e2}")
                    break

        n = len(markets)
        total += n
        # Show endDate of first market on this page — sanity check that sort is holding
        first_end = markets[0].get("endDate", "?") if markets else "?"
        print(f"  page {page_num:>4}  offset={offset:>6}  got {n:>4} markets  "
              f"[{source}]  first endDate: {first_end}  running total: {total:,}")

        if n == 0:
            empty_in_a_row += 1
            if empty_in_a_row >= 2:
                print("  Two empty pages — end of resolved-market universe.")
                break
        else:
            empty_in_a_row = 0

        page_num += 1
        offset += page_limit

    print(f"\nDone. Collected {total:,} markets across {page_num + 1} page(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull resolved Polymarket markets via Gamma API.")
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Hard cap on markets to pull. Omit to pull everything (recommended).",
    )
    args = parser.parse_args()
    try:
        main(max_markets=args.max_markets)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — re-run to resume.")
        sys.exit(130)