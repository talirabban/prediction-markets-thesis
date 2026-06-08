"""
collect_prices.py  (PATCHED v3)
===============================
Pull YES-token price history from CLOB for markets that pass two filters:
  1. lifetime volume >= --min-volume
  2. resolved within the last --max-age-days

The age filter exists because Polymarket's CLOB only retains price time-series
for ~30 days after a market closes (metadata persists forever; bars do not).
Confirmed empirically: markets older than ~28d return empty histories,
markets within 28d return full histories.

Changes from v2:
  - New --max-age-days (default 25, giving a 5-day safety buffer).
  - New --retry-empty: re-fetches cached files that came back with 0 bars
    (use this once after upgrading; the previous run cached empties that
    were really misses).
  - Caches startDate / endDate inside each price file for downstream use.

Output: data/raw/prices/<condition_id>.json

Usage:
    python src/collect_prices.py                            # full pull, default filters
    python src/collect_prices.py --max-markets 50           # smoke test
    python src/collect_prices.py --retry-empty              # one-time cleanup of bad cache
    python src/collect_prices.py --max-age-days 25 --min-volume 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import requests

CLOB_BASE = "https://clob.polymarket.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKETS_DIR = PROJECT_ROOT / "data" / "raw" / "markets"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_BETWEEN = 0.05
REQUEST_TIMEOUT = 30


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "columbia-thesis-research/0.1 (academic use)",
        "Accept": "application/json",
    })
    return s


def load_all_markets() -> list[dict]:
    out: list[dict] = []
    for p in sorted(glob(str(MARKETS_DIR / "page_*.json"))):
        try:
            out.extend(json.loads(Path(p).read_text()))
        except json.JSONDecodeError:
            print(f"  ! skipping unreadable page: {p}")
    return out


def parse_token_ids(market: dict) -> list[str]:
    """clobTokenIds usually arrives as a JSON-encoded STRING."""
    raw = market.get("clobTokenIds")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def get_volume(market: dict) -> float:
    v = market.get("volumeNum")
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(market.get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def age_days(market: dict, now: datetime) -> float | None:
    """Days since the market's endDate. None if endDate is missing/malformed."""
    d = parse_iso(market.get("endDate"))
    if d is None:
        return None
    return (now - d).total_seconds() / 86400.0


def fetch_price_history(
    session: requests.Session, token_id: str, fidelity: int
) -> list[dict] | None:
    params = {"market": token_id, "interval": "max", "fidelity": fidelity}
    try:
        r = session.get(f"{CLOB_BASE}/prices-history", params=params, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    payload = r.json()
    return payload.get("history", []) if isinstance(payload, dict) else None


def main(
    fidelity: int,
    min_volume: float,
    max_age_days: float,
    max_markets: int | None,
    retry_empty: bool,
) -> None:
    session = make_session()

    markets = load_all_markets()
    if not markets:
        print("No cached markets found. Run collect_markets.py first.")
        sys.exit(1)

    now = datetime.now(timezone.utc)

    # Filter by volume AND age.
    eligible: list[dict] = []
    for m in markets:
        if get_volume(m) < min_volume:
            continue
        age = age_days(m, now)
        if age is None or age < 0 or age > max_age_days:
            continue
        eligible.append(m)

    print(f"Loaded {len(markets):,} markets")
    print(f"After filters (volume >= ${min_volume:,.0f}, age <= {max_age_days:.0f}d): "
          f"{len(eligible):,} eligible")

    if max_markets is not None:
        eligible = eligible[:max_markets]
        print(f"Capped at {len(eligible):,} for this run.")

    print(f"Fidelity: {fidelity} min/bar")
    if retry_empty:
        print("--retry-empty: will re-fetch any cached file with 0 bars.\n")
    else:
        print()

    new = cached = no_token = http_fail = retried = 0

    for i, m in enumerate(eligible, start=1):
        cid = m.get("conditionId")
        if not cid:
            no_token += 1
            continue

        out_file = PRICES_DIR / f"{cid}.json"

        # If file exists, decide whether to skip or retry.
        if out_file.exists():
            try:
                existing = json.loads(out_file.read_text())
                existing_bars = len(existing.get("history") or [])
            except Exception:
                existing_bars = 0

            if existing_bars > 0:
                cached += 1
                continue
            if not retry_empty:
                cached += 1
                continue
            # retry_empty flag is on AND existing file is empty — fall through to re-fetch
            retried += 1

        token_ids = parse_token_ids(m)
        if len(token_ids) == 0:
            no_token += 1
            continue

        yes_token = token_ids[0]
        history = fetch_price_history(session, yes_token, fidelity)

        if history is None:
            http_fail += 1
            time.sleep(SLEEP_BETWEEN)
            continue

        out_file.write_text(json.dumps({
            "condition_id": cid,
            "yes_token": yes_token,
            "fidelity_minutes": fidelity,
            "volume": get_volume(m),
            "start_date": m.get("startDate"),
            "end_date": m.get("endDate"),
            "history": history,
        }))
        new += 1

        if i % 50 == 0 or i == len(eligible):
            print(f"  {i:>5}/{len(eligible):<5}  "
                  f"new={new:<5}  cached={cached:<5}  retried={retried:<4}  "
                  f"no_token={no_token:<3}  http_fail={http_fail:<3}")

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. New: {new}, Cached (kept): {cached}, Retried: {retried}, "
          f"No tokens: {no_token}, HTTP failures: {http_fail}.")
    if http_fail:
        print("Rerun the script — HTTP failures are not cached and will be retried.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Polymarket price history.")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="Minutes per price bar. 60=hourly (default), 1440=daily.")
    parser.add_argument("--min-volume", type=float, default=1000.0,
                        help="Skip markets with volume below this (default $1000).")
    parser.add_argument("--max-age-days", type=float, default=25.0,
                        help="Skip markets that closed more than this many days ago "
                             "(default 25; CLOB cuts off around 30).")
    parser.add_argument("--max-markets", type=int, default=None,
                        help="Optional cap on number of markets (for smoke testing).")
    parser.add_argument("--retry-empty", action="store_true",
                        help="Re-fetch cached price files that have 0 bars "
                             "(use this once after upgrading from v2).")
    args = parser.parse_args()
    try:
        main(
            fidelity=args.fidelity,
            min_volume=args.min_volume,
            max_age_days=args.max_age_days,
            max_markets=args.max_markets,
            retry_empty=args.retry_empty,
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved — re-run to resume.")
        sys.exit(130)