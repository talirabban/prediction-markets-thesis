"""
build_price_manifest.py
=======================
Create a reproducible manifest for cached Polymarket CLOB price-history files.

Run from project root:
    python src/build_price_manifest.py --min-volume 10000 --max-age-days 25

Outputs:
    data/processed/price_manifest_all.csv
    data/processed/price_manifest_cohort.csv
    data/processed/price_manifest_summary.json

Why this exists:
    collect_prices.py caches raw price files, but the directory alone does not
    record which research cohort/filter produced them. This script joins price
    files back to Gamma market metadata and records bar counts, dates, volume,
    category, and whether each file satisfies the chosen cohort definition.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKETS_DIR = PROJECT_ROOT / "data" / "raw" / "markets"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def age_days(dt_str: str | None, now: datetime) -> float | None:
    dt = parse_iso(dt_str)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 86400.0


def get_volume(market: dict[str, Any]) -> float:
    v = market.get("volumeNum")
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(market.get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def get_category(market: dict[str, Any]) -> str:
    # Gamma schemas vary; keep this defensive.
    for key in ("category", "categoryName", "groupItemTitle"):
        val = market.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    events = market.get("events")
    if isinstance(events, list) and events:
        event0 = events[0]
        if isinstance(event0, dict):
            val = event0.get("category") or event0.get("categoryName")
            if isinstance(val, str) and val.strip():
                return val.strip()

    return "Unknown"


def load_markets_by_condition_id() -> dict[str, dict[str, Any]]:
    markets: dict[str, dict[str, Any]] = {}
    for p in sorted(glob(str(MARKETS_DIR / "page_*.json"))):
        try:
            page = json.loads(Path(p).read_text())
        except Exception:
            continue
        if not isinstance(page, list):
            continue
        for m in page:
            if not isinstance(m, dict):
                continue
            cid = m.get("conditionId")
            if cid:
                markets[str(cid)] = m
    return markets


def price_time_bounds(history: list[Any]) -> tuple[Any, Any]:
    if not history:
        return None, None

    def get_t(row: Any) -> Any:
        if isinstance(row, dict):
            return row.get("t") or row.get("timestamp") or row.get("time")
        return None

    return get_t(history[0]), get_t(history[-1])


def load_price_rows(
    markets_by_cid: dict[str, dict[str, Any]],
    min_volume: float,
    max_age_days: float,
    now: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for p in sorted(PRICES_DIR.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
        except Exception:
            payload = {}

        cid = str(payload.get("condition_id") or p.stem)
        history = payload.get("history") if isinstance(payload, dict) else []
        if not isinstance(history, list):
            history = []

        m = markets_by_cid.get(cid, {})
        volume = get_volume(m) if m else float(payload.get("volume") or 0)
        end_date = m.get("endDate") or payload.get("end_date")
        closed_time = m.get("closedTime") or m.get("closed_time")
        end_age = age_days(end_date, now)
        closed_age = age_days(closed_time, now)
        first_bar_time, last_bar_time = price_time_bounds(history)

        in_cohort = (
            volume >= min_volume
            and end_age is not None
            and 0 <= end_age <= max_age_days
        )

        rows.append({
            "condition_id": cid,
            "price_file": str(p.relative_to(PROJECT_ROOT)),
            "matched_market_metadata": bool(m),
            "question": m.get("question", ""),
            "category": get_category(m) if m else "Unknown",
            "volume": volume,
            "end_date": end_date or "",
            "closed_time": closed_time or "",
            "age_days_since_end_date": "" if end_age is None else round(end_age, 4),
            "age_days_since_closed_time": "" if closed_age is None else round(closed_age, 4),
            "fidelity_minutes": payload.get("fidelity_minutes", ""),
            "bar_count": len(history),
            "first_bar_time": first_bar_time or "",
            "last_bar_time": last_bar_time or "",
            "in_cohort": in_cohort,
        })

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], cohort_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def bar_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        bars = [int(r["bar_count"]) for r in rs]
        if not bars:
            return {}
        nonempty = [b for b in bars if b > 0]
        return {
            "files": len(bars),
            "empty_files": sum(b == 0 for b in bars),
            "files_with_10_plus_bars": sum(b >= 10 for b in bars),
            "files_with_24_plus_bars": sum(b >= 24 for b in bars),
            "files_with_168_plus_bars": sum(b >= 168 for b in bars),
            "min_bars_nonempty": min(nonempty) if nonempty else None,
            "median_bars_nonempty": median(nonempty) if nonempty else None,
            "max_bars_nonempty": max(nonempty) if nonempty else None,
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_definition": {
            "min_volume": args.min_volume,
            "max_age_days": args.max_age_days,
            "age_field": "endDate",
        },
        "all_cached_price_files": bar_stats(rows),
        "cohort_price_files": bar_stats(cohort_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for cached Polymarket price histories.")
    parser.add_argument("--min-volume", type=float, default=10000.0)
    parser.add_argument("--max-age-days", type=float, default=25.0)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    markets_by_cid = load_markets_by_condition_id()
    rows = load_price_rows(markets_by_cid, args.min_volume, args.max_age_days, now)
    cohort_rows = [r for r in rows if r["in_cohort"]]

    all_path = OUT_DIR / "price_manifest_all.csv"
    cohort_path = OUT_DIR / "price_manifest_cohort.csv"
    summary_path = OUT_DIR / "price_manifest_summary.json"

    write_csv(all_path, rows)
    write_csv(cohort_path, cohort_rows)
    summary = summarize(rows, cohort_rows, args)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Loaded market metadata for {len(markets_by_cid):,} condition IDs")
    print(f"Found {len(rows):,} cached price files")
    print(f"Cohort files matching volume >= ${args.min_volume:,.0f}, age <= {args.max_age_days:g}d: {len(cohort_rows):,}")
    print()
    print(f"Wrote {all_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {cohort_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
