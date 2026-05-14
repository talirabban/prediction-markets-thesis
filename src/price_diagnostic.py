"""
price_diagnostic.py
===================
Why are most price histories empty? Compare metadata of markets that
returned empty histories vs ones that returned data. No API calls —
just reads cached files.

Run: python src/price_diagnostic.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKETS_DIR = PROJECT_ROOT / "data" / "raw" / "markets"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"


def load_markets_by_cid() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(glob(str(MARKETS_DIR / "page_*.json"))):
        try:
            for m in json.loads(Path(p).read_text()):
                cid = m.get("conditionId")
                if cid:
                    out[cid] = m
        except json.JSONDecodeError:
            pass
    return out


def parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def summarize(label: str, bucket: list[dict]) -> None:
    if not bucket:
        print(f"  ({label} bucket is empty)")
        return
    print(f"\n--- {label}  (n={len(bucket)}) ---")

    # Volume
    vols = sorted(x["volume"] for x in bucket)
    print(f"  volume:   median ${vols[len(vols) // 2]:>12,.0f}   "
          f"min ${vols[0]:>10,.0f}   max ${vols[-1]:>14,.0f}")

    # Age (days since endDate)
    now = datetime.now(timezone.utc)
    ages = []
    for x in bucket:
        d = parse_iso(x["endDate"])
        if d:
            ages.append((now - d).days)
    if ages:
        ages.sort()
        n = len(ages)
        print(f"  age (d):  median {ages[n // 2]:>12}   "
              f"min {ages[0]:>10}   max {ages[-1]:>14}")
        print(f"  age buckets:  <30d: {sum(1 for a in ages if a < 30)}   "
              f"30-180d: {sum(1 for a in ages if 30 <= a < 180)}   "
              f"180-730d: {sum(1 for a in ages if 180 <= a < 730)}   "
              f">=730d: {sum(1 for a in ages if a >= 730)}")

    # negRisk flag
    neg = sum(1 for x in bucket if x.get("negRisk"))
    print(f"  negRisk:  {neg}/{len(bucket)} = {100 * neg / len(bucket):.0f}%")

    # outcomePrices distribution — useful for understanding outcome shape
    op_counts: Counter[str] = Counter()
    for x in bucket:
        op = x.get("outcomePrices")
        if isinstance(op, str):
            op_counts[op] += 1
        else:
            op_counts[json.dumps(op)] += 1
    print(f"  top outcomePrices values: {dict(op_counts.most_common(3))}")

    # Examples
    print("  examples:")
    for x in bucket[:5]:
        print(f"    [{x['endDate']}] vol=${x['volume']:>10,.0f}  "
              f"neg={str(x.get('negRisk')):<5}  bars={x['n_bars']:>4}  "
              f"{x['question'][:65]}")


def main() -> None:
    markets = load_markets_by_cid()
    files = sorted(glob(str(PRICES_DIR / "*.json")))
    print(f"Comparing {len(files)} cached price files against "
          f"{len(markets):,} cached markets")

    empty: list[dict] = []
    nonempty: list[dict] = []

    for f in files:
        try:
            payload = json.loads(Path(f).read_text())
        except Exception:
            continue
        cid = payload.get("condition_id")
        m = markets.get(cid)
        if not m:
            continue
        n_bars = len(payload.get("history") or [])
        info = {
            "cid": cid,
            "volume": float(m.get("volumeNum") or 0),
            "endDate": m.get("endDate"),
            "startDate": m.get("startDate"),
            "negRisk": m.get("negRisk"),
            "outcomePrices": m.get("outcomePrices"),
            "question": (m.get("question") or "")[:70],
            "n_bars": n_bars,
        }
        (empty if n_bars == 0 else nonempty).append(info)

    print(f"\nEMPTY: {len(empty)}    NON-EMPTY: {len(nonempty)}")

    summarize("EMPTY price histories", empty)
    summarize("NON-EMPTY price histories", nonempty)

    # Specific call-out: are the empties OLDER?
    now = datetime.now(timezone.utc)
    def median_age(bucket):
        ages = sorted([(now - parse_iso(x["endDate"])).days
                       for x in bucket if parse_iso(x["endDate"])])
        return ages[len(ages) // 2] if ages else None

    me, mn = median_age(empty), median_age(nonempty)
    if me is not None and mn is not None:
        print(f"\n>>> Median age: EMPTY={me}d   NON-EMPTY={mn}d")
        if me > mn * 3:
            print(">>> Verdict: AGE likely matters. Empties are much older.")
        elif mn > me * 3:
            print(">>> Verdict: weird — empties are NEWER. Probably not age.")
        else:
            print(">>> Verdict: age looks similar. Probably not age — check negRisk.")


if __name__ == "__main__":
    main()