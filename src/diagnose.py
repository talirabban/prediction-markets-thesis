"""
diagnose.py
===========
One-shot diagnostic script — dumps the info needed to patch the collectors.
Tells us whether the volume sort is working, the real shape of events/tags,
and the price-history depth distribution.

Run: python src/diagnose.py
Then paste the full output back in chat.
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKETS_DIR = PROJECT_ROOT / "data" / "raw" / "markets"
PRICES_DIR = PROJECT_ROOT / "data" / "raw" / "prices"


def load_all_markets() -> list[dict]:
    out: list[dict] = []
    for p in sorted(glob(str(MARKETS_DIR / "page_*.json"))):
        try:
            out.extend(json.loads(Path(p).read_text()))
        except json.JSONDecodeError:
            pass
    return out


def main() -> None:
    markets = load_all_markets()
    print(f"Loaded {len(markets)} markets\n")

    # ---- A. Is the volume sort working? -----------------------------------
    print("=" * 60)
    print("A. VOLUME SORT CHECK")
    print("=" * 60)
    print("If sort is working, the volumeNum values below should be")
    print("a strictly DECREASING list of large numbers.\n")
    print("First 20 markets' volumeNum (in order returned by API):")
    for i, m in enumerate(markets[:20]):
        v = m.get("volumeNum")
        q = (m.get("question") or "")[:55]
        print(f"  {i:>3}.  volumeNum = {v!s:>15}   {q}")

    # Quick distribution
    vols = [m.get("volumeNum") for m in markets if isinstance(m.get("volumeNum"), (int, float))]
    if vols:
        vols_sorted = sorted(vols, reverse=True)
        n = len(vols)
        print(f"\nVolume stats across all {n} markets:")
        print(f"  max:    ${vols_sorted[0]:,.0f}")
        print(f"  p90:    ${vols_sorted[n//10]:,.0f}")
        print(f"  median: ${vols_sorted[n//2]:,.0f}")
        print(f"  p10:    ${vols_sorted[9*n//10]:,.0f}")
        print(f"  min:    ${vols_sorted[-1]:,.0f}")

    # ---- B. What does the events/tags structure actually look like? -------
    print("\n" + "=" * 60)
    print("B. EVENTS / TAGS RAW SHAPE")
    print("=" * 60)
    sample = next((m for m in markets if m.get("events")), None)
    if sample:
        events = sample.get("events")
        print(f"type(events) = {type(events).__name__}")
        if isinstance(events, list) and events:
            ev = events[0]
            print(f"type(events[0]) = {type(ev).__name__}")
            if isinstance(ev, dict):
                print(f"events[0] keys: {sorted(ev.keys())}")
                # Look for anything that smells like tags / category
                for key in ev.keys():
                    if "tag" in key.lower() or "categor" in key.lower():
                        val = ev[key]
                        print(f"\n  events[0]['{key}'] = (type: {type(val).__name__})")
                        # Pretty-print first chunk
                        s = json.dumps(val, indent=2, default=str)
                        if len(s) > 800:
                            s = s[:800] + "  ...(truncated)"
                        print(s)
        else:
            print(f"events list empty or wrong type: {events!r}")

    # ---- C. Top-level keys on a market ------------------------------------
    print("\n" + "=" * 60)
    print("C. ALL TOP-LEVEL KEYS ON A MARKET")
    print("=" * 60)
    if markets:
        keys = sorted(markets[0].keys())
        for k in keys:
            v = markets[0].get(k)
            t = type(v).__name__
            preview = json.dumps(v, default=str)
            if len(preview) > 60:
                preview = preview[:60] + "..."
            print(f"  {k:<25} ({t}): {preview}")

    # ---- D. Price history depth distribution ------------------------------
    print("\n" + "=" * 60)
    print("D. PRICE HISTORY DEPTH")
    print("=" * 60)
    price_files = sorted(glob(str(PRICES_DIR / "*.json")))
    bars = []
    for p in price_files:
        try:
            payload = json.loads(Path(p).read_text())
            bars.append(len(payload.get("history") or []))
        except Exception:
            pass
    if bars:
        bars.sort()
        n = len(bars)
        print(f"Price files: {n}")
        print(f"  empty (0 bars):    {sum(1 for b in bars if b == 0)}")
        print(f"  < 10 bars:         {sum(1 for b in bars if b < 10)}")
        print(f"  >= 24 bars:        {sum(1 for b in bars if b >= 24)}")
        print(f"  >= 100 bars:       {sum(1 for b in bars if b >= 100)}")
        if any(b > 0 for b in bars):
            nonzero = [b for b in bars if b > 0]
            nonzero.sort()
            print(f"  among non-empty: min={nonzero[0]}, "
                  f"median={nonzero[len(nonzero)//2]}, max={nonzero[-1]}")


if __name__ == "__main__":
    main()
