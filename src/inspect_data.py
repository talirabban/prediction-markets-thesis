"""
inspect_data.py  (PATCHED v3)
=============================
Sanity report after running the collectors. New in v3:
  - "Eligible window" section: counts markets that pass both --min-volume
    and --max-age-days filters at common threshold combinations. Tells you
    how big your usable sample is for the thesis.
  - Better example-market selection.
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


# ---------------------------------------------------------------------------
# Heuristic categorizer (reused in feature engineering downstream).
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Crypto",        ("bitcoin", "ethereum", "solana", "btc ", "eth ", "doge",
                       "crypto", "xrp", "polygon ", " sol ", "altcoin")),
    ("Politics",      ("trump", "biden", "harris", "election", "president",
                       "congress", "senate", "supreme court", "republican",
                       "democrat", "vote ", "voter", "primary", "gop ",
                       "kamala", "vance")),
    ("Sports",        ("nfl", "nba", "nhl", "mlb", "ncaa", "f1 ", "nascar",
                       "soccer", "premier league", "champions league", "ucl",
                       "match o/u", "set 1", "set 2", "total kills",
                       "spread:", "win on 20", "fc ", "vs.", " vs ")),
    ("Macro/Econ",    ("fed ", "interest rate", "inflation", "cpi", "gdp",
                       "recession", "unemployment", "jobs report", "yield curve")),
    ("Tech/Business", ("openai", "gpt", "altman", "google", "apple", "tesla",
                       "microsoft", " ai ", "model release", "anthropic",
                       "claude ", "nvidia")),
    ("Geopolitics",   ("ukraine", "russia", "putin", "china", "taiwan",
                       "israel", "gaza", "iran", "north korea", "nato",
                       "ceasefire", "peace deal")),
    ("Entertainment", ("oscar", "grammy", "emmy", "golden globe", "billboard",
                       "movie", "box office", "tour", "album", "song of the")),
]


def categorize(question: str) -> str:
    q = (question or "").lower()
    for label, keywords in CATEGORY_KEYWORDS:
        if any(k in q for k in keywords):
            return label
    return "Other"


def load_all_markets() -> list[dict]:
    out: list[dict] = []
    for p in sorted(glob(str(MARKETS_DIR / "page_*.json"))):
        try:
            out.extend(json.loads(Path(p).read_text()))
        except json.JSONDecodeError:
            pass
    return out


def get_volume(m: dict) -> float:
    v = m.get("volumeNum")
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(m.get("volume") or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def age_days(m: dict, now: datetime) -> float | None:
    d = parse_iso(m.get("endDate"))
    if d is None:
        return None
    return (now - d).total_seconds() / 86400.0


def main() -> None:
    markets = load_all_markets()
    n = len(markets)
    now = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"MARKET METADATA  ({n:,} resolved markets cached)")
    print("=" * 60)
    if n == 0:
        print("(No markets — run collect_markets.py first.)")
        return

    # Volume distribution
    vols = sorted([get_volume(m) for m in markets], reverse=True)
    print("\nLifetime volume distribution:")
    print(f"  max    ${vols[0]:>14,.0f}")
    print(f"  p90    ${vols[n // 10]:>14,.0f}")
    print(f"  median ${vols[n // 2]:>14,.0f}")
    print(f"  p10    ${vols[9 * n // 10]:>14,.0f}")

    # Age distribution
    ages = [a for m in markets if (a := age_days(m, now)) is not None and a >= 0]
    ages.sort()
    print(f"\nAge distribution (days since endDate, {len(ages):,} markets with valid date):")
    if ages:
        print(f"  min {ages[0]:>6.1f}d   p10 {ages[len(ages) // 10]:>6.1f}d   "
              f"median {ages[len(ages) // 2]:>6.1f}d   p90 {ages[9 * len(ages) // 10]:>6.0f}d   "
              f"max {ages[-1]:>7.0f}d")
        for cutoff in (7, 14, 25, 30, 60, 180):
            in_window = sum(1 for a in ages if a <= cutoff)
            print(f"  closed within {cutoff:>3}d: {in_window:>6,} ({100 * in_window / n:>5.1f}%)")

    # ELIGIBLE WINDOW — the headline number for the thesis
    print("\n" + "=" * 60)
    print("ELIGIBLE WINDOW  (volume AND age filter together)")
    print("=" * 60)
    print(f"{'min_volume':>12}  {'max_age_days':>12}  {'eligible':>10}")
    print("-" * 40)
    for min_vol in (100, 1_000, 10_000, 100_000):
        for max_age in (7, 25, 30):
            cnt = sum(1 for m in markets
                      if get_volume(m) >= min_vol
                      and (a := age_days(m, now)) is not None
                      and 0 <= a <= max_age)
            print(f"  ${min_vol:>10,}  {max_age:>12}  {cnt:>10,}")
        print()

    # Categories within the recommended eligible window
    eligible = [m for m in markets
                if get_volume(m) >= 1000
                and (a := age_days(m, now)) is not None
                and 0 <= a <= 25]
    if eligible:
        print(f"Category mix within the recommended window "
              f"(vol>=$1K, age<=25d, n={len(eligible):,}):")
        cats = Counter(categorize(m.get("question", "")) for m in eligible)
        for label, count in cats.most_common():
            print(f"  {label:<15} {count:>6,}  ({100 * count / len(eligible):>5.1f}%)")

    # Price history depth
    price_files = sorted(glob(str(PRICES_DIR / "*.json")))
    print("\n" + "=" * 60)
    print(f"PRICE HISTORY  ({len(price_files):,} files cached)")
    print("=" * 60)
    if not price_files:
        print("(No price files yet — run collect_prices.py.)")
        return

    bars: list[int] = []
    for p in price_files:
        try:
            payload = json.loads(Path(p).read_text())
            bars.append(len(payload.get("history") or []))
        except Exception:
            pass

    if bars:
        bars.sort()
        m = len(bars)
        print(f"Bars per market across {m:,} files:")
        print(f"  empty (0 bars):    {sum(1 for b in bars if b == 0):>5,}  "
              f"({100 * sum(1 for b in bars if b == 0) / m:>4.1f}%)")
        print(f"  >= 10 bars:        {sum(1 for b in bars if b >= 10):>5,}")
        print(f"  >= 24 bars (≈1d):  {sum(1 for b in bars if b >= 24):>5,}")
        print(f"  >= 168 bars (≈1w): {sum(1 for b in bars if b >= 168):>5,}")
        print(f"  >= 720 bars (≈1m): {sum(1 for b in bars if b >= 720):>5,}")
        nonzero = [b for b in bars if b > 0]
        if nonzero:
            print(f"  among non-empty: min={nonzero[0]}, "
                  f"median={nonzero[len(nonzero) // 2]}, max={nonzero[-1]}")


if __name__ == "__main__":
    main()