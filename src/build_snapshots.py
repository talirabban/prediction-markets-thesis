"""
src/build_snapshots.py

Build the snapshot panel for modeling.
Input:  data/raw/markets/page_*.json   (metadata)
        data/raw/prices/<cid>.json     (hourly YES price history)
Output: data/processed/snapshots.parquet
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- Config ----------
DATA_DIR     = Path("data")
RAW_MARKETS  = DATA_DIR / "raw" / "markets"
RAW_PRICES   = DATA_DIR / "raw" / "prices"
OUT_PATH     = DATA_DIR / "processed" / "snapshots.parquet"

SNAPSHOT_OFFSETS_HRS = [1, 6, 24, 72]                # T - offset before close
MOMENTUM_WINDOWS = {                                  # name -> hours lookback
    "momentum_1h": 1, "momentum_6h": 6,
    "momentum_1d": 24, "momentum_3d": 72,
}
VOL_WINDOW_HRS   = 24      # rolling window for price_volatility
MAX_STALENESS_HR = 6       # snapshot is NaN if no bar within this many hours

LIMIT = None  # set to e.g. 200 for a quick test run; None = full panel


# ---------- Category classifier (heuristic) ----------
CATEGORY_KEYWORDS = {
    "Crypto":   ["bitcoin", "btc", "ethereum", "eth", "solana", "sol ", "crypto",
                 "blockchain", "doge", "xrp", "binance", "coinbase", "stablecoin",
                 "altcoin", "cardano", "ada ", "polygon", "avalanche"],
    "Politics": ["election", "president", "senate", "congress", "voter", "vote ",
                 "republican", "democrat", "gop ", "primary", "trump", "biden",
                 "harris", "kamala", "governor", "scotus", "supreme court",
                 "ballot", "midterm"],
    "Sports":   ["nfl", "nba", "mlb", "nhl", "ncaa", "soccer", "uefa", "fifa",
                 "tennis", "wimbledon", "golf", "pga", "masters", "f1 ",
                 "formula 1", "mma", "ufc", "boxing", "world cup", "premier league",
                 "champions league", "la liga", "serie a", "bundesliga",
                 " vs ", " v. ", "playoff", "championship", "match"],
    "Tech":     ["stock", "earnings", "apple", "tesla", "microsoft", "google",
                 "meta ", "amazon", "nvda", "nvidia", "ipo", "merger",
                 "openai", "anthropic", "sam altman"],
}

def categorize(question: str) -> str:
    """Bucket a question into Crypto/Politics/Sports/Tech/Other by keyword match."""
    q = (question or "").lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat
    return "Other"


# ---------- Utilities ----------
def parse_iso(s):
    """ISO-8601 string -> unix seconds (UTC). NaN on failure or empty."""
    if not s:
        return np.nan
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return np.nan


def parse_outcome(op):
    """outcomePrices is usually a JSON-encoded string like '["1","0"]'.
    Returns float resolution for YES token, or NaN."""
    if op is None:
        return np.nan
    try:
        parsed = json.loads(op) if isinstance(op, str) else op
        if isinstance(parsed, list) and len(parsed) >= 1:
            return float(parsed[0])
    except Exception:
        pass
    return np.nan

# ---------- Metadata loader ----------
def load_market_metadata() -> pd.DataFrame:
    """Walk all metadata pages, dedupe by conditionId, parse fields."""
    rows = []
    files = sorted(RAW_MARKETS.glob("page_*.json"))
    for f in tqdm(files, desc="Loading metadata pages"):
        with open(f) as fh:
            data = json.load(fh)
        markets = data if isinstance(data, list) else data.get("markets", [])
        for m in markets:
            cid = m.get("conditionId")
            if not cid:
                continue
            rows.append({
                "condition_id":   cid,
                "question":       m.get("question", "") or "",
                "yes_resolution": parse_outcome(m.get("outcomePrices")),
                "end_date":       parse_iso(m.get("endDate")),
                "volume":         float(m.get("volumeNum") or 0.0),
            })
    df = pd.DataFrame(rows).drop_duplicates("condition_id", keep="first")
    df["category"] = df["question"].map(categorize)
    return df

# ---------- Per-market snapshot builder ----------
def price_at_or_before(times_arr, prices_arr, t_target, max_staleness_sec):
    """Return the most recent price at or before t_target, if within
    max_staleness_sec of t_target. Otherwise NaN. Assumes times_arr sorted."""
    idx = np.searchsorted(times_arr, t_target, side="right") - 1
    if idx < 0:
        return np.nan
    if t_target - times_arr[idx] > max_staleness_sec:
        return np.nan
    return prices_arr[idx]


def build_market_snapshots(meta_row, history):
    """For one market, emit up to 4 snapshot rows (one per offset)."""
    close = meta_row["end_date"]
    if not np.isfinite(close) or not np.isfinite(meta_row["yes_resolution"]):
        return []
    # Only keep clean binary outcomes for calibration work
    if meta_row["yes_resolution"] not in (0.0, 1.0):
        return []
    if not history:
        return []

    # Sorted numpy arrays for fast lookup
    hist_sorted = sorted(history, key=lambda r: r["t"])
    times  = np.array([r["t"] for r in hist_sorted], dtype=np.float64)
    prices = np.array([r["p"] for r in hist_sorted], dtype=np.float64)

    max_stale_sec = MAX_STALENESS_HR * 3600
    rows = []

    for offset_h in SNAPSHOT_OFFSETS_HRS:
        snap_t = close - offset_h * 3600
        price  = price_at_or_before(times, prices, snap_t, max_stale_sec)
        if not np.isfinite(price):
            continue

        # Momentum: price now minus price at lookback time
        moms = {}
        for name, win_h in MOMENTUM_WINDOWS.items():
            past_t  = snap_t - win_h * 3600
            past_p  = price_at_or_before(times, prices, past_t, max_stale_sec)
            moms[name] = price - past_p if np.isfinite(past_p) else np.nan

        # Volatility: std of consecutive bar-to-bar changes within last 24h
        vol_start = snap_t - VOL_WINDOW_HRS * 3600
        mask = (times >= vol_start) & (times <= snap_t)
        window_p = prices[mask]
        if window_p.size >= 3:
            volatility = float(np.diff(window_p).std(ddof=1))
        else:
            volatility = np.nan

        rows.append({
            "condition_id":     meta_row["condition_id"],
            "snapshot_offset_h": offset_h,
            "snapshot_time":    snap_t,
            "close_time":       close,
            "hours_to_expiry":  float(offset_h),
            "price_level":      price,
            **moms,
            "price_volatility": volatility,
            "log_volume":       np.log1p(meta_row["volume"]),
            "category":         meta_row["category"],
            "yes_resolution":   meta_row["yes_resolution"],
            "calibration_error": meta_row["yes_resolution"] - price,
        })
    return rows

# ---------- Main ----------
def main():
    print("Loading market metadata…")
    meta = load_market_metadata()
    print(f"  {len(meta):,} unique markets, "
          f"{meta['yes_resolution'].isin([0,1]).sum():,} with clean binary outcomes")

    meta_idx = meta.set_index("condition_id")

    price_files = sorted(RAW_PRICES.glob("*.json"))
    if LIMIT:
        price_files = price_files[:LIMIT]
    print(f"Processing {len(price_files):,} price files…")

    all_rows = []
    skip = {"no_meta": 0, "no_history": 0, "no_resolution": 0, "no_snapshots": 0}

    for pf in tqdm(price_files, desc="Building snapshots"):
        cid = pf.stem
        if cid not in meta_idx.index:
            skip["no_meta"] += 1
            continue
        with open(pf) as fh:
            pdata = json.load(fh)
        hist = pdata.get("history", [])
        if not hist:
            skip["no_history"] += 1
            continue

        m = meta_idx.loc[cid].to_dict()
        m["condition_id"] = cid
        if m["yes_resolution"] not in (0.0, 1.0):
            skip["no_resolution"] += 1
            continue

        rows = build_market_snapshots(m, hist)
        if not rows:
            skip["no_snapshots"] += 1
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"\nSkipped: {skip}")
    print(f"Built {len(df):,} snapshot rows across "
          f"{df['condition_id'].nunique():,} markets")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")

    # Quick sanity report
    print("\n--- Snapshots per offset ---")
    print(df["snapshot_offset_h"].value_counts().sort_index().to_string())
    print("\n--- Category distribution ---")
    print(df["category"].value_counts().to_string())
    print("\n--- Feature null-rate (%) ---")
    feats = ["price_level", "momentum_1h", "momentum_6h",
             "momentum_1d", "momentum_3d", "price_volatility"]
    print((df[feats].isna().mean() * 100).round(1).to_string())
    print("\n--- price_level + calibration_error stats ---")
    print(df[["price_level", "calibration_error"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()