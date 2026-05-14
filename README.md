# Alpha Generation in Prediction Markets

Master's thesis — Columbia Applied Mathematics.
Applying quantitative trading methods to Polymarket event contracts.

## Layout

```
prediction-markets-thesis/
├── requirements.txt
├── src/
│   ├── collect_markets.py        # Phase 1: pull resolved market metadata from Gamma API
│   ├── collect_prices.py         # Phase 1: pull hourly price history per market
│   ├── build_price_manifest.py   # Phase 1: build cohort manifest from cached price files
│   ├── inspect_data.py           # sanity-check raw pulls
│   ├── diagnose.py               # diagnostic: volume sort, tag shapes, price-history depth
│   ├── price_diagnostic.py       # diagnostic: why are most price histories empty?
│   ├── test_sort.py              # probe Gamma API sort/filter parameters
│   ├── build_snapshots.py        # Phase 2: build snapshot panel for modeling
│   ├── explore_snapshots.py      # Phase 2: cross-sectional calibration analysis
│   ├── train_models.py           # Phase 3: train ElasticNet & Random Forest models
│   └── backtest.py               # Phase 3: backtest residual predictions into P&L
├── data/
│   ├── raw/                      # raw API JSON, cached for resume
│   │   ├── markets/              # one JSON per page of resolved markets
│   │   └── prices/               # one JSON per market (price history)
│   └── processed/                # cleaned parquet/csv used downstream
├── notebooks/                    # exploratory analysis
├── results/
│   ├── figures/                  # plots for thesis
│   └── tables/                   # CSVs for thesis
└── thesis/                       # writeup drafts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

### Phase 1 — Data Collection

```bash
# Pull resolved-market metadata (~1-3 min for 2000 markets)
python src/collect_markets.py --max-markets 2000

# Pull hourly price history for each market (~10-30 min)
python src/collect_prices.py

# Build cohort manifest from cached files
python src/build_price_manifest.py --min-volume 10000 --max-age-days 25

# Sanity-check what you have
python src/inspect_data.py
```

Both collectors are resumable — they cache to disk and skip what's already there.

### Phase 2 — Feature Engineering & EDA

```bash
# Build snapshot panel (joins metadata + price history)
python src/build_snapshots.py

# Cross-sectional calibration analysis; writes figures/ and tables/
python src/explore_snapshots.py
```

### Phase 3 — Modeling & Backtest

```bash
# Train ElasticNet and Random Forest on calibration error
python src/train_models.py

# Backtest residual predictions into a flat-stake P&L
python src/backtest.py
```

## Diagnostics

```bash
python src/diagnose.py          # volume sort check, tag shapes, price-history depth
python src/price_diagnostic.py  # compare markets with vs. without price data
python src/test_sort.py         # probe Gamma API sort parameters
```
