# Alpha Generation in Prediction Markets

Master's thesis — Columbia Applied Mathematics.
Applying quantitative trading methods to Polymarket event contracts.

## Layout

```
prediction-markets-thesis/
├── requirements.txt
├── src/
│   ├── collect_markets.py   # Phase 1 Step 2a: pull resolved market metadata
│   ├── collect_prices.py    # Phase 1 Step 2b: pull price history per market
│   └── inspect_data.py      # quick sanity-check of what was pulled
├── data/
│   ├── raw/                 # raw API JSON, cached for resume
│   │   ├── markets/         # one JSON per page of resolved markets
│   │   └── prices/          # one JSON per market (price history)
│   └── processed/           # cleaned parquet/csv used downstream
├── notebooks/               # exploratory analysis
├── results/                 # figures, tables for thesis
└── thesis/                  # writeup drafts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Phase 1 commands

```bash
# 1. Pull resolved-market metadata (~1-3 min for default 2000 markets)
python src/collect_markets.py --max-markets 2000

# 2. Pull price history for each market (~10-30 min)
python src/collect_prices.py

# 3. Sanity-check what you have
python src/inspect_data.py
```

Both collectors are resumable — they cache to disk and skip what's already there.
