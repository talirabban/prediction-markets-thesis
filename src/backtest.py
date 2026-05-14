"""
src/backtest.py
================
Phase 3 backtest: convert RF/EN residual predictions into a flat-stake
betting strategy and evaluate P&L, win rate, and Sharpe-equivalent.

Strategy:
    For each test snapshot with predicted residual pred:
      side =  +1  if pred >  threshold   (buy YES at price_level)
              -1  if pred < -threshold   (buy NO  at 1-price_level)
               0  otherwise              (skip)
    Payoff per $1 bet = side * (yes_resolution - price_level)
    Stake = flat $1 per bet.

Inputs:
    data/processed/test_predictions.parquet

Outputs:
    results/tables/backtest_threshold_sweep.csv
    results/tables/backtest_by_category.csv
    results/tables/backtest_by_horizon.csv
    results/tables/backtest_cost_sensitivity.csv
    results/figures/cumulative_pnl.png
    results/figures/threshold_sweep.png
    data/processed/backtest_bets.parquet
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH    = PROJECT_ROOT / "data" / "processed" / "test_predictions.parquet"
OUT_TABLES   = PROJECT_ROOT / "results" / "tables"
OUT_FIGURES  = PROJECT_ROOT / "results" / "figures"
OUT_BETS     = PROJECT_ROOT / "data" / "processed" / "backtest_bets.parquet"

OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
COST_LEVELS = [0.000, 0.005, 0.010, 0.020]
PRIMARY_SIGNAL = "pred_RF"


# ---------------------------------------------------------------------------
# Core mechanics
# ---------------------------------------------------------------------------
def compute_bets(df: pd.DataFrame, signal_col: str, threshold: float,
                 cost: float = 0.0) -> pd.DataFrame:
    """Apply threshold rule + cost; return df with side/payoff columns."""
    signal = df[signal_col].values
    side = np.where(np.abs(signal) > threshold, np.sign(signal), 0).astype(int)

    # Gross payoff per $1 bet = side * (outcome - price).
    # Costs assessed only on placed bets.
    gross = side * (df["yes_resolution"].values - df["price_level"].values)
    net = gross - (side != 0).astype(float) * cost

    out = df.copy()
    out["signal"] = signal
    out["side"] = side
    out["payoff_gross"] = gross
    out["payoff"] = net
    return out


def summarize(bets: pd.DataFrame, label: str) -> dict:
    """One-row summary: n_bets, total/mean/std P&L, Sharpe-per-bet, win-rate."""
    placed = bets[bets["side"] != 0]
    n = len(placed)
    if n == 0:
        return dict(label=label, n_bets=0, total_pnl=0.0, mean_pnl=np.nan,
                    std_pnl=np.nan, sharpe_per_bet=np.nan, win_rate=np.nan)
    mean = placed["payoff"].mean()
    std  = placed["payoff"].std(ddof=1)
    return dict(
        label=label,
        n_bets=n,
        total_pnl=float(placed["payoff"].sum()),
        mean_pnl=float(mean),
        std_pnl=float(std),
        sharpe_per_bet=float(mean / std) if std > 0 else np.nan,
        win_rate=float((placed["payoff_gross"] > 0).mean()),  # pre-cost
    )


def main() -> None:
    print(f"Loading {PRED_PATH}")
    df = pd.read_parquet(PRED_PATH)
    df = df.sort_values("close_time").reset_index(drop=True)
    print(f"  {len(df):,} test rows")
    print(f"  close_time: {df['close_time'].min()} to {df['close_time'].max()}")

    # -----------------------------------------------------------------------
    # 1) Threshold sweep, both signals, no costs
    # -----------------------------------------------------------------------
    rows = []
    for sig in ["pred_RF", "pred_EN"]:
        for tau in THRESHOLDS:
            b = compute_bets(df, sig, tau, cost=0.0)
            s = summarize(b, label=f"{sig}_tau_{tau:.3f}")
            s["signal"] = sig
            s["threshold"] = tau
            rows.append(s)
    sweep = pd.DataFrame(rows)
    cols = ["signal", "threshold", "n_bets", "total_pnl", "mean_pnl",
            "std_pnl", "sharpe_per_bet", "win_rate"]
    sweep = sweep[cols]
    sweep.to_csv(OUT_TABLES / "backtest_threshold_sweep.csv", index=False)
    print("\nThreshold sweep (no costs):")
    print(sweep.to_string(index=False))

    # Pick the threshold that maximizes Sharpe-per-bet for RF
    rf_rows = sweep[sweep["signal"] == PRIMARY_SIGNAL].copy()
    best_idx = rf_rows["sharpe_per_bet"].idxmax()
    BEST_TAU = float(rf_rows.loc[best_idx, "threshold"])
    print(f"\nBest RF threshold by Sharpe-per-bet: {BEST_TAU:.3f}  "
          f"(Sharpe={rf_rows.loc[best_idx, 'sharpe_per_bet']:.4f})")

    # -----------------------------------------------------------------------
    # 2) Primary strategy: RF, threshold=0 (bet everywhere)
    #    Secondary: RF, best threshold
    # -----------------------------------------------------------------------
    primary = compute_bets(df, PRIMARY_SIGNAL, 0.0, cost=0.0)
    best_b  = compute_bets(df, PRIMARY_SIGNAL, BEST_TAU, cost=0.0)

    # -----------------------------------------------------------------------
    # 3) Per-category breakdown (using primary)
    # -----------------------------------------------------------------------
    by_cat = []
    for cat, g in primary.groupby("category"):
        s = summarize(g, label=f"{PRIMARY_SIGNAL}_cat_{cat}")
        s["category"] = cat
        by_cat.append(s)
    by_cat_df = (pd.DataFrame(by_cat)
                   .sort_values("total_pnl", ascending=False)
                   [["category", "n_bets", "total_pnl", "mean_pnl",
                     "sharpe_per_bet", "win_rate"]])
    by_cat_df.to_csv(OUT_TABLES / "backtest_by_category.csv", index=False)
    print("\nP&L by category (RF, τ=0):")
    print(by_cat_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 4) Per-horizon breakdown
    # -----------------------------------------------------------------------
    by_h = []
    for h, g in primary.groupby("horizon_h"):
        s = summarize(g, label=f"{PRIMARY_SIGNAL}_h_{int(h)}")
        s["horizon_h"] = int(h)
        by_h.append(s)
    by_h_df = (pd.DataFrame(by_h).sort_values("horizon_h")
               [["horizon_h", "n_bets", "total_pnl", "mean_pnl",
                 "sharpe_per_bet", "win_rate"]])
    by_h_df.to_csv(OUT_TABLES / "backtest_by_horizon.csv", index=False)
    print("\nP&L by horizon (RF, τ=0):")
    print(by_h_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 5) Transaction-cost sensitivity at τ=0 and τ=BEST
    # -----------------------------------------------------------------------
    cost_rows = []
    for tau, label in [(0.0, "tau_0"), (BEST_TAU, f"tau_{BEST_TAU:.3f}")]:
        for c in COST_LEVELS:
            b = compute_bets(df, PRIMARY_SIGNAL, tau, cost=c)
            s = summarize(b, label=f"{label}_cost_{c:.3f}")
            s["threshold"] = tau
            s["cost_per_bet"] = c
            cost_rows.append(s)
    cost_df = pd.DataFrame(cost_rows)[
        ["threshold", "cost_per_bet", "n_bets", "total_pnl",
         "mean_pnl", "sharpe_per_bet", "win_rate"]
    ]
    cost_df.to_csv(OUT_TABLES / "backtest_cost_sensitivity.csv", index=False)
    print("\nTransaction cost sensitivity (RF):")
    print(cost_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # 6) Cumulative P&L figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    pri_sorted  = primary.sort_values("close_time")
    best_sorted = best_b.sort_values("close_time")
    en_b        = compute_bets(df, "pred_EN", 0.0, cost=0.0).sort_values("close_time")

    ax.plot(pri_sorted["close_time"],  pri_sorted["payoff"].cumsum(),
            label=f"RF  τ=0  (n={(pri_sorted['side']!=0).sum():,})", color="C0")
    ax.plot(best_sorted["close_time"], best_sorted["payoff"].cumsum(),
            label=f"RF  τ={BEST_TAU:.3f}  (n={(best_sorted['side']!=0).sum():,})",
            color="C2", linestyle="--")
    ax.plot(en_b["close_time"], en_b["payoff"].cumsum(),
            label=f"EN  τ=0  (n={(en_b['side']!=0).sum():,})",
            color="C1", alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Market close time")
    ax.set_ylabel("Cumulative P&L (units of $1)")
    ax.set_title("Cumulative P&L on test set (flat $1 stakes, no costs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "cumulative_pnl.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 7) Threshold sweep figure (total P&L and Sharpe vs threshold)
    # -----------------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    for sig, color in [("pred_RF", "C0"), ("pred_EN", "C1")]:
        sub = sweep[sweep["signal"] == sig]
        axL.plot(sub["threshold"], sub["total_pnl"], "o-", label=sig, color=color)
        axR.plot(sub["threshold"], sub["sharpe_per_bet"], "o-", label=sig, color=color)
    axL.axhline(0, color="k", linewidth=0.5)
    axL.set_xlabel("Threshold |pred|")
    axL.set_ylabel("Total P&L ($)")
    axL.set_title("Total P&L vs threshold")
    axL.grid(alpha=0.3); axL.legend()
    axR.set_xlabel("Threshold |pred|")
    axR.set_ylabel("Sharpe per bet")
    axR.set_title("Sharpe-per-bet vs threshold")
    axR.grid(alpha=0.3); axR.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 8) Save bets parquet (primary + best, with model label)
    # -----------------------------------------------------------------------
    pri_out = primary.assign(model=f"RF_tau_0.000")
    best_out = best_b.assign(model=f"RF_tau_{BEST_TAU:.3f}")
    pd.concat([pri_out, best_out], ignore_index=True).to_parquet(OUT_BETS,
                                                                 index=False)
    print(f"\nSaved bets -> {OUT_BETS}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    s0 = summarize(primary, "RF_tau_0")
    s1 = summarize(best_b,  f"RF_tau_{BEST_TAU:.3f}")
    print("\n==SUMMARY==")
    for tag, s in [("RF τ=0 (all bets)", s0),
                   (f"RF τ={BEST_TAU:.3f} (best Sharpe)", s1)]:
        print(f"  {tag}:")
        print(f"    n_bets       : {s['n_bets']:,}")
        print(f"    total P&L    : ${s['total_pnl']:.2f}")
        print(f"    mean P&L/bet : ${s['mean_pnl']:.4f}")
        print(f"    Sharpe/bet   : {s['sharpe_per_bet']:.4f}")
        print(f"    win rate     : {s['win_rate']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
