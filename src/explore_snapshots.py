"""
src/explore_snapshots.py

Part I analysis: cross-sectional calibration of Polymarket prices.
Reads:  data/processed/snapshots.parquet
Writes: results/figures/*.png, results/tables/*.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
SNAP_PATH    = Path("data/processed/snapshots.parquet")
FIG_DIR      = Path("results/figures")
TBL_DIR      = Path("results/tables")
N_BINS       = 10
N_BOOTSTRAP  = 1000
RNG_SEED     = 42

# Active-sample filter: keep rows with evidence of recent price activity
# based on nonzero recent volatility or nonzero 6h momentum.
def active_mask(df):
    """Keep rows with evidence of recent price activity.
    Operationalized as nonzero 6h momentum OR nonzero recent volatility."""
    return (
        (df["price_volatility"].fillna(0) > 0) |
        (df["momentum_6h"].fillna(0).abs() > 1e-6)
    )

# Thesis-quality matplotlib defaults
plt.rcParams.update({
    "figure.figsize":  (7, 5.5),
    "figure.dpi":      120,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

# ---------- Bootstrap helpers ----------
def bootstrap_ci(values, statistic=np.mean, n_boot=N_BOOTSTRAP, alpha=0.05, seed=RNG_SEED):
    """Returns (lower, upper) percentile CI for a statistic of `values`."""
    rng = np.random.default_rng(seed)
    n   = len(values)
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = statistic(values[idx], axis=1)
    return float(np.quantile(boot, alpha/2)), float(np.quantile(boot, 1 - alpha/2))


def calibration_table(df, bin_edges=None):
    """Bin prices and compute calibration stats with bootstrap CIs on empirical rate."""
    if bin_edges is None:
        bin_edges = np.linspace(0, 1, N_BINS + 1)
    d = df.dropna(subset=["price_level", "yes_resolution"]).copy()
    d["price_bin"] = pd.cut(d["price_level"], bins=bin_edges, include_lowest=True)
    rows = []
    for b, sub in d.groupby("price_bin", observed=True):
        outcomes = sub["yes_resolution"].to_numpy()
        emp_rate = float(outcomes.mean())
        avg_p    = float(sub["price_level"].mean())
        lo, hi   = bootstrap_ci(outcomes)
        # Significance: does the CI on the empirical rate exclude the average price?
        significant = not (lo <= avg_p <= hi)
        rows.append({
            "price_bin":          str(b),
            "n":                  len(sub),
            "avg_price":          avg_p,
            "empirical_yes_rate": emp_rate,
            "ci_low":             lo,
            "ci_high":            hi,
            "calibration_error":  emp_rate - avg_p,
            "significant":        significant,
        })
    return pd.DataFrame(rows)

# ---------- Plot helpers ----------
def plot_calibration_curve(cal_df, ax=None, label="Empirical", color="C0",
                           show_ci=True, show_n=True):
    """Plot one calibration curve with CIs and sample-size annotations."""
    if ax is None:
        fig, ax = plt.subplots()
    x = cal_df["avg_price"].to_numpy()
    y = cal_df["empirical_yes_rate"].to_numpy()
    if show_ci:
        yerr = np.vstack([y - cal_df["ci_low"], cal_df["ci_high"] - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, label=label, color=color)
    else:
        ax.plot(x, y, "o-", label=label, color=color)

    # Add perfect-calibration diagonal only once per axis
    existing_labels = [text.get_text() for text in ax.get_legend().get_texts()] if ax.get_legend() else []
    if "Perfect calibration" not in existing_labels:
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    else:
        ax.plot([0, 1], [0, 1], "--", color="gray")

    if show_n:
        for xi, yi, n in zip(x, y, cal_df["n"]):
            ax.annotate(f"n={n:,}", (xi, yi),
                        textcoords="offset points", xytext=(5, -10), fontsize=7,
                        color="gray")
    ax.set_xlabel("Average market price (implied probability)")
    ax.set_ylabel("Empirical YES resolution rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left")
    return ax

# ---------- Section 1: Main calibration (full + active sample) ----------
def section_main_calibration(df):
    print("\n=== Section 1: Main calibration ===")
    cal_full   = calibration_table(df)
    cal_active = calibration_table(df[active_mask(df)])

    fig, ax = plt.subplots()
    plot_calibration_curve(cal_full,   ax=ax, label="Full sample",   color="C0")
    plot_calibration_curve(cal_active, ax=ax, label="Active sample", color="C3",
                           show_ci=True, show_n=False)
    ax.set_title("Polymarket calibration: market price vs. empirical outcome\n"
                 "(error bars: 95% bootstrap CI on empirical rate)")
    fig.savefig(FIG_DIR / "01_calibration_curve_main.png")
    plt.close(fig)

    # Combine into one table with full/active suffixes
    out = cal_full.merge(cal_active, on="price_bin", suffixes=("_full", "_active"))
    out.to_csv(TBL_DIR / "calibration_main.csv", index=False)

    # Print favorite-longshot summary
    print(f"  Full sample:   {len(df):,} rows")
    print(f"  Active sample: {active_mask(df).sum():,} rows "
          f"({100*active_mask(df).mean():.1f}%)")
    print("\n  Calibration errors by bin (full sample):")
    for _, r in cal_full.iterrows():
        flag = " *" if r["significant"] else "  "
        print(f"    {r['price_bin']:>18s}  n={r['n']:>6,}  "
              f"price={r['avg_price']:.3f}  emp={r['empirical_yes_rate']:.3f}  "
              f"err={r['calibration_error']:+.3f}{flag}")
    print("  (* = 95% CI on empirical rate excludes avg market price)")

    # Save explicit favorite-longshot bias table
    flb = cal_full.copy()
    flb["bias_direction"] = np.where(
        flb["calibration_error"] > 0, "underpriced", "overpriced"
    )
    flb.to_csv(TBL_DIR / "flb_summary.csv", index=False)

    return cal_full, cal_active

# ---------- Section 2: Calibration by snapshot horizon ----------
def section_by_horizon(df):
    print("\n=== Section 2: Calibration by horizon ===")
    horizons = [int(h) for h in sorted(df["snapshot_offset_h"].dropna().unique())]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    rows = []
    for ax, h in zip(axes.flat, horizons):
        sub = df[df["snapshot_offset_h"] == h]
        cal = calibration_table(sub)
        plot_calibration_curve(cal, ax=ax, label=f"T−{int(h)}h", color="C0",
                               show_n=False)
        ax.set_title(f"T−{int(h)}h before close (n={len(sub):,})")
        cal["horizon_h"] = h
        rows.append(cal)
    fig.suptitle("Calibration by pre-resolution horizon", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_calibration_by_horizon.png")
    plt.close(fig)

    pd.concat(rows).to_csv(TBL_DIR / "calibration_by_horizon.csv", index=False)
    print(f"  Computed for horizons: {horizons}")


# ---------- Section 3: Calibration by category ----------
def section_by_category(df, min_n=300):
    print("\n=== Section 3: Calibration by category ===")
    cats = (df["category"].value_counts()
            .loc[lambda s: s >= min_n].index.tolist())
    fig, ax = plt.subplots()
    plot_calibration_curve(calibration_table(df), ax=ax,
                           label="All", color="black", show_n=False)
    colors = ["C0", "C1", "C2", "C3", "C4"]
    rows = []
    for cat, c in zip(cats, colors):
        sub = df[df["category"] == cat]
        cal = calibration_table(sub)
        plot_calibration_curve(cal, ax=ax, label=f"{cat} (n={len(sub):,})",
                               color=c, show_ci=False, show_n=False)
        cal["category"] = cat
        rows.append(cal)
    ax.set_title("Calibration by market category")
    fig.savefig(FIG_DIR / "03_calibration_by_category.png")
    plt.close(fig)
    pd.concat(rows).to_csv(TBL_DIR / "calibration_by_category.csv", index=False)
    print(f"  Categories shown (>= {min_n} rows): {cats}")

# ---------- Section 4: Brier breakdown ----------
def section_brier(df):
    print("\n=== Section 4: Brier scores ===")
    d = df.copy()
    d["brier"] = (d["price_level"] - d["yes_resolution"]) ** 2

    overall = d["brier"].mean()
    by_horizon  = (d.groupby("snapshot_offset_h")
                    .agg(n=("brier","size"), brier=("brier","mean"),
                         avg_price=("price_level","mean"),
                         yes_rate=("yes_resolution","mean")).reset_index())
    by_category = (d.groupby("category")
                    .agg(n=("brier","size"), brier=("brier","mean"),
                         avg_price=("price_level","mean"),
                         yes_rate=("yes_resolution","mean"))
                    .sort_values("n", ascending=False).reset_index())

    summary = pd.concat([
        pd.DataFrame([{"group":"OVERALL", "level":"-", "n":len(d),
                       "brier":overall, "avg_price":d["price_level"].mean(),
                       "yes_rate":d["yes_resolution"].mean()}]),
        by_horizon.assign(group="horizon").rename(columns={"snapshot_offset_h":"level"}),
        by_category.assign(group="category").rename(columns={"category":"level"}),
    ], ignore_index=True)[["group","level","n","brier","avg_price","yes_rate"]]
    summary.to_csv(TBL_DIR / "brier_summary.csv", index=False)

    # Bar chart of Brier by category
    fig, ax = plt.subplots()
    ax.bar(by_category["category"], by_category["brier"], color="C0")
    ax.axhline(0.25, color="gray", linestyle="--", label="Naive 50/50 forecast = 0.25")
    ax.axhline(overall, color="C3", linestyle="--", label=f"Overall = {overall:.3f}")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Brier score by market category")
    ax.legend()
    fig.savefig(FIG_DIR / "04_brier_by_category.png")
    plt.close(fig)

    print(f"  Overall Brier: {overall:.4f} (naive 50/50 = 0.25)")
    print(f"  By horizon:\n{by_horizon.round(4).to_string(index=False)}")
    print(f"  By category:\n{by_category.round(4).to_string(index=False)}")

# ---------- Section 5: Price distribution (motivation for 0.50 discussion) ----------
def section_price_dist(df):
    print("\n=== Section 5: Price distribution ===")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(df["price_level"], bins=50, color="C0", alpha=0.8)
    axes[0].set_title(f"Full sample (n={len(df):,})")
    axes[0].set_xlabel("YES price"); axes[0].set_ylabel("Count")
    sub = df[active_mask(df)]
    axes[1].hist(sub["price_level"], bins=50, color="C3", alpha=0.8)
    axes[1].set_title(f"Active sample (n={len(sub):,})")
    axes[1].set_xlabel("YES price")
    fig.suptitle("Distribution of snapshot prices: full vs active sample",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_price_distribution.png")
    plt.close(fig)
    pct_at_half = float((np.isclose(df["price_level"], 0.50) |
                         np.isclose(df["price_level"], 0.5050)).mean())
    print(f"  Fraction of rows at exactly 0.50/0.5050: {pct_at_half:.1%}")

# ---------- Main ----------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SNAP_PATH}…")
    df = pd.read_parquet(SNAP_PATH)
    print(f"  {len(df):,} rows × {df.shape[1]} cols")

    cal_full, cal_active = section_main_calibration(df)
    section_by_horizon(df)
    section_by_category(df)
    section_brier(df)
    section_price_dist(df)

    print(f"\nDone. Figures → {FIG_DIR}/   Tables → {TBL_DIR}/")


if __name__ == "__main__":
    main()