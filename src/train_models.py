"""
src/train_models.py (PATCHED v2)
====================

Phase 3: Train ElasticNet and Random Forest to predict the
calibration error (yes_resolution - price_level) on Polymarket
snapshots, then evaluate adjusted-probability Brier scores against
naive and market baselines.

Inputs:
    data/processed/snapshots.parquet

Outputs:
    data/processed/test_predictions.parquet
    results/tables/model_comparison.csv
    results/tables/elasticnet_coefs.csv
    results/tables/rf_feature_importance.csv
    results/tables/per_horizon_brier.csv
    results/tables/decile_table.csv
    results/figures/predicted_vs_actual_decile.png
    results/models/{elasticnet,random_forest,scaler,feature_cols}.joblib
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "snapshots.parquet"

OUT_PRED_PATH = PROJECT_ROOT / "data" / "processed" / "test_predictions.parquet"
OUT_TABLES = PROJECT_ROOT / "results" / "tables"
OUT_FIGURES = PROJECT_ROOT / "results" / "figures"
OUT_MODELS = PROJECT_ROOT / "results" / "models"

for d in (OUT_TABLES, OUT_FIGURES, OUT_MODELS):
    d.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2026-05-07")
DATE_COL = "close_time"
TARGET_COL = "calibration_error"
PRICE_COL = "price_level"
OUTCOME_COL = "yes_resolution"
ACTIVE_EPS = 1e-12
RANDOM_STATE = 42


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Brier score: mean squared error between probability and binary outcome."""
    return float(np.mean((p - y) ** 2))


def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1: Load and validate schema
    # -----------------------------------------------------------------------
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  loaded {len(df):,} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")

    assert DATE_COL in df.columns, (
        f"Expected date column '{DATE_COL}' in snapshots.parquet. "
        f"Available columns: {list(df.columns)}"
    )
    assert TARGET_COL in df.columns, f"Missing target column '{TARGET_COL}'."
    assert PRICE_COL in df.columns, f"Missing price column '{PRICE_COL}'."
    assert OUTCOME_COL in df.columns, f"Missing outcome column '{OUTCOME_COL}'."

    # Coerce date column to tz-naive datetime so comparison with SPLIT_DATE works.
    # Polymarket stores close_time as float Unix epoch seconds; default
    # pd.to_datetime would misread floats as nanoseconds, so dispatch on dtype.
    if pd.api.types.is_numeric_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], unit="s")
        print(f"  parsed {DATE_COL} as Unix epoch seconds")
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        if df[DATE_COL].dt.tz is not None:
            df[DATE_COL] = df[DATE_COL].dt.tz_convert("UTC").dt.tz_localize(None)
        print(f"  parsed {DATE_COL} as datetime")
    print(f"  {DATE_COL} range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")

    # Build the unified horizon variable. Prefer snapshot_offset_h
    # (the design-time offset) over hours_to_expiry (which is derived).
    if "snapshot_offset_h" in df.columns:
        df["horizon_h"] = df["snapshot_offset_h"]
        print("  horizon_h <- snapshot_offset_h")
    elif "hours_to_expiry" in df.columns:
        df["horizon_h"] = df["hours_to_expiry"]
        print("  horizon_h <- hours_to_expiry (fallback)")
    else:
        raise ValueError("Need snapshot_offset_h or hours_to_expiry in parquet.")

    # -----------------------------------------------------------------------
    # Step 2: Active-sample filter (matches Phase 2 EDA rule)
    # -----------------------------------------------------------------------
    pv = df.get("price_volatility", pd.Series(0.0, index=df.index)).fillna(0.0)
    m6 = df.get("momentum_6h",      pd.Series(0.0, index=df.index)).fillna(0.0)
    active_mask = (pv.abs() > ACTIVE_EPS) | (m6.abs() > ACTIVE_EPS)
    df = df.loc[active_mask].copy()
    print(f"  active sample: {len(df):,} rows "
          f"({100 * active_mask.mean():.1f}% of full)")
    if not (24_000 <= len(df) <= 26_500):
        print(f"  WARNING: active-sample count {len(df):,} differs from "
              f"Phase 2 EDA reference (~25,290). Sanity-check the filter rule.")

    # -----------------------------------------------------------------------
    # Step 3: Feature engineering
    # -----------------------------------------------------------------------
    # 3a. Drop momentum_3d entirely (56% null in Phase 2 EDA).
    if "momentum_3d" in df.columns:
        df = df.drop(columns=["momentum_3d"])
        print("  dropped momentum_3d (too many nulls to impute defensibly)")

    # 3b. momentum_1d: median-impute + missingness indicator.
    if "momentum_1d" in df.columns:
        df["momentum_1d_missing"] = df["momentum_1d"].isna().astype(int)
        median_m1d = df["momentum_1d"].median()
        df["momentum_1d"] = df["momentum_1d"].fillna(median_m1d)
        miss_pct = df["momentum_1d_missing"].mean() * 100
        print(f"  imputed momentum_1d nulls with median={median_m1d:.4f} "
              f"({miss_pct:.1f}% flagged via momentum_1d_missing)")

    # 3c. One-hot encode category (drop_first avoids collinearity for ElasticNet).
    cat_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True,
                                 dtype=int)
    df = pd.concat([df, cat_dummies], axis=1)
    print(f"  one-hot encoded category -> {list(cat_dummies.columns)}")

    # 3d. Final feature list. Include horizon_h ONLY (not hours_to_expiry).
    base_features = [
        "price_level",
        "momentum_1h",
        "momentum_6h",
        "momentum_1d",
        "momentum_1d_missing",
        "log_volume",
        "price_volatility",
        "horizon_h",
    ]
    cat_features = list(cat_dummies.columns)
    feature_cols = [c for c in base_features if c in df.columns] + cat_features
    print(f"  final feature set ({len(feature_cols)} cols): {feature_cols}")

    needed = feature_cols + [TARGET_COL, PRICE_COL, OUTCOME_COL, DATE_COL]
    before = len(df)
    df = df.dropna(subset=needed)
    print(f"  dropped {before - len(df):,} rows with NaN in features/target")

    # -----------------------------------------------------------------------
    # Step 4: Chronological train/test split at SPLIT_DATE
    # -----------------------------------------------------------------------
    train = (df.loc[df[DATE_COL] < SPLIT_DATE]
               .sort_values(DATE_COL).reset_index(drop=True))
    test  = (df.loc[df[DATE_COL] >= SPLIT_DATE]
               .sort_values(DATE_COL).reset_index(drop=True))
    print(f"  train: {len(train):,} rows  test: {len(test):,} rows  "
          f"split at {SPLIT_DATE.date()}")

    X_train = train[feature_cols].values
    y_train = train[TARGET_COL].values
    X_test  = test[feature_cols].values
    y_test  = test[TARGET_COL].values

    # -----------------------------------------------------------------------
    # Step 5: Standardize features for ElasticNet (RF doesn't need it)
    # -----------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # Step 6: ElasticNet with TimeSeriesSplit CV
    # -----------------------------------------------------------------------
    print("\nTraining ElasticNet (TimeSeriesSplit CV) ...")
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-4, 0, 30),
        cv=TimeSeriesSplit(n_splits=5),
        max_iter=10_000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    en.fit(X_train_sc, y_train)
    print(f"  best alpha={en.alpha_:.5f}  best l1_ratio={en.l1_ratio_}")

    # -----------------------------------------------------------------------
    # Step 7: Random Forest with RandomizedSearchCV + TimeSeriesSplit
    # -----------------------------------------------------------------------
    print("\nTraining Random Forest (RandomizedSearchCV, TimeSeriesSplit) ...")
    rf_dist = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [4, 8, 16, None],
        "min_samples_leaf":[5, 20, 50],
        "max_features":    ["sqrt", 0.5],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_dist,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    print(f"  best params: {rf_search.best_params_}")

    # -----------------------------------------------------------------------
    # Step 8: Predict on test set + compute baselines
    # -----------------------------------------------------------------------
    pred_en = en.predict(X_test_sc)
    pred_rf = rf.predict(X_test)

    p_market = test[PRICE_COL].values
    p_adj_en = np.clip(p_market + pred_en, 0.0, 1.0)
    p_adj_rf = np.clip(p_market + pred_rf, 0.0, 1.0)
    y = test[OUTCOME_COL].values.astype(float)

    metrics = {
        "naive_0.5":         brier(np.full_like(y, 0.5), y),
        "market":            brier(p_market, y),
        "elasticnet_adj":    brier(p_adj_en, y),
        "random_forest_adj": brier(p_adj_rf, y),
    }

    print("\nBrier scores on test set (lower is better):")
    for k, v in metrics.items():
        print(f"  {k:<22s} {v:.4f}")

    rmse_en = float(np.sqrt(mean_squared_error(y_test, pred_en)))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, pred_rf)))
    r2_en   = float(r2_score(y_test, pred_en))
    r2_rf   = float(r2_score(y_test, pred_rf))

    print("\nResidual regression metrics (target = calibration_error):")
    print(f"  ElasticNet     RMSE={rmse_en:.4f}  R^2={r2_en:.4f}")
    print(f"  Random Forest  RMSE={rmse_rf:.4f}  R^2={r2_rf:.4f}")

    # -----------------------------------------------------------------------
    # Step 9: Per-horizon Brier breakdown (robustness)
    # -----------------------------------------------------------------------
    per_h = []
    for h in sorted(test["horizon_h"].unique()):
        m = (test["horizon_h"].values == h)
        per_h.append({
            "horizon_h":            int(h),
            "n":                    int(m.sum()),
            "brier_naive":          brier(np.full(m.sum(), 0.5), y[m]),
            "brier_market":         brier(p_market[m], y[m]),
            "brier_elasticnet":     brier(p_adj_en[m], y[m]),
            "brier_random_forest":  brier(p_adj_rf[m], y[m]),
        })
    per_h_df = pd.DataFrame(per_h)
    print("\nPer-horizon Brier:")
    print(per_h_df.to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 10: Decile table — does pred_RF rank actual residuals correctly?
    # -----------------------------------------------------------------------
    test = test.assign(
        pred_EN=pred_en,
        pred_RF=pred_rf,
        p_adj_EN=p_adj_en,
        p_adj_RF=p_adj_rf,
    )
    test["pred_decile"] = pd.qcut(test["pred_RF"], q=10, labels=False,
                                  duplicates="drop")
    decile = (test.groupby("pred_decile")
                  .agg(n=("pred_RF", "size"),
                       mean_pred=("pred_RF", "mean"),
                       mean_actual=(TARGET_COL, "mean"),
                       mean_price=(PRICE_COL, "mean"),
                       mean_outcome=(OUTCOME_COL, "mean"))
                  .reset_index())
    print("\nPrediction-decile table (RF):")
    print(decile.to_string(index=False))

    # Decile diagnostic figure.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(decile["mean_pred"], decile["mean_actual"], "o-", label="RF deciles")
    lo = min(decile["mean_pred"].min(), decile["mean_actual"].min())
    hi = max(decile["mean_pred"].max(), decile["mean_actual"].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x (perfect)")
    ax.set_xlabel("Mean predicted calibration error (by decile)")
    ax.set_ylabel("Mean realized calibration error")
    ax.set_title("RF predicted vs realized calibration error (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "predicted_vs_actual_decile.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Step 11: Save all artifacts
    # -----------------------------------------------------------------------
    comp = pd.DataFrame([
        {"model": "naive_0.5",       "brier": metrics["naive_0.5"],
         "rmse_residual": np.nan,    "r2_residual": np.nan},
        {"model": "market_baseline", "brier": metrics["market"],
         "rmse_residual": np.nan,    "r2_residual": np.nan},
        {"model": "elasticnet",      "brier": metrics["elasticnet_adj"],
         "rmse_residual": rmse_en,   "r2_residual": r2_en},
        {"model": "random_forest",   "brier": metrics["random_forest_adj"],
         "rmse_residual": rmse_rf,   "r2_residual": r2_rf},
    ])
    comp.to_csv(OUT_TABLES / "model_comparison.csv", index=False)

    # ElasticNet coefficients in both scaled and original feature units.
    en_coefs = pd.DataFrame({
        "feature":             feature_cols,
        "coef_scaled":         en.coef_,
        "coef_original_units": en.coef_ / scaler.scale_,
    })
    en_coefs = en_coefs.iloc[np.argsort(-np.abs(en_coefs["coef_scaled"].values))]
    en_coefs.to_csv(OUT_TABLES / "elasticnet_coefs.csv", index=False)

    rf_imp = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    rf_imp.to_csv(OUT_TABLES / "rf_feature_importance.csv", index=False)

    per_h_df.to_csv(OUT_TABLES / "per_horizon_brier.csv", index=False)
    decile.to_csv(OUT_TABLES / "decile_table.csv", index=False)

    # Test predictions for backtest.py to consume.
    keep_cols = [c for c in [
        "market_id", DATE_COL, "horizon_h", PRICE_COL, OUTCOME_COL,
        TARGET_COL, "category", "pred_EN", "pred_RF", "p_adj_EN", "p_adj_RF",
    ] if c in test.columns]
    test[keep_cols].to_parquet(OUT_PRED_PATH, index=False)
    print(f"\nSaved test predictions -> {OUT_PRED_PATH}")

    joblib.dump(en,           OUT_MODELS / "elasticnet.joblib")
    joblib.dump(rf,           OUT_MODELS / "random_forest.joblib")
    joblib.dump(scaler,       OUT_MODELS / "scaler.joblib")
    joblib.dump(feature_cols, OUT_MODELS / "feature_cols.joblib")
    print(f"Saved models    -> {OUT_MODELS}")

    print("\n==SUMMARY==")
    print(f"  Brier  naive   : {metrics['naive_0.5']:.4f}")
    print(f"  Brier  market  : {metrics['market']:.4f}")
    print(f"  Brier  EN adj  : {metrics['elasticnet_adj']:.4f}")
    print(f"  Brier  RF adj  : {metrics['random_forest_adj']:.4f}")
    print(f"  Train rows     : {len(train):,}")
    print(f"  Test  rows     : {len(test):,}")
    print("Done.")


if __name__ == "__main__":
    main()