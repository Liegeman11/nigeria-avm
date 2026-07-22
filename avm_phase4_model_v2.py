"""
Nigerian AVM V2 — Phase 4 (Modelling) + Phase 5 (Evaluation)
Trains on the expanded dataset with LGA quality scores.
Compares directly against V1 (28.4% MdAPE) to measure improvement.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

print("=" * 55)
print("PHASE 4 (V2) — MODELLING")
print("=" * 55)

df = pd.read_csv("nigeria_property_clean_v2.csv")
print(f"Loaded {len(df)} clean listings (V1 had 1,635)\n")

type_cols = [c for c in df.columns if c.startswith("type_")]

FEATURES = [
    "bedrooms", "bathrooms", "bath_per_bed",
    "lga_median_log_price", "lga_mean_log_price", "lga_count",
    "lga_quality_score",  # NEW — independent location signal
    "state_median_log_price",
    "is_premium_area",
    "is_furnished", "is_serviced", "is_newly_built",
    "has_cof_o", "has_bq", "has_pool", "is_gated",
] + type_cols

TARGET = "log_price"

model_df = df[FEATURES + [TARGET, "price_ngn", "lga", "state"]].dropna()
print(f"Modelling dataset: {len(model_df)} rows × {len(FEATURES)} features")

X = model_df[FEATURES]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

def evaluate(model, name):
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = (np.abs(y_true - y_pred) / y_true).mean() * 100
    mdape = (np.abs(y_true - y_pred) / y_true).median() * 100

    print(f"  {name}")
    print(f"    MAE:   NGN {mae:>15,.0f}")
    print(f"    RMSE:  NGN {rmse:>15,.0f}")
    print(f"    MAPE:  {mape:>6.1f}%")
    print(f"    MdAPE: {mdape:>6.1f}%\n")
    return y_pred_log, mape, mdape

print("-- Baseline: Ridge Regression --------------------------")
ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
ridge.fit(X_train, y_train)
ridge_pred_log, ridge_mape, ridge_mdape = evaluate(ridge, "Ridge Regression")

print("-- Main model: XGBoost -----------------------------------")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.04, max_depth=5,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    early_stopping_rounds=30, eval_metric="rmse", verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred_log, xgb_mape, xgb_mdape = evaluate(xgb_model, "XGBoost")

# ── V1 vs V2 comparison ──
V1_MDAPE = 28.4
improvement = V1_MDAPE - xgb_mdape

print("=" * 55)
print("V1 vs V2 COMPARISON")
print("=" * 55)
print(f"  V1 MdAPE (1,635 listings, no quality score):  {V1_MDAPE:.1f}%")
print(f"  V2 MdAPE ({len(model_df)} listings, +quality score): {xgb_mdape:.1f}%")
if improvement > 0:
    print(f"  Improvement: -{improvement:.1f} percentage points ({improvement/V1_MDAPE*100:.0f}% relative reduction)")
else:
    print(f"  Change: +{-improvement:.1f} percentage points (worse — investigate)")

# ─────────────────────────────────────────────────────────────
# PHASE 5 — EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 5 (V2) — EVALUATION")
print("=" * 55)

y_pred_ngn = np.expm1(xgb_pred_log)
y_true_ngn = np.expm1(y_test)
pct_errors = ((y_pred_ngn - y_true_ngn) / y_true_ngn * 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Nigerian AVM V2 — XGBoost Model Evaluation (MdAPE {xgb_mdape:.1f}%)",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.scatter(y_true_ngn / 1e6, y_pred_ngn / 1e6, alpha=0.4, s=20, color="#4C72B0")
lims = [0, min(y_true_ngn.max(), y_pred_ngn.max()) / 1e6 * 1.05]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual price (NGN M)")
ax.set_ylabel("Predicted price (NGN M)")
ax.set_title("Predicted vs Actual")
ax.legend()

ax = axes[1]
ax.hist(pct_errors.clip(-100, 100), bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Prediction error (%)")
ax.set_ylabel("Count")
ax.set_title(f"Error Distribution\nMdAPE = {xgb_mdape:.1f}% (V1 was 28.4%)")

ax = axes[2]
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(12)
importances.plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Top 12 Feature Importances")
ax.set_xlabel("Importance score")

plt.tight_layout()
plt.savefig("avm_evaluation_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved avm_evaluation_v2.png")

# ── LGA summary ──
lga_summary = (
    model_df.groupby(["state", "lga"])
    .agg(median_price_M=("price_ngn", lambda x: x.median() / 1e6), count=("price_ngn", "count"))
    .reset_index().query("count >= 5").sort_values("median_price_M", ascending=False)
)
lga_summary.to_csv("lga_price_summary_v2.csv", index=False)
print("[✓] Saved lga_price_summary_v2.csv")
print(f"\nTop 10 LGAs by median price:")
print(lga_summary.head(10).to_string(index=False))
print(f"\nBottom 10 LGAs by median price:")
print(lga_summary.tail(10).to_string(index=False))

# ── Save model ──
import pickle
with open("avm_model_v2.pkl", "wb") as f:
    pickle.dump({"model": xgb_model, "features": FEATURES}, f)
print("[✓] Saved avm_model_v2.pkl")

print(f"\n{'='*55}")
print(f"  V2 COMPLETE — MdAPE: {xgb_mdape:.1f}% (down from V1's 28.4%)")
print(f"{'='*55}")
