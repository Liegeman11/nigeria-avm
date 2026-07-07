"""
Nigerian AVM — Phase 4 (Modelling) + Phase 5 (Evaluation & SHAP)
Run after avm_phase2_3.py has produced nigeria_property_clean.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# ─────────────────────────────────────────────────────────────
# PHASE 4 — MODELLING
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 4 — MODELLING")
print("=" * 55)

df = pd.read_csv("nigeria_property_clean.csv")
print(f"Loaded {len(df)} clean listings\n")

# ── 4.1  Define feature set ───────────────────────────────────
type_cols = [c for c in df.columns if c.startswith("type_")]

FEATURES = [
    "bedrooms", "bathrooms", "bath_per_bed",
    "lga_median_log_price", "lga_mean_log_price", "lga_count",
    "state_median_log_price",
    "is_premium_area",
    "is_furnished", "is_serviced", "is_newly_built",
    "has_cof_o", "has_bq", "has_pool", "is_gated",
] + type_cols

TARGET = "log_price"

# Drop rows with any NaN in features or target
model_df = df[FEATURES + [TARGET, "price_ngn", "lga", "state"]].dropna()
print(f"Modelling dataset: {len(model_df)} rows × {len(FEATURES)} features")

X = model_df[FEATURES]
y = model_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Helper: evaluate on test set ──────────────────────────────
def evaluate(model, name):
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = (np.abs(y_true - y_pred) / y_true).mean() * 100
    # Median absolute percentage error (more robust)
    mdape = (np.abs(y_true - y_pred) / y_true).median() * 100

    print(f"  {name}")
    print(f"    MAE:   ₦{mae:>15,.0f}")
    print(f"    RMSE:  ₦{rmse:>15,.0f}")
    print(f"    MAPE:  {mape:>6.1f}%   (mean % error)")
    print(f"    MdAPE: {mdape:>6.1f}%   (median % error — more robust)\n")
    return y_pred_log, mape, mdape

# ── 4.2  Baseline — Ridge Regression ─────────────────────────
print("── Baseline: Ridge Regression ──────────────────────────")
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=10.0))
])
ridge.fit(X_train, y_train)
ridge_pred_log, ridge_mape, _ = evaluate(ridge, "Ridge Regression")

# ── 4.3  Main model — XGBoost ─────────────────────────────────
print("── Main model: XGBoost ──────────────────────────────────")
xgb_model = xgb.XGBRegressor(
    n_estimators   = 500,
    learning_rate  = 0.04,
    max_depth      = 5,
    min_child_weight = 3,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    reg_alpha      = 0.1,
    reg_lambda     = 1.0,
    random_state   = 42,
    early_stopping_rounds = 30,
    eval_metric    = "rmse",
    verbosity      = 0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_pred_log, xgb_mape, xgb_mdape = evaluate(xgb_model, "XGBoost")

# Best model
best_model = xgb_model
best_pred_log = xgb_pred_log

# ─────────────────────────────────────────────────────────────
# PHASE 5 — EVALUATION PLOTS + SHAP
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 5 — EVALUATION & EXPLAINABILITY")
print("=" * 55)

y_pred_ngn = np.expm1(best_pred_log)
y_true_ngn = np.expm1(y_test)
pct_errors = ((y_pred_ngn - y_true_ngn) / y_true_ngn * 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Nigerian AVM — XGBoost Model Evaluation", fontsize=14, fontweight="bold")

# ── Plot 1: Predicted vs Actual ───────────────────────────────
ax = axes[0]
ax.scatter(y_true_ngn / 1e6, y_pred_ngn / 1e6,
           alpha=0.4, s=20, color="#4C72B0")
lims = [0, min(y_true_ngn.max(), y_pred_ngn.max()) / 1e6 * 1.05]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual price (₦M)")
ax.set_ylabel("Predicted price (₦M)")
ax.set_title("Predicted vs Actual")
ax.legend()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₦{x:.0f}M"))

# ── Plot 2: Residuals distribution ───────────────────────────
ax = axes[1]
ax.hist(pct_errors.clip(-100, 100), bins=40,
        color="#55A868", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xlabel("Prediction error (%)")
ax.set_ylabel("Count")
ax.set_title(f"Error Distribution\nMdAPE = {xgb_mdape:.1f}%")

# ── Plot 3: Feature importance ────────────────────────────────
ax = axes[2]
importances = pd.Series(
    best_model.feature_importances_, index=FEATURES
).sort_values(ascending=True).tail(12)
importances.plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Top 12 Feature Importances")
ax.set_xlabel("Importance score")

plt.tight_layout()
plt.savefig("avm_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("[✓] Saved avm_evaluation.png")

# ── SHAP explainability ───────────────────────────────────────
try:
    import shap
    print("\n── SHAP Explainability ──────────────────────────────────")
    explainer   = shap.Explainer(best_model)
    shap_values = explainer(X_test)

    # SHAP summary bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Nigerian AVM")
    plt.tight_layout()
    plt.savefig("avm_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] Saved avm_shap_importance.png")

    # Single prediction explanation (first test row)
    plt.figure(figsize=(10, 4))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title("SHAP — Single Property Explanation")
    plt.tight_layout()
    plt.savefig("avm_shap_single.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] Saved avm_shap_single.png")

except ImportError:
    print("[!] shap not installed. Run: pip install shap")
    print("    Skipping SHAP plots — all other outputs are complete.")

# ── Price by LGA heatmap ──────────────────────────────────────
print("\n── LGA Price Summary ────────────────────────────────────")
lga_summary = (
    model_df.groupby(["state", "lga"])
    .agg(
        median_price_M=("price_ngn", lambda x: x.median() / 1e6),
        count=("price_ngn", "count")
    )
    .reset_index()
    .query("count >= 5")   # only LGAs with 5+ listings
    .sort_values("median_price_M", ascending=False)
)
print(lga_summary.head(15).to_string(index=False))
lga_summary.to_csv("lga_price_summary.csv", index=False)
print("[✓] Saved lga_price_summary.csv")

# ── Save model ────────────────────────────────────────────────
import pickle
with open("avm_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURES}, f)
print("[✓] Saved avm_model.pkl")

print(f"\n{'='*55}")
print(f"  ALL PHASES COMPLETE")
print(f"  Model MAPE:  {xgb_mape:.1f}%")
print(f"  Model MdAPE: {xgb_mdape:.1f}%")
print(f"{'='*55}")
print("\nFiles produced:")
print("  nigeria_property_clean.csv  — cleaned dataset")
print("  avm_evaluation.png          — 3-panel evaluation chart")
print("  avm_shap_importance.png     — SHAP feature importance")
print("  avm_shap_single.png         — single prediction explanation")
print("  lga_price_summary.csv       — median price per LGA")
print("  avm_model.pkl               — trained model (for deployment)")
print("\nNext: run avm_deploy_api.py to serve predictions via FastAPI")

