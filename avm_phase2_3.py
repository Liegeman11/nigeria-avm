"""
Nigerian AVM — Phase 2 (Cleaning) + Phase 3 (Feature Engineering)
Run this after nigeria_avm_scraper_final.py has produced nigeria_property_raw.csv
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PHASE 2 — DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 2 — DATA CLEANING")
print("=" * 55)

df = pd.read_csv("nigeria_property_raw.csv")
print(f"\nRaw shape: {df.shape}")
print(f"Columns:   {list(df.columns)}\n")

# ── 2.1  Remove Land (no beds/baths — not suitable for AVM) ──
df = df[df["prop_type"].str.lower() != "land"].copy()
print(f"After removing Land listings: {len(df)} rows")

# ── 2.2  Focus on residential only ────────────────────────────
residential_types = ["house", "flat apartment", "flat / apartment",
                     "detached", "semi-detached", "terrace", "bungalow",
                     "maisonette", "duplex"]
mask = df["prop_type"].str.lower().str.contains(
    "|".join(residential_types), na=False
)
df = df[mask].copy()
print(f"After keeping residential only: {len(df)} rows")

# ── 2.3  Price outlier removal ────────────────────────────────
# Floor:  ₦5M  (below this = data error or non-residential)
# Ceiling: ₦5B (above this = ultra-luxury, too sparse to model)
df = df[(df["price_ngn"] >= 5_000_000) &
        (df["price_ngn"] <= 5_000_000_000)].copy()
print(f"After price floor/ceiling filter: {len(df)} rows")

# ── 2.4  Drop rows with no bedrooms ──────────────────────────
df = df[df["bedrooms"].notna()].copy()
df["bedrooms"]  = df["bedrooms"].astype(int)
df["bathrooms"] = df["bathrooms"].fillna(df["bedrooms"]).astype(int)
print(f"After dropping missing bedrooms: {len(df)} rows")

# ── 2.5  Bedroom sanity check (1–10) ─────────────────────────
df = df[(df["bedrooms"] >= 1) & (df["bedrooms"] <= 10)].copy()
print(f"After bedroom sanity (1-10):     {len(df)} rows")

# ── 2.6  Log-transform price ──────────────────────────────────
df["log_price"] = np.log1p(df["price_ngn"])

# ── 2.7  Standardise state names ─────────────────────────────
state_map = {
    "fct": "Abuja", "abuja": "Abuja", "fct abuja": "Abuja",
    "lagos": "Lagos", "oyo": "Oyo", "ogun": "Ogun",
    "rivers": "Rivers", "delta": "Delta", "anambra": "Anambra",
}
df["state"] = df["state"].str.strip().str.lower().map(
    lambda x: state_map.get(x, x.title())
)

# ── 2.8  Standardise property type ───────────────────────────
def clean_type(t):
    t = str(t).lower()
    if "flat" in t or "apartment" in t: return "Flat/Apartment"
    if "detached" in t and "semi" not in t: return "Detached House"
    if "semi" in t: return "Semi-Detached"
    if "terrace" in t: return "Terrace"
    if "bungalow" in t: return "Bungalow"
    if "duplex" in t: return "Duplex"
    return "House"

df["prop_type_clean"] = df["prop_type"].apply(clean_type)

# ── 2.9  Parse date_added to datetime ────────────────────────
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce",
                                   dayfirst=True)
df["days_on_market"] = (pd.Timestamp("today") - df["date_added"]).dt.days

print(f"\nCleaning complete. Final shape: {df.shape}")
print(f"\nState distribution:\n{df['state'].value_counts()}")
print(f"\nProperty type distribution:\n{df['prop_type_clean'].value_counts()}")
print(f"\nPrice summary (₦):")
print(df["price_ngn"].describe().apply(lambda x: f"₦{x:,.0f}"))

# ─────────────────────────────────────────────────────────────
# PHASE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 — FEATURE ENGINEERING")
print("=" * 55)

# ── 3.1  LGA-level price statistics (powerful proxy for location) ──
lga_stats = df.groupby("lga")["log_price"].agg(
    lga_median_log_price="median",
    lga_mean_log_price="mean",
    lga_count="count"
).reset_index()
df = df.merge(lga_stats, on="lga", how="left")
print(f"\n[✓] LGA price stats added ({df['lga'].nunique()} unique LGAs)")

# ── 3.2  State-level encoding ──────────────────────────────────
state_price = df.groupby("state")["log_price"].median().rename("state_median_log_price")
df = df.join(state_price, on="state")
print(f"[✓] State median price encoded")

# ── 3.3  Property type dummies ────────────────────────────────
type_dummies = pd.get_dummies(df["prop_type_clean"], prefix="type", drop_first=True)
df = pd.concat([df, type_dummies], axis=1)
print(f"[✓] Property type dummies: {list(type_dummies.columns)}")

# ── 3.4  Feature flags from listing features ─────────────────
feature_flags = {
    "is_furnished":  r"furnished",
    "is_serviced":   r"serviced",
    "is_newly_built":r"newly built|new build",
    "has_cof_o":     r"c\s*of\s*o|cofo",
    "has_bq":        r"\bbq\b|boys? quarter",
    "has_pool":      r"pool|swimming",
    "is_gated":      r"gated|estate",
}
for col, pattern in feature_flags.items():
    df[col] = (
        df["features"].fillna("").str.lower().str.contains(pattern, regex=True) |
        df["title"].fillna("").str.lower().str.contains(pattern, regex=True)
    ).astype(int)
    count = df[col].sum()
    print(f"[✓] {col}: {count} listings ({count/len(df)*100:.1f}%)")

# ── 3.5  Bed/bath ratio (luxury signal) ──────────────────────
df["bath_per_bed"] = (df["bathrooms"] / df["bedrooms"]).round(2)
df["bath_per_bed"] = df["bath_per_bed"].clip(0.5, 3.0)   # cap extremes
print(f"[✓] bath_per_bed ratio added")

# ── 3.6  Premium area flag (Lagos top LGAs) ──────────────────
premium_lgas = {
    "Ikoyi", "Victoria Island", "Lekki", "Banana Island",
    "Maitama", "Asokoro", "Wuse 2", "Jabi",   # Abuja premium
}
df["is_premium_area"] = df["lga"].isin(premium_lgas).astype(int)
print(f"[✓] is_premium_area: {df['is_premium_area'].sum()} listings")

# ── 3.7  Save cleaned + engineered dataset ────────────────────
df.to_csv("nigeria_property_clean.csv", index=False)
print(f"\n[✓] Saved to nigeria_property_clean.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")

# ── 3.8  Final feature summary ────────────────────────────────
model_features = [
    "bedrooms", "bathrooms", "bath_per_bed",
    "lga_median_log_price", "lga_mean_log_price", "lga_count",
    "state_median_log_price",
    "is_premium_area",
    "is_furnished", "is_serviced", "is_newly_built",
    "has_cof_o", "has_bq", "has_pool", "is_gated",
    "days_on_market",
] + list(type_dummies.columns)

print(f"\n{'='*55}")
print(f"MODEL-READY FEATURES ({len(model_features)} total):")
print(f"{'='*55}")
for f in model_features:
    non_null = df[f].notna().sum()
    print(f"  {f:<30} {non_null:>5} non-null  ({non_null/len(df)*100:.0f}%)")

print(f"\nTarget variable: log_price (log of price_ngn)")
print(f"\nReady for Phase 4 — Modelling!")
print(f"Next cell: run avm_phase4_model.py")

