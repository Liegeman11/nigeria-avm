"""
Nigerian AVM V2 — Phase 2 (Cleaning) + Phase 3 (Feature Engineering)
Uses the expanded dataset: nigeria_property_raw_v2_midmarket.csv (4,143 listings)
Adds manual LGA quality scores as an independent location signal.
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# LGA QUALITY SCORES (manual, independent of sample size)
# ─────────────────────────────────────────────────────────────
LGA_QUALITY_SCORES = {
    "Ikoyi": 10, "Victoria Island": 10, "Banana Island": 10, "Eko Atlantic": 10,
    "Lekki": 9, "Lekki Phase 1": 9,
    "Ikeja": 8, "Ikeja Gra": 8, "GRA Ikeja": 8,
    "Maryland": 7, "Magodo": 7, "Omole": 7, "Opebi": 7, "Allen": 7, "Oregun": 7,
    "Yaba": 6, "Gbagada": 6,
    "Surulere": 5, "Ojodu": 5, "Ojota": 5, "Ketu": 5, "Mile 12": 5, "Shomolu": 5,
    "Ikorodu": 4, "Agege": 3, "Mushin": 3, "Oshodi": 3, "Isolo": 3,
    "Alimosho": 3, "Badagry": 3,
    "Maitama": 10, "Asokoro": 10,
    "Wuse 2": 9, "Jabi": 8, "Wuse": 7, "Garki": 7, "Guzape": 8, "Katampe": 7,
    "Kubwa": 4, "Lugbe": 4, "Karu": 3, "Nyanya": 3, "Gwagwalada": 3,
    "Ibadan": 4, "GRA Ibadan": 7, "Bodija": 6, "Oluyole": 5, "Agodi": 6,
    "DEFAULT": 4,
}

def get_lga_score(lga: str) -> int:
    if not lga or pd.isna(lga):
        return LGA_QUALITY_SCORES["DEFAULT"]
    lga_clean = str(lga).strip().title()
    if lga_clean in LGA_QUALITY_SCORES:
        return LGA_QUALITY_SCORES[lga_clean]
    for key, score in LGA_QUALITY_SCORES.items():
        if key.lower() in lga_clean.lower() or lga_clean.lower() in key.lower():
            return score
    return LGA_QUALITY_SCORES["DEFAULT"]


# ─────────────────────────────────────────────────────────────
# PHASE 2 — DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("PHASE 2 (V2) — DATA CLEANING")
print("=" * 55)

df = pd.read_csv("nigeria_property_raw_v2_midmarket.csv")
print(f"\nRaw shape: {df.shape}")

# ── 2.1  Remove Land ──
df = df[df["prop_type"].str.lower() != "land"].copy()
print(f"After removing Land listings: {len(df)} rows")

# ── 2.2  Residential only ──
residential_types = ["house", "flat apartment", "flat / apartment",
                     "detached", "semi-detached", "terrace", "bungalow",
                     "maisonette", "duplex"]
mask = df["prop_type"].str.lower().str.contains("|".join(residential_types), na=False)
df = df[mask].copy()
print(f"After keeping residential only: {len(df)} rows")

# ── 2.3  Price floor/ceiling ──
df = df[(df["price_ngn"] >= 5_000_000) & (df["price_ngn"] <= 5_000_000_000)].copy()
print(f"After price floor/ceiling filter: {len(df)} rows")

# ── 2.4  Drop missing bedrooms ──
df = df[df["bedrooms"].notna()].copy()
df["bedrooms"]  = df["bedrooms"].astype(int)
df["bathrooms"] = df["bathrooms"].fillna(df["bedrooms"]).astype(int)
print(f"After dropping missing bedrooms: {len(df)} rows")

# ── 2.5  Bedroom sanity (1-10) ──
df = df[(df["bedrooms"] >= 1) & (df["bedrooms"] <= 10)].copy()
print(f"After bedroom sanity (1-10):     {len(df)} rows")

# ── 2.6  Deduplicate on PID (in case v1 and midmarket overlap) ──
df.drop_duplicates(subset=["pid"], inplace=True)
print(f"After deduplication on PID:      {len(df)} rows")

# ── 2.7  Log-transform price ──
df["log_price"] = np.log1p(df["price_ngn"])

# ── 2.8  Standardise state ──
state_map = {
    "fct": "Abuja", "abuja": "Abuja", "fct abuja": "Abuja",
    "lagos": "Lagos", "oyo": "Oyo", "ogun": "Ogun",
    "rivers": "Rivers", "delta": "Delta", "anambra": "Anambra",
}
df["state"] = df["state"].str.strip().str.lower().map(lambda x: state_map.get(x, str(x).title()))

# ── 2.9  Standardise property type ──
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

# ── 2.10  Fix "Lagos" as LGA (unparsed location strings) ──
# Where lga == "Lagos" (generic), try to recover from location text
def fix_generic_lga(row):
    if str(row["lga"]).strip().lower() == "lagos":
        loc = str(row["location"]).lower()
        for known_lga in ["surulere", "yaba", "gbagada", "ikorodu", "mushin",
                          "agege", "shomolu", "maryland", "lekki", "ikeja",
                          "ikoyi", "victoria island", "ajah"]:
            if known_lga in loc:
                return known_lga.title()
    return row["lga"]

df["lga"] = df.apply(fix_generic_lga, axis=1)
print(f"\nFixed generic 'Lagos' LGA entries where recoverable from location text")

# ── 2.11  Parse date ──
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce", dayfirst=True)
df["days_on_market"] = (pd.Timestamp("today") - df["date_added"]).dt.days

print(f"\nCleaning complete. Final shape: {df.shape}")
print(f"\nState distribution:\n{df['state'].value_counts()}")
print(f"\nTop 15 LGAs:\n{df['lga'].value_counts().head(15)}")
print(f"\nProperty type distribution:\n{df['prop_type_clean'].value_counts()}")

# ─────────────────────────────────────────────────────────────
# PHASE 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PHASE 3 (V2) — FEATURE ENGINEERING")
print("=" * 55)

# ── 3.1  LGA-level price statistics ──
lga_stats = df.groupby("lga")["log_price"].agg(
    lga_median_log_price="median",
    lga_mean_log_price="mean",
    lga_count="count"
).reset_index()
df = df.merge(lga_stats, on="lga", how="left")
print(f"\n[✓] LGA price stats added ({df['lga'].nunique()} unique LGAs)")

# ── 3.2  NEW: Manual LGA quality score ──
df["lga_quality_score"] = df["lga"].apply(get_lga_score)
print(f"[✓] LGA quality score added (independent of sample size)")
print(f"    Distribution:\n{df['lga_quality_score'].value_counts().sort_index()}")

# ── 3.3  State-level encoding ──
state_price = df.groupby("state")["log_price"].median().rename("state_median_log_price")
df = df.join(state_price, on="state")
print(f"[✓] State median price encoded")

# ── 3.4  Property type dummies ──
type_dummies = pd.get_dummies(df["prop_type_clean"], prefix="type", drop_first=True)
df = pd.concat([df, type_dummies], axis=1)
print(f"[✓] Property type dummies: {list(type_dummies.columns)}")

# ── 3.5  Feature flags ──
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

# ── 3.6  Bed/bath ratio ──
df["bath_per_bed"] = (df["bathrooms"] / df["bedrooms"]).round(2)
df["bath_per_bed"] = df["bath_per_bed"].clip(0.5, 3.0)

# ── 3.7  Premium area flag ──
premium_lgas = {"Ikoyi", "Victoria Island", "Lekki", "Banana Island",
                "Maitama", "Asokoro", "Wuse 2", "Jabi"}
df["is_premium_area"] = df["lga"].isin(premium_lgas).astype(int)
print(f"[✓] is_premium_area: {df['is_premium_area'].sum()} listings")

# ── 3.8  Save ──
df.to_csv("nigeria_property_clean_v2.csv", index=False)
print(f"\n[✓] Saved to nigeria_property_clean_v2.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")

model_features = [
    "bedrooms", "bathrooms", "bath_per_bed",
    "lga_median_log_price", "lga_mean_log_price", "lga_count",
    "lga_quality_score",  # NEW in v2
    "state_median_log_price",
    "is_premium_area",
    "is_furnished", "is_serviced", "is_newly_built",
    "has_cof_o", "has_bq", "has_pool", "is_gated",
    "days_on_market",
] + list(type_dummies.columns)

print(f"\n{'='*55}")
print(f"MODEL-READY FEATURES ({len(model_features)} total, incl. lga_quality_score):")
print(f"{'='*55}")
for f in model_features:
    non_null = df[f].notna().sum()
    print(f"  {f:<30} {non_null:>5} non-null  ({non_null/len(df)*100:.0f}%)")

print(f"\nReady for Phase 4 (V2) — Modelling!")
