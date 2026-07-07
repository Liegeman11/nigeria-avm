"""
Nigerian AVM — Phase 6: FastAPI deployment
Run: uvicorn avm_deploy_api:app --reload
Then open: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pickle
import pandas as pd

# ── Load model ─────────────────────────────────────────────────
with open("avm_model.pkl", "rb") as f:
    bundle  = pickle.load(f)
    MODEL   = bundle["model"]
    FEATURES= bundle["features"]

# ── Load LGA price stats (needed for encoding) ─────────────────
LGA_STATS = pd.read_csv("lga_price_summary.csv")
GLOBAL_MEDIAN_LOG = np.log1p(LGA_STATS["median_price_M"].median() * 1e6)

# Precompute LGA → log price median lookup
LGA_LOG_LOOKUP = (
    pd.read_csv("nigeria_property_clean.csv")
    .groupby("lga")["log_price"]
    .agg(["median", "mean", "count"])
    .rename(columns={"median": "lga_median_log_price",
                     "mean":   "lga_mean_log_price",
                     "count":  "lga_count"})
)
STATE_LOG_LOOKUP = (
    pd.read_csv("nigeria_property_clean.csv")
    .groupby("state")["log_price"]
    .median()
    .rename("state_median_log_price")
)

PREMIUM_LGAS = {
    "Ikoyi", "Victoria Island", "Lekki", "Banana Island",
    "Maitama", "Asokoro", "Wuse 2", "Jabi",
}

app = FastAPI(
    title="Nigerian Property AVM",
    description="Automated Valuation Model for Nigerian residential property. "
                "Built on ~1,900 scraped PropertyPro.ng listings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────
class PropertyInput(BaseModel):
    bedrooms:     int   = Field(..., ge=1, le=10,  example=3,
                                 description="Number of bedrooms")
    bathrooms:    int   = Field(..., ge=1, le=15,  example=3,
                                 description="Number of bathrooms")
    lga:          str   = Field(..., example="Lekki",
                                 description="Local Government Area (e.g. Lekki, Ikeja, Maitama)")
    state:        str   = Field(..., example="Lagos",
                                 description="State (e.g. Lagos, Abuja, Oyo)")
    prop_type:    str   = Field("House", example="Flat/Apartment",
                                 description="House | Flat/Apartment | Detached House | Semi-Detached | Terrace | Bungalow | Duplex")
    is_furnished:  int  = Field(0, ge=0, le=1, example=0)
    is_serviced:   int  = Field(0, ge=0, le=1, example=0)
    is_newly_built:int  = Field(0, ge=0, le=1, example=0)
    has_cof_o:     int  = Field(0, ge=0, le=1, example=1)
    has_bq:        int  = Field(0, ge=0, le=1, example=0)
    has_pool:      int  = Field(0, ge=0, le=1, example=0)
    is_gated:      int  = Field(0, ge=0, le=1, example=0)


class ValuationResponse(BaseModel):
    estimated_value_ngn: int
    estimated_value_usd: float
    confidence_range_ngn: dict
    lga:   str
    state: str
    inputs_used: dict


# ── Prediction logic ───────────────────────────────────────────
def build_feature_vector(inp: PropertyInput) -> np.ndarray:
    lga   = inp.lga.strip().title()
    state = inp.state.strip().title()

    # LGA encoding
    if lga in LGA_LOG_LOOKUP.index:
        lga_median = LGA_LOG_LOOKUP.loc[lga, "lga_median_log_price"]
        lga_mean   = LGA_LOG_LOOKUP.loc[lga, "lga_mean_log_price"]
        lga_count  = LGA_LOG_LOOKUP.loc[lga, "lga_count"]
    else:
        lga_median = GLOBAL_MEDIAN_LOG
        lga_mean   = GLOBAL_MEDIAN_LOG
        lga_count  = 1

    # State encoding
    state_median = (STATE_LOG_LOOKUP.get(state, GLOBAL_MEDIAN_LOG))

    # Property type dummies (match training columns)
    type_cols   = [c for c in FEATURES if c.startswith("type_")]
    type_vector = {c: 0 for c in type_cols}
    type_key    = f"type_{inp.prop_type.replace('/', '').replace(' ', '_')}"
    if type_key in type_vector:
        type_vector[type_key] = 1

    bath_per_bed = min(max(inp.bathrooms / inp.bedrooms, 0.5), 3.0)
    is_premium   = 1 if lga in PREMIUM_LGAS else 0

    base = {
        "bedrooms":              inp.bedrooms,
        "bathrooms":             inp.bathrooms,
        "bath_per_bed":          bath_per_bed,
        "lga_median_log_price":  lga_median,
        "lga_mean_log_price":    lga_mean,
        "lga_count":             lga_count,
        "state_median_log_price":state_median,
        "is_premium_area":       is_premium,
        "is_furnished":          inp.is_furnished,
        "is_serviced":           inp.is_serviced,
        "is_newly_built":        inp.is_newly_built,
        "has_cof_o":             inp.has_cof_o,
        "has_bq":                inp.has_bq,
        "has_pool":              inp.has_pool,
        "is_gated":              inp.is_gated,
        **type_vector,
    }

    return np.array([[base[f] for f in FEATURES]])


@app.post("/predict", response_model=ValuationResponse)
def predict(prop: PropertyInput):
    try:
        X = build_feature_vector(prop)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature error: {e}")

    log_pred  = MODEL.predict(X)[0]
    price_ngn = int(np.expm1(log_pred))

    # ±20% confidence band (reflects ~20% MdAPE)
    low  = int(price_ngn * 0.80)
    high = int(price_ngn * 1.20)

    # USD conversion (update rate as needed)
    USD_RATE = 1600
    price_usd = round(price_ngn / USD_RATE, 2)

    return ValuationResponse(
        estimated_value_ngn=price_ngn,
        estimated_value_usd=price_usd,
        confidence_range_ngn={"low": low, "high": high},
        lga=prop.lga,
        state=prop.state,
        inputs_used=prop.dict(),
    )


@app.get("/lgas")
def list_lgas():
    """List all LGAs the model has data for."""
    return {"lgas": sorted(LGA_LOG_LOOKUP.index.tolist())}


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost AVM v1.0", "features": len(FEATURES)}


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

