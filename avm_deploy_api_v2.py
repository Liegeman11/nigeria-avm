"""
Nigerian AVM V2 — FastAPI deployment
Uses avm_model_v2.pkl with lga_quality_score feature included.
Run: uvicorn avm_deploy_api_v2:app --reload --port 8001
Docs: http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pickle
import pandas as pd

with open("avm_model_v2.pkl", "rb") as f:
    bundle   = pickle.load(f)
    MODEL    = bundle["model"]
    FEATURES = bundle["features"]

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
    lga_clean = str(lga).strip().title()
    if lga_clean in LGA_QUALITY_SCORES:
        return LGA_QUALITY_SCORES[lga_clean]
    for key, score in LGA_QUALITY_SCORES.items():
        if key.lower() in lga_clean.lower():
            return score
    return LGA_QUALITY_SCORES["DEFAULT"]

CLEAN_DF = pd.read_csv("nigeria_property_clean_v2.csv")
GLOBAL_MEDIAN_LOG = CLEAN_DF["log_price"].median()

LGA_LOG_LOOKUP = (
    CLEAN_DF.groupby("lga")["log_price"]
    .agg(["median", "mean", "count"])
    .rename(columns={"median": "lga_median_log_price",
                     "mean": "lga_mean_log_price",
                     "count": "lga_count"})
)
STATE_LOG_LOOKUP = CLEAN_DF.groupby("state")["log_price"].median()

PREMIUM_LGAS = {"Ikoyi", "Victoria Island", "Lekki", "Banana Island",
                "Maitama", "Asokoro", "Wuse 2", "Jabi"}

app = FastAPI(title="Nigerian Property AVM V2",
             description="V2: expanded to 3,161 listings + LGA quality scores")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PropertyInput(BaseModel):
    bedrooms:     int   = Field(..., ge=1, le=10)
    bathrooms:    int   = Field(..., ge=1, le=15)
    lga:          str
    state:        str
    prop_type:    str   = Field("House")
    is_furnished:  int  = Field(0, ge=0, le=1)
    is_serviced:   int  = Field(0, ge=0, le=1)
    is_newly_built:int  = Field(0, ge=0, le=1)
    has_cof_o:     int  = Field(0, ge=0, le=1)
    has_bq:        int  = Field(0, ge=0, le=1)
    has_pool:      int  = Field(0, ge=0, le=1)
    is_gated:      int  = Field(0, ge=0, le=1)


def build_feature_vector(inp: PropertyInput) -> np.ndarray:
    lga   = inp.lga.strip().title()
    state = inp.state.strip().title()

    if lga in LGA_LOG_LOOKUP.index:
        lga_median = LGA_LOG_LOOKUP.loc[lga, "lga_median_log_price"]
        lga_mean   = LGA_LOG_LOOKUP.loc[lga, "lga_mean_log_price"]
        lga_count  = LGA_LOG_LOOKUP.loc[lga, "lga_count"]
    else:
        lga_median = lga_mean = GLOBAL_MEDIAN_LOG
        lga_count  = 1

    state_median = STATE_LOG_LOOKUP.get(state, GLOBAL_MEDIAN_LOG)
    lga_quality  = get_lga_score(lga)

    type_cols   = [c for c in FEATURES if c.startswith("type_")]
    type_vector = {c: 0 for c in type_cols}
    type_key    = f"type_{inp.prop_type.replace('/', '').replace(' ', '_')}"
    if type_key in type_vector:
        type_vector[type_key] = 1

    bath_per_bed = min(max(inp.bathrooms / inp.bedrooms, 0.5), 3.0)
    is_premium   = 1 if lga in PREMIUM_LGAS else 0

    base = {
        "bedrooms": inp.bedrooms, "bathrooms": inp.bathrooms,
        "bath_per_bed": bath_per_bed,
        "lga_median_log_price": lga_median, "lga_mean_log_price": lga_mean,
        "lga_count": lga_count, "lga_quality_score": lga_quality,
        "state_median_log_price": state_median, "is_premium_area": is_premium,
        "is_furnished": inp.is_furnished, "is_serviced": inp.is_serviced,
        "is_newly_built": inp.is_newly_built, "has_cof_o": inp.has_cof_o,
        "has_bq": inp.has_bq, "has_pool": inp.has_pool, "is_gated": inp.is_gated,
        **type_vector,
    }
    return np.array([[base[f] for f in FEATURES]])


@app.post("/predict")
def predict(prop: PropertyInput):
    X = build_feature_vector(prop)
    log_pred  = MODEL.predict(X)[0]
    price_ngn = int(np.expm1(log_pred))
    return {
        "estimated_value_ngn": price_ngn,
        "estimated_value_usd": round(price_ngn / 1600, 2),
        "confidence_range_ngn": {"low": int(price_ngn * 0.80), "high": int(price_ngn * 1.20)},
        "lga": prop.lga, "state": prop.state,
        "lga_quality_score": get_lga_score(prop.lga),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost AVM V2", "features": len(FEATURES),
            "training_rows": len(CLEAN_DF)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
