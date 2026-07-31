# Nigeria Property AVM — Automated Valuation Model

> ## Nigeria's first open-source Automated Valuation Model for residential property.
> Built entirely from scraped data, deployed as a live REST API.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)](https://fastapi.tiangolo.com)
[![MdAPE](https://img.shields.io/badge/MdAPE-27.6%25-yellow)](##model-performance)

## The Problem

Nigeria has no open land registry. No public transaction database. No Zillow.
When a buyer, seller, bank, or investor needs to value a Nigerian property, they hire a human surveyor, a process that takes days and costs ₦50,000–₦200,000, with no transparency on how the number was reached.

This project changes that.

## What I Built

A full end-to-end machine learning pipeline that:

1. Scrapes live property listings from PropertyPro.ng (1,940 listings collected in V1; expanded to 4,143 in V2)
2. Cleans messy, inconsistent Nigerian property data (prices in mixed formats, unstandardised locations)
3. Engineers Nigerian-specific features — LGA-level price encoding, C of O status, BQ presence, premium area flags, LGA quality scores
4. Trains Ridge regression (baseline) and XGBoost (main model)
5. Deploys the model as a production REST API with confidence intervals and automatic documentation

## Live Demo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 4,
    "bathrooms": 4,
    "lga": "Lekki",
    "state": "Lagos",
    "prop_type": "House",
    "has_cof_o": 1,
    "has_bq": 1,
    "has_pool": 0,
    "is_furnished": 0,
    "is_serviced": 0,
    "is_newly_built": 0,
    "is_gated": 0
  }'
```

**Response:**
```json
{
  "estimated_value_ngn": 391014048,
  "estimated_value_usd": 244383.78,
  "confidence_range_ngn": {
    "low": 312811238,
    "high": 469216857
  },
  "lga": "Lekki",
  "state": "Lagos"
}
```

## Model Performance (V1)

| Metric | Ridge (Baseline) | XGBoost (Final) | Improvement |
|---|---|---|---|
| MAE | ₦212,938,822 | ₦195,281,484 | -8.3% |
| RMSE | ₦417,689,904 | ₦398,752,440 | -4.5% |
| MAPE | 59.5% | 50.9% | -8.6pp |
| **MdAPE** | **34.1%** | **28.4%** | **-5.7pp** |

> **Why MdAPE?** Nigerian property data contains extreme outliers. Median absolute percentage error is more robust than mean-based metrics in right-skewed distributions.

## Key Findings From Testing (V1)

### 1. Premium features add a 59% price uplift
A 4-bedroom house in Lekki with no features: **₦245M**
Same house, fully premium (C of O + BQ + pool + furnished + serviced + newly built): **₦391M**
**Premium uplift: +₦145.7M (+59.4%)**

### 2. Bedroom count drives value correctly

| Bedrooms | Bathrooms | LGA | Estimated Value |
|---|---|---|---|
| 2 | 2 | Ikeja | ₦281,597,664 |
| 5 | 5 | Ikeja | ₦831,385,984 |

### 3. Inter-city pricing correctly captured

| Location | 4-bed basic | Market range (PropertyPro) |
|---|---|---|
| Lekki, Lagos | ₦245,259,456 | ₦200M–₦500M |
| Maitama, Abuja | ₦675,186,624 | ₦500M–₦1.2B |

### 4. Identified limitation
Intra-city Lagos location sensitivity is weak due to sample bias in mid-market LGAs. This is the primary target for V2 — addressed below.

## Nigerian-Specific Feature Engineering

| Feature | Description | Why it matters |
|---|---|---|
| `has_cof_o` | Certificate of Occupancy | Gold-standard legal title; buyers pay premium |
| `has_bq` | Boys' Quarters | Rental income potential adds value |
| `lga_median_log_price` | LGA-level price encoding | Captures micro-market location value |
| `lga_quality_score` | Manual 1-10 LGA prestige rating (V2) | Independent location signal, not dependent on scrape sample size |
| `is_premium_area` | Ikoyi/VI/Lekki/Maitama flag | Nigeria's ultra-premium residential clusters |
| `bath_per_bed` | Bathroom-to-bedroom ratio | Luxury signal in Nigerian market |

## Technical Stack
Data Collection → requests, BeautifulSoup4
Data Processing → pandas, numpy
Machine Learning → scikit-learn, XGBoost
Explainability → SHAP
API Deployment → FastAPI, Uvicorn

## Project Structure
├── nigeria_avm_scraper_final.py # V1 scraper
├── scrape_midmarket_lagos_v3.py # V2 mid-market Lagos scraper
├── lga_quality_scores.py # V2 manual LGA quality scores
├── avm_phase2_3.py # V1 cleaning + feature engineering
├── avm_phase2_3_v2.py # V2 cleaning + feature engineering
├── avm_phase4_model.py # V1 model training
├── avm_phase4_model_v2.py # V2 model training
├── avm_deploy_api.py # V1 FastAPI REST API
├── avm_deploy_api_v2.py # V2 FastAPI REST API
├── nigeria_property_raw.csv # V1 raw data (1,940 listings)
├── nigeria_property_clean_v2.csv # V2 cleaned data (3,161 listings)
├── lga_price_summary_v2.csv # V2 LGA price statistics
├── avm_evaluation_v2.png # V2 evaluation chart
└── requirements.txt

## How to Reproduce

```bash
git clone https://github.com/Liegeman11/nigeria-avm
cd nigeria-avm
pip install -r requirements.txt

# V1 pipeline
python avm_phase2_3.py
python avm_phase4_model.py

# V2 pipeline (recommended — better location accuracy)
python scrape_midmarket_lagos_v3.py
python avm_phase2_3_v2.py
python avm_phase4_model_v2.py
python avm_deploy_api_v2.py
# Open http://localhost:8001/docs
```


---

## Version 2 — Fixing the Location Weakness

V1 identified a specific limitation: Lekki and Surulere returned near-identical valuations for the same property, when the real market gap is 2-3×.

**What changed:**
- Scraped 2,276 new listings from underrepresented mid-market Lagos areas (Surulere, Yaba, Gbagada, Ikorodu, Agege, Shomolu, Maryland)
- Dataset grew from 1,635 to 3,161 listings
- Added manual LGA quality scores (1-10 prestige rating) as an independent location signal, not dependent on scrape sample size

**Note on floor area:** V2 originally planned to add floor area (sqm) as a feature. Testing showed PropertyPro listings almost never report square metreage — Nigerian agents describe properties by bedroom/bathroom count instead. This was dropped as a non-viable feature for this data source.

**Results — same test, V1 vs V2:**

| Property | V1 estimate | V2 estimate |
|---|---|---|
| 3-bed house, Lekki | ₦225,997,472 | ₦233,405,008 |
| 3-bed house, Surulere | ₦235,146,368 (wrongly higher) | ₦170,810,960 |

Lekki is now correctly valued ~37% higher than Surulere, matching real market dynamics. In V1, Surulere was priced above Lekki, which was wrong.

| Metric | V1 | V2 |
|---|---|---|
| MdAPE | 28.4% | 27.6% |
| MAE | ₦195,281,484 | ₦152,615,998 (-22%) |
| RMSE | ₦398,752,440 | ₦365,641,889 |
| Training listings | 1,635 | 3,161 |

`lga_quality_score` is now among the top 6 most important features in the model.


---

## Version 3 — Fixing `is_furnished`, Removing `is_serviced`, and a Real Insight on `has_cof_o`

V2 left two sparse binary features (`is_furnished`, `is_serviced`) with very low training coverage (1.9% and 1.3%), producing inconsistent, direction-flipping predictions depending on property location and size.

**What changed:**
- Scraped PropertyPro's dedicated filter pages (`/is-furnished`, `/is-serviced`, `/is-new`) to directly target these sparse features rather than hoping to encounter them by chance
- Dataset grew from 3,161 to 3,923 usable listings (after removing noise from out-of-scope states picked up by the nationwide filter scrape)
- `is_furnished` coverage grew from 1.9% to 4.4%

**Result — `is_furnished` now behaves consistently:**

| LGA | Bedrooms | Unfurnished | Furnished | Diff |
|---|---|---|---|---|
| Agege | 5 | ₦336,800,128 | ₦368,185,088 | +₦31.4M |
| Lekki | 5 | ₦595,722,816 | ₦666,559,744 | +₦70.8M |
| Ikoyi | 4 | ₦980,902,208 | ₦1,115,104,896 | +₦134.2M |

Furnished now consistently increases price, scaling sensibly with property value — a complete turnaround from V2, where the effect flipped sign depending on location.

**`is_serviced` was removed entirely.** After two attempts to fix it with more targeted data, coverage barely moved (1.3% → 1.6%) — the `/is-serviced` filter URL doesn't reliably return genuinely serviced listings (validated at just 0-5% actual match rate). Re-testing showed the feature producing a **−₦239M swing** on a 4-bed Ikoyi property — actively misleading, not just weak. Rather than ship a feature that confidently gives the wrong answer, it was excluded from the model.

**A real insight, not a bug — `has_cof_o`:**

Testing showed `has_cof_o` consistently *decreases* predicted price across every LGA and bedroom combination tested (−₦4.7M to −₦38M). Rather than remove it, further diagnosis revealed why: agents rarely mention Certificate of Occupancy in ultra-premium areas (Asokoro: 0% of listings mention it, Ikoyi: 2.9%), but mention it far more often in mid-tier areas (Ojodu: 16%, Ikeja: 9%) where buyers are more likely to want title reassurance. The correlation between LGA price and C of O mention rate is −0.213.

This means `has_cof_o` isn't really measuring "does this property have legal title" — it's acting as a weak proxy for market tier, since agents in premium areas simply assume title is fine and don't bother stating it. The model isn't wrong; the feature is measuring something more subtle than its name suggests.

**`is_gated`, by contrast, is a genuine premium signal** — consistently positive across all tests (+₦5M to +₦196M), with no meaningful price-tier correlation (−0.072), confirming it's a direct amenity effect rather than a marketing artifact.

| Metric | V2 | V3 |
|---|---|---|
| MdAPE | 27.6% | 28.7% |
| Training listings | 3,161 | 3,923 |
| `is_furnished` coverage | 1.9% | 4.4% |
| `is_serviced` | included (unreliable) | removed |


## About

Built by **Taiwo Micheal Emmanuel** — Statistician, Data Analyst, and Senior Data Scientist.

- 🔗 [LinkedIn](https://www.linkedin.com/in/taiwo-micheal-emmanuel-633602325/)
- 💻 [GitHub](https://github.com/Liegeman11)
