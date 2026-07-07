# Nigeria Property AVM — Automated Valuation Model

> ## Nigeria's first open-source Automated Valuation Model for residential property.
> Built entirely from scraped data, deployed as a live REST API.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138-green)](https://fastapi.tiangolo.com)
[![MdAPE](https://img.shields.io/badge/MdAPE-28.4%25-yellow)](##model-performance)

## The Problem

Nigeria has no open land registry. No public transaction database. No Zillow.  
When a buyer, seller, bank, or investor needs to value a Nigerian property, they hire a human surveyor, a process that takes days and costs ₦50,000–₦200,000, with no transparency on how the number was reached.

This project changes that.

## What I Built

A full end-to-end machine learning pipeline that:

1. Scrapes live property listings from PropertyPro.ng (1,940 listings collected)
2. Cleans messy, inconsistent Nigerian property data (prices in mixed formats, unstandardised locations)
3. Engineers Nigerian-specific features, LGA-level price encoding, C of O status, BQ presence, premium area flags
4. Trains Ridge regression (baseline) and XGBoost (main model) on 1,635 cleaned listings
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

## Model Performance

| Metric | Ridge (Baseline) | XGBoost (Final) | Improvement |

| MAE | ₦212,938,822 | ₦195,281,484 | -8.3% |
| RMSE | ₦417,689,904 | ₦398,752,440 | -4.5% |
| MAPE | 59.5% | 50.9% | -8.6pp |
| **MdAPE** | **34.1%** | **28.4%** | **-5.7pp** |

> Why MdAPE?

Nigerian property data contains extreme outliers. Median absolute percentage error is more robust than mean-based metrics in right-skewed distributions.

## Key Findings From Testing

### 1. Premium features add a 59% price uplift
A 4-bedroom house in Lekki with no features: **₦245M**  
Same house, fully premium (C of O + BQ + pool + furnished + serviced + newly built): **₦391M**  
**Premium uplift: +₦145.7M (+59.4%)**

### 2. Bedroom count drives value correctly
| Bedrooms | Bathrooms | LGA | Estimated Value |

| 2 | 2 | Ikeja | ₦281,597,664 |
| 5 | 5 | Ikeja | ₦831,385,984 |

### 3. Inter-city pricing correctly captured
| Location | 4-bed basic | Market range (PropertyPro) |

| Lekki, Lagos | ₦245,259,456 | ₦200M–₦500M |
| Maitama, Abuja | ₦675,186,624 | ₦500M–₦1.2B |

### 4. Identified limitation
Intra-city Lagos location sensitivity is weak due to sample bias in mid-market LGAs. This is the primary target for v2.

## Nigerian-Specific Feature Engineering

| Feature | Description | Why it matters |

| `has_cof_o` | Certificate of Occupancy | Gold-standard legal title; buyers pay premium |
| `has_bq` | Boys' Quarters | Rental income potential adds value |
| `lga_median_log_price` | LGA-level price encoding | Captures micro-market location value |
| `is_premium_area` |
