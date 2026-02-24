# 💜 ML Matchmaking — Predicting Relationship Longevity

> **Can we predict whether a couple will last 10+ years?**
> A full-stack ML project — from raw data exploration to a deployed API and interactive web app.

---

## 🎯 Project Objective

This project builds a **binary classification model** that estimates the probability that a relationship will last **10 years or more**, based on personality, values, and compatibility features.

The model is exposed through a **FastAPI backend** and a **web frontend** offering three interaction modes:
- 🧠 **AI Interview** — free-text conversation (Groq/Llama) that extracts personality traits and predicts compatibility
- 📋 **Questionnaire** — structured quiz on personality and relationship style
- 🎯 **Manual Scores** — direct feature input for power users

---

## 🔬 ML Pipeline — 8 Notebooks

The full analytical journey is documented step by step:

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `data_audit` | Data exploration, types, distributions, missing values |
| 02 | `features_engineering` | Crafting relational features from raw personality data |
| 03 | `modeling_longevity_monthly_target` | Regression benchmark — R² ≈ 0.11, insufficient → pivot to classification |
| 04 | `modeling_longevity_binary_target` | Binary classification (≥10 years) — **Logistic Regression retained (AUC = 0.75)** |
| 05 | `feature_importance` | SHAP values analysis — identifying key predictors |
| 06 | `optimize_feature_selection` | Sequential Backward Elimination → **6 features selected** |
| 07 | `model_export` | Final pipeline (StandardScaler + LR) retrained on full dataset and serialized |
| 08 | `paired_features` *(optional)* | Exploring paired category interactions for richer granularity |

**Key modelling decisions:**
- Regression was dropped (R² ≈ 0.11, high MAE) in favour of **binary classification**
- Logistic Regression with `class_weight="balanced"` outperformed tree-based models on this dataset
- SHAP + Sequential Backward Elimination reduced the feature set to **6 robust predictors**
- Final model: **AUC = 0.75**

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| **ML** | scikit-learn, LightGBM, SHAP |
| **API** | FastAPI, Uvicorn, Docker |
| **AI layer** | Groq API (Llama 3) |
| **Frontend** | Vanilla JS, HTML/CSS |
| **Deployment** | Hugging Face Spaces (Docker SDK), Vercel |

---

## 📁 Project Structure

```
├── notebooks/          # Full ML pipeline (01 → 08)
├── src/
│   ├── models/         # Serialized production models (.pkl)
│   ├── ai_extractor.py # LLM-based personality extraction
│   ├── config.py       # Feature definitions and constants
│   └── utils/          # Shared utilities
├── app.py              # FastAPI application
├── frontend.html       # Web app (single-file)
├── Dockerfile          # Container for Hugging Face Spaces
└── requirements.txt
```

---

## 🚀 Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn app:app --reload

# Open frontend.html in your browser
```

---

## 👩‍💻 Author

**Céline Apéry**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/c%C3%A9line-ap%C3%A9ry-936b1491/)

---

*This project was built as a data science portfolio piece to demonstrate end-to-end ML skills: data wrangling, feature engineering, model selection, interpretability (SHAP), API design, and deployment.*
