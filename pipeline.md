# Prediction Pipeline

Documents how raw user inputs are transformed into model features before calling the model.

## Raw inputs (12 fields)

| Field | Type | Values |
|---|---|---|
| `a_career_ambition` / `b_career_ambition` | float | 0.0–1.0 |
| `a_chronotype` / `b_chronotype` | float | 0.0–1.0 |
| `a_emotional_expressiveness` / `b_emotional_expressiveness` | float | 0.0–1.0 |
| `a_openness` / `b_openness` | float | 0.0–1.0 |
| `a_spontaneity` / `b_spontaneity` | float | 0.0–1.0 |
| `a_love_language` / `b_love_language` | str | Acts of Service, Physical Touch, Quality Time, Receiving Gifts, Words of Affirmation |

## Feature engineering (6 final features)

### Personality traits → absolute difference
```
career_ambition_diff          = abs(a_career_ambition - b_career_ambition)
chronotype_diff               = abs(a_chronotype - b_chronotype)
emotional_expressiveness_diff = abs(a_emotional_expressiveness - b_emotional_expressiveness)
openness_diff                 = abs(a_openness - b_openness)
spontaneity_diff              = abs(a_spontaneity - b_spontaneity)
```

### Love language → match flag
```
same_love_language = 1 if a_love_language == b_love_language else 0
```

## Features dropped (sequential backward elimination — notebook 06)

All 10 features below were removed with a cumulative AUC loss of −0.0044 (within the −0.005 tolerance):

| Feature | Reason |
|---|---|
| `age_mean` | Negative permutation importance (adds noise) |
| `age_diff` | Near-zero signal |
| `education_diff` | Near-zero signal |
| `same_career_field` | Near-zero signal |
| `conscientiousness_diff` | Near-zero signal |
| `large_age_gap` | Near-zero signal |
| `large_education_gap` | Near-zero signal |
| `same_location` | Near-zero signal |
| `extraversion_diff` | Near-zero signal |
| `agreeableness_diff` | Near-zero signal |

## Model

**Logistic Regression** with `class_weight="balanced"`, wrapped in a `sklearn.pipeline.Pipeline` (StandardScaler → LR).
Serialised to `src/models/lr_pipeline_final.pkl` via `joblib`.

## Model output

The model returns `P(long-lasting)` — the probability that the relationship lasts **10 years or more** (threshold fixed at 120 months, see notebook 04).

Only ~**0.9%** of couples in the training set reach this milestone.
Class imbalance is handled via `class_weight="balanced"` in Logistic Regression.
Primary evaluation metric: **ROC-AUC** (accuracy is uninformative with 99.1% negatives). CV AUC = 0.7501 on the 6-feature optimal set.
