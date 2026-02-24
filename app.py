from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
from pathlib import Path
import time
from collections import defaultdict

from src.ai_extractor import extract_personality, neutral_personality

# ── Load model ────────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "src" / "models" / "lr_pipeline_final.pkl"
model = joblib.load(MODEL_PATH)

# ── Load quiz question bank ────────────────────────────────────────────────────

QUIZ_PATH = Path(__file__).parent / "src" / "quiz_questions.json"
with open(QUIZ_PATH, encoding="utf-8") as _f:
    QUIZ_QUESTIONS = json.load(_f)

app = FastAPI(
    title="ML Matchmaking API",
    description="Predicts the probability that a couple lasts 10 years or more (threshold: 120 months).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Input schema ──────────────────────────────────────────────────────────────

class CoupleInput(BaseModel):
    # Personality scores (0.0 – 1.0)
    a_career_ambition: float = Field(..., ge=0, le=1)
    b_career_ambition: float = Field(..., ge=0, le=1)
    a_chronotype: float = Field(..., ge=0, le=1)
    b_chronotype: float = Field(..., ge=0, le=1)
    a_emotional_expressiveness: float = Field(..., ge=0, le=1)
    b_emotional_expressiveness: float = Field(..., ge=0, le=1)
    a_openness: float = Field(..., ge=0, le=1)
    b_openness: float = Field(..., ge=0, le=1)
    a_spontaneity: float = Field(..., ge=0, le=1)
    b_spontaneity: float = Field(..., ge=0, le=1)

    # Love language (categorical match)
    a_love_language: str
    b_love_language: str


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(data: CoupleInput) -> pd.DataFrame:
    """Transform raw couple inputs into the 6 model features. See pipeline.md."""
    features = {
        "career_ambition_diff":          abs(data.a_career_ambition - data.b_career_ambition),
        "chronotype_diff":               abs(data.a_chronotype - data.b_chronotype),
        "emotional_expressiveness_diff": abs(data.a_emotional_expressiveness - data.b_emotional_expressiveness),
        "openness_diff":                 abs(data.a_openness - data.b_openness),
        "spontaneity_diff":              abs(data.a_spontaneity - data.b_spontaneity),
        "same_love_language":            int(data.a_love_language == data.b_love_language),
    }
    return pd.DataFrame([features])


# ── Endpoint ──────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "career_ambition_diff":          "Career Ambition gap",
    "chronotype_diff":               "Chronotype gap",
    "emotional_expressiveness_diff": "Emotional Expressiveness gap",
    "openness_diff":                 "Openness gap",
    "spontaneity_diff":              "Spontaneity gap",
    "same_love_language":            "Same Love Language",
}


def get_feature_contributions(features_df: pd.DataFrame) -> list[dict]:
    """
    Compute per-feature log-odds contributions for a single prediction.
    contribution_i = coef_i * scaled_value_i
    Positive → pushes toward long-lasting; negative → pushes against.
    """
    scaler = model.named_steps["scaler"]
    lr     = model.named_steps["lr"]

    scaled = scaler.transform(features_df)          # shape (1, n_features)
    coefs  = lr.coef_[0]                            # shape (n_features,)

    contributions = []
    for i, col in enumerate(features_df.columns):
        contrib = float(coefs[i] * scaled[0][i])
        contributions.append({
            "feature": col,
            "label":   FEATURE_LABELS.get(col, col),
            "raw_value":    round(float(features_df.iloc[0][col]), 4),
            "contribution": round(contrib, 4),
        })

    # Sort by absolute contribution, largest first
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


@app.post("/predict")
def predict(data: CoupleInput):
    """
    Returns P(long-lasting) — the probability that the relationship lasts 10 years or more
    (classification threshold fixed at 120 months, see notebook 04).
    Only ~0.9 % of couples in the training set reach this milestone.
    Also returns per-feature log-odds contributions.
    """
    features = build_features(data)
    proba = model.predict_proba(features)[0][1]
    contributions = get_feature_contributions(features)
    return {
        "probability":    round(float(proba), 4),
        "long_lasting":   bool(proba >= 0.5),
        "contributions":  contributions,
    }


@app.get("/quiz_questions")
def quiz_questions():
    """Returns the full question bank used by the quiz mode."""
    return QUIZ_QUESTIONS


# ── Rate limiting (in-memory, par IP) ────────────────────────────────────────

RATE_LIMIT_WINDOW  = 60    # secondes
RATE_LIMIT_MAX     = 5     # requêtes max par IP par fenêtre
MAX_TRANSCRIPT_LEN = 2000  # caractères max par transcript

_rate_store: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(ip: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Garder uniquement les timestamps dans la fenêtre courante
    _rate_store[ip] = [t for t in _rate_store[ip] if t > window_start]
    if len(_rate_store[ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Max {RATE_LIMIT_MAX} AI analyses per minute per IP."
        )
    _rate_store[ip].append(now)


# ── AI interview endpoint ─────────────────────────────────────────────────────

class InterviewInput(BaseModel):
    a_transcript: str = Field(..., description="Description of person A")
    b_transcript: str = Field(..., description="Description of person B")
    run_prediction: bool = Field(True, description="If True, runs the ML model and returns the prediction alongside the extracted scores")


@app.post("/ai_interview")
def ai_interview(data: InterviewInput, request: Request):
    """
    Extracts personality scores from free-text descriptions
    using Groq/Llama (temperature=0), then optionally runs the ML prediction.

    - a_transcript / b_transcript : free-text description of each person
    - run_prediction               : set to false to only get extracted scores
    """
    # ── Rate limiting ─────────────────────────────────────────────────────────
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    # ── Longueur max des transcripts ──────────────────────────────────────────
    if len(data.a_transcript) > MAX_TRANSCRIPT_LEN:
        raise HTTPException(status_code=400, detail=f"Person A description too long (max {MAX_TRANSCRIPT_LEN} chars).")
    if len(data.b_transcript) > MAX_TRANSCRIPT_LEN:
        raise HTTPException(status_code=400, detail=f"Person B description too long (max {MAX_TRANSCRIPT_LEN} chars).")

    # ── Extraction pour A ────────────────────────────────────────────────────
    try:
        scores_a = extract_personality(data.a_transcript, person_label="A")
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed for person A: {e}")

    # ── Extraction pour B ────────────────────────────────────────────────────
    try:
        scores_b = extract_personality(data.b_transcript, person_label="B")
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed for person B: {e}")

    # ── Résultat de base ─────────────────────────────────────────────────────
    result = {
        "person_a": scores_a,
        "person_b": scores_b,
    }

    # ── Prédiction ML (optionnel) ─────────────────────────────────────────────
    if data.run_prediction:
        couple = CoupleInput(
            a_career_ambition          = scores_a["career_ambition"],
            b_career_ambition          = scores_b["career_ambition"],
            a_chronotype               = scores_a["chronotype"],
            b_chronotype               = scores_b["chronotype"],
            a_emotional_expressiveness = scores_a["emotional_expressiveness"],
            b_emotional_expressiveness = scores_b["emotional_expressiveness"],
            a_openness                 = scores_a["openness"],
            b_openness                 = scores_b["openness"],
            a_spontaneity              = scores_a["spontaneity"],
            b_spontaneity              = scores_b["spontaneity"],
            a_love_language            = scores_a["love_language"],
            b_love_language            = scores_b["love_language"],
        )
        features = build_features(couple)
        proba = model.predict_proba(features)[0][1]
        contributions = get_feature_contributions(features)
        result["prediction"] = {
            "probability":   round(float(proba), 4),
            "long_lasting":  bool(proba >= 0.5),
            "contributions": contributions,
        }

    return result


@app.get("/health")
def health():
    return {"status": "ok"}
