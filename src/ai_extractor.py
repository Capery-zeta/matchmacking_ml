"""
ai_extractor.py
───────────────
Couche IA : à partir d'une description libre d'une personne,
extrait les 5 scores de personnalité + le love language via l'API Groq (gratuit).

Scores extraits :
  career_ambition          float  0.0 – 1.0
  chronotype               float  0.0 – 1.0   (0 = night owl, 1 = early bird)
  emotional_expressiveness float  0.0 – 1.0
  openness                 float  0.0 – 1.0
  spontaneity              float  0.0 – 1.0
  love_language            str    l'une des 5 valeurs canoniques

Usage rapide :
  from src.ai_extractor import extract_personality
  scores = extract_personality(transcript="...", person_label="A")

Prérequis :
  pip install groq
  export GROQ_API_KEY="gsk_..."
"""

import json
import re
import os
from groq import Groq

# ── Valeurs autorisées ────────────────────────────────────────────────────────

VALID_LOVE_LANGUAGES = {
    "Acts of Service",
    "Physical Touch",
    "Quality Time",
    "Receiving Gifts",
    "Words of Affirmation",
}

NUMERIC_FIELDS = [
    "career_ambition",
    "chronotype",
    "emotional_expressiveness",
    "openness",
    "spontaneity",
]

# ── Prompt système ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personality-scoring assistant for a relationship compatibility model.

Your job: read a description of one person, then infer their scores on 5 personality dimensions and their primary love language.

## Scoring definitions

| Dimension               | 0.0 (low end)                          | 1.0 (high end)                             |
|-------------------------|----------------------------------------|--------------------------------------------|
| career_ambition         | Work-life balance first, works to live | Career is top priority, lives to excel     |
| chronotype              | Night owl, active late, slow mornings  | Early bird, energised in the morning       |
| emotional_expressiveness| Reserved, rarely shares feelings       | Open book, expresses emotions freely       |
| openness                | Prefers routine, familiar experiences  | Craves novelty, ideas, diverse experiences |
| spontaneity             | Needs to plan everything in advance    | Loves last-minute decisions, improvises    |

## Love language options (pick exactly one)
- Acts of Service
- Physical Touch
- Quality Time
- Receiving Gifts
- Words of Affirmation

## Output format
Return ONLY a valid JSON object — no prose, no markdown fences, no explanation:

{
  "career_ambition": <float 0.0–1.0>,
  "chronotype": <float 0.0–1.0>,
  "emotional_expressiveness": <float 0.0–1.0>,
  "openness": <float 0.0–1.0>,
  "spontaneity": <float 0.0–1.0>,
  "love_language": "<one of the 5 canonical strings>",
  "confidence": <float 0.0–1.0>,
  "notes": "<optional short explanation of uncertain scores, or empty string>"
}

## Rules
- Use 0.5 as the default when you have no signal for a dimension.
- The `confidence` field reflects your overall certainty (0 = guessing, 1 = very confident).
- Never invent information. If the description is too short, lower confidence and use 0.5 defaults.
- Return ONLY the JSON object — nothing else."""

# ── Prompt utilisateur ────────────────────────────────────────────────────────

def _build_user_prompt(transcript: str, person_label: str) -> str:
    return (
        f"Here is the description for Person {person_label}:\n\n"
        f"<description>\n{transcript.strip()}\n</description>\n\n"
        f"Extract the personality scores for Person {person_label} and return the JSON object."
    )

# ── Parsing + validation ──────────────────────────────────────────────────────

def _parse_and_validate(raw: str) -> dict:
    """Parse la réponse brute du LLM et valide les champs."""
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Aucun JSON trouvé dans la réponse : {raw!r}")

    data = json.loads(json_match.group())

    for field in NUMERIC_FIELDS:
        if field not in data:
            raise ValueError(f"Champ manquant : {field}")
        val = float(data[field])
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{field} hors plage [0, 1] : {val}")
        data[field] = round(val, 4)

    ll = data.get("love_language", "")
    if ll not in VALID_LOVE_LANGUAGES:
        raise ValueError(
            f"Love language invalide : {ll!r}. "
            f"Valeurs acceptées : {sorted(VALID_LOVE_LANGUAGES)}"
        )

    data.setdefault("confidence", 0.5)
    data.setdefault("notes", "")
    data["confidence"] = round(float(data["confidence"]), 4)

    return data

# ── Fonction principale ───────────────────────────────────────────────────────

def extract_personality(
    transcript: str,
    person_label: str = "A",
    model: str = "llama-3.1-8b-instant",
    max_retries: int = 2,
    api_key: str | None = None,
) -> dict:
    """
    Appelle Groq pour extraire les scores de personnalité depuis une description.

    Paramètres
    ----------
    transcript    : description libre de la personne
    person_label  : "A" ou "B" (utilisé dans le prompt)
    model         : modèle Groq (llama-3.1-8b-instant par défaut — rapide et gratuit)
    max_retries   : tentatives en cas d'échec de parsing
    api_key       : clé Groq (utilise GROQ_API_KEY si None)

    Retourne
    --------
    dict avec les clés :
      career_ambition, chronotype, emotional_expressiveness,
      openness, spontaneity, love_language, confidence, notes
    """
    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    last_error = None

    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            temperature=0,          # quasi-déterministe
            max_tokens=512,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(transcript, person_label)},
            ],
        )

        raw = response.choices[0].message.content

        try:
            return _parse_and_validate(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < max_retries:
                continue

    raise RuntimeError(
        f"Extraction échouée après {max_retries} tentatives. "
        f"Dernière erreur : {last_error}"
    )


# ── Fallback : valeurs neutres ─────────────────────────────────────────────────

def neutral_personality(love_language: str = "Quality Time") -> dict:
    """Profil neutre (scores = 0.5) quand l'extraction échoue définitivement."""
    if love_language not in VALID_LOVE_LANGUAGES:
        love_language = "Quality Time"
    return {
        "career_ambition": 0.5,
        "chronotype": 0.5,
        "emotional_expressiveness": 0.5,
        "openness": 0.5,
        "spontaneity": 0.5,
        "love_language": love_language,
        "confidence": 0.0,
        "notes": "Fallback : aucun transcript valide fourni.",
    }
