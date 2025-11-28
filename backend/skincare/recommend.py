from typing import Dict, Any, Callable, Optional
import json
import textwrap

__all__ = [
    "recommend_from_inputs",
    "dummy_llm_call",
    "openai_llm_call",
    "generate_recommendations",
    "build_prompt"
]


def build_prompt(profile: Dict[str, Any]) -> str:
    profile_json = json.dumps(profile, indent=2, ensure_ascii=False)
    prompt = textwrap.dedent(f"""
    You are a certified skincare advisor. Use the PROFILE below and produce ONLY valid JSON matching the schema described.

    PROFILE:
    {profile_json}

    INSTRUCTIONS:
    - Output JSON with keys: routine, products, safety_notes, progress_plan.
      * routine: ordered list of steps. Each step: {{ "step": int, "when": "AM/PM", "action": str, "notes": str (optional) }}
      * products: list of up to 5 product suggestions. Each product: {{ "name": str, "why": str, "key_ingredients": [str,...], "suitable_for": str }}
      * safety_notes: list of strings (allergies, medication interactions, sun guidance).
      * progress_plan: {{ "weeks": int, "checkpoints": [str,...], "photo_schedule": "e.g., every 2 weeks" }}

    GUIDELINES:
    - Respect sensitivity, allergies, and medication flags in profile.risk_flags.
    - If acne severity is 'Severe', include a note to consult dermatologist and avoid strong actives without medical approval.
    - Prefer ingredient-focused suggestions (e.g., 'niacinamide', 'salicylic acid', 'hyaluronic acid', 'zinc oxide').
    - Avoid vendor/brand marketing; use descriptive product types when possible.
    - Keep answers concise and practical.

    RETURN only JSON (no extra text).
    """).strip()
    return prompt

def generate_recommendations(profile: Dict[str, Any], llm_call: Callable[[str], str]) -> Dict[str, Any]:
    prompt = build_prompt(profile)
    raw = llm_call(prompt)

    try:
        parsed = json.loads(raw)
        return {"ok": True, "recommendations": parsed}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end+1])
                return {"ok": True, "recommendations": parsed, "raw": raw}
            except Exception:
                pass
        return {"ok": False, "error": "LLM output not valid JSON", "raw": raw}


#for testing 
def dummy_llm_call(prompt: str) -> str:
    out = {
        "routine": [
            {"step":1, "when":"AM", "action":"Gentle cleanser", "notes":"Use lukewarm water"},
            {"step":2, "when":"AM", "action":"Lightweight non-comedogenic moisturizer", "notes":""},
            {"step":3, "when":"AM", "action":"Broad-spectrum sunscreen SPF50", "notes":"Apply 2mg/cm2"},
            {"step":4, "when":"PM", "action":"Cleanser then targeted acne serum (salicylic acid 2%)", "notes":"Patch test first if sensitive"}
        ],
        "products": [
            {"name":"Gentle foaming cleanser", "why":"removes oil without stripping", "key_ingredients":["glycerin","niacinamide"], "suitable_for":"Oily/Combination"},
            {"name":"Light gel moisturizer", "why":"hydrates without clogging pores", "key_ingredients":["hyaluronic acid"], "suitable_for":"All"},
            {"name":"Mineral SPF 50", "why":"broad spectrum, gentle", "key_ingredients":["zinc oxide"], "suitable_for":"Sensitive/All"}
        ],
        "safety_notes": [
            "If on isotretinoin or oral medication, consult dermatologist before starting exfoliating acids.",
            "Avoid combining benzoyl peroxide with retinoids without guidance.",
            "Patch test any new product if you have known allergies."
        ],
        "progress_plan": {
            "weeks": 4,
            "checkpoints": ["Week 2: check for irritation", "Week 4: photo and acne_count"],
            "photo_schedule": "Take front-facing photo every 2 weeks under consistent lighting"
        }
    }
    return json.dumps(out)

#forimplementation
def openai_llm_call(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key="API_KEY")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",     # or any model you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
    )

    return response.choices[0].message["content"]


def recommend_from_inputs(q_profile: Dict[str, Any], image_metrics: Dict[str, Any], llm_call: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    if llm_call is None:
        llm_call=openai_llm_call

    try:
        from .questionnaire import combine_profile_with_image_metrics
    except Exception:
        # fallback: naive merge
        combined = dict(q_profile)
        combined["image_metrics"] = image_metrics
    else:
        combined = combine_profile_with_image_metrics(q_profile, image_metrics)

    return generate_recommendations(combined, llm_call)


# quick CLI test
# if __name__ == "__main__":
#     import json
#     # minimal fake inputs
#     q = {
#         "skin_type":"Oily",
#         "sensitivity":False,
#         "known_allergies":"",
#         "current_skincare_routine":["Cleanser","Sunscreen"],
#         "frequency_of_product_use":"Daily",
#         "frequency_score":1.0,
#         "medication_affecting_skin":False,
#         "medication_text":"",
#         "diet_habits":"high_oily",
#         "sun_exposure_hours":2.0,
#         "sleep_quality":"Average",
#         "goal":"Reduce acne"
#     }
#     img = {
#         "tone":{"tone_label":"Medium","tone_hex":"#d1a68f","luminance":130.2},
#         "oiliness":{"oiliness_label":"High","oiliness_score":0.04,"dryness_flag":False},
#         "acne":{"acne_count":8,"acne_severity":"Moderate","acne_confidence":0.42},
#         "blackheads":{"blackhead_count":12}
#     }
#     out = recommend_from_inputs(q, img, llm_call=dummy_llm_call)
#     print(json.dumps(out, indent=2))
