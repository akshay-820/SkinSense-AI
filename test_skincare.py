from backend.skincare.recommend import recommend_from_inputs,dummy_llm_call
from backend.skincare.questionnaire import questionnaire_to_profile


q_raw = {
    "skin_type": "Oily",
    "sensitivity": "Yes",
    "known_allergies": "",
    "current_skincare_routine": "Cleanser,Sunscreen",
    "frequency_of_product_use": "Daily",
    "medication_affecting_skin": "No",
    "diet_habits": "High oily food",
    "sun_exposure": "Moderate",
    "sleep_quality": "Average",
    "goal": "Reduce acne"
}

q_profile = questionnaire_to_profile(q_raw)


img_metrics = {
    "tone": {"tone_label": "Medium", "tone_hex": "#d1a68f", "luminance": 130},
    "oiliness": {"oiliness_label": "High", "oiliness_score": 0.045, "dryness_flag": False},
    "acne": {"acne_count": 8, "acne_severity": "Moderate", "acne_confidence": 0.4},
    "blackheads": {"blackhead_count": 12},
    "face_roi_shape": (256,256,3)
}

result = recommend_from_inputs(q_profile, img_metrics)

import json
print(json.dumps(result, indent=2))
