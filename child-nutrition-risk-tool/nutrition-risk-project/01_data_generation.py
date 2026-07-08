# ============================================================================
# 01_data_generation.py
# Child Nutrition Risk Screening & Intervention Support Tool
# ----------------------------------------------------------------------------
# Generates a synthetic, WHO/ICDS-aligned dataset for children aged 6-59 months.
#
# METHODOLOGY (read this before using the data):
#   1. MUAC (Mid-Upper Arm Circumference) thresholds used here are REAL WHO
#      cutoffs for children 6-59 months:
#         MUAC < 11.5 cm        -> Severe Acute Malnutrition (SAM)
#         11.5 cm <= MUAC < 12.5 cm -> Moderate Acute Malnutrition (MAM)
#         MUAC >= 12.5 cm       -> Normal
#      Reference: WHO/UNICEF Joint Statement on MUAC screening.
#
#   2. Weight-for-age and height-for-age Z-scores are APPROXIMATED using
#      simplified age-bucketed mean/SD reference values, NOT the official
#      WHO LMS growth tables (those require the full WHO Anthro lookup
#      tables, which are not embedded here). This is an intentional,
#      documented simplification -- flagged again in the README.
#
#   3. The final risk label is NOT a pure deterministic threshold rule.
#      It is built as: MUAC-based base label -> probabilistic adjustment
#      using household/behavioural risk factors -> label noise injection.
#      This avoids trivial leakage (model just re-deriving a fixed rule)
#      and forces the model to learn genuine multi-factor signal.
#
# Output: data/child_nutrition_data.csv
# ============================================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES = 6000
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISTRICTS = [
    "Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain",
    "Rewa", "Sagar", "Satna", "Ratlam", "Dewas"
]

# Approximate age-bucketed weight/height reference means & SDs (kg / cm).
# NOTE: simplified reference values for demonstration -- see README limitations.
AGE_REFERENCE = {
    # (age_min, age_max): (weight_mean, weight_sd, height_mean, height_sd)
    (6, 12):  (8.5, 1.1, 70.0, 3.0),
    (12, 24): (10.5, 1.3, 79.0, 3.5),
    (24, 36): (12.7, 1.5, 89.0, 4.0),
    (36, 48): (14.5, 1.7, 97.0, 4.2),
    (48, 60): (16.3, 1.9, 104.0, 4.5),
}


def get_reference(age_months):
    for (lo, hi), ref in AGE_REFERENCE.items():
        if lo <= age_months < hi:
            return ref
    return AGE_REFERENCE[(48, 60)]


def generate_row():
    age_months = np.random.randint(6, 59)
    gender = np.random.choice(["Male", "Female"])
    w_mean, w_sd, h_mean, h_sd = get_reference(age_months)

    # Household / behavioural risk factors
    income_bracket = np.random.choice(
        ["Low", "Lower-Middle", "Middle"], p=[0.45, 0.35, 0.20]
    )
    sanitation_access = np.random.choice(["No", "Partial", "Yes"], p=[0.30, 0.30, 0.40])
    mother_literacy = np.random.choice(["Illiterate", "Primary", "Secondary+"], p=[0.35, 0.35, 0.30])
    dietary_diversity_score = np.clip(np.random.normal(4.5, 1.8), 0, 9)  # 0-9 food groups
    immunization_status = np.random.choice(["Incomplete", "Complete"], p=[0.30, 0.70])
    district = np.random.choice(DISTRICTS)

    # Base risk pressure from socio-economic factors (drives correlated,
    # not independent, measurement generation -> realistic dataset)
    risk_pressure = 0.0
    risk_pressure += {"Low": 0.35, "Lower-Middle": 0.15, "Middle": 0.0}[income_bracket]
    risk_pressure += {"No": 0.25, "Partial": 0.10, "Yes": 0.0}[sanitation_access]
    risk_pressure += {"Illiterate": 0.20, "Primary": 0.08, "Secondary+": 0.0}[mother_literacy]
    risk_pressure += max(0, (5 - dietary_diversity_score)) * 0.05
    risk_pressure += 0.15 if immunization_status == "Incomplete" else 0.0

    # Weight/height drawn from age reference, nudged down by risk pressure
    weight = np.random.normal(w_mean - risk_pressure * 1.8, w_sd)
    height = np.random.normal(h_mean - risk_pressure * 2.5, h_sd)
    weight = max(3.0, weight)
    height = max(55.0, height)

    # MUAC correlated with weight-for-age deficit + independent noise
    weight_for_age_z = (weight - w_mean) / w_sd
    height_for_age_z = (height - h_mean) / h_sd
    muac_base = 13.5 + weight_for_age_z * 0.9
    muac = np.clip(np.random.normal(muac_base, 0.6), 8.0, 17.0)

    # ---- Base label purely from real WHO MUAC cutoffs ----
    if muac < 11.5:
        base_label = "SAM"
    elif muac < 12.5:
        base_label = "MAM"
    else:
        base_label = "Normal"

    # ---- Probabilistic adjustment using non-MUAC risk factors ----
    # Gives the model genuine multi-factor signal to learn instead of
    # just re-deriving the MUAC threshold rule.
    labels_order = ["Normal", "MAM", "SAM"]
    idx = labels_order.index(base_label)
    shift_prob = min(0.25, risk_pressure * 0.4)
    if np.random.rand() < shift_prob and idx < 2:
        idx += 1  # push toward higher risk category
    elif np.random.rand() < 0.05 and idx > 0:
        idx -= 1  # small chance of improvement (natural variance)
    final_label = labels_order[idx]

    # ---- Label noise injection (~8%) to simulate real-world
    # measurement error / imperfect field conditions ----
    if np.random.rand() < 0.08:
        final_label = np.random.choice(labels_order)

    return {
        "age_months": age_months,
        "gender": gender,
        "district": district,
        "weight_kg": round(weight, 2),
        "height_cm": round(height, 1),
        "muac_cm": round(muac, 2),
        "weight_for_age_z": round(weight_for_age_z, 2),
        "height_for_age_z": round(height_for_age_z, 2),
        "income_bracket": income_bracket,
        "sanitation_access": sanitation_access,
        "mother_literacy": mother_literacy,
        "dietary_diversity_score": round(dietary_diversity_score, 1),
        "immunization_status": immunization_status,
        "risk_category": final_label,
    }


def main():
    rows = [generate_row() for _ in range(N_SAMPLES)]
    df = pd.DataFrame(rows)

    out_path = os.path.join(OUTPUT_DIR, "child_nutrition_data.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass distribution:")
    print(df["risk_category"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
