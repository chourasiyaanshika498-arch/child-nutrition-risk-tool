# ============================================================================
# app.py -- Child Nutrition Risk Screening & Intervention Support Tool
# Streamlit front-end styled as an official government health-portal.
#
# NOTE ON THE EMBLEM: India's State Emblem (the Ashoka Lion Capital) is a
# protected symbol under the State Emblem of India (Prohibition of Improper
# Use) Act, 2005, and cannot be reproduced without government authorisation.
# This UI instead uses an original circular seal/shield motif in the same
# official tricolour palette to achieve a "govt. portal" look without using
# the protected emblem itself.
# ============================================================================

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

# Resolve paths relative to this script's own location, not the process's
# working directory (Streamlit Cloud's cwd is the repo root, not the app's
# folder, which breaks bare relative paths like "models/xgb_model.json").
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Child Nutrition Risk Screening Portal",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Clean government-portal styling (white background, red accent, card layout)
# NOTE: India's State Emblem (Ashoka Lion Capital) is a protected symbol under
# the State Emblem of India (Prohibition of Improper Use) Act, 2005, and is
# NOT reproduced here. The circular badge below is an original tricolour
# motif, not the official emblem.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
    }

    .gov-header {
        background: linear-gradient(90deg, #FF9933 0%, #FF9933 33%, #FFFFFF 33%, #FFFFFF 34%, #FFFFFF 66%, #138808 66%, #138808 100%);
        height: 5px;
        width: 100%;
        margin-bottom: 0px;
    }

    .gov-banner {
        background-color: #FFFFFF;
        padding: 20px 30px;
        display: flex;
        align-items: center;
        gap: 18px;
        border-bottom: 3px solid #B31217;
    }

    .gov-title {
        color: #1A1A1A;
        font-family: 'Merriweather', serif;
        font-size: 22px;
        font-weight: 700;
        line-height: 1.3;
        margin: 0;
    }

    .gov-subtitle {
        color: #5A5A5A;
        font-size: 13px;
        margin-top: 2px;
    }

    .section-heading {
        font-family: 'Merriweather', serif;
        font-size: 16px;
        color: #1A1A1A;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 2px solid #B31217;
        display: inline-block;
        padding-bottom: 4px;
    }

    .risk-badge {
        padding: 14px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 18px;
        text-align: center;
        color: white;
        margin-bottom: 10px;
    }

    .risk-normal   { background-color: #138808; }
    .risk-mam      { background-color: #E24B4A; }
    .risk-sam      { background-color: #8C0E12; }

    .footer-note {
        font-size: 12px;
        color: #6B7280;
        text-align: center;
        margin-top: 30px;
        padding: 14px;
        border-top: 1px solid #E5E7EB;
    }

    div.stButton > button {
        background-color: #B31217;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 10px 26px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #8C0E12;
        color: white;
    }
</style>

<div class="gov-header"></div>
<div class="gov-banner">
    <div>
        <p class="gov-title">Child Nutrition Risk Screening &amp; Intervention Support Portal</p>
        <p class="gov-subtitle">Field Screening Tool for Anganwadi &amp; ASHA Workers &nbsp;|&nbsp; ICDS-Aligned Prototype</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# Load model artifacts (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODELS_DIR, "xgb_model.json"))

    ohe = joblib.load(os.path.join(MODELS_DIR, "onehot_encoder.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    explainer = shap.TreeExplainer(model)
    return model, ohe, label_encoder, feature_columns, explainer


model, ohe, label_encoder, feature_columns, explainer = load_artifacts()

CATEGORICAL_COLS = [
    "gender", "district", "income_bracket",
    "sanitation_access", "mother_literacy", "immunization_status",
]
NUMERIC_COLS = [
    "age_months", "weight_kg", "height_cm", "muac_cm",
    "weight_for_age_z", "height_for_age_z", "dietary_diversity_score",
]

DISTRICTS = [
    "Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain",
    "Rewa", "Sagar", "Satna", "Ratlam", "Dewas",
]

# ----------------------------------------------------------------------------
# Input form -- deliberately minimal, field-worker friendly
# ----------------------------------------------------------------------------
st.markdown('<p class="section-heading">Child &amp; Household Details</p>', unsafe_allow_html=True)
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        age_months = st.slider("Age (months)", 6, 59, 24)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        district = st.selectbox("District", DISTRICTS)

    with col2:
        weight_kg = st.number_input("Weight (kg)", min_value=3.0, max_value=25.0, value=11.0, step=0.1)
        height_cm = st.number_input("Height / Length (cm)", min_value=55.0, max_value=120.0, value=80.0, step=0.5)
        muac_cm = st.number_input("MUAC -- Mid-Upper Arm Circumference (cm)", min_value=8.0, max_value=17.0, value=13.5, step=0.1)

    with col3:
        income_bracket = st.selectbox("Household Income Bracket", ["Low", "Lower-Middle", "Middle"])
        sanitation_access = st.selectbox("Sanitation Access", ["No", "Partial", "Yes"])
        mother_literacy = st.selectbox("Mother's Literacy Level", ["Illiterate", "Primary", "Secondary+"])

    col4, col5 = st.columns(2)
    with col4:
        dietary_diversity_score = st.slider("Dietary Diversity Score (food groups consumed, 0-9)", 0, 9, 4)
    with col5:
        immunization_status = st.selectbox("Immunization Status", ["Incomplete", "Complete"])

run = st.button("Run Screening Assessment")

# ----------------------------------------------------------------------------
# Prediction + explanation
# ----------------------------------------------------------------------------
if run:
    # Approximate age-adjusted Z-scores using the same simplified reference
    # logic as training data generation (documented limitation -- see README)
    ref_table = {
        (6, 12):  (8.5, 1.1, 70.0, 3.0),
        (12, 24): (10.5, 1.3, 79.0, 3.5),
        (24, 36): (12.7, 1.5, 89.0, 4.0),
        (36, 48): (14.5, 1.7, 97.0, 4.2),
        (48, 60): (16.3, 1.9, 104.0, 4.5),
    }
    w_mean, w_sd, h_mean, h_sd = next(
        v for (lo, hi), v in ref_table.items() if lo <= age_months < hi
    )
    weight_for_age_z = (weight_kg - w_mean) / w_sd
    height_for_age_z = (height_cm - h_mean) / h_sd

    input_dict = {
        "age_months": age_months, "weight_kg": weight_kg, "height_cm": height_cm,
        "muac_cm": muac_cm, "weight_for_age_z": weight_for_age_z,
        "height_for_age_z": height_for_age_z,
        "dietary_diversity_score": dietary_diversity_score,
        "gender": gender, "district": district, "income_bracket": income_bracket,
        "sanitation_access": sanitation_access, "mother_literacy": mother_literacy,
        "immunization_status": immunization_status,
    }
    input_df = pd.DataFrame([input_dict])

    cat_encoded = ohe.transform(input_df[CATEGORICAL_COLS])
    cat_cols_out = ohe.get_feature_names_out(CATEGORICAL_COLS)
    X_input = pd.concat(
        [input_df[NUMERIC_COLS].reset_index(drop=True),
         pd.DataFrame(cat_encoded, columns=cat_cols_out)],
        axis=1,
    ).reindex(columns=feature_columns, fill_value=0)

    pred_class_idx = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_class_idx])[0]

    shap_values = explainer.shap_values(X_input)
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_class_idx][0]
    else:
        class_shap = shap_values[0, :, pred_class_idx]

    top_idx = np.argsort(-np.abs(class_shap))[:3]
    top_features = [(X_input.columns[i], class_shap[i]) for i in top_idx]

    st.markdown('<p class="section-heading">Screening Result</p>', unsafe_allow_html=True)
    with st.container(border=True):
        badge_class = {"Normal": "risk-normal", "MAM": "risk-mam", "SAM": "risk-sam"}[pred_label]
        badge_text = {
            "Normal": "NORMAL -- No Immediate Risk Indicated",
            "MAM": "MODERATE ACUTE MALNUTRITION (MAM) -- Monitoring Advised",
            "SAM": "SEVERE ACUTE MALNUTRITION (SAM) -- Urgent Referral Advised",
        }[pred_label]
        st.markdown(f'<div class="risk-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)

        proba_df = pd.DataFrame({
            "Category": label_encoder.classes_,
            "Confidence": pred_proba,
        }).sort_values("Confidence", ascending=False)
        st.bar_chart(proba_df.set_index("Category"))

        st.markdown("**Key contributing factors (SHAP-based explanation):**")
        factor_labels = {
            "muac_cm": "MUAC measurement",
            "weight_for_age_z": "Weight-for-age (Z-score)",
            "height_for_age_z": "Height-for-age (Z-score)",
            "dietary_diversity_score": "Dietary diversity",
            "weight_kg": "Weight",
            "height_cm": "Height",
            "age_months": "Age",
        }
        action_map = {
            "muac_cm": "Recommend repeat MUAC measurement and PHC referral if confirmed.",
            "weight_for_age_z": "Advise increased caloric and protein intake; schedule follow-up weigh-in.",
            "height_for_age_z": "Flag for chronic growth monitoring over subsequent visits.",
            "dietary_diversity_score": "Counsel household on food group diversity (proteins, fruits, vegetables).",
            "weight_kg": "Track weight trend over next 3 monthly visits.",
            "height_cm": "Track height/length trend over next 3 monthly visits.",
            "age_months": "No direct action -- contextual factor only.",
        }

        for feat, val in top_features:
            direction = "increased" if val > 0 else "decreased"
            label = factor_labels.get(feat, feat)
            st.markdown(f"- **{label}** {direction} the predicted risk level.")

        st.markdown("**Suggested Action(s):**")
        for feat, _ in top_features:
            if feat in action_map:
                st.markdown(f"- {action_map[feat]}")
        st.markdown(
            "- This is a **screening aid only**. Final clinical decisions must "
            "be made by a qualified health worker."
        )

st.markdown("""
<div class="footer-note">
    Prototype developed for demonstration purposes on synthetic data aligned to WHO MUAC
    standards and ICDS categories. Not an officially endorsed government system.
    Not for use in actual clinical diagnosis without field validation.
</div>
""", unsafe_allow_html=True)
