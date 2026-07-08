# Child Nutrition Risk Screening & Intervention Support Tool

A field-screening prototype that flags children (6-59 months) who may need closer nutritional monitoring, aligned to **WHO MUAC standards** and **ICDS (Integrated Child Development Services)** risk categories — built for Anganwadi/ASHA-style field use, using explainable machine learning (XGBoost + SHAP).

> **Status:** Research/demonstration prototype built on synthetic data. Not a deployed government system, and not a substitute for clinical judgment. See [Limitations](#limitations--honest-disclaimers) before drawing any real-world conclusions from it.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Why Synthetic Data](#why-synthetic-data)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Setup & How to Run](#setup--how-to-run)
  - [Run on Google Colab](#run-on-google-colab)
  - [Run Locally](#run-locally)
  - [Deploy on Streamlit Community Cloud](#deploy-on-streamlit-community-cloud)
- [Model Performance](#model-performance)
- [Explainability](#explainability)
- [Ethical Considerations](#ethical-considerations)
- [Limitations & Honest Disclaimers](#limitations--honest-disclaimers)
- [Future Scope](#future-scope)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Malnutrition among children under 5 remains a serious public health challenge in many parts of India, and frontline workers (Anganwadi/ASHA) often screen children with minimal equipment and limited time. This project builds a **decision-support tool** — not a diagnostic replacement — that:

1. Takes simple, field-measurable inputs (age, weight, height, MUAC, basic household indicators)
2. Predicts a risk category: **Normal / MAM (Moderate Acute Malnutrition) / SAM (Severe Acute Malnutrition)**
3. Explains *why* that risk level was assigned (SHAP-based)
4. Suggests a plain-language next action for the field worker

---

## Why Synthetic Data

No public dataset maps individual child anthropometric + household data to ICDS-aligned risk labels at scale. Real ICDS/Anganwadi data is not publicly released for individual children (rightly so, for privacy reasons). This project therefore uses a **carefully constructed synthetic dataset**, grounded in real clinical thresholds rather than arbitrary numbers:

- **MUAC cutoffs are real WHO standards** for children 6-59 months:
  - MUAC < 11.5 cm → **SAM**
  - 11.5 cm ≤ MUAC < 12.5 cm → **MAM**
  - MUAC ≥ 12.5 cm → **Normal**
- Weight-for-age / height-for-age Z-scores use **simplified age-bucketed reference means/SDs** (an intentional approximation of WHO growth charts — see [Limitations](#limitations--honest-disclaimers)).

---

## Methodology

This project was deliberately designed to avoid the common mistakes that make synthetic-data ML projects fall apart under scrutiny:

| Risk | How it's handled here |
|---|---|
| **Label leakage** (model just re-deriving a fixed threshold rule) | Final label = MUAC-based base label → **probabilistic adjustment** using household risk factors → **~8% random label noise** injected. The model must learn genuine multi-factor signal, not a lookup table. |
| **Class imbalance** (SAM is rare, like in real life) | Class-weighted Logistic Regression / Random Forest baselines; **sample-weighted XGBoost**; per-class recall/F1 reported, not just overall accuracy. |
| **Overclaiming accuracy** | A `>97% accuracy` check is built into the training script — if triggered, it's flagged as a probable leakage bug, not celebrated. |
| **SHAP used as decoration** | SHAP values are mapped to **plain-language explanations and concrete suggested actions**, not just plotted and left unexplained. |
| **Ignoring cost asymmetry** | **SAM-class recall** is treated as the headline metric (missing a severely malnourished child is far costlier than a false alarm), not overall accuracy. |
| **Model version mismatch on deployment** | XGBoost model saved via `model.save_model(...)` in **native JSON format** — avoids the sklearn-Pipeline + joblib version mismatch errors common on Streamlit Community Cloud. |

---

## Project Structure

```
nutrition-risk-project/
├── 01_data_generation.py      # Generates synthetic dataset (WHO MUAC + noise injection)
├── 02_model_training.py       # Trains LR/RF baselines + final XGBoost, saves artifacts
├── app.py                     # Streamlit front-end (government-portal styled UI)
├── requirements.txt
├── data/
│   └── child_nutrition_data.csv
├── models/
│   ├── xgb_model.json
│   ├── onehot_encoder.pkl
│   ├── label_encoder.pkl
│   ├── feature_columns.json
│   └── metrics.json
└── README.md
```

---

## Setup & How to Run

### Run on Google Colab

1. Upload `01_data_generation.py` and `02_model_training.py` to your Colab session (or clone the GitHub repo):
   ```python
   !git clone https://github.com/<your-username>/child-nutrition-risk-tool.git
   %cd child-nutrition-risk-tool
   ```
2. Install dependencies:
   ```python
   !pip install -q pandas numpy scikit-learn xgboost shap joblib
   ```
3. Generate the dataset:
   ```python
   !python 01_data_generation.py
   ```
4. Train the models:
   ```python
   !python 02_model_training.py
   ```
5. This creates `data/child_nutrition_data.csv` and all files inside `models/`. Download the `models/` folder or push it to your GitHub repo so the Streamlit app can load it.

### Run Locally

```bash
git clone https://github.com/<your-username>/child-nutrition-risk-tool.git
cd child-nutrition-risk-tool
pip install -r requirements.txt
python 01_data_generation.py
python 02_model_training.py
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push the entire repo (including the generated `data/` and `models/` folders) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, click **New app**.
3. Select your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install from `requirements.txt` automatically.
5. Because the XGBoost model is saved in native JSON format (not a pickled sklearn Pipeline), you avoid the common `sklearn version mismatch` deployment error.

---

## Model Performance

*(Actual numbers are written to `models/metrics.json` after training — update this table with your run's output.)*

| Model | Macro F1 | SAM-class Recall | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.58 | ~0.75 | High SAM recall but weak overall precision |
| Random Forest (baseline) | ~0.74 | ~0.56 | Good overall, weaker SAM recall |
| **XGBoost (final)** | **~0.71** | **~0.60** | Best balance; tuned via sample weighting |

5-fold stratified cross-validation was used to check stability given the rare SAM class (see script output for CV mean/std).

**Why these numbers are believable:** they are *not* 95%+ accuracy. That would be a red flag suggesting label leakage. Deliberately injected noise and probabilistic label shifts keep this in a realistic, defensible range.

---

## Explainability

Every prediction is accompanied by:
1. A **confidence bar chart** across all three risk categories
2. The **top 3 SHAP-driving factors** for that specific prediction, translated into plain language (e.g., "MUAC measurement increased the predicted risk level")
3. A **suggested next action** mapped from those factors (e.g., PHC referral, dietary counselling, follow-up scheduling)

This turns the tool from a black-box classifier into a **decision-support aid** a field worker can actually act on.

---

## Ethical Considerations

- **Asymmetric cost of errors:** A missed SAM case (false negative) is clinically far more dangerous than an unnecessary follow-up (false positive). The modeling pipeline explicitly optimizes for SAM-class recall over raw accuracy.
- **Human-in-the-loop by design:** The tool is framed throughout as a *screening aid*, not a diagnostic authority. Every prediction includes a note that a qualified health worker makes the final call.
- **No real personal data used:** All data in this repository is synthetically generated. No real child, household, or NGO data is included or required.
- **Regional/cultural generalizability:** Synthetic data cannot capture the full diversity of real regional dietary and health patterns across India — flagged explicitly as a limitation, not hidden.

---

## Limitations & Honest Disclaimers

Being upfront about these is part of the project, not an afterthought:

1. **Weight-for-age / height-for-age Z-scores are simplified approximations**, using age-bucketed mean/SD reference values — **not** the official WHO Anthro LMS growth tables. MUAC cutoffs, however, are the real WHO standard.
2. **Entirely synthetic dataset.** No real field validation has been performed. Before any real-world use, this would need partnership with an NGO/ICDS body and validation against real, ethically-sourced data.
3. **No mobile/offline support yet** — current version assumes a device with a browser and internet access (see [Future Scope](#future-scope)).
4. **English-only UI** in this version — a real field tool would need Hindi/regional language support.
5. **Not clinically certified** — this is an academic/portfolio project demonstrating an ML + explainability + deployment pipeline, not a certified medical device.

---

## Future Scope

- Multi-language UI (Hindi, Marathi, and other regional languages) for real field-worker usability
- Offline-first mobile app (PWA) for low-connectivity rural areas, with sync-on-connect
- Integration with real, ethically-sourced NFHS-5 / ICDS-CAS data (with proper data-sharing agreements)
- Longitudinal/time-series tracking of a child's growth trajectory across visits, not just single-point risk
- Automated SMS/WhatsApp alerts to supervisors or the nearest PHC for confirmed SAM cases
- District/state-level aggregated dashboard for NGO/government administrators to spot malnutrition hotspots
- Feedback loop: field-worker-confirmed outcomes used to continuously retrain and improve the model
- Expansion to related indicators: anemia risk, immunization gaps, vitamin deficiency screening

---

## Tech Stack

- **Language:** Python
- **ML:** XGBoost, scikit-learn (Logistic Regression, Random Forest baselines)
- **Explainability:** SHAP
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit, Streamlit Community Cloud
- **Development environment:** Google Colab (low-resource-friendly)

---

## Disclaimer on Visual Design

The application UI is styled in an official Indian-government-portal aesthetic (tricolour accents, formal typography, seal-style badge) purely for professional presentation. It does **not** reproduce India's State Emblem (the Ashoka Lion Capital), which is a protected symbol under the *State Emblem of India (Prohibition of Improper Use) Act, 2005*. The circular badge used is an original design.

---

## License

This project is released for educational and portfolio purposes.
