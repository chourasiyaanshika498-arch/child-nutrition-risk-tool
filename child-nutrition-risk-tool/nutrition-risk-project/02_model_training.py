# ============================================================================
# 02_model_training.py
# Trains baseline models (Logistic Regression, Random Forest) and the final
# XGBoost model on the synthetic child nutrition dataset, with:
#   - Stratified K-fold cross-validation (class imbalance is real here)
#   - Class-weighting to handle rare SAM class
#   - Per-class precision/recall/F1 reporting (NOT just accuracy)
#   - SHAP explainability
#   - Model saved in XGBoost's native JSON format (avoids sklearn/joblib
#     version-mismatch errors on Streamlit Cloud)
# ============================================================================

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATA_PATH = "data/child_nutrition_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_ORDER = ["Normal", "MAM", "SAM"]  # low -> high risk

CATEGORICAL_COLS = [
    "gender", "district", "income_bracket",
    "sanitation_access", "mother_literacy", "immunization_status",
]
NUMERIC_COLS = [
    "age_months", "weight_kg", "height_cm", "muac_cm",
    "weight_for_age_z", "height_for_age_z", "dietary_diversity_score",
]


def load_and_encode():
    df = pd.read_csv(DATA_PATH)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = ohe.fit_transform(df[CATEGORICAL_COLS])
    cat_cols_out = ohe.get_feature_names_out(CATEGORICAL_COLS)

    X = pd.concat(
        [df[NUMERIC_COLS].reset_index(drop=True),
         pd.DataFrame(cat_encoded, columns=cat_cols_out)],
        axis=1,
    )

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df["risk_category"])

    return X, y, le, ohe


def sam_recall(y_true, y_pred, sam_idx):
    """Recall specifically on the SAM class -- the metric that matters most,
    since a missed SAM case (false negative) is far costlier than a false
    positive flag."""
    mask = y_true == sam_idx
    if mask.sum() == 0:
        return np.nan
    return (y_pred[mask] == sam_idx).mean()


def evaluate_model(model, X_test, y_test, label_encoder, name):
    y_pred = model.predict(X_test)
    sam_idx = list(label_encoder.classes_).index("SAM")

    print(f"\n===== {name} =====")
    print(classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print(f"SAM-class recall (most important metric): "
          f"{sam_recall(y_test, y_pred, sam_idx):.3f}")

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.3f}")
    return macro_f1


def main():
    X, y, label_encoder, ohe = load_and_encode()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------------- Baseline: Logistic Regression ----------------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    evaluate_model(lr, X_test, y_test, label_encoder, "Logistic Regression (baseline)")

    # ---------------- Baseline: Random Forest ----------------
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, label_encoder, "Random Forest (baseline)")

    # ---------------- Final model: XGBoost with sample weights ----------------
    # class_weight isn't native to XGBoost multi-class; use sample_weight
    # computed from inverse class frequency to address SAM being rare.
    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weight = np.array([class_weights[label] for label in y_train])

    # 5-fold stratified CV to sanity-check stability (rare class -> variance risk)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_model = xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=42,
        )
        fold_model.fit(
            X_train.iloc[train_idx], y_train[train_idx],
            sample_weight=sample_weight[train_idx],
        )
        fold_pred = fold_model.predict(X_train.iloc[val_idx])
        cv_f1_scores.append(f1_score(y_train[val_idx], fold_pred, average="macro"))

    print(f"\n5-fold CV macro F1 (XGBoost): "
          f"mean={np.mean(cv_f1_scores):.3f}, std={np.std(cv_f1_scores):.3f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.08,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    final_macro_f1 = evaluate_model(
        xgb_model, X_test, y_test, label_encoder, "XGBoost (final model)"
    )

    # Sanity check against suspiciously high accuracy (possible leakage)
    y_pred = xgb_model.predict(X_test)
    acc = (y_pred == y_test).mean()
    if acc > 0.97:
        print(
            "\nWARNING: accuracy > 97% -- re-check for label leakage before "
            "reporting this number. (Not expected with this pipeline's noise "
            "injection, but always verify.)"
        )
    else:
        print(f"\nOverall accuracy: {acc:.3f} (plausible range -- no leakage red flag)")

    # ---------------- SHAP explainability ----------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    print("\nSHAP values computed successfully "
          f"(shape info: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape})")

    # ---------------- Save artifacts ----------------
    # Native XGBoost JSON format -- avoids sklearn Pipeline + joblib
    # version-mismatch errors when loading on Streamlit Community Cloud.
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    joblib.dump(ohe, os.path.join(MODEL_DIR, "onehot_encoder.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)

    metrics = {
        "cv_macro_f1_mean": float(np.mean(cv_f1_scores)),
        "cv_macro_f1_std": float(np.std(cv_f1_scores)),
        "test_macro_f1": float(final_macro_f1),
        "test_accuracy": float(acc),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll artifacts saved to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
