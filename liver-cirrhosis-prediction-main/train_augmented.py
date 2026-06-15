"""
Train ensemble on augmented dataset (real + 4800 synthetic patients).
Saves best models to backend/models/.
"""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import joblib

# ── Load augmented dataset ───────────────────────────────────────────────────
print("Loading augmented dataset …")
df = pd.read_csv("datasets/cirrhosis_augmented.csv")
df = df.dropna(subset=["Stage"])

# Fill any remaining nulls with median
for c in ["Cholesterol","Copper","Alk_Phos","SGOT","Tryglicerides",
          "Platelets","Prothrombin","Bilirubin","Albumin"]:
    df[c] = df[c].fillna(df[c].median())

# Encode categoricals
for c in ["Sex","Ascites","Hepatomegaly","Spiders","Edema"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

FEATURES = [
    "Age","Sex","Ascites","Hepatomegaly","Spiders","Edema",
    "Bilirubin","Cholesterol","Albumin","Copper","Alk_Phos",
    "SGOT","Tryglicerides","Platelets","Prothrombin",
]

X = df[FEATURES].values.astype(np.float32)
y = (df["Stage"].values.astype(int) - 1)   # 1-4 → 0-3

print(f"Total samples: {len(X)}  |  Classes: {np.bincount(y)}")

# ── Train / test split (stratified, 20% test) ────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")
print()

# ── Train models ─────────────────────────────────────────────────────────────
print("Training XGBoost …")
xgb_m = xgb.XGBClassifier(
    n_estimators=600, learning_rate=0.03, max_depth=6,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
    objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", random_state=42, tree_method="hist",
    early_stopping_rounds=40,
)
xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)

print("Training Gradient Boosting …")
gb_m = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.85, min_samples_leaf=3, random_state=42
)
gb_m.fit(X_tr_s, y_tr)

print("Training Random Forest …")
rf_m = RandomForestClassifier(
    n_estimators=500, max_depth=14, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
rf_m.fit(X_tr_s, y_tr)

# ── Evaluate individual models ────────────────────────────────────────────────
print()
print("=" * 55)
print("Individual model performance on held-out test set:")
print("=" * 55)
for name, m in [("XGBoost", xgb_m), ("GradBoost", gb_m), ("RandomForest", rf_m)]:
    pred = m.predict(X_te_s)
    acc  = accuracy_score(y_te, pred)
    f1   = f1_score(y_te, pred, average="weighted")
    print(f"  {name:14s}: Accuracy={acc:.4f}  F1={f1:.4f}")

# ── Ensemble (soft-vote weighted) ─────────────────────────────────────────────
p_xgb = xgb_m.predict_proba(X_te_s)
p_gb  = gb_m.predict_proba(X_te_s)
p_rf  = rf_m.predict_proba(X_te_s)

ensemble_proba = 0.40 * p_xgb + 0.40 * p_gb + 0.20 * p_rf
ens_pred       = np.argmax(ensemble_proba, axis=1)
ens_acc        = accuracy_score(y_te, ens_pred)
ens_f1         = f1_score(y_te, ens_pred, average="weighted")

print()
print("=" * 55)
print(f"  ENSEMBLE         : Accuracy={ens_acc:.4f}  F1={ens_f1:.4f}")
print("=" * 55)
print()
print("Classification Report (Ensemble):")
print(classification_report(
    y_te, ens_pred,
    target_names=["Stage 1","Stage 2","Stage 3","Stage 4"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_te, ens_pred))
print()

# Stage 4 AUC (most clinically important)
y_bin      = (y_te == 3).astype(int)
stage4_auc = roc_auc_score(y_bin, ensemble_proba[:, 3])
print(f"Stage 4 (Cirrhosis) Detection AUC: {stage4_auc:.4f}")

# 5-fold cross-validation on full dataset
print()
print("5-fold cross-validation (GradBoost on full augmented dataset):")
cv_scores = cross_val_score(
    gb_m, scaler.transform(X), y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1
)
print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Per-fold:   {[round(s,4) for s in cv_scores]}")

# ── Feature importance ────────────────────────────────────────────────────────
importances = dict(zip(FEATURES, xgb_m.feature_importances_))
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print()
print("Top feature importances (XGBoost):")
for feat, imp in top_features:
    bar = "█" * int(imp * 200)
    print(f"  {feat:16s}: {imp:.4f}  {bar}")

# ── Save models ───────────────────────────────────────────────────────────────
Path("backend/models").mkdir(exist_ok=True)
joblib.dump(scaler, "backend/models/symptom_scaler.pkl")
joblib.dump(xgb_m,  "backend/models/xgb_model.pkl")
joblib.dump(gb_m,   "backend/models/gb_model.pkl")
joblib.dump(rf_m,   "backend/models/rf_model.pkl")
joblib.dump(gb_m,   "backend/models/symptom_model.pkl")

metadata = {
    "features":          FEATURES,
    "n_classes":         4,
    "stage_labels": {
        "0": "Stage 1 - Mild Fibrosis",
        "1": "Stage 2 - Moderate Fibrosis",
        "2": "Stage 3 - Severe Fibrosis",
        "3": "Stage 4 - Cirrhosis",
    },
    "dataset":           "cirrhosis_augmented.csv",
    "total_samples":     int(len(X)),
    "real_samples":      412,
    "synthetic_samples": int(len(X)) - 412,
    "train_size":        int(len(X_tr)),
    "test_size":         int(len(X_te)),
    "accuracy_test":     float(ens_acc),
    "f1_weighted":       float(ens_f1),
    "stage4_auc":        float(stage4_auc),
    "cv_mean":           float(cv_scores.mean()),
    "cv_std":            float(cv_scores.std()),
    "ensemble_weights":  {"xgboost": 0.40, "gradboost": 0.40, "random_forest": 0.20},
    "feature_importance": {k: float(v) for k, v in top_features},
}

with open("backend/models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print()
print("✓ All models saved to backend/models/")
print(f"✓ Metadata saved  →  Accuracy={ens_acc:.4f}  F1={ens_f1:.4f}  Stage4_AUC={stage4_auc:.4f}")
