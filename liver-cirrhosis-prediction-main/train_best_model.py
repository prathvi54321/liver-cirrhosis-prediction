"""Retrain best ensemble on cirrhosis.csv with SMOTE + tuned hyperparameters."""
import warnings, numpy as np, json
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from pathlib import Path

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv("datasets/cirrhosis.csv")
df = df.drop(columns=["ID", "N_Days", "Status", "Drug"], errors="ignore")
df = df.dropna(subset=["Stage"])

for c in ["Cholesterol","Copper","Alk_Phos","SGOT","Tryglicerides","Platelets","Prothrombin","Bilirubin","Albumin"]:
    df[c] = df[c].fillna(df[c].median())

for c in ["Sex","Ascites","Hepatomegaly","Spiders","Edema"]:
    df[c] = LabelEncoder().fit_transform(df[c].astype(str))

FEATURES = [
    "Age","Sex","Ascites","Hepatomegaly","Spiders","Edema",
    "Bilirubin","Cholesterol","Albumin","Copper","Alk_Phos",
    "SGOT","Tryglicerides","Platelets","Prothrombin",
]

X = df[FEATURES].values.astype(np.float32)
y = (df["Stage"].values.astype(int) - 1)   # 1-4 → 0-3

print(f"Dataset: {len(X)} samples  |  Classes: {np.bincount(y)}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── SMOTE on training only ────────────────────────────────────────────────────
sm = SMOTE(random_state=42, k_neighbors=3)
X_tr_sm, y_tr_sm = sm.fit_resample(X_tr_s, y_tr)
print(f"After SMOTE: {np.bincount(y_tr_sm)}")

# ── Train ─────────────────────────────────────────────────────────────────────
xgb_m = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=6,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=1,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
    objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", random_state=42, tree_method="hist"
)
xgb_m.fit(X_tr_sm, y_tr_sm, eval_set=[(X_te_s, y_te)], verbose=False)

gb_m = GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.85, min_samples_leaf=2, random_state=42
)
gb_m.fit(X_tr_sm, y_tr_sm)

rf_m = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
rf_m.fit(X_tr_sm, y_tr_sm)

# ── Evaluate ──────────────────────────────────────────────────────────────────
for name, m in [("XGBoost", xgb_m), ("GradBoost", gb_m), ("RandomForest", rf_m)]:
    p   = m.predict(X_te_s)
    acc = accuracy_score(y_te, p)
    f1  = f1_score(y_te, p, average="weighted")
    print(f"{name:14s}: acc={acc:.3f}  f1={f1:.3f}")

p_ens    = 0.40*xgb_m.predict_proba(X_te_s) + 0.40*gb_m.predict_proba(X_te_s) + 0.20*rf_m.predict_proba(X_te_s)
ens_pred = np.argmax(p_ens, axis=1)
ens_acc  = accuracy_score(y_te, ens_pred)
ens_f1   = f1_score(y_te, ens_pred, average="weighted")
print(f"{'Ensemble':14s}: acc={ens_acc:.3f}  f1={ens_f1:.3f}")
print()
print(classification_report(y_te, ens_pred,
      target_names=["Stage 1 (n=4)","Stage 2 (n=19)","Stage 3 (n=31)","Stage 4 (n=29)"]))

# Stage 4 (cirrhosis) AUC — most important clinically
y_te_bin = (y_te == 3).astype(int)
stage4_auc = roc_auc_score(y_te_bin, p_ens[:, 3])
print(f"Stage 4 detection AUC: {stage4_auc:.3f}  (most clinically important)")

# 5-fold CV on full dataset
cv = cross_val_score(gb_m, scaler.transform(X), y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring="accuracy")
print(f"5-fold CV: {cv.mean():.3f} +/- {cv.std():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
Path("backend/models").mkdir(exist_ok=True)
joblib.dump(scaler, "backend/models/symptom_scaler.pkl")
joblib.dump(xgb_m,  "backend/models/xgb_model.pkl")
joblib.dump(gb_m,   "backend/models/gb_model.pkl")
joblib.dump(rf_m,   "backend/models/rf_model.pkl")
joblib.dump(gb_m,   "backend/models/symptom_model.pkl")   # best single

meta = {
    "features": FEATURES,
    "n_classes": 4,
    "stage_labels": {
        "0": "Stage 1 - Mild Fibrosis",
        "1": "Stage 2 - Moderate Fibrosis",
        "2": "Stage 3 - Severe Fibrosis",
        "3": "Stage 4 - Cirrhosis",
    },
    "accuracy_test": float(ens_acc),
    "f1_weighted": float(ens_f1),
    "stage4_auc": float(stage4_auc),
    "cv_mean": float(cv.mean()),
    "cv_std": float(cv.std()),
    "smote_used": True,
    "train_size": int(len(y_tr_sm)),
    "test_size": int(len(y_te)),
    "dataset": "cirrhosis.csv (412 samples)",
}
with open("backend/models/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nAll models saved to backend/models/")
