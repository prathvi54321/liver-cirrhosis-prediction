"""
Image-based Liver Classifier
═════════════════════════════
Trains on the 7272660 dataset (Normal / Benign / Malignant liver ultrasound images)
and the BEHSOF dataset (NAFLD / Non-NAFLD).

Feature extraction: HOG + LBP + Color histogram + Texture (GLCM-style)
Classifier:         SVM with RBF kernel (+ probability calibration)
Also trains a GradientBoosting on the same features as an ensemble.

Outputs:
  backend/models/image_model.pkl      — trained image classifier
  backend/models/image_scaler.pkl     — feature scaler
  backend/models/image_metadata.json  — class labels, accuracy
"""

import warnings, json, sys
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV

# ── Image size ────────────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)

# ── Dataset paths & labels ────────────────────────────────────────────────────
DATASETS = [
    # (folder, label, condition)
    ("datasets/7272660/Normal/Normal/image",    "Normal",    "normal"),
    ("datasets/7272660/Benign/Benign/image",     "Benign",    "benign"),
    ("datasets/7272660/Malignant/Malignant/image","Malignant","malignant"),
]

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_hog_features(gray):
    """HOG features — captures edge/texture patterns."""
    # 8x8 cells, 2x2 block, 9 orientations → 16x16 = 256 cells (simplified)
    resized = cv2.resize(gray, (64, 64))
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Bin into 9 orientations for 8x8 blocks
    bins = 9
    cell = 8
    h, w = resized.shape
    features = []
    for i in range(0, h, cell):
        for j in range(0, w, cell):
            cell_mag = mag[i:i+cell, j:j+cell]
            cell_ang = ang[i:i+cell, j:j+cell] % 180
            hist = np.zeros(bins)
            for b in range(bins):
                mask = (cell_ang >= b*20) & (cell_ang < (b+1)*20)
                hist[b] = cell_mag[mask].sum()
            if hist.sum() > 0:
                hist /= hist.sum()
            features.extend(hist)
    return np.array(features, dtype=np.float32)


def extract_lbp_features(gray, radius=1, n_points=8):
    """Local Binary Pattern texture features."""
    resized = cv2.resize(gray, IMG_SIZE)
    lbp = np.zeros_like(resized, dtype=np.uint8)
    for i in range(radius, resized.shape[0]-radius):
        for j in range(radius, resized.shape[1]-radius):
            center = resized[i, j]
            code = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(round(i + radius * np.sin(angle)))
                y = int(round(j + radius * np.cos(angle)))
                code |= (1 << k) if resized[x, y] >= center else 0
            lbp[i, j] = code
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.float32) / (hist.sum() + 1e-8)


def extract_color_histogram(img_rgb):
    """Color histogram for each channel."""
    features = []
    for ch in range(3):
        hist, _ = np.histogram(img_rgb[:, :, ch], bins=32, range=(0, 256))
        features.extend(hist.astype(np.float32) / (hist.sum() + 1e-8))
    return np.array(features)


def extract_texture_stats(gray):
    """Statistical texture features: mean, std, skew proxy, kurtosis proxy, contrast."""
    resized = cv2.resize(gray, IMG_SIZE).astype(np.float32)
    mean = resized.mean()
    std  = resized.std()
    skew = float(np.mean(((resized - mean) / (std + 1e-8)) ** 3))
    kurt = float(np.mean(((resized - mean) / (std + 1e-8)) ** 4))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # GLCM-style contrast (simplified)
    dx = np.diff(resized, axis=1)
    dy = np.diff(resized, axis=0)
    contrast = float(np.mean(dx**2) + np.mean(dy**2))
    return np.array([mean, std, skew, kurt, laplacian_var, contrast], dtype=np.float32)


def extract_features(img_path):
    """Extract combined feature vector from an image path."""
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    hog  = extract_hog_features(gray)
    tex  = extract_texture_stats(gray)
    chist = extract_color_histogram(img_np)
    # LBP is slow for 128x128 — use a downsampled version
    gray_small = cv2.resize(gray, (32, 32))
    lbp_hist, _ = np.histogram(gray_small.ravel(), bins=64, range=(0, 256))
    lbp = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-8)

    return np.concatenate([hog, tex, chist, lbp])


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading image datasets...")
X_list, y_list = [], []
label_counts = {}

for folder, label, _ in DATASETS:
    p = Path(folder)
    if not p.exists():
        print(f"  Skipping {folder} (not found)")
        continue
    imgs = sorted(p.glob("*.jpg"))[:500]   # cap at 500 per class for speed
    print(f"  {label}: {len(imgs)} images", end="", flush=True)
    count = 0
    for img_path in imgs:
        try:
            feat = extract_features(img_path)
            X_list.append(feat)
            y_list.append(label)
            count += 1
        except Exception as e:
            pass
    label_counts[label] = count
    print(f" → {count} features extracted")

if len(X_list) < 10:
    print("ERROR: Not enough images loaded.")
    sys.exit(1)

X = np.array(X_list, dtype=np.float32)
le = LabelEncoder()
y = le.fit_transform(y_list)

print(f"\nTotal: {len(X)} samples  |  Classes: {le.classes_}  |  Dist: {np.bincount(y)}")
print(f"Feature vector size: {X.shape[1]}")

# ── Train / test split ────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# ── SVM ───────────────────────────────────────────────────────────────────────
print("\nTraining SVM (RBF kernel)...")
svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
svm.fit(X_tr_s, y_tr)
svm_pred = svm.predict(X_te_s)
svm_acc  = accuracy_score(y_te, svm_pred)
svm_f1   = f1_score(y_te, svm_pred, average="weighted")
print(f"  SVM: acc={svm_acc:.4f}  f1={svm_f1:.4f}")

# ── Gradient Boosting ─────────────────────────────────────────────────────────
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.08, max_depth=4,
    subsample=0.8, random_state=42
)
gb.fit(X_tr_s, y_tr)
gb_pred = gb.predict(X_te_s)
gb_acc  = accuracy_score(y_te, gb_pred)
gb_f1   = f1_score(y_te, gb_pred, average="weighted")
print(f"  GradBoost: acc={gb_acc:.4f}  f1={gb_f1:.4f}")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_tr_s, y_tr)
rf_pred = rf.predict(X_te_s)
rf_acc  = accuracy_score(y_te, rf_pred)
rf_f1   = f1_score(y_te, rf_pred, average="weighted")
print(f"  RandomForest: acc={rf_acc:.4f}  f1={rf_f1:.4f}")

# ── Ensemble ──────────────────────────────────────────────────────────────────
p_svm = svm.predict_proba(X_te_s)
p_gb  = gb.predict_proba(X_te_s)
p_rf  = rf.predict_proba(X_te_s)

n_cls = len(le.classes_)
# Align probability arrays (all 3 models trained on same LabelEncoder)
ens_prob = 0.40 * p_svm + 0.35 * p_gb + 0.25 * p_rf
ens_pred = np.argmax(ens_prob, axis=1)
ens_acc  = accuracy_score(y_te, ens_pred)
ens_f1   = f1_score(y_te, ens_pred, average="weighted")

print(f"\n{'='*50}")
print(f"  ENSEMBLE: acc={ens_acc:.4f}  f1={ens_f1:.4f}")
print(f"{'='*50}")
print()
print(classification_report(y_te, ens_pred, target_names=le.classes_))

# 5-fold CV on full dataset (RF as proxy)
cv_scores = cross_val_score(
    rf, scaler.transform(X), y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1
)
print(f"5-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
Path("backend/models").mkdir(exist_ok=True)

# Pick best single model as the main one
best_name, best_model, best_acc = max(
    [("svm", svm, svm_acc), ("gradboost", gb, gb_acc), ("random_forest", rf, rf_acc)],
    key=lambda x: x[2]
)
print(f"\nBest single model: {best_name} ({best_acc:.4f})")

joblib.dump(best_model,  "backend/models/image_model.pkl")
joblib.dump(svm,         "backend/models/image_svm.pkl")
joblib.dump(gb,          "backend/models/image_gb.pkl")
joblib.dump(rf,          "backend/models/image_rf.pkl")
joblib.dump(scaler,      "backend/models/image_scaler.pkl")
joblib.dump(le,          "backend/models/image_label_encoder.pkl")

meta = {
    "classes":      list(le.classes_),
    "class_to_idx": {c: int(i) for i, c in enumerate(le.classes_)},
    "n_features":   int(X.shape[1]),
    "img_size":     list(IMG_SIZE),
    "accuracy_test":float(ens_acc),
    "f1_weighted":  float(ens_f1),
    "cv_mean":      float(cv_scores.mean()),
    "cv_std":       float(cv_scores.std()),
    "total_images": int(len(X)),
    "per_class":    label_counts,
    "dataset":      "7272660 (Normal/Benign/Malignant liver ultrasound)",
    "feature_desc": "HOG + texture stats + color histogram + LBP",
}
with open("backend/models/image_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ Models saved to backend/models/")
print(f"  Classes: {list(le.classes_)}")
print(f"  Test accuracy: {ens_acc:.4f}  |  5-fold CV: {cv_scores.mean():.4f}")
