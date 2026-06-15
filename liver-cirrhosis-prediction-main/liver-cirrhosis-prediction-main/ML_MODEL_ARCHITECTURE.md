# Machine Learning Model Architecture
## Traditional ML Models for Symptom-Based Prediction

---

## 📊 Overview

This document details the machine learning models used for symptom-based liver cirrhosis prediction. The system employs an ensemble approach combining Random Forest, XGBoost, and SVM models for robust prediction.

---

## 🏗️ Model Architecture

### 1. Data Preprocessing Pipeline

```python
class SymptomDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)

    def preprocess(self, data):
        # Handle missing values
        data = self.imputer.fit_transform(data)

        # Scale features
        data = self.scaler.fit_transform(data)

        # Feature selection
        data = self.feature_selector.fit_transform(data, labels)

        return data
```

### 2. Random Forest Model

```python
class RandomForestPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        return self.model.feature_importances_
```

**Hyperparameters:**
- `n_estimators`: 500 trees
- `max_depth`: 15 levels
- `min_samples_split`: 5 samples
- `min_samples_leaf`: 2 samples
- `max_features`: sqrt(n_features)
- `bootstrap`: True

### 3. XGBoost Model

```python
class XGBoostPredictor:
    def __init__(self):
        self.model = XGBClassifier(
            objective='multi:softprob',
            num_class=4,  # stages 0-3
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        return self.model.feature_importances_
```

**Hyperparameters:**
- `n_estimators`: 300 trees
- `max_depth`: 8 levels
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `gamma`: 0.1 (regularization)
- `min_child_weight`: 1
- `reg_alpha`: 0.1 (L1 regularization)
- `reg_lambda`: 1.0 (L2 regularization)

### 4. Support Vector Machine Model

```python
class SVMPredictor:
    def __init__(self):
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            max_iter=10000
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

**Hyperparameters:**
- `kernel`: Radial Basis Function (RBF)
- `C`: 1.0 (regularization parameter)
- `gamma`: 'scale' (kernel coefficient)
- `probability`: True (enable probability estimates)

---

## 🔄 Ensemble Model

### Weighted Voting Ensemble

```python
class EnsemblePredictor:
    def __init__(self, weights=None):
        self.weights = weights or [0.4, 0.4, 0.2]  # RF, XGB, SVM
        self.models = {
            'rf': RandomForestPredictor(),
            'xgb': XGBoostPredictor(),
            'svm': SVMPredictor()
        }

    def train(self, X_train, y_train):
        for model in self.models.values():
            model.train(X_train, y_train)

    def predict_proba(self, X):
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

    def predict_stage(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_confidence_score(self, X):
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)
```

**Ensemble Weights:**
- Random Forest: 40%
- XGBoost: 40%
- SVM: 20%

---

## 📈 Feature Engineering

### Input Features (22 features)

**Demographic Features:**
- `age`: Patient age (years)
- `sex`: Gender (0=female, 1=male)

**Clinical Symptoms:**
- `fatigue_level`: Fatigue severity (1-10 scale)
- `alcohol_consumption`: Alcohol units per week
- `weight_loss_kg`: Weight loss in kg
- `abdominal_swelling`: Presence of ascites (0-1)
- `appetite_loss`: Loss of appetite (0-1)
- `jaundice`: Yellow skin/eyes (0-1)
- `fever`: Presence of fever (0-1)

**Physical Examination:**
- `ascites`: Ascites severity (0-3 scale)
- `hepatomegaly`: Liver enlargement (0-3 scale)
- `spiders`: Spider angiomas (0-3 scale)
- `edema`: Peripheral edema (0-3 scale)

**Laboratory Tests:**
- `bilirubin`: Serum bilirubin (mg/dL)
- `cholesterol`: Total cholesterol (mg/dL)
- `albumin`: Serum albumin (g/dL)
- `copper`: Serum copper (μg/dL)
- `alk_phos`: Alkaline phosphatase (IU/L)
- `ast`: Aspartate aminotransferase (IU/L)
- `triglycerides`: Serum triglycerides (mg/dL)
- `platelets`: Platelet count (×10^9/L)
- `prothrombin`: Prothrombin time (seconds)

### Feature Processing

```python
def create_features(df):
    # Create derived features
    df['bilirubin_albumin_ratio'] = df['bilirubin'] / (df['albumin'] + 1e-6)
    df['ast_alt_ratio'] = df['ast'] / (df['alt'] + 1e-6)  # if ALT available
    df['platelet_albumin_score'] = df['platelets'] * df['albumin']

    # Categorize continuous features
    df['bilirubin_category'] = pd.cut(df['bilirubin'],
                                      bins=[0, 1.2, 2.0, 5.0, np.inf],
                                      labels=['normal', 'mild', 'moderate', 'severe'])

    df['albumin_category'] = pd.cut(df['albumin'],
                                    bins=[0, 3.5, 2.8, 2.0, np.inf],
                                    labels=['severe', 'moderate', 'mild', 'normal'])

    return df
```

---

## 🎯 Model Evaluation

### Performance Metrics

```python
def evaluate_model(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_true, y_proba, multi_class='ovr')
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred)

    return metrics, cm, report
```

### Cross-Validation Strategy

```python
def cross_validate_model(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv,
                               scoring=['accuracy', 'precision_weighted',
                                       'recall_weighted', 'f1_weighted'])

    return {
        'accuracy_mean': cv_scores['test_accuracy'].mean(),
        'accuracy_std': cv_scores['test_accuracy'].std(),
        'precision_mean': cv_scores['test_precision_weighted'].mean(),
        'recall_mean': cv_scores['test_recall_weighted'].mean(),
        'f1_mean': cv_scores['test_f1_weighted'].mean()
    }
```

---

## 🔧 Hyperparameter Tuning

### Grid Search Configuration

```python
rf_param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}
```

### Bayesian Optimization

```python
from skopt import BayesSearchCV

def bayesian_tune(model, param_space, X, y):
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=50,
        cv=5,
        scoring='f1_weighted',
        random_state=42
    )

    opt.fit(X, y)
    return opt.best_params_, opt.best_score_
```

---

## 📊 Model Interpretability

### Feature Importance Analysis

```python
def analyze_feature_importance(models, feature_names):
    importance_dict = {}

    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importance_dict[name] = dict(zip(feature_names,
                                           model.feature_importances_))
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_dict[name] = dict(zip(feature_names,
                                           np.abs(model.coef_[0])))

    return importance_dict
```

### SHAP Integration

```python
import shap

def explain_prediction(model, X_sample, feature_names):
    if isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)

    shap_values = explainer(X_sample)
    feature_importance = dict(zip(feature_names,
                                 np.abs(shap_values.values).mean(axis=0)))

    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance,
        'base_value': explainer.expected_value
    }
```

---

## 💾 Model Persistence

### Save/Load Models

```python
import joblib

def save_model(model, filepath):
    """Save trained model to disk"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load trained model from disk"""
    return joblib.load(filepath)

def save_model_metadata(model_info, filepath):
    """Save model metadata (version, training date, performance)"""
    metadata = {
        'model_version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'performance_metrics': model_info.get('metrics', {}),
        'feature_names': model_info.get('feature_names', []),
        'hyperparameters': model_info.get('hyperparameters', {})
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## 🔄 Model Updates

### Online Learning Strategy

```python
class OnlineLearner:
    def __init__(self, base_model):
        self.base_model = base_model
        self.new_data_buffer = []
        self.update_threshold = 100  # Update after 100 new samples

    def add_feedback(self, X_new, y_true, y_pred):
        """Add new labeled data for model update"""
        self.new_data_buffer.append((X_new, y_true))

        if len(self.new_data_buffer) >= self.update_threshold:
            self.update_model()

    def update_model(self):
        """Update model with new data"""
        if not self.new_data_buffer:
            return

        # Combine old and new data
        X_old, y_old = self.base_model.training_data
        X_new = np.array([x for x, y in self.new_data_buffer])
        y_new = np.array([y for x, y in self.new_data_buffer])

        X_combined = np.vstack([X_old, X_new])
        y_combined = np.hstack([y_old, y_new])

        # Retrain model
        self.base_model.train(X_combined, y_combined)

        # Clear buffer
        self.new_data_buffer = []
```

---

## 📈 Performance Monitoring

### Model Drift Detection

```python
from scipy.stats import ks_2samp

def detect_drift(X_current, X_reference, threshold=0.05):
    """Detect data drift using Kolmogorov-Smirnov test"""
    drift_features = []

    for i, feature in enumerate(X_current.columns):
        stat, p_value = ks_2samp(X_current.iloc[:, i], X_reference.iloc[:, i])

        if p_value < threshold:
            drift_features.append({
                'feature': feature,
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': True
            })

    return drift_features
```

### Model Performance Tracking

```python
class ModelMonitor:
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'precision_drop': 0.05,
            'recall_drop': 0.05
        }

    def log_performance(self, metrics):
        """Log model performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

    def check_alerts(self):
        """Check if performance has degraded"""
        if len(self.performance_history) < 2:
            return []

        current = self.performance_history[-1]['metrics']
        previous = self.performance_history[-2]['metrics']

        alerts = []
        for metric, threshold in self.alert_thresholds.items():
            if metric in current and metric in previous:
                drop = previous[metric] - current[metric]
                if drop > threshold:
                    alerts.append({
                        'metric': metric,
                        'previous_value': previous[metric],
                        'current_value': current[metric],
                        'drop': drop,
                        'threshold': threshold
                    })

        return alerts
```

---

## 🚀 Production Deployment

### Model Serving

```python
class ModelService:
    def __init__(self, model_path, preprocessor_path):
        self.model = load_model(model_path)
        self.preprocessor = load_model(preprocessor_path)
        self.monitor = ModelMonitor()

    def predict(self, input_data):
        """Make prediction with monitoring"""
        start_time = time.time()

        try:
            # Preprocess input
            processed_data = self.preprocessor.transform(input_data)

            # Make prediction
            prediction = self.model.predict(processed_data)
            probability = self.model.predict_proba(processed_data)

            # Log performance
            inference_time = time.time() - start_time
            self.monitor.log_inference_time(inference_time)

            return {
                'prediction': prediction,
                'probability': probability,
                'inference_time': inference_time
            }

        except Exception as e:
            self.monitor.log_error(str(e))
            raise
```

---

**For implementation details, see** `ai_pipeline/train_ml.py` and `backend/models_handler.py`
