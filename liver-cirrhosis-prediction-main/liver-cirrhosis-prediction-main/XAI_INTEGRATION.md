# Explainable AI Integration
## SHAP and Grad-CAM Implementation

---

## 📊 Overview

This document details the Explainable AI (XAI) implementation for both traditional ML models (SHAP) and deep learning models (Grad-CAM) in the liver cirrhosis detection system.

---

## 🔍 SHAP for Traditional ML Models

### 1. SHAP Implementation

```python
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class SHAPExplainer:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type"""
        if isinstance(self.model, xgb.XGBClassifier):
            return shap.TreeExplainer(self.model)
        elif isinstance(self.model, RandomForestClassifier):
            return shap.TreeExplainer(self.model)
        else:
            # For other models, use KernelExplainer (slower but general)
            background = shap.sample(self.X_train, 100)  # Background dataset
            return shap.KernelExplainer(self.model.predict_proba, background)

    def explain_prediction(self, X_sample):
        """Explain a single prediction"""
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)

        # For multi-class, shap_values is a list of arrays
        if isinstance(shap_values, list):
            # Take SHAP values for predicted class
            predicted_class = np.argmax(self.model.predict_proba(X_sample)[0])
            shap_values_single = shap_values[predicted_class]
        else:
            shap_values_single = shap_values

        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = {
                'shap_value': float(shap_values_single[0][i]),
                'feature_value': float(X_sample.iloc[0, i]),
                'importance': abs(float(shap_values_single[0][i]))
            }

        # Sort by importance
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: x[1]['importance'], reverse=True)

        return {
            'feature_importance': dict(sorted_features),
            'expected_value': float(self.explainer.expected_value),
            'shap_values': shap_values_single.tolist()
        }

    def explain_global(self, X_test, max_evals=1000):
        """Global feature importance explanation"""
        if hasattr(self.explainer, 'shap_values'):
            # For tree models, use all training data
            shap_values = self.explainer.shap_values(self.X_train[:max_evals])
        else:
            # For kernel explainer, use sample
            X_sample = shap.sample(self.X_train, min(max_evals, len(self.X_train)))
            shap_values = self.explainer.shap_values(X_sample)

        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            # Multi-class case
            global_importance = {}
            for class_idx in range(len(shap_values)):
                class_importance = np.abs(shap_values[class_idx]).mean(axis=0)
                for i, feature in enumerate(self.feature_names):
                    key = f"{feature}_class_{class_idx}"
                    global_importance[key] = float(class_importance[i])
        else:
            # Binary or regression case
            global_importance = np.abs(shap_values).mean(axis=0)
            global_importance = dict(zip(self.feature_names, global_importance))

        return global_importance
```

### 2. SHAP Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

class SHAPVisualizer:
    def __init__(self, explainer):
        self.explainer = explainer

    def plot_waterfall(self, X_sample, prediction_id=None):
        """Create waterfall plot for single prediction"""
        explanation = self.explainer.explain_prediction(X_sample)

        # Prepare data for waterfall plot
        features = []
        values = []
        cumulative = explanation['expected_value']

        # Sort features by absolute SHAP value
        sorted_features = sorted(explanation['feature_importance'].items(),
                               key=lambda x: abs(x[1]['shap_value']), reverse=True)

        for feature, data in sorted_features[:10]:  # Top 10 features
            features.append(feature)
            values.append(data['shap_value'])
            cumulative += data['shap_value']

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Starting point
        ax.barh(['Base'], [explanation['expected_value']],
               color='lightblue', label='Base Value')

        # Feature contributions
        bottom = explanation['expected_value']
        for i, (feature, value) in enumerate(zip(features, values)):
            color = 'red' if value > 0 else 'blue'
            ax.barh([feature], [value], left=[bottom], color=color)
            bottom += value

        ax.set_xlabel('SHAP Value Contribution')
        ax.set_title(f'Prediction Explanation (ID: {prediction_id})')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_summary(self, X_test, max_samples=1000):
        """Create summary plot for global feature importance"""
        X_sample = X_test.sample(min(max_samples, len(X_test)))

        if hasattr(self.explainer.explainer, 'shap_values'):
            shap_values = self.explainer.explainer.shap_values(X_sample)
        else:
            shap_values = self.explainer.explainer.shap_values(X_sample)

        # Create summary plot
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            # Multi-class case - show for first class
            shap.summary_plot(shap_values[0], X_sample,
                            feature_names=self.explainer.feature_names,
                            show=False)
        else:
            shap.summary_plot(shap_values, X_sample,
                            feature_names=self.explainer.feature_names,
                            show=False)

        plt.title('Global Feature Importance (SHAP Summary)')
        plt.tight_layout()
        return plt.gcf()

    def plot_dependence(self, feature_name, X_test, interaction_feature=None):
        """Create dependence plot for a specific feature"""
        if hasattr(self.explainer.explainer, 'shap_values'):
            shap_values = self.explainer.explainer.shap_values(X_test)
        else:
            shap_values = self.explainer.explainer.shap_values(X_test)

        plt.figure(figsize=(8, 6))
        if isinstance(shap_values, list):
            # Multi-class case
            shap.dependence_plot(feature_name, shap_values[0], X_test,
                               interaction_index=interaction_feature,
                               feature_names=self.explainer.feature_names,
                               show=False)
        else:
            shap.dependence_plot(feature_name, shap_values, X_test,
                               interaction_index=interaction_feature,
                               feature_names=self.explainer.feature_names,
                               show=False)

        plt.title(f'SHAP Dependence Plot: {feature_name}')
        plt.tight_layout()
        return plt.gcf()
```

---

## 🎨 Grad-CAM for Deep Learning Models

### 1. Grad-CAM Implementation

```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        """Save feature maps from forward pass"""
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        """Save gradients from backward pass"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class):
        """Generate Class Activation Map"""
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get weights from gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) - torch.min(cam) + 1e-8)

        return cam.squeeze().cpu().numpy()

    def generate_guided_gradcam(self, input_tensor, target_class):
        """Generate Guided Grad-CAM (Grad-CAM + Guided Backpropagation)"""
        # Generate Grad-CAM
        cam = self.generate_cam(input_tensor, target_class)

        # Generate guided backpropagation
        guided_grads = self.guided_backprop(input_tensor, target_class)

        # Combine Grad-CAM with guided gradients
        guided_cam = guided_grads * cam[..., np.newaxis]

        return guided_cam, cam

    def guided_backprop(self, input_tensor, target_class):
        """Perform guided backpropagation"""
        def guided_hook(module, grad_input, grad_output):
            # Only keep positive gradients
            if grad_input[0] is not None:
                grad_input = (torch.clamp(grad_input[0], min=0.0),) + grad_input[1:]

            return grad_input

        # Register hooks for guided backprop
        hooks = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_backward_hook(guided_hook))

        # Forward and backward pass
        self.model.zero_grad()
        output = self.model(input_tensor)

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)

        # Get gradients
        guided_grads = input_tensor.grad.data.cpu().numpy()[0]

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return guided_grads
```

### 2. Advanced Grad-CAM Variants

```python
class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation for better localization"""

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ weights calculation
        grads = self.gradients  # [batch, channels, height, width]
        features = self.feature_maps  # [batch, channels, height, width]

        # First derivative
        first_derivative = grads ** 2

        # Second derivative
        second_derivative = grads ** 3 * features

        # Global average pool
        global_avg_first = torch.mean(first_derivative, dim=[2, 3], keepdim=True)
        global_avg_second = torch.mean(second_derivative, dim=[2, 3], keepdim=True)

        # Compute weights
        numerator = grads ** 2
        denominator = 2 * grads ** 2 + global_avg_first * features * 2 + global_avg_second
        weights = numerator / (denominator + 1e-8)

        # Weighted combination
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) - torch.min(cam) + 1e-8)

        return cam.squeeze().cpu().numpy()

class AblationCAM(GradCAM):
    """Ablation-CAM for more precise localization"""

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(input_tensor)
            original_score = original_output[0][target_class]

        # Generate CAM by ablating feature maps
        num_channels = self.feature_maps.shape[1]
        cam = torch.zeros(self.feature_maps.shape[2:])

        for channel in range(num_channels):
            # Ablate channel (set to zero)
            ablated_features = self.feature_maps.clone()
            ablated_features[0, channel, :, :] = 0

            # Forward pass with ablated features
            with torch.no_grad():
                ablated_output = self.model.classifier(
                    self.model.encoder(ablated_features).view(ablated_features.size(0), -1)
                )
                ablated_score = ablated_output[0][target_class]

            # Calculate importance
            importance = original_score - ablated_score
            cam += importance * torch.mean(ablated_features[0, channel, :, :], dim=[0, 1])

        # Normalize
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) - torch.min(cam) + 1e-8)

        return cam.cpu().numpy()
```

---

## 🎯 XAI Integration Service

### 1. Unified XAI Service

```python
from typing import Dict, Any, Optional
import json
from pathlib import Path

class XAIService:
    def __init__(self, ml_model, dl_model, feature_names):
        self.ml_explainer = SHAPExplainer(ml_model, None, feature_names)
        self.dl_explainer = None  # Will be set when DL model is loaded
        self.visualizer = SHAPVisualizer(self.ml_explainer)

    def set_dl_model(self, dl_model, target_layer):
        """Set deep learning model for Grad-CAM"""
        self.dl_explainer = GradCAM(dl_model, target_layer)

    def explain_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanation for a prediction"""
        prediction_type = prediction_data.get('prediction_type', 'symptom')

        explanation = {
            'prediction_id': prediction_data.get('prediction_id'),
            'prediction_type': prediction_type,
            'timestamp': datetime.now().isoformat(),
            'explanations': {}
        }

        if prediction_type in ['symptom', 'hybrid']:
            # SHAP explanation for symptoms
            symptom_data = pd.DataFrame([prediction_data['symptom_features']])
            ml_explanation = self.ml_explainer.explain_prediction(symptom_data)

            explanation['explanations']['symptom_shap'] = {
                'method': 'shap',
                'feature_importance': ml_explanation['feature_importance'],
                'expected_value': ml_explanation['expected_value'],
                'top_contributing_features': self._get_top_features(ml_explanation, 5)
            }

        if prediction_type in ['imaging', 'hybrid']:
            # Grad-CAM explanation for images
            if self.dl_explainer:
                image_tensor = self._preprocess_image(prediction_data['image_path'])
                predicted_class = prediction_data['predicted_class']

                cam = self.dl_explainer.generate_cam(image_tensor, predicted_class)
                heatmap_path = self._save_heatmap(cam, prediction_data['image_path'])

                explanation['explanations']['image_gradcam'] = {
                    'method': 'gradcam',
                    'heatmap_path': heatmap_path,
                    'attention_regions': self._analyze_attention_regions(cam),
                    'confidence_score': prediction_data.get('confidence', 0.0)
                }

        # Generate narrative explanation
        explanation['narrative'] = self._generate_narrative(explanation)

        return explanation

    def _get_top_features(self, explanation, top_n=5):
        """Get top contributing features"""
        features = explanation['feature_importance']
        sorted_features = sorted(features.items(),
                               key=lambda x: abs(x[1]['importance']), reverse=True)

        top_features = []
        for feature, data in sorted_features[:top_n]:
            top_features.append({
                'feature': feature,
                'importance': data['importance'],
                'value': data['feature_value'],
                'direction': 'increases_risk' if data['shap_value'] > 0 else 'decreases_risk'
            })

        return top_features

    def _preprocess_image(self, image_path):
        """Preprocess image for DL model"""
        # Implementation depends on your DL model preprocessing
        pass

    def _save_heatmap(self, cam, original_image_path):
        """Save Grad-CAM heatmap overlay"""
        # Implementation for saving heatmap
        pass

    def _analyze_attention_regions(self, cam):
        """Analyze which regions the model focused on"""
        # Implementation for region analysis
        pass

    def _generate_narrative(self, explanation):
        """Generate human-readable explanation"""
        narrative = "Based on the analysis:\n\n"

        if 'symptom_shap' in explanation['explanations']:
            shap_data = explanation['explanations']['symptom_shap']
            narrative += "Key symptom factors:\n"
            for feature in shap_data['top_contributing_features'][:3]:
                direction = "increased" if feature['direction'] == 'increases_risk' else "decreased"
                narrative += f"- {feature['feature']} ({feature['value']}) {direction} the risk assessment\n"

        if 'image_gradcam' in explanation['explanations']:
            gradcam_data = explanation['explanations']['image_gradcam']
            narrative += f"\nThe AI focused on specific regions in the medical image, "
            narrative += f"with {len(gradcam_data['attention_regions'])} key areas identified.\n"

        return narrative
```

### 2. XAI Report Generation

```python
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

class XAIReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()

    def generate_xai_report(self, explanation_data, output_path):
        """Generate comprehensive XAI report"""
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # Title
        title = Paragraph("Explainable AI Report", self.styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Prediction Summary
        pred_id = explanation_data.get('prediction_id', 'N/A')
        pred_type = explanation_data.get('prediction_type', 'N/A')

        summary = Paragraph(f"Prediction ID: {pred_id}<br/>Type: {pred_type}",
                          self.styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 12))

        # SHAP Explanation
        if 'symptom_shap' in explanation_data.get('explanations', {}):
            story.append(Paragraph("Symptom-Based Analysis (SHAP)", self.styles['Heading2']))
            shap_data = explanation_data['explanations']['symptom_shap']

            for feature in shap_data['top_contributing_features'][:5]:
                feature_text = f"{feature['feature']}: {feature['value']:.2f} "
                feature_text += f"(Impact: {feature['importance']:.3f})"
                story.append(Paragraph(feature_text, self.styles['Normal']))

            story.append(Spacer(1, 12))

        # Grad-CAM Explanation
        if 'image_gradcam' in explanation_data.get('explanations', {}):
            story.append(Paragraph("Medical Image Analysis (Grad-CAM)", self.styles['Heading2']))
            gradcam_data = explanation_data['explanations']['image_gradcam']

            gradcam_text = f"AI Confidence: {gradcam_data['confidence_score']:.2%}<br/>"
            gradcam_text += f"Attention Regions: {len(gradcam_data['attention_regions'])}"
            story.append(Paragraph(gradcam_text, self.styles['Normal']))

            # Add heatmap image if available
            if 'heatmap_path' in gradcam_data:
                try:
                    img = Image(gradcam_data['heatmap_path'], width=400, height=300)
                    story.append(img)
                except:
                    pass

            story.append(Spacer(1, 12))

        # Narrative Explanation
        if 'narrative' in explanation_data:
            story.append(Paragraph("Clinical Interpretation", self.styles['Heading2']))
            narrative = explanation_data['narrative'].replace('\n', '<br/>')
            story.append(Paragraph(narrative, self.styles['Normal']))

        # Build PDF
        doc.build(story)

        return output_path
```

---

## 📊 XAI Dashboard Components

### 1. Feature Importance Chart

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_feature_importance_chart(explanation_data):
    """Create interactive feature importance visualization"""
    if 'symptom_shap' not in explanation_data.get('explanations', {}):
        return None

    shap_data = explanation_data['explanations']['symptom_shap']
    features = shap_data['top_contributing_features']

    # Prepare data
    feature_names = [f['feature'] for f in features]
    importances = [f['importance'] for f in features]
    directions = ['red' if f['direction'] == 'increases_risk' else 'blue' for f in features]

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=feature_names,
        x=importances,
        orientation='h',
        marker_color=directions,
        name='Feature Importance'
    ))

    fig.update_layout(
        title="Key Factors Influencing Prediction",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )

    return fig
```

### 2. Grad-CAM Heatmap Display

```python
def create_gradcam_visualization(image_path, heatmap_path):
    """Create overlay visualization for Grad-CAM"""
    # Load original image and heatmap
    original = cv2.imread(image_path)
    heatmap = cv2.imread(heatmap_path)

    # Resize heatmap to match original
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

    # Create overlay
    overlay = cv2.addWeighted(original, 0.7, heatmap, 0.3, 0)

    # Convert to base64 for web display
    _, buffer = cv2.imencode('.png', overlay)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"
```

---

## 🔧 XAI Configuration

### 1. Configuration Settings

```python
class XAIConfig:
    def __init__(self):
        # SHAP settings
        self.shap_max_evals = 1000
        self.shap_background_samples = 100

        # Grad-CAM settings
        self.gradcam_target_layer = 'layer4'  # For ResNet
        self.gradcam_transparency = 0.4

        # Report settings
        self.report_format = 'pdf'  # pdf, json, html
        self.include_visualizations = True
        self.max_features_display = 10

        # Performance settings
        self.enable_caching = True
        self.cache_expiry_hours = 24
```

### 2. Performance Optimization

```python
import functools
import hashlib
import json
from datetime import datetime, timedelta

class XAICache:
    def __init__(self, expiry_hours=24):
        self.cache = {}
        self.expiry_hours = expiry_hours

    def _get_cache_key(self, data):
        """Generate cache key from input data"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def get(self, key):
        """Get cached result if not expired"""
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now() - cached_item['timestamp'] < timedelta(hours=self.expiry_hours):
                return cached_item['result']
            else:
                del self.cache[key]
        return None

    def set(self, key, result):
        """Cache result with timestamp"""
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now()
        }

def cached_explanation(func):
    """Decorator for caching XAI explanations"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'cache'):
            cache_key = self.cache._get_cache_key(args[0] if args else kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        result = func(self, *args, **kwargs)

        if hasattr(self, 'cache'):
            self.cache.set(cache_key, result)

        return result

    return wrapper
```

---

## 📈 XAI Evaluation Metrics

### 1. Explanation Quality Metrics

```python
def evaluate_explanation_quality(true_importance, predicted_importance, k=5):
    """Evaluate how well explanations match ground truth"""
    # Get top-k features for both
    true_top_k = set(sorted(true_importance.items(), key=lambda x: x[1], reverse=True)[:k])
    pred_top_k = set(sorted(predicted_importance.items(), key=lambda x: x[1], reverse=True)[:k])

    # Calculate overlap
    overlap = len(true_top_k.intersection(pred_top_k))
    precision = overlap / k if k > 0 else 0
    recall = overlap / len(true_top_k) if len(true_top_k) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'overlap': overlap
    }

def evaluate_gradcam_localization(ground_truth_masks, predicted_cams):
    """Evaluate Grad-CAM localization accuracy"""
    iou_scores = []

    for gt_mask, pred_cam in zip(ground_truth_masks, predicted_cams):
        # Threshold prediction
        pred_mask = (pred_cam > np.percentile(pred_cam, 80)).astype(np.uint8)

        # Calculate IoU
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)

    return {
        'mean_iou': np.mean(iou_scores),
        'median_iou': np.median(iou_scores),
        'iou_std': np.std(iou_scores)
    }
```

---

## 🚀 Production Deployment

### 1. XAI Service API

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Explainable AI Service")

class ExplanationRequest(BaseModel):
    prediction_id: str
    prediction_type: str
    symptom_features: Optional[Dict[str, float]] = None
    image_path: Optional[str] = None
    predicted_class: Optional[int] = None

@app.post("/api/v1/xai/explain")
async def explain_prediction(request: ExplanationRequest):
    """Generate XAI explanation for prediction"""
    try:
        xai_service = XAIService.get_instance()
        explanation = xai_service.explain_prediction(request.dict())

        return {
            "status": "success",
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/api/v1/xai/report/{prediction_id}")
async def get_xai_report(prediction_id: str):
    """Get XAI report for prediction"""
    try:
        report_generator = XAIReportGenerator()
        report_path = report_generator.generate_xai_report(
            {"prediction_id": prediction_id},  # Load from database
            f"/tmp/xai_report_{prediction_id}.pdf"
        )

        return FileResponse(report_path, media_type='application/pdf',
                          filename=f'xai_report_{prediction_id}.pdf')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
```

---

**For implementation details, see** `backend/utils/xai_logic.py` and `backend/services/xai_service.py`
