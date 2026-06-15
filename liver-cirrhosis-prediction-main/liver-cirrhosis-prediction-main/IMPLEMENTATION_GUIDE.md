# Liver Cirrhosis AI System - Complete Implementation Guide

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Installation & Setup](#installation--setup)
3. [AI Pipeline Components](#ai-pipeline-components)
4. [API Documentation](#api-documentation)
5. [Training Workflows](#training-workflows)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│              (Dashboard, Chat, Upload UI)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
    ┌─────▼─────┐         ┌────▼─────┐
    │  Nginx    │         │ Reverse  │
    │ Proxy     │         │ Proxy    │
    └─────┬─────┘         └────┬─────┘
          │                    │
┌─────────▼──────────────────────────────┐
│        FastAPI Backend (Port 8000)     │
├──────────────────────────────────────┤
│  Authentication & Authorization      │
│  Request Routing & Validation        │
│  Database Management                 │
└───────┬──────────────────────────────┘
        │
   ┌────┴──────────────────────────────┐
   │    AI System Integration Layer     │
   └────┬───────────────────────────────┘
        │
   ┌────┴────────────────────────────────────┐
   │                                          │
┌──▼───────────┐  ┌──────────┐  ┌──────────┐│
│ ML Engine    │  │DL Engine │  │XAI Engine││
│(XGBoost)     │  │(EfficientNet) (SHAP)   │
└─┬────────────┘  └────┬─────┘  └──┬───────┘│
  │                    │           │        │
┌─▼────────────────────▼───────────▼──┐     │
│      Fusion Engine (Hybrid)          │◄────┘
│   (Dynamic Weighting)                │
└──────┬───────────────────────────────┘
       │
    ┌──▼──────────────┐
    │ PostgreSQL DB   │
    │ Redis Cache     │
    │ File Storage    │
    └─────────────────┘
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- PostgreSQL 13+
- 8GB RAM (minimum)
- NVIDIA GPU (recommended for DL training)

### Local Development Setup

#### 1. Clone and Setup Environment

```bash
cd liver-cirrhosis-project
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter CUDA issues, install CPU version
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Set Environment Variables

```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings:
DATABASE_URL=postgresql://user:password@localhost:5432/liver_cirrhosis_db
JWT_SECRET_KEY=your-secret-key
OPENAI_API_KEY=sk-...
```

#### 4. Initialize Database

```bash
cd backend
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"
```

---

## AI Pipeline Components

### 1. ML Engine (Tabular Data - XGBoost)

**Location:** `ai_pipeline/ml_engine.py`

#### Features:
- XGBoost classifier for symptom-based prediction
- Handles class imbalance with SMOTE
- Automatic hyperparameter optimization
- Feature importance calculation

#### Training Script:

```python
from ai_pipeline.ml_engine import MLEngine

engine = MLEngine()

# Train on your dataset
metrics = engine.train(
    data_path='datasets/cirrhosis.csv',
    target_col='Status',
    test_size=0.2
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

#### Expected Dataset Format:

```csv
Age,Gender,Alcohol,Bilirubin,Albumin,AST,ALT,Platelets,...,Status
45,M,20,1.5,3.8,60,35,150000,...,Normal
52,F,35,2.1,3.2,120,80,120000,...,Stage 1
```

---

### 2. DL Engine (Imaging - EfficientNet)

**Location:** `ai_pipeline/dl_engine.py`

#### Features:
- EfficientNet-B0 transfer learning
- Multi-stage training (frozen base → fine-tuning)
- Data augmentation pipeline
- Grad-CAM visualization

#### Training Script:

```python
from ai_pipeline.dl_engine import DLEngine

engine = DLEngine()
engine.build_model()

# Train on medical images
result = engine.train(
    train_dir='data/train_images',
    val_dir='data/val_images',
    epochs=50,
    batch_size=32
)

print(f"Model saved: {result['model_path']}")
```

#### Dataset Directory Structure:

```
data/
├── train_images/
│   ├── F0/  # Normal (class 0)
│   ├── F1/  # Mild Fibrosis (class 1)
│   ├── F2/  # Moderate Fibrosis (class 2)
│   ├── F3/  # Severe Fibrosis (class 3)
│   └── F4/  # Cirrhosis (class 4)
└── val_images/
    ├── F0/
    ├── F1/
    ...
```

---

### 3. XAI Engine (Explainability)

**Location:** `ai_pipeline/xai_engine.py`

#### Features:
- SHAP for feature importance (tabular)
- Grad-CAM for imaging attention maps
- Clinical interpretation generation
- Recommendations based on predictions

#### Example Usage:

```python
from ai_pipeline.xai_engine import create_xai_engine
from ai_pipeline.ml_engine import MLEngine

xai = create_xai_engine()
ml_engine = MLEngine()
ml_engine.load()

# Generate SHAP explanation
explanation = xai.generate_shap_explanation(
    model=ml_engine.model,
    X_background=X_train,
    X_explain=X_test,
    feature_names=feature_list
)

print("Top Features:", explanation['feature_importance'])
```

---

### 4. Fusion Engine (Hybrid Predictions)

**Location:** `ai_pipeline/fusion_engine.py`

#### Fusion Strategies:
1. **Weighted Average** - Static weights for ML/DL
2. **Dynamic Weight** - Adjusts weights based on confidence
3. **Voting** - Majority voting with confidence weighting
4. **Stacking** - Meta-learner approach

#### Example:

```python
from ai_pipeline.fusion_engine import create_fusion_engine

fusion = create_fusion_engine('dynamic_weight')

# Combine predictions
ml_pred = {'stage': 2, 'confidence': 0.85, 'probabilities': [...]}
dl_pred = {'stage': 3, 'confidence': 0.90, 'probabilities': [...]}

fused = fusion.fuse_predictions(ml_pred, dl_pred)
print(f"Final Stage: {fused['stage_name']}")
print(f"Ensemble Confidence: {fused['ensemble_confidence']}")
```

---

### 5. Chatbot with RAG

**Location:** `ai_pipeline/chatbot.py`

#### Features:
- Retrieval-Augmented Generation (RAG)
- Medical knowledge base
- Support for OpenAI, Groq, and local LLMs
- Conversation history tracking

#### Usage:

```python
from ai_pipeline.chatbot import create_chatbot

# Using OpenAI
chatbot = create_chatbot(api_key='sk-...', provider='openai')

# Using local provider (no API key needed)
chatbot = create_chatbot(provider='local')

# Ask medical questions
response = chatbot.chat("What is Stage 3 cirrhosis?")
print(response['response'])
```

---

## API Documentation

### Authentication

All protected endpoints require JWT token in header:

```
Authorization: Bearer <token>
```

### Core Endpoints

#### 1. **Symptom Prediction**

```bash
POST /ai/predict/symptoms
Content-Type: application/json

{
  "age": 45,
  "gender": "M",
  "alcohol_consumption": 25,
  "bilirubin": 1.8,
  "albumin": 3.5,
  "ast": 75,
  "alt": 60,
  "platelets": 150000,
  "fatigue": true,
  "jaundice": false,
  "ascites": 0,
  ...
}
```

**Response:**

```json
{
  "prediction": {
    "stage": 2,
    "confidence": 0.87,
    "probabilities": [0.05, 0.15, 0.62, 0.18, 0.0],
    "risk_level": "medium"
  },
  "timestamp": "2024-06-10T10:30:00Z"
}
```

#### 2. **Image Prediction**

```bash
POST /ai/predict/image
Content-Type: multipart/form-data

[Binary JPEG/PNG image data]
```

**Response:**

```json
{
  "prediction": {
    "stage": 3,
    "confidence": 0.91,
    "probabilities": [0.02, 0.05, 0.15, 0.78, 0.0]
  },
  "image_name": "ultrasound_scan.jpg"
}
```

#### 3. **Hybrid Prediction (Symptoms + Image)**

```bash
POST /ai/predict/hybrid
Content-Type: multipart/form-data

symptoms={JSON object}
image={binary image file}
```

**Response:** (Includes XAI explanations)

```json
{
  "diagnosis_id": 123,
  "prediction": {...},
  "xai_report": {
    "tabular_analysis": {
      "top_features": [
        {"feature": "platelets", "importance": 0.35},
        {"feature": "bilirubin", "importance": 0.28}
      ]
    },
    "imaging_analysis": {
      "heatmap_base64": "iVBORw0KGgo...",
      "attention_regions": {...}
    }
  },
  "report_url": "/reports/123"
}
```

#### 4. **Medical Chatbot**

```bash
POST /ai/chat/medical
Content-Type: application/json

{
  "message": "What are the symptoms of cirrhosis?"
}
```

**Response:**

```json
{
  "session_id": 456,
  "message": "Cirrhosis symptoms include jaundice, abdominal swelling, fatigue...",
  "sources": ["Medical knowledge base"],
  "confidence": "high"
}
```

---

## Training Workflows

### Complete Training Pipeline

```python
from ai_pipeline.ml_engine import MLEngine
from ai_pipeline.dl_engine import DLEngine
from ai_pipeline.preprocessing import create_preprocessors

# Setup
preprocessors = create_preprocessors()
ml_engine = MLEngine()
dl_engine = DLEngine()

# 1. Preprocess tabular data
df = pd.read_csv('datasets/cirrhosis.csv')
df_clean = preprocessors['tabular'].preprocess(df)

# 2. Train ML model
print("Training ML model...")
ml_metrics = ml_engine.train('datasets/cirrhosis_clean.csv')

# 3. Preprocess images
print("Preprocessing images...")
train_images = preprocessors['image'].preprocess_batch(image_paths)

# 4. Train DL model
print("Training DL model...")
dl_engine.build_model()
dl_result = dl_engine.train(
    train_dir='data/train_images',
    val_dir='data/val_images',
    epochs=50
)

# 5. Validate fusion
from ai_pipeline.fusion_engine import create_fusion_engine
fusion = create_fusion_engine()

print(f"ML Accuracy: {ml_metrics['accuracy']:.4f}")
print(f"DL Results: {dl_result}")
print("Training complete!")
```

---

## Deployment

### Docker Deployment (Recommended)

#### 1. Build and Start Containers

```bash
# Set environment variables
export OPENAI_API_KEY=sk-...
export DATABASE_PASSWORD=secure_password

# Build images
docker-compose build

# Start services
docker-compose up -d
```

#### 2. Verify Deployment

```bash
# Check container health
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs frontend

# Test API
curl http://localhost/health
curl http://localhost/ai/health
```

#### 3. Access Services

- **API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

### Cloud Deployment (AWS)

#### 1. Prepare EC2 Instance

```bash
# SSH into instance
ssh -i key.pem ubuntu@instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repository
git clone <repo-url>
cd liver-cirrhosis-project
```

#### 2. Deploy with Docker Compose

```bash
# Create .env file with secrets
echo "DATABASE_PASSWORD=secure_pass" > .env
echo "JWT_SECRET_KEY=random_secret" >> .env

# Start services
docker-compose up -d

# Setup SSL with Let's Encrypt
docker-compose exec nginx certbot certonly --standalone -d yourdomain.com
```

#### 3. Scale Application (Optional)

```bash
# Update docker-compose.yml to add replicas
# Then recreate services
docker-compose up -d --scale backend=3
```

---

## Troubleshooting

### Common Issues

#### 1. **"ML Engine not available"**

```bash
# Verify model files exist
ls models/ml_model.pkl
ls models/ml_scaler.pkl

# Retrain if missing
python -c "from ai_pipeline.ml_engine import MLEngine; MLEngine().train('datasets/cirrhosis.csv')"
```

#### 2. **"DL Engine not available"**

```bash
# Check TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"

# Rebuild model
python -c "from ai_pipeline.dl_engine import DLEngine; DLEngine().build_model()"
```

#### 3. **Database Connection Issues**

```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Verify connection string in .env
psql postgresql://user:password@localhost:5432/liver_cirrhosis_db

# Reinitialize database
python backend/database.py
```

#### 4. **Memory Issues During Training**

```python
# Reduce batch size
dl_engine.train(..., batch_size=8)  # Instead of 32

# Use CPU for testing
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### 5. **GPU Not Detected**

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install PyTorch with CUDA
pip install torch::cuda118  # Replace 118 with your CUDA version
```

---

## Performance Optimization

### Model Inference Speed

```python
# Use batch prediction
predictions = dl_engine.predict_batch(images)

# Use model quantization (for deployment)
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('models/dl_model')
tflite_model = converter.convert()
```

### Database Optimization

```python
# Add indexes
from sqlalchemy import Index
Index('idx_user_diagnosis', DiagnosisRecord.user_id, DiagnosisRecord.created_at)

# Use connection pooling
from sqlalchemy.pool import QueuePool
engine = create_engine(db_url, poolclass=QueuePool, pool_size=20, max_overflow=40)
```

### Caching

```python
# Redis caching for predictions
from redis import Redis
cache = Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Cache predictions
key = f"prediction:{user_id}:{symptom_hash}"
cached = cache.get(key)
if cached:
    return json.loads(cached)
```

---

## Monitoring & Logging

### Enable Detailed Logging

```python
# In backend/main.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Monitor Metrics

```bash
# View API performance
docker-compose exec backend tail -f logs/app.log

# Monitor resource usage
docker stats
```

---

## References

- **IEEE Standards for Medical AI**: IEEE 42010 (Architecture Description)
- **DICOM Standard**: For medical imaging
- **HIPAA Compliance**: For data privacy
- **FDA Guidance**: For medical device software

---

**Version:** 1.0  
**Last Updated:** June 2024  
**Maintained By:** AI Team
