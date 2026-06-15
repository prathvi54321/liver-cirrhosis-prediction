# 🚀 Quick Start Guide

## Installation (5 minutes)

### Option 1: Local Development (Recommended for Development)

```bash
# 1. Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your database details

# 4. Initialize database
cd backend
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"

# 5. Start backend
uvicorn main:app --reload

# 6. In another terminal, start frontend
cd frontend
npm install
npm start

# Access: http://localhost:3000
```

### Option 2: Docker (Recommended for Production)

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for services to start (30 seconds)
docker-compose ps

# 3. Access services:
# - Frontend: http://localhost:3000
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

---

## Testing the System

### 1. Create Test User

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "secure123",
    "full_name": "Test User",
    "role": "patient"
  }'

# Login and get token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "secure123"
  }' | jq -r '.access_token')
```

### 2. Test Symptom Prediction

```bash
curl -X POST http://localhost:8000/ai/predict/symptoms \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "M",
    "alcohol_consumption": 25,
    "bilirubin": 1.8,
    "albumin": 3.5,
    "ast": 75,
    "alt": 60,
    "platelets": 150000,
    "prothrombin": 13,
    "creatinine": 1.0,
    "ascites": 0,
    "hepatomegaly": 0,
    "spiders": 0,
    "edema": 0,
    "encephalopathy": 0,
    "fatigue": 1,
    "jaundice": 0,
    "abdominal_pain": 0,
    "weight_loss": 0
  }'
```

### 3. Test Image Prediction

```bash
curl -X POST http://localhost:8000/ai/predict/image \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@path/to/liver_scan.jpg"
```

### 4. Test Medical Chatbot

```bash
curl -X POST http://localhost:8000/ai/chat/medical \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the symptoms of cirrhosis?"
  }'
```

---

## Training Models

### Train ML Model (XGBoost)

```python
from ai_pipeline.ml_engine import MLEngine

engine = MLEngine()
metrics = engine.train('datasets/cirrhosis.csv')

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

### Train DL Model (EfficientNet)

```python
from ai_pipeline.dl_engine import DLEngine

engine = DLEngine()
engine.build_model()

result = engine.train(
    train_dir='data/train_images',
    val_dir='data/val_images',
    epochs=50
)

print(f"Model saved: {result['model_path']}")
```

---

## Project Structure

```
liver-cirrhosis-project/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application file
│   ├── database.py         # Database models
│   ├── schemas.py          # Pydantic schemas
│   ├── auth.py             # Authentication logic
│   ├── models_handler.py   # Demo model handler
│   └── utils/              # Utility functions
│
├── frontend/               # React frontend
│   ├── src/
│   │   ├── pages/          # Page components
│   │   ├── components/     # Reusable components
│   │   ├── styles/         # CSS files
│   │   └── api.js          # API client
│   └── public/             # Static files
│
├── ai_pipeline/            # AI/ML components
│   ├── ml_engine.py        # XGBoost training & prediction
│   ├── dl_engine.py        # EfficientNet implementation
│   ├── xai_engine.py       # SHAP & Grad-CAM
│   ├── fusion_engine.py    # Hybrid prediction fusion
│   ├── chatbot.py          # RAG-based medical chatbot
│   ├── preprocessing.py    # Data preprocessing
│   ├── integration.py      # System integration
│   ├── init_models.py      # Model initialization
│   ├── train_ml.py         # ML training script
│   └── train_dl.py         # DL training script
│
├── datasets/               # Data folder (create manually)
│   ├── cirrhosis.csv       # Tabular training data
│   └── train_images/       # Medical images
│
├── models/                 # Trained model files (auto-created)
│   ├── ml_model.pkl        # Trained XGBoost
│   ├── dl_model.h5         # Trained EfficientNet
│   └── metadata/           # Model metadata
│
├── reports/                # Generated reports (auto-created)
├── uploads/                # User uploads (auto-created)
├── logs/                   # Application logs (auto-created)
│
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Backend container
├── nginx.conf              # Nginx configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── IMPLEMENTATION_GUIDE.md # Detailed documentation
```

---

## Key Features Implemented

### ✅ ML Engine
- [x] XGBoost training with 20+ features
- [x] Class imbalance handling (SMOTE)
- [x] Cross-validation and hyperparameter tuning
- [x] Feature importance extraction
- [x] Model persistence

### ✅ DL Engine
- [x] EfficientNet-B0 architecture
- [x] Transfer learning from ImageNet
- [x] Multi-stage training (frozen → fine-tuning)
- [x] Data augmentation pipeline
- [x] Batch prediction support

### ✅ XAI Components
- [x] SHAP explanations for ML (feature importance)
- [x] Grad-CAM heatmaps for DL (attention maps)
- [x] Clinical interpretation generator
- [x] Personalized recommendations

### ✅ Fusion Engine
- [x] Weighted average fusion
- [x] Dynamic weighting based on confidence
- [x] Voting strategy
- [x] Stacking with meta-learner
- [x] Agreement-based confidence boosting

### ✅ Medical Chatbot
- [x] RAG (Retrieval-Augmented Generation)
- [x] Medical knowledge base
- [x] OpenAI integration (optional)
- [x] Local fallback model
- [x] Conversation history

### ✅ Backend API
- [x] JWT authentication
- [x] Symptom prediction endpoint
- [x] Image prediction endpoint
- [x] Hybrid prediction endpoint
- [x] Medical chatbot endpoint
- [x] XAI report generation
- [x] Diagnosis history
- [x] PDF report generation
- [x] User management

### ✅ Data Preprocessing
- [x] Tabular data preprocessing
- [x] Image preprocessing & enhancement
- [x] CLAHE contrast enhancement
- [x] Data augmentation (rotation, flip, brightness)
- [x] Missing value handling
- [x] Outlier detection & removal

### ✅ Deployment
- [x] Docker containerization
- [x] Docker Compose orchestration
- [x] Nginx reverse proxy
- [x] PostgreSQL integration
- [x] Redis caching support
- [x] Environment configuration

---

## API Reference

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login & get JWT token

### Predictions
- `POST /ai/predict/symptoms` - Predict from symptoms (ML)
- `POST /ai/predict/image` - Predict from image (DL)
- `POST /ai/predict/hybrid` - Combined prediction + XAI

### Medical Assistance
- `POST /ai/chat/medical` - Chat with medical AI

### Reports & History
- `GET /ai/diagnosis/{id}/xai` - Get XAI report
- `GET /ai/diagnosis-history` - Get prediction history
- `GET /reports/{id}` - Download PDF report

### Health
- `GET /health` - System health check
- `GET /ai/health` - AI components status

---

## Performance Metrics

### Expected Accuracy
- **ML Model (XGBoost)**: 90-95% (tabular data)
- **DL Model (EfficientNet)**: 85-92% (imaging)
- **Ensemble (Fused)**: 92-96% (combined)

### Inference Speed
- **Symptom Prediction**: <50ms
- **Image Prediction**: 200-400ms (GPU), 500-1000ms (CPU)
- **Hybrid Prediction**: 500-800ms (GPU)

### System Resources
- **Backend Memory**: 2-4 GB
- **Model Storage**: 500-800 MB
- **Database**: 100-500 MB

---

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt --upgrade
```

### Port already in use
```bash
# Change port in .env or docker-compose.yml
# Or kill the process using the port
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

### Database connection error
```bash
# Check PostgreSQL is running
docker-compose ps database

# Reset database
docker-compose down -v
docker-compose up database
```

### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Use CPU instead
export CUDA_VISIBLE_DEVICES=-1
```

---

## Next Steps

1. **Prepare Data**: Collect liver cirrhosis dataset (symptoms + images)
2. **Train Models**: Run training scripts with your data
3. **Fine-tune**: Adjust hyperparameters for better accuracy
4. **Deploy**: Use Docker Compose for production deployment
5. **Monitor**: Set up logging and monitoring
6. **Iterate**: Collect feedback and improve models

---

## Support & Documentation

- **Full Documentation**: See `IMPLEMENTATION_GUIDE.md`
- **API Documentation**: Visit `/docs` endpoint when running
- **Architecture**: See `SYSTEM_ARCHITECTURE.md`
- **Database Schema**: See `DATABASE_SCHEMA.md`

---

**Ready to get started? Run:**

```bash
docker-compose up -d
```

Then visit: **http://localhost:3000**
