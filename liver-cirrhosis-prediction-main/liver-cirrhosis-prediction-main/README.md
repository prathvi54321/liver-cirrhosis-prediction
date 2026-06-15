# Liver Cirrhosis Prediction System

An AI-powered smart healthcare assistant system for liver cirrhosis prediction using hybrid ML+DL models.

## Project Structure

```
liver-cirrhosis-project/
├── backend/                    # FastAPI Backend
│   ├── main.py                 # API entry point
│   ├── models_handler.py       # ML/DL hybrid inference
│   ├── schemas.py              # Data validation
│   ├── database.py             # SQLAlchemy models
│   ├── auth.py                 # Authentication logic
│   ├── models/                 # Pre-trained models
│   ├── uploads/                # Temporary file storage
│   └── utils/                  # Helper functions
│       ├── xai_logic.py        # Explainable AI (GradCAM)
│       ├── chatbot_api.py      # Chatbot integration
│       └── pdf_gen.py          # PDF report generation
├── ai_pipeline/                # Model Training
│   ├── train_ml.py             # XGBoost training
│   ├── train_dl.py             # CNN training
│   └── datasets/               # Training data
├── frontend/                   # React.js Frontend (Future)
└── requirements.txt            # Python dependencies
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Database

```bash
cd backend
python
>>> from database import Base, engine
>>> Base.metadata.create_all(bind=engine)
>>> exit()
```

### 3. Run the Backend

```bash
cd backend
python main.py
```

Or use uvicorn directly:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- **GET** `/` - Health check
- **POST** `/signup` - User registration
- **POST** `/login` - User login
- **POST** `/predict` - Liver cirrhosis prediction
- **POST** `/generate-report/{record_id}` - Download PDF report
- **GET** `/ask-assistant?query=...` - Chat with assistant

## Example API Requests

### Signup
```bash
curl -X POST "http://localhost:8000/signup?username=test&password=pass123&role=patient"
```

### Login
```bash
curl -X POST "http://localhost:8000/login?username=test&password=pass123"
```

### Predict
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"age\": 45,
    \"sex\": 1,
    \"ascites\": 0,
    \"hepatomegaly\": 0,
    \"spiders\": 0,
    \"edema\": 0,
    \"bilirubin\": 0.8,
    \"cholesterol\": 210,
    \"albumin\": 3.5,
    \"copper\": 50,
    \"alk_phos\": 75,
    \"ast\": 40,
    \"triglycerides\": 150,
    \"platelets\": 250,
    \"prothrombin\": 11
  }"
```

## Model Setup

Models are expected in `backend/models/`:
- `symptom_model.pkl` - XGBoost classifier
- `liver_cnn_model.h5` - EfficientNet deep learning model

For now, the system uses mock predictions if models don't exist.

## Technologies Used

- **Backend**: FastAPI, SQLAlchemy
- **ML**: XGBoost, TensorFlow/Keras
- **XAI**: GradCAM, SHAP
- **Database**: SQLite (configurable to PostgreSQL/MySQL)
- **Auth**: JWT with Passlib
- **PDF**: FPDF2

## Next Steps

1. Train and save models in `backend/models/`
2. Implement React frontend
3. Add image upload for medical imaging analysis
4. Deploy with Docker & Kubernetes
