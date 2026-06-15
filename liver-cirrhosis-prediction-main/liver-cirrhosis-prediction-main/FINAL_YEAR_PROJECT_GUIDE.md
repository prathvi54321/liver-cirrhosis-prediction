# Explainable AI Liver Cirrhosis Detection System - Master Guide

## 1. Professional System Architecture

The system uses a modular healthcare AI architecture:

- React.js frontend for patient, doctor, and admin workflows.
- FastAPI backend for authentication, prediction APIs, chatbot APIs, diagnosis history, and PDF reports.
- Hybrid AI engine that combines symptom/lab scoring with optional medical image analysis.
- Explainable AI layer for human-readable explanations, SHAP-ready tabular interpretation, and Grad-CAM-style image heatmaps.
- SQLite for local demo, with a clean path to MySQL/PostgreSQL for deployment.

Production flow:

1. User logs in.
2. Patient enters symptoms and optional lab/imaging data.
3. Backend validates input.
4. ML symptom model produces risk probability.
5. CNN imaging model produces image-based severity probability.
6. Hybrid fusion generates final disease stage and confidence.
7. XAI layer explains major clinical/image factors.
8. Recommendations and PDF report are generated.
9. Dashboard displays historical risk trends.

## 2. Folder Structure

Important folders:

- `backend/`: FastAPI APIs, authentication, database models, prediction handler, chatbot, PDF reports.
- `frontend/`: React app with pages for home/login, diagnosis, results, reports, dashboard, chatbot.
- `ai_pipeline/`: training scripts for ML and DL models.
- `backend/models/`: place trained `.pkl`, `.h5`, `.pth`, or exported model files here.
- `backend/reports/`: generated PDF reports.
- root `.md` files: detailed architecture, database, dataset, training, XAI, chatbot, and deployment documentation.

## 3. Database Schema

Core tables:

- `users`: patient, doctor, admin accounts.
- `diagnosis_records`: symptoms, uploaded image data, prediction result, explanations, confidence, risk level, recommendations.
- `chat_sessions`: chatbot conversations.
- `chat_messages`: individual user/bot messages.
- `model_versions`: model metadata and performance tracking.
- `audit_logs`: security and healthcare access audit trail.

## 4. Detailed Workflow

Patient workflow:

1. Register/login.
2. Fill symptom form.
3. Optionally upload ultrasound/CT/MRI image.
4. Submit diagnosis.
5. View risk level, stage, confidence, explanation, recommendations.
6. Download PDF report.
7. Use chatbot for liver-health education.

Doctor workflow:

1. Login as doctor.
2. View dashboard analytics.
3. Review patient prediction history.
4. Monitor risk trends.
5. Use reports for consultation support.

Admin workflow:

1. Manage users.
2. Track model versions.
3. Review audit logs.
4. Prepare system for deployment.

## 5. Frontend Pages

- Home/Login/Register page.
- Diagnosis page with symptoms, lab values, and imaging upload.
- Results page with stage, risk, confidence, probability distribution, XAI explanation, and recommendations.
- Reports page for PDF access.
- Dashboard page for analytics and monitoring.
- Chatbot page for medical Q&A and symptom collection.

## 6. Backend APIs

Current main APIs:

- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /predict`
- `POST /diagnosis/predict`
- `GET /history`
- `GET /diagnosis/results/{diagnosis_id}`
- `POST /chat/start`
- `POST /chat/message`
- `GET /chat/history/{session_id}`
- `GET /reports/{diagnosis_id}`

## 7. AI Model Pipeline

Recommended ML pipeline:

1. Collect symptom, clinical, and lab data.
2. Clean missing values and outliers.
3. Encode categorical fields.
4. Normalize numeric values.
5. Train Random Forest, XGBoost, and SVM.
6. Compare accuracy, precision, recall, F1-score, ROC-AUC.
7. Calibrate probabilities.
8. Export best models using `joblib`.

Recommended DL pipeline:

1. Collect ultrasound/CT/MRI images.
2. Remove corrupted images.
3. Resize and normalize images.
4. Apply augmentation.
5. Train CNN/ResNet/EfficientNet.
6. Evaluate with accuracy, sensitivity, specificity, AUC, confusion matrix.
7. Export trained weights.

## 8. Dataset Requirements

Tabular data:

- Age, sex, fatigue, alcohol intake, weight loss, appetite loss, abdominal swelling, jaundice.
- Clinical fields: ascites, hepatomegaly, edema, spider angiomas.
- Optional lab fields: bilirubin, albumin, AST, platelets, prothrombin.
- Target label: normal, mild fibrosis, moderate fibrosis, severe fibrosis, cirrhosis.

Imaging data:

- Ultrasound, CT, or MRI images.
- Labels by fibrosis/cirrhosis stage.
- Expert annotation preferred for segmentation and damaged-area localization.

## 9. Training Pipeline

Use `ai_pipeline/train_ml.py` and `ai_pipeline/train_dl.py` as starting scripts. Add real datasets under `ai_pipeline/datasets/`, train models, then save outputs into `backend/models/`.

## 10. CNN Model Architecture

Recommended architecture:

- Input: `224x224x3` or `256x256x1`.
- Convolution blocks with batch normalization and ReLU.
- Max pooling.
- Dropout.
- Global average pooling.
- Dense classification head.
- Softmax output for 5 stages.

For publication-quality results, use transfer learning:

- ResNet50
- EfficientNetB0/B3
- DenseNet121

## 11. Explainable AI Integration

Tabular explanation:

- SHAP feature importance for symptoms and lab values.
- Explain high-risk factors such as jaundice, ascites, low albumin, high bilirubin, high AST, low platelets.

Image explanation:

- Grad-CAM heatmap over uploaded liver image.
- Highlight regions that influenced CNN classification.

Human explanation:

- Convert technical factors into patient-friendly language.

## 12. Chatbot Implementation

Current chatbot supports rule-based medical Q&A. It can be extended with:

- LLM API integration.
- Retrieval-augmented generation over liver-health documents.
- Symptom collection flow.
- Follow-up reminders.
- Emergency warning detection.

Medical safety rule:

- The chatbot must never claim final diagnosis.
- It should recommend doctor consultation for serious symptoms.

## 13. Deployment Architecture

Local demo:

- React on `http://127.0.0.1:3000`
- FastAPI on `http://127.0.0.1:8000`
- SQLite database.

Cloud deployment:

- Frontend: Vercel/Netlify/S3 + CloudFront.
- Backend: Render/Railway/AWS EC2/Azure App Service.
- Database: PostgreSQL/MySQL managed service.
- Model storage: S3/Azure Blob.
- Containerization: Docker + Docker Compose.

## 14. UML Diagrams

Recommended diagrams for report:

- Use case diagram: patient, doctor, admin.
- Class diagram: User, DiagnosisRecord, ChatSession, ChatMessage, ModelVersion.
- Sequence diagram: login -> prediction -> XAI -> report.
- Activity diagram: symptom/image submission workflow.
- Deployment diagram: frontend, backend, database, model store.

## 15. Implementation Roadmap

Phase 1: Make app run locally.

Phase 2: Add real tabular dataset and train ML models.

Phase 3: Add imaging dataset and train CNN/ResNet.

Phase 4: Add SHAP and Grad-CAM visual outputs.

Phase 5: Improve dashboard analytics.

Phase 6: Add PDF report polish.

Phase 7: Add cloud deployment.

Phase 8: Prepare IEEE-style paper and evaluation results.

## 16. Research Paper Ideas

- Hybrid ML and deep learning framework for non-invasive liver cirrhosis screening.
- Explainable AI for liver fibrosis staging using tabular and imaging data.
- Smart healthcare assistant for liver disease monitoring using NLP and XAI.
- Comparative study of Random Forest, XGBoost, SVM, CNN, ResNet, and EfficientNet for cirrhosis detection.

## 17. Accuracy Optimization Techniques

- Use medically validated features.
- Balance classes with SMOTE or weighted loss.
- Use data augmentation for images.
- Tune hyperparameters with grid search or Optuna.
- Use ensemble fusion.
- Calibrate probability outputs.
- Validate on separate test data.
- Track sensitivity and specificity, not only accuracy.

## 18. Security Considerations

- JWT authentication.
- Role-based access control.
- Password hashing.
- Input validation with Pydantic.
- File upload validation.
- Audit logs.
- Avoid storing unnecessary medical data.
- Use HTTPS in production.
- Encrypt database backups.

## 19. Cloud Deployment Steps

1. Create production `.env`.
2. Replace SQLite with PostgreSQL/MySQL.
3. Build React frontend.
4. Containerize backend.
5. Upload trained models.
6. Deploy backend API.
7. Deploy frontend with API URL configured.
8. Enable HTTPS.
9. Run smoke tests.
10. Monitor logs and errors.

## 20. End-to-End Data Flow

Input data starts from the React form. It is sent as multipart form data to FastAPI. The backend validates symptoms, decodes uploaded image, runs hybrid prediction, creates explanation, stores diagnosis in the database, generates report metadata, and returns the result to the frontend. The frontend displays stage, risk, confidence, probability, explanation, and recommendations. Doctors can later review diagnosis history and trends.

## Current Demo Note

The current runnable project uses deterministic demo-safe inference when real trained model files are unavailable. For final submission, train models with real datasets and replace the placeholder scoring with saved ML/DL models in `backend/models/`.
