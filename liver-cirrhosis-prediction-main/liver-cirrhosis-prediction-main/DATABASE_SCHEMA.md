# Database Schema Design
## Complete SQLAlchemy Models & Relationships

```sql
=== USER MANAGEMENT ===

TABLE: users
├── id (INT, PK)
├── email (VARCHAR, UNIQUE)
├── username (VARCHAR, UNIQUE)
├── password_hash (VARCHAR)
├── first_name (VARCHAR)
├── last_name (VARCHAR)
├── role (ENUM: 'patient', 'doctor', 'admin')
├── phone_number (VARCHAR)
├── date_of_birth (DATE)
├── gender (ENUM: 'M', 'F', 'Other')
├── is_active (BOOLEAN, default=True)
├── is_verified (BOOLEAN, default=False)
├── created_at (DATETIME, default=now)
├── updated_at (DATETIME)
├── last_login (DATETIME)
└── mfa_enabled (BOOLEAN, default=False)


TABLE: patients (extends users)
├── id (INT, PK, FK to users.id)
├── medical_id (VARCHAR, UNIQUE)
├── blood_group (VARCHAR)
├── allergies (TEXT)
├── medications (TEXT)
├── family_history (TEXT)
├── emergency_contact (VARCHAR)
├── insurance_provider (VARCHAR)
├── insurance_number (VARCHAR)
├── doctor_id (INT, FK to doctors.id, nullable)
├── height_cm (DECIMAL)
├── weight_kg (DECIMAL)
├── bmi (DECIMAL, computed)
├── lifestyle_notes (TEXT)
└── risk_level (ENUM: 'low', 'medium', 'high', 'critical')


TABLE: doctors (extends users)
├── id (INT, PK, FK to users.id)
├── license_number (VARCHAR, UNIQUE)
├── specialization (VARCHAR)
├── qualification (VARCHAR)
├── hospital_affiliation (VARCHAR)
├── years_experience (INT)
├── is_verified (BOOLEAN)
├── patients_count (INT)
├── consultation_fee (DECIMAL, nullable)
├── availability_status (ENUM: 'available', 'busy', 'offline')
└── profile_image_url (VARCHAR, nullable)


TABLE: admin_users (extends users)
├── id (INT, PK, FK to users.id)
├── permissions (JSON)
├── department (VARCHAR)
└── access_level (INT, 1-5)


=== PREDICTIONS ===

TABLE: predictions
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── prediction_type (ENUM: 'symptom', 'imaging', 'hybrid')
├── input_data (JSON)  # Stores input features
├── symptom_score (DECIMAL, 0-1)
├── imaging_score (DECIMAL, 0-1)
├── final_score (DECIMAL, 0-1)
├── predicted_stage (ENUM: 'healthy', 'stage_1', 'stage_2', 'stage_3', 'stage_4')
├── confidence_score (DECIMAL, 0-1)
├── risk_level (ENUM: 'none', 'low', 'medium', 'high', 'critical')
├── model_version (VARCHAR)
├── inference_time_ms (INT)
├── created_at (DATETIME)
├── reviewed_by_doctor (INT, FK to doctors.id, nullable)
├── doctor_approval (BOOLEAN, nullable)
├── doctor_notes (TEXT, nullable)
├── clinical_validation (BOOLEAN)
└── archived (BOOLEAN, default=False)


TABLE: prediction_details
├── id (INT, PK)
├── prediction_id (INT, FK to predictions.id)
├── model_name (VARCHAR)  # e.g., 'xgboost', 'resnet50'
├── individual_score (DECIMAL)
├── processing_time_ms (INT)
├── feature_importance (JSON)  # SHAP values
├── raw_output (JSON)
└── created_at (DATETIME)


=== MEDICAL IMAGING ===

TABLE: medical_images
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── prediction_id (INT, FK to predictions.id, nullable)
├── image_type (ENUM: 'ultrasound', 'ct_scan', 'mri_scan', 'xray')
├── file_name (VARCHAR)
├── file_path (VARCHAR)  # S3 URL
├── file_size_kb (INT)
├── mime_type (VARCHAR)
├── dicom_metadata (JSON, nullable)  # DICOM info
├── image_quality_score (DECIMAL, nullable)
├── is_preprocessed (BOOLEAN, default=False)
├── preprocessing_status (ENUM: 'pending', 'processing', 'completed', 'failed')
├── uploaded_at (DATETIME)
├── uploaded_by (INT, FK to users.id)
├── radiologist_id (INT, FK to doctors.id, nullable)
├── radiologist_notes (TEXT, nullable)
├── is_flagged (BOOLEAN, default=False)
├── flag_reason (VARCHAR, nullable)
└── encrypted (BOOLEAN, default=True)


TABLE: image_segmentation
├── id (INT, PK)
├── medical_image_id (INT, FK to medical_images.id)
├── segmentation_mask_path (VARCHAR)  # Liver region mask
├── liver_volume_ml (DECIMAL, nullable)
├── fibrosis_percentage (DECIMAL, nullable)
├── cirrhotic_area_percentage (DECIMAL, nullable)
├── healthy_tissue_percentage (DECIMAL, nullable)
├── segmentation_model (VARCHAR)
├── processing_time_ms (INT)
└── created_at (DATETIME)


TABLE: image_annotations
├── id (INT, PK)
├── medical_image_id (INT, FK to medical_images.id)
├── annotation_type (ENUM: 'abnormality', 'landmark', 'measurement')
├── description (TEXT)
├── coordinates_json (JSON)  # x, y, width, height
├── severity (ENUM: 'mild', 'moderate', 'severe')
├── annotated_by (INT, FK to doctors.id)
├── created_at (DATETIME)
└── updated_at (DATETIME)


=== SYMPTOMS ===

TABLE: symptom_categories
├── id (INT, PK)
├── name (VARCHAR)
├── description (TEXT)
└── priority (INT)


TABLE: symptoms
├── id (INT, PK)
├── category_id (INT, FK to symptom_categories.id)
├── name (VARCHAR)
├── medical_code (VARCHAR)  # ICD-10
├── description (TEXT)
├── severity_scale (INT, 1-10)
└── common_in_stage (VARCHAR)


TABLE: patient_symptoms
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── symptom_id (INT, FK to symptoms.id)
├── severity_level (INT, 1-10)
├── duration_days (INT)
├── onset_date (DATE)
├── is_resolved (BOOLEAN)
├── resolved_date (DATE, nullable)
├── notes (TEXT)
├── reported_at (DATETIME)
└── reported_by (INT, FK to users.id)


=== CHATBOT ===

TABLE: chatbot_sessions
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── session_start (DATETIME)
├── session_end (DATETIME, nullable)
├── session_duration_min (INT, nullable)
├── is_active (BOOLEAN)
├── conversation_summary (TEXT, nullable)
├── symptoms_collected (JSON)
├── actions_suggested (JSON)
├── satisfaction_rating (INT, 1-5, nullable)
└── feedback (TEXT, nullable)


TABLE: chatbot_messages
├── id (INT, PK)
├── session_id (INT, FK to chatbot_sessions.id)
├── message_type (ENUM: 'user', 'bot')
├── content (TEXT)
├── intent (VARCHAR, nullable)  # 'symptom_query', 'appointment', etc.
├── sentiment (ENUM: 'positive', 'neutral', 'negative')
├── confidence_score (DECIMAL)
├── timestamp (DATETIME)
├── helpful (BOOLEAN, nullable)
└── response_time_ms (INT, nullable)


TABLE: chatbot_faq
├── id (INT, PK)
├── question (TEXT)
├── answer (TEXT)
├── category (VARCHAR)
├── tags (JSON)
├── view_count (INT, default=0)
├── helpful_count (INT, default=0)
├── not_helpful_count (INT, default=0)
├── last_updated (DATETIME)
└── created_by (INT, FK to users.id)


=== RECOMMENDATIONS ===

TABLE: recommendations
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── prediction_id (INT, FK to predictions.id, nullable)
├── recommendation_type (ENUM: 'diet', 'lifestyle', 'medication', 'appointment', 'alert')
├── title (VARCHAR)
├── description (TEXT)
├── priority_level (ENUM: 'low', 'medium', 'high', 'urgent')
├── is_personalized (BOOLEAN)
├── reference_stage (VARCHAR)
├── doctor_approved (BOOLEAN, nullable)
├── approved_by (INT, FK to doctors.id, nullable)
├── created_at (DATETIME)
├── scheduled_for (DATETIME, nullable)
├── completed (BOOLEAN, default=False)
├── completed_at (DATETIME, nullable)
└── feedback_score (INT, 1-5, nullable)


TABLE: diet_recommendations
├── id (INT, PK)
├── recommendation_id (INT, FK to recommendations.id)
├── food_item (VARCHAR)
├── quantity (VARCHAR)
├── frequency (VARCHAR)
├── reason (TEXT)
├── nutrition_info (JSON)  # Calories, protein, etc.
└── restrictions (TEXT)


TABLE: lifestyle_recommendations
├── id (INT, PK)
├── recommendation_id (INT, FK to recommendations.id)
├── activity_type (VARCHAR)
├── duration_min (INT)
├── frequency_per_week (INT)
├── intensity (ENUM: 'light', 'moderate', 'vigorous')
├── precautions (TEXT)
└── expected_benefits (TEXT)


=== REPORTS ===

TABLE: medical_reports
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── prediction_id (INT, FK to predictions.id)
├── report_type (ENUM: 'comprehensive', 'simple', 'follow_up')
├── title (VARCHAR)
├── summary (TEXT)
├── detailed_findings (TEXT)
├── xai_explanation (JSON)  # Explainability data
├── recommendations_text (TEXT)
├── doctor_signature (VARCHAR, nullable)
├── report_status (ENUM: 'draft', 'pending_review', 'approved', 'finalized')
├── reviewed_by (INT, FK to doctors.id, nullable)
├── file_path (VARCHAR)  # PDF location
├── generated_at (DATETIME)
├── finalized_at (DATETIME, nullable)
├── expiry_date (DATE, nullable)
└── archived (BOOLEAN, default=False)


TABLE: report_sections
├── id (INT, PK)
├── report_id (INT, FK to medical_reports.id)
├── section_name (VARCHAR)
├── section_content (TEXT)
├── section_order (INT)
├── is_visible (BOOLEAN)
└── updated_at (DATETIME)


=== RISK & MONITORING ===

TABLE: risk_assessments
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── assessment_date (DATETIME)
├── overall_risk_score (DECIMAL, 0-1)
├── cirrhosis_probability (DECIMAL, 0-1)
├── progression_risk (DECIMAL, 0-1)
├── mortality_risk (DECIMAL, 0-1)
├── risk_factors (JSON)  # List of contributing factors
├── risk_category (ENUM: 'none', 'low', 'medium', 'high', 'critical')
├── recommended_action (VARCHAR)
├── monitoring_frequency (VARCHAR)  # e.g., 'monthly', 'quarterly'
└── assessed_by (INT, FK to doctors.id, nullable)


TABLE: survival_predictions
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── prediction_date (DATETIME)
├── one_year_survival (DECIMAL, 0-1)
├── five_year_survival (DECIMAL, 0-1)
├── ten_year_survival (DECIMAL, 0-1)
├── median_survival_months (INT)
├── confidence_interval (JSON)
├── clinical_stage (VARCHAR)
├── model_used (VARCHAR)
└── notes (TEXT, nullable)


TABLE: patient_monitoring_logs
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── log_date (DATETIME)
├── health_status (VARCHAR)
├── symptoms_status (JSON)
├── last_prediction_date (DATE, nullable)
├── follow_up_due (DATE, nullable)
├── alert_status (ENUM: 'no_alert', 'warning', 'critical')
├── alert_message (TEXT, nullable)
└── logged_by (INT, FK to users.id)


=== APPOINTMENTS & CONSULTATIONS ===

TABLE: appointments
├── id (INT, PK)
├── patient_id (INT, FK to patients.id)
├── doctor_id (INT, FK to doctors.id)
├── scheduled_datetime (DATETIME)
├── consultation_type (ENUM: 'follow_up', 'initial', 'emergency', 'video')
├── reason (TEXT)
├── status (ENUM: 'scheduled', 'completed', 'cancelled', 'no_show')
├── consultation_notes (TEXT, nullable)
├── prescription (TEXT, nullable)
├── completed_at (DATETIME, nullable)
├── duration_min (INT, nullable)
├── follow_up_required (BOOLEAN)
├── next_appointment_date (DATE, nullable)
└── reminder_sent (BOOLEAN, default=False)


TABLE: prescription_templates
├── id (INT, PK)
├── name (VARCHAR)
├── disease_stage (VARCHAR)
├── medication_1 (VARCHAR)
├── dosage_1 (VARCHAR)
├── frequency_1 (VARCHAR)
├── medication_2 (VARCHAR)
├── dosage_2 (VARCHAR)
├── frequency_2 (VARCHAR)
├── medication_3 (VARCHAR)
├── dosage_3 (VARCHAR)
├── frequency_3 (VARCHAR)
├── precautions (TEXT)
├── created_by (INT, FK to doctors.id)
└── last_updated (DATETIME)


=== ANALYTICS & AUDIT ===

TABLE: analytics_dashboard_stats
├── id (INT, PK)
├── date (DATE)
├── total_patients (INT)
├── new_patients_today (INT)
├── total_predictions (INT)
├── predictions_today (INT)
├── avg_prediction_time_ms (DECIMAL)
├── model_accuracy (DECIMAL)
├── system_uptime_percent (DECIMAL)
├── active_users (INT)
└── api_calls (INT)


TABLE: audit_logs
├── id (INT, PK)
├── user_id (INT, FK to users.id)
├── action (VARCHAR)
├── resource_type (VARCHAR)
├── resource_id (INT)
├── timestamp (DATETIME)
├── ip_address (VARCHAR)
├── user_agent (VARCHAR)
├── status (ENUM: 'success', 'failure')
├── details (JSON)
└── severity (ENUM: 'info', 'warning', 'error', 'critical')


TABLE: system_logs
├── id (INT, PK)
├── log_level (ENUM: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
├── service (VARCHAR)
├── message (TEXT)
├── stack_trace (TEXT, nullable)
├── timestamp (DATETIME)
├── request_id (VARCHAR, nullable)
└── resolved (BOOLEAN, default=False)


=== NOTIFICATIONS ===

TABLE: notifications
├── id (INT, PK)
├── recipient_id (INT, FK to users.id)
├── notification_type (ENUM: 'appointment', 'result', 'alert', 'recommendation', 'system')
├── title (VARCHAR)
├── message (TEXT)
├── related_entity_type (VARCHAR)
├── related_entity_id (INT)
├── is_read (BOOLEAN, default=False)
├── read_at (DATETIME, nullable)
├── priority (ENUM: 'low', 'medium', 'high', 'urgent')
├── created_at (DATETIME)
└── expires_at (DATETIME, nullable)


TABLE: notification_preferences
├── id (INT, PK)
├── user_id (INT, FK to users.id)
├── notification_type (VARCHAR)
├── email_enabled (BOOLEAN, default=True)
├── sms_enabled (BOOLEAN, default=False)
├── push_enabled (BOOLEAN, default=True)
├── quiet_hours_start (TIME, nullable)
├── quiet_hours_end (TIME, nullable)
└── updated_at (DATETIME)


=== FEEDBACK & RATINGS ===

TABLE: user_feedback
├── id (INT, PK)
├── user_id (INT, FK to users.id)
├── feedback_type (ENUM: 'bug', 'feature_request', 'general', 'complaint')
├── title (VARCHAR)
├── description (TEXT)
├── rating (INT, 1-5, nullable)
├── attachments (JSON, nullable)
├── status (ENUM: 'open', 'in_progress', 'resolved', 'closed')
├── response (TEXT, nullable)
├── responded_by (INT, FK to users.id, nullable)
├── created_at (DATETIME)
├── resolved_at (DATETIME, nullable)
└── archived (BOOLEAN, default=False)


TABLE: prediction_feedback
├── id (INT, PK)
├── prediction_id (INT, FK to predictions.id)
├── user_id (INT, FK to users.id)
├── accuracy_rating (INT, 1-5)
├── usefulness_rating (INT, 1-5)
├── explanation_clarity (INT, 1-5)
├── comments (TEXT, nullable)
├── would_recommend (BOOLEAN)
├── created_at (DATETIME)
└── helpful_count (INT, default=0)
```

## 🔑 Key Relationships

```
users (parent)
├── patients (1:1 - extends users)
├── doctors (1:1 - extends users)
└── admin_users (1:1 - extends users)

patients (parent)
├── medical_images (1:M)
├── predictions (1:M)
├── chatbot_sessions (1:M)
├── recommendations (1:M)
├── risk_assessments (1:M)
├── patient_symptoms (1:M)
├── appointments (1:M)
├── medical_reports (1:M)
└── patient_monitoring_logs (1:M)

predictions (parent)
├── prediction_details (1:M)
├── image_annotations (1:M)
└── medical_reports (1:M)

medical_images (parent)
├── prediction_details (1:M)
├── image_segmentation (1:M)
└── image_annotations (1:M)

doctors (parent)
├── patients (1:M)  # As assigned doctor
├── appointments (1:M)  # As consulting doctor
├── medical_reports (1:M)  # As reviewer
└── recommendations (1:M)  # As approver
```

## 📊 Database Normalization

- **Normalization Level:** 3NF (Third Normal Form)
- **No Denormalization:** Except for computed fields (BMI, aggregate counts)
- **Referential Integrity:** All FKs properly defined
- **Index Strategy:**
  - Primary keys on all tables
  - Composite indexes on foreign keys
  - Search indexes on email, username
  - Date range indexes on temporal fields

## 🔐 Security

- Passwords stored as bcrypt hashes (never plain text)
- Medical images encrypted at rest (AES-256)
- PII masked in logs
- Audit trail for all data modifications
- Role-based access control via SQL views

## 📝 Example SQLAlchemy Models

See `backend/app/models/` directory for complete implementation.

## 🔄 Migration Strategy

Use Alembic for database migrations:
```bash
alembic init backend/migrations
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

## ✅ Testing

- Foreign key constraints enabled
- Referential integrity tests
- Cascade delete tests for sensitive operations
- Data type validation tests
