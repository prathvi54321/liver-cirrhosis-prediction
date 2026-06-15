# API Specifications & Endpoints
## Complete RESTful API Documentation

---

## 🔐 Authentication Endpoints

### 1. User Registration
```
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "patient@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe",
  "role": "patient",
  "phone_number": "+1234567890"
}

Response (201):
{
  "id": 1,
  "email": "patient@example.com",
  "username": "john_doe_123",
  "role": "patient",
  "created_at": "2024-05-14T10:30:00Z",
  "message": "User created successfully"
}
```

### 2. User Login
```
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "patient@example.com",
  "password": "SecurePass123!"
}

Response (200):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "email": "patient@example.com",
    "role": "patient"
  }
}
```

### 3. Refresh Token
```
POST /api/v1/auth/refresh
Authorization: Bearer <refresh_token>

Response (200):
{
  "access_token": "new_token...",
  "token_type": "bearer"
}
```

### 4. Logout
```
POST /api/v1/auth/logout
Authorization: Bearer <access_token>

Response (200):
{
  "message": "Logged out successfully"
}
```

---

## 👥 User Management Endpoints

### 5. Get Current User Profile
```
GET /api/v1/users/me
Authorization: Bearer <token>

Response (200):
{
  "id": 1,
  "email": "patient@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "patient",
  "is_verified": true,
  "profile_image_url": "https://s3.../profile.jpg",
  "created_at": "2024-05-14T10:30:00Z"
}
```

### 6. Update User Profile
```
PUT /api/v1/users/me
Authorization: Bearer <token>
Content-Type: application/json

{
  "first_name": "John",
  "last_name": "Doe",
  "phone_number": "+1234567890",
  "profile_image_url": "https://..."
}

Response (200):
{
  "message": "Profile updated successfully",
  "user": {...}
}
```

### 7. Change Password
```
POST /api/v1/users/change-password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "OldPass123!",
  "new_password": "NewPass456!",
  "confirm_password": "NewPass456!"
}

Response (200):
{
  "message": "Password changed successfully"
}
```

### 8. Get User Preferences
```
GET /api/v1/users/preferences
Authorization: Bearer <token>

Response (200):
{
  "notifications_email": true,
  "notifications_sms": false,
  "notifications_push": true,
  "theme": "dark",
  "language": "en"
}
```

### 9. Update Preferences
```
PUT /api/v1/users/preferences
Authorization: Bearer <token>

{
  "notifications_email": true,
  "theme": "dark"
}

Response (200):
{
  "message": "Preferences updated"
}
```

---

## 🩺 Prediction Endpoints

### 10. Create Prediction (Symptoms Only)
```
POST /api/v1/predictions/symptom
Authorization: Bearer <token>
Content-Type: application/json

{
  "age": 45,
  "sex": 1,
  "fatigue_level": 7,
  "alcohol_consumption": 3,
  "weight_loss_kg": 5,
  "abdominal_swelling": true,
  "appetite_loss": true,
  "jaundice": false,
  "fever": false,
  "ascites": 1,
  "hepatomegaly": 0,
  "spiders": 0,
  "edema": 0,
  "bilirubin": 0.8,
  "cholesterol": 210,
  "albumin": 3.5,
  "copper": 50,
  "alk_phos": 75,
  "ast": 40,
  "triglycerides": 150,
  "platelets": 250,
  "prothrombin": 11
}

Response (201):
{
  "prediction_id": 42,
  "prediction_type": "symptom",
  "symptom_score": 0.82,
  "imaging_score": null,
  "final_score": 0.82,
  "predicted_stage": "stage_2",
  "confidence_score": 0.82,
  "risk_level": "high",
  "requires_imaging": true,
  "created_at": "2024-05-14T11:00:00Z"
}
```

### 11. Create Prediction (Medical Image)
```
POST /api/v1/predictions/imaging
Authorization: Bearer <token>
Content-Type: multipart/form-data

Parameters:
  image: <file.jpg>
  image_type: "ultrasound"  # ultrasound, ct_scan, mri_scan
  radiologist_notes: "Routine screening"

Response (202):  # Asynchronous processing
{
  "prediction_id": 43,
  "status": "processing",
  "message": "Image uploaded and processing",
  "estimated_time_sec": 30,
  "job_id": "job_uuid_123"
}

# Poll for results:
GET /api/v1/predictions/43/status
Response (200):
{
  "prediction_id": 43,
  "status": "completed",
  "imaging_score": 0.85,
  "predicted_stage": "stage_2",
  "confidence_score": 0.85,
  "risk_level": "high"
}
```

### 12. Create Hybrid Prediction (Symptoms + Image)
```
POST /api/v1/predictions/hybrid
Authorization: Bearer <token>
Content-Type: multipart/form-data

Parameters:
  image: <file.jpg>
  image_type: "ultrasound"
  symptom_data: {
    "age": 45,
    "fatigue_level": 7,
    ...
  }

Response (202):
{
  "prediction_id": 44,
  "prediction_type": "hybrid",
  "status": "processing",
  "estimated_time_sec": 45
}

# When completed:
{
  "prediction_id": 44,
  "symptom_score": 0.82,
  "imaging_score": 0.85,
  "final_score": 0.837,  # Weighted: 60% imaging + 40% symptoms
  "predicted_stage": "stage_2",
  "confidence_score": 0.84,
  "risk_level": "high"
}
```

### 13. Get Prediction Details
```
GET /api/v1/predictions/44
Authorization: Bearer <token>

Response (200):
{
  "prediction_id": 44,
  "patient_id": 1,
  "prediction_type": "hybrid",
  "symptom_score": 0.82,
  "imaging_score": 0.85,
  "final_score": 0.837,
  "predicted_stage": "stage_2",
  "confidence_score": 0.84,
  "risk_level": "high",
  "model_version": "v2.1",
  "created_at": "2024-05-14T11:00:00Z",
  "doctor_review": {
    "reviewed_by": "Dr. Smith",
    "approval": true,
    "notes": "Results align with clinical findings"
  }
}
```

### 14. Get Prediction History
```
GET /api/v1/predictions/history?page=1&limit=10&sort=-created_at
Authorization: Bearer <token>

Response (200):
{
  "total": 25,
  "page": 1,
  "limit": 10,
  "items": [
    {
      "prediction_id": 44,
      "created_at": "2024-05-14T11:00:00Z",
      "predicted_stage": "stage_2",
      "confidence_score": 0.84
    },
    ...
  ]
}
```

### 15. Delete Prediction
```
DELETE /api/v1/predictions/44
Authorization: Bearer <token>

Response (200):
{
  "message": "Prediction archived successfully"
}
```

---

## 🧠 Explainability Endpoints

### 16. Get Explainability (SHAP + Grad-CAM)
```
GET /api/v1/predictions/44/explainability
Authorization: Bearer <token>

Response (200):
{
  "prediction_id": 44,
  "explainability_type": "hybrid",
  
  "symptom_explanation": {
    "method": "shap",
    "top_features": [
      {
        "feature": "bilirubin",
        "importance": 0.28,
        "value": 0.8,
        "direction": "increase_risk"
      },
      {
        "feature": "albumin",
        "importance": 0.22,
        "value": 3.5,
        "direction": "decrease_risk"
      },
      {
        "feature": "age",
        "importance": 0.15,
        "value": 45,
        "direction": "increase_risk"
      }
    ],
    "summary": "High bilirubin and low albumin strongly indicate cirrhosis risk",
    "shap_plot_url": "https://s3.../shap_plot.png"
  },

  "image_explanation": {
    "method": "grad_cam",
    "focus_areas": [
      {
        "region": "upper_right_lobe",
        "attention_intensity": 0.92,
        "findings": "Increased echo density suggesting fibrosis"
      },
      {
        "region": "portal_vein",
        "attention_intensity": 0.78,
        "findings": "Portal hypertension signs"
      }
    ],
    "heatmap_url": "https://s3.../gradcam_heatmap.jpg",
    "segmentation_mask_url": "https://s3.../liver_mask.png"
  },

  "risk_factors": {
    "modifiable": [
      "Reduce alcohol consumption",
      "Improve diet quality",
      "Increase physical activity"
    ],
    "non_modifiable": [
      "Age: 45 years",
      "Sex: Male"
    ]
  },

  "prediction_narrative": "The patient shows stage 2 cirrhosis risk..."
}
```

### 17. Get XAI Report
```
GET /api/v1/predictions/44/xai-report
Authorization: Bearer <token>

Response (200):
{
  "report_type": "explainability_report",
  "prediction_id": 44,
  "file_url": "https://s3.../xai_report.pdf"
}
```

---

## 📸 Image Management Endpoints

### 18. Upload Medical Image
```
POST /api/v1/images/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

Parameters:
  image: <file.jpg or file.dcm>
  image_type: "ultrasound"
  notes: "Routine liver screening"

Response (201):
{
  "image_id": 52,
  "file_path": "https://s3.../images/52.jpg",
  "status": "uploaded",
  "preprocessing_status": "pending"
}
```

### 19. Get Image Details
```
GET /api/v1/images/52
Authorization: Bearer <token>

Response (200):
{
  "image_id": 52,
  "image_type": "ultrasound",
  "uploaded_at": "2024-05-14T10:00:00Z",
  "file_path": "https://s3.../images/52.jpg",
  "segmentation_mask_url": "https://s3.../mask_52.png",
  "liver_volume_ml": 1850,
  "fibrosis_percentage": 35,
  "annotations": [
    {
      "type": "abnormality",
      "description": "Heterogeneous echo",
      "severity": "moderate"
    }
  ]
}
```

### 20. List Patient Images
```
GET /api/v1/images?filter=all&page=1&limit=10
Authorization: Bearer <token>

Response (200):
{
  "total": 15,
  "items": [
    {
      "image_id": 52,
      "image_type": "ultrasound",
      "uploaded_at": "2024-05-14T10:00:00Z"
    }
  ]
}
```

---

## 💬 Chatbot Endpoints

### 21. Start Chat Session
```
POST /api/v1/chatbot/start-session
Authorization: Bearer <token>

Response (201):
{
  "session_id": "session_abc_123",
  "status": "active",
  "initial_message": "Hello! I'm your liver health assistant. How can I help you today?"
}
```

### 22. Send Chat Message
```
POST /api/v1/chatbot/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": "session_abc_123",
  "message": "I've been feeling very tired for 2 weeks"
}

Response (200):
{
  "message_id": "msg_123",
  "user_message": "I've been feeling very tired for 2 weeks",
  "bot_response": "Fatigue can be a sign of various conditions. Let me ask you more...",
  "intent": "symptom_reporting",
  "symptoms_extracted": ["fatigue"],
  "next_question": "How severe is your fatigue on a scale of 1-10?"
}
```

### 23. Get FAQ
```
GET /api/v1/chatbot/faq?category=symptoms&query=jaundice
Authorization: Bearer <token>

Response (200):
{
  "items": [
    {
      "faq_id": 1,
      "question": "What is jaundice?",
      "answer": "Jaundice is yellowing of skin and eyes...",
      "category": "symptoms"
    }
  ]
}
```

### 24. Get Chat History
```
GET /api/v1/chatbot/history?session_id=session_abc_123
Authorization: Bearer <token>

Response (200):
{
  "session_id": "session_abc_123",
  "messages": [
    {
      "type": "bot",
      "content": "Hello!...",
      "timestamp": "2024-05-14T11:00:00Z"
    },
    {
      "type": "user",
      "content": "I've been tired",
      "timestamp": "2024-05-14T11:00:30Z"
    }
  ]
}
```

---

## 📋 Reports Endpoints

### 25. Generate Report
```
POST /api/v1/reports/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "prediction_id": 44,
  "report_type": "comprehensive",
  "include_recommendations": true,
  "include_xai": true
}

Response (202):
{
  "report_id": 78,
  "status": "generating",
  "message": "Report is being generated..."
}

# Check status:
GET /api/v1/reports/78/status
Response (200):
{
  "report_id": 78,
  "status": "completed",
  "file_url": "https://s3.../report_78.pdf",
  "download_token": "token_xyz"
}
```

### 26. Download Report
```
GET /api/v1/reports/78/download?token=token_xyz
Response: PDF file stream
```

### 27. Get Report List
```
GET /api/v1/reports?page=1&limit=10
Authorization: Bearer <token>

Response (200):
{
  "total": 8,
  "items": [
    {
      "report_id": 78,
      "generated_at": "2024-05-14T11:30:00Z",
      "report_type": "comprehensive",
      "status": "finalized"
    }
  ]
}
```

---

## 📊 Dashboard & Analytics

### 28. Get Patient Dashboard
```
GET /api/v1/dashboard/patient
Authorization: Bearer <token>

Response (200):
{
  "health_summary": {
    "current_risk_level": "high",
    "last_prediction_date": "2024-05-14",
    "predictions_this_month": 3
  },
  "risk_trends": {
    "trend": "increasing",
    "change_percent": 12.5
  },
  "recent_predictions": [...],
  "upcoming_appointments": [...],
  "active_recommendations": [...],
  "alerts": [
    {
      "severity": "high",
      "message": "Follow-up consultation due",
      "action_url": "/appointments"
    }
  ]
}
```

### 29. Get Doctor Dashboard
```
GET /api/v1/dashboard/doctor
Authorization: Bearer <token>

Response (200):
{
  "patients_under_care": 42,
  "pending_reviews": 5,
  "critical_alerts": 2,
  "recent_cases": [...],
  "pending_cases": [
    {
      "case_id": 123,
      "patient_name": "John Doe",
      "prediction_date": "2024-05-14",
      "risk_level": "high",
      "status": "awaiting_review"
    }
  ]
}
```

### 30. Get Analytics
```
GET /api/v1/analytics?period=month&metric=accuracy
Authorization: Bearer <token>

Response (200):
{
  "metric": "accuracy",
  "period": "month",
  "value": 94.2,
  "trend": "up",
  "comparison_previous_period": 92.8,
  "historical_data": [
    {"date": "2024-05-01", "value": 91.5},
    {"date": "2024-05-02", "value": 92.0},
    ...
  ]
}
```

---

## 👨‍⚕️ Doctor Endpoints

### 31. Get Assigned Patients
```
GET /api/v1/doctors/patients?page=1&limit=20
Authorization: Bearer <token> (doctor only)

Response (200):
{
  "total": 42,
  "items": [
    {
      "patient_id": 1,
      "name": "John Doe",
      "last_visit": "2024-05-10",
      "risk_level": "high",
      "next_appointment": "2024-05-20"
    }
  ]
}
```

### 32. Review Prediction
```
POST /api/v1/doctors/predictions/44/review
Authorization: Bearer <token> (doctor only)
Content-Type: application/json

{
  "approved": true,
  "notes": "Results align with clinical findings"
}

Response (200):
{
  "message": "Prediction reviewed",
  "reviewed_at": "2024-05-14T12:00:00Z"
}
```

### 33. Add Clinical Notes
```
POST /api/v1/doctors/patients/1/notes
Authorization: Bearer <token> (doctor only)
Content-Type: application/json

{
  "notes": "Patient showing improvement...",
  "visit_date": "2024-05-14"
}

Response (201):
{
  "note_id": 456,
  "created_at": "2024-05-14T12:00:00Z"
}
```

---

## 🎯 Recommendation Endpoints

### 34. Get Recommendations
```
GET /api/v1/recommendations?category=diet
Authorization: Bearer <token>

Response (200):
{
  "items": [
    {
      "recommendation_id": 10,
      "title": "Reduce salt intake",
      "description": "Limit to 2g per day",
      "priority": "high",
      "category": "diet"
    }
  ]
}
```

### 35. Mark Recommendation Complete
```
PUT /api/v1/recommendations/10/complete
Authorization: Bearer <token>

Response (200):
{
  "message": "Recommendation marked as completed"
}
```

---

## 📱 Notification Endpoints

### 36. Get Notifications
```
GET /api/v1/notifications?unread=true
Authorization: Bearer <token>

Response (200):
{
  "total_unread": 3,
  "items": [
    {
      "notification_id": 789,
      "type": "alert",
      "title": "Follow-up due",
      "created_at": "2024-05-14T10:00:00Z"
    }
  ]
}
```

### 37. Mark Notification Read
```
PUT /api/v1/notifications/789/read
Authorization: Bearer <token>

Response (200):
{
  "message": "Marked as read"
}
```

---

## ⚙️ Admin Endpoints

### 38. Get System Statistics
```
GET /api/v1/admin/statistics
Authorization: Bearer <token> (admin only)

Response (200):
{
  "total_users": 1250,
  "total_patients": 980,
  "total_doctors": 45,
  "total_predictions": 15420,
  "system_uptime_percent": 99.95,
  "avg_response_time_ms": 245
}
```

### 39. List All Users
```
GET /api/v1/admin/users?page=1&limit=50&role=patient
Authorization: Bearer <token> (admin only)

Response (200):
{
  "total": 980,
  "items": [...]
}
```

### 40. Manage Users (Activate/Deactivate)
```
PUT /api/v1/admin/users/1/status
Authorization: Bearer <token> (admin only)
Content-Type: application/json

{
  "is_active": false,
  "reason": "Inactivity"
}

Response (200):
{
  "message": "User status updated"
}
```

---

## 🔄 Appointment Endpoints

### 41. Schedule Appointment
```
POST /api/v1/appointments/schedule
Authorization: Bearer <token>
Content-Type: application/json

{
  "doctor_id": 5,
  "scheduled_datetime": "2024-05-20T14:30:00Z",
  "consultation_type": "follow_up",
  "reason": "Review previous prediction"
}

Response (201):
{
  "appointment_id": 100,
  "status": "scheduled",
  "confirmation_id": "APPT_20240520_001"
}
```

### 42. Get Appointments
```
GET /api/v1/appointments?status=scheduled
Authorization: Bearer <token>

Response (200):
{
  "items": [
    {
      "appointment_id": 100,
      "doctor_name": "Dr. Smith",
      "scheduled_datetime": "2024-05-20T14:30:00Z",
      "status": "scheduled"
    }
  ]
}
```

### 43. Cancel Appointment
```
DELETE /api/v1/appointments/100
Authorization: Bearer <token>

Response (200):
{
  "message": "Appointment cancelled"
}
```

---

## 📝 Error Responses

### Standard Error Format
```
{
  "error": {
    "code": "PREDICTION_FAILED",
    "message": "Image preprocessing failed",
    "details": "Invalid DICOM file format",
    "timestamp": "2024-05-14T11:00:00Z",
    "request_id": "req_xyz_123"
  }
}
```

### Common HTTP Status Codes
- **200:** Success
- **201:** Created
- **202:** Accepted (async processing)
- **400:** Bad Request
- **401:** Unauthorized
- **403:** Forbidden
- **404:** Not Found
- **409:** Conflict
- **429:** Rate Limited
- **500:** Internal Server Error

---

## 🔒 Security Headers

All responses include:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000
```

---

## 📚 API Versioning

Current Version: **v1**
Base URL: `https://api.liverhealth.com/api/v1`

Future versions available under `/api/v2/`, `/api/v3/`, etc.

---

## 🧪 Testing Endpoints

**Testing Base URL:** `https://api-test.liverhealth.com/api/v1`

All endpoints available with test data in non-production environment.

---

**For detailed implementation, see** `backend/app/api/v1/` directory
