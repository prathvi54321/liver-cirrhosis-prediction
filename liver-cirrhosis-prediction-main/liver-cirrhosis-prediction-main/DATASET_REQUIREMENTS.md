# Dataset Requirements
## Medical Imaging and Clinical Data Specifications

---

## 📊 Overview

This document outlines the dataset requirements for training and validating the Explainable AI-Based Non-Invasive Liver Cirrhosis Detection System. The system requires both medical imaging data and clinical/symptom data for comprehensive model training.

---

## 🏥 Medical Imaging Datasets

### 1. Primary Imaging Modalities

#### Ultrasound Liver Images
**Requirements:**
- **Format:** DICOM (.dcm) or PNG/JPG (minimum 512x512 pixels)
- **Resolution:** Minimum 256x256 pixels, preferred 512x512 or higher
- **Color Mode:** Grayscale (8-bit) for ultrasound
- **Annotations:** Liver region segmentation masks
- **Classes:** Normal, Fatty Liver, Cirrhosis (Stage 1-4)

**Dataset Specifications:**
```python
ULTRASOUND_DATASET = {
    'total_images': 5000,
    'train_split': 0.7,    # 3500 images
    'val_split': 0.15,     # 750 images
    'test_split': 0.15,    # 750 images
    'class_distribution': {
        'normal': 1500,
        'fatty_liver': 1500,
        'cirrhosis_stage1': 750,
        'cirrhosis_stage2': 625,
        'cirrhosis_stage3': 375,
        'cirrhosis_stage4': 250
    },
    'image_properties': {
        'min_resolution': (256, 256),
        'preferred_resolution': (512, 512),
        'color_mode': 'grayscale',
        'bit_depth': 8
    }
}
```

#### CT Scan Liver Images
**Requirements:**
- **Format:** DICOM (.dcm) or NIfTI (.nii)
- **Resolution:** Minimum 256x256 pixels per slice
- **Color Mode:** Grayscale (12-16 bit)
- **Volume:** Multi-slice volumes (50-200 slices per patient)
- **Annotations:** 3D liver segmentation masks

**Dataset Specifications:**
```python
CT_SCAN_DATASET = {
    'total_volumes': 800,
    'total_slices': 80000,  # ~100 slices per volume
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'class_distribution': {
        'normal': 300,
        'fatty_liver': 200,
        'cirrhosis_stage1': 120,
        'cirrhosis_stage2': 100,
        'cirrhosis_stage3': 50,
        'cirrhosis_stage4': 30
    }
}
```

#### MRI Liver Images
**Requirements:**
- **Format:** DICOM or NIfTI
- **Sequences:** T1-weighted, T2-weighted, DWI
- **Resolution:** Minimum 256x256 pixels
- **Annotations:** Liver segmentation and fibrosis region marking

### 2. Image Annotation Standards

#### Segmentation Masks
```python
ANNOTATION_REQUIREMENTS = {
    'format': 'PNG',  # or NIfTI for 3D
    'color_coding': {
        'background': [0, 0, 0],      # Black
        'liver': [255, 255, 255],     # White
        'fibrosis': [255, 0, 0],      # Red
        'tumor': [0, 255, 0],         # Green
        'cyst': [0, 0, 255]           # Blue
    },
    'annotation_tool': 'ITK-SNAP or 3D Slicer',
    'inter_annotator_agreement': '>0.85 Dice coefficient'
}
```

#### Classification Labels
```python
CLASSIFICATION_LABELS = {
    'stage_0': {
        'description': 'Normal liver',
        'fibrosis_score': 0,
        'clinical_features': ['normal_architecture', 'no_fibrosis']
    },
    'stage_1': {
        'description': 'Mild fibrosis',
        'fibrosis_score': 1,
        'clinical_features': ['portal_fibrosis', 'minimal_distortion']
    },
    'stage_2': {
        'description': 'Moderate fibrosis',
        'fibrosis_score': 2,
        'clinical_features': ['septal_fibrosis', 'architecture_distortion']
    },
    'stage_3': {
        'description': 'Severe fibrosis',
        'fibrosis_score': 3,
        'clinical_features': ['bridging_fibrosis', 'marked_distortion']
    },
    'stage_4': {
        'description': 'Cirrhosis',
        'fibrosis_score': 4,
        'clinical_features': ['regenerative_nodules', 'severe_distortion']
    }
}
```

---

## 📋 Clinical and Symptom Data

### 1. Tabular Dataset Structure

#### Core Features (22 features)
```python
CLINICAL_FEATURES = {
    # Demographic
    'age': {'type': 'numeric', 'range': [18, 100], 'unit': 'years'},
    'sex': {'type': 'categorical', 'values': [0, 1], 'labels': ['female', 'male']},

    # Symptoms (1-10 scale)
    'fatigue_level': {'type': 'ordinal', 'range': [1, 10], 'description': 'Fatigue severity'},
    'alcohol_consumption': {'type': 'numeric', 'range': [0, 50], 'unit': 'drinks/week'},

    # Physical symptoms
    'weight_loss_kg': {'type': 'numeric', 'range': [0, 20], 'unit': 'kg'},
    'abdominal_swelling': {'type': 'binary', 'values': [0, 1]},
    'appetite_loss': {'type': 'binary', 'values': [0, 1]},
    'jaundice': {'type': 'binary', 'values': [0, 1]},
    'fever': {'type': 'binary', 'values': [0, 1]},

    # Clinical examination
    'ascites': {'type': 'ordinal', 'range': [0, 3], 'description': 'Ascites severity'},
    'hepatomegaly': {'type': 'ordinal', 'range': [0, 3], 'description': 'Liver enlargement'},
    'spiders': {'type': 'ordinal', 'range': [0, 3], 'description': 'Spider angiomas'},
    'edema': {'type': 'ordinal', 'range': [0, 3], 'description': 'Peripheral edema'},

    # Laboratory tests
    'bilirubin': {'type': 'numeric', 'range': [0.1, 20.0], 'unit': 'mg/dL'},
    'cholesterol': {'type': 'numeric', 'range': [100, 400], 'unit': 'mg/dL'},
    'albumin': {'type': 'numeric', 'range': [1.0, 5.0], 'unit': 'g/dL'},
    'copper': {'type': 'numeric', 'range': [10, 200], 'unit': 'μg/dL'},
    'alk_phos': {'type': 'numeric', 'range': [40, 500], 'unit': 'IU/L'},
    'ast': {'type': 'numeric', 'range': [10, 1000], 'unit': 'IU/L'},
    'triglycerides': {'type': 'numeric', 'range': [50, 500], 'unit': 'mg/dL'},
    'platelets': {'type': 'numeric', 'range': [50000, 500000], 'unit': '×10^9/L'},
    'prothrombin': {'type': 'numeric', 'range': [8, 20], 'unit': 'seconds'}
}
```

#### Dataset Size Requirements
```python
CLINICAL_DATASET = {
    'total_samples': 10000,
    'train_split': 0.7,    # 7000 samples
    'val_split': 0.15,     # 1500 samples
    'test_split': 0.15,    # 1500 samples
    'class_distribution': {
        'stage_0': 3000,   # Normal
        'stage_1': 2500,   # Mild fibrosis
        'stage_2': 2000,   # Moderate fibrosis
        'stage_3': 1500,   # Severe fibrosis
        'stage_4': 1000    # Cirrhosis
    },
    'missing_data_tolerance': 0.1,  # Maximum 10% missing values
    'outlier_handling': 'IQR method with 1.5 multiplier'
}
```

### 2. Data Quality Standards

#### Missing Data Handling
```python
MISSING_DATA_STRATEGY = {
    'numerical_features': 'median_imputation',
    'categorical_features': 'mode_imputation',
    'maximum_missing_percentage': 15,
    'advanced_imputation': {
        'method': 'iterative_imputer',
        'estimator': 'RandomForestRegressor',
        'max_iter': 10
    }
}
```

#### Outlier Detection and Treatment
```python
OUTLIER_HANDLING = {
    'detection_methods': [
        'IQR_method',
        'z_score_method',
        'isolation_forest'
    ],
    'treatment_strategies': [
        'clipping',
        'removal',
        'transformation'
    ],
    'clinical_validation': True  # Domain expert review required
}
```

---

## 🔄 Data Preprocessing Pipeline

### 1. Image Preprocessing

```python
class MedicalImagePreprocessor:
    def __init__(self):
        self.target_size = (256, 256)
        self.normalization = 'z_score'  # or 'min_max'

    def preprocess_ultrasound(self, image_path):
        """Preprocess ultrasound image"""
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize
        image = cv2.resize(image, self.target_size)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Normalization
        if self.normalization == 'z_score':
            image = (image - np.mean(image)) / np.std(image)
        elif self.normalization == 'min_max':
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        return image

    def preprocess_ct_scan(self, volume_path):
        """Preprocess CT scan volume"""
        # Load DICOM volume
        volume = self.load_dicom_volume(volume_path)

        # Windowing (liver window: 40-400 HU)
        volume = np.clip(volume, 40, 400)

        # Normalization to [0, 1]
        volume = (volume - 40) / (400 - 40)

        # Resize slices
        resized_volume = []
        for slice in volume:
            resized_slice = cv2.resize(slice, self.target_size)
            resized_volume.append(resized_slice)

        return np.array(resized_volume)

    def augment_image(self, image):
        """Apply data augmentation"""
        augmented = []

        # Original
        augmented.append(image)

        # Horizontal flip
        augmented.append(cv2.flip(image, 1))

        # Vertical flip
        augmented.append(cv2.flip(image, 0))

        # Rotation (slight)
        for angle in [-10, 10]:
            matrix = cv2.getRotationMatrix2D((image.shape[1]/2, image.shape[0]/2), angle, 1)
            rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            augmented.append(rotated)

        return augmented
```

### 2. Clinical Data Preprocessing

```python
class ClinicalDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoder = LabelEncoder()

    def preprocess_dataset(self, df):
        """Complete preprocessing pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical variables
        df = self.encode_categorical(df)

        # Create derived features
        df = self.create_derived_features(df)

        # Handle outliers
        df = self.handle_outliers(df)

        # Scale features
        df = self.scale_features(df)

        return df

    def handle_missing_values(self, df):
        """Handle missing values appropriately"""
        # Clinical thresholds for imputation
        imputation_rules = {
            'bilirubin': 'median',
            'albumin': 'median',
            'platelets': 'median',
            'age': 'median',
            'sex': 'mode'
        }

        for column, method in imputation_rules.items():
            if column in df.columns:
                if method == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif method == 'mode':
                    df[column].fillna(df[column].mode()[0], inplace=True)

        return df

    def create_derived_features(self, df):
        """Create clinically relevant derived features"""
        # Liver function ratios
        df['bilirubin_albumin_ratio'] = df['bilirubin'] / (df['albumin'] + 1e-6)
        df['ast_alt_ratio'] = df['ast'] / (df.get('alt', df['ast']) + 1e-6)

        # Platelet ratios
        df['platelet_albumin_score'] = df['platelets'] * df['albumin']

        # Categorize continuous variables
        df['bilirubin_category'] = pd.cut(df['bilirubin'],
                                        bins=[0, 1.2, 2.0, 5.0, np.inf],
                                        labels=['normal', 'mild', 'moderate', 'severe'])

        df['albumin_category'] = pd.cut(df['albumin'],
                                      bins=[0, 3.5, 2.8, 2.0, np.inf],
                                      labels=['severe', 'moderate', 'mild', 'normal'])

        return df

    def handle_outliers(self, df):
        """Handle outliers using clinical knowledge"""
        # Define clinical ranges
        clinical_ranges = {
            'bilirubin': (0.1, 20.0),
            'albumin': (1.0, 5.0),
            'platelets': (50000, 500000),
            'age': (18, 100)
        }

        for column, (min_val, max_val) in clinical_ranges.items():
            if column in df.columns:
                df[column] = df[column].clip(min_val, max_val)

        return df
```

---

## 📊 Data Sources and Acquisition

### 1. Public Datasets

#### Medical Imaging Datasets
```python
PUBLIC_DATASETS = {
    'liver_ultrasound': {
        'source': 'Medical Segmentation Decathlon',
        'url': 'https://decathlon-10.grand-challenge.org/',
        'size': '200+ liver ultrasound volumes',
        'format': 'NIfTI',
        'license': 'CC BY-SA 4.0'
    },
    'chaos_challenge': {
        'source': 'CHAOS Challenge',
        'url': 'https://chaos.grand-challenge.org/',
        'modalities': ['CT', 'MRI'],
        'organs': ['Liver', 'Kidney', 'Spleen'],
        'size': '80+ volumes'
    },
    'lits_challenge': {
        'source': 'LiTS - Liver Tumor Segmentation Challenge',
        'url': 'https://competitions.codalab.org/competitions/17094',
        'size': '131 CT volumes',
        'annotations': 'liver_masks, tumor_masks'
    },
    'sliver_dataset': {
        'source': 'SLIVER - Segmentation of Liver in VOlumetric Images',
        'url': 'https://sliver07.grand-challenge.org/',
        'size': '20 CT volumes',
        'annotations': 'manual_segmentations'
    }
}
```

#### Clinical Datasets
```python
CLINICAL_DATASETS = {
    'indian_liver_patient': {
        'source': 'UCI Machine Learning Repository',
        'url': 'https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)',
        'size': '583 samples',
        'features': 10,
        'classes': ['liver_disease', 'normal']
    },
    'cirrhosis_patient_survival': {
        'source': 'Kaggle',
        'url': 'https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction',
        'size': '418 samples',
        'features': 18,
        'target': 'survival_prediction'
    }
}
```

### 2. Data Acquisition Strategy

#### Institutional Partnerships
```python
DATA_ACQUISITION_PLAN = {
    'hospitals': [
        'Apollo Hospitals',
        'AIIMS Delhi',
        'Christian Medical College',
        'Tata Memorial Hospital'
    ],
    'data_types': [
        'ultrasound_images',
        'ct_scans',
        'mri_scans',
        'clinical_records',
        'laboratory_results'
    ],
    'ethical_approval': {
        'irb_committee': True,
        'patient_consent': True,
        'data_anonymization': True,
        'hipaa_compliance': True
    },
    'data_volume_target': {
        'year_1': 5000,
        'year_2': 15000,
        'year_3': 30000
    }
}
```

#### Synthetic Data Generation
```python
SYNTHETIC_DATA_STRATEGY = {
    'gan_based': {
        'method': 'StyleGAN2 or Progressive GAN',
        'purpose': 'Generate diverse ultrasound images',
        'validation': 'Domain expert review required'
    },
    'physics_based': {
        'method': 'Finite element modeling',
        'purpose': 'Simulate liver deformation',
        'validation': 'Clinical correlation studies'
    },
    'hybrid_approach': {
        'method': 'GAN + Clinical constraints',
        'purpose': 'Generate realistic clinical scenarios',
        'validation': 'Statistical similarity testing'
    }
}
```

---

## 🔒 Data Privacy and Ethics

### 1. HIPAA and GDPR Compliance

```python
COMPLIANCE_REQUIREMENTS = {
    'data_anonymization': {
        'methods': ['deidentification', 'pseudonymization'],
        'phi_removal': ['names', 'dates', 'locations', 'contact_info'],
        'retention_policy': '7 years post-treatment'
    },
    'access_control': {
        'role_based_access': True,
        'audit_logging': True,
        'encryption_at_rest': True,
        'encryption_in_transit': True
    },
    'consent_management': {
        'informed_consent': True,
        'withdrawal_rights': True,
        'purpose_limitation': True
    }
}
```

### 2. Data Quality Assurance

```python
QUALITY_ASSURANCE = {
    'annotation_quality': {
        'inter_annotator_agreement': '>0.85',
        'expert_validation': True,
        'quality_control_checks': ['completeness', 'accuracy', 'consistency']
    },
    'data_integrity': {
        'checksums': True,
        'backup_redundancy': True,
        'corruption_detection': True
    },
    'bias_detection': {
        'demographic_analysis': True,
        'performance_fairness': True,
        'bias_mitigation_strategies': ['reweighting', 'augmentation', 'fair_representation']
    }
}
```

---

## 📈 Dataset Versioning and Management

### 1. Data Version Control

```python
DATA_VERSIONING = {
    'version_format': 'v{major}.{minor}.{patch}',
    'metadata_tracking': {
        'creation_date': True,
        'source_institution': True,
        'annotation_version': True,
        'preprocessing_version': True
    },
    'change_log': {
        'additions': 'New samples added',
        'corrections': 'Annotation corrections',
        'removals': 'Invalid samples removed'
    }
}
```

### 2. Dataset Storage Architecture

```python
DATA_STORAGE = {
    'raw_data': {
        'format': 'DICOM/NIfTI/CSV',
        'storage': 'AWS S3 with versioning',
        'backup': 'Cross-region replication',
        'retention': '10 years'
    },
    'processed_data': {
        'format': 'HDF5/Parquet/TFRecords',
        'storage': 'High-performance storage',
        'caching': 'Redis for frequent access'
    },
    'metadata': {
        'format': 'JSON/YAML',
        'storage': 'PostgreSQL with indexing',
        'search': 'Elasticsearch integration'
    }
}
```

---

## 🎯 Minimum Viable Dataset

### For Initial Model Development
```python
MINIMUM_VIABLE_DATASET = {
    'clinical_data': {
        'samples': 2000,
        'features': 22,
        'class_balance': 'stratified_split'
    },
    'ultrasound_images': {
        'samples': 1000,
        'resolution': '256x256',
        'annotations': 'basic_segmentation'
    },
    'validation_metrics': {
        'accuracy_target': 0.75,
        'auc_target': 0.80,
        'clinical_validation': True
    }
}
```

### Scaling Strategy
```python
DATA_SCALING_STRATEGY = {
    'phase_1': 'MVP dataset (2000 clinical + 1000 images)',
    'phase_2': 'Expanded dataset (5000 clinical + 3000 images)',
    'phase_3': 'Full dataset (10000+ clinical + 5000+ images)',
    'continuous_improvement': 'Active learning + new data acquisition'
}
```

---

**For data acquisition and preprocessing implementation, see** `ai_pipeline/data_preprocessing.py` and `ai_pipeline/dataset_manager.py`
