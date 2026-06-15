# Training Pipeline
## End-to-End Model Training and Validation

---

## 📊 Overview

This document outlines the complete training pipeline for the Explainable AI-Based Non-Invasive Liver Cirrhosis Detection System, covering data preparation, model training, validation, and deployment.

---

## 🔄 Complete Training Workflow

### 1. Pipeline Architecture

```python
class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model_factory = ModelFactory(config)
        self.trainer = ModelTrainer(config)
        self.validator = ModelValidator(config)
        self.explainer = XAIExplainer(config)

    def run_pipeline(self):
        """Execute complete training pipeline"""
        # Phase 1: Data Preparation
        datasets = self.data_manager.prepare_datasets()

        # Phase 2: Model Training
        models = self.model_factory.create_models()
        trained_models = self.trainer.train_models(models, datasets)

        # Phase 3: Model Validation
        validation_results = self.validator.validate_models(trained_models, datasets)

        # Phase 4: Explainability Integration
        explainers = self.explainer.create_explainers(trained_models)

        # Phase 5: Model Deployment Preparation
        deployment_package = self.prepare_deployment_package(
            trained_models, explainers, validation_results
        )

        return deployment_package
```

---

## 📥 Data Preparation Phase

### 1. Data Loading and Validation

```python
class DataManager:
    def __init__(self, config):
        self.config = config
        self.image_processor = MedicalImageProcessor()
        self.clinical_processor = ClinicalDataProcessor()

    def prepare_datasets(self):
        """Prepare all datasets for training"""
        # Load clinical data
        clinical_data = self.load_clinical_data()

        # Load imaging data
        imaging_data = self.load_imaging_data()

        # Validate data quality
        self.validate_data_quality(clinical_data, imaging_data)

        # Create data splits
        splits = self.create_train_val_test_splits(clinical_data, imaging_data)

        # Apply preprocessing
        processed_splits = self.apply_preprocessing(splits)

        return processed_splits

    def load_clinical_data(self):
        """Load and validate clinical dataset"""
        df = pd.read_csv(self.config.clinical_data_path)

        # Validate required columns
        required_columns = CLINICAL_FEATURES.keys()
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Basic data validation
        self.validate_clinical_data(df)

        return df

    def load_imaging_data(self):
        """Load and organize imaging data"""
        image_paths = []
        labels = []

        for class_name in CLASSIFICATION_LABELS.keys():
            class_path = os.path.join(self.config.imaging_data_path, class_name)
            if os.path.exists(class_path):
                for image_file in os.listdir(class_path):
                    if image_file.endswith(('.png', '.jpg', '.dcm')):
                        image_paths.append(os.path.join(class_path, image_file))
                        labels.append(class_name)

        return {
            'image_paths': image_paths,
            'labels': labels,
            'label_encoder': self.create_label_encoder(labels)
        }

    def create_train_val_test_splits(self, clinical_data, imaging_data):
        """Create stratified train/validation/test splits"""
        from sklearn.model_selection import train_test_split

        # Clinical data split
        clinical_train, clinical_temp = train_test_split(
            clinical_data, test_size=0.3, stratify=clinical_data['stage'],
            random_state=self.config.random_seed
        )
        clinical_val, clinical_test = train_test_split(
            clinical_temp, test_size=0.5, stratify=clinical_temp['stage'],
            random_state=self.config.random_seed
        )

        # Imaging data split (maintaining class balance)
        train_images, temp_images, train_labels, temp_labels = train_test_split(
            imaging_data['image_paths'], imaging_data['labels'],
            test_size=0.3, stratify=imaging_data['labels'],
            random_state=self.config.random_seed
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            temp_images, temp_labels, test_size=0.5,
            stratify=temp_labels, random_state=self.config.random_seed
        )

        return {
            'clinical': {
                'train': clinical_train,
                'val': clinical_val,
                'test': clinical_test
            },
            'imaging': {
                'train': {'paths': train_images, 'labels': train_labels},
                'val': {'paths': val_images, 'labels': val_labels},
                'test': {'paths': test_images, 'labels': test_labels}
            }
        }
```

### 2. Data Preprocessing

```python
class MedicalImageProcessor:
    def __init__(self):
        self.target_size = (256, 256)
        self.augmentation = self.setup_augmentation()

    def setup_augmentation(self):
        """Setup data augmentation pipeline"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_batch(self, image_paths, labels, training=True):
        """Preprocess batch of images"""
        processed_images = []
        processed_labels = []

        for image_path, label in zip(image_paths, labels):
            # Load and preprocess image
            image = self.load_and_preprocess_image(image_path)

            if training:
                # Apply augmentation
                image = self.augmentation(image)

            processed_images.append(image)
            processed_labels.append(label)

        return torch.stack(processed_images), torch.tensor(processed_labels)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess single image"""
        if image_path.endswith('.dcm'):
            # Handle DICOM files
            import pydicom
            dicom = pydicom.dcmread(image_path)
            image = dicom.pixel_array.astype(np.float32)
        else:
            # Handle regular images
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32)

        # Resize
        image = cv2.resize(image, self.target_size)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image.astype(np.uint8))

        # Convert to tensor
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0

        return image
```

---

## 🏭 Model Training Phase

### 1. Model Factory

```python
class ModelFactory:
    def __init__(self, config):
        self.config = config

    def create_models(self):
        """Create all required models"""
        models = {}

        # Traditional ML models
        models['rf'] = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=self.config.random_seed
        )

        models['xgb'] = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            random_state=self.config.random_seed
        )

        models['svm'] = SVC(
            kernel=self.config.svm_kernel,
            C=self.config.svm_c,
            probability=True,
            random_state=self.config.random_seed
        )

        # Deep Learning models
        models['cnn'] = self.create_cnn_model()
        models['resnet'] = self.create_resnet_model()
        models['efficientnet'] = self.create_efficientnet_model()

        return models

    def create_cnn_model(self):
        """Create custom CNN model"""
        return CustomCNN(
            num_classes=self.config.num_classes,
            input_channels=1,  # Grayscale
            dropout_rate=self.config.cnn_dropout
        )

    def create_resnet_model(self):
        """Create ResNet model"""
        model = models.resnet50(pretrained=True)

        # Modify first layer for grayscale
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(self.config.resnet_dropout),
            nn.Linear(512, self.config.num_classes)
        )

        return model

    def create_efficientnet_model(self):
        """Create EfficientNet model"""
        model = EfficientNet.from_pretrained('efficientnet-b0')

        # Modify for grayscale input
        model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Modify classifier
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(self.config.efficientnet_dropout),
            nn.Linear(512, self.config.num_classes)
        )

        return model
```

### 2. Model Trainer

```python
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_models(self, models, datasets):
        """Train all models"""
        trained_models = {}

        # Train traditional ML models
        ml_models = {k: v for k, v in models.items() if k in ['rf', 'xgb', 'svm']}
        trained_models.update(self.train_ml_models(ml_models, datasets['clinical']))

        # Train deep learning models
        dl_models = {k: v for k, v in models.items() if k in ['cnn', 'resnet', 'efficientnet']}
        trained_models.update(self.train_dl_models(dl_models, datasets['imaging']))

        return trained_models

    def train_ml_models(self, models, clinical_data):
        """Train traditional ML models"""
        trained_models = {}

        # Prepare training data
        X_train = clinical_data['train'].drop('stage', axis=1)
        y_train = clinical_data['train']['stage']

        # Preprocess features
        preprocessor = ClinicalDataPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Train model
            model.fit(X_train_processed, y_train)

            # Save model
            self.save_ml_model(model, model_name, preprocessor)

            trained_models[model_name] = {
                'model': model,
                'preprocessor': preprocessor,
                'training_date': datetime.now()
            }

        return trained_models

    def train_dl_models(self, models, imaging_data):
        """Train deep learning models"""
        trained_models = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Create data loaders
            train_loader = self.create_data_loader(
                imaging_data['train']['paths'],
                imaging_data['train']['labels'],
                training=True
            )

            val_loader = self.create_data_loader(
                imaging_data['val']['paths'],
                imaging_data['val']['labels'],
                training=False
            )

            # Train model
            trained_model = self.train_single_dl_model(model, train_loader, val_loader)

            # Save model
            self.save_dl_model(trained_model, model_name)

            trained_models[model_name] = {
                'model': trained_model,
                'training_date': datetime.now(),
                'best_val_accuracy': self.get_best_val_accuracy(trained_model)
            }

        return trained_models

    def train_single_dl_model(self, model, train_loader, val_loader):
        """Train single deep learning model"""
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()

            # Calculate metrics
            train_acc = train_correct / len(train_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            scheduler.step()

        # Load best model state
        model.load_state_dict(best_model_state)
        return model
```

---

## ✅ Model Validation Phase

### 1. Comprehensive Validation

```python
class ModelValidator:
    def __init__(self, config):
        self.config = config
        self.metrics_calculator = MetricsCalculator()

    def validate_models(self, trained_models, datasets):
        """Comprehensive model validation"""
        validation_results = {}

        # Validate ML models
        ml_models = {k: v for k, v in trained_models.items() if k in ['rf', 'xgb', 'svm']}
        validation_results['ml'] = self.validate_ml_models(ml_models, datasets['clinical'])

        # Validate DL models
        dl_models = {k: v for k, v in trained_models.items() if k in ['cnn', 'resnet', 'efficientnet']}
        validation_results['dl'] = self.validate_dl_models(dl_models, datasets['imaging'])

        # Cross-validation
        validation_results['cross_validation'] = self.perform_cross_validation(
            trained_models, datasets
        )

        # Ensemble validation
        validation_results['ensemble'] = self.validate_ensemble_models(
            trained_models, datasets
        )

        return validation_results

    def validate_ml_models(self, models, clinical_data):
        """Validate traditional ML models"""
        results = {}

        # Prepare test data
        X_test = clinical_data['test'].drop('stage', axis=1)
        y_test = clinical_data['test']['stage']

        for model_name, model_info in models.items():
            model = model_info['model']
            preprocessor = model_info['preprocessor']

            # Preprocess test data
            X_test_processed = preprocessor.transform(X_test)

            # Make predictions
            y_pred = model.predict(X_test_processed)
            y_proba = model.predict_proba(X_test_processed)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_classification_metrics(
                y_test, y_pred, y_proba
            )

            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba,
                'feature_importance': self.get_feature_importance(model, model_name)
            }

        return results

    def validate_dl_models(self, models, imaging_data):
        """Validate deep learning models"""
        results = {}

        # Create test data loader
        test_loader = self.create_test_data_loader(
            imaging_data['test']['paths'],
            imaging_data['test']['labels']
        )

        for model_name, model_info in models.items():
            model = model_info['model']

            # Evaluate on test set
            test_metrics = self.evaluate_dl_model(model, test_loader)

            results[model_name] = {
                'metrics': test_metrics,
                'confusion_matrix': test_metrics['confusion_matrix'],
                'roc_curves': test_metrics['roc_curves']
            }

        return results

    def perform_cross_validation(self, trained_models, datasets):
        """Perform cross-validation for robustness assessment"""
        cv_results = {}

        # 5-fold cross-validation for ML models
        ml_models = {k: v for k, v in trained_models.items() if k in ['rf', 'xgb', 'svm']}

        for model_name, model_info in ml_models.items():
            model = model_info['model']
            X = datasets['clinical']['train'].drop('stage', axis=1)
            y = datasets['clinical']['train']['stage']

            cv_scores = cross_val_score(
                model, X, y, cv=5,
                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            )

            cv_results[model_name] = {
                'accuracy': {
                    'mean': cv_scores['test_accuracy'].mean(),
                    'std': cv_scores['test_accuracy'].std()
                },
                'precision': {
                    'mean': cv_scores['test_precision_weighted'].mean(),
                    'std': cv_scores['test_precision_weighted'].std()
                },
                'recall': {
                    'mean': cv_scores['test_recall_weighted'].mean(),
                    'std': cv_scores['test_recall_weighted'].std()
                },
                'f1': {
                    'mean': cv_scores['test_f1_weighted'].mean(),
                    'std': cv_scores['test_f1_weighted'].std()
                }
            }

        return cv_results
```

### 2. Metrics Calculator

```python
class MetricsCalculator:
    def calculate_classification_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }

        # Per-class metrics
        metrics['per_class'] = classification_report(y_true, y_pred, output_dict=True)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                if y_proba.shape[1] > 2:  # Multi-class
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                else:  # Binary
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['auc_roc'] = None

        return metrics

    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression metrics for continuous predictions"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
```

---

## 🔍 Explainability Integration

### 1. XAI Explainer

```python
class XAIExplainer:
    def __init__(self, config):
        self.config = config

    def create_explainers(self, trained_models):
        """Create explainers for all trained models"""
        explainers = {}

        # SHAP explainers for ML models
        ml_models = {k: v for k, v in trained_models.items() if k in ['rf', 'xgb', 'svm']}
        explainers['ml'] = self.create_ml_explainers(ml_models)

        # Grad-CAM explainers for DL models
        dl_models = {k: v for k, v in trained_models.items() if k in ['cnn', 'resnet', 'efficientnet']}
        explainers['dl'] = self.create_dl_explainers(dl_models)

        return explainers

    def create_ml_explainers(self, ml_models):
        """Create SHAP explainers for ML models"""
        explainers = {}

        for model_name, model_info in ml_models.items():
            model = model_info['model']

            if model_name == 'xgb':
                explainer = shap.TreeExplainer(model)
            elif model_name == 'rf':
                explainer = shap.TreeExplainer(model)
            else:  # SVM
                # For SVM, use KernelExplainer with background data
                background_data = shap.sample(model_info['training_data'], 100)
                explainer = shap.KernelExplainer(model.predict_proba, background_data)

            explainers[model_name] = {
                'explainer': explainer,
                'model_info': model_info
            }

        return explainers

    def create_dl_explainers(self, dl_models):
        """Create Grad-CAM explainers for DL models"""
        explainers = {}

        for model_name, model_info in dl_models.items():
            model = model_info['model']

            # Determine target layer for Grad-CAM
            if model_name == 'resnet':
                target_layer = model.layer4[-1]
            elif model_name == 'efficientnet':
                target_layer = model._blocks[-1]
            else:  # Custom CNN
                target_layer = model.conv4

            explainer = GradCAMExplainer(model, target_layer)

            explainers[model_name] = {
                'explainer': explainer,
                'target_layer': target_layer,
                'model_info': model_info
            }

        return explainers
```

---

## 📦 Deployment Preparation

### 1. Model Packaging

```python
class DeploymentPreparer:
    def __init__(self, config):
        self.config = config

    def prepare_deployment_package(self, trained_models, explainers, validation_results):
        """Prepare complete deployment package"""
        deployment_package = {
            'models': {},
            'explainers': {},
            'validation_results': validation_results,
            'metadata': self.create_metadata(),
            'config': self.config,
            'version': self.config.model_version
        }

        # Package ML models
        for model_name, model_info in trained_models.items():
            if model_name in ['rf', 'xgb', 'svm']:
                deployment_package['models'][model_name] = self.package_ml_model(
                    model_info, model_name
                )

        # Package DL models
        for model_name, model_info in trained_models.items():
            if model_name in ['cnn', 'resnet', 'efficientnet']:
                deployment_package['models'][model_name] = self.package_dl_model(
                    model_info, model_name
                )

        # Package explainers
        deployment_package['explainers'] = explainers

        # Save deployment package
        self.save_deployment_package(deployment_package)

        return deployment_package

    def package_ml_model(self, model_info, model_name):
        """Package ML model for deployment"""
        return {
            'model': model_info['model'],
            'preprocessor': model_info['preprocessor'],
            'feature_names': model_info.get('feature_names', []),
            'training_date': model_info['training_date'],
            'model_type': 'ml',
            'framework': 'scikit-learn'
        }

    def package_dl_model(self, model_info, model_name):
        """Package DL model for deployment"""
        return {
            'model': model_info['model'],
            'model_state_dict': model_info['model'].state_dict(),
            'input_shape': self.config.input_shape,
            'training_date': model_info['training_date'],
            'model_type': 'dl',
            'framework': 'pytorch'
        }

    def create_metadata(self):
        """Create deployment metadata"""
        return {
            'deployment_date': datetime.now(),
            'model_version': self.config.model_version,
            'training_config': {
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.num_epochs,
                'optimizer': self.config.optimizer,
                'loss_function': self.config.loss_function
            },
            'data_info': {
                'training_samples': self.config.training_samples,
                'validation_samples': self.config.validation_samples,
                'test_samples': self.config.test_samples,
                'feature_count': self.config.feature_count
            },
            'performance_summary': {
                'best_accuracy': self.config.best_accuracy,
                'best_f1_score': self.config.best_f1_score,
                'auc_roc': self.config.auc_roc
            }
        }

    def save_deployment_package(self, deployment_package):
        """Save deployment package to disk"""
        import pickle

        # Save models separately (large files)
        for model_name, model_data in deployment_package['models'].items():
            if model_data['model_type'] == 'ml':
                joblib.dump(model_data['model'], f'models/{model_name}_model.pkl')
                joblib.dump(model_data['preprocessor'], f'models/{model_name}_preprocessor.pkl')
            else:  # DL model
                torch.save(model_data['model_state_dict'], f'models/{model_name}_model.pth')

        # Save metadata and config
        with open('models/deployment_metadata.pkl', 'wb') as f:
            pickle.dump({
                'metadata': deployment_package['metadata'],
                'validation_results': deployment_package['validation_results'],
                'explainers_config': deployment_package['explainers']
            }, f)

        # Save configuration
        with open('models/config.json', 'w') as f:
            json.dump(deployment_package['config'].__dict__, f, indent=2, default=str)
```

---

## 🔄 Pipeline Execution

### 1. Main Training Script

```python
def main():
    """Main training pipeline execution"""
    # Load configuration
    config = TrainingConfig()

    # Initialize pipeline
    pipeline = TrainingPipeline(config)

    try:
        # Run complete pipeline
        print("Starting training pipeline...")
        deployment_package = pipeline.run_pipeline()

        print("Training pipeline completed successfully!")
        print(f"Models saved to: models/")
        print(f"Best validation accuracy: {deployment_package['metadata']['performance_summary']['best_accuracy']:.4f}")

        # Generate training report
        generate_training_report(deployment_package)

    except Exception as e:
        print(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

### 2. Training Configuration

```python
class TrainingConfig:
    def __init__(self):
        # Data paths
        self.clinical_data_path = "data/clinical_data.csv"
        self.imaging_data_path = "data/imaging_data/"

        # Model hyperparameters
        self.rf_n_estimators = 500
        self.rf_max_depth = 15
        self.xgb_n_estimators = 300
        self.xgb_max_depth = 8
        self.xgb_learning_rate = 0.1
        self.svm_kernel = 'rbf'
        self.svm_c = 1.0

        # Training parameters
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.num_classes = 5  # stages 0-4

        # Data parameters
        self.random_seed = 42
        self.val_split = 0.15
        self.test_split = 0.15

        # Model version
        self.model_version = "1.0.0"
```

---

## 📊 Training Monitoring and Logging

### 1. Training Logger

```python
class TrainingLogger:
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.logger = logging.getLogger(__name__)

        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

    def log_epoch(self, epoch, train_metrics, val_metrics):
        """Log metrics for training epoch"""
        self.logger.info(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                        f"Train Acc={train_metrics['accuracy']:.4f}, "
                        f"Val Loss={val_metrics['loss']:.4f}, "
                        f"Val Acc={val_metrics['accuracy']:.4f}")

        # Store metrics
        self.metrics_history['train_loss'].append(train_metrics['loss'])
        self.metrics_history['val_loss'].append(val_metrics['loss'])
        self.metrics_history['train_accuracy'].append(train_metrics['accuracy'])
        self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])

    def log_model_save(self, model_name, val_accuracy):
        """Log model saving"""
        self.logger.info(f"Saved {model_name} model with validation accuracy: {val_accuracy:.4f}")

    def save_metrics_plot(self):
        """Save training curves plot"""
        plt.figure(figsize=(12, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics_history['train_loss'], label='Train Loss')
        plt.plot(self.metrics_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics_history['train_accuracy'], label='Train Accuracy')
        plt.plot(self.metrics_history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
```

---

## 🚨 Error Handling and Recovery

### 1. Pipeline Error Handler

```python
class PipelineErrorHandler:
    def __init__(self):
        self.errors = []
        self.checkpoints = {}

    def handle_error(self, stage, error, context=None):
        """Handle pipeline errors"""
        error_info = {
            'stage': stage,
            'error': str(error),
            'context': context,
            'timestamp': datetime.now(),
            'traceback': traceback.format_exc()
        }

        self.errors.append(error_info)

        # Log error
        logging.error(f"Pipeline error in {stage}: {str(error)}")

        # Attempt recovery based on stage
        if stage == 'data_preparation':
            return self.recover_data_preparation(error, context)
        elif stage == 'model_training':
            return self.recover_model_training(error, context)
        elif stage == 'validation':
            return self.recover_validation(error, context)

        return False  # Cannot recover

    def create_checkpoint(self, stage, data):
        """Create pipeline checkpoint"""
        checkpoint_file = f"checkpoints/{stage}_checkpoint.pkl"
        os.makedirs("checkpoints", exist_ok=True)

        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'stage': stage,
                'data': data,
                'timestamp': datetime.now()
            }, f)

        self.checkpoints[stage] = checkpoint_file

    def load_checkpoint(self, stage):
        """Load pipeline checkpoint"""
        checkpoint_file = self.checkpoints.get(stage)
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            return checkpoint['data']

        return None
```

---

**For implementation details, see** `ai_pipeline/train_pipeline.py` and `ai_pipeline/model_trainer.py`
