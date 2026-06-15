# Deep Learning Model Architecture
## CNN Models for Medical Image Analysis

---

## 📊 Overview

This document details the deep learning models used for medical image analysis in liver cirrhosis detection. The system employs multiple CNN architectures optimized for medical imaging tasks.

---

## 🏗️ CNN Architecture Components

### 1. Base CNN Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BaseCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(-1, 512 * 8 * 8)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

**Architecture Details:**
- Input: 256x256x3 RGB images
- Conv layers: 4 blocks with increasing filters (64→128→256→512)
- Pooling: MaxPool2d(2,2) after each conv block
- FC layers: 1024→512→4 (cirrhosis stages)
- Regularization: BatchNorm + Dropout(0.5)

### 2. ResNet50 Architecture

```python
import torchvision.models as models

class ResNet50Cirrhosis(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNet50Cirrhosis, self).__init__()

        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)

        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # Freeze early layers for fine-tuning
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
```

**Fine-tuning Strategy:**
- Freeze first 40 layers (up to conv4_x)
- Train last 10 layers + classifier
- Learning rate: 1e-4 for frozen layers, 1e-3 for trainable layers

### 3. EfficientNetB0 Architecture

```python
import efficientnet_pytorch as efn

class EfficientNetB0Cirrhosis(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetB0Cirrhosis, self).__init__()

        # Load EfficientNetB0
        self.efficientnet = efn.EfficientNet.from_pretrained('efficientnet-b0')

        # Replace classifier
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)
```

**Advantages:**
- Better parameter efficiency than ResNet
- Compound scaling for optimal performance
- Pretrained on ImageNet

### 4. Custom CNN with Attention

```python
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AttentionCNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            AttentionBlock(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            AttentionBlock(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            AttentionBlock(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
```

---

## 🏭 Training Pipeline

### 1. Data Preprocessing

```python
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Data augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 2. Training Configuration

```python
class TrainingConfig:
    def __init__(self):
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.patience = 10
        self.num_classes = 4

        # Loss function with class weights
        self.class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Higher weight for advanced stages
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3. Training Loop

```python
def train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    optimizer = config.optimizer(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    scheduler = config.scheduler(optimizer, T_max=config.num_epochs)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = config.criterion(outputs, labels)
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
                images, labels = images.to(config.device), labels.to(config.device)

                outputs = model(images)
                loss = config.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered")
                break

        scheduler.step()

    return model
```

---

## 🔍 Grad-CAM Implementation

### 1. Grad-CAM Class

```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # Hook to capture gradients
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

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
        cam = cam / torch.max(cam)

        return cam.squeeze().detach().cpu().numpy()
```

### 2. Grad-CAM Usage

```python
def explain_prediction(model, image_path, target_class):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Get target layer (last conv layer)
    if isinstance(model, ResNet50Cirrhosis):
        target_layer = model.resnet.layer4[-1]
    elif isinstance(model, EfficientNetB0Cirrhosis):
        target_layer = model.efficientnet._blocks[-1]
    else:
        target_layer = model.conv4  # For custom CNN

    # Generate Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor, target_class)

    # Resize CAM to original image size
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap on original image
    superimposed_img = heatmap * 0.4 + np.float32(image) / 255

    return superimposed_img, cam
```

---

## 📊 Model Evaluation

### 1. Performance Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    report = classification_report(all_labels, all_preds,
                                 target_names=class_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return report, cm
```

### 2. Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate_model(model_class, dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.image_paths, dataset.labels)):
        print(f'Fold {fold + 1}/{n_splits}')

        # Create data subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

        # Initialize model
        model = model_class()

        # Train model
        trained_model = train_model(model, train_loader, val_loader, config)

        # Evaluate
        report, cm = evaluate_model(trained_model, val_loader, class_names)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        })

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    summary = {
        'mean_accuracy': results_df['accuracy'].mean(),
        'std_accuracy': results_df['accuracy'].std(),
        'mean_f1': results_df['f1_score'].mean(),
        'std_f1': results_df['f1_score'].std()
    }

    return summary, results_df
```

---

## 🔧 Model Optimization

### 1. Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

def train_with_mixed_precision(model, train_loader, config):
    scaler = GradScaler()

    for epoch in range(config.num_epochs):
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                outputs = model(images)
                loss = config.criterion(outputs, labels)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

### 2. Model Quantization

```python
def quantize_model(model, test_loader):
    """Quantize model for faster inference"""
    model.eval()

    # Fuse Conv+BN+ReLU layers
    model = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu1']])

    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with test data
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            model(images)
            break  # Just one batch for calibration

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return model
```

---

## 💾 Model Deployment

### 1. ONNX Export

```python
import torch.onnx

def export_to_onnx(model, input_shape, onnx_path):
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
```

### 2. TorchScript Conversion

```python
def convert_to_torchscript(model, example_input):
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Save TorchScript model
    traced_model.save('model.pt')

    return traced_model
```

---

## 📈 Performance Monitoring

### 1. Model Monitoring

```python
class ModelMonitor:
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}

    def log_prediction(self, prediction_data):
        """Log prediction for monitoring"""
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction_data['prediction'],
            'confidence': prediction_data['confidence'],
            'inference_time': prediction_data['inference_time']
        })

    def get_performance_stats(self):
        """Calculate performance statistics"""
        if not self.prediction_history:
            return {}

        df = pd.DataFrame(self.prediction_history)

        stats = {
            'total_predictions': len(df),
            'avg_inference_time': df['inference_time'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'predictions_per_hour': len(df) / ((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
        }

        return stats
```

---

**For implementation details, see** `ai_pipeline/train_dl.py` and `backend/utils/xai_logic.py`
