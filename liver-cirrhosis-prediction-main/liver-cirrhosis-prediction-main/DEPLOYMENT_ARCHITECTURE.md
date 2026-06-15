# Deployment Architecture
## Production-Ready Healthcare AI System

---

## 🏗️ Overview

This document outlines the complete deployment architecture for the Explainable AI-Based Non-Invasive Liver Cirrhosis Detection and Smart Healthcare Assistant System, designed for scalability, reliability, and compliance with healthcare standards.

---

## 🏛️ System Architecture

### 1. Multi-Tier Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Application<br/>React.js]
        MOBILE[Mobile App<br/>React Native]
        API_CLI[API Clients<br/>Python/Node.js]
    end

    subgraph "API Gateway Layer"
        GATEWAY[API Gateway<br/>Kong/Traefik]
        AUTH[Authentication Service<br/>JWT/OAuth2]
        RATE_LIMIT[Rate Limiting<br/>Redis]
        LOGGING[Request Logging<br/>ELK Stack]
    end

    subgraph "Application Layer"
        FASTAPI[FastAPI Backend<br/>Prediction Service]
        CHATBOT[Chatbot Service<br/>NLP Engine]
        REPORT[Report Generation<br/>PDF Service]
        MONITOR[Monitoring Service<br/>Health Checks]
    end

    subgraph "AI/ML Layer"
        MODEL_API[Model Serving<br/>TorchServe/TensorFlow Serving]
        XAI_API[XAI Service<br/>SHAP/Grad-CAM]
        ENSEMBLE[Ensemble Engine<br/>Model Fusion]
        CACHE[Prediction Cache<br/>Redis]
    end

    subgraph "Data Layer"
        POSTGRES[(PostgreSQL<br/>User Data)]
        REDIS[(Redis<br/>Cache/Session)]
        MINIO[(MinIO<br/>Object Storage)]
        ELASTIC[(Elasticsearch<br/>Search/Indexing)]
    end

    subgraph "Infrastructure Layer"
        KUBERNETES[Kubernetes<br/>Orchestration]
        DOCKER[Docker<br/>Containerization]
        PROMETHEUS[Prometheus<br/>Monitoring]
        GRAFANA[Grafana<br/>Visualization]
    end

    WEB --> GATEWAY
    MOBILE --> GATEWAY
    API_CLI --> GATEWAY

    GATEWAY --> AUTH
    GATEWAY --> RATE_LIMIT
    GATEWAY --> LOGGING

    AUTH --> FASTAPI
    RATE_LIMIT --> FASTAPI
    LOGGING --> FASTAPI

    FASTAPI --> MODEL_API
    FASTAPI --> CHATBOT
    FASTAPI --> REPORT

    MODEL_API --> XAI_API
    MODEL_API --> ENSEMBLE
    MODEL_API --> CACHE

    FASTAPI --> POSTGRES
    FASTAPI --> REDIS
    FASTAPI --> MINIO
    FASTAPI --> ELASTIC

    KUBERNETES --> DOCKER
    PROMETHEUS --> GRAFANA
```

---

## 🐳 Containerization Strategy

### 1. Docker Architecture

#### Base Images
```dockerfile
# Multi-stage Dockerfile for FastAPI backend
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /home/app
USER app

WORKDIR /home/app

# Python dependencies
FROM base as dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime image
FROM base as runtime

COPY --from=dependencies /home/app/.local /home/app/.local
ENV PATH=/home/app/.local/bin:$PATH

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Service-Specific Containers

```dockerfile
# Model serving container
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

RUN pip install torchserve torch-model-archiver torch-workflow-archiver

# Copy models and configuration
COPY models/ /home/model-server/model-store/
COPY config/ /home/model-server/config/

EXPOSE 8080 8081
CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--ts-config", "/home/model-server/config/config.properties"]
```

```dockerfile
# React frontend container
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Production server
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 2. Docker Compose for Development

```yaml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: liver_cirrhosis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # MinIO object storage
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  # Backend API
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/liver_cirrhosis
      - REDIS_URL=redis://redis:6379
      - MINIO_ENDPOINT=minio:9000
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      minio:
        condition: service_started
    volumes:
      - ./backend:/app
      - /app/__pycache__

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  # Model serving
  model-server:
    build:
      context: ./ai_pipeline
      dockerfile: Dockerfile.torchserve
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - MODEL_STORE=/home/model-server/model-store
    volumes:
      - ./models:/home/model-server/model-store

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

---

## ☸️ Kubernetes Production Deployment

### 1. Kubernetes Manifests

#### Backend Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liver-cirrhosis-backend
  labels:
    app: liver-cirrhosis-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: liver-cirrhosis-backend
  template:
    metadata:
      labels:
        app: liver-cirrhosis-backend
    spec:
      containers:
      - name: backend
        image: liver-cirrhosis/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

#### Model Serving Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
  labels:
    app: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-k80  # GPU nodes
      containers:
      - name: torchserve
        image: liver-cirrhosis/model-server:latest
        ports:
        - containerPort: 8080
        - containerPort: 8081
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: "/ping"
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        volumeMounts:
        - name: model-storage
          mountPath: /home/model-server/model-store
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
```

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: liver-cirrhosis-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.liver-cirrhosis.com
    - app.liver-cirrhosis.com
    secretName: liver-cirrhosis-tls
  rules:
  - host: api.liver-cirrhosis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 80
  - host: app.liver-cirrhosis.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### 2. Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: liver-cirrhosis-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

---

## ☁️ Cloud Deployment Options

### 1. AWS Architecture

```mermaid
graph TB
    subgraph "AWS Cloud"
        subgraph "VPC"
            IGW[Internet Gateway]
            
            subgraph "Public Subnet"
                ALB[Application Load Balancer]
                NAT[NAT Gateway]
            end
            
            subgraph "Private Subnet A"
                ECS_BACKEND[ECS Backend Service<br/>Fargate]
                RDS[RDS PostgreSQL<br/>Multi-AZ]
                ELASTICACHE[ElastiCache Redis<br/>Cluster Mode]
            end
            
            subgraph "Private Subnet B"
                ECS_MODELS[ECS Model Service<br/>GPU Instances]
                SAGEMAKER[SageMaker<br/>Model Hosting]
            end
            
            subgraph "Private Subnet C"
                ECS_FRONTEND[ECS Frontend Service<br/>Fargate]
                S3[S3 Bucket<br/>Static Assets]
            end
        end
        
        ECR[Elastic Container Registry]
        CLOUDFRONT[CloudFront CDN]
        WAF[WAF & Shield]
        COGNITO[Cognito<br/>User Auth]
        API_GATEWAY[API Gateway]
    end

    USER[Healthcare Users] --> CLOUDFRONT
    CLOUDFRONT --> WAF
    WAF --> ALB
    ALB --> ECS_FRONTEND
    ALB --> API_GATEWAY
    
    API_GATEWAY --> COGNITO
    COGNITO --> ECS_BACKEND
    
    ECS_BACKEND --> RDS
    ECS_BACKEND --> ELASTICACHE
    ECS_BACKEND --> ECS_MODELS
    ECS_BACKEND --> SAGEMAKER
    
    ECR -.-> ECS_BACKEND
    ECR -.-> ECS_FRONTEND
    ECR -.-> ECS_MODELS
```

#### AWS Services Configuration

```hcl
# Terraform configuration for AWS deployment
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# VPC and networking
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support = true

  tags = {
    Name = "liver-cirrhosis-vpc"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "liver-cirrhosis-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition for Backend
resource "aws_ecs_task_definition" "backend" {
  family                   = "liver-cirrhosis-backend"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "backend"
      image = "${aws_ecr_repository.backend.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "DATABASE_URL"
          value = "postgresql://${aws_db_instance.postgres.username}:${var.db_password}@${aws_db_instance.postgres.endpoint}/${aws_db_instance.postgres.db_name}"
        },
        {
          name  = "REDIS_URL"
          value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:${aws_elasticache_cluster.redis.cache_nodes[0].port}"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/liver-cirrhosis-backend"
          "awslogs-region"        = "us-east-1"
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command = ["CMD-SHELL", "python -c \"import requests; requests.get('http://localhost:8000/health')\""]
        interval = 30
        timeout = 5
        retries = 3
      }
    }
  ])
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier             = "liver-cirrhosis-db"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.medium"
  allocated_storage      = 20
  max_allocated_storage  = 100
  db_name                = "liver_cirrhosis"
  username               = "postgres"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  multi_az               = true
  backup_retention_period = 7
  skip_final_snapshot    = true
  
  tags = {
    Name = "liver-cirrhosis-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "liver-cirrhosis-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Name = "liver-cirrhosis-cache"
  }
}

# S3 Bucket for storage
resource "aws_s3_bucket" "storage" {
  bucket = "liver-cirrhosis-storage-${random_string.suffix.result}"
  
  tags = {
    Name = "liver-cirrhosis-storage"
  }
}

resource "aws_s3_bucket_versioning" "storage" {
  bucket = aws_s3_bucket.storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

# CloudFront CDN
resource "aws_cloudfront_distribution" "cdn" {
  origin {
    domain_name = aws_s3_bucket.storage.bucket_regional_domain_name
    origin_id   = "S3-liver-cirrhosis-storage"
  }

  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-liver-cirrhosis-storage"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = {
    Name = "liver-cirrhosis-cdn"
  }
}
```

### 2. GCP Architecture

```yaml
# Cloud Run deployment for serverless
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: liver-cirrhosis-backend
spec:
  template:
    spec:
      containers:
      - image: gcr.io/PROJECT-ID/liver-cirrhosis/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
        resources:
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3. Azure Architecture

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerRegistry/registries",
      "apiVersion": "2023-07-01",
      "name": "livercirrhosisacr",
      "location": "[parameters('location')]",
      "sku": {
        "name": "Basic"
      },
      "properties": {
        "adminUserEnabled": true
      }
    },
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2023-05-01",
      "name": "liver-cirrhosis-backend",
      "location": "[parameters('location')]",
      "properties": {
        "containers": [
          {
            "name": "backend",
            "properties": {
              "image": "livercirrhosisacr.azurecr.io/backend:latest",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "DATABASE_URL",
                  "value": "[concat('postgresql://', parameters('dbUsername'), '@', reference(resourceId('Microsoft.DBforPostgreSQL/flexibleServers', parameters('dbName'))).fullyQualifiedDomainName, ':5432/', parameters('dbName'))]"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 1.5
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        }
      },
      "dependsOn": [
        "[resourceId('Microsoft.ContainerRegistry/registries', 'livercirrhosisacr')]"
      ]
    }
  ]
}
```

---

## 📊 Monitoring and Observability

### 1. Prometheus Metrics

```yaml
# Prometheus configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'model-server'
    static_configs:
      - targets: ['model-server:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
```

#### Custom Metrics Collection

```python
# FastAPI metrics middleware
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_type', 'prediction_class']
)

MODEL_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model_type']
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(time.time() - start_time)
        
        return response

# Model prediction metrics
def record_prediction_metrics(model_type, prediction_class, latency):
    PREDICTION_COUNT.labels(
        model_type=model_type,
        prediction_class=prediction_class
    ).inc()
    
    MODEL_LATENCY.labels(model_type=model_type).observe(latency)
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Liver Cirrhosis AI System",
    "tags": ["healthcare", "ai", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(model_prediction_duration_seconds_sum[5m]) / rate(model_prediction_duration_seconds_count[5m])",
            "legendFormat": "{{model_type}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "5xx errors (%)"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_users",
            "legendFormat": "Active Users"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules

```yaml
groups:
  - name: liver_cirrhosis_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"[5][0-9][0-9]"}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}% which is above 5%"

      - alert: SlowAPIResponse
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow API response time"
          description: "95th percentile response time is {{ $value }}s"

      - alert: ModelServerDown
        expr: up{job="model-server"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Model server is down"
          description: "Model server has been down for more than 1 minute"

      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
```

---

## 🔒 Security Architecture

### 1. Authentication and Authorization

```python
# JWT Authentication Service
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

class AuthService:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise JWTError
            return username
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

# Role-based access control
class RBACService:
    def __init__(self):
        self.roles = {
            "admin": ["read", "write", "delete", "admin"],
            "doctor": ["read", "write", "predict"],
            "patient": ["read", "predict"],
            "researcher": ["read", "export"]
        }

    def check_permission(self, user_role: str, required_permission: str) -> bool:
        if user_role not in self.roles:
            return False
        return required_permission in self.roles[user_role]
```

### 2. Data Encryption

```python
# Data encryption service
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class EncryptionService:
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.fernet = Fernet(key)

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_file(self, input_path: str, output_path: str):
        """Encrypt file"""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)

    def decrypt_file(self, input_path: str, output_path: str):
        """Decrypt file"""
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)

# Database field-level encryption
class DatabaseEncryption:
    def __init__(self, encryption_service: EncryptionService):
        self.encryption = encryption_service

    def encrypt_sensitive_fields(self, data: dict) -> dict:
        """Encrypt sensitive fields before storing in database"""
        sensitive_fields = ['medical_history', 'personal_info', 'contact_details']
        
        encrypted_data = data.copy()
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encryption.encrypt_data(str(encrypted_data[field]))
        
        return encrypted_data

    def decrypt_sensitive_fields(self, data: dict) -> dict:
        """Decrypt sensitive fields when retrieving from database"""
        sensitive_fields = ['medical_history', 'personal_info', 'contact_details']
        
        decrypted_data = data.copy()
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = self.encryption.decrypt_data(decrypted_data[field])
                except:
                    # If decryption fails, keep original (might not be encrypted)
                    pass
        
        return decrypted_data
```

### 3. HIPAA Compliance

```python
# HIPAA compliance service
class HIPAAComplianceService:
    def __init__(self):
        self.audit_log = []
        
    def log_access(self, user_id: str, resource: str, action: str, ip_address: str):
        """Log all data access for HIPAA compliance"""
        log_entry = {
            'timestamp': datetime.utcnow(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'ip_address': ip_address,
            'user_agent': request.headers.get('User-Agent', ''),
            'session_id': request.cookies.get('session_id', '')
        }
        
        self.audit_log.append(log_entry)
        
        # Write to secure audit log
        self.write_audit_log(log_entry)

    def check_minimum_necessary(self, user_role: str, requested_data: list) -> list:
        """Implement minimum necessary access principle"""
        role_permissions = {
            'doctor': ['diagnosis', 'treatment_plan', 'medical_history'],
            'nurse': ['vital_signs', 'medication_list'],
            'admin': ['all'],
            'patient': ['own_records']
        }
        
        allowed_data = role_permissions.get(user_role, [])
        if 'all' in allowed_data:
            return requested_data
        
        return [data for data in requested_data if data in allowed_data]

    def enforce_retention_policy(self):
        """Automatically delete data beyond retention period"""
        retention_days = {
            'medical_records': 365 * 7,  # 7 years
            'audit_logs': 365 * 6,       # 6 years
            'diagnostic_images': 365 * 10 # 10 years
        }
        
        for data_type, days in retention_days.items():
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            self.delete_expired_data(data_type, cutoff_date)

    def anonymize_data_for_research(self, data: dict) -> dict:
        """Anonymize data for research purposes"""
        anonymized = data.copy()
        
        # Remove direct identifiers
        direct_identifiers = ['name', 'ssn', 'phone', 'email', 'address']
        for identifier in direct_identifiers:
            anonymized.pop(identifier, None)
        
        # Generalize quasi-identifiers
        if 'age' in anonymized:
            anonymized['age'] = self.generalize_age(anonymized['age'])
        if 'zip_code' in anonymized:
            anonymized['zip_code'] = anonymized['zip_code'][:3] + '**'
        
        return anonymized
```

---

## 🚀 CI/CD Pipeline

### 1. GitHub Actions Workflow

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push backend image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: liver-cirrhosis-backend
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest ./backend
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
    
    - name: Build and push frontend image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: liver-cirrhosis-frontend
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest ./frontend
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Update ECS service
      run: |
        aws ecs update-service \
          --cluster liver-cirrhosis-cluster \
          --service liver-cirrhosis-backend \
          --force-new-deployment \
          --region us-east-1
```

### 2. Infrastructure as Code

```yaml
# CloudFormation template for infrastructure
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Liver Cirrhosis AI System Infrastructure'

Parameters:
  EnvironmentName:
    Type: String
    Default: 'production'
    AllowedValues: ['development', 'staging', 'production']

Resources:
  # VPC
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-vpc'

  # Subnets
  PublicSubnetA:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true

  PrivateSubnetA:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [0, !GetAZs '']

  PrivateSubnetB:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.3.0/24
      AvailabilityZone: !Select [1, !GetAZs '']

  # Security Groups
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: 'ALB Security Group'
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0

  ECSSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: 'ECS Security Group'
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          SourceSecurityGroupId: !Ref ALBSecurityGroup

  # Application Load Balancer
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub '${EnvironmentName}-alb'
      Type: application
      Scheme: internet-facing
      IpAddressType: ipv4
      Subnets:
        - !Ref PublicSubnetA
      SecurityGroups:
        - !Ref ALBSecurityGroup

  # Target Group
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub '${EnvironmentName}-tg'
      Protocol: HTTP
      Port: 8000
      VpcId: !Ref VPC
      TargetType: ip
      HealthCheckPath: /health

  # Listener
  Listener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Protocol: HTTP
      Port: 80
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${EnvironmentName}-cluster'
      ClusterSettings:
        - Name: containerInsights
          Value: enabled

  # Task Definition
  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub '${EnvironmentName}-backend'
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: '1024'
      Memory: '2048'
      ExecutionRoleArn: !GetAtt ECSExecutionRole.Arn
      TaskRoleArn: !GetAtt ECSTaskRole.Arn
      ContainerDefinitions:
        - Name: backend
          Image: !Sub '${AWS::Account}.dkr.ecr.${AWS::Region}.amazonaws.com/liver-cirrhosis-backend:latest'
          PortMappings:
            - ContainerPort: 8000
              Protocol: tcp
          Environment:
            - Name: ENVIRONMENT
              Value: !Ref EnvironmentName
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: ecs
          HealthCheck:
            Command:
              - CMD-SHELL
              - python -c "import requests; requests.get('http://localhost:8000/health')"
            Interval: 30
            Timeout: 5
            Retries: 3

  # ECS Service
  ECSService:
    Type: AWS::ECS::Service
    Properties:
      ServiceName: !Sub '${EnvironmentName}-backend'
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 3
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets:
            - !Ref PrivateSubnetA
            - !Ref PrivateSubnetB
          SecurityGroups:
            - !Ref ECSSecurityGroup
          AssignPublicIp: DISABLED
      LoadBalancers:
        - TargetGroupArn: !Ref TargetGroup
          ContainerName: backend
          ContainerPort: 8000
      HealthCheckGracePeriodSeconds: 60

Outputs:
  LoadBalancerDNS:
    Description: 'DNS name of the load balancer'
    Value: !GetAtt ApplicationLoadBalancer.DNSName
    Export:
      Name: !Sub '${EnvironmentName}-alb-dns'
```

---

## 📈 Performance Optimization

### 1. Caching Strategy

```python
# Multi-level caching service
class CacheService:
    def __init__(self, redis_client, memory_cache_size=1000):
        self.redis = redis_client
        self.memory_cache = LRUCache(memory_cache_size)
        self.cache_ttl = {
            'predictions': 3600,    # 1 hour
            'user_data': 1800,      # 30 minutes
            'model_metadata': 86400 # 24 hours
        }

    async def get_cached_prediction(self, input_hash: str) -> Optional[dict]:
        """Get cached prediction result"""
        # Check memory cache first
        if input_hash in self.memory_cache:
            return self.memory_cache[input_hash]
        
        # Check Redis cache
        cached_result = await self.redis.get(f"prediction:{input_hash}")
        if cached_result:
            result = json.loads(cached_result)
            self.memory_cache[input_hash] = result  # Update memory cache
            return result
        
        return None

    async def cache_prediction(self, input_hash: str, result: dict):
        """Cache prediction result"""
        # Cache in memory
        self.memory_cache[input_hash] = result
        
        # Cache in Redis
        await self.redis.setex(
            f"prediction:{input_hash}",
            self.cache_ttl['predictions'],
            json.dumps(result)
        )

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate user-specific cache"""
        pattern = f"user:{user_id}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

    async def warmup_cache(self, common_inputs: list):
        """Warm up cache with common inputs"""
        for input_data in common_inputs:
            input_hash = self.hash_input(input_data)
            if not await self.get_cached_prediction(input_hash):
                # Pre-compute and cache
                prediction = await self.compute_prediction(input_data)
                await self.cache_prediction(input_hash, prediction)
```

### 2. Model Optimization

```python
# Model optimization service
class ModelOptimizationService:
    def __init__(self):
        self.quantization_config = {
            'dynamic': {'dtype': torch.qint8},
            'static': {'dtype': torch.qint8, 'qconfig': 'fbgemm'},
            'quantization_aware': {'dtype': torch.qint8, 'epochs': 5}
        }

    def optimize_model(self, model, optimization_type='dynamic'):
        """Optimize model for inference"""
        if optimization_type == 'dynamic':
            return self.dynamic_quantization(model)
        elif optimization_type == 'static':
            return self.static_quantization(model)
        elif optimization_type == 'pruning':
            return self.prune_model(model)
        elif optimization_type == 'distillation':
            return self.knowledge_distillation(model)

    def dynamic_quantization(self, model):
        """Apply dynamic quantization"""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        return quantized_model

    def static_quantization(self, model):
        """Apply static quantization"""
        model.eval()
        
        # Fuse layers
        model = torch.quantization.fuse_modules(
            model,
            [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']]
        )
        
        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        self.calibrate_model(model)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        return model

    def prune_model(self, model, pruning_ratio=0.3):
        """Apply model pruning"""
        parameters_to_prune = []
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )
        
        # Remove pruning reparameterization
        for module, param in parameters_to_prune:
            prune.remove(module, param)
        
        return model

    def knowledge_distillation(self, teacher_model, student_model, train_loader):
        """Apply knowledge distillation"""
        optimizer = torch.optim.Adam(student_model.parameters())
        criterion = nn.KLDivLoss(reduction='batchmean')
        temperature = 3.0
        
        student_model.train()
        teacher_model.eval()
        
        for epoch in range(10):
            for images, _ in train_loader:
                optimizer.zero_grad()
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(images) / temperature
                
                # Student predictions
                student_logits = student_model(images) / temperature
                
                # Knowledge distillation loss
                loss = criterion(
                    F.log_softmax(student_logits, dim=1),
                    F.softmax(teacher_logits, dim=1)
                ) * (temperature ** 2)
                
                loss.backward()
                optimizer.step()
        
        return student_model
```

---

## 🔄 Backup and Disaster Recovery

### 1. Backup Strategy

```yaml
# Backup configuration
backup:
  databases:
    postgresql:
      schedule: "0 2 * * *"  # Daily at 2 AM
      retention: 30  # days
      type: "full"
      destination: "s3://liver-cirrhosis-backups/db/"
      
    redis:
      schedule: "0 */4 * * *"  # Every 4 hours
      retention: 7  # days
      type: "snapshot"
      destination: "s3://liver-cirrhosis-backups/cache/"

  models:
    schedule: "0 3 * * 1"  # Weekly on Monday at 3 AM
    retention: 52  # weeks
    destination: "s3://liver-cirrhosis-backups/models/"
    include_weights: true
    include_metadata: true

  user_data:
    schedule: "0 1 * * *"  # Daily at 1 AM
    retention: 2555  # 7 years
    encryption: true
    destination: "s3://liver-cirrhosis-backups/user-data/"
    compliance: "hipaa"

  logs:
    schedule: "0 */6 * * *"  # Every 6 hours
    retention: 90  # days
    destination: "s3://liver-cirrhosis-backups/logs/"
    compression: "gzip"
```

### 2. Disaster Recovery

```yaml
# Disaster recovery configuration
disaster_recovery:
  rto: 4  # Recovery Time Objective: 4 hours
  rpo: 1  # Recovery Point Objective: 1 hour
  
  multi_region:
    primary: "us-east-1"
    secondary: "us-west-2"
    tertiary: "eu-west-1"
  
  failover_strategy:
    automatic: true
    health_checks: 3  # consecutive failures
    dns_ttl: 60  # seconds
    
  data_replication:
    postgresql: "synchronous"
    redis: "async"
    s3: "cross-region-replication"
  
  monitoring:
    alerts:
      - "database_replication_lag > 5min"
      - "region_unavailable"
      - "backup_failure"
```

---

**For deployment scripts and configurations, see** `deployment/` directory and `infrastructure/` directory
