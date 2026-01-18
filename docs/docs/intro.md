---
sidebar_position: 1
---

# Sentiment Analysis Pipeline

Welcome to the **Sentiment Production Pipeline** documentation. This project provides a complete, production-ready sentiment analysis system with both Classical Machine Learning and Deep Learning models.

## Overview

This pipeline demonstrates a real-world ML system that includes:

- **Dual Model Architecture**: Classical ML (Logistic Regression) and Deep Learning (DistilBERT) models
- **Production API**: FastAPI-based REST endpoints with health checks and logging
- **MLflow Integration**: Experiment tracking, model versioning, and artifact storage
- **Docker Support**: Containerized deployment for consistent environments
- **Comprehensive Testing**: Unit tests for preprocessing and model validation

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Nouman64-cat/sentiment-production-pipeline.git
cd sentiment-production-pipeline

# Install dependencies
pip install -r requirements.txt

# Train models
python -m src.models.train_ml
python -m src.models.train_dl

# Start the API
uvicorn src.api.main:app --reload
```

## Documentation Structure

| Section                          | Description                           |
| -------------------------------- | ------------------------------------- |
| [Installation](./installation)   | Complete setup instructions           |
| [Architecture](./architecture)   | Project folder structure explained    |
| [Dataset](./dataset)             | How the training data was curated     |
| [Preprocessing](./preprocessing) | Text cleaning pipeline details        |
| [Model Choices](./model-choices) | Why we chose these specific models    |
| [Results](./results)             | Performance comparison and benchmarks |
| [API Usage](./api-usage)         | Endpoint documentation with examples  |

## Key Features

### Classical ML Model

- **TF-IDF Vectorization** with optimized n-gram ranges
- **Logistic Regression** with GridSearchCV hyperparameter tuning
- **Sub-millisecond inference** ideal for real-time applications

### Deep Learning Model

- **DistilBERT** transformer architecture for contextual understanding
- **Fine-tuned** on sentiment-specific data
- **Higher accuracy** for complex or nuanced text

### Production Features

- SQLite logging for prediction auditing
- MLflow experiment tracking
- Health check endpoints
- Docker containerization
