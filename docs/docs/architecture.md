---
sidebar_position: 3
---

# Project Architecture

Understanding the folder structure and design decisions behind the Sentiment Analysis Pipeline.

## Directory Structure

```
sentiment-production-pipeline/
├── dataset/                    # Training data
│   └── data.csv               # Curated sentiment dataset
│
├── docs/                       # Docusaurus documentation (you are here)
│
├── models/                     # Trained model artifacts
│   ├── ml_model.joblib        # Serialized scikit-learn pipeline
│   └── dl_model.pth           # PyTorch model weights
│
├── mlruns/                     # MLflow experiment tracking data
│
├── notebooks/                  # Jupyter notebooks for analysis
│   └── analysis.ipynb         # Model comparison notebook
│
├── src/                        # Main source code
│   ├── __init__.py
│   │
│   ├── api/                   # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py           # API routes and startup
│   │   └── database.py       # SQLite logging utilities
│   │
│   ├── models/                # Model training scripts
│   │   ├── __init__.py
│   │   ├── train_ml.py       # Classical ML training
│   │   └── train_dl.py       # Deep Learning training
│   │
│   ├── preprocessing/         # Text preprocessing
│   │   ├── __init__.py
│   │   └── clean.py          # Text cleaning functions
│   │
│   └── scripts/               # Utility scripts
│       └── download_dataset.py
│
├── tests/                      # Test suite
│   └── test_preprocessing.py  # Preprocessing unit tests
│
├── Dockerfile                  # Container definition
├── pyproject.toml             # Project metadata
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

## Architecture Decisions

### Modular Design

The project follows a modular architecture where each component has a single responsibility:

| Module               | Responsibility                  |
| -------------------- | ------------------------------- |
| `src/preprocessing/` | Text cleaning and normalization |
| `src/models/`        | Model training and evaluation   |
| `src/api/`           | REST API and request handling   |
| `src/scripts/`       | Data preparation utilities      |

### Why This Structure?

1. **Separation of Concerns**: Each module handles one aspect of the pipeline
2. **Easy Testing**: Isolated modules can be unit tested independently
3. **Reusability**: Preprocessing can be imported anywhere without circular dependencies
4. **Scalability**: New models or endpoints can be added without modifying existing code

## Data Flow

```
┌──────────────┐    ┌─────────────────┐    ┌───────────────┐
│  Raw Text    │───▶│  Preprocessing  │───▶│  Clean Text   │
└──────────────┘    │  (clean.py)     │    └───────────────┘
                    └─────────────────┘            │
                                                   ▼
                    ┌─────────────────┐    ┌───────────────┐
                    │   Model A (ML)  │◀───│  API Request  │
                    │   TF-IDF + LR   │    └───────────────┘
                    └─────────────────┘            │
                           │                       ▼
                           ▼              ┌───────────────┐
                    ┌─────────────────┐   │  Model B (DL) │
                    │   Prediction    │   │  DistilBERT   │
                    │   Response      │   └───────────────┘
                    └─────────────────┘            │
                           ▲                       │
                           └───────────────────────┘
```

## Key Components

### Preprocessing Module (`src/preprocessing/`)

The preprocessing module provides a single, consistent text cleaning function used by:

- Training scripts (data preparation)
- API endpoints (inference-time cleaning)
- Analysis notebooks (evaluation)

This ensures identical preprocessing during training and inference, preventing data drift.

### Models Module (`src/models/`)

Contains training scripts with:

- **MLflow integration** for experiment tracking
- **Hyperparameter tuning** via GridSearchCV
- **Model serialization** for production deployment

### API Module (`src/api/`)

FastAPI application with:

- **Lifespan management** for model loading
- **Prediction logging** to SQLite
- **Health checks** for monitoring
- **MLflow tracing** for observability

## Configuration Management

| File               | Purpose                                 |
| ------------------ | --------------------------------------- |
| `.env`             | Environment variables (API keys, paths) |
| `pyproject.toml`   | Project metadata and tool configuration |
| `requirements.txt` | Pinned Python dependencies              |

## Artifact Storage

| Artifact    | Location                 | Format                             |
| ----------- | ------------------------ | ---------------------------------- |
| ML Model    | `models/ml_model.joblib` | Joblib serialized sklearn pipeline |
| DL Model    | `models/dl_model.pth`    | PyTorch state dict                 |
| MLflow Data | `mlruns/`                | MLflow artifact store              |
| Logs        | `logs.db`                | SQLite database                    |

## Scaling Considerations

The current architecture supports:

1. **Horizontal Scaling**: API is stateless (models loaded at startup)
2. **Model Versioning**: MLflow tracks all experiments
3. **Containerization**: Dockerfile included for consistent deployments
4. **A/B Testing**: Separate endpoints for each model enable gradual rollouts
