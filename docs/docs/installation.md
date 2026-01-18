---
sidebar_position: 2
---

# Installation

Complete guide to setting up the Sentiment Analysis Pipeline.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Git** for cloning the repository
- **Docker & Docker Compose** (recommended) OR
- **Python 3.10+** with **pip** or **uv** (for local development)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Nouman64-cat/sentiment-production-pipeline.git
cd sentiment-production-pipeline
```

---

## Quick Start with Docker (Recommended)

Docker is the fastest way to get started. It automatically downloads data, trains both models, and serves the API.

### Option A: Docker Compose (Best Experience)

Docker Compose runs both the API and MLflow UI together:

```bash
docker compose up --build
```

**Access the services:**

| Service       | URL                   |
| ------------- | --------------------- |
| **API**       | http://localhost:8000 |
| **MLflow UI** | http://localhost:5001 |

:::tip Why Docker Compose?
Docker Compose is the recommended approach because it gives you instant access to MLflow experiment tracking without any additional setup.
:::

To stop the services:

```bash
docker compose down
```

### Option B: Docker Only (API)

If you only need the API without MLflow UI:

```bash
# Build the image (includes model training, ~5 mins)
docker build -t sentiment-pipeline .

# Run the container
docker run -p 8000:8000 sentiment-pipeline
```

#### Viewing MLflow with Docker Only

When running Docker without Compose, MLflow experiments are stored inside the container. To access them, mount a volume:

```bash
# Run with volume mount
docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns sentiment-pipeline

# In another terminal, run MLflow UI locally
pip install mlflow
mlflow ui --backend-store-uri ./mlruns
```

Then visit http://localhost:5001

---

## Local Development (Alternative)

For developers who want to modify the code or run without Docker.

### Step 1: Create Virtual Environment

**Using uv (Recommended):**

```bash
# Install uv if not already installed
pip install uv

# Sync dependencies
uv sync
```

**Using venv (Standard Python):**

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Environment Configuration

Copy the environment template:

```bash
cp .env.example .env
```

### Step 3: Run the Pipeline

```bash
# Download dataset
PYTHONPATH=. uv run python src/scripts/download_dataset.py

# Train Classical ML Model
PYTHONPATH=. uv run python src/models/train_ml.py

# Train Deep Learning Model
PYTHONPATH=. uv run python src/models/train_dl.py
```

:::info Training Time

- ML Model: ~30 seconds
- DL Model: ~5-15 minutes (depending on GPU availability)
  :::

### Step 4: Start the API

```bash
PYTHONPATH=. uv run uvicorn src.api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

### Step 5: View MLflow Experiments

```bash
uv run mlflow ui
```

MLflow UI will be available at http://localhost:5000

---

## Verify Installation

Test the health check endpoint:

```bash
curl http://localhost:8000/healthcheck
```

Expected response:

```json
{
  "status": "ok",
  "ml_model": true,
  "dl_model": true
}
```

---

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'src'**

Ensure you're running commands from the project root directory with `PYTHONPATH=.` prefix.

**CUDA/GPU Issues**

If you encounter GPU-related errors, the system will automatically fall back to CPU. For explicit CPU usage:

```python
device = torch.device("cpu")
```

**MLflow Database Errors**

Delete the existing database and restart:

```bash
rm mlflow.db
python -m src.models.train_ml
```

**Docker: No space left on device**

Clean up Docker resources:

```bash
docker system prune -a
```
