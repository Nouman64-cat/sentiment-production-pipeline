---
sidebar_position: 2
---

# Installation

Complete guide to setting up the Sentiment Analysis Pipeline on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (3.12 recommended)
- **pip** or **uv** package manager
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Nouman64-cat/sentiment-production-pipeline.git
cd sentiment-production-pipeline
```

## Step 2: Create Virtual Environment

### Using venv (Standard Python)

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Using uv (Faster Alternative)

```bash
# Install uv if not already installed
pip install uv

# Create and activate environment
uv venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate on Windows
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

## Step 4: Environment Configuration

Copy the environment template and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

## Step 5: Download/Prepare Dataset

The dataset should be placed in the `dataset/` directory as `data.csv`. If you have a download script:

```bash
python -m src.scripts.download_dataset
```

## Step 6: Train Models

Train both models to generate the required artifacts:

```bash
# Train Classical ML Model (Logistic Regression)
python -m src.models.train_ml

# Train Deep Learning Model (DistilBERT)
python -m src.models.train_dl
```

:::info Training Time

- ML Model: ~30 seconds
- DL Model: ~5-15 minutes (depending on GPU availability)
  :::

## Step 7: Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## Step 8: Verify Installation

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

## Docker Installation (Alternative)

Build and run using Docker:

```bash
# Build the image
docker build -t sentiment-pipeline .

# Run the container
docker run -p 8000:8000 sentiment-pipeline
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'src'**

Ensure you're running commands from the project root directory, not from subdirectories.

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
