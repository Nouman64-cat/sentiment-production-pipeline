# Sentiment Production Pipeline

## Setup Guide

Clone the repository:

```bash
git clone https://github.com/Nouman64-cat/sentiment-production-pipeline.git
cd sentiment-production-pipeline
```

### Option 1: Docker (Recommended for Evaluation)

Builds the environment, downloads data, trains both models, and serves the API in one container.

```bash
# Build the image (Approx. 5 mins - includes Model Training)
docker build -t sentiment-api .

# Run the container
docker run -p 8000:8000 sentiment-api
```

### Option 2: Local Development

If you wish to modify code or run locally.

#### Step 1: Install uv (Recommended) or use pip

**using pipx (any platform):**

```bash
pipx install uv
```

**Or using pip with virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
```

#### Step 2: Setup Environment

| Step            | Option A: uv (Recommended) | Option B: pip (Standard)                                                      |
| :-------------- | :------------------------- | :---------------------------------------------------------------------------- |
| **1. Init**     | `uv sync`                  | `python -m venv .venv`                                                        |
| **2. Activate** | _(Handled by `uv run`)_    | `source .venv/bin/activate` (Mac/Linux)<br>`.venv\Scripts\activate` (Windows) |
| **3. Deps**     | **Done!**                  | `pip install -r requirements.txt`                                             |

#### Manual Training (Only for Option 2)

If running locally, you must execute the pipeline steps manually:

```bash
# Using uv (set PYTHONPATH so 'src' module is found)
PYTHONPATH=. uv run python src/scripts/download_dataset.py
PYTHONPATH=. uv run python src/models/train_ml.py
PYTHONPATH=. uv run python src/models/train_dl.py
PYTHONPATH=. uv run uvicorn src.api.main:app --reload

# For Mlflow tracking
uv run mlflow ui
```
