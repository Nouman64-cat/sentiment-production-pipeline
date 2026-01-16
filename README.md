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

| Step            | Option A: uv (Recommended) | Option B: pip (Standard)                                                      |
| :-------------- | :------------------------- | :---------------------------------------------------------------------------- |
| **1. Install**  | `pip install uv`           | (Pre-installed)                                                               |
| **2. Init**     | `uv sync`                  | `python -m venv .venv`                                                        |
| **3. Activate** | _(Handled by `uv run`)_    | `source .venv/bin/activate` (Mac/Linux)<br>`.venv\Scripts\activate` (Windows) |
| **4. Deps**     | **Done!**                  | `pip install -r requirements.txt`                                             |

#### Manual Training (Only for Option 2)

If running locally, you must execute the pipeline steps manually:

```bash
# Using uv (or python if using pip)
uv run python src/data/make_dataset.py
uv run python src/models/train_ml.py
uv run python src/models/train_dl.py
uv run uvicorn src.api.main:app --reload
```
