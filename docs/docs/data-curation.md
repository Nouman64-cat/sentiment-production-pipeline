---
sidebar_position: 4
---

# Data Curation

This page explains how the sentiment analysis dataset was curated from multiple sources, processed, and made available for the pipeline.

## Data Sources

The final dataset is a combination of two well-known sentiment datasets from Kaggle:

### 1. Sentiment140 (Twitter Dataset)

- **Source**: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Description**: 1.6 million tweets extracted using the Twitter API, labeled for sentiment analysis.
- **Labels**: Negative (0), Positive (4)

### 2. IMDB Movie Reviews

- **Source**: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description**: 50,000 movie reviews from IMDB, evenly split between positive and negative sentiments.
- **Labels**: Negative, Positive

## Data Processing Pipeline

The raw datasets were processed using Jupyter notebooks in Google Colab. The workflow is documented in the `notebooks/` directory:

### Step 1: Exploratory Data Analysis

Each dataset was analyzed independently to understand its structure, class distribution, and text characteristics.

| Notebook                                                                                                                 | Purpose                                                 |
| :----------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------ |
| [`EDA_IMDB.ipynb`](https://github.com/Nouman64-cat/sentiment-production-pipeline/blob/main/notebooks/EDA_IMDB.ipynb)     | Exploratory analysis of the IMDB reviews dataset        |
| [`EDA_Tweets.ipynb`](https://github.com/Nouman64-cat/sentiment-production-pipeline/blob/main/notebooks/EDA_Tweets.ipynb) | Exploratory analysis of the Sentiment140 tweets dataset |

### Step 2: Merging Datasets

The two datasets were combined, standardized to a common schema (`text`, `sentiment`), and balanced to create a unified training corpus.

| Notebook                                                                                                                               | Purpose                                            |
| :------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- |
| [`Merge_IMDB_Tweets.ipynb`](https://github.com/Nouman64-cat/sentiment-production-pipeline/blob/main/notebooks/Merge_IMDB_Tweets.ipynb) | Merging, cleaning, and exporting the final dataset |

### Step 3: Hosting the Dataset

The final curated dataset was uploaded to an S3 bucket for easy access during training:

```
https://ai-ml-dl---datasets.s3.us-east-1.amazonaws.com/test-task-dataset/data.csv
```

## Automatic Download

The pipeline includes a script to automatically download the curated dataset:

```bash
uv run python -m src.scripts.download_dataset
```

This script (`src/scripts/download_dataset.py`) fetches the CSV from S3 and saves it to `src/data/data.csv`.

```python
DATA_URL = "https://ai-ml-dl---datasets.s3.us-east-1.amazonaws.com/test-task-dataset/data.csv"
OUTPUT_FILE = "src/data/data.csv"
```

## Final Dataset Schema

| Column      | Type   | Description                              |
| :---------- | :----- | :--------------------------------------- |
| `text`      | string | The review or tweet text                 |
| `sentiment` | int    | Binary label: 0 (Negative), 1 (Positive) |

## Why Combine Datasets?

1. **Diversity**: Tweets are short and informal; IMDB reviews are longer and structured. This diversity helps the model generalize better.
2. **Volume**: More data improves model training, especially for deep learning.
3. **Balance**: The merged dataset maintains a balanced class distribution to prevent bias.
