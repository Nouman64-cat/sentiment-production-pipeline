---
sidebar_position: 7
---

# Evaluation & Metrics

This project uses a centralized evaluation module to ensure that all models are benchmarked using the exact same logic.

## Metrics Used

| Metric               | Description                                                     | Purpose                                                                                   |
| :------------------- | :-------------------------------------------------------------- | :---------------------------------------------------------------------------------------- |
| **Accuracy**         | Fraction of correct predictions.                                | General performance indicator.                                                            |
| **F1 Score**         | Harmonic mean of precision and recall.                          | Better than accuracy for imbalanced datasets (e.g., if we had very few negative reviews). |
| **Confusion Matrix** | Visualizes partial errors (False Positives vs False Negatives). | Helps understand _how_ the model is failing.                                              |

## Running Evaluations

You can evaluate trained models independently using the `src/scripts/evaluate.py` script. This script loads the model artifacts and the test portion of the dataset (defined in `config/*.yaml`) to verify performance.

### 1. Evaluate Classical ML Model

```bash
# Recommend module execution (cleaner path handling)
uv run python -m src.scripts.evaluate --model ml

# Alternative: Run script directly
uv run python src/scripts/evaluate.py --model ml
```

**Output:**

- Prints Accuracy and F1 Score to console.
- Saves confusion matrix plot to `models/ml_confusion_matrix.png`.

### 2. Evaluate Deep Learning Model

```bash
# Recommend module execution
uv run python -m src.scripts.evaluate --model dl

# Alternative: Run script directly
uv run python src/scripts/evaluate.py --model dl
```

**Output:**

- Prints Accuracy and F1 Score to console.
- Saves confusion matrix plot to `models/dl_confusion_matrix.png`.

### 3. View in MLflow (Docker / Local)

If you are running the pipeline via `docker compose`, you can view these metrics and the confusion matrix in the MLflow UI:

1.  Open [http://localhost:5001](http://localhost:5001).
2.  Select your experiment (`Sentiment_Analysis_ML` or `Sentiment_Analysis_DL`).
3.  Click on a run to see **Metrics** (Accuracy, F1).
4.  Scroll to **Artifacts** to view the `confusion_matrix.png`.

## Code Functionality

The evaluation logic is decoupled into:

1.  **Definitions (`src/evaluation/metrics.py`)**:
    - Pure functions that take `y_true` and `y_pred` arrays.
    - Used by _both_ the training loop (for real-time logs) and the evaluation script (for final reports).

2.  **Runner (`src/scripts/evaluate.py`)**:
    - Handles data loading, model instantiation, and batch inference.
    - Ensures the test set split is identical to training by reading `src/config/`.

## Key Dependencies

- `scikit-learn`: For metric calculations.
- `matplotlib` & `seaborn`: For generating confusion matrix plots.
