# Sentiment Production Pipeline

[![Documentation Status](https://img.shields.io/badge/docs-live-brightgreen)](https://nouman64-cat.github.io/sentiment-production-pipeline/)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)](https://fastapi.tiangolo.com/)

**Production-ready Sentiment Analysis System** featuring a "Champion/Challenger" architecture with Classical ML (Logistic Regression) and Deep Learning (DistilBERT) models.

> **Full Documentation:** [View the Live Documentation Site](https://nouman64-cat.github.io/sentiment-production-pipeline/) for architecture diagrams, deep-dives, and EDA.

---

## Model Comparison & Results (The Bottom Line)

We compared a lightweight **Classical ML** model against a heavy **Deep Learning** Transformer.

| Metric              | Model A: Classical ML | Model B: DistilBERT  |
| :------------------ | :-------------------- | :------------------- |
| **F1 Score**        | 0.62                  | **0.82**             |
| **Inference Speed** | **~0.15 ms/sample**   | ~17.18 ms/sample     |
| **Model Size**      | **0.2 MB**            | 255 MB               |
| **Training Time**   | < 5 seconds           | ~10 minutes (on CPU) |

### Deployment Decision

**Recommendation:** Deploy **Model A (Classical ML)** for the initial V1 release.

**Reasoning:**

1.  **Latency:** Model A is **100x faster** (0.15ms vs 17ms), essential for a high-throughput API.
2.  **Cost:** At **0.2MB**, Model A is significantly cheaper to host (e.g., AWS Lambda) compared to the memory-heavy Transformer.
3.  **Baseline:** While Model B is 20% more accurate, Model A provides a "good enough" baseline with near-zero operational overhead. Model B remains available as a premium/offline option.

---

## Architecture

The project follows a strict modular design pattern to ensure reproducibility and separation of concerns:

```text
├── src/
│   ├── data/           # Dataset storage
│   ├── preprocessing/  # Text cleaning pipeline
│   ├── models/         # Training logic (ML & DL)
│   ├── evaluation/     # Model evaluation utilities
│   ├── api/            # FastAPI application & SQLite logging
│   ├── config/         # Configuration management
│   └── scripts/        # Utilities (data download)
├── tests/              # Unit tests for data pipeline
├── notebooks/          # EDA and Model Comparison (Jupyter)
├── Dockerfile          # Multi-stage build for production
├── docker-compose.yml  # Multi-service setup (API + MLflow)
└── pyproject.toml      # Dependency management (uv)
```
