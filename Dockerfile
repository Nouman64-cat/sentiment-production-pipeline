# 1. Base Image
FROM python:3.13-slim

# 2. Set working directory
WORKDIR /app

# 3. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Install Dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 5. Copy the Source Code
COPY src/ src/

# 6. Install the project (so 'src' becomes an importable package)
RUN uv sync --frozen

# Set PYTHONPATH to ensure src module is found
ENV PYTHONPATH="/app"

# 7. Create mlflow data directory and set tracking URI
RUN mkdir -p /app/mlflow_data
ENV MLFLOW_TRACKING_URI="sqlite:////app/mlflow_data/mlflow.db"

# 8. PIPELINE EXECUTION

# Step A: Download Data
RUN uv run python src/scripts/download_dataset.py

# Step B: Train Classical ML Model
RUN uv run python src/models/train_ml.py

# Step C: Train Deep Learning Model
RUN uv run python src/models/train_dl.py

# 9. Save mlflow data to build directory (will be copied to volume at runtime)
RUN cp -r /app/mlflow_data /app/mlflow_data_build

# 10. Setup Environment
ENV PATH="/app/.venv/bin:$PATH"

# 11. Copy and setup entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# 12. Expose & Run
EXPOSE 8000
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]