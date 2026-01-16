# 1. Base Image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install 'uv'
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Install Dependencies
# Copy lockfiles first to leverage Docker caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 5. Copy the Source Code
COPY src/ src/

# 6. PIPELINE EXECUTION

# Step A: Download Data
RUN uv run python src/data/make_dataset.py

# Step B: Train Classical ML Model
RUN uv run python src/models/train_ml.py

# Step C: Train Deep Learning Model
RUN uv run python src/models/train_dl.py

# 7. Setup Environment
ENV PATH="/app/.venv/bin:$PATH"

# 8. Expose & Run
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]