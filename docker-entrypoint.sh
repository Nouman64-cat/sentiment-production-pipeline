#!/bin/sh
set -e

# Copy pre-trained mlflow database to the shared volume (if not already present)
if [ -f "/app/mlflow_data_build/mlflow.db" ] && [ ! -f "/app/mlflow_data/mlflow.db" ]; then
    echo "Copying MLflow database to shared volume..."
    cp /app/mlflow_data_build/mlflow.db /app/mlflow_data/mlflow.db
    echo "MLflow database copied successfully."
fi

# Execute the main command
exec "$@"
