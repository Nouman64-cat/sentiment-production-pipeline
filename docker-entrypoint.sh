#!/bin/sh
set -e

# Copy pre-trained mlruns to the shared volume (if not already present)
if [ -d "/app/mlruns_build" ] && [ ! -f "/app/mlruns/.copied" ]; then
    echo "Copying MLflow experiments to shared volume..."
    cp -r /app/mlruns_build/* /app/mlruns/ 2>/dev/null || true
    touch /app/mlruns/.copied
    echo "MLflow experiments copied successfully."
fi

# Execute the main command
exec "$@"
