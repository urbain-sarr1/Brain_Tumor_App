#!/bin/sh

set -e

echo "🚀 Démarrage de FastAPI..."

exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 10000