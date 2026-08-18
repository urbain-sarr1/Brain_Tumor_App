#!/bin/sh

set -e

echo "🚀 Démarrage de FastAPI..."

uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 &

echo "🚀 Démarrage de Streamlit..."

exec streamlit run streamlit_app/app.py \
    --server.port 10000 \
    --server.address 0.0.0.0