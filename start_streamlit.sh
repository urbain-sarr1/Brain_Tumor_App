#!/bin/sh

set -e

echo "🚀 Démarrage de Streamlit..."

exec streamlit run streamlit_app/app.py \
    --server.port 10000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false