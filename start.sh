#!/bin/sh

uvicorn api.main:app --host 0.0.0.0 --port 8000 &

streamlit run streamlit_app/app.py \
    --server.port=10000 \
    --server.address=0.0.0.0