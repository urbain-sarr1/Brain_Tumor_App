FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY streamlit_app ./streamlit_app

COPY start.sh .
RUN chmod +x start.sh

ENV API_URL=http://127.0.0.1:8000

EXPOSE 10000

CMD ["./start.sh"]