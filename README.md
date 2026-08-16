services:
  - type: web
    name: brain-tumor-api
    env: docker
    dockerfilePath: api/Dockerfile
    dockerContext: api
    plan: starter
    healthCheckPath: /health

    envVars:
      - key: MODEL_PATH
        value: /app/models/best.pt
      - key: MAX_FILE_SIZE_MB
        value: "10"
      - key: CONFIDENCE_THRESHOLD
        value: "0.5"
      - key: ALLOWED_ORIGINS
        sync: false

  - type: web
    name: brain-tumor-ui
    env: docker
    dockerfilePath: streamlit_app/Dockerfile
    dockerContext: streamlit_app
    plan: starter

    envVars:
      - key: API_URL
        sync: false