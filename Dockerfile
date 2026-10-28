FROM python:3.11.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY config ./config

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
