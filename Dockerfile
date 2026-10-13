FROM python:3.15-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser app.py ./app.py
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/fraud_pipeline.pkl models/metrics.json ./models/

USER appuser
EXPOSE 8501

CMD ["sh", "-c", "exec streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}"]
