# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="PASD Praktikum - Telkom University"
LABEL description="ShopeeSentiment — Analisis Sentimen Ulasan Shopee"
LABEL version="2.0"

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy source code
COPY app.py             ./app.py
COPY sentiment_engine.py ./sentiment_engine.py
COPY templates/         ./templates/
COPY static/            ./static/
COPY Shopee_Sampled_Reviews.csv ./Shopee_Sampled_Reviews.csv

# Non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Production: gunicorn dengan 4 worker
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120"]
