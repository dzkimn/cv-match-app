# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="PASD Praktikum - Telkom University"
LABEL description="CV-Match AI — Semantic CV Profiling & Job Matcher"
LABEL version="1.0"

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Create uploads directory
RUN mkdir -p /app/uploads

# Copy source code
COPY app.py                 ./app.py
COPY cv_engine.py           ./cv_engine.py
COPY generate_jobs.py       ./generate_jobs.py
COPY job_roles_dataset.csv  ./job_roles_dataset.csv
COPY templates/             ./templates/

# Set ownership to non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Production: gunicorn dengan 4 worker
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "4", "--timeout", "120"]

