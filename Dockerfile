# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

# ─────────────────────────────────────────────────────────────
# STAGE 1 — builder
# ─────────────────────────────────────────────────────────────
FROM python:3.10.9-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Remove pywin32 (Windows-only)
RUN grep -v pywin32 requirements.txt > requirements-linux.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements-linux.txt && \
    rm requirements-linux.txt


# ─────────────────────────────────────────────────────────────
# STAGE 2 — runtime
# ─────────────────────────────────────────────────────────────
FROM python:3.10.9-slim

# Dependência mínima para torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar libs instaladas
COPY --from=builder /install /usr/local

ENV PYTHONPATH=/app/app
WORKDIR /app

# Código da API
COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/

# Artefatos do modelo
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# .env opcional
COPY .env* .

# Render fornece PORT automaticamente
ENV PORT=8000
EXPOSE 8000

# ⚠️ Render quer 1 processo simples
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info"]