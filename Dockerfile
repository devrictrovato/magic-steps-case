# =============================================================
# Dockerfile — Magic Steps Prediction API (produção | CPU)
# =============================================================
#
# Multi-stage build:
#   Stage 1 «builder»  → instala dependências numa imagem com
#                         compiladores (gcc, etc.)
#   Stage 2 «runtime»  → imagem magra; copia apenas dependências
#                         + código da API + artefactos do modelo
#
# Uso:
#   docker build -t magic-steps-api .
#   docker run -p 8000:8000 magic-steps-api
#
# =============================================================

# ─────────────────────────────────────────────────────────────
# STAGE 1 — builder: instalar dependências
# ─────────────────────────────────────────────────────────────
FROM python:3.10.9-slim AS builder

# Compiladores necessários para pacotes com extensões C
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copiar o ficheiro de dependências
COPY requirements.txt .

# Instalar dependências numa pasta isolada (não no sistema).
# Remove pywin32 (somente Windows) de forma compatível com /bin/sh
RUN grep -v pywin32 requirements.txt > requirements-linux.txt && \
    pip install --no-cache-dir --prefix=/install -r requirements-linux.txt && \
    rm requirements-linux.txt


# ─────────────────────────────────────────────────────────────
# STAGE 2 — runtime: imagem final magra
# ─────────────────────────────────────────────────────────────
FROM python:3.10.9-slim AS runtime

# ── utilizador não-root (segurança) ──────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# ── dependências mínimas do sistema ──────────────────────────
# libgomp1 → OpenMP (necessário para torch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── copiar dependências do builder ───────────────────────────
COPY --from=builder /install /usr/local

# ── PYTHONPATH ───────────────────────────────────────────────
# main.py / routes.py / context.py vivem em /app/app/
# PYTHONPATH=/app/app permite imports planos
ENV PYTHONPATH=/app/app

# ── diretório de trabalho ────────────────────────────────────
WORKDIR /app

# ── código da API ────────────────────────────────────────────
COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/

# ── artefactos do modelo ─────────────────────────────────────
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# ── .env (opcional) ──────────────────────────────────────────
COPY .env* .

# ── permissões ───────────────────────────────────────────────
RUN chown -R appuser:appuser /app

USER appuser

# ── porta ────────────────────────────────────────────────────
EXPOSE 8000

# ── healthcheck ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
  CMD wget --no-check-certificate -qO- http://localhost:8000/health || exit 1

# ── entry point ──────────────────────────────────────────────
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "1", "--log-level", "info", "--access-log", "--no-reload"]