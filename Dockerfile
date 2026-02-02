# =============================================================
# Dockerfile — Magic Steps Prediction API (produção)
# =============================================================
#
# Multi-stage build:
#   Stage 1 «builder»  → instala dependências numa imagem com
#                         compiladores (gcc, etc.) e gera wheels.
#   Stage 2 «runtime»  → imagem magra; copia apenas wheels +
#                         código da API + artefactos do modelo.
#
# Uso:
#   docker build -t magic-steps-api .
#   docker run -p 8000:8000 magic-steps-api
#
# Com GPU (NVIDIA):
#   docker build --platform linux/amd64 -t magic-steps-api .
#   docker run --gpus all -p 8000:8000 magic-steps-api
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

# Copiar o ficheiro de dependências do requirements.txt
COPY requirements.txt .

# Instalar dependências numa pasta isolada (não no sistema).
# --no-cache-dir     → não guardar cache do pip dentro da imagem
# --prefix=/install  → instalar em /install para copiar depois
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt


# ─────────────────────────────────────────────────────────────
# STAGE 2 — runtime: imagem final magra
# ─────────────────────────────────────────────────────────────
FROM python:3.10.9-slim AS runtime

# ── utilizador não-root (segurança) ──────────────────────
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# ── dependências do sistema mínimas ──────────────────────
# libgomp1  → OpenMP, necessário pelo torch
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── copiar dependências compiladas do builder ────────────
COPY --from=builder /install /usr/local

# ── PYTHONPATH ────────────────────────────────────────────
# main.py / routes.py / context.py vivem em /app/app/ e usam
# imports planos ("from context import …", não "from app.context").
# PYTHONPATH=/app/app torna esses módulos visíveis ao Python.
ENV PYTHONPATH=/app/app

# ── diretório de trabalho ─────────────────────────────────
WORKDIR /app

# ── código da API (apenas os 3 ficheiros que o runtime precisa)
COPY app/context.py   app/
COPY app/main.py      app/
COPY app/routes.py    app/

# ── artefactos do modelo ──────────────────────────────────
# main.py resolve os caminhos com:
#   BASE_DIR = Path(__file__).resolve().parent.parent  →  /app
#   MODEL_PATH        = BASE_DIR / "out" / "model_magic_steps_dl.pt"
#   PREPROCESSOR_PATH = BASE_DIR / "out" / "preprocessor.joblib"
COPY out/model_magic_steps_dl.pt   out/
COPY out/preprocessor.joblib       out/

# ── .env (opcional) ───────────────────────────────────────
# pydantic-settings lê .env se presente; se não existir usa defaults.
# O wildcard .env* impede que o COPY falhe quando o ficheiro não existe.
COPY .env* .

# ── ownership para utilizador não-root ────────────────────
RUN chown -R appuser:appuser /app

USER appuser

# ── porta ─────────────────────────────────────────────────
EXPOSE 8000

# ── healthcheck ───────────────────────────────────────────
# wget está disponível no slim; sem necessidade de instalar curl.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
  CMD wget --no-check-certificate -qO- http://localhost:8000/health || exit 1

# ── entry point ───────────────────────────────────────────
# "main:app" → Python procura main.py no PYTHONPATH (/app/app/)
#              e dentro dele a instância FastAPI chamada "app".
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "1", "--log-level", "info", "--access-log", "--no-reload"]