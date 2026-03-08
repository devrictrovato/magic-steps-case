# ============================================================
# Magic Steps MLOps — Dockerfile
# ============================================================
FROM python:3.10-slim

WORKDIR /app

# ── dependências do sistema ───────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── dependências Python ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── código da aplicação ───────────────────────────────────────
COPY . .

# ── variáveis de ambiente padrão (sobrescritas pelo .env) ─────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATABASE_URL=postgresql://db_magic_steps_user:0hT1fLv2GkMb4McXAhZeUBAJUTgQtSLV@dpg-d6mo2hlm5p6s73fv0dt0-a.virginia-postgres.render.com/db_magic_steps

EXPOSE 8000

# ── healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]