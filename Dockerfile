# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

FROM python:3.10-slim

# Evita .pyc e melhora logs no Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências mínimas do sistema (scikit-learn, scipy, psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Diretório da aplicação
WORKDIR /app

# Copia apenas requirements primeiro (melhor cache)
COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# PYTHONPATH para imports planos
ENV PYTHONPATH=/app/app

# Código da API (mantendo estrutura original)
COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/

# Artefatos do modelo
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# .env opcional
COPY .env* .

# Porta padrão Render
ENV PORT=8000
EXPOSE 8000

# Processo único (Render-friendly)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info"]