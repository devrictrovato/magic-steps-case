# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

FROM python:3.10-slim

# Evita .pyc e melhora logs no Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências mínimas do sistema + redis e mongodb para serviços locais
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    gcc \
    g++ \
    redis-server \
    mongodb \
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

# environment defaults (can be overridden by docker run or compose)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MONGO_URI=mongodb://mongo:27017
ENV MONGO_DB=magic_steps_logs

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
# se estiver usando contêiner único para demos, iniciamos redis e mongo antes de levantar a API
CMD ["sh", "-c", "redis-server --daemonize yes && \
              mongod --fork --logpath /var/log/mongodb.log && \
              uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info"]