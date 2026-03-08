# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

FROM python:3.10-slim

# Evita .pyc e melhora logs no Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Atualiza certificados SSL e dependências mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    gcc \
    g++ \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Diretório da aplicação
WORKDIR /app

# Copia apenas requirements primeiro (melhor cache)
COPY requirements.txt .

# Atualiza pip e instala dependências
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install certifi

# PYTHONPATH para imports planos
ENV PYTHONPATH=/app/app:/app:/app/src

# =============================
# Variáveis de ambiente
# =============================

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# URI corrigida MongoDB Atlas
ENV MONGO_URI="mongodb+srv://dbuser:6r2jt27yAF7Sn4ZH@cluster-magic-steps.ksobi7u.mongodb.net/magic_steps_logs?retryWrites=true&w=majority&tls=true"

ENV MONGO_DB=magic_steps_logs

# =============================
# Código da API
# =============================

COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/

# incluir código auxiliar
COPY src/ src/

# Artefatos do modelo
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# .env opcional
COPY .env* .

# Porta padrão Render
ENV PORT=8000
EXPOSE 8000

# =============================
# Inicialização da API
# =============================

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]