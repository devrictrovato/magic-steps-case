# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

FROM python:3.10-slim

# Evita .pyc e melhora logs no Render
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências mínimas do sistema (REMOVIDO mongodb e redis)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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
# include project root, the app package, and src helpers so modules can be imported
ENV PYTHONPATH=/app/app:/app:/app/src

# Variáveis de ambiente (usar serviços externos no Render)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV MONGO_URI=mongodb+srv://dbuser:6r2jt27yAF7Sn4ZH@cluster-magic-steps.ksobi7u.mongodb.net/?appName=cluster-magic-steps
ENV MONGO_DB=magic_steps_logs

# Código da API
COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/
# incluir código auxiliar que vive em src
COPY src/ src/

# Artefatos do modelo
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# .env opcional
COPY .env* .

# Porta padrão Render
ENV PORT=8000
EXPOSE 8000

# Apenas sobe a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]