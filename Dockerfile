# =============================================================
# Dockerfile — Magic Steps Prediction API (Render | CPU)
# =============================================================

FROM python:3.10.9-slim

# Dependências mínimas do sistema (torch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
 && rm -rf /var/lib/apt/lists/*

# Diretório da app
WORKDIR /app

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# PYTHONPATH para imports planos
ENV PYTHONPATH=/app/app

# Código da API
COPY app/context.py app/
COPY app/main.py    app/
COPY app/routes.py  app/

# Artefatos do modelo
COPY app/model/model_magic_steps_dl.pt app/model/
COPY out/preprocessor.joblib out/

# .env opcional
COPY .env* .

# Render fornece PORT
ENV PORT=8000
EXPOSE 8000

# Processo único (Render-friendly)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info"]