# Magic Steps Prediction API

API de inferência para o modelo de predição de **atingimento do Valor Prognóstico (PV)** — projeto PEDE 2024.

## Recursos principais

- **Autenticação JWT**: endpoints protegidos por token, com cadastro (`/register`) e obtenção de token (`/token`).
- **Logger de uso**: todas as predições são registradas em um banco MongoDB via `MongoLogger`.
- **Feature Store**: integração com Feast para armazenar e materializar features usando Redis como store online.
- **Microserviço FastAPI** com rotas de predição individual e em lote, consultas de metadados e limiar ajustável.
- **Dockerizado**: imagem preparada para produção com dependências atualizadas.

## Utilização

1. Configure variáveis de ambiente em `.env` (modelo incluído no repositório).
   - `SECRET_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES` para JWT
   - `MONGO_URI`, `MONGO_DB` para logs de predição

   **Nota:** Por padrão o host de Redis é `localhost`; se você estiver
   executando fora de Docker certifique‑se de que o serviço esteja ativo ou
   ajuste `REDIS_HOST` em `.env`. A rota de monitoramento retornará lista vazia
   em vez de erro caso não consiga conectar.

2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute em modo de desenvolvimento:
   ```bash
   cd app
   uvicorn main:app --reload
   ```

4. Acesse Swagger UI: `http://localhost:8000/docs`.

5. Fluxo básico:
   - `POST /register` para criar usuário
   - `POST /token` para obter token JWT
   - Use token em `Authorization: Bearer <token>` para demais chamadas
   - Predição em `/predict` ou `/predict/batch`
   - Os logs de uso aparecerão na coleção `predictions` do MongoDB

## Características MLOps

- **Feature Engineering**: `src/feature_engineering.py` define e prepara features, salva em Feast (no Redis online store).
- **Treinamento**: `src/train.py` carrega features (opcional via Feast) e treina a rede neural.
- **Monitoramento**: MongoDB armazena cada requisição de predição, incluindo usuário, features e saída. A API expõe rotas adicionais para consultar esses logs e as features que foram registradas:
  - `GET /monitor/logs` – retorna todos os registros de predição (pode ser filtrado por usuário ou `student_id`).
  - `GET /monitor/features` – retorna a lista de chaves presentes no store online Redis (Feast).
  Ambos exigem autenticação JWT.

## Docker

O `Dockerfile` já está configurado para produção e inclui os serviços
necessários para rodar Redis e MongoDB localmente (gestão de features via
Feast e logs de uso).

### O que o Dockerfile faz

> **Atenção:** o código de configuração (`src/settings.py`) agora resolve o
> caminho do `.env` usando o diretório raiz do projeto em vez do diretório
> atual. Isso garante que variáveis como `MONGO_URI` definidas em `.env` sejam
> sempre carregadas mesmo quando Uvicorn é iniciado a partir de `app/`.


```dockerfile
FROM python:3.10-slim

# Instala dependências de compilação e os serviços Redis/Mongo
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

# Durante o bootstrap iniciamos redis e mongodb em background e depois a API
CMD ["sh", "-c", "redis-server --daemonize yes && \
              mongod --fork --logpath /var/log/mongodb.log && \
              uvicorn main:app --host 0.0.0.0 --port ${PORT} --log-level info"]
```

### Executando

```bash
docker build -t magic-steps-api .
docker run -e PORT=8000 -e SECRET_KEY="..." \
           -e MONGO_URI="mongodb://localhost:27017" \
           magic-steps-api
```

O contêiner iniciará Redis e MongoDB internamente; a API estará disponível
em `http://localhost:8000` e os dados de uso serão gravados no banco embutido.

> **Obs.** Para produção real recomendo usar um `docker-compose.yml` com
> contêineres separados para `redis` e `mongo`, mas a configuração acima serve
> para ambientes simples ou demonstrações.

## Testes

Executar `pytest` no diretório raiz. A suíte já cobre autenticação, predições, validações e logs (com `MongoLogger` mockado).

---

Este repositório serve como base para um pipeline MLOps leve com foco em reproducibilidade, observabilidade e deploy rápido.