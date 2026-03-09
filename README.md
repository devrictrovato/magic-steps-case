# 🎯 Magic Steps — Prediction API

> API de inferência para predição de **atingimento do Valor Prognóstico (PV)** — Projeto PEDE 2024.

Pipeline MLOps completo com rede neural PyTorch, feature store via Feast/Redis, autenticação JWT, logging MongoDB e deploy Docker-ready.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Pré-requisitos](#-pré-requisitos)
- [Configuração](#-configuração)
- [Instalação e Execução](#-instalação-e-execução)
- [Docker](#-docker)
- [Endpoints da API](#-endpoints-da-api)
- [Features do Modelo](#-features-do-modelo)
- [Pipeline MLOps](#-pipeline-mlops)
- [Testes](#-testes)

---

## 🔍 Visão Geral

O Magic Steps é um microserviço de Machine Learning que prediz se um aluno atingirá o seu **Valor Prognóstico (PV)** com base em indicadores acadêmicos e socioeconômicos coletados pelo programa PEDE.

**O que a API faz:**
- Recebe dados brutos de um aluno (sem necessidade de normalização prévia)
- Aplica automaticamente o pré-processador (`MinMaxScaler` + `OrdinalEncoder`)
- Executa a rede neural (`MagicStepsNet`) e retorna a probabilidade e classe predita
- Registra cada inferência no MongoDB para monitoramento e auditoria
- Suporta predições individuais e em lote (até 500 alunos por requisição)

---

## 🏗️ Arquitetura

```
Cliente HTTP
     │
     ▼
┌─────────────────────────────────────┐
│           FastAPI (app/)            │
│  ┌──────────┐   ┌─────────────────┐ │
│  │  routes  │──▶│   MagicStepsNet │ │
│  │  (JWT)   │   │   (PyTorch)     │ │
│  └──────────┘   └─────────────────┘ │
│        │               │            │
│        ▼               ▼            │
│  ┌──────────┐   ┌─────────────────┐ │
│  │ MongoDB  │   │  preprocessor   │ │
│  │ (logger) │   │   (.joblib)     │ │
│  └──────────┘   └─────────────────┘ │
└─────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│ Redis/Feast  │  ← Feature Store online
└──────────────┘
```

**Stack principal:** Python 3.10 · FastAPI · PyTorch · scikit-learn · Feast · Redis · MongoDB · Docker

---

## 📁 Estrutura do Projeto

```
magic-steps-case/
├── app/
│   ├── main.py              # Entrypoint FastAPI — carrega modelo e preprocessador
│   ├── routes.py            # Rotas de predição, features e thresholds
│   ├── context.py           # Estado global compartilhado (modelo, device, preprocessador)
│   └── model/
│       └── model_magic_steps_dl.pt   # Checkpoint do modelo treinado
├── src/
│   ├── feature_engineering.py  # Define e materializa features no Feast
│   ├── preprocessing.py         # Pré-processamento e geração do preprocessor.joblib
│   ├── train.py                 # Treinamento da rede neural com grid search
│   ├── evaluate.py              # Avaliação de métricas e geração de relatórios
│   ├── settings.py              # Configurações centralizadas (paths, env, modelo)
│   └── utils.py                 # AWSClient, DWClient, FileManager, ModelRegistry
├── notebooks/
│   └── magic_steps_analytics.ipynb   # Análise exploratória e experimentos
├── tests/
│   ├── test_preprocessing.py    # Testes do pipeline de pré-processamento
│   └── test_model.py            # Testes de predição e validações
├── out/
│   └── preprocessor.joblib      # ColumnTransformer serializado
├── Dockerfile
├── requirements.txt
└── .env                         # Variáveis de ambiente (não versionado)
```

---

## ✅ Pré-requisitos

- Python **3.10+**
- Docker (opcional, para deploy containerizado)
- Redis (para Feature Store via Feast)
- MongoDB (para logging de predições)

---

## ⚙️ Configuração

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Autenticação JWT
SECRET_KEY=sua_chave_secreta_aqui
ACCESS_TOKEN_EXPIRE_MINUTES=30

# MongoDB (logging de predições)
MONGO_URI=mongodb://localhost:27017
MONGO_DB=magic_steps

# Redis (Feast online store)
REDIS_HOST=localhost
REDIS_PORT=6379

# AWS (opcional — para artefatos no S3)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET=magic-steps-ml

# MLflow (opcional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=magic_steps_experiment
```

> **Nota:** O `REDIS_HOST` padrão é `localhost`. Se Redis não estiver disponível, a rota `/monitor/features` retornará lista vazia sem gerar erro.

---

## 🚀 Instalação e Execução

### Local (desenvolvimento)

```bash
# 1. Clone o repositório
git clone https://github.com/devrictrovato/magic-steps-case.git
cd magic-steps-case

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Inicie a API
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Acesse a documentação interativa em **http://localhost:8000/docs**

### Fluxo básico de uso

```bash
# 1. Cadastrar usuário
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "senha123"}'

# 2. Obter token JWT
curl -X POST http://localhost:8000/token \
  -d "username=user&password=senha123"

# 3. Fazer uma predição
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <seu_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "RA-42",
    "features": {
      "fase": 2, "idade": 12, "ano_ingresso": 2021,
      "score_inde": 7.5, "score_iaa": 8.5, "score_ieg": 8.0,
      "score_ips": 7.0, "score_ida": 6.5, "score_ipv": 7.8,
      "score_ian": 5.0, "nota_cg": 400, "nota_cf": 70,
      "nota_ct": 6, "num_avaliacoes": 3, "defasagem": -1,
      "turma": "a", "genero": "menina", "pedra_modal": "ametista"
    }
  }'
```

---

## 🐳 Docker

### Build e execução

```bash
# Build da imagem
docker build -t magic-steps-api .

# Executar o contêiner
docker run -p 8000:8000 \
  -e SECRET_KEY="sua_chave_aqui" \
  -e MONGO_URI="mongodb://localhost:27017" \
  magic-steps-api
```

O contêiner inicializa Redis e MongoDB internamente. A API ficará disponível em `http://localhost:8000`.

> **Para produção**, recomenda-se separar os serviços com `docker-compose`:
> ```yaml
> services:
>   api:     { build: . }
>   redis:   { image: redis:7-alpine }
>   mongodb: { image: mongo:6 }
> ```

---

## 📡 Endpoints da API

| Método | Rota | Descrição | Auth |
|--------|------|-----------|------|
| `POST` | `/register` | Cadastrar novo usuário | ❌ |
| `POST` | `/token` | Obter token JWT | ❌ |
| `GET` | `/health` | Status da API, modelo e preprocessador | ❌ |
| `GET` | `/info` | Metadados completos do modelo e features | ❌ |
| `GET` | `/features` | Lista de features com tipos e intervalos válidos | ❌ |
| `GET` | `/thresholds` | Limiar de classificação atual | ✅ |
| `PUT` | `/thresholds` | Atualizar limiar sem reiniciar a API | ✅ |
| `POST` | `/predict` | Predição para um único aluno | ✅ |
| `POST` | `/predict/batch` | Predição em lote (até 500 alunos) | ✅ |
| `GET` | `/monitor/logs` | Histórico de predições (filtrável) | ✅ |
| `GET` | `/monitor/features` | Features registradas no Redis/Feast | ✅ |

### Exemplo de resposta — `/predict`

```json
{
  "student_id": "RA-42",
  "probability": 0.823456,
  "prediction": 1,
  "confidence": "alta",
  "prediction_id": "a3f2c1d0-...",
  "timestamp": "2024-11-01T14:32:00Z"
}
```

**Níveis de confiança:** `alta` (|P − 0.5| ≥ 0.30) · `média` (≥ 0.15) · `baixa` (< 0.15)

---

## 📊 Features do Modelo

O modelo utiliza **18 features** extraídas da base PEDE 2024 (860 alunos):

| Feature | Tipo | Intervalo |
|---------|------|-----------|
| `fase` | int | 0 – 7 |
| `idade` | int | 7 – 21 anos |
| `ano_ingresso` | int | 2016 – 2022 |
| `score_inde` | float | 3.0 – 9.5 |
| `score_iaa` | float | 0.0 – 10.0 |
| `score_ieg` | float | 0.0 – 10.0 |
| `score_ips` | float | 2.5 – 10.0 |
| `score_ida` | float | 0.0 – 9.9 |
| `score_ipv` | float | 2.5 – 10.0 |
| `score_ian` | float | 2.5 – 10.0 |
| `nota_cg` | int | 1 – 862 |
| `nota_cf` | int | 1 – 192 |
| `nota_ct` | int | 1 – 18 |
| `num_avaliacoes` | int | 2 – 4 |
| `defasagem` | int | -5 – 2 |
| `turma` | categorical | a – z |
| `genero` | categorical | menina \| menino |
| `pedra_modal` | categorical | ametista \| quartzo \| topázio \| ágata |

> O pré-processamento (`MinMaxScaler` + `OrdinalEncoder`) é aplicado **automaticamente** pela API — envie os valores brutos sem normalização.

---

## 🔬 Pipeline MLOps

```
Dados brutos (PEDE)
       │
       ▼
preprocessing.py       → preprocessor.joblib
       │
       ▼
feature_engineering.py → Feast feature store (Redis)
       │
       ▼
train.py               → model_magic_steps_dl.pt
       │                  (grid search + early stopping)
       ▼
evaluate.py            → métricas, relatórios, artefatos
       │
       ▼
API (FastAPI + Docker)  → inferência em produção
       │
       ▼
MongoDB                → logs de uso e auditoria
```

### Modelo — `MagicStepsNet`

Rede neural fully-connected com:
- Camadas ocultas configuráveis (`hidden_layers`)
- Ativação ReLU → BatchNorm → Dropout após cada camada
- Saída: 1 logit com sigmoid na inferência
- Otimização via grid search sobre arquitetura, learning rate e batch size
- Early stopping com `patience=10`

---

## 🧪 Testes

```bash
# Rodar toda a suíte de testes
pytest

# Com cobertura
pytest --cov=app --cov=src
```

A suíte cobre autenticação JWT, predições individuais e em lote, validação de features fora do intervalo e logging com `MongoLogger` mockado.

---

## 🤝 Contribuição

1. Fork o repositório
2. Crie uma branch: `git checkout -b feat/minha-feature`
3. Commit: `git commit -m "feat: descrição da mudança"`
4. Push: `git push origin feat/minha-feature`
5. Abra um Pull Request

---

<p align="center">
  Desenvolvido com ❤️ para o Projeto PEDE 2024
</p>
