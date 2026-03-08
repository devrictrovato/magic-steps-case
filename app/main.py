# main.py
# ============================================================
# Magic Steps — API FastAPI
# ============================================================
#
#   Execução:
#       uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#   Swagger UI:  http://localhost:8000/docs
#   ReDoc:       http://localhost:8000/redoc
#
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path

import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("magic_steps_api")

# ════════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════════

BASE_DIR          = Path(__file__).resolve().parent.parent
MODEL_PATH        = BASE_DIR / "app/model" / "model_magic_steps_dl.pt"
PREPROCESSOR_PATH = BASE_DIR / "out" / "preprocessor.joblib"

# ── contexto global ───────────────────────────────────────────────────────────
from .context import DEVICE, _MODEL_CONTEXT, get_model_context
from src.db import ensure_schema


# ════════════════════════════════════════════════════════════════════════════════
# ARQUITETURA DO MODELO
# Espelha EXATAMENTE o train.py — num_classes variável, sem .squeeze(1)
# ════════════════════════════════════════════════════════════════════════════════

class MagicStepsNet(nn.Module):
    """
    Rede neural classificadora multiclasse.
    Arquitetura: FC → ReLU → BatchNorm → Dropout (repetido por camada oculta).
    Saída: logits shape (batch, num_classes).  Softmax aplicado na inferência.
    """

    def __init__(
        self,
        input_dim:     int,
        hidden_layers: List[int],
        dropout:       float,
        num_classes:   int = 3,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retorna logits (batch, num_classes) — sem squeeze, sem sigmoid
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DO MODELO
# ════════════════════════════════════════════════════════════════════════════════

def load_model() -> None:
    """
    Carrega o checkpoint .pt salvo pelo train.py.
    Usa num_classes do próprio checkpoint (default 3 para compatibilidade).
    """
    if not MODEL_PATH.exists():
        logger.error("Arquivo de modelo não encontrado: %s", MODEL_PATH)
        logger.warning("A API iniciará sem modelo — /predict retornará 503.")
        return

    logger.info("Carregando modelo de %s …", MODEL_PATH)

    checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
    input_dim   = checkpoint["input_dim"]
    best_params = checkpoint["best_params"]
    num_classes = checkpoint.get("num_classes", 3)

    model = MagicStepsNet(
        input_dim=input_dim,
        hidden_layers=best_params["hidden_layers"],
        dropout=best_params["dropout"],
        num_classes=num_classes,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _MODEL_CONTEXT["model"]       = model
    _MODEL_CONTEXT["input_dim"]   = input_dim
    _MODEL_CONTEXT["best_params"] = best_params
    _MODEL_CONTEXT["num_classes"] = num_classes

    logger.info(
        "✅ Modelo carregado — input_dim=%d | layers=%s | dropout=%.2f | "
        "num_classes=%d | device=%s",
        input_dim, best_params["hidden_layers"], best_params["dropout"],
        num_classes, DEVICE,
    )


def load_preprocessor() -> None:
    """
    Carrega o ColumnTransformer (MinMaxScaler + OrdinalEncoder) salvo pelo
    preprocessing.py.  As features (e a ordem delas) são determinadas pelo
    preprocessador — a API usa exatamente as mesmas colunas do treino.
    """
    if not PREPROCESSOR_PATH.exists():
        logger.error("Preprocessador não encontrado: %s", PREPROCESSOR_PATH)
        logger.warning("A API iniciará sem preprocessador — /predict retornará 503.")
        return

    import joblib

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    _MODEL_CONTEXT["preprocessor"] = preprocessor

    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    # ── detectar se o modelo usa feature engineering ──────────────────────
    # Se input_dim > len(num_cols) + len(cat_cols), o treino aplicou FE antes
    # de preprocessar.  Armazenamos essa informação no contexto para que
    # routes.py saiba se deve rodar FeatureEngineer antes de preprocessar.
    base_feature_count = len(num_cols) + len(cat_cols)
    input_dim = _MODEL_CONTEXT.get("input_dim")

    if input_dim is not None and input_dim > base_feature_count:
        _MODEL_CONTEXT["uses_feature_engineering"] = True
        logger.info(
            "🔍 Modelo treinado COM feature engineering "
            "(input_dim=%d > base=%d — diferença de %d features extras).",
            input_dim, base_feature_count, input_dim - base_feature_count,
        )
    else:
        _MODEL_CONTEXT["uses_feature_engineering"] = False

    logger.info(
        "✅ Preprocessador carregado — %d numéricas, %d categóricas | "
        "base_output_dim=%d",
        len(num_cols), len(cat_cols), base_feature_count,
    )


# ════════════════════════════════════════════════════════════════════════════════
# APLICAÇÃO FASTAPI
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🎯 Magic Steps — Prediction API",
    description=(
        "API de inferência para o modelo de predição de "
        "**defasagem escolar** — Projeto PEDE 2024.\n\n"
        "### Como usar\n"
        "1. `GET /features` — veja as features esperadas e intervalos válidos.\n"
        "2. `POST /predict` — envie dados reais de um aluno; "
        "o preprocessamento e o feature engineering são aplicados internamente.\n"
        "3. `POST /predict/batch` — até 500 alunos por chamada.\n\n"
        "### Pré-processamento interno\n"
        "Você **não** precisa normalizar nada. O pipeline "
        "(MinMaxScaler + OrdinalEncoder + Feature Engineering) "
        "é aplicado automaticamente antes da inferência."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── rotas ─────────────────────────────────────────────────────────────────────
from .routes import router as prediction_router  # noqa: E402
from .root_route import router as root_router    # noqa: E402
app.include_router(root_router)
app.include_router(prediction_router)


# ════════════════════════════════════════════════════════════════════════════════
# EVENTOS DE CICLO DE VIDA
# ════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Magic Steps API …")

    # ── PostgreSQL schema ─────────────────────────────────────────────────
    try:
        from settings import settings as _s
        ensure_schema(_s.database_url)
        logger.info("✅ Schema PostgreSQL verificado.")
    except Exception as e:
        logger.warning("Não foi possível inicializar schema PostgreSQL: %s", e)

    # ── modelo e preprocessador ───────────────────────────────────────────
    load_model()
    load_preprocessor()
    logger.info("🟢 API pronta para receber requisições.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔴 Encerrando Magic Steps API …")
    _MODEL_CONTEXT["model"]        = None
    _MODEL_CONTEXT["preprocessor"] = None


# ════════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")