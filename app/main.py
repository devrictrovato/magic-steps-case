# main.py
# ============================================================
# Magic Steps — API FastAPI
# ============================================================
#
#   Execução:
#       uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
#   Swagger UI:
#       http://localhost:8000/docs
#
#   ReDoc:
#       http://localhost:8000/redoc
#
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path

# make project root and src directory importable for top-level modules
import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
from typing import Dict, List, Any, Optional


import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("magic_steps_api")

# ════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ════════════════════════════════════════════════════════════

# Caminhos — main.py vive em app/, mas out/ está na raiz do projeto
BASE_DIR          = Path(__file__).resolve().parent.parent   # sobe de app/ → raiz
MODEL_PATH        = BASE_DIR / "app/model" / "model_magic_steps_dl.pt"
PREPROCESSOR_PATH = BASE_DIR / "out" / "preprocessor.joblib"

# ── contexto global — importado do módulo neutro ─────────
from context import DEVICE, _MODEL_CONTEXT, get_model_context  # noqa: F401


# ════════════════════════════════════════════════════════════
# ARQUITETURA DO MODELO  (espelhada do train.py)
# ════════════════════════════════════════════════════════════

class MagicStepsNet(nn.Module):
    """
    Rede neural classificadora.

    Arquitetura: camadas fully-connected empilhadas com
    ReLU → BatchNorm → Dropout após cada hidden layer.
    Saída: 1 logit (sigmoid aplicado na inferência).
    """

    def __init__(
        self,
        input_dim:     int,
        hidden_layers: List[int],
        dropout:       float,
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

        layers.append(nn.Linear(prev_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ════════════════════════════════════════════════════════════
# CARREGAMENTO DO MODELO
# ════════════════════════════════════════════════════════════

def load_model() -> None:
    """
    Carrega o checkpoint .pt salvo pelo train.py e
    preenche o contexto global com modelo + metadados.
    """
    if not MODEL_PATH.exists():
        logger.error(
            "Arquivo de modelo não encontrado: %s", MODEL_PATH
        )
        logger.warning("A API iniciará sem modelo — /predict retornará 503.")
        return

    logger.info("Carregando modelo de %s …", MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    input_dim   = checkpoint["input_dim"]
    best_params = checkpoint["best_params"]

    model = MagicStepsNet(
        input_dim=input_dim,
        hidden_layers=best_params["hidden_layers"],
        dropout=best_params["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # desativa dropout e normaliza BatchNorm

    _MODEL_CONTEXT["model"]       = model
    _MODEL_CONTEXT["input_dim"]   = input_dim
    _MODEL_CONTEXT["best_params"] = best_params

    logger.info(
        "✅ Modelo carregado — input_dim=%d | layers=%s | dropout=%.2f | device=%s",
        input_dim,
        best_params["hidden_layers"],
        best_params["dropout"],
        DEVICE,
    )


def load_preprocessor() -> None:
    """
    Carrega o ColumnTransformer (MinMaxScaler + OrdinalEncoder)
    salvo pelo preprocessing.py e o coloca no contexto global.
    """
    if not PREPROCESSOR_PATH.exists():
        logger.error("Preprocessador não encontrado: %s", PREPROCESSOR_PATH)
        logger.warning("A API iniciará sem preprocessador — /predict retornará 503.")
        return

    import joblib  # noqa: E402 — importado aqui para não pesar no topo

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    _MODEL_CONTEXT["preprocessor"] = preprocessor

    # log das features que o preprocessador conhece
    num_cols = preprocessor.transformers_[0][2]   # MinMaxScaler
    cat_cols = preprocessor.transformers_[1][2]   # OrdinalEncoder
    logger.info(
        "✅ Preprocessador carregado — %d numéricas, %d categóricas | output_dim=%d",
        len(num_cols), len(cat_cols), len(num_cols) + len(cat_cols),
    )


# ════════════════════════════════════════════════════════════
# APLICAÇÃO FASTAPI
# ════════════════════════════════════════════════════════════

# ── instância ─────────────────────────────────────────────
app = FastAPI(
    title="🎯 Magic Steps — Prediction API",
    description=(
        "API de inferência para o modelo de predição de "
        "**atingimento do Valor Prognóstico (PV)** — Projeto PEDE 2024.\n\n"
        "### Como usar\n"
        "1. Consulte `GET /features` para ver as features esperadas, suas descrições e **intervalos válidos**.\n"
        "2. Envie os dados do aluno com **valores reais** (ex: `nota_cg: 430`, `pedra_modal: \"ametista\"`) "
        "para `POST /predict`. A API aplica o preprocessador internamente.\n"
        "3. Para processamento em lote use `POST /predict/batch` (até 500 alunos).\n\n"
        "### Pré-processamento interno\n"
        "Você **não** precisa normalizar nada. O `preprocessor.joblib` (MinMaxScaler + OrdinalEncoder) "
        "é aplicado automaticamente antes da inferência do modelo."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ajuste para produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── registra rotas do router ──────────────────────────────
from routes import router as prediction_router   # noqa: E402
app.include_router(prediction_router)


# ════════════════════════════════════════════════════════════
# EVENTOS DE CICLO DE VIDA
# ════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Carrega modelo + preprocessador no momento em que a aplicação sobe."""
    logger.info("🚀 Iniciando Magic Steps API …")
    load_model()
    load_preprocessor()
    logger.info("🟢 API pronta para receber requisições.")


@app.on_event("shutdown")
async def shutdown_event():
    """Libera recursos ao derrubar a aplicação."""
    logger.info("🔴 Encerrando Magic Steps API …")
    _MODEL_CONTEXT["model"]        = None
    _MODEL_CONTEXT["preprocessor"] = None


# ════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA  (python main.py)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn  # noqa: E402

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )