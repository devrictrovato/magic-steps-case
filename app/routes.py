# routes.py
# ============================================================
# Rotas da API Magic Steps — predição de atingimento do PV
# ============================================================
#
# Grupo          | Rotas
# ───────────────|─────────────────────────────────────────────
# Health         | GET  /health
# Info           | GET  /info
# Predict        | POST /predict          (1 aluno)
#                | POST /predict/batch    (até 500 alunos)
# Features       | GET  /features         (schema + intervalos)
# Thresholds     | GET  /thresholds       (limiar atual)
#                | PUT  /thresholds       (atualizar limiar)
# ============================================================
#
# v2 — o cliente envia valores REAIS.
# A API aplica preprocessor.joblib internamente antes do modelo.
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from context import get_model_context

router = APIRouter()


# ════════════════════════════════════════════════════════════
# ENUMS  — valores exatos que o OrdinalEncoder conhece
# ════════════════════════════════════════════════════════════

class TurmaEnum(str, Enum):
    a="a"; b="b"; c="c"; d="d"; e="e"; f="f"
    g="g"; h="h"; i="i"; j="j"; k="k"; l="l"
    m="m"; n="n"; o="o"; p="p"; q="q"; r="r"
    s="s"; t="t"; u="u"; v="v"; y="y"; z="z"

class GeneroEnum(str, Enum):
    menina = "menina"
    menino = "menino"

class PedraEnum(str, Enum):
    ametista = "ametista"
    quartzo  = "quartzo"
    topazio  = "topázio"
    agata    = "ágata"


# ════════════════════════════════════════════════════════════
# ESQUEMAS DE ENTRADA  (valores humanos / originais da base)
# ════════════════════════════════════════════════════════════

class StudentFeatures(BaseModel):
    """
    Dados brutos de um aluno — exatamente como saem da planilha.

    Intervalos extraídos do preprocessor.joblib ajustado nos dados
    de treinamento (860 alunos, BASE DE DADOS PEDE 2024).
    """

    # ── numéricas ─────────────────────────────────────────
    fase:           int   = Field(..., ge=0,     le=7,
                                  description="Fase educacional (0 a 7)")
    idade:          int   = Field(..., ge=7,     le=21,
                                  description="Idade do aluno em anos")
    ano_ingresso:   int   = Field(..., ge=2016,  le=2022,
                                  description="Ano de ingresso no programa")
    score_inde:     float = Field(..., ge=3.0,   le=9.5,
                                  description="INDE 22 — Índice de Desenvolvimento Educacional")
    score_iaa:      float = Field(..., ge=0.0,   le=10.0,
                                  description="IAA — Índice de Aprendizagem Ativa")
    score_ieg:      float = Field(..., ge=0.0,   le=10.0,
                                  description="IEG — Índice de Engajamento")
    score_ips:      float = Field(..., ge=2.5,   le=10.0,
                                  description="IPS — Índice de Progresso Social")
    score_ida:      float = Field(..., ge=0.0,   le=9.9,
                                  description="IDA — Índice de Desempenho Acadêmico")
    score_ipv:      float = Field(..., ge=2.5,   le=10.0,
                                  description="IPV — Índice de Progresso de Valor")
    score_ian:      float = Field(..., ge=2.5,   le=10.0,
                                  description="IAN — Índice de Atingimento das Necessidades")
    nota_cg:        int   = Field(..., ge=1,     le=862,
                                  description="Caderneta Global (pontuação acumulada)")
    nota_cf:        int   = Field(..., ge=1,     le=192,
                                  description="Caderneta de Formação")
    nota_ct:        int   = Field(..., ge=1,     le=18,
                                  description="Caderneta de Tradição")
    num_avaliacoes: int   = Field(..., ge=2,     le=4,
                                  description="Número de avaliações realizadas")
    defasagem:      int   = Field(..., ge=-5,    le=2,
                                  description="Defasagem acadêmica (pode ser negativa)")

    # ── categóricas ───────────────────────────────────────
    turma:       TurmaEnum  = Field(..., description="Letra da turma (minúscula)")
    genero:      GeneroEnum = Field(..., description="Gênero do aluno")
    pedra_modal: PedraEnum  = Field(..., description="Pedra modal (melhor classificação nos 3 anos)")


class StudentInput(BaseModel):
    """Envelope com ID opcional + features."""
    student_id: Optional[str]   = Field(None, description="Identificador do aluno (opcional)")
    features:   StudentFeatures


# ════════════════════════════════════════════════════════════
# ESQUEMAS DE RESPOSTA
# ════════════════════════════════════════════════════════════

class PredictionResult(BaseModel):
    student_id:     Optional[str]
    probability:    float = Field(..., description="P(atingiu PV) entre 0 e 1")
    prediction:     int   = Field(..., description="Classe predita: 0 = não atingiu, 1 = atingiu PV")
    confidence:     str   = Field(..., description="alta / média / baixa")
    prediction_id:  str   = Field(..., description="UUID da inferência")
    timestamp:      str   = Field(..., description="Timestamp ISO-8601")


class BatchRequest(BaseModel):
    students: List[StudentInput] = Field(
        ..., min_length=1, max_length=500,
        description="Lista de até 500 alunos",
    )


class BatchResponse(BaseModel):
    total:       int
    predictions: List[PredictionResult]
    timestamp:   str


class FeatureInfo(BaseModel):
    name:        str
    description: str
    type:        str                      # int | float | categorical
    range:       Optional[str] = None


class ThresholdInfo(BaseModel):
    threshold:  float = Field(..., ge=0.0, le=1.0)
    updated_at: str


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0,
                             description="Novo limiar de classificação")


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

_THRESHOLD: float = 0.5

# Colunas na ordem que o ColumnTransformer recebe.
_INPUT_COLUMNS = [
    "fase", "idade", "ano_ingresso",
    "score_inde", "score_iaa", "score_ieg", "score_ips",
    "score_ida", "score_ipv", "score_ian",
    "nota_cg", "nota_cf", "nota_ct",
    "num_avaliacoes", "defasagem",
    "turma", "genero", "pedra_modal",
]

# Descrições estáticas para /features e /info
_FEATURE_META = {
    "fase":           ("int",   "Fase educacional do aluno"),
    "idade":          ("int",   "Idade em anos"),
    "ano_ingresso":   ("int",   "Ano de ingresso no programa"),
    "score_inde":     ("float", "INDE — Índice de Desenvolvimento Educacional"),
    "score_iaa":      ("float", "IAA  — Índice de Aprendizagem Ativa"),
    "score_ieg":      ("float", "IEG  — Índice de Engajamento"),
    "score_ips":      ("float", "IPS  — Índice de Progresso Social"),
    "score_ida":      ("float", "IDA  — Índice de Desempenho Acadêmico"),
    "score_ipv":      ("float", "IPV  — Índice de Progresso de Valor"),
    "score_ian":      ("float", "IAN  — Índice de Atingimento das Necessidades"),
    "nota_cg":        ("int",   "Caderneta Global (pontuação acumulada)"),
    "nota_cf":        ("int",   "Caderneta de Formação"),
    "nota_ct":        ("int",   "Caderneta de Tradição"),
    "num_avaliacoes": ("int",   "Número de avaliações realizadas"),
    "defasagem":      ("int",   "Defasagem acadêmica (pode ser negativa)"),
    "turma":          ("categorical", "Letra da turma"),
    "genero":         ("categorical", "Gênero do aluno"),
    "pedra_modal":    ("categorical", "Pedra modal (melhor classificação nos 3 anos)"),
}


def _confidence_label(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist >= 0.30:
        return "alta"
    if dist >= 0.15:
        return "média"
    return "baixa"


def _check_ready(ctx: dict) -> None:
    """503 se modelo ou preprocessador não estiverem carregados."""
    missing = []
    if ctx["model"] is None:
        missing.append("modelo (model_magic_steps_dl.pt)")
    if ctx["preprocessor"] is None:
        missing.append("preprocessador (preprocessor.joblib)")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Artefatos não carregados: {', '.join(missing)}. "
                   "Verifique se os arquivos existem em out/.",
        )


def _features_to_row(features: StudentFeatures) -> dict:
    """Pydantic → dict com nomes de coluna do preprocessador."""
    row = {}
    for col in _INPUT_COLUMNS:
        val = getattr(features, col)
        # enums → string do valor
        if isinstance(val, Enum):
            val = val.value
        row[col] = val
    return row


def _preprocess_and_tensorize(rows: List[dict], ctx: dict) -> torch.Tensor:
    """
    DataFrame com valores reais
      → preprocessor.transform()   (MinMaxScaler + OrdinalEncoder)
      → tensor float32 no device do modelo
    """
    df   = pd.DataFrame(rows, columns=_INPUT_COLUMNS)
    arr  = ctx["preprocessor"].transform(df).astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32).to(ctx["device"])


def _build_feature_info(ctx: dict) -> List[dict]:
    """
    Monta lista de FeatureInfo extraindo min/max do MinMaxScaler
    e categorias do OrdinalEncoder ao vivo do preprocessador.
    """
    pre = ctx.get("preprocessor")

    num_meta, cat_meta = {}, {}
    if pre is not None:
        num_trans = pre.transformers_[0][1]
        num_cols  = pre.transformers_[0][2]
        cat_trans = pre.transformers_[1][1]
        cat_cols  = pre.transformers_[1][2]

        num_meta = {
            col: (float(mn), float(mx))
            for col, mn, mx in zip(num_cols, num_trans.data_min_, num_trans.data_max_)
        }
        cat_meta = {
            col: cats.tolist()
            for col, cats in zip(cat_cols, cat_trans.categories_)
        }

    result = []
    for col in _INPUT_COLUMNS:
        dtype, desc = _FEATURE_META[col]

        if col in num_meta:
            mn, mx = num_meta[col]
            rang = f"{int(mn)} – {int(mx)}" if dtype == "int" else f"{mn} – {mx}"
        elif col in cat_meta:
            rang = " | ".join(cat_meta[col])
        else:
            rang = None

        result.append({"name": col, "description": desc, "type": dtype, "range": rang})

    return result


# ════════════════════════════════════════════════════════════
# ROTAS — HEALTH & INFO
# ════════════════════════════════════════════════════════════

@router.get("/health", status_code=status.HTTP_200_OK, tags=["Health"],
            summary="Health-check",
            description="Verifica se a API, o modelo e o preprocessador estão operacionais.")
def health():
    ctx = get_model_context()
    return {
        "status":               "ok",
        "model_loaded":         ctx["model"] is not None,
        "preprocessor_loaded":  ctx["preprocessor"] is not None,
        "device":               str(ctx["device"]),
        "timestamp":            datetime.now(timezone.utc).isoformat(),
    }


@router.get("/info", status_code=status.HTTP_200_OK, tags=["Health"],
            summary="Informações do modelo e preprocessador",
            description="Metadados completos: arquitetura, hiperparâmetros, features com intervalos.")
def info():
    ctx = get_model_context()
    return {
        "project":              "Magic Steps",
        "model_class":          "MagicStepsNet",
        "input_dim":            ctx.get("input_dim"),
        "best_params":          ctx.get("best_params"),
        "device":               str(ctx["device"]),
        "preprocessor_loaded":  ctx["preprocessor"] is not None,
        "n_features":           len(_INPUT_COLUMNS),
        "threshold":            _THRESHOLD,
        "features":             _build_feature_info(ctx),
    }


# ════════════════════════════════════════════════════════════
# ROTAS — PREDICT
# ════════════════════════════════════════════════════════════

_PREDICT_EXAMPLE = (
    "Recebe os dados **reais** de um único aluno e retorna a predição.\n\n"
    "A API aplica o preprocessador (MinMaxScaler + OrdinalEncoder) "
    "internamente — **não é necessário normalizar**.\n\n"
    "Exemplo:\n"
    "```json\n"
    '{\n'
    '  "student_id": "RA-42",\n'
    '  "features": {\n'
    '    "fase": 2,\n'
    '    "idade": 12,\n'
    '    "ano_ingresso": 2021,\n'
    '    "score_inde": 7.5,\n'
    '    "score_iaa": 8.5,\n'
    '    "score_ieg": 8.0,\n'
    '    "score_ips": 7.0,\n'
    '    "score_ida": 6.5,\n'
    '    "score_ipv": 7.8,\n'
    '    "score_ian": 5.0,\n'
    '    "nota_cg": 400,\n'
    '    "nota_cf": 70,\n'
    '    "nota_ct": 6,\n'
    '    "num_avaliacoes": 3,\n'
    '    "defasagem": -1,\n'
    '    "turma": "a",\n'
    '    "genero": "menina",\n'
    '    "pedra_modal": "ametista"\n'
    '  }\n'
    '}\n'
    "```"
)


@router.post("/predict", response_model=PredictionResult,
             status_code=status.HTTP_200_OK, tags=["Predict"],
             summary="Predição individual", description=_PREDICT_EXAMPLE)
def predict(student: StudentInput):
    ctx = get_model_context()
    _check_ready(ctx)

    row    = _features_to_row(student.features)
    tensor = _preprocess_and_tensorize([row], ctx)

    with torch.no_grad():
        prob = torch.sigmoid(ctx["model"](tensor)).item()

    return PredictionResult(
        student_id=student.student_id,
        probability=round(prob, 6),
        prediction=int(prob >= _THRESHOLD),
        confidence=_confidence_label(prob),
        prediction_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/predict/batch", response_model=BatchResponse,
             status_code=status.HTTP_200_OK, tags=["Predict"],
             summary="Predição em lote",
             description=(
                 "Até **500 alunos** com valores reais.\n\n"
                 "O preprocessador é aplicado uma vez ao DataFrame completo e o modelo "
                 "faz um único forward pass — muito mais eficiente que N chamadas individuais."
             ))
def predict_batch(body: BatchRequest):
    ctx = get_model_context()
    _check_ready(ctx)

    rows   = [_features_to_row(s.features) for s in body.students]
    tensor = _preprocess_and_tensorize(rows, ctx)

    with torch.no_grad():
        probs = torch.sigmoid(ctx["model"](tensor)).cpu().numpy()

    now     = datetime.now(timezone.utc).isoformat()
    results = []
    for student, prob in zip(body.students, probs):
        p = float(prob)
        results.append(PredictionResult(
            student_id=student.student_id,
            probability=round(p, 6),
            prediction=int(p >= _THRESHOLD),
            confidence=_confidence_label(p),
            prediction_id=str(uuid4()),
            timestamp=now,
        ))

    return BatchResponse(total=len(results), predictions=results, timestamp=now)


# ════════════════════════════════════════════════════════════
# ROTAS — FEATURES & THRESHOLDS
# ════════════════════════════════════════════════════════════

@router.get("/features", response_model=List[FeatureInfo],
            status_code=status.HTTP_200_OK, tags=["Features"],
            summary="Lista de features esperadas",
            description=(
                "Features com descrição, tipo e **intervalo válido** extraído "
                "do preprocessador ajustado. Envie valores dentro desses intervalos."
            ))
def features():
    return _build_feature_info(get_model_context())


@router.get("/thresholds", response_model=ThresholdInfo,
            status_code=status.HTTP_200_OK, tags=["Thresholds"],
            summary="Limiar atual", description="Classe = 1 se P ≥ threshold.")
def get_threshold():
    return ThresholdInfo(threshold=_THRESHOLD,
                         updated_at=datetime.now(timezone.utc).isoformat())


@router.put("/thresholds", response_model=ThresholdInfo,
            status_code=status.HTTP_200_OK, tags=["Thresholds"],
            summary="Atualizar limiar",
            description="Valor alto → conservador. Valor baixo → agressivo. Mudança imediata.")
def update_threshold(body: ThresholdUpdate):
    global _THRESHOLD
    _THRESHOLD = body.threshold
    return ThresholdInfo(threshold=_THRESHOLD,
                         updated_at=datetime.now(timezone.utc).isoformat())