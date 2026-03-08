# routes.py  v2.2
# ============================================================
# Pipeline de inferência corrigido:
#
#   INPUT (16 features brutas do usuário)
#     ↓
#   preprocessor.transform()   →  16 features normalizadas [0,1]
#     ↓
#   _apply_fe_on_normalized()  →  + 7 features de FE
#     ↓
#   tensor (1 × 23)            →  modelo
#
# Ordem das 23 colunas deve ser idêntica ao dataset de treino.
# As colunas FE são calculadas APÓS a normalização, exatamente
# como feito em feature_engineering.py durante o treino.
# ============================================================

from __future__ import annotations

import logging
import math
from pathlib import Path

import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, List, Optional, Dict
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import jwt

from context import get_model_context
from settings import settings
from utils import PostgresLogger
from db import (
    ensure_schema,
    log_prediction as db_log_prediction,
    query_predictions,
    get_drift_summary,
    list_model_runs,
    query_monitoring_logs,
    log_monitoring_event,
)

logger = logging.getLogger("magic_steps_routes")

try:
    ensure_schema(settings.database_url)
except Exception as _e:
    logger.warning("Não foi possível inicializar schema PostgreSQL: %s", _e)


# ============================================================
# AUTH
# ============================================================

SECRET_KEY                  = settings.secret_key
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
oauth2_scheme               = OAuth2PasswordBearer(tokenUrl="/token")
fake_users_db: Dict[str, Dict] = {}


class Token(BaseModel):
    access_token: str
    token_type:   str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username:  str
    full_name: Optional[str] = None
    disabled:  Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username:  str
    password:  str
    full_name: Optional[str] = None


def get_password_hash(password: str) -> str:
    return password

def get_user(db, username: str) -> Optional[UserInDB]:
    u = db.get(username)
    return UserInDB(**u) if u else None

def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    return get_user(db, username)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None: raise exc
    except jwt.PyJWTError:
        raise exc
    user = get_user(fake_users_db, username)
    if user is None: raise exc
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ============================================================
# ENUMS & ESQUEMAS
# ============================================================

_DEFASAGEM_LABELS: Dict[int, str] = {0: "atraso", 1: "neutro", 2: "avanço"}
pg_logger = PostgresLogger()
router    = APIRouter()


class TurmaEnum(str, Enum):
    a="a"; b="b"; c="c"; d="d"; e="e"; f="f"
    g="g"; h="h"; i="i"; j="j"; k="k"; l="l"
    m="m"; n="n"; o="o"; p="p"; q="q"; r="r"
    s="s"; t="t"; u="u"; v="v"; y="y"; z="z"

class GeneroEnum(str, Enum):
    menina="menina"; menino="menino"

class PedraEnum(str, Enum):
    ametista="ametista"; quartzo="quartzo"
    topazio="topázio";   agata="ágata"


class StudentFeatures(BaseModel):
    """
    16 features brutas do aluno (valores originais da planilha).
    O pipeline interno faz:  normalização → FE → inferência.
    """
    fase:           int   = Field(..., ge=0,    le=7)
    ano_ingresso:   int   = Field(..., ge=2016, le=2022)
    score_inde:     float = Field(..., ge=3.0,  le=9.5)
    score_iaa:      float = Field(..., ge=0.0,  le=10.0)
    score_ieg:      float = Field(..., ge=0.0,  le=10.0)
    score_ips:      float = Field(..., ge=2.5,  le=10.0)
    score_ida:      float = Field(..., ge=0.0,  le=9.9)
    score_ipv:      float = Field(..., ge=2.5,  le=10.0)
    score_ian:      float = Field(..., ge=2.5,  le=10.0)
    nota_cg:        int   = Field(..., ge=1,    le=862)
    nota_cf:        int   = Field(..., ge=1,    le=192)
    nota_ct:        int   = Field(..., ge=1,    le=18)
    num_avaliacoes: int   = Field(..., ge=2,    le=4)
    turma:          TurmaEnum
    genero:         GeneroEnum
    pedra_modal:    PedraEnum

class StudentInput(BaseModel):
    student_id: Optional[str] = None
    features:   StudentFeatures

class PredictionResult(BaseModel):
    student_id:       Optional[str]
    defasagem_classe: int
    defasagem_label:  str
    probabilities:    Dict[str, float]
    confidence:       str
    prediction_id:    str
    timestamp:        str

class BatchRequest(BaseModel):
    students: List[StudentInput] = Field(..., min_length=1, max_length=500)

class BatchResponse(BaseModel):
    total:       int
    predictions: List[PredictionResult]
    timestamp:   str

class FeatureInfo(BaseModel):
    name:        str
    description: str
    type:        str
    range:       Optional[str] = None

class ThresholdInfo(BaseModel):
    threshold:  float = Field(..., ge=0.0, le=1.0)
    updated_at: str

class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# COLUNAS BASE — mesma ordem que o ColumnTransformer espera
# ============================================================

# Numéricas (13) + categóricas (3) — exatamente como em preprocessing.py
_NUM_COLS = [
    "fase", "ano_ingresso",
    "score_inde", "score_iaa", "score_ieg", "score_ips",
    "score_ida", "score_ipv", "score_ian",
    "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
]
_CAT_COLS = ["turma", "genero", "pedra_modal"]
_BASE_COLUMNS = _NUM_COLS + _CAT_COLS   # 16 colunas, ordem do preprocessador

_FEATURE_META = {
    "fase":           ("int",         "Fase educacional"),
    "ano_ingresso":   ("int",         "Ano de ingresso no programa"),
    "score_inde":     ("float",       "INDE — Índice de Desenvolvimento Educacional"),
    "score_iaa":      ("float",       "IAA  — Índice de Aprendizagem Ativa"),
    "score_ieg":      ("float",       "IEG  — Índice de Engajamento"),
    "score_ips":      ("float",       "IPS  — Índice de Progresso Social"),
    "score_ida":      ("float",       "IDA  — Índice de Desempenho Acadêmico"),
    "score_ipv":      ("float",       "IPV  — Índice de Progresso de Valor"),
    "score_ian":      ("float",       "IAN  — Índice de Atingimento das Necessidades"),
    "nota_cg":        ("int",         "Caderneta Global"),
    "nota_cf":        ("int",         "Caderneta de Formação"),
    "nota_ct":        ("int",         "Caderneta de Tradição"),
    "num_avaliacoes": ("int",         "Número de avaliações realizadas"),
    "turma":          ("categorical", "Letra da turma"),
    "genero":         ("categorical", "Gênero do aluno"),
    "pedra_modal":    ("categorical", "Pedra modal"),
}


# ============================================================
# PIPELINE DE INFERÊNCIA
# ============================================================

def _features_to_row(features: StudentFeatures) -> dict:
    """Pydantic → dict com as 16 colunas base."""
    row = {}
    for col in _BASE_COLUMNS:
        val = getattr(features, col)
        row[col] = val.value if isinstance(val, Enum) else val
    return row


def _apply_fe_on_normalized(arr_16: np.ndarray, num_col_names: list) -> np.ndarray:
    """
    Calcula as 7 features de feature engineering sobre o array já
    normalizado (output do ColumnTransformer, 16 colunas).

    O ColumnTransformer concatena: [num_cols(13), cat_cols(3)]
    Os índices das colunas numéricas no array transformado:
      idx 0  = fase
      idx 1  = ano_ingresso
      idx 2  = score_inde
      idx 3  = score_iaa
      idx 4  = score_ieg
      idx 5  = score_ips
      idx 6  = score_ida
      idx 7  = score_ipv
      idx 8  = score_ian
      idx 9  = nota_cg
      idx 10 = nota_cf
      idx 11 = nota_ct
      idx 12 = num_avaliacoes
      idx 13 = turma     (ordinal)
      idx 14 = genero    (ordinal)
      idx 15 = pedra_modal (ordinal)

    As 7 features de FE (calculadas sobre os valores normalizados):
      score_medio          = mean(score_inde..score_ian)  → idx 2–8
      nota_media           = mean(nota_cg, nota_cf, nota_ct) → idx 9–11
      ratio_aval_fase      = num_avaliacoes / (fase + 1)  → idx 12, 0
      score_inde_squared   = score_inde²                  → idx 2
      num_avaliacoes_squared = num_avaliacoes²             → idx 12
      inde_x_fase          = score_inde * fase            → idx 2, 0
      aval_x_fase          = num_avaliacoes * fase        → idx 12, 0
    """
    n = arr_16.shape[0]

    # índices no array de 16 colunas (num_cols vêm primeiro no ColumnTransformer)
    # Buscamos os índices reais a partir dos nomes do transformador
    idx = {name: i for i, name in enumerate(num_col_names)}

    i_fase       = idx.get("fase",           0)
    i_inde       = idx.get("score_inde",     2)
    i_iaa        = idx.get("score_iaa",      3)
    i_ieg        = idx.get("score_ieg",      4)
    i_ips        = idx.get("score_ips",      5)
    i_ida        = idx.get("score_ida",      6)
    i_ipv        = idx.get("score_ipv",      7)
    i_ian        = idx.get("score_ian",      8)
    i_nota_cg    = idx.get("nota_cg",        9)
    i_nota_cf    = idx.get("nota_cf",       10)
    i_nota_ct    = idx.get("nota_ct",       11)
    i_num_av     = idx.get("num_avaliacoes",12)

    fase    = arr_16[:, i_fase]
    inde    = arr_16[:, i_inde]
    num_av  = arr_16[:, i_num_av]

    score_medio          = arr_16[:, [i_iaa, i_ieg, i_ips, i_ida, i_ipv, i_ian, i_inde]].mean(axis=1)
    nota_media           = arr_16[:, [i_nota_cg, i_nota_cf, i_nota_ct]].mean(axis=1)
    ratio_aval_fase      = num_av / (fase + 1e-8)   # +ε para evitar ÷0 quando fase=0
    score_inde_sq        = inde    ** 2
    num_av_sq            = num_av  ** 2
    inde_x_fase          = inde    * fase
    aval_x_fase          = num_av  * fase

    # shape: (n, 7)
    fe_cols = np.stack([
        score_medio,
        nota_media,
        ratio_aval_fase,
        score_inde_sq,
        num_av_sq,
        inde_x_fase,
        aval_x_fase,
    ], axis=1)

    # concatenar: (n, 16) + (n, 7) → (n, 23)
    return np.concatenate([arr_16, fe_cols], axis=1).astype(np.float32)


def _preprocess_and_tensorize(rows: List[dict], ctx: dict) -> torch.Tensor:
    """
    Fluxo completo:
      1. DataFrame 16 cols brutas
      2. ColumnTransformer.transform() → array (n, 16) normalizado
      3. _apply_fe_on_normalized()     → array (n, 23) com FE
      4. torch.tensor → device do modelo
    """
    pre = ctx["preprocessor"]

    # colunas numéricas do preprocessador (em ordem)
    num_col_names = list(pre.transformers_[0][2])   # MinMaxScaler cols
    cat_col_names = list(pre.transformers_[1][2])   # OrdinalEncoder cols
    all_cols      = num_col_names + cat_col_names   # 16 colunas, ordem certa

    df = pd.DataFrame(rows, columns=_BASE_COLUMNS)

    # garantir ordem exata que o preprocessador espera
    df_ordered = df[all_cols]

    # step 1 → 16 colunas normalizadas
    arr_16 = pre.transform(df_ordered).astype(np.float32)

    # step 2 → 23 colunas (16 + 7 FE calculadas sobre valores normalizados)
    uses_fe = ctx.get("uses_feature_engineering", False)
    if uses_fe:
        arr_23 = _apply_fe_on_normalized(arr_16, num_col_names)
    else:
        arr_23 = arr_16

    expected_dim = ctx.get("input_dim")
    if expected_dim is not None and arr_23.shape[1] != expected_dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Dimensão do tensor ({arr_23.shape[1]}) não bate com "
                f"input_dim do modelo ({expected_dim}). "
                "Verifique se o pipeline de FE está correto."
            ),
        )

    return torch.tensor(arr_23, dtype=torch.float32).to(ctx["device"])


def _confidence_label(max_prob: float) -> str:
    if max_prob >= 0.60: return "alta"
    if max_prob >= 0.40: return "média"
    return "baixa"


def _check_ready(ctx: dict) -> None:
    missing = []
    if ctx["model"]        is None: missing.append("modelo")
    if ctx["preprocessor"] is None: missing.append("preprocessador")
    if missing:
        raise HTTPException(status_code=503, detail=f"Artefatos não carregados: {', '.join(missing)}.")


def _build_feature_info(ctx: dict) -> List[dict]:
    pre = ctx.get("preprocessor")
    num_meta, cat_meta = {}, {}
    if pre is not None:
        num_trans = pre.transformers_[0][1]
        num_cols  = pre.transformers_[0][2]
        cat_trans = pre.transformers_[1][1]
        cat_cols  = pre.transformers_[1][2]
        num_meta  = {col: (float(mn), float(mx))
                     for col, mn, mx in zip(num_cols, num_trans.data_min_, num_trans.data_max_)}
        cat_meta  = {col: cats.tolist() for col, cats in zip(cat_cols, cat_trans.categories_)}
    result = []
    for col in _BASE_COLUMNS:
        dtype, desc = _FEATURE_META[col]
        if col in num_meta:
            mn, mx = num_meta[col]
            rang = f"{int(mn)} – {int(mx)}" if dtype == "int" else f"{mn:.2f} – {mx:.2f}"
        elif col in cat_meta:
            rang = " | ".join(cat_meta[col])
        else:
            rang = None
        result.append({"name": col, "description": desc, "type": dtype, "range": rang})
    return result


def _serialize(record: dict) -> dict:
    import json
    return json.loads(json.dumps(record, default=str))


# ============================================================
# HEALTH & AUTH ROUTES
# ============================================================

@router.get("/health", tags=["Health"])
def health():
    ctx = get_model_context()
    return {
        "status":                   "ok",
        "model_loaded":             ctx["model"] is not None,
        "preprocessor_loaded":      ctx["preprocessor"] is not None,
        "input_dim":                ctx.get("input_dim"),
        "uses_feature_engineering": ctx.get("uses_feature_engineering"),
        "device":                   str(ctx["device"]),
        "timestamp":                datetime.now(timezone.utc).isoformat(),
    }


@router.post("/register", status_code=201)
def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    fake_users_db[user.username] = {
        "username": user.username, "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
    }
    return {"msg": "user created"}


@router.post("/token", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token({"sub": user.username},
                                timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}


@router.get("/info", tags=["Health"], dependencies=[Depends(get_current_active_user)])
def info():
    ctx = get_model_context()
    return {
        "project": "Magic Steps", "model_class": "MagicStepsNet",
        "input_dim": ctx.get("input_dim"), "num_classes": ctx.get("num_classes"),
        "best_params": ctx.get("best_params"), "device": str(ctx["device"]),
        "uses_feature_engineering": ctx.get("uses_feature_engineering"),
        "preprocessor_loaded": ctx["preprocessor"] is not None,
        "n_base_features": len(_BASE_COLUMNS),
        "features": _build_feature_info(ctx),
    }


# ============================================================
# PREDICT
# ============================================================

@router.post("/predict", response_model=PredictionResult, tags=["Predict"])
def predict(student: StudentInput,
            current_user: User = Depends(get_current_active_user)):
    ctx = get_model_context()
    _check_ready(ctx)

    row    = _features_to_row(student.features)
    tensor = _preprocess_and_tensorize([row], ctx)

    with torch.no_grad():
        logits = ctx["model"](tensor)                           # (1, C)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (C,)

    pred_idx  = int(np.argmax(probs))
    prob_dict = {_DEFASAGEM_LABELS[i]: round(float(p), 6) for i, p in enumerate(probs)}

    result = PredictionResult(
        student_id=student.student_id,
        defasagem_classe=pred_idx,
        defasagem_label=_DEFASAGEM_LABELS[pred_idx],
        probabilities=prob_dict,
        confidence=_confidence_label(float(probs[pred_idx])),
        prediction_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        db_log_prediction(_serialize({
            "prediction_id": result.prediction_id, "timestamp": result.timestamp,
            "user": current_user.username, "student_id": student.student_id,
            "features": row, "defasagem_classe": result.defasagem_classe,
            "defasagem_label": result.defasagem_label,
            "probabilities": result.probabilities, "confidence": result.confidence,
        }), settings.database_url)
    except Exception as e:
        logger.warning("Não foi possível gravar predição no PostgreSQL: %s", e)

    return result


@router.post("/predict/batch", response_model=BatchResponse, tags=["Predict"])
def predict_batch(body: BatchRequest,
                  current_user: User = Depends(get_current_active_user)):
    ctx = get_model_context()
    _check_ready(ctx)

    rows   = [_features_to_row(s.features) for s in body.students]
    tensor = _preprocess_and_tensorize(rows, ctx)

    with torch.no_grad():
        probs_all = torch.softmax(ctx["model"](tensor), dim=1).cpu().numpy()

    now = datetime.now(timezone.utc).isoformat()
    results = []
    for student, probs in zip(body.students, probs_all):
        pred_idx  = int(np.argmax(probs))
        prob_dict = {_DEFASAGEM_LABELS[i]: round(float(p), 6) for i, p in enumerate(probs)}
        results.append(PredictionResult(
            student_id=student.student_id,
            defasagem_classe=pred_idx,
            defasagem_label=_DEFASAGEM_LABELS[pred_idx],
            probabilities=prob_dict,
            confidence=_confidence_label(float(probs[pred_idx])),
            prediction_id=str(uuid4()),
            timestamp=now,
        ))

    for student, res in zip(body.students, results):
        try:
            db_log_prediction(_serialize({
                "prediction_id": res.prediction_id, "timestamp": now,
                "user": current_user.username, "student_id": student.student_id,
                "features": _features_to_row(student.features),
                "defasagem_classe": res.defasagem_classe,
                "defasagem_label": res.defasagem_label,
                "probabilities": res.probabilities, "confidence": res.confidence,
            }), settings.database_url)
        except Exception as e:
            logger.warning("Erro ao gravar predição batch: %s", e)

    return BatchResponse(total=len(results), predictions=results, timestamp=now)


# ============================================================
# FEATURES
# ============================================================

@router.get("/features", response_model=List[FeatureInfo], tags=["Features"])
def features():
    """Lista as 16 features de entrada com tipo e range do preprocessador."""
    return _build_feature_info(get_model_context())


class ProcessRequest(BaseModel):
    features: StudentFeatures


class PreprocessorInfo(BaseModel):
    num_cols:       List[str]
    cat_cols:       List[str]
    num_min:        Dict[str, float]
    num_max:        Dict[str, float]
    cat_categories: Dict[str, List[str]]


class FeatureProcessResponse(BaseModel):
    """
    Resposta detalhada do pipeline de transformação para uma amostra.

    raw        → valores originais enviados (16 campos — numéricas como float,
                 categóricas como string)
    normalized → output do ColumnTransformer (16 valores float normalizados/encoded)
    engineered → array final de 23 features float que entra no modelo
    preprocessor → metadados extraídos do preprocessor.joblib
    """
    raw:          Dict[str, Any]   # str para categóricas, float para numéricas
    normalized:   Dict[str, float]
    engineered:   Dict[str, float]
    preprocessor: PreprocessorInfo


@router.post(
    "/features/process",
    response_model=FeatureProcessResponse,
    tags=["Features"],
    summary="Aplica o pipeline completo e retorna cada etapa de transformação",
    dependencies=[Depends(get_current_active_user)],
)
def features_process(body: ProcessRequest) -> FeatureProcessResponse:
    """
    Recebe as 16 features brutas de um aluno e retorna:

    - **raw**: valores exatamente como recebidos
    - **normalized**: saída do ColumnTransformer (MinMaxScaler + OrdinalEncoder)
    - **engineered**: as 23 features finais (16 normalizadas + 7 FE) que
      alimentam o modelo
    - **preprocessor**: key-value do preprocessor.joblib —
      colunas, min/max de cada numérica, categorias de cada categórica
    """
    ctx = get_model_context()
    if ctx["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Preprocessador não carregado.")

    pre = ctx["preprocessor"]

    # ── metadados do preprocessor ─────────────────────────────────────────
    num_trans    = pre.transformers_[0][1]   # MinMaxScaler
    num_cols     = list(pre.transformers_[0][2])
    cat_trans    = pre.transformers_[1][1]   # OrdinalEncoder
    cat_cols     = list(pre.transformers_[1][2])
    all_cols     = num_cols + cat_cols

    num_min = {c: float(v) for c, v in zip(num_cols, num_trans.data_min_)}
    num_max = {c: float(v) for c, v in zip(num_cols, num_trans.data_max_)}
    cat_cats = {c: cats.tolist()
                for c, cats in zip(cat_cols, cat_trans.categories_)}

    preprocessor_info = PreprocessorInfo(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_min=num_min,
        num_max=num_max,
        cat_categories=cat_cats,
    )

    # ── raw ───────────────────────────────────────────────────────────────
    row = _features_to_row(body.features)
    raw_out = {k: float(v) if not isinstance(v, str) else v
               for k, v in row.items()}

    # ── normalized (16 colunas) ───────────────────────────────────────────
    df = pd.DataFrame([row], columns=_BASE_COLUMNS)[all_cols]
    arr_16 = pre.transform(df).astype(np.float32)[0]   # shape (16,)
    normalized_out = {col: round(float(val), 8)
                      for col, val in zip(all_cols, arr_16)}

    # ── engineered (23 colunas) ───────────────────────────────────────────
    uses_fe = ctx.get("uses_feature_engineering", False)
    if uses_fe:
        arr_23 = _apply_fe_on_normalized(arr_16.reshape(1, -1), num_cols)[0]
    else:
        arr_23 = arr_16

    # nomes das 7 features de FE (na mesma ordem de _apply_fe_on_normalized)
    _FE_NAMES = [
        "score_medio", "nota_media", "ratio_aval_fase",
        "score_inde_squared", "num_avaliacoes_squared",
        "inde_x_fase", "aval_x_fase",
    ]
    engineered_cols = all_cols + (_FE_NAMES if uses_fe else [])
    engineered_out  = {col: round(float(val), 8)
                       for col, val in zip(engineered_cols, arr_23)}

    return FeatureProcessResponse(
        raw=raw_out,
        normalized=normalized_out,
        engineered=engineered_out,
        preprocessor=preprocessor_info,
    )


# ============================================================
# THRESHOLDS (stateful — persiste valor em memória)
# ============================================================

# estado em memória para o threshold de confiança
_threshold_state: Dict[str, float | str] = {
    "threshold":  0.5,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}


@router.get("/thresholds", response_model=ThresholdInfo, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)])
def get_threshold():
    return ThresholdInfo(**_threshold_state)


@router.put("/thresholds", response_model=ThresholdInfo, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)])
def update_threshold(body: ThresholdUpdate):
    _threshold_state["threshold"]  = body.threshold
    _threshold_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return ThresholdInfo(**_threshold_state)


# ============================================================
# MONITORING
# ============================================================

@router.get("/monitor/logs", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)])
def monitor_logs(user:       Optional[str] = Query(None),
                 student_id: Optional[str] = Query(None),
                 limit:      int           = Query(200, ge=1, le=5000)):
    try:
        records = query_predictions(settings.database_url, user=user,
                                    student_id=student_id, limit=limit)
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"): r[k] = v.isoformat()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/drift", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)])
def monitor_drift():
    try:
        summary = get_drift_summary(settings.database_url)
        for row in summary.get("class_distribution", []):
            for k, v in row.items():
                if hasattr(v, "isoformat"): row[k] = v.isoformat()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/runs", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)])
def monitor_runs():
    try:
        runs = list_model_runs(settings.database_url)
        for r in runs:
            for k, v in r.items():
                if hasattr(v, "isoformat"): r[k] = v.isoformat()
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/events", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)])
def monitor_events(event_type: Optional[str] = Query(None),
                   severity:   Optional[str] = Query(None),
                   limit:      int           = Query(200, ge=1, le=2000)):
    try:
        events = query_monitoring_logs(settings.database_url,
                                       event_type=event_type,
                                       severity=severity, limit=limit)
        for e in events:
            for k, v in e.items():
                if hasattr(v, "isoformat"): e[k] = v.isoformat()
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))