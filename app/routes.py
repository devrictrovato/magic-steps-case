# routes.py  v2.1
# ============================================================
# Rotas da API Magic Steps
# ============================================================
#
# Grupo          | Rotas
# ───────────────|──────────────────────────────────────────────
# Health         | GET  /health
# Auth           | POST /register  |  POST /token
# Info           | GET  /info
# Predict        | POST /predict          (1 aluno)
#                | POST /predict/batch    (até 500 alunos)
# Features       | GET  /features
# Thresholds     | GET  /thresholds  |  PUT /thresholds
# Monitoring     | GET  /monitor/logs
#                | GET  /monitor/drift
#                | GET  /monitor/runs
#                | GET  /monitor/events
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
from typing import List, Optional, Dict, Any
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

SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
fake_users_db: Dict[str, Dict] = {}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False


class UserInDB(User):
    hashed_password: str


def get_password_hash(password: str) -> str:
    return password


def get_user(db, username: str) -> Optional[UserInDB]:
    user = db.get(username)
    return UserInDB(**user) if user else None


def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    return user if user else None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ============================================================
# RÓTULOS E ENUMS
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
    menina = "menina"
    menino = "menino"


class PedraEnum(str, Enum):
    ametista = "ametista"
    quartzo  = "quartzo"
    topazio  = "topázio"
    agata    = "ágata"


# ============================================================
# ESQUEMAS DE ENTRADA / SAÍDA
# ============================================================

class StudentFeatures(BaseModel):
    """
    Dados brutos de um aluno — exatamente como saem da planilha.
    O preprocessamento (MinMaxScaler + OrdinalEncoder) e o
    feature engineering são aplicados internamente pela API.
    """
    fase:           int   = Field(..., ge=0,    le=7,    description="Fase educacional (0–7)")
    ano_ingresso:   int   = Field(..., ge=2016, le=2022, description="Ano de ingresso no programa")
    score_inde:     float = Field(..., ge=3.0,  le=9.5,  description="INDE 22")
    score_iaa:      float = Field(..., ge=0.0,  le=10.0, description="IAA")
    score_ieg:      float = Field(..., ge=0.0,  le=10.0, description="IEG")
    score_ips:      float = Field(..., ge=2.5,  le=10.0, description="IPS")
    score_ida:      float = Field(..., ge=0.0,  le=9.9,  description="IDA")
    score_ipv:      float = Field(..., ge=2.5,  le=10.0, description="IPV")
    score_ian:      float = Field(..., ge=2.5,  le=10.0, description="IAN")
    nota_cg:        int   = Field(..., ge=1,    le=862,  description="Caderneta Global")
    nota_cf:        int   = Field(..., ge=1,    le=192,  description="Caderneta de Formação")
    nota_ct:        int   = Field(..., ge=1,    le=18,   description="Caderneta de Tradição")
    num_avaliacoes: int   = Field(..., ge=2,    le=4,    description="Nº de avaliações")
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


class UserCreate(BaseModel):
    username:  str
    password:  str
    full_name: Optional[str] = None


# ============================================================
# COLUNAS BASE (entrada do preprocessador)
# ============================================================

_BASE_COLUMNS = [
    "fase", "ano_ingresso",
    "score_inde", "score_iaa", "score_ieg", "score_ips",
    "score_ida", "score_ipv", "score_ian",
    "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
    "turma", "genero", "pedra_modal",
]

_FEATURE_META = {
    "fase":           ("int",         "Fase educacional do aluno"),
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
# HELPERS
# ============================================================

def _confidence_label(max_prob: float) -> str:
    if max_prob >= 0.60: return "alta"
    if max_prob >= 0.40: return "média"
    return "baixa"


def _check_ready(ctx: dict) -> None:
    missing = []
    if ctx["model"]        is None: missing.append("modelo")
    if ctx["preprocessor"] is None: missing.append("preprocessador")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Artefatos não carregados: {', '.join(missing)}.",
        )


def _features_to_row(features: StudentFeatures) -> dict:
    row = {}
    for col in _BASE_COLUMNS:
        val = getattr(features, col)
        row[col] = val.value if isinstance(val, Enum) else val
    return row


def _apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduz exatamente as mesmas transformações do FeatureEngineer,
    adicionando as features derivadas usadas no treinamento.
    """
    df = df.copy()

    # ── features agregadas ─────────────────────────────────────────────────
    score_cols = ["score_inde","score_iaa","score_ieg","score_ips",
                  "score_ida","score_ipv","score_ian"]
    df["score_medio"] = df[[c for c in score_cols if c in df.columns]].mean(axis=1)

    nota_cols = ["nota_cg", "nota_cf", "nota_ct"]
    df["nota_media"] = df[[c for c in nota_cols if c in df.columns]].mean(axis=1)

    # ── razão ─────────────────────────────────────────────────────────────
    df["ratio_aval_fase"] = df["num_avaliacoes"] / (df["fase"] + 1)

    # ── polinomiais ────────────────────────────────────────────────────────
    df["score_inde_squared"]     = df["score_inde"]     ** 2
    df["num_avaliacoes_squared"] = df["num_avaliacoes"] ** 2

    # ── interações ────────────────────────────────────────────────────────
    df["inde_x_fase"] = df["score_inde"]     * df["fase"]
    df["aval_x_fase"] = df["num_avaliacoes"] * df["fase"]

    return df


def _preprocess_and_tensorize(rows: List[dict], ctx: dict) -> torch.Tensor:
    """
    Pipeline de inferência:
      1. Monta DataFrame com colunas base
      2. Se o modelo foi treinado com feature engineering, aplica FE
      3. Separa numéricas vs categóricas conforme o preprocessador
      4. Aplica ColumnTransformer (MinMaxScaler + OrdinalEncoder)
      5. Retorna tensor float32 no device correto
    """
    pre = ctx["preprocessor"]

    # ── colunas que o preprocessador conhece ──────────────────────────────
    num_cols = list(pre.transformers_[0][2])   # MinMaxScaler
    cat_cols = list(pre.transformers_[1][2])   # OrdinalEncoder
    preprocessor_cols = num_cols + cat_cols    # ordem exata esperada

    # ── montar DataFrame base ──────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=_BASE_COLUMNS)

    # ── aplicar feature engineering se necessário ─────────────────────────
    uses_fe = ctx.get("uses_feature_engineering", False)
    if uses_fe:
        df = _apply_feature_engineering(df)

    # ── selecionar apenas as colunas que o preprocessador conhece ─────────
    # (garante ordem correta e ignora colunas extras não vistas no treino)
    missing = [c for c in preprocessor_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Colunas ausentes após feature engineering: {missing}. "
                "Verifique se o preprocessador foi salvo com o mesmo pipeline de FE."
            ),
        )

    df_input = df[preprocessor_cols]

    # ── transformar ───────────────────────────────────────────────────────
    arr = pre.transform(df_input).astype(np.float32)
    return torch.tensor(arr, dtype=torch.float32).to(ctx["device"])


def _build_feature_info(ctx: dict) -> List[dict]:
    pre = ctx.get("preprocessor")
    num_meta, cat_meta = {}, {}
    if pre is not None:
        num_trans = pre.transformers_[0][1]
        num_cols  = pre.transformers_[0][2]
        cat_trans = pre.transformers_[1][1]
        cat_cols  = pre.transformers_[1][2]
        num_meta = {col: (float(mn), float(mx)) for col, mn, mx in
                    zip(num_cols, num_trans.data_min_, num_trans.data_max_)}
        cat_meta = {col: cats.tolist() for col, cats in
                    zip(cat_cols, cat_trans.categories_)}

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
# HEALTH & AUTH
# ============================================================

@router.get("/health", tags=["Health"])
def health():
    ctx = get_model_context()
    return {
        "status":              "ok",
        "model_loaded":        ctx["model"] is not None,
        "preprocessor_loaded": ctx["preprocessor"] is not None,
        "input_dim":           ctx.get("input_dim"),
        "uses_feature_engineering": ctx.get("uses_feature_engineering"),
        "device":              str(ctx["device"]),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    fake_users_db[user.username] = {
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
    }
    return {"msg": "user created"}


@router.post("/token", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/info", tags=["Health"],
            dependencies=[Depends(get_current_active_user)])
def info():
    ctx = get_model_context()
    return {
        "project":             "Magic Steps",
        "model_class":         "MagicStepsNet",
        "input_dim":           ctx.get("input_dim"),
        "num_classes":         ctx.get("num_classes"),
        "best_params":         ctx.get("best_params"),
        "device":              str(ctx["device"]),
        "uses_feature_engineering": ctx.get("uses_feature_engineering"),
        "preprocessor_loaded": ctx["preprocessor"] is not None,
        "n_base_features":     len(_BASE_COLUMNS),
        "features":            _build_feature_info(ctx),
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
    label     = _DEFASAGEM_LABELS[pred_idx]
    prob_dict = {_DEFASAGEM_LABELS[i]: round(float(p), 6) for i, p in enumerate(probs)}
    max_prob  = float(probs[pred_idx])

    result = PredictionResult(
        student_id=student.student_id,
        defasagem_classe=pred_idx,
        defasagem_label=label,
        probabilities=prob_dict,
        confidence=_confidence_label(max_prob),
        prediction_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        db_log_prediction(_serialize({
            "prediction_id":    result.prediction_id,
            "timestamp":        result.timestamp,
            "user":             current_user.username,
            "student_id":       student.student_id,
            "features":         row,
            "defasagem_classe": result.defasagem_classe,
            "defasagem_label":  result.defasagem_label,
            "probabilities":    result.probabilities,
            "confidence":       result.confidence,
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
        logits    = ctx["model"](tensor)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy()

    now = datetime.now(timezone.utc).isoformat()
    results = []
    for student, probs in zip(body.students, probs_all):
        pred_idx  = int(np.argmax(probs))
        label     = _DEFASAGEM_LABELS[pred_idx]
        prob_dict = {_DEFASAGEM_LABELS[i]: round(float(p), 6) for i, p in enumerate(probs)}
        results.append(PredictionResult(
            student_id=student.student_id,
            defasagem_classe=pred_idx,
            defasagem_label=label,
            probabilities=prob_dict,
            confidence=_confidence_label(float(probs[pred_idx])),
            prediction_id=str(uuid4()),
            timestamp=now,
        ))

    for student, res in zip(body.students, results):
        try:
            db_log_prediction(_serialize({
                "prediction_id":    res.prediction_id,
                "timestamp":        now,
                "user":             current_user.username,
                "student_id":       student.student_id,
                "features":         _features_to_row(student.features),
                "defasagem_classe": res.defasagem_classe,
                "defasagem_label":  res.defasagem_label,
                "probabilities":    res.probabilities,
                "confidence":       res.confidence,
            }), settings.database_url)
        except Exception as e:
            logger.warning("Erro ao gravar predição batch: %s", e)

    return BatchResponse(total=len(results), predictions=results, timestamp=now)


# ============================================================
# FEATURES & THRESHOLDS
# ============================================================

@router.get("/features", response_model=List[FeatureInfo], tags=["Features"])
def features():
    return _build_feature_info(get_model_context())


@router.get("/thresholds", response_model=ThresholdInfo, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)])
def get_threshold():
    return ThresholdInfo(threshold=0.0, updated_at=datetime.now(timezone.utc).isoformat())


@router.put("/thresholds", response_model=ThresholdInfo, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)])
def update_threshold(body: ThresholdUpdate):
    return ThresholdInfo(threshold=body.threshold,
                         updated_at=datetime.now(timezone.utc).isoformat())


# ============================================================
# MONITORING
# ============================================================

@router.get("/monitor/logs", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Logs de predições (PostgreSQL)")
def monitor_logs(
    user:       Optional[str] = Query(None),
    student_id: Optional[str] = Query(None),
    limit:      int           = Query(200, ge=1, le=5000),
):
    try:
        records = query_predictions(settings.database_url, user=user,
                                    student_id=student_id, limit=limit)
        for r in records:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/drift", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Resumo de drift — distribuição das predições")
def monitor_drift():
    try:
        summary = get_drift_summary(settings.database_url)
        for row in summary.get("class_distribution", []):
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/runs", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Histórico de runs de treinamento")
def monitor_runs():
    try:
        runs = list_model_runs(settings.database_url)
        for r in runs:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/events", tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Eventos de monitoramento do sistema")
def monitor_events(
    event_type: Optional[str] = Query(None),
    severity:   Optional[str] = Query(None),
    limit:      int           = Query(200, ge=1, le=2000),
):
    try:
        events = query_monitoring_logs(settings.database_url,
                                       event_type=event_type,
                                       severity=severity, limit=limit)
        for e in events:
            for k, v in e.items():
                if hasattr(v, "isoformat"):
                    e[k] = v.isoformat()
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))