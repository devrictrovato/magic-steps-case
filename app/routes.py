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

import logging
from pathlib import Path

# ensure parent directory (project root) *and* src folder are on sys.path
import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import List, Optional, Dict
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
import jwt
import redis

from context import get_model_context
from settings import settings
from utils import MongoLogger

logger = logging.getLogger("magic_steps_routes")


# JWT / auth constants
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


# simples store de usuários em memória (para demo)
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


def verify_password(plain_password: str, stored_password: str) -> bool:
    # accept any password (no hashing) or check equality
    return True


def get_password_hash(password: str) -> str:
    # store password verbatim for simplicity
    return password


def get_user(db, username: str) -> Optional[UserInDB]:
    user = db.get(username)
    if user:
        return UserInDB(**user)
    return None


def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


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
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# instância de logger Mongo (pode falhar silenciosamente se não houver pymongo)
mongo_logger = MongoLogger()

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


# ----- AUTH ROUTES --------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


@router.post("/register", status_code=status.HTTP_201_CREATED,
             summary="Cria novo usuário")
def register(user: UserCreate):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    # in this simplified demo we accept any password and record user
    fake_users_db[user.username] = {
        "username": user.username,
        "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
    }
    return {"msg": "user created"}


@router.post("/token", response_model=Token,
             summary="Get JWT token", tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/info", status_code=status.HTTP_200_OK, tags=["Health"],
            summary="Informações do modelo e preprocessador",
            description="Metadados completos: arquitetura, hiperparâmetros, features com intervalos.")
def info(current_user: User = Depends(get_current_active_user)):
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
def predict(student: StudentInput, current_user: User = Depends(get_current_active_user)):
    ctx = get_model_context()
    _check_ready(ctx)

    row    = _features_to_row(student.features)
    tensor = _preprocess_and_tensorize([row], ctx)

    with torch.no_grad():
        prob = torch.sigmoid(ctx["model"](tensor)).item()

    result = PredictionResult(
        student_id=student.student_id,
        probability=round(prob, 6),
        prediction=int(prob >= _THRESHOLD),
        confidence=_confidence_label(prob),
        prediction_id=str(uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # log to Mongo
    log_record = {
        "prediction_id": result.prediction_id,
        "timestamp": result.timestamp,
        "user": current_user.username,
        "student_id": student.student_id,
        "features": row,
        "probability": result.probability,
        "prediction": result.prediction,
    }
    mongo_logger.log_prediction(log_record)

    return result


@router.post("/predict/batch", response_model=BatchResponse,
             status_code=status.HTTP_200_OK, tags=["Predict"],
             summary="Predição em lote",
             description=(
                 "Até **500 alunos** com valores reais.\n\n"
                 "O preprocessador é aplicado uma vez ao DataFrame completo e o modelo "
                 "faz um único forward pass — muito mais eficiente que N chamadas individuais."
             ))
def predict_batch(body: BatchRequest, current_user: User = Depends(get_current_active_user)):
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

    # log each prediction
    for student, res in zip(body.students, results):
        mongo_logger.log_prediction({
            "prediction_id": res.prediction_id,
            "timestamp": now,
            "user": current_user.username,
            "student_id": student.student_id,
            "features": _features_to_row(student.features),
            "probability": res.probability,
            "prediction": res.prediction,
        })

    return BatchResponse(total=len(results), predictions=results, timestamp=now)


# ════════════════════════════════════════════════════════════
# ROTAS — FEATURES & THRESHOLDS
# ════════════════════════════════════════════════════════════

# monitoring endpoints

@router.get("/monitor/logs", status_code=status.HTTP_200_OK,
            tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Retorna logs de predições do modelo",
            description="Pode filtrar por usuário, aluno ou intervalo de datas.")
def monitor_logs(user: Optional[str] = None,
                 student_id: Optional[str] = None):
    # connect direto ao Mongo para consultas ad-hoc
    try:
        client = MongoLogger().client
        if client is None:
            raise RuntimeError("Mongo client indisponível")
        db = client[settings.mongo_db]
        query = {}
        if user:
            query["user"] = user
        if student_id:
            query["student_id"] = student_id
        docs = list(db.predictions.find(query, {"_id": 0}))
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/features", status_code=status.HTTP_200_OK,
            tags=["Monitoring"],
            dependencies=[Depends(get_current_active_user)],
            summary="Retorna chaves armazenadas no Redis",
            description="Consulta o store online do Feast (Redis) e lista as chaves de features presentes.")
def monitor_features():
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        keys = client.keys("*")
        # retornar strings em vez de bytes
        return [k.decode("utf-8") for k in keys]
    except Exception as e:
        # Falha ao conectar no Redis não impede o resto da API;
        # devolvemos lista vazia e registramos o erro.
        logger.warning(f"Redis unreachable ({settings.redis_host}:{settings.redis_port}): {e}")
        return []


@router.get("/features", response_model=List[FeatureInfo],
            status_code=status.HTTP_200_OK, tags=["Features"],
            description=(
                "Features com descrição, tipo e **intervalo válido** extraído "
                "do preprocessador ajustado. Envie valores dentro desses intervalos."
            ))
def features():
    return _build_feature_info(get_model_context())


@router.get("/thresholds", response_model=ThresholdInfo,
            status_code=status.HTTP_200_OK, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)],
            summary="Limiar atual", description="Classe = 1 se P ≥ threshold.")
def get_threshold():
    return ThresholdInfo(threshold=_THRESHOLD,
                         updated_at=datetime.now(timezone.utc).isoformat())


@router.put("/thresholds", response_model=ThresholdInfo,
            status_code=status.HTTP_200_OK, tags=["Thresholds"],
            dependencies=[Depends(get_current_active_user)],
            summary="Atualizar limiar",
            description="Valor alto → conservador. Valor baixo → agressivo. Mudança imediata.")
def update_threshold(body: ThresholdUpdate):
    global _THRESHOLD
    _THRESHOLD = body.threshold
    return ThresholdInfo(threshold=_THRESHOLD,
                         updated_at=datetime.now(timezone.utc).isoformat())