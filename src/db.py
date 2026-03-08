"""
db.py — Cliente PostgreSQL central para o projeto Magic Steps MLOps.

Responsabilidades:
  • Criar/garantir todas as tabelas necessárias (DDL automático no startup)
  • Expor helpers para cada domínio:
      - raw_data        → tabela de dados brutos vindos do Excel/CSV
      - engineered_features → features após feature_engineering
      - predictions     → logs de inferência da API
      - model_runs      → metadados de treino / avaliação (drift, métricas)
      - monitoring_logs → eventos gerais de monitoramento
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger("magic_steps.db")


# ============================================================
# CONNECTION
# ============================================================

def get_connection(database_url: str) -> PgConnection:
    """Abre e retorna uma conexão psycopg2."""
    conn = psycopg2.connect(database_url, connect_timeout=30)
    conn.autocommit = False
    return conn


@contextmanager
def db_cursor(database_url: str):
    """Context manager: abre conexão + cursor e faz commit/rollback."""
    conn = get_connection(database_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# DDL — criação automática das tabelas
# ============================================================

_DDL = """
-- ── 1. Dados brutos ingeridos (origem: Excel/CSV) ──────────────────────────
CREATE TABLE IF NOT EXISTS raw_student_data (
    id                  SERIAL PRIMARY KEY,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_file         TEXT,
    fase                INT,
    turma               TEXT,
    ano_nascimento      INT,
    idade               INT,
    genero              TEXT,
    ano_ingresso        INT,
    pedra_2020          TEXT,
    pedra_2021          TEXT,
    pedra_2022          TEXT,
    score_inde          FLOAT,
    score_iaa           FLOAT,
    score_ieg           FLOAT,
    score_ips           FLOAT,
    score_ida           FLOAT,
    score_ipv           FLOAT,
    score_ian           FLOAT,
    nota_matematica     FLOAT,
    nota_portugues      FLOAT,
    nota_ingles         FLOAT,
    nota_cg             FLOAT,
    nota_cf             FLOAT,
    nota_ct             FLOAT,
    num_avaliacoes      INT,
    defasagem           FLOAT,
    flag_atingiu_pv     TEXT,
    extra_columns       JSONB
);

-- ── 2. Features após feature_engineering ───────────────────────────────────
CREATE TABLE IF NOT EXISTS engineered_features (
    id                  SERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_run_id     TEXT,
    student_id          INT,
    event_timestamp     TIMESTAMPTZ,
    fase                FLOAT,
    ano_ingresso        FLOAT,
    score_inde          FLOAT,
    score_iaa           FLOAT,
    score_ieg           FLOAT,
    score_ips           FLOAT,
    score_ida           FLOAT,
    score_ipv           FLOAT,
    score_ian           FLOAT,
    nota_cg             FLOAT,
    nota_cf             FLOAT,
    nota_ct             FLOAT,
    num_avaliacoes      FLOAT,
    turma               TEXT,
    genero              TEXT,
    pedra_modal         TEXT,
    score_medio         FLOAT,
    nota_media          FLOAT,
    ratio_aval_fase     FLOAT,
    score_inde_squared  FLOAT,
    num_avaliacoes_squared FLOAT,
    inde_x_fase         FLOAT,
    aval_x_fase         FLOAT,
    defasagem           INT,
    extra_features      JSONB
);

-- ── 3. Logs de predição da API ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    prediction_id       TEXT UNIQUE NOT NULL,
    predicted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    api_user            TEXT,
    student_id          TEXT,
    features            JSONB NOT NULL,
    defasagem_classe    INT NOT NULL,
    defasagem_label     TEXT NOT NULL,
    prob_atraso         FLOAT,
    prob_neutro         FLOAT,
    prob_avanco         FLOAT,
    confidence          TEXT,
    model_version       TEXT
);

-- ── 4. Runs de treino / avaliação ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_runs (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT UNIQUE NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT DEFAULT 'running',
    model_name          TEXT,
    best_params         JSONB,
    metrics             JSONB,
    confusion_matrix    JSONB,
    classification_report TEXT,
    notes               TEXT
);

-- ── 5. Monitoramento / drift ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS monitoring_logs (
    id                  SERIAL PRIMARY KEY,
    logged_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type          TEXT NOT NULL,
    details             JSONB,
    severity            TEXT DEFAULT 'info'
);
"""


def ensure_schema(database_url: str) -> None:
    """Executa o DDL para garantir que todas as tabelas existam."""
    conn = get_connection(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        logger.info("✅ Schema PostgreSQL verificado/criado com sucesso.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erro ao criar schema: {e}")
        raise
    finally:
        conn.close()


# ============================================================
# RAW DATA
# ============================================================

def insert_raw_dataframe(
    df: pd.DataFrame,
    database_url: str,
    source_file: str = "",
) -> int:
    """
    Persiste um DataFrame de dados brutos na tabela raw_student_data.
    Colunas extras (não mapeadas) são salvas no campo extra_columns (JSONB).

    Returns:
        Número de linhas inseridas.
    """
    known_cols = {
        "fase", "turma", "ano_nascimento", "idade", "genero", "ano_ingresso",
        "pedra_2020", "pedra_2021", "pedra_2022",
        "score_inde", "score_iaa", "score_ieg", "score_ips", "score_ida",
        "score_ipv", "score_ian",
        "nota_matematica", "nota_portugues", "nota_ingles",
        "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
        "defasagem", "flag_atingiu_pv",
    }

    sql = """
        INSERT INTO raw_student_data (
            source_file, fase, turma, ano_nascimento, idade, genero, ano_ingresso,
            pedra_2020, pedra_2021, pedra_2022,
            score_inde, score_iaa, score_ieg, score_ips, score_ida,
            score_ipv, score_ian,
            nota_matematica, nota_portugues, nota_ingles,
            nota_cg, nota_cf, nota_ct, num_avaliacoes,
            defasagem, flag_atingiu_pv, extra_columns
        ) VALUES (
            %(source_file)s, %(fase)s, %(turma)s, %(ano_nascimento)s, %(idade)s,
            %(genero)s, %(ano_ingresso)s,
            %(pedra_2020)s, %(pedra_2021)s, %(pedra_2022)s,
            %(score_inde)s, %(score_iaa)s, %(score_ieg)s, %(score_ips)s,
            %(score_ida)s, %(score_ipv)s, %(score_ian)s,
            %(nota_matematica)s, %(nota_portugues)s, %(nota_ingles)s,
            %(nota_cg)s, %(nota_cf)s, %(nota_ct)s, %(num_avaliacoes)s,
            %(defasagem)s, %(flag_atingiu_pv)s, %(extra_columns)s
        )
    """

    conn = get_connection(database_url)
    count = 0
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                extras = {c: _safe(row[c]) for c in df.columns if c not in known_cols}
                params = {c: _safe(row.get(c)) for c in known_cols}
                params["source_file"] = source_file
                params["extra_columns"] = json.dumps(extras, default=str)
                cur.execute(sql, params)
                count += 1
        conn.commit()
        logger.info(f"Inseridas {count} linhas em raw_student_data.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erro ao inserir dados brutos: {e}")
        raise
    finally:
        conn.close()
    return count


def load_raw_dataframe(database_url: str) -> pd.DataFrame:
    """Carrega todos os dados brutos da tabela raw_student_data."""
    with db_cursor(database_url) as cur:
        cur.execute("SELECT * FROM raw_student_data ORDER BY id")
        rows = cur.fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ============================================================
# ENGINEERED FEATURES
# ============================================================

def insert_engineered_features(
    df: pd.DataFrame,
    database_url: str,
    pipeline_run_id: str = "",
) -> int:
    """
    Persiste DataFrame de features engenheiradas.
    Colunas extras são salvas em extra_features (JSONB).
    """
    known_cols = {
        "student_id", "event_timestamp",
        "fase", "ano_ingresso", "score_inde", "score_iaa", "score_ieg",
        "score_ips", "score_ida", "score_ipv", "score_ian",
        "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
        "turma", "genero", "pedra_modal",
        "score_medio", "nota_media", "ratio_aval_fase",
        "score_inde_squared", "num_avaliacoes_squared",
        "inde_x_fase", "aval_x_fase", "defasagem",
    }

    sql = """
        INSERT INTO engineered_features (
            pipeline_run_id, student_id, event_timestamp,
            fase, ano_ingresso, score_inde, score_iaa, score_ieg,
            score_ips, score_ida, score_ipv, score_ian,
            nota_cg, nota_cf, nota_ct, num_avaliacoes,
            turma, genero, pedra_modal,
            score_medio, nota_media, ratio_aval_fase,
            score_inde_squared, num_avaliacoes_squared,
            inde_x_fase, aval_x_fase, defasagem, extra_features
        ) VALUES (
            %(pipeline_run_id)s, %(student_id)s, %(event_timestamp)s,
            %(fase)s, %(ano_ingresso)s, %(score_inde)s, %(score_iaa)s,
            %(score_ieg)s, %(score_ips)s, %(score_ida)s, %(score_ipv)s,
            %(score_ian)s, %(nota_cg)s, %(nota_cf)s, %(nota_ct)s,
            %(num_avaliacoes)s, %(turma)s, %(genero)s, %(pedra_modal)s,
            %(score_medio)s, %(nota_media)s, %(ratio_aval_fase)s,
            %(score_inde_squared)s, %(num_avaliacoes_squared)s,
            %(inde_x_fase)s, %(aval_x_fase)s, %(defasagem)s, %(extra_features)s
        )
    """

    conn = get_connection(database_url)
    count = 0
    try:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                extras = {c: _safe(row[c]) for c in df.columns if c not in known_cols}
                params = {c: _safe(row.get(c)) for c in known_cols}
                params["pipeline_run_id"] = pipeline_run_id
                params["extra_features"] = json.dumps(extras, default=str)
                cur.execute(sql, params)
                count += 1
        conn.commit()
        logger.info(f"Inseridas {count} linhas em engineered_features.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erro ao inserir features engenheiradas: {e}")
        raise
    finally:
        conn.close()
    return count


def load_engineered_features(
    database_url: str,
    pipeline_run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carrega features engenheiradas do PostgreSQL.
    Se pipeline_run_id for fornecido, filtra pela execução específica;
    caso contrário, retorna a execução mais recente.
    """
    with db_cursor(database_url) as cur:
        if pipeline_run_id:
            cur.execute(
                "SELECT * FROM engineered_features WHERE pipeline_run_id = %s ORDER BY id",
                (pipeline_run_id,),
            )
        else:
            # pega o run_id mais recente
            cur.execute(
                """
                SELECT * FROM engineered_features
                WHERE pipeline_run_id = (
                    SELECT pipeline_run_id FROM engineered_features
                    ORDER BY created_at DESC LIMIT 1
                )
                ORDER BY id
                """
            )
        rows = cur.fetchall()

    if not rows:
        raise ValueError(
            "Nenhuma feature engenheirada encontrada no PostgreSQL. "
            "Execute o pipeline de feature_engineering primeiro."
        )

    df = pd.DataFrame([dict(r) for r in rows])
    # remover colunas de controle
    drop_cols = [
        c for c in [
            "id",
            "created_at",
            "pipeline_run_id",
            "student_id",
            "event_timestamp",
            "extra_features",
            "score_medio",
            "nota_media",
            "ratio_aval_fase",
            "score_inde_squared",
            "num_avaliacoes_squared",
            "inde_x_fase",
            "aval_x_fase",
        ]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)
    return df


# ============================================================
# PREDICTIONS (API logs)
# ============================================================

def log_prediction(record: dict, database_url: str) -> None:
    """Insere um registro de predição na tabela predictions."""
    sql = """
        INSERT INTO predictions (
            prediction_id, predicted_at, api_user, student_id,
            features, defasagem_classe, defasagem_label,
            prob_atraso, prob_neutro, prob_avanco,
            confidence, model_version
        ) VALUES (
            %(prediction_id)s, %(predicted_at)s, %(api_user)s, %(student_id)s,
            %(features)s, %(defasagem_classe)s, %(defasagem_label)s,
            %(prob_atraso)s, %(prob_neutro)s, %(prob_avanco)s,
            %(confidence)s, %(model_version)s
        )
        ON CONFLICT (prediction_id) DO NOTHING
    """
    probs = record.get("probabilities", {})
    conn = get_connection(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "prediction_id":   record.get("prediction_id"),
                "predicted_at":    record.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "api_user":        record.get("user"),
                "student_id":      record.get("student_id"),
                "features":        json.dumps(record.get("features", {}), default=str),
                "defasagem_classe": record.get("defasagem_classe"),
                "defasagem_label": record.get("defasagem_label"),
                "prob_atraso":     probs.get("atraso"),
                "prob_neutro":     probs.get("neutro"),
                "prob_avanco":     probs.get("avanço"),
                "confidence":      record.get("confidence"),
                "model_version":   record.get("model_version"),
            })
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erro ao inserir predição: {e}")
    finally:
        conn.close()


def query_predictions(
    database_url: str,
    user: Optional[str] = None,
    student_id: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict]:
    """Consulta logs de predição com filtros opcionais."""
    conditions = []
    params: list = []
    if user:
        conditions.append("api_user = %s")
        params.append(user)
    if student_id:
        conditions.append("student_id = %s")
        params.append(student_id)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM predictions {where} ORDER BY predicted_at DESC LIMIT %s"
    params.append(limit)

    with db_cursor(database_url) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ============================================================
# MODEL RUNS
# ============================================================

def create_model_run(run_id: str, model_name: str, database_url: str) -> None:
    """Registra início de um run de treinamento."""
    with db_cursor(database_url) as cur:
        cur.execute(
            """
            INSERT INTO model_runs (run_id, model_name, status)
            VALUES (%s, %s, 'running')
            ON CONFLICT (run_id) DO NOTHING
            """,
            (run_id, model_name),
        )


def update_model_run(
    run_id: str,
    database_url: str,
    best_params: Optional[dict] = None,
    metrics: Optional[dict] = None,
    confusion_matrix: Optional[list] = None,
    classification_report: Optional[str] = None,
    status: str = "completed",
    notes: str = "",
) -> None:
    """Atualiza metadados de um run ao final do treinamento/avaliação."""
    with db_cursor(database_url) as cur:
        cur.execute(
            """
            UPDATE model_runs SET
                finished_at           = NOW(),
                status                = %s,
                best_params           = %s,
                metrics               = %s,
                confusion_matrix      = %s,
                classification_report = %s,
                notes                 = %s
            WHERE run_id = %s
            """,
            (
                status,
                json.dumps(best_params or {}, default=str),
                json.dumps(metrics or {}, default=str),
                json.dumps(confusion_matrix or [], default=str),
                classification_report or "",
                notes,
                run_id,
            ),
        )


def list_model_runs(database_url: str) -> List[Dict]:
    """Lista todos os runs de modelo registrados."""
    with db_cursor(database_url) as cur:
        cur.execute("SELECT * FROM model_runs ORDER BY started_at DESC")
        return [dict(r) for r in cur.fetchall()]


# ============================================================
# MONITORING
# ============================================================

def log_monitoring_event(
    event_type: str,
    details: dict,
    database_url: str,
    severity: str = "info",
) -> None:
    """Registra evento de monitoramento (drift, erro, alerta, etc.)."""
    conn = get_connection(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monitoring_logs (event_type, details, severity)
                VALUES (%s, %s, %s)
                """,
                (event_type, json.dumps(details, default=str), severity),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Erro ao gravar evento de monitoramento: {e}")
    finally:
        conn.close()


def query_monitoring_logs(
    database_url: str,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 500,
) -> List[Dict]:
    """Consulta eventos de monitoramento."""
    conditions, params = [], []
    if event_type:
        conditions.append("event_type = %s"); params.append(event_type)
    if severity:
        conditions.append("severity = %s"); params.append(severity)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM monitoring_logs {where} ORDER BY logged_at DESC LIMIT %s"
    params.append(limit)
    with db_cursor(database_url) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def get_drift_summary(database_url: str) -> Dict:
    """
    Sumariza distribuição das predições recentes vs. distribuição de treino.
    Usado pelo endpoint /monitor/drift.
    """
    with db_cursor(database_url) as cur:
        cur.execute(
            """
            SELECT
                defasagem_label,
                COUNT(*)                             AS total,
                ROUND(AVG(prob_atraso)::numeric, 4)  AS avg_prob_atraso,
                ROUND(AVG(prob_neutro)::numeric, 4)  AS avg_prob_neutro,
                ROUND(AVG(prob_avanco)::numeric, 4)  AS avg_prob_avanco,
                MIN(predicted_at)                    AS first_pred,
                MAX(predicted_at)                    AS last_pred
            FROM predictions
            GROUP BY defasagem_label
            ORDER BY defasagem_label
            """
        )
        rows = cur.fetchall()

        cur.execute("SELECT COUNT(*) AS total FROM predictions")
        grand_total = cur.fetchone()["total"]

    return {
        "total_predictions": grand_total,
        "class_distribution": [dict(r) for r in rows],
    }


# ============================================================
# HELPERS INTERNOS
# ============================================================

def _safe(val: Any) -> Any:
    """
    Converte tipos NumPy/pandas para Python nativo e trata NaN/Inf → None.
    O psycopg2 não aceita np.float64/np.int64 diretamente como valores —
    eles são interpretados como nomes de schema na interpolação SQL.
    """
    import math

    if val is None:
        return None

    # ── tipos inteiros NumPy ──────────────────────────────────────────────
    try:
        import numpy as np
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass

    # ── pandas NA/NaT ─────────────────────────────────────────────────────
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except (ImportError, TypeError, ValueError):
        pass

    # ── float nativo ──────────────────────────────────────────────────────
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None

    return val