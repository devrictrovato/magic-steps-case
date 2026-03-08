"""
Configurações centralizadas do projeto Magic Steps MLOps.
"""

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "out"
MODELS_DIR = PROJECT_ROOT / "app/model"
FEAST_REPO_DIR = PROJECT_ROOT / "out/features"
ARTIFACTS_DIR = PROJECT_ROOT / "out/"
LOGS_DIR = PROJECT_ROOT / "logs"

for directory in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

class Settings(BaseSettings):
    """Configurações de ambiente do projeto."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    project_name: str = "magic_steps_mlops"
    environment: str = "development"

    # ── PostgreSQL (banco principal — dados brutos, features, logs, monitoramento) ──
    database_url: str = (
        "postgresql://db_magic_steps_user:0hT1fLv2GkMb4McXAhZeUBAJUTgQtSLV"
        "@dpg-d6mo2hlm5p6s73fv0dt0-a.virginia-postgres.render.com/db_magic_steps"
    )

    random_state: int = 42

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "magic_steps_experiment"

    secret_key: str = Field("change_me")
    access_token_expire_minutes: int = Field(60)


# ============================================================
# DATA CONFIGURATION
# ============================================================

@dataclass
class DataConfig:
    target_column: str = "defasagem"
    target_map: Dict[str, int] = field(default_factory=lambda: {})
    defasagem_class_labels: Dict[int, str] = field(default_factory=lambda: {
        0: "atraso", 1: "neutro", 2: "avanço",
    })
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "nota_ingles", "nota_matematica", "nota_portugues",
        "flag_atingiu_pv", "ano_nascimento", "idade",
    ])
    categorical_columns: List[str] = field(default_factory=lambda: [
        "turma", "genero", "pedra_modal",
    ])
    column_alias_map: Dict[str, str] = field(default_factory=lambda: {
        "Fase": "fase", "Turma": "turma", "Ano nasc": "ano_nascimento",
        "Idade 22": "idade", "Gênero": "genero", "Ano ingresso": "ano_ingresso",
        "Pedra 20": "pedra_2020", "Pedra 21": "pedra_2021", "Pedra 22": "pedra_2022",
        "INDE 22": "score_inde", "IAA": "score_iaa", "IEG": "score_ieg",
        "IPS": "score_ips", "IDA": "score_ida", "IPV": "score_ipv",
        "IAN": "score_ian", "Matem": "nota_matematica", "Portug": "nota_portugues",
        "Inglês": "nota_ingles", "Cg": "nota_cg", "Cf": "nota_cf",
        "Ct": "nota_ct", "Nº Av": "num_avaliacoes", "Defas": "defasagem",
        "Atingiu PV": "flag_atingiu_pv",
    })
    final_feature_order: List[str] = field(default_factory=lambda: [
        "fase", "ano_ingresso", "score_inde", "score_iaa", "score_ieg",
        "score_ips", "score_ida", "score_ipv", "score_ian",
        "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
        "turma", "genero", "pedra_modal",
    ])


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    val_size: float = 0.2
    test_size: float = 0.15
    patience: int = 10
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "hidden_layers": [[64],[128],[128, 64],[256, 128],[256, 128, 64]],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "lr": [1e-2, 5e-3, 1e-3, 5e-4],
        "batch_size": [16, 32, 64],
        "epochs": [100],
    })


# ============================================================
# FEAST CONFIGURATION
# ============================================================

@dataclass
class FeastConfig:
    project_name: str = "magic_steps"
    entity_id_column: str = "student_id"
    event_timestamp_column: str = "event_timestamp"
    feature_views: List[str] = field(default_factory=lambda: [
        "student_academic_features",
        "student_demographic_features",
        "student_performance_features",
    ])


# ============================================================
# INSTANTIATE
# ============================================================

settings = Settings()
data_config = DataConfig()
model_config = ModelConfig()
feast_config = FeastConfig()