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
ARTIFACTS_DIR = PROJECT_ROOT / "out"
MODELS_DIR = PROJECT_ROOT / "app/model"
FEAST_REPO_DIR = PROJECT_ROOT / "out/features"
LOGS_DIR = PROJECT_ROOT / "logs"

# Criar diretórios se não existirem
for directory in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

class Settings(BaseSettings):
    """Configurações de ambiente do projeto."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Projeto
    project_name: str = "magic_steps_mlops"
    environment: str = "development"
    
    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = "magic-steps-ml"
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "magic_steps_features"
    db_user: str = "postgres"
    db_password: str = ""
    
    # Redis (para Feast online store)
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "magic_steps_experiment"
    
    # Random State
    random_state: int = 42


# ============================================================
# DATA CONFIGURATION
# ============================================================

@dataclass
class DataConfig:
    """Configurações relacionadas aos dados."""
    
    target_column: str = "flag_atingiu_pv"
    
    target_map: Dict[str, int] = field(default_factory=lambda: {
        "sim": 1,
        "não": 0,
    })
    
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "nota_ingles",
        "nota_matematica",
        "nota_portugues",
    ])
    
    categorical_columns: List[str] = field(default_factory=lambda: [
        "turma",
        "genero",
        "pedra_modal",
    ])
    
    column_alias_map: Dict[str, str] = field(default_factory=lambda: {
        "Fase": "fase",
        "Turma": "turma",
        "Ano nasc": "ano_nascimento",
        "Idade 22": "idade",
        "Gênero": "genero",
        "Ano ingresso": "ano_ingresso",
        "Pedra 20": "pedra_2020",
        "Pedra 21": "pedra_2021",
        "Pedra 22": "pedra_2022",
        "INDE 22": "score_inde",
        "IAA": "score_iaa",
        "IEG": "score_ieg",
        "IPS": "score_ips",
        "IDA": "score_ida",
        "IPV": "score_ipv",
        "IAN": "score_ian",
        "Matem": "nota_matematica",
        "Portug": "nota_portugues",
        "Inglês": "nota_ingles",
        "Cg": "nota_cg",
        "Cf": "nota_cf",
        "Ct": "nota_ct",
        "Nº Av": "num_avaliacoes",
        "Defas": "defasagem",
        "Atingiu PV": "flag_atingiu_pv",
    })
    
    final_feature_order: List[str] = field(default_factory=lambda: [
        "fase",
        "idade",
        "ano_ingresso",
        "score_inde",
        "score_iaa",
        "score_ieg",
        "score_ips",
        "score_ida",
        "score_ipv",
        "score_ian",
        "nota_cg",
        "nota_cf",
        "nota_ct",
        "num_avaliacoes",
        "defasagem",
        "turma",
        "genero",
        "pedra_modal",
    ])


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """Configurações do modelo."""
    
    # Training
    val_size: float = 0.2
    test_size: float = 0.15
    patience: int = 10
    
    # Grid Search
    param_grid: Dict[str, List[Any]] = field(default_factory=lambda: {
        "hidden_layers": [
            [64],
            [128],
            [128, 64],
            [256, 128],
            [256, 128, 64],
        ],
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
    """Configurações do Feast Feature Store."""
    
    project_name: str = "magic_steps"
    
    # Entity
    entity_id_column: str = "student_id"
    event_timestamp_column: str = "event_timestamp"
    
    # Feature views
    feature_views: List[str] = field(default_factory=lambda: [
        "student_academic_features",
        "student_demographic_features",
        "student_performance_features",
    ])


# ============================================================
# INSTANTIATE CONFIGS
# ============================================================

settings = Settings()
data_config = DataConfig()
model_config = ModelConfig()
feast_config = FeastConfig()