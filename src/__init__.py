"""
Core ML pipeline package (src/).

Os módulos deste pacote utilizam imports absolutos entre si
(ex.: `from settings import ...`, `from utils import ...`),
o que requer que o diretório `src/` esteja no sys.path.
Isso é garantido por main.py e routes.py no startup da API,
e pelo bloco `sys.path.append(...)` presente em cada módulo
quando executados diretamente.

Reexportamos aqui apenas os símbolos públicos estáveis de cada
módulo, usando seus `__all__` quando definidos. Imports com
wildcard (`*`) sem `__all__` foram removidos para evitar
importações implícitas que mascaram erros e poluem o namespace.
"""

from .settings import settings, data_config, model_config, feast_config
from .utils import setup_logger, FileManager, ModelRegistry, PostgresLogger, MongoLogger
from .db import (
    ensure_schema,
    insert_engineered_features,
    load_engineered_features,
    log_prediction,
    query_predictions,
    get_drift_summary,
    create_model_run,
    update_model_run,
    list_model_runs,
    log_monitoring_event,
    query_monitoring_logs,
)
from .feature_engineering import (
    FeatureEngineer,
    FeatureStoreManager,
    FeatureLoader,
    load_features_for_training,
)
from .preprocessing import *   # mantém wildcard pois preprocessing.py define __all__
from .evaluate import *        # mantém wildcard pois evaluate.py define __all__
from .train import (
    MagicStepsDataset,
    MagicStepsNet,
    Trainer,
    GridSearchCV,
    TrainingPipeline,
)

__all__ = [
    # settings
    "settings",
    "data_config",
    "model_config",
    "feast_config",
    # utils
    "setup_logger",
    "FileManager",
    "ModelRegistry",
    "PostgresLogger",
    "MongoLogger",
    # db
    "ensure_schema",
    "insert_engineered_features",
    "load_engineered_features",
    "log_prediction",
    "query_predictions",
    "get_drift_summary",
    "create_model_run",
    "update_model_run",
    "list_model_runs",
    "log_monitoring_event",
    "query_monitoring_logs",
    # feature_engineering
    "FeatureEngineer",
    "FeatureStoreManager",
    "FeatureLoader",
    "load_features_for_training",
    # train
    "MagicStepsDataset",
    "MagicStepsNet",
    "Trainer",
    "GridSearchCV",
    "TrainingPipeline",
]