"""
Classes utilitárias para o projeto Magic Steps MLOps.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from settings import settings, LOGS_DIR


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        if log_file:
            log_path = LOGS_DIR / log_file
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


# ============================================================
# FILE MANAGER
# ============================================================

class FileManager:
    def __init__(self, base_path: Optional[Path] = None):
        from settings import DATA_DIR
        self.base_path = base_path or DATA_DIR
        self.logger = setup_logger(self.__class__.__name__)

    def get_data_path(self, filename: str) -> Path:
        return self.base_path / filename

    def list_files(self, pattern: str = "*") -> list:
        return list(self.base_path.glob(pattern))

    def ensure_directory(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:
    def __init__(self, registry_path: Optional[Path] = None):
        from settings import MODELS_DIR
        self.registry_path = registry_path or MODELS_DIR
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)

    def save_model_metadata(self, model_name: str, metadata: Dict[str, Any]) -> None:
        import json
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Metadados salvos em {metadata_path}")

    def load_model_metadata(self, model_name: str) -> Dict[str, Any]:
        import json
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_models(self) -> list:
        model_files = list(self.registry_path.glob("*_metadata.json"))
        return [f.stem.replace('_metadata', '') for f in model_files]


# ============================================================
# POSTGRES LOGGER  (substitui MongoLogger)
# ============================================================

class PostgresLogger:
    """
    Registra logs de predição e eventos de monitoramento no PostgreSQL.
    Drop-in replacement do MongoLogger — mesma interface pública.
    """

    def __init__(self, database_url: Optional[str] = None):
        from settings import settings as _s
        import logging as _logging
        from db import ensure_schema

        self.database_url = database_url or _s.database_url
        self.logger = _logging.getLogger(self.__class__.__name__)
        self._available = False

        try:
            ensure_schema(self.database_url)
            self._available = True
            self.logger.info("PostgresLogger: schema ok.")
        except Exception as e:
            self.logger.error(f"PostgresLogger indisponível: {e}")

    def log_prediction(self, record: dict) -> None:
        """Insere um registro de predição no PostgreSQL."""
        if not self._available:
            self.logger.debug("PostgreSQL não disponível, log ignorado.")
            return
        try:
            from db import log_prediction as _log
            _log(record, self.database_url)
        except Exception as e:
            self.logger.error(f"Falha ao gravar log de predição: {e}")

    def log_monitoring(self, event_type: str, details: dict,
                       severity: str = "info") -> None:
        """Registra evento de monitoramento."""
        if not self._available:
            return
        try:
            from db import log_monitoring_event
            log_monitoring_event(event_type, details, self.database_url, severity)
        except Exception as e:
            self.logger.error(f"Falha ao gravar evento de monitoramento: {e}")


# Alias para manter compatibilidade com código que importava MongoLogger
MongoLogger = PostgresLogger


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'setup_logger',
    'FileManager',
    'ModelRegistry',
    'PostgresLogger',
    'MongoLogger',
]