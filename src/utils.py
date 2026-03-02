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
    """
    Configura e retorna um logger.
    
    Args:
        name: Nome do logger
        log_file: Caminho do arquivo de log (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
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
    """
    Gerenciador de arquivos do projeto.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Inicializa o gerenciador de arquivos.
        
        Args:
            base_path: Caminho base para os arquivos
        """
        from settings import DATA_DIR
        self.base_path = base_path or DATA_DIR
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_data_path(self, filename: str) -> Path:
        """
        Retorna o caminho completo para um arquivo de dados.
        
        Args:
            filename: Nome do arquivo
        
        Returns:
            Caminho completo do arquivo
        """
        return self.base_path / filename
    
    def list_files(self, pattern: str = "*") -> list:
        """
        Lista arquivos no diretório base.
        
        Args:
            pattern: Padrão para filtrar arquivos
        
        Returns:
            Lista de caminhos de arquivos
        """
        return list(self.base_path.glob(pattern))
    
    def ensure_directory(self, path: Path) -> Path:
        """
        Garante que um diretório existe.
        
        Args:
            path: Caminho do diretório
        
        Returns:
            Caminho do diretório
        """
        path.mkdir(parents=True, exist_ok=True)
        return path


# ============================================================
# MODEL REGISTRY
# ============================================================

class ModelRegistry:
    """
    Registro de modelos treinados.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Inicializa o registro de modelos.
        
        Args:
            registry_path: Caminho do registro de modelos
        """
        from settings import MODELS_DIR
        self.registry_path = registry_path or MODELS_DIR
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)
    
    def save_model_metadata(
        self,
        model_name: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Salva metadados de um modelo.
        
        Args:
            model_name: Nome do modelo
            metadata: Metadados do modelo
        """
        import json
        
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metadados salvos em {metadata_path}")
    
    def load_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Carrega metadados de um modelo.
        
        Args:
            model_name: Nome do modelo
        
        Returns:
            Metadados do modelo
        """
        import json
        
        metadata_path = self.registry_path / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> list:
        """
        Lista todos os modelos registrados.
        
        Returns:
            Lista de nomes de modelos
        """
        model_files = list(self.registry_path.glob("*_metadata.json"))
        models = [f.stem.replace('_metadata', '') for f in model_files]
        return models


# ============================================================
# MONGODB LOGGER
# ============================================================

try:
    from pymongo import MongoClient as _PymongoClient
except ImportError:  # pymongo may not be installed during tests
    _PymongoClient = None


class MongoLogger:
    """Cliente simples para gravar logs de predição em MongoDB."""

    def __init__(self, uri: str | None = None, db_name: str | None = None):
        from settings import settings

        self.uri = uri or settings.mongo_uri
        self.db_name = db_name or settings.mongo_db
        self.logger = setup_logger(self.__class__.__name__)

        if _PymongoClient is None:
            self.logger.warning("pymongo não instalado, MongoLogger ficará inoperante")
            self.client = None
            self.db = None
        else:
            try:
                self.client = _PymongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=5_000,
                    connectTimeoutMS=5_000,
                    socketTimeoutMS=10_000,
                    tls=True,
                    tlsAllowInvalidCertificates=True,
                )
                # Valida conectividade imediatamente
                self.client.admin.command("ping")
                self.db = self.client[self.db_name]
            except Exception as e:
                self.logger.error(f"Erro ao conectar MongoDB: {e}")
                self.client = None
                self.db = None

    def log_prediction(self, record: dict) -> None:
        """Insere um documento na coleção `predictions`."""
        if self.db is None:
            return
        try:
            self.db.predictions.insert_one(record)
            self.logger.debug("log de predição gravado em MongoDB")
        except Exception as e:
            self.logger.error(f"falha ao inserir log de predição: {e}")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'setup_logger',
    'FileManager',
    'ModelRegistry',
    'MongoLogger',
]