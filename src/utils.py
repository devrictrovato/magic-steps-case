"""
Classes utilitárias para o projeto Magic Steps MLOps.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

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
# AWS CLIENT
# ============================================================

class AWSClient:
    """
    Cliente para interação com serviços AWS (S3, SageMaker, etc).
    """
    
    def __init__(
        self,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ):
        """
        Inicializa o cliente AWS.
        
        Args:
            region: Região AWS
            access_key_id: AWS Access Key ID
            secret_access_key: AWS Secret Access Key
        """
        self.region = region or settings.aws_region
        self.access_key_id = access_key_id or settings.aws_access_key_id
        self.secret_access_key = secret_access_key or settings.aws_secret_access_key
        
        self.logger = setup_logger(self.__class__.__name__)
        
        # Inicializar clientes
        self._s3_client = None
        self._sagemaker_client = None
    
    @property
    def s3(self):
        """Retorna o cliente S3."""
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            )
        return self._s3_client
    
    @property
    def sagemaker(self):
        """Retorna o cliente SageMaker."""
        if self._sagemaker_client is None:
            self._sagemaker_client = boto3.client(
                'sagemaker',
                region_name=self.region,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            )
        return self._sagemaker_client
    
    def upload_file_to_s3(
        self,
        file_path: Path,
        bucket: Optional[str] = None,
        s3_key: Optional[str] = None,
    ) -> str:
        """
        Faz upload de um arquivo para o S3.
        
        Args:
            file_path: Caminho do arquivo local
            bucket: Nome do bucket S3
            s3_key: Chave do objeto no S3
        
        Returns:
            URI do objeto no S3
        """
        bucket = bucket or settings.s3_bucket
        s3_key = s3_key or file_path.name
        
        try:
            self.s3.upload_file(str(file_path), bucket, s3_key)
            s3_uri = f"s3://{bucket}/{s3_key}"
            self.logger.info(f"Arquivo enviado para {s3_uri}")
            return s3_uri
        except ClientError as e:
            self.logger.error(f"Erro ao fazer upload para S3: {e}")
            raise
    
    def download_file_from_s3(
        self,
        s3_key: str,
        local_path: Path,
        bucket: Optional[str] = None,
    ) -> Path:
        """
        Baixa um arquivo do S3.
        
        Args:
            s3_key: Chave do objeto no S3
            local_path: Caminho local para salvar o arquivo
            bucket: Nome do bucket S3
        
        Returns:
            Caminho do arquivo baixado
        """
        bucket = bucket or settings.s3_bucket
        
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(bucket, s3_key, str(local_path))
            self.logger.info(f"Arquivo baixado de s3://{bucket}/{s3_key}")
            return local_path
        except ClientError as e:
            self.logger.error(f"Erro ao baixar do S3: {e}")
            raise
    
    def list_s3_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
    ) -> list:
        """
        Lista objetos em um bucket S3.
        
        Args:
            prefix: Prefixo para filtrar objetos
            bucket: Nome do bucket S3
        
        Returns:
            Lista de chaves dos objetos
        """
        bucket = bucket or settings.s3_bucket
        
        try:
            response = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            objects = [obj['Key'] for obj in response.get('Contents', [])]
            return objects
        except ClientError as e:
            self.logger.error(f"Erro ao listar objetos do S3: {e}")
            raise


# ============================================================
# DATABASE CLIENT
# ============================================================

class DWClient:
    """
    Cliente para interação com o Data Warehouse (PostgreSQL).
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Inicializa o cliente do Data Warehouse.
        
        Args:
            host: Host do banco de dados
            port: Porta do banco de dados
            database: Nome do banco de dados
            user: Usuário do banco de dados
            password: Senha do banco de dados
        """
        self.host = host or settings.db_host
        self.port = port or settings.db_port
        self.database = database or settings.db_name
        self.user = user or settings.db_user
        self.password = password or settings.db_password
        
        self.logger = setup_logger(self.__class__.__name__)
        self._connection = None
    
    @contextmanager
    def get_connection(self):
        """
        Context manager para gerenciar conexões com o banco.
        
        Yields:
            Conexão com o banco de dados
        """
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Erro no banco de dados: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
    ) -> Optional[list]:
        """
        Executa uma query no banco de dados.
        
        Args:
            query: Query SQL a ser executada
            params: Parâmetros da query
            fetch: Se True, retorna os resultados
        
        Returns:
            Lista de resultados ou None
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                
                if fetch:
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                
                return None
    
    def execute_many(
        self,
        query: str,
        data: list,
    ) -> None:
        """
        Executa múltiplas operações no banco de dados.
        
        Args:
            query: Query SQL a ser executada
            data: Lista de tuplas com os dados
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, data)
        
        self.logger.info(f"Executadas {len(data)} operações no banco")
    
    def insert_dataframe(
        self,
        df,
        table_name: str,
        if_exists: str = 'append',
    ) -> None:
        """
        Insere um DataFrame no banco de dados.
        
        Args:
            df: DataFrame a ser inserido
            table_name: Nome da tabela
            if_exists: Ação se a tabela existir ('append', 'replace', 'fail')
        """
        from sqlalchemy import create_engine
        
        engine = create_engine(
            f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
        )
        
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        self.logger.info(f"DataFrame inserido na tabela {table_name}")


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
# EXPORTS
# ============================================================

__all__ = [
    'setup_logger',
    'AWSClient',
    'DWClient',
    'FileManager',
    'ModelRegistry',
]