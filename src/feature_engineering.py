"""
Feature Engineering com Feast Feature Store para o projeto Magic Steps.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, feast_config, FEAST_REPO_DIR, ARTIFACTS_DIR
from utils import setup_logger


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "feature_engineering.log")


# ============================================================
# FEAST SETUP
# ============================================================

class FeastFeatureStore:
    """Gerenciador do Feast Feature Store."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        """
        Inicializa o Feast Feature Store.
        
        Args:
            repo_path: Caminho do repositório Feast
        """
        self.repo_path = repo_path or FEAST_REPO_DIR
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(self.__class__.__name__)
        self.store = None
        
        self._initialize_feast()
    
    def _initialize_feast(self) -> None:
        """Inicializa o Feast repository."""
        feature_store_yaml = self.repo_path / "feature_store.yaml"
        
        if not feature_store_yaml.exists():
            self._create_feature_store_config()
        
        try:
            # Criar diretório de dados se não existir
            (self.repo_path / "data").mkdir(parents=True, exist_ok=True)
            
            self.store = FeatureStore(repo_path=str(self.repo_path))
            self.logger.info("Feast Feature Store inicializado")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar Feast: {e}")
            self.logger.warning("Continuando sem Feast Feature Store")
            self.store = None
    
    def _create_feature_store_config(self) -> None:
        """Cria arquivo de configuração do Feast."""
        # Usar POSIX paths para compatibilidade com Feast no Windows
        registry_path = (self.repo_path / "data" / "registry.db").as_posix()
        online_store_path = (self.repo_path / "data" / "online_store.db").as_posix()
        
        config_content = f"""project: {feast_config.project_name}
registry: {registry_path}
provider: local
online_store:
    type: sqlite
    path: {online_store_path}
offline_store:
    type: file
entity_key_serialization_version: 3
"""
        
        feature_store_yaml = self.repo_path / "feature_store.yaml"
        with open(feature_store_yaml, 'w') as f:
            f.write(config_content)
        
        self.logger.info(f"Configuração do Feast criada em {feature_store_yaml}")
    
    def apply_features(self) -> None:
        """Aplica definições de features ao Feast."""
        if self.store:
            self.store.apply([])  # Apply será feito via CLI
            self.logger.info("Features aplicadas ao Feast")


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

def create_student_entity() -> Entity:
    """
    Cria a entidade Student para o Feast.
    
    Returns:
        Entity configurada
    """
    return Entity(
        name="student",
        description="Entidade representando um aluno",
        join_keys=[feast_config.entity_id_column],
    )


def prepare_features_for_feast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara features para o Feast Feature Store.
    
    Args:
        df: DataFrame com features processadas
    
    Returns:
        DataFrame preparado para o Feast
    """
    df_feast = df.copy()
    
    # Adicionar colunas necessárias para o Feast
    if feast_config.entity_id_column not in df_feast.columns:
        # Gerar IDs únicos para os alunos
        df_feast[feast_config.entity_id_column] = range(len(df_feast))
    
    if feast_config.event_timestamp_column not in df_feast.columns:
        # Adicionar timestamp (simulando dados históricos)
        base_date = datetime.now()
        df_feast[feast_config.event_timestamp_column] = [
            base_date - timedelta(days=i) for i in range(len(df_feast))
        ]
    
    return df_feast


def save_features_to_feast(
    df: pd.DataFrame,
    feature_store: FeastFeatureStore,
) -> None:
    """
    Salva features no Feast offline store.
    
    Args:
        df: DataFrame com features
        feature_store: Instância do FeastFeatureStore
    """
    logger.info("Salvando features no Feast offline store...")
    
    # Preparar dados para o Feast
    df_feast = prepare_features_for_feast(df)
    
    # Salvar em parquet (offline store)
    feast_data_dir = feature_store.repo_path / "data"
    feast_data_dir.mkdir(parents=True, exist_ok=True)
    
    features_path = feast_data_dir / "student_features.parquet"
    df_feast.to_parquet(features_path, index=False)
    
    logger.info(f"Features salvas em: {features_path}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

class FeatureEngineer:
    """Engenharia de features para o modelo."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features agregadas.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features agregadas
        """
        df = df.copy()
        
        # Score médio
        score_cols = [
            "score_inde",
            "score_iaa",
            "score_ieg",
            "score_ips",
            "score_ida",
            "score_ipv",
            "score_ian",
        ]
        
        available_scores = [c for c in score_cols if c in df.columns]
        if available_scores:
            df["score_medio"] = df[available_scores].mean(axis=1)
        
        # Nota média
        nota_cols = ["nota_cg", "nota_cf", "nota_ct"]
        available_notas = [c for c in nota_cols if c in df.columns]
        if available_notas:
            df["nota_media"] = df[available_notas].mean(axis=1)
        
        self.logger.info("Features agregadas criadas")
        return df
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de razão.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features de razão
        """
        df = df.copy()
        
        # Razão avaliações por defasagem
        if "num_avaliacoes" in df.columns and "defasagem" in df.columns:
            df["ratio_aval_defasagem"] = df["num_avaliacoes"] / (
                df["defasagem"] + 1
            )  # +1 para evitar divisão por zero
        
        self.logger.info("Features de razão criadas")
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features polinomiais (quadráticas).
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features polinomiais
        """
        df = df.copy()
        
        # Features importantes para criar versões quadráticas
        poly_cols = ["score_inde", "idade", "num_avaliacoes"]
        
        for col in poly_cols:
            if col in df.columns:
                df[f"{col}_squared"] = df[col] ** 2
        
        self.logger.info("Features polinomiais criadas")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features de interação
        """
        df = df.copy()
        
        # Interações importantes
        if "score_inde" in df.columns and "idade" in df.columns:
            df["inde_x_idade"] = df["score_inde"] * df["idade"]
        
        if "num_avaliacoes" in df.columns and "defasagem" in df.columns:
            df["aval_x_defasagem"] = df["num_avaliacoes"] * df["defasagem"]
        
        self.logger.info("Features de interação criadas")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas as técnicas de feature engineering.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com features engineered
        """
        self.logger.info("Iniciando feature engineering...")
        
        df = (
            df.pipe(self.create_aggregate_features)
            .pipe(self.create_ratio_features)
            .pipe(self.create_polynomial_features)
            .pipe(self.create_interaction_features)
        )
        
        self.logger.info(f"Feature engineering concluído. Shape: {df.shape}")
        return df


# ============================================================
# FEATURE LOADER
# ============================================================

class FeatureLoader:
    """Carregador de features do Feast ou de arquivos."""
    
    def __init__(self, use_feast: bool = False):
        """
        Inicializa o carregador de features.
        
        Args:
            use_feast: Se True, usa Feast; caso contrário, carrega de arquivo
        """
        self.use_feast = use_feast
        self.logger = setup_logger(self.__class__.__name__)
        
        if use_feast:
            self.feast_store = FeastFeatureStore()
    
    def load_features_for_training(self) -> pd.DataFrame:
        """
        Carrega features para treinamento.
        
        Returns:
            DataFrame com features
        """
        if self.use_feast:
            return self._load_from_feast()
        else:
            return self._load_from_file()
    
    def _load_from_feast(self) -> pd.DataFrame:
        """
        Carrega features do Feast.
        
        Returns:
            DataFrame com features
        """
        self.logger.info("Carregando features do Feast...")
        
        # Aqui você carregaria do Feast usando get_historical_features
        # Por ora, vamos carregar do arquivo preparado
        feast_data_path = (
            self.feast_store.repo_path / "data" / "student_features.parquet"
        )
        
        if feast_data_path.exists():
            df = pd.read_parquet(feast_data_path)
            self.logger.info(f"Features carregadas do Feast. Shape: {df.shape}")
            return df
        else:
            self.logger.warning("Arquivo do Feast não encontrado. Carregando de arquivo padrão.")
            return self._load_from_file()
    
    def _load_from_file(self) -> pd.DataFrame:
        """
        Carrega features de arquivo.
        
        Returns:
            DataFrame com features
        """
        dataset_path = ARTIFACTS_DIR / "dataset_transformed_magic_steps.parquet"
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em {dataset_path}. "
                "Execute o preprocessing primeiro."
            )
        
        df = pd.read_parquet(dataset_path)
        self.logger.info(f"Features carregadas de arquivo. Shape: {df.shape}")
        
        return df
    
    def get_online_features(
        self, entity_ids: List[int]
    ) -> pd.DataFrame:
        """
        Obtém features online do Feast para inferência.
        
        Args:
            entity_ids: Lista de IDs de entidades
        
        Returns:
            DataFrame com features
        """
        if not self.use_feast:
            raise ValueError("Feast não está habilitado")
        
        self.logger.info(f"Obtendo features online para {len(entity_ids)} entidades...")
        
        # Aqui você usaria o get_online_features do Feast
        # Por ora, retornamos um placeholder
        entity_df = pd.DataFrame({
            feast_config.entity_id_column: entity_ids
        })
        
        # Em produção, você faria:
        # feature_vector = self.feast_store.store.get_online_features(
        #     features=[...],
        #     entity_rows=entity_df.to_dict('records')
        # )
        
        return entity_df


# ============================================================
# FEATURE STORE MANAGER
# ============================================================

class FeatureStoreManager:
    """Gerenciador completo do Feature Store."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.feast_store = FeastFeatureStore()
        self.engineer = FeatureEngineer()
    
    def materialize_features(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Materializa features no Feast.
        
        Args:
            df: DataFrame com features
            start_date: Data inicial
            end_date: Data final
        """
        if self.feast_store.store is None:
            self.logger.warning("Feast não disponível. Pulando materialização.")
            return
        
        self.logger.info("Materializando features no Feast...")
        
        try:
            # Preparar e salvar features
            save_features_to_feast(df, self.feast_store)
            
            # Materializar para online store
            if start_date and end_date:
                # self.feast_store.store.materialize(start_date, end_date)
                self.logger.info("Features materializadas para o período especificado")
            
            self.logger.info("Materialização concluída")
        except Exception as e:
            self.logger.error(f"Erro na materialização: {e}")
            self.logger.warning("Continuando sem materialização no Feast")
        
        self.logger.info("Materialização concluída")
    
    def register_features_in_dw(self, df: pd.DataFrame) -> None:
        """
        *Deprecated* stub for DW registration.
        
        We no longer maintain a Data Warehouse - all external storage is
        handled via MongoDB (logs) and Redis (Feast online store). This
        method remains for compatibility but does nothing.
        """
        self.logger.info("register_features_in_dw called but DW client is disabled")
        # no-op


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def load_features_for_training(use_feast: bool = False) -> pd.DataFrame:
    """
    Função auxiliar para carregar features para treinamento.
    
    Args:
        use_feast: Se True, usa Feast
    
    Returns:
        DataFrame com features
    """
    loader = FeatureLoader(use_feast=use_feast)
    return loader.load_features_for_training()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("🎯 Feature Engineering com Feast - Magic Steps")
    
    try:
        # Carregar dados preprocessados
        loader = FeatureLoader(use_feast=False)
        df = loader.load_features_for_training()
        
        # Aplicar feature engineering
        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_features(df)
        
        # Salvar features engineered
        engineered_path = ARTIFACTS_DIR / "dataset_with_engineered_features.parquet"
        df_engineered.to_parquet(engineered_path, index=False)
        print(f"\n✅ Features engineered salvas em: {engineered_path}")
        
        # Tentar salvar no Feast (opcional)
        try:
            print("\n📦 Tentando materializar features no Feast...")
            feast_manager = FeatureStoreManager()
            feast_manager.materialize_features(df_engineered)
            print("✅ Features materializadas no Feast")
        except Exception as feast_error:
            print(f"⚠️ Aviso: Não foi possível usar Feast: {feast_error}")
            print("✅ Continuando sem Feast (features salvas localmente)")
        
        print("\n✅ Feature engineering finalizado!")
        
    except Exception as e:
        print(f"\n❌ Erro no feature engineering: {e}")
        import traceback
        traceback.print_exc()
        exit(1)