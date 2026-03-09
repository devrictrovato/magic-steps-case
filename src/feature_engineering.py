"""
Feature Engineering para o projeto Magic Steps.

Armazenamento:
  - Features engenheiradas são persistidas no PostgreSQL (tabela engineered_features)
  - Leitura para treinamento também usa o PostgreSQL como fonte primária
  - Parquet local é mantido como fallback
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, feast_config, OUTPUT_DIR, settings
from utils import setup_logger
from db import (
    ensure_schema,
    insert_engineered_features,
    load_engineered_features,
    log_monitoring_event,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "feature_engineering.log")


# ============================================================
# FEATURE ENGINEER
# ============================================================

class FeatureEngineer:
    """Engenharia de features para o modelo."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config

    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        score_cols = ["score_inde","score_iaa","score_ieg","score_ips","score_ida","score_ipv","score_ian"]
        available_scores = [c for c in score_cols if c in df.columns]
        if available_scores:
            df["score_medio"] = df[available_scores].mean(axis=1)
        nota_cols = ["nota_cg", "nota_cf", "nota_ct"]
        available_notas = [c for c in nota_cols if c in df.columns]
        if available_notas:
            df["nota_media"] = df[available_notas].mean(axis=1)
        self.logger.info("Features agregadas criadas")
        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "num_avaliacoes" in df.columns and "fase" in df.columns:
            df["ratio_aval_fase"] = df["num_avaliacoes"] / (df["fase"] + 1)
        self.logger.info("Features de razão criadas")
        return df

    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["score_inde", "num_avaliacoes"]:
            if col in df.columns:
                df[f"{col}_squared"] = df[col] ** 2
        self.logger.info("Features polinomiais criadas")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "score_inde" in df.columns and "fase" in df.columns:
            df["inde_x_fase"] = df["score_inde"] * df["fase"]
        if "num_avaliacoes" in df.columns and "fase" in df.columns:
            df["aval_x_fase"] = df["num_avaliacoes"] * df["fase"]
        self.logger.info("Features de interação criadas")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
# FEATURE STORE MANAGER  (PostgreSQL como store central)
# ============================================================

class FeatureStoreManager:
    """
    Gerenciador do Feature Store usando PostgreSQL como backend.

    O Feast (Redis/SQLite) foi substituído pelo PostgreSQL para unificar
    toda a persistência do projeto em um único banco de dados.
    """

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.engineer = FeatureEngineer()

    def materialize_features(
        self,
        df: pd.DataFrame,
        pipeline_run_id: Optional[str] = None,
    ) -> str:
        """
        Persiste features no PostgreSQL (engineered_features).

        Args:
            df:               DataFrame com features engenheiradas
            pipeline_run_id:  ID da execução do pipeline (gerado automaticamente se None)

        Returns:
            pipeline_run_id utilizado
        """
        if pipeline_run_id is None:
            pipeline_run_id = str(uuid.uuid4())

        self.logger.info(f"Materializando features no PostgreSQL (run_id={pipeline_run_id})…")

        try:
            ensure_schema(settings.database_url)

            # Adicionar metadados para o Feast virtual
            df_feast = df.copy()
            if feast_config.entity_id_column not in df_feast.columns:
                df_feast[feast_config.entity_id_column] = range(len(df_feast))
            if feast_config.event_timestamp_column not in df_feast.columns:
                base_date = datetime.now()
                df_feast[feast_config.event_timestamp_column] = [
                    base_date - timedelta(days=i) for i in range(len(df_feast))
                ]

            n = insert_engineered_features(
                df_feast, settings.database_url, pipeline_run_id=pipeline_run_id
            )
            self.logger.info(f"✅ {n} features materializadas no PostgreSQL.")

            log_monitoring_event(
                event_type="features_materialized",
                details={"pipeline_run_id": pipeline_run_id, "rows": n, "shape": list(df.shape)},
                database_url=settings.database_url,
            )
        except Exception as e:
            self.logger.error(f"Erro na materialização: {e}")
            raise

        return pipeline_run_id

    def register_features_in_dw(self, df: pd.DataFrame) -> None:
        """Stub de compatibilidade — redireciona para materialize_features."""
        self.logger.info("register_features_in_dw → redirecionando para materialize_features")
        self.materialize_features(df)


# ============================================================
# FEATURE LOADER
# ============================================================

class FeatureLoader:
    """
    Carregador de features.
    Prioridade: PostgreSQL → parquet local → erro.
    """

    def __init__(self, use_db: bool = True):
        self.use_db = use_db
        self.logger = setup_logger(self.__class__.__name__)

    def load_features_for_training(self, pipeline_run_id: Optional[str] = None) -> pd.DataFrame:
        if self.use_db:
            return self._load_from_db(pipeline_run_id)
        return self._load_from_file()

    def _load_from_db(self, pipeline_run_id: Optional[str] = None) -> pd.DataFrame:
        self.logger.info("Carregando features do PostgreSQL…")
        try:
            ensure_schema(settings.database_url)
            df = load_engineered_features(settings.database_url, pipeline_run_id)
            self.logger.info(f"Features carregadas do PostgreSQL. Shape: {df.shape}")
            return df
        except ValueError as e:
            self.logger.warning(f"Nenhuma feature no PostgreSQL ({e}). Tentando parquet local…")
            return self._load_from_file()

    def _load_from_file(self) -> pd.DataFrame:
        dataset_path = OUTPUT_DIR / "dataset_transformed_magic_steps.parquet"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em {dataset_path}. "
                "Execute o preprocessing primeiro."
            )
        df = pd.read_parquet(dataset_path)
        self.logger.info(f"Features carregadas de arquivo. Shape: {df.shape}")
        return df

    def get_online_features(self, entity_ids: List[int]) -> pd.DataFrame:
        """Obtém features online do PostgreSQL para inferência."""
        self.logger.info(f"Obtendo features para {len(entity_ids)} entidades…")
        try:
            ensure_schema(settings.database_url)
            from db import db_cursor
            with db_cursor(settings.database_url) as cur:
                cur.execute(
                    "SELECT * FROM engineered_features WHERE student_id = ANY(%s) "
                    "ORDER BY created_at DESC",
                    (entity_ids,),
                )
                rows = cur.fetchall()
            return pd.DataFrame([dict(r) for r in rows])
        except Exception as e:
            self.logger.error(f"Erro ao buscar features online: {e}")
            return pd.DataFrame({feast_config.entity_id_column: entity_ids})


# ============================================================
# FUNÇÃO AUXILIAR PÚBLICA
# ============================================================

def load_features_for_training(
    use_feast: bool = False,
    use_db: bool = True,
    pipeline_run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carrega features para treinamento.

    Args:
        use_feast:        Ignorado (legado — mantido para compatibilidade de assinatura)
        use_db:           Se True, usa PostgreSQL como fonte primária
        pipeline_run_id:  Filtra por execução específica do pipeline

    Returns:
        DataFrame com features
    """
    loader = FeatureLoader(use_db=use_db)
    return loader.load_features_for_training(pipeline_run_id=pipeline_run_id)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("🎯 Feature Engineering com PostgreSQL — Magic Steps")

    try:
        loader = FeatureLoader(use_db=True)
        df = loader.load_features_for_training()

        engineer = FeatureEngineer()
        df_engineered = engineer.engineer_features(df)

        # Salvar parquet local como backup
        engineered_path = OUTPUT_DIR / "dataset_with_engineered_features.parquet"
        df_engineered.to_parquet(engineered_path, index=False)
        print(f"\n✅ Features salvas localmente em: {engineered_path}")

        # Materializar no PostgreSQL
        fsm = FeatureStoreManager()
        run_id = fsm.materialize_features(df_engineered)
        print(f"✅ Features materializadas no PostgreSQL (run_id={run_id})")

        print("\n✅ Feature engineering finalizado!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback; traceback.print_exc()
        exit(1)