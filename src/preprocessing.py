"""
Pipeline de pré-processamento de dados para o projeto Magic Steps.

Fluxo de dados:
  1. DataExtractor  → lê Excel/CSV  ──OR──  carrega de raw_student_data (PostgreSQL)
  2. DataTransformer → limpeza + bucketing
  3. DataBalancer   → oversampling sintético
  4. Preprocessor   → MinMaxScaler + OrdinalEncoder  (joblib salvo em out/)
  5. _save_artifacts → salva parquet local  +  persiste no PostgreSQL
"""

from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, settings, OUTPUT_DIR
from utils import setup_logger, FileManager
from db import (
    ensure_schema,
    insert_raw_dataframe,
    load_raw_dataframe,
    insert_engineered_features,
    log_monitoring_event,
)


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "preprocessing.log")


# ============================================================
# DATA EXTRACTION
# ============================================================

class DataExtractor:
    """Extrator de dados — arquivo Excel/CSV  ou  PostgreSQL (raw_student_data)."""

    def __init__(self):
        self.file_manager = FileManager()
        self.logger = setup_logger(self.__class__.__name__)

    def extract_from_file(
        self,
        file_path: str | Path,
        sheet_name: Optional[str] = None,
        save_to_db: bool = True,
    ) -> pd.DataFrame:
        """
        Extrai dados de arquivo e, opcionalmente, persiste na tabela raw_student_data.

        Args:
            file_path:   Caminho do arquivo Excel/CSV/Parquet
            sheet_name:  Aba da planilha (apenas Excel)
            save_to_db:  Se True, insere os dados brutos no PostgreSQL

        Returns:
            DataFrame com os dados extraídos
        """
        file_path = Path(file_path)
        if not file_path.exists():
            file_path = self.file_manager.get_data_path(file_path.name)

        self.logger.info(f"Extraindo dados de: {file_path}")

        if file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if isinstance(df, dict):
                df = next(iter(df.values()))
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Formato não suportado: {file_path.suffix}")

        self.logger.info(f"Dados extraídos. Shape: {df.shape}")

        if save_to_db:
            try:
                ensure_schema(settings.database_url)
                # renomear colunas para nomes internos antes de persistir
                alias = data_config.column_alias_map
                df_to_save = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
                n = insert_raw_dataframe(
                    df_to_save, settings.database_url, source_file=str(file_path)
                )
                self.logger.info(f"✅ {n} linhas brutas persistidas no PostgreSQL (raw_student_data).")
            except Exception as e:
                self.logger.warning(f"Não foi possível salvar dados brutos no PostgreSQL: {e}")

        return df

    def extract_from_db(self) -> pd.DataFrame:
        """
        Carrega dados brutos diretamente do PostgreSQL (tabela raw_student_data).
        Útil quando a planilha original não está disponível no ambiente de treino.
        """
        self.logger.info("Carregando dados brutos do PostgreSQL…")
        ensure_schema(settings.database_url)
        df = load_raw_dataframe(settings.database_url)
        self.logger.info(f"Dados carregados do PostgreSQL. Shape: {df.shape}")
        return df

    # Mantém assinatura legada
    def extract_data(
        self,
        file_path: str | Path,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.extract_from_file(file_path, sheet_name, save_to_db=True)


# ============================================================
# DATA TRANSFORMATION
# ============================================================

class DataTransformer:
    """Transformador de dados."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config

    def select_and_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        selected_columns = list(self.config.column_alias_map.keys())
        missing_cols = set(selected_columns) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes (serão ignoradas): {missing_cols}")
            selected_columns = [c for c in selected_columns if c in df.columns]
        df_selected = df[selected_columns].rename(columns=self.config.column_alias_map).copy()
        self.logger.info(f"Colunas selecionadas e renomeadas: {df_selected.shape}")
        return df_selected

    def consolidate_pedra(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pedra_cols = [c for c in ["pedra_2020", "pedra_2021", "pedra_2022"] if c in df.columns]
        if not pedra_cols:
            self.logger.warning("Nenhuma coluna de pedra encontrada")
            return df

        def row_mode(row):
            values = row.dropna()
            return values.mode().iloc[0] if not values.mode().empty else np.nan

        df["pedra_modal"] = df[pedra_cols].apply(row_mode, axis=1)
        df = df.drop(columns=pedra_cols)
        self.logger.info("Colunas de pedra consolidadas")
        return df

    def clean_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).str.strip().str.lower()
        self.logger.info("Strings limpas")
        return df

    def drop_invalid_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.config.columns_to_drop if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"Colunas removidas: {cols_to_drop}")
        return df

    def map_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target_col = self.config.target_column
        target_map = self.config.target_map
        if target_col not in df.columns:
            return df
        if not target_map:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            self.logger.info(f"Target '{target_col}' é numérico — mapeamento textual ignorado.")
            return df
        df[target_col] = df[target_col].map(target_map)
        self.logger.info("Target mapeado")
        return df

    def bucket_defasagem(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        target_col = self.config.target_column
        if target_col not in df.columns:
            self.logger.warning(f"Coluna '{target_col}' não encontrada para bucketing")
            return df
        raw = pd.to_numeric(df[target_col], errors="coerce")
        df[target_col] = np.where(raw < 0, 0, np.where(raw == 0, 1, 2)).astype("int64")
        counts = df[target_col].value_counts().sort_index()
        labels = {0: "atraso", 1: "neutro", 2: "avanço"}
        self.logger.info("Defasagem convertida em classes ternárias:")
        for cls, cnt in counts.items():
            self.logger.info(f"  Classe {cls} ({labels[cls]:>6}): {cnt:>4} amostras")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.info("Valores nulos por coluna:")
            self.logger.info(null_counts[null_counts > 0].to_string())
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        return df


# ============================================================
# DATA BALANCER
# ============================================================

class DataBalancer:
    """Balanceador de dados multiclasse por oversampling sintético."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config

    def balance_target(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(settings.random_state)
        target_col = self.config.target_column
        class_counts = df[target_col].value_counts()
        n_majority = int(class_counts.max())
        unique_classes = class_counts.index.tolist()
        self.logger.info(f"Distribuição original:\n{class_counts.to_string()}")
        if class_counts.nunique() == 1:
            self.logger.info("Dataset já está balanceado")
            return df

        balanced_parts = []
        for cls in unique_classes:
            df_cls = df[df[target_col] == cls]
            n_new = n_majority - len(df_cls)
            balanced_parts.append(df_cls)
            if n_new <= 0:
                continue
            numeric_cols = (
                df_cls.select_dtypes(include=["int64", "float64"])
                .columns.drop(target_col, errors="ignore")
            )
            categorical_cols = df_cls.select_dtypes(include="object").columns
            synthetic_rows = []
            for _ in range(n_new):
                row: dict = {}
                for col in numeric_cols:
                    col_min, col_max = float(df_cls[col].min()), float(df_cls[col].max())
                    row[col] = col_min if col_min == col_max else float(rng.uniform(col_min, col_max))
                for col in categorical_cols:
                    row[col] = rng.choice(df_cls[col].values)
                row[target_col] = cls
                synthetic_rows.append(row)
            if synthetic_rows:
                balanced_parts.append(pd.DataFrame(synthetic_rows))
            self.logger.info(f"Classe {cls}: {len(df_cls)} originais + {n_new} sintéticos = {n_majority}")

        df_balanced = pd.concat(balanced_parts, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=settings.random_state).reset_index(drop=True)
        self.logger.info(f"Dataset balanceado. Shape: {df_balanced.shape}")
        return df_balanced


# ============================================================
# PREPROCESSOR
# ============================================================

class Preprocessor:
    """Preprocessador de features (MinMaxScaler + OrdinalEncoder)."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config
        self.transformer = None

    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        X = df.drop(columns=[self.config.target_column], errors="ignore")
        numeric_features = [
            c for c in self.config.final_feature_order
            if c in X.columns and c not in self.config.categorical_columns
        ]
        categorical_features = [c for c in self.config.categorical_columns if c in X.columns]
        self.logger.info(f"Features numéricas: {len(numeric_features)}")
        self.logger.info(f"Features categóricas: {len(categorical_features)}")
        return ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), numeric_features),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
            ],
            remainder="drop",
        )

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
        X = df.drop(columns=[self.config.target_column], errors="ignore")
        y = df[self.config.target_column]
        self.transformer = self.build_preprocessor(df)
        X_transformed = self.transformer.fit_transform(X)
        feature_names = [name.split("__")[-1] for name in self.transformer.get_feature_names_out()]
        df_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        df_transformed[self.config.target_column] = y.values
        self.logger.info(f"Transformação concluída. Shape: {df_transformed.shape}")
        return df_transformed, self.transformer

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None:
            raise ValueError("Preprocessador não foi ajustado ainda")
        X = df.drop(columns=[self.config.target_column], errors='ignore')
        X_transformed = self.transformer.transform(X)
        df_transformed = pd.DataFrame(X_transformed, columns=self.config.final_feature_order)
        if self.config.target_column in df.columns:
            df_transformed[self.config.target_column] = df[self.config.target_column].values
        return df_transformed


# ============================================================
# PIPELINE
# ============================================================

class PreprocessingPipeline:
    """Pipeline completo de pré-processamento."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.balancer = DataBalancer()
        self.preprocessor = Preprocessor()

    def run_pipeline(
        self,
        file_path: Optional[str | Path] = None,
        sheet_name: Optional[str] = None,
        balance: bool = True,
        save_artifacts: bool = True,
        use_db_source: bool = False,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de pré-processamento.

        Args:
            file_path:      Caminho do arquivo (ignorado se use_db_source=True)
            sheet_name:     Aba da planilha
            balance:        Se True, aplica oversampling
            save_artifacts: Se True, salva parquet local + persiste no PostgreSQL
            use_db_source:  Se True, lê dados brutos do PostgreSQL em vez de arquivo

        Returns:
            Dict com dataset transformado e preprocessador
        """
        pipeline_run_id = str(uuid.uuid4())
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        self.logger.info(f"Run ID: {pipeline_run_id}")
        self.logger.info("=" * 60)

        # 1. Extrair dados
        if use_db_source:
            df = self.extractor.extract_from_db()
        else:
            df = self.extractor.extract_from_file(file_path, sheet_name, save_to_db=True)

        # 2. Transformar
        df = (
            df.pipe(self.transformer.select_and_rename)
            .pipe(self.transformer.consolidate_pedra)
            .pipe(self.transformer.clean_strings)
            .pipe(self.transformer.drop_invalid_columns)
            .pipe(self.transformer.map_target)
            .pipe(self.transformer.bucket_defasagem)
            .pipe(self.transformer.handle_missing_values)
        )

        # 3. Balancear
        if balance:
            df = self.balancer.balance_target(df)

        # 4. Preparar features (fit_transform)
        df_final, preprocessor = self.preprocessor.fit_transform(df)

        # 5. Salvar artefatos
        if save_artifacts:
            self._save_artifacts(df_final, preprocessor, pipeline_run_id)

        # 6. Estatísticas finais
        self._print_statistics(df_final)

        # Registro de monitoramento no PostgreSQL
        try:
            log_monitoring_event(
                event_type="preprocessing_completed",
                details={
                    "pipeline_run_id": pipeline_run_id,
                    "shape": list(df_final.shape),
                    "balance": balance,
                    "source": "db" if use_db_source else str(file_path),
                },
                database_url=settings.database_url,
            )
        except Exception as e:
            self.logger.warning(f"Não foi possível registrar evento no PostgreSQL: {e}")

        self.logger.info("=" * 60)
        self.logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
        self.logger.info("=" * 60)

        return {"dataset": df_final, "preprocessor": preprocessor}

    def _save_artifacts(
        self, dataset: pd.DataFrame, preprocessor: ColumnTransformer,
        pipeline_run_id: str = ""
    ) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Parquet local (fallback para treino offline)
        dataset_path = OUTPUT_DIR / "dataset_transformed_magic_steps.parquet"
        dataset.to_parquet(dataset_path, index=False)
        self.logger.info(f"Dataset salvo em: {dataset_path}")

        # Preprocessador serializado
        preprocessor_path = OUTPUT_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        self.logger.info(f"Preprocessador salvo em: {preprocessor_path}")

        # Persistir features transformadas no PostgreSQL
        try:
            ensure_schema(settings.database_url)
            n = insert_engineered_features(
                dataset, settings.database_url, pipeline_run_id=pipeline_run_id
            )
            self.logger.info(f"✅ {n} features engenheiradas persistidas no PostgreSQL (engineered_features).")
        except Exception as e:
            self.logger.warning(f"Não foi possível persistir features no PostgreSQL: {e}")

    def _print_statistics(self, dataset: pd.DataFrame) -> None:
        self.logger.info(f"\nShape do dataset: {dataset.shape}")
        self.logger.info(f"Colunas: {list(dataset.columns)}")
        target_col = data_config.target_column
        if target_col in dataset.columns:
            labels = data_config.defasagem_class_labels
            self.logger.info(f"\nDistribuição do target '{target_col}':")
            counts = dataset[target_col].value_counts().sort_index()
            for val, cnt in counts.items():
                pct = cnt / len(dataset)
                label = labels.get(int(val), str(val))
                self.logger.info(f"  Classe {val} ({label:>6}): {cnt:>4} amostras ({pct:.1%})")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATA_FILE = r"data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    pipeline = PreprocessingPipeline()
    results = pipeline.run_pipeline(DATA_FILE)
    print("\n✅ Pipeline finalizado com sucesso!")