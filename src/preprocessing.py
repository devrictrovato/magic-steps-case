"""
Pipeline de pré-processamento de dados para o projeto Magic Steps.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, settings, OUTPUT_DIR
from utils import setup_logger, FileManager


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "preprocessing.log")


# ============================================================
# DATA EXTRACTION
# ============================================================

class DataExtractor:
    """Extrator de dados de diferentes fontes."""
    
    def __init__(self):
        self.file_manager = FileManager()
        self.logger = setup_logger(self.__class__.__name__)
    
    def extract_data(
        self,
        file_path: str | Path,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extrai dados de arquivo Excel ou CSV.
        
        Args:
            file_path: Caminho do arquivo
            sheet_name: Nome da planilha (para Excel)
        
        Returns:
            DataFrame com os dados
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
            raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")
        
        self.logger.info(f"Dados extraídos com sucesso. Shape: {df.shape}")
        return df


# ============================================================
# DATA TRANSFORMATION
# ============================================================

class DataTransformer:
    """Transformador de dados."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config
    
    def select_and_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Seleciona e renomeia colunas.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame transformado
        """
        selected_columns = list(self.config.column_alias_map.keys())
        
        # Verificar se todas as colunas existem
        missing_cols = set(selected_columns) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes: {missing_cols}")
            selected_columns = [c for c in selected_columns if c in df.columns]
        
        df_selected = df[selected_columns].rename(
            columns=self.config.column_alias_map
        ).copy()
        
        self.logger.info(f"Colunas selecionadas e renomeadas: {df_selected.shape}")
        return df_selected
    
    def consolidate_pedra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolida colunas de pedra em uma única coluna modal.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com coluna pedra_modal
        """
        df = df.copy()
        pedra_cols = ["pedra_2020", "pedra_2021", "pedra_2022"]
        
        # Verificar se as colunas existem
        pedra_cols = [c for c in pedra_cols if c in df.columns]
        
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
        """
        Limpa strings (strip e lowercase).
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com strings limpas
        """
        df = df.copy()
        
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).str.strip().str.lower()
        
        self.logger.info("Strings limpas")
        return df
    
    def drop_invalid_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove colunas inválidas.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame sem colunas inválidas
        """
        cols_to_drop = [
            c for c in self.config.columns_to_drop if c in df.columns
        ]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"Colunas removidas: {cols_to_drop}")
        
        return df
    
    def map_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mapeia variável target para valores numéricos (apenas quando target_map
        não está vazio). Para targets já numéricos (ex: defasagem) esta etapa
        é ignorada automaticamente.

        Args:
            df: DataFrame original

        Returns:
            DataFrame com target mapeado
        """
        df = df.copy()

        target_col = self.config.target_column
        target_map = self.config.target_map

        if target_col not in df.columns:
            return df

        if not target_map:
            # Target já numérico — garantir dtype numérico.
            # O bucketing ternário é feito em bucket_defasagem() logo a seguir.
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            self.logger.info(
                f"Target '{target_col}' é numérico — mapeamento textual ignorado."
            )
            return df

        df[target_col] = df[target_col].map(target_map)
        self.logger.info("Target mapeado")
        return df
    
    def bucket_defasagem(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a coluna numérica 'defasagem' em classes ternárias:
            0 → atraso  (defasagem < 0)
            1 → neutro  (defasagem == 0)
            2 → avanço  (defasagem > 0)

        A coluna original é substituída pelo bucket para ser usada como target
        pelo DataBalancer e pelo modelo.

        Args:
            df: DataFrame com coluna 'defasagem' numérica

        Returns:
            DataFrame com 'defasagem' convertida em 0 / 1 / 2
        """
        df = df.copy()
        target_col = self.config.target_column   # "defasagem"

        if target_col not in df.columns:
            self.logger.warning(f"Coluna '{target_col}' não encontrada para bucketing")
            return df

        raw = pd.to_numeric(df[target_col], errors="coerce")

        df[target_col] = np.where(
            raw < 0, 0,
            np.where(raw == 0, 1, 2)
        ).astype("int64")

        counts = df[target_col].value_counts().sort_index()
        labels = {0: "atraso", 1: "neutro", 2: "avanço"}
        self.logger.info("Defasagem convertida em classes ternárias:")
        for cls, cnt in counts.items():
            self.logger.info(f"  Classe {cls} ({labels[cls]:>6}): {cnt:>4} amostras")

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores ausentes.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame com valores ausentes tratados
        """
        df = df.copy()
        
        # Log de valores ausentes
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            self.logger.info("\nValores nulos por coluna:")
            self.logger.info(null_counts[null_counts > 0].to_string())
        
        # Preencher valores numéricos com mediana
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Preencher valores categóricos com moda
        categorical_cols = df.select_dtypes(include='object').columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        return df


# ============================================================
# DATA BALANCER
# ============================================================

class DataBalancer:
    """Balanceador de dados para classes desbalanceadas (suporta multiclasse)."""

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config

    def balance_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balanceia as classes do target usando oversampling sintético.

        Para classificação binária (0/1) o comportamento original é mantido.
        Para classificação multiclasse, todas as classes minoritárias são
        sobreamostradas até a contagem da classe majoritária.

        Args:
            df: DataFrame original

        Returns:
            DataFrame balanceado
        """
        rng = np.random.default_rng(settings.random_state)
        target_col = self.config.target_column

        class_counts = df[target_col].value_counts()
        n_majority = int(class_counts.max())
        unique_classes = class_counts.index.tolist()

        self.logger.info(f"Distribuição original das classes:\n{class_counts.to_string()}")
        self.logger.info(f"Alvo de balanceamento: {n_majority} amostras por classe")

        # Se todas as classes já têm a mesma contagem, não faz nada
        if class_counts.nunique() == 1:
            self.logger.info("Dataset já está balanceado")
            return df

        # Para classificação binária legada (0/1): mantém comportamento anterior
        # Para multiclasse: sobreamostra todas as classes até n_majority
        balanced_parts = []

        for cls in unique_classes:
            df_cls = df[df[target_col] == cls]
            n_cls = len(df_cls)
            n_new = n_majority - n_cls

            balanced_parts.append(df_cls)

            if n_new <= 0:
                continue  # já é a classe majoritária

            numeric_cols = (
                df_cls.select_dtypes(include=["int64", "float64"])
                .columns
                .drop(target_col, errors="ignore")
            )
            categorical_cols = df_cls.select_dtypes(include="object").columns

            synthetic_rows = []
            for _ in range(n_new):
                row: dict = {}

                for col in numeric_cols:
                    col_min, col_max = float(df_cls[col].min()), float(df_cls[col].max())
                    row[col] = (
                        col_min
                        if col_min == col_max
                        else float(rng.uniform(col_min, col_max))
                    )

                for col in categorical_cols:
                    row[col] = rng.choice(df_cls[col].values)

                row[target_col] = cls
                synthetic_rows.append(row)

            if synthetic_rows:
                balanced_parts.append(pd.DataFrame(synthetic_rows))

            self.logger.info(
                f"Classe {cls}: {n_cls} originais + {n_new} sintéticos = {n_majority}"
            )

        df_balanced = pd.concat(balanced_parts, ignore_index=True)
        df_balanced = df_balanced.sample(
            frac=1, random_state=settings.random_state
        ).reset_index(drop=True)

        self.logger.info(
            f"Dataset balanceado. Shape: {df_balanced.shape} | "
            f"Classes: {df_balanced[target_col].value_counts().to_dict()}"
        )
        return df_balanced


# ============================================================
# PREPROCESSOR
# ============================================================

class Preprocessor:
    """Preprocessador de features para treinamento."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = data_config
        self.transformer = None
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Constrói o preprocessador de features.
        
        Args:
            df: DataFrame com features
        
        Returns:
            ColumnTransformer configurado
        """
        X = df.drop(columns=[self.config.target_column], errors="ignore")

        numeric_features = [
            c
            for c in self.config.final_feature_order
            if c in X.columns and c not in self.config.categorical_columns
        ]
        
        categorical_features = [
            c for c in self.config.categorical_columns if c in X.columns
        ]
        
        self.logger.info(f"Features numéricas: {len(numeric_features)}")
        self.logger.info(f"Features categóricas: {len(categorical_features)}")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", MinMaxScaler(), numeric_features),
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )
        
        return preprocessor
    
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, ColumnTransformer]:
        """
        Ajusta e transforma o DataFrame.
        
        Args:
            df: DataFrame original
        
        Returns:
            Tupla (DataFrame transformado, preprocessador)
        """
        X = df.drop(columns=[self.config.target_column], errors="ignore")
        y = df[self.config.target_column]
        
        self.transformer = self.build_preprocessor(df)
        X_transformed = self.transformer.fit_transform(X)
        
        feature_names = self.transformer.get_feature_names_out()
        feature_names = [name.split("__")[-1] for name in feature_names]

        df_transformed = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )
        df_transformed[self.config.target_column] = y.values
        
        self.logger.info(f"Transformação concluída. Shape: {df_transformed.shape}")
        
        return df_transformed, self.transformer
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma o DataFrame usando preprocessador já ajustado.
        
        Args:
            df: DataFrame original
        
        Returns:
            DataFrame transformado
        """
        if self.transformer is None:
            raise ValueError("Preprocessador não foi ajustado ainda")
        
        X = df.drop(columns=[self.config.target_column], errors='ignore')
        X_transformed = self.transformer.transform(X)
        
        df_transformed = pd.DataFrame(
            X_transformed, columns=self.config.final_feature_order
        )
        
        if self.config.target_column in df.columns:
            df_transformed[self.config.target_column] = df[
                self.config.target_column
            ].values
        
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
        file_path: str | Path,
        sheet_name: Optional[str] = None,
        balance: bool = True,
        save_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo de pré-processamento.
        
        Args:
            file_path: Caminho do arquivo de dados
            sheet_name: Nome da planilha (para Excel)
            balance: Se True, balanceia as classes
            save_artifacts: Se True, salva artefatos
        
        Returns:
            Dicionário com dataset e preprocessador
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        self.logger.info("=" * 60)
        
        # 1. Extrair dados
        df = self.extractor.extract_data(file_path, sheet_name)
        
        # 2. Transformar dados
        df = (
            df.pipe(self.transformer.select_and_rename)
            .pipe(self.transformer.consolidate_pedra)
            .pipe(self.transformer.clean_strings)
            .pipe(self.transformer.drop_invalid_columns)
            .pipe(self.transformer.map_target)
            .pipe(self.transformer.bucket_defasagem)
            .pipe(self.transformer.handle_missing_values)
        )
        
        # 3. Balancear classes (opcional)
        if balance:
            df = self.balancer.balance_target(df)
        
        # 4. Preparar features
        df_final, preprocessor = self.preprocessor.fit_transform(df)
        
        # 5. Salvar artefatos (opcional)
        if save_artifacts:
            self._save_artifacts(df_final, preprocessor)
        
        # 6. Estatísticas finais
        self._print_statistics(df_final)
        
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
        self.logger.info("=" * 60)
        
        return {
            "dataset": df_final,
            "preprocessor": preprocessor,
        }
    
    def _save_artifacts(
        self, dataset: pd.DataFrame, preprocessor: ColumnTransformer
    ) -> None:
        """Salva artefatos do pipeline."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Salvar dataset
        dataset_path = OUTPUT_DIR / "dataset_transformed_magic_steps.parquet"
        dataset.to_parquet(dataset_path, index=False)
        self.logger.info(f"Dataset salvo em: {dataset_path}")
        
        # Salvar preprocessador
        preprocessor_path = OUTPUT_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        self.logger.info(f"Preprocessador salvo em: {preprocessor_path}")
    
    def _print_statistics(self, dataset: pd.DataFrame) -> None:
        """Imprime estatísticas do dataset."""
        self.logger.info(f"\nShape do dataset: {dataset.shape}")
        self.logger.info(f"Colunas: {list(dataset.columns)}")
        
        target_col = data_config.target_column
        if target_col in dataset.columns:
            labels = data_config.defasagem_class_labels
            self.logger.info(f"\nDistribuição do target '{target_col}' (após balanceamento):")
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