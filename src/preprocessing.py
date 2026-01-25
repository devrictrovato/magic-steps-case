# preprocessing.py

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# ============================================================
# CONFIGURAÇÕES
# ============================================================

TARGET_COLUMN = "flag_atingiu_pv"

TARGET_MAP = {
    "sim": 1,
    "não": 0,
}

COLUMNS_TO_DROP = [
    "nota_ingles",
    "nota_matematica",
    "nota_portugues",
]

CATEGORICAL_COLUMNS = [
    "turma",
    "genero",
    "pedra_modal",
]

COLUMN_ALIAS_MAP = {
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
}

SELECTED_COLUMNS = list(COLUMN_ALIAS_MAP.keys())

FINAL_FEATURE_ORDER = [
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
]

ARTIFACTS_DIR = Path("out")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ============================================================
# EXTRACT
# ============================================================

def extract_data(file_path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    file_path = Path(file_path)

    if file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if isinstance(df, dict):
            df = next(iter(df.values()))
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado")

    return df


# ============================================================
# TRANSFORM
# ============================================================

def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    return df[SELECTED_COLUMNS].rename(columns=COLUMN_ALIAS_MAP).copy()


def consolidate_pedra(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pedra_cols = ["pedra_2020", "pedra_2021", "pedra_2022"]

    def row_mode(row):
        values = row.dropna()
        return values.mode().iloc[0] if not values.mode().empty else np.nan

    df["pedra_modal"] = df[pedra_cols].apply(row_mode, axis=1)
    return df.drop(columns=pedra_cols)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def drop_invalid_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)
    return df


# ============================================================
# BALANCEAMENTO (GERAÇÃO DE SIM)
# ============================================================

def balance_target(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    df_major = df[df[TARGET_COLUMN] == 0]
    df_minor = df[df[TARGET_COLUMN] == 1]

    if len(df_minor) >= len(df_major):
        return df

    n_new = len(df_major) - len(df_minor)

    numeric_cols = df_minor.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET_COLUMN)
    categorical_cols = df_minor.select_dtypes(include="object").columns

    synthetic_rows = []

    for _ in range(n_new):
        row = {}

        for col in numeric_cols:
            col_min, col_max = df_minor[col].min(), df_minor[col].max()
            row[col] = col_min if col_min == col_max else rng.uniform(col_min, col_max)

        for col in categorical_cols:
            row[col] = rng.choice(df_minor[col].values)

        row[TARGET_COLUMN] = 1
        synthetic_rows.append(row)

    df_syn = pd.DataFrame(synthetic_rows)
    df_balanced = pd.concat([df_major, df_minor, df_syn], ignore_index=True)

    return df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    X = df.drop(columns=[TARGET_COLUMN])

    numeric_features = [c for c in FINAL_FEATURE_ORDER if c in X.columns and c not in CATEGORICAL_COLUMNS]
    categorical_features = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    return ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numeric_features),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(file_path: str | Path, sheet_name: str | None = None) -> Dict[str, Any]:

    df = (
        extract_data(file_path, sheet_name)
        .pipe(select_and_rename)
        .pipe(consolidate_pedra)
        .pipe(clean_strings)
        .pipe(drop_invalid_columns)
        .pipe(map_target)
    )

    print("\n🔍 Valores nulos por coluna:")
    print(df.isnull().sum().sort_values(ascending=False))

    df = balance_target(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    preprocessor = build_preprocessor(df)
    X_t = preprocessor.fit_transform(X)

    dataset_final = pd.DataFrame(X_t, columns=FINAL_FEATURE_ORDER)
    dataset_final[TARGET_COLUMN] = y.values

    dataset_final.to_parquet(ARTIFACTS_DIR / "dataset_transformed_magic_steps.parquet", index=False)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "normalizador.joblib")

    return {"dataset": dataset_final, "preprocessor": preprocessor}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATA_FILE = r"data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

    print("🚀 Executando pipeline de preprocessing (produção)...")

    results = run_pipeline(DATA_FILE)
    dataset = results["dataset"]

    print("\n✅ Pipeline finalizado com sucesso!")
    print("📦 Shape:", dataset.shape)
    print("\n🎯 Distribuição do target:")
    print(dataset[TARGET_COLUMN].value_counts(normalize=True))
