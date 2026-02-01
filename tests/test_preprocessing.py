# test_preprocessing.py
# ============================================================
# Testes unitários e de integração para preprocessing.py
# ============================================================
#
# Execução:
#     pytest test_preprocessing.py -v
#
# Estrutura:
#   Fixtures         → dados sintéticos que espelhham a estrutura
#                       real (colunas originais da planilha).
#   TestExtractData  → extract_data() com .csv temporário e erro
#                       para formato não suportado.
#   TestSelectAndRename  → select_and_rename(): seleção + renomeação.
#   TestConsolidatePedra → consolidate_pedra(): modo por linha,
#                       NaN quando todas ausentes, mapeamento correto.
#   TestCleanStrings → clean_strings(): strip, lower, não toca
#                       colunas numéricas.
#   TestDropInvalidColumns → drop_invalid_columns(): remove as três
#                       colunas proibidas, ignora se já ausentes.
#   TestMapTarget    → map_target(): sim→1, não→0, valor
#                       inesperado→NaN.
#   TestBalanceTarget → balance_target(): classe já balanceada
#                       retorna sem alteração; imbalance gera
#                       sintéticos corretos.
#   TestBuildPreprocessor → build_preprocessor(): retorna
#                       ColumnTransformer com MinMaxScaler +
#                       OrdinalEncoder, fit_transform produz
#                       shape e range corretos.
#   TestPreprocessorJoblib → smoke test com o preprocessor.joblib
#                       real: transform de valores conhecidos
#                       produz vetor esperado.
# ============================================================

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# ── importa o módulo sob teste ────────────────────────────
import sys
# raiz do projecto + src/ no path.
# src/ é obrigatório porque preprocessing.py usa imports planos:
#   "from settings import …"  e  "from utils import …"
# logo settings.py e utils.py precisam ser visíveis directamente.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_SRC_DIR      = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SRC_DIR)

# =═══════════════════════════════════════════════════════════
# Instancias globais do teste
# ════════════════════════════════════════════════════════════

from src.settings import data_config
from src.preprocessing import PreprocessingPipeline

pp = PreprocessingPipeline()

# ════════════════════════════════════════════════════════════
# FIXTURES  — dados sintéticos com a estrutura real
# ════════════════════════════════════════════════════════════

def _raw_row(overrides: dict | None = None) -> dict:
    """Retorna uma linha com os nomes originais da planilha."""
    base = {
        "Fase": 2,
        "Turma": "A",
        "Ano nasc": 2010,
        "Idade 22": 12,
        "Gênero": "Menina",
        "Ano ingresso": 2021,
        "Pedra 20": "Ágata",
        "Pedra 21": "Topázio",
        "Pedra 22": "Topázio",
        "INDE 22": 7.5,
        "IAA": 8.5,
        "IEG": 8.0,
        "IPS": 7.0,
        "IDA": 6.5,
        "IPV": 7.8,
        "IAN": 5.0,
        "Matem": 7.0,
        "Portug": 6.5,
        "Inglês": 5.0,
        "Cg": 400,
        "Cf": 70,
        "Ct": 6,
        "Nº Av": 3,
        "Defas": -1,
        "Atingiu PV": "Sim",
    }
    if overrides:
        base.update(overrides)
    return base


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """DataFrame com 6 linhas e todas as colunas originais."""
    rows = [
        _raw_row({"Atingiu PV": "Sim",  "Turma": "A", "Pedra 22": "Topázio"}),
        _raw_row({"Atingiu PV": "Não",  "Turma": "B", "Pedra 22": "Ágata"}),
        _raw_row({"Atingiu PV": "Não",  "Turma": "C", "Pedra 22": "Quartzo"}),
        _raw_row({"Atingiu PV": "Não",  "Turma": "D", "Pedra 22": "Ametista"}),
        _raw_row({"Atingiu PV": "Sim",  "Turma": "E", "Pedra 22": "Topázio"}),
        _raw_row({"Atingiu PV": "Não",  "Turma": "F", "Pedra 22": "Quartzo"}),
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def renamed_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Já aplicou select_and_rename."""
    return pp.transformer.select_and_rename(raw_df)


@pytest.fixture()
def clean_df(renamed_df: pd.DataFrame) -> pd.DataFrame:
    """select → pedra → clean → drop → map (pipeline completo sem balance)."""
    return (
        renamed_df
        .pipe(pp.transformer.consolidate_pedra)
        .pipe(pp.transformer.clean_strings)
        .pipe(pp.transformer.drop_invalid_columns)
        .pipe(pp.transformer.map_target)
    )


# ════════════════════════════════════════════════════════════
# TestExtractData
# ════════════════════════════════════════════════════════════

class TestExtractData:
    """Testa extract_data() com ficheiros reais e erros."""

    def test_csv_leitura(self, raw_df: pd.DataFrame, tmp_path: Path):
        """Deve ler um .csv e retornar DataFrame com as mesmas linhas."""
        csv_path = tmp_path / "test.csv"
        raw_df.to_csv(csv_path, index=False)

        result = pp.extractor.extract_data(csv_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(raw_df)
        assert list(result.columns) == list(raw_df.columns)

    def test_xlsx_leitura(self, raw_df: pd.DataFrame, tmp_path: Path):
        """Deve ler um .xlsx e retornar DataFrame equivalente."""
        xlsx_path = tmp_path / "test.xlsx"
        raw_df.to_excel(xlsx_path, index=False)

        result = pp.extractor.extract_data(xlsx_path)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(raw_df)

    def test_formato_nao_suportado(self, tmp_path: Path):
        """Deve levantar ValueError para extensão não suportada."""
        bad_path = tmp_path / "test.json"
        bad_path.write_text("{}")

        with pytest.raises(ValueError, match="não suportado"):
            pp.extractor.extract_data(bad_path)


# ════════════════════════════════════════════════════════════
# TestSelectAndRename
# ════════════════════════════════════════════════════════════

class TestSelectAndRename:
    """Testa select_and_rename()."""

    def test_colunas_renomeadas(self, raw_df: pd.DataFrame):
        """Todas as colunas devem estar no COLUMN_ALIAS_MAP como valores."""
        result = pp.transformer.select_and_rename(raw_df)
        expected_cols = set(data_config.column_alias_map.values())

        assert set(result.columns) == expected_cols

    def test_nao_modifica_original(self, raw_df: pd.DataFrame):
        """Original não deve ser mutado (copy)."""
        original_cols = list(raw_df.columns)
        pp.transformer.select_and_rename(raw_df)

        assert list(raw_df.columns) == original_cols

    def test_tamanho_preservado(self, raw_df: pd.DataFrame):
        """Número de linhas deve ser mantido."""
        result = pp.transformer.select_and_rename(raw_df)

        assert len(result) == len(raw_df)

    def test_valores_preservados(self, raw_df: pd.DataFrame):
        """Valor de 'Fase' deve aparecer em 'fase' sem alteração."""
        result = pp.transformer.select_and_rename(raw_df)

        pd.testing.assert_series_equal(
            result["fase"].reset_index(drop=True),
            raw_df["Fase"].reset_index(drop=True),
            check_names=False,
        )


# ════════════════════════════════════════════════════════════
# TestConsolidatePedra
# ════════════════════════════════════════════════════════════

class TestConsolidatePedra:
    """Testa consolidate_pedra() — cálculo do modo por linha."""

    def test_modo_correto_consenso(self):
        """Quando duas pedras são iguais, o modo deve ser essa pedra."""
        df = pd.DataFrame({
            "pedra_2020": ["Ágata",    "Quartzo"],
            "pedra_2021": ["Topázio",  "Quartzo"],
            "pedra_2022": ["Topázio",  "Quartzo"],
        })
        result = pp.transformer.consolidate_pedra(df)

        assert result["pedra_modal"].iloc[0] == "Topázio"
        assert result["pedra_modal"].iloc[1] == "Quartzo"

    def test_todas_nan_retorna_nan(self):
        """Linha com todas as pedras NaN deve produzir pedra_modal NaN."""
        df = pd.DataFrame({
            "pedra_2020": [np.nan],
            "pedra_2021": [np.nan],
            "pedra_2022": [np.nan],
        })
        result = pp.transformer.consolidate_pedra(df)

        assert pd.isna(result["pedra_modal"].iloc[0])

    def test_uma_pedra_presente(self):
        """Com apenas uma pedra não-nula, o modo deve ser ela."""
        df = pd.DataFrame({
            "pedra_2020": [np.nan],
            "pedra_2021": [np.nan],
            "pedra_2022": ["Ametista"],
        })
        result = pp.transformer.consolidate_pedra(df)

        assert result["pedra_modal"].iloc[0] == "Ametista"

    def test_colunas_pedra_removidas(self):
        """As três colunas pedra_20xx não devem existir na saída."""
        df = pd.DataFrame({
            "pedra_2020": ["Ágata"],
            "pedra_2021": ["Ágata"],
            "pedra_2022": ["Ágata"],
            "outra_coluna": [1],
        })
        result = pp.transformer.consolidate_pedra(df)

        assert "pedra_2020" not in result.columns
        assert "pedra_2021" not in result.columns
        assert "pedra_2022" not in result.columns
        assert "pedra_modal" in result.columns
        assert "outra_coluna" in result.columns

    def test_nao_modifica_original(self):
        """DataFrame original não deve ser mutado."""
        df = pd.DataFrame({
            "pedra_2020": ["Ágata"],
            "pedra_2021": ["Topázio"],
            "pedra_2022": ["Topázio"],
        })
        cols_before = list(df.columns)
        pp.transformer.consolidate_pedra(df)

        assert list(df.columns) == cols_before


# ════════════════════════════════════════════════════════════
# TestCleanStrings
# ════════════════════════════════════════════════════════════

class TestCleanStrings:
    """Testa clean_strings() — strip + lower em colunas object."""

    def test_lower_e_strip(self):
        """Deve converter para minúsculo e remover espaços."""
        df = pd.DataFrame({
            "texto": ["  MENINA ", " Topázio  ", "  aMeTiStA"],
            "numero": [1, 2, 3],
        })
        result = pp.transformer.clean_strings(df)

        assert result["texto"].tolist() == ["menina", "topázio", "ametista"]

    def test_colunas_numericas_intactas(self):
        """Colunas int/float não devem ser alteradas."""
        df = pd.DataFrame({
            "texto": ["Sim"],
            "valor": [42],
            "real": [3.14],
        })
        result = pp.transformer.clean_strings(df)

        assert result["valor"].iloc[0] == 42
        assert result["real"].iloc[0] == 3.14

    def test_nao_modifica_original(self):
        df = pd.DataFrame({"texto": ["  OLA  "]})
        original_val = df["texto"].iloc[0]
        pp.transformer.clean_strings(df)

        assert df["texto"].iloc[0] == original_val


# ════════════════════════════════════════════════════════════
# TestDropInvalidColumns
# ════════════════════════════════════════════════════════════

class TestDropInvalidColumns:
    """Testa drop_invalid_columns()."""

    def test_remove_colunas_proibidas(self):
        """As três colunas de COLUMNS_TO_DROP devem desaparecer."""
        df = pd.DataFrame({
            "nota_ingles": [5],
            "nota_matematica": [7],
            "nota_portugues": [6],
            "fase": [2],
        })
        result = pp.transformer.drop_invalid_columns(df)

        for col in data_config.columns_to_drop:
            assert col not in result.columns
        assert "fase" in result.columns

    def test_sem_colunas_proibidas_nao_falha(self):
        """Se as colunas já não existem, não deve levantar erro."""
        df = pd.DataFrame({"fase": [2], "idade": [12]})
        result = pp.transformer.drop_invalid_columns(df)

        assert list(result.columns) == ["fase", "idade"]

    def test_remove_apenas_as_proibidas(self):
        """Colunas que não estão em COLUMNS_TO_DROP devem permanecer."""
        df = pd.DataFrame({
            "nota_ingles": [5],
            "score_inde": [7.5],
            "turma": ["a"],
        })
        result = pp.transformer.drop_invalid_columns(df)

        assert "score_inde" in result.columns
        assert "turma" in result.columns


# ════════════════════════════════════════════════════════════
# TestMapTarget
# ════════════════════════════════════════════════════════════

class TestMapTarget:
    """Testa map_target() — converte sim/não → 1/0."""

    def test_mapeamento_correto(self):
        """'sim'→1, 'não'→0 após clean_strings (já lowercase)."""
        df = pd.DataFrame({
            data_config.target_column: ["sim", "não", "sim", "não"],
        })
        result = pp.transformer.map_target(df)

        assert result[data_config.target_column].tolist() == [1, 0, 1, 0]

    def test_valor_inesperado_gera_nan(self):
        """Valor fora do TARGET_MAP deve resultar em NaN."""
        df = pd.DataFrame({
            data_config.target_column: ["sim", "talvez", "não"],
        })
        result = pp.transformer.map_target(df)

        assert result[data_config.target_column].iloc[0] == 1
        assert pd.isna(result[data_config.target_column].iloc[1])
        assert result[data_config.target_column].iloc[2] == 0

    def test_nao_modifica_original(self):
        df = pd.DataFrame({data_config.target_column: ["sim"]})
        pp.transformer.map_target(df)

        assert df[data_config.target_column].iloc[0] == "sim"


# ════════════════════════════════════════════════════════════
# TestBalanceTarget
# ════════════════════════════════════════════════════════════

class TestBalanceTarget:
    """Testa balance_target() — geração de sintéticos."""

    @pytest.fixture()
    def imbalanced_df(self) -> pd.DataFrame:
        """10 classe-0, 3 classe-1 → deve gerar 7 sintéticos."""
        rows = []
        # classe 0
        for i in range(10):
            rows.append({
                "fase": 2, "idade": 12, "nota_cg": 200 + i * 10,
                "turma": "a",
                data_config.target_column: 0,
            })
        # classe 1
        for i in range(3):
            rows.append({
                "fase": 3, "idade": 13, "nota_cg": 400 + i * 20,
                "turma": "b",
                data_config.target_column: 1,
            })
        return pd.DataFrame(rows)

    def test_balance_gera_igual(self, imbalanced_df: pd.DataFrame):
        """Após balanceamento, contagens das duas classes devem ser iguais."""
        result = pp.balancer.balance_target(imbalanced_df)
        counts = result[data_config.target_column].value_counts()

        assert counts[0] == counts[1]

    def test_balance_total_correto(self, imbalanced_df: pd.DataFrame):
        """Total deve ser 2 × len(classe_maior) = 20."""
        result = pp.balancer.balance_target(imbalanced_df)

        assert len(result) == 20  # 10 + 10

    def test_sintéticos_dentro_dos_limites(self, imbalanced_df: pd.DataFrame):
        """Valores sintéticos numéricos devem estar dentro do range da classe minority."""
        minor = imbalanced_df[imbalanced_df[data_config.target_column] == 1]
        result = pp.balancer.balance_target(imbalanced_df)
        synthetic = result[result[data_config.target_column] == 1].iloc[3:]  # os 7 novos

        for col in ["fase", "idade", "nota_cg"]:
            assert synthetic[col].min() >= minor[col].min()
            assert synthetic[col].max() <= minor[col].max()

    def test_já_balanceado_sem_alteração(self):
        """Se minority >= majority, retorna DataFrame original."""
        df = pd.DataFrame({
            "fase": [1, 2, 3, 4],
            data_config.target_column: [0, 0, 1, 1],
        })
        result = pp.balancer.balance_target(df)

        pd.testing.assert_frame_equal(result, df)

    def test_determinismo(self, imbalanced_df: pd.DataFrame):
        """Duas chamadas com mesmo RANDOM_STATE devem produzir o mesmo resultado."""
        r1 = pp.balancer.balance_target(imbalanced_df)
        r2 = pp.balancer.balance_target(imbalanced_df)

        pd.testing.assert_frame_equal(r1, r2)

    def test_categoricas_sintéticas_válidas(self, imbalanced_df: pd.DataFrame):
        """Categoricas sintéticas devem vir do universo da classe minority."""
        minor_cats = set(
            imbalanced_df.loc[imbalanced_df[data_config.target_column] == 1, "turma"].unique()
        )
        result = pp.balancer.balance_target(imbalanced_df)
        synthetic = result[result[data_config.target_column] == 1].iloc[3:]

        for val in synthetic["turma"]:
            assert val in minor_cats


# ════════════════════════════════════════════════════════════
# TestBuildPreprocessor
# ════════════════════════════════════════════════════════════

class TestBuildPreprocessor:
    """Testa build_preprocessor() e seu fit_transform."""

    @pytest.fixture()
    def preprocessor_and_data(self, clean_df: pd.DataFrame):
        """Retorna (preprocessor não ajustado, X, y)."""
        pre = pp.preprocessor.build_preprocessor(clean_df)
        X = clean_df.drop(columns=[data_config.target_column])
        y = clean_df[data_config.target_column]
        return pre, X, y

    def test_retorna_column_transformer(self, preprocessor_and_data):
        pre, _, _ = preprocessor_and_data

        assert isinstance(pre, ColumnTransformer)

    def test_dois_transformadores(self, preprocessor_and_data):
        """Deve ter exatamente 'num' e 'cat' (+ remainder=drop)."""
        pre, _, _ = preprocessor_and_data
        names = [name for name, _, _ in pre.transformers]

        assert "num" in names
        assert "cat" in names

    def test_num_é_minmaxscaler(self, preprocessor_and_data):
        pre, _, _ = preprocessor_and_data
        num_trans = pre.transformers[0][1]

        assert isinstance(num_trans, MinMaxScaler)

    def test_cat_é_ordinalencoder(self, preprocessor_and_data):
        pre, _, _ = preprocessor_and_data
        cat_trans = pre.transformers[1][1]

        assert isinstance(cat_trans, OrdinalEncoder)

    def test_remainder_drop(self, preprocessor_and_data):
        pre, _, _ = preprocessor_and_data

        assert pre.remainder == "drop"

    def test_fit_transform_shape(self, preprocessor_and_data):
        """Saída deve ter n_rows × 18 (15 num + 3 cat)."""
        pre, X, _ = preprocessor_and_data
        X_t = pre.fit_transform(X)

        assert X_t.shape == (len(X), 18)

    def test_fit_transform_range_numericas(self, preprocessor_and_data):
        """Após fit_transform, as 15 colunas numéricas devem estar em [0, 1]."""
        pre, X, _ = preprocessor_and_data
        X_t = pre.fit_transform(X)

        num_part = X_t[:, :15]
        assert num_part.min() >= 0.0
        assert num_part.max() <= 1.0

    def test_colunas_numericas_corretas(self, preprocessor_and_data):
        """Colunas numéricas devem ser exatamente as que não são categóricas."""
        pre, _, _ = preprocessor_and_data
        num_cols = pre.transformers[0][2]
        expected = [
            c for c in data_config.final_feature_order
            if c not in data_config.categorical_columns
        ]

        assert num_cols == expected

    def test_colunas_categoricas_corretas(self, preprocessor_and_data):
        pre, _, _ = preprocessor_and_data
        cat_cols = pre.transformers[1][2]

        assert cat_cols == data_config.categorical_columns


# ════════════════════════════════════════════════════════════
# TestPreprocessorJoblib  — smoke com o artefato real
# ════════════════════════════════════════════════════════════

JOBLIB_PATH = Path("out/preprocessor.joblib")


@pytest.mark.skipif(
    not JOBLIB_PATH.exists(),
    reason="preprocessor.joblib não disponível",
)
class TestPreprocessorJoblib:
    """Testa o preprocessor.joblib real com valores conhecidos."""

    @pytest.fixture(scope="class")
    def preprocessor(self):
        import joblib
        return joblib.load(JOBLIB_PATH)

    # ── estrutura ───────────────────────────────────────────
    def test_n_features_in(self, preprocessor):
        """Deve esperar 19 colunas de entrada (inclui ano_nascimento)."""
        assert preprocessor.n_features_in_ == 19

    def test_output_dim(self, preprocessor):
        """Saída deve ter 18 features (15 num + 3 cat)."""
        names = preprocessor.get_feature_names_out()
        assert len(names) == 18

    def test_remainder_drop(self, preprocessor):
        assert preprocessor.remainder == "drop"

    def test_ano_nascimento_é_dropped(self, preprocessor):
        """ano_nascimento deve estar na lista de remainder (dropped)."""
        dropped = preprocessor.transformers_[2][2]
        assert "ano_nascimento" in dropped

    # ── MinMaxScaler min/max ajustados ──────────────────────
    def test_min_fase_é_zero(self, preprocessor):
        scaler = preprocessor.transformers_[0][1]
        num_cols = preprocessor.transformers_[0][2]
        idx = num_cols.index("fase")

        assert scaler.data_min_[idx] == 0.0

    def test_max_nota_cg_é_862(self, preprocessor):
        scaler = preprocessor.transformers_[0][1]
        num_cols = preprocessor.transformers_[0][2]
        idx = num_cols.index("nota_cg")

        assert scaler.data_max_[idx] == 862.0

    def test_min_defasagem_é_neg5(self, preprocessor):
        scaler = preprocessor.transformers_[0][1]
        num_cols = preprocessor.transformers_[0][2]
        idx = num_cols.index("defasagem")

        assert scaler.data_min_[idx] == -5.0

    # ── OrdinalEncoder categorias ───────────────────────────
    def test_categorias_turma(self, preprocessor):
        encoder = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        idx = cat_cols.index("turma")
        cats = encoder.categories_[idx].tolist()

        # deve ter 24 letras, começar em 'a' e terminar em 'z'
        assert len(cats) == 24
        assert cats[0] == "a"
        assert cats[-1] == "z"
        # 'w' e 'x' não existem na base
        assert "w" not in cats
        assert "x" not in cats

    def test_categorias_genero(self, preprocessor):
        encoder = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        idx = cat_cols.index("genero")
        cats = encoder.categories_[idx].tolist()

        assert cats == ["menina", "menino"]

    def test_categorias_pedra_modal(self, preprocessor):
        encoder = preprocessor.transformers_[1][1]
        cat_cols = preprocessor.transformers_[1][2]
        idx = cat_cols.index("pedra_modal")
        cats = encoder.categories_[idx].tolist()

        assert cats == ["ametista", "quartzo", "topázio", "ágata"]

    # ── transform com valores conhecidos ────────────────────
    def _make_input(self, **overrides) -> pd.DataFrame:
        """Monta um DataFrame de 1 linha com valores padrão."""
        base = {
            "fase": 2, "turma": "a", "ano_nascimento": 2010,
            "idade": 12, "genero": "menina", "ano_ingresso": 2021,
            "score_inde": 7.5, "score_iaa": 8.5, "score_ieg": 8.0,
            "score_ips": 7.0, "score_ida": 6.5, "score_ipv": 7.8,
            "score_ian": 5.0, "nota_cg": 400, "nota_cf": 70,
            "nota_ct": 6, "num_avaliacoes": 3, "defasagem": -1,
            "pedra_modal": "ametista",
        }
        base.update(overrides)
        return pd.DataFrame([base])

    def test_transform_shape(self, preprocessor):
        """Output deve ser (1, 18)."""
        out = preprocessor.transform(self._make_input())

        assert out.shape == (1, 18)

    def test_transform_valores_min_produzem_zero(self, preprocessor):
        """Dar os valores mínimos do scaler deve produzir 0.0 para cada numérica."""
        inp = self._make_input(
            fase=0, idade=7, ano_ingresso=2016,
            score_inde=3.032, score_iaa=0.0, score_ieg=0.0,
            score_ips=2.5, score_ida=0.0, score_ipv=2.5,
            score_ian=2.5, nota_cg=1, nota_cf=1,
            nota_ct=1, num_avaliacoes=2, defasagem=-5,
        )
        out = preprocessor.transform(inp).flatten()

        # primeiras 15 posições = numéricas
        # nota: fase=0 → (0-0)/7 = 0, mas fase está na posição 0
        # verificamos que TODAS as numéricas são 0 exceto fase
        # que aqui é 0 também (min=0, valor=0)
        np.testing.assert_allclose(out[:15], 0.0, atol=1e-7)

    def test_transform_valores_max_produzem_um(self, preprocessor):
        """Dar os valores máximos deve produzir 1.0 para cada numérica."""
        inp = self._make_input(
            fase=7, idade=21, ano_ingresso=2022,
            score_inde=9.442, score_iaa=10.0, score_ieg=10.0,
            score_ips=10.0, score_ida=9.9, score_ipv=10.0,
            score_ian=10.0, nota_cg=862, nota_cf=192,
            nota_ct=18, num_avaliacoes=4, defasagem=2,
        )
        out = preprocessor.transform(inp).flatten()

        np.testing.assert_allclose(out[:15], 1.0, atol=1e-7)

    def test_transform_ordinal_menina_é_zero(self, preprocessor):
        """genero='menina' deve ser codificado como 0."""
        out = preprocessor.transform(
            self._make_input(genero="menina")
        ).flatten()
        # posição 16 = cat__genero
        assert out[16] == 0.0

    def test_transform_ordinal_menino_é_um(self, preprocessor):
        """genero='menino' deve ser codificado como 1."""
        out = preprocessor.transform(
            self._make_input(genero="menino")
        ).flatten()

        assert out[16] == 1.0

    def test_transform_ordinal_turma_b_é_um(self, preprocessor):
        """turma='b' é a segunda categoria → ordinal 1."""
        out = preprocessor.transform(
            self._make_input(turma="b")
        ).flatten()
        # posição 15 = cat__turma
        assert out[15] == 1.0

    def test_transform_ordinal_pedra_topazio_é_dois(self, preprocessor):
        """pedra_modal='topázio' é a terceira categoria → ordinal 2."""
        out = preprocessor.transform(
            self._make_input(**{"pedra_modal": "topázio"})
        ).flatten()
        # posição 17 = cat__pedra_modal
        assert out[17] == 2.0

    def test_transform_nota_cg_meio_range(self, preprocessor):
        """nota_cg=431.5 deve mapear para ~0.5 (meio do range 1–862)."""
        inp = self._make_input(nota_cg=431)
        out = preprocessor.transform(inp).flatten()
        num_cols = preprocessor.transformers_[0][2]
        idx = num_cols.index("nota_cg")

        expected = (431 - 1.0) / (862.0 - 1.0)
        np.testing.assert_allclose(out[idx], expected, atol=1e-6)

    def test_transform_defasagem_negativa(self, preprocessor):
        """defasagem=-3 deve mapear corretamente: (-3 - (-5)) / (2 - (-5)) = 2/7."""
        inp = self._make_input(defasagem=-3)
        out = preprocessor.transform(inp).flatten()
        num_cols = preprocessor.transformers_[0][2]
        idx = num_cols.index("defasagem")

        expected = (-3.0 - (-5.0)) / (2.0 - (-5.0))  # 2/7 ≈ 0.2857
        np.testing.assert_allclose(out[idx], expected, atol=1e-6)

    def test_transform_batch_shape(self, preprocessor):
        """Batch de 5 linhas deve produzir (5, 18)."""
        rows = pd.concat([self._make_input(nota_cg=100 * i) for i in range(1, 6)])
        out = preprocessor.transform(rows)

        assert out.shape == (5, 18)

    def test_transform_batch_consistente(self, preprocessor):
        """Cada linha do batch deve ser igual ao transform individual."""
        inputs = [self._make_input(nota_cg=100 * i) for i in range(1, 4)]
        batch = pd.concat(inputs, ignore_index=True)
        batch_out = preprocessor.transform(batch)

        for i, single in enumerate(inputs):
            single_out = preprocessor.transform(single).flatten()
            np.testing.assert_allclose(
                batch_out[i], single_out, atol=1e-7,
                err_msg=f"Linha {i} do batch diverge do transform individual",
            )


# ════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ════════════════════════════════════════════════════════════
#
#   python test_preprocessing.py          → rodan todos
#   python test_preprocessing.py -v       → verbose
#   python test_preprocessing.py -k pedra → filtrar por nome
#
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))