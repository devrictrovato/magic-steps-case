# test_model.py
# ============================================================
# Testes unitários e de integração para:
#   • MagicStepsNet        (arquitetura, forward, checkpoint)
#   • MagicStepsDataset    (Dataset PyTorch)
#   • Routes da API        (health, info, predict, batch,
#                            features, thresholds)
# ============================================================
#
# Execução:
#     pytest test_model.py -v
#
# Estrutura:
#   Fixtures globais   → modelo sintetico, preprocessor real,
#                         contexto injetado no _MODEL_CONTEXT.
#   TestMagicStepsNet  → construção, forward shape, squeeze,
#                         eval vs train, checkpoint save/load.
#   TestMagicStepsDataset → len, getitem, dtype, batch via
#                         DataLoader.
#   TestHealthEndpoint → GET /health com e sem artefatos.
#   TestInfoEndpoint   → GET /info campos obrigatórios.
#   TestFeaturesEndpoint → GET /features 18 items, tipos,
#                         ranges extraídos do preprocessor.
#   TestThresholdEndpoints → GET/PUT /thresholds.
#   TestPredictEndpoint   → POST /predict validação de
#                         entrada, resposta, confidência,
#                         503 sem modelo.
#   TestPredictBatchEndpoint → POST /predict/batch lote,
#                         limites, consistência com individual.
# ============================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

# ensure src directory is on path so `import settings` resolves during tests
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── garantir que raiz + app/ estão no path ─────────────────
# app/ é obrigatório porque main.py / routes.py / context.py
# usam imports planos: "from context import …".
# Sem app/ no path, "context" nunca é resolvido.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_APP_DIR      = str(Path(__file__).resolve().parent.parent / "app")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _APP_DIR)

# ── importações do projeto ────────────────────────────────
# MagicStepsNet & Dataset vêm do train.py;
# usamos a versão do main.py (idêntica) para evitar import
# do feature_engineering no topo do train.
# Definimos aqui uma cópia fiel para o teste ser autónomo.


class MagicStepsNet(nn.Module):
    """Cópia exata da arquitetura usada em train.py / main.py."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout: float,
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class MagicStepsDataset(torch.utils.data.Dataset):
    """Cópia exata do Dataset usado em train.py."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── caminho do preprocessor real ──────────────────────────
# caminho absoluto derivado de __file__ — independente do CWD
JOBLIB_PATH = Path(__file__).resolve().parent.parent / "out" / "preprocessor.joblib"

INPUT_DIM = 18  # 15 num + 3 cat

# ── example body reutilizável nos testes da API ──────────
_VALID_FEATURES = {
    "fase": 2,
    "idade": 12,
    "ano_ingresso": 2021,
    "score_inde": 7.5,
    "score_iaa": 8.5,
    "score_ieg": 8.0,
    "score_ips": 7.0,
    "score_ida": 6.5,
    "score_ipv": 7.8,
    "score_ian": 5.0,
    "nota_cg": 400,
    "nota_cf": 70,
    "nota_ct": 6,
    "num_avaliacoes": 3,
    "defasagem": -1,
    "turma": "a",
    "genero": "menina",
    "pedra_modal": "ametista",
}


def _student_body(student_id: str | None = "RA-1", features: dict | None = None) -> dict:
    return {
        "student_id": student_id,
        "features": features or _VALID_FEATURES,
    }


# ════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════


@pytest.fixture()
def auth_client(client):
    """Retorna TestClient com usuário criado e token válido."""
    # register a user
    resp = client.post("/register", json={"username":"alice","password":"secret"})
    assert resp.status_code == 201
    # get token
    resp = client.post("/token", data={"username":"alice","password":"secret"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client

@pytest.fixture()
def simple_params() -> dict:
    """Hiperparâmetros mínimos para construir um modelo rápido."""
    return {
        "hidden_layers": [32, 16],
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 2,
    }


@pytest.fixture()
def model(simple_params) -> MagicStepsNet:
    """Modelo não treinado com input_dim=18."""
    return MagicStepsNet(
        input_dim=INPUT_DIM,
        hidden_layers=simple_params["hidden_layers"],
        dropout=simple_params["dropout"],
    )


@pytest.fixture()
def preprocessor():
    """Preprocessor real do joblib (skippa se indisponível)."""
    if not JOBLIB_PATH.exists():
        pytest.skip("preprocessor.joblib não disponível")
    import joblib
    return joblib.load(JOBLIB_PATH)


@pytest.fixture()
def fake_context(model, preprocessor, simple_params) -> dict:
    """Contexto completo que simula o _MODEL_CONTEXT após startup."""
    model.eval()
    return {
        "model": model,
        "device": torch.device("cpu"),
        "input_dim": INPUT_DIM,
        "best_params": simple_params,
        "preprocessor": preprocessor,
    }


@pytest.fixture()
def client(fake_context):
    """
    TestClient da FastAPI com contexto injetado.
    Substitui get_model_context pelo fake_context.
    Retorna também um atributo `logger` na instância para inspeção.
    """
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock

    # importa app DEPOIS de preparar o mock para evitar
    # que o startup tente carregar arquivos reais
    with patch("context.get_model_context", return_value=fake_context), \
         patch("routes.MongoLogger") as fake_logger_class:
        # make sure logger instance methods exist but do nothing
        fake_logger = fake_logger_class.return_value
        fake_logger.log_prediction = MagicMock()

        # routes importa get_model_context de context;
        # precisamos also do patch no módulo routes
        with patch("routes.get_model_context", return_value=fake_context):
            from app.main import app
            client = TestClient(app, raise_server_exceptions=True)
            # attach fake_logger to client so tests can inspect call count
            client.mongo_logger = fake_logger
            yield client


# ════════════════════════════════════════════════════════════
# TestMagicStepsNet  — arquitetura e forward
# ════════════════════════════════════════════════════════════

class TestMagicStepsNet:
    """Testa a classe MagicStepsNet isoladamente."""

    def test_instancia(self, model):
        assert isinstance(model, nn.Module)

    # ── forward shape ─────────────────────────────────────
    def test_forward_single(self, model):
        """Input (1, 18) → output (1,) após squeeze."""
        x = torch.randn(1, INPUT_DIM)
        model.eval()
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1,)

    def test_forward_batch(self, model):
        """Input (64, 18) → output (64,)."""
        x = torch.randn(64, INPUT_DIM)
        model.eval()
        with torch.no_grad():
            out = model(x)

        assert out.shape == (64,)

    def test_forward_dtype_float32(self, model):
        """Saída deve ser float32."""
        x = torch.randn(4, INPUT_DIM)
        model.eval()
        with torch.no_grad():
            out = model(x)

        assert out.dtype == torch.float32

    # ── sigmoid para probabilidade ────────────────────────
    def test_sigmoid_range(self, model):
        """sigmoid(logit) deve estar em (0, 1) para entradas aleatórias."""
        x = torch.randn(128, INPUT_DIM)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(x))

        assert probs.min().item() > 0.0
        assert probs.max().item() < 1.0

    # ── determinismo em eval ──────────────────────────────
    def test_eval_determinismo(self, model):
        """Duas passagens em eval com mesmo input devem dar mesmo output."""
        model.eval()
        x = torch.randn(8, INPUT_DIM)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)

        torch.testing.assert_close(o1, o2)

    # ── dropout ativo em train ────────────────────────────
    def test_train_dropout_variabilidade(self):
        """Em modo train com dropout>0, outputs devem (muito provavelmente) divergir."""
        m = MagicStepsNet(INPUT_DIM, [64, 32], dropout=0.5)
        m.train()
        x = torch.randn(32, INPUT_DIM)

        torch.manual_seed(0)
        o1 = m(x)
        torch.manual_seed(1)
        o2 = m(x)

        # com dropout 0.5 e seed diferente é quase impossível serem iguais
        assert not torch.allclose(o1, o2)

    # ── arquitetura interna ───────────────────────────────
    def test_ultima_camada_linear(self, model):
        """A última camada deve ser Linear com out_features=1."""
        last = list(model.net.children())[-1]

        assert isinstance(last, nn.Linear)
        assert last.out_features == 1

    def test_hidden_layers_contagem(self):
        """Para hidden_layers=[128, 64], deve haver 2 blocos (Linear+ReLU+BN+Drop)
        mais a camada final → 9 módulos no Sequential."""
        m = MagicStepsNet(INPUT_DIM, [128, 64], dropout=0.2)
        # 2 blocos × 4 + 1 final = 9
        assert len(list(m.net.children())) == 9

    def test_hidden_layers_single(self):
        """hidden_layers=[64] → 1 bloco × 4 + 1 final = 5 módulos."""
        m = MagicStepsNet(INPUT_DIM, [64], dropout=0.1)

        assert len(list(m.net.children())) == 5

    def test_hidden_layers_tres(self):
        """hidden_layers=[256, 128, 64] → 3 × 4 + 1 = 13 módulos."""
        m = MagicStepsNet(INPUT_DIM, [256, 128, 64], dropout=0.3)

        assert len(list(m.net.children())) == 13

    # ── checkpoint save / load ────────────────────────────
    def test_checkpoint_round_trip(self, model, simple_params, tmp_path):
        """Salvar e recarregar checkpoint deve produzir modelo equivalente."""
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            logits_before = model(x)

        # salvar no formato do train.py
        ckpt_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": INPUT_DIM,
                "best_params": simple_params,
            },
            ckpt_path,
        )

        # recarregar
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2 = MagicStepsNet(
            input_dim=ckpt["input_dim"],
            hidden_layers=ckpt["best_params"]["hidden_layers"],
            dropout=ckpt["best_params"]["dropout"],
        )
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()

        with torch.no_grad():
            logits_after = model2(x)

        torch.testing.assert_close(logits_before, logits_after)

    def test_checkpoint_campos_obrigatórios(self, model, simple_params, tmp_path):
        """Checkpoint deve conter as três chaves esperadas."""
        ckpt_path = tmp_path / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": INPUT_DIM,
                "best_params": simple_params,
            },
            ckpt_path,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")

        assert "model_state_dict" in ckpt
        assert "input_dim" in ckpt
        assert "best_params" in ckpt
        assert ckpt["input_dim"] == INPUT_DIM


# ════════════════════════════════════════════════════════════
# TestMagicStepsDataset
# ════════════════════════════════════════════════════════════

class TestMagicStepsDataset:
    """Testa o Dataset PyTorch."""

    @pytest.fixture()
    def dataset(self) -> MagicStepsDataset:
        X = np.random.rand(50, INPUT_DIM).astype("float32")
        y = np.random.randint(0, 2, size=50).astype("float32")
        return MagicStepsDataset(X, y)

    def test_len(self, dataset):
        assert len(dataset) == 50

    def test_getitem_retorna_tupla(self, dataset):
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_getitem_shapes(self, dataset):
        x, y = dataset[0]

        assert x.shape == (INPUT_DIM,)
        assert y.shape == ()  # escalar

    def test_getitem_dtype(self, dataset):
        x, y = dataset[0]

        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_valores_preservados(self):
        """Valores originais devem aparecer no tensor."""
        X = np.array([[1.0] * INPUT_DIM], dtype="float32")
        y = np.array([1.0], dtype="float32")
        ds = MagicStepsDataset(X, y)
        x_t, y_t = ds[0]

        assert x_t.sum().item() == INPUT_DIM
        assert y_t.item() == 1.0

    def test_dataloader_batch(self, dataset):
        """DataLoader deve produzir batches com shape correto."""
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        batches = list(loader)

        # 50 amostras / 16 = 3 batches completos + 1 com 2
        assert len(batches) == 4

        x_batch, y_batch = batches[0]
        assert x_batch.shape == (16, INPUT_DIM)
        assert y_batch.shape == (16,)

    def test_dataloader_ultimo_batch(self, dataset):
        """Último batch pode ter menos que batch_size."""
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        x_last, y_last = list(loader)[-1]

        assert x_last.shape[0] == 2  # 50 - 3×16 = 2
        assert y_last.shape[0] == 2


# ════════════════════════════════════════════════════════════
# TestHealthEndpoint
# ════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """GET /health"""

    def test_health_200(self, client):
        resp = client.get("/health")

        assert resp.status_code == 200

    def test_health_campos(self, client):
        body = client.get("/health").json()

        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["preprocessor_loaded"] is True
        assert "device" in body
        assert "timestamp" in body

    def test_health_sem_modelo(self, fake_context):
        """Sem modelo, model_loaded deve ser False mas status ainda 'ok'."""
        from fastapi.testclient import TestClient

        ctx_no_model = {**fake_context, "model": None}
        with patch("context.get_model_context", return_value=ctx_no_model), \
             patch("routes.get_model_context", return_value=ctx_no_model):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=True)
            body = c.get("/health").json()

        assert body["model_loaded"] is False
        assert body["status"] == "ok"

    def test_health_sem_preprocessor(self, fake_context):
        """Sem preprocessor, preprocessor_loaded deve ser False."""
        from fastapi.testclient import TestClient

        ctx_no_pre = {**fake_context, "preprocessor": None}
        with patch("context.get_model_context", return_value=ctx_no_pre), \
             patch("routes.get_model_context", return_value=ctx_no_pre):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=True)
            body = c.get("/health").json()

        assert body["preprocessor_loaded"] is False


# ════════════════════════════════════════════════════════════
# TestInfoEndpoint
# ════════════════════════════════════════════════════════════

class TestInfoEndpoint:
    """GET /info"""

    def test_info_200(self, client, auth_client):
        # public access should be forbidden
        resp = client.get("/info")
        assert resp.status_code == 401

        # authorized client works
        resp = auth_client.get("/info")
        assert resp.status_code == 200

    def test_info_campos_obrigatórios(self, client, auth_client):
        body = auth_client.get("/info").json()

        assert body["project"] == "Magic Steps"
        assert body["model_class"] == "MagicStepsNet"
        assert body["input_dim"] == INPUT_DIM
        assert body["preprocessor_loaded"] is True
        assert body["n_features"] == 18
        assert "best_params" in body
        assert "features" in body
        assert "threshold" in body

    def test_info_features_tem_18_items(self, client, auth_client):
        body = auth_client.get("/info").json()

        assert len(body["features"]) == 18

    def test_info_best_params_fields(self, client, auth_client):
        params = auth_client.get("/info").json()["best_params"]

        assert "hidden_layers" in params
        assert "dropout" in params


# ════════════════════════════════════════════════════════════
# TestFeaturesEndpoint
# ════════════════════════════════════════════════════════════

class TestFeaturesEndpoint:
    """GET /features"""

    def test_features_200(self, client, auth_client):
        assert client.get("/features").status_code == 401
        resp = auth_client.get("/features")
        assert resp.status_code == 200

    def test_features_quantidade(self, client, auth_client):
        body = auth_client.get("/features").json()

        assert len(body) == 18

    def test_features_campos_presentes(self, client, auth_client):
        """Cada item deve ter name, description, type e range."""
        for feat in auth_client.get("/features").json():
            assert "name" in feat
            assert "description" in feat
            assert "type" in feat
            assert "range" in feat

    def test_features_tipos_válidos(self, client, auth_client):
        tipos = {f["type"] for f in auth_client.get("/features").json()}

        assert tipos == {"int", "float", "categorical"}

    def test_features_numericas_têm_range(self, client, auth_client):
        """Todas as numéricas devem ter range não-None."""
        for feat in auth_client.get("/features").json():
            if feat["type"] in ("int", "float"):
                assert feat["range"] is not None, f"{feat['name']} sem range"

    def test_features_categoricas_têm_range(self, client, auth_client):
        """Categoricas devem ter range com ' | ' como separador."""
        for feat in auth_client.get("/features").json():
            if feat["type"] == "categorical":
                assert feat["range"] is not None
                assert " | " in feat["range"]

    def test_features_nomes_esperados(self, client, auth_client):
        nomes = [f["name"] for f in auth_client.get("/features").json()]
        expected = [
            "fase", "idade", "ano_ingresso",
            "score_inde", "score_iaa", "score_ieg", "score_ips",
            "score_ida", "score_ipv", "score_ian",
            "nota_cg", "nota_cf", "nota_ct",
            "num_avaliacoes", "defasagem",
            "turma", "genero", "pedra_modal",
        ]

        assert nomes == expected

    def test_features_turma_range_contem_24(self, client, auth_client):
        """turma deve listar 24 letras."""
        turma = next(
            f for f in auth_client.get("/features").json() if f["name"] == "turma"
        )
        letras = turma["range"].split(" | ")

        assert len(letras) == 24

    def test_features_genero_range(self, client, auth_client):
        genero = next(
            f for f in auth_client.get("/features").json() if f["name"] == "genero"
        )

        assert "menina" in genero["range"]
        assert "menino" in genero["range"]

    def test_features_pedra_range(self, client, auth_client):
        pedra = next(
            f for f in auth_client.get("/features").json() if f["name"] == "pedra_modal"
        )

        for p in ["ametista", "quartzo", "topázio", "ágata"]:
            assert p in pedra["range"]


# ════════════════════════════════════════════════════════════
# TestThresholdEndpoints
# ════════════════════════════════════════════════════════════

class TestThresholdEndpoints:
    """GET / PUT /thresholds"""

    def test_get_threshold_200(self, client, auth_client):
        assert client.get("/thresholds").status_code == 401
        resp = auth_client.get("/thresholds")

        assert resp.status_code == 200

    def test_get_threshold_campos(self, client, auth_client):
        body = auth_client.get("/thresholds").json()

        assert "threshold" in body
        assert "updated_at" in body
        assert 0.0 <= body["threshold"] <= 1.0

    def test_put_threshold_atualiza(self, client, auth_client):
        auth_client.put("/thresholds", json={"threshold": 0.7})
        body = auth_client.get("/thresholds").json()

        assert body["threshold"] == 0.7

    def test_put_threshold_resposta(self, client, auth_client):
        resp = auth_client.put("/thresholds", json={"threshold": 0.3})

        assert resp.status_code == 200
        assert resp.json()["threshold"] == 0.3

    def test_put_threshold_fora_range(self, client):
        """Valor > 1.0 deve ser rejeitado com 422."""
        resp = client.put("/thresholds", json={"threshold": 1.5})

        assert resp.status_code == 422

    def test_put_threshold_negativo(self, client):
        resp = client.put("/thresholds", json={"threshold": -0.1})

        assert resp.status_code == 422

    def test_put_threshold_restaura_default(self, client):
        """Restaurar para 0.5 deve funcionar."""
        client.put("/thresholds", json={"threshold": 0.9})
        client.put("/thresholds", json={"threshold": 0.5})
        body = client.get("/thresholds").json()

        assert body["threshold"] == 0.5


# ════════════════════════════════════════════════════════════
# TestPredictEndpoint
# ════════════════════════════════════════════════════════════

class TestPredictEndpoint:
    """POST /predict"""

    def test_predict_200(self, client, auth_client):
        assert client.post("/predict", json=_student_body()).status_code == 401
        resp = auth_client.post("/predict", json=_student_body())

        assert resp.status_code == 200
        # confirm logging
        assert auth_client.mongo_logger.log_prediction.call_count == 1

    def test_monitor_logs(self, client, auth_client):
        # after a prediction we should see at least one log entry
        auth_client.post("/predict", json=_student_body())
        resp = auth_client.get("/monitor/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_monitor_features(self, client, auth_client):
        # Redis successfully returns no keys
        with patch("routes.redis") as fake_redis:
            fake_conn = fake_redis.Redis.return_value
            fake_conn.keys.return_value = []
            resp = auth_client.get("/monitor/features")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_monitor_features_redis_down(self, client, auth_client):
        # Simulate connection error: should still return 200 with empty list
        from redis.exceptions import ConnectionError
        with patch("routes.redis") as fake_redis:
            fake_redis.Redis.side_effect = ConnectionError("getaddrinfo failed")
            resp = auth_client.get("/monitor/features")
            assert resp.status_code == 200
            assert resp.json() == []

    # ── campos da resposta ────────────────────────────────
    def test_predict_campos_obrigatórios(self, client, auth_client):
        body = auth_client.post("/predict", json=_student_body()).json()

        assert "student_id" in body
        assert "probability" in body
        assert "prediction" in body
        assert "confidence" in body
        assert "prediction_id" in body
        assert "timestamp" in body

    def test_predict_probability_range(self, client, auth_client):
        """Probabilidade deve estar em [0, 1]."""
        body = auth_client.post("/predict", json=_student_body()).json()

        assert 0.0 <= body["probability"] <= 1.0

    def test_predict_prediction_binário(self, client, auth_client):
        body = auth_client.post("/predict", json=_student_body()).json()

        assert body["prediction"] in (0, 1)

    def test_predict_confidence_válido(self, client, auth_client):
        body = auth_client.post("/predict", json=_student_body()).json()

        assert body["confidence"] in ("alta", "média", "baixa")

    def test_predict_student_id_preservado(self, client, auth_client):
        body = auth_client.post(
            "/predict", json=_student_body(student_id="TESTE-99")
        ).json()

        assert body["student_id"] == "TESTE-99"

    def test_predict_student_id_nulo(self, client, auth_client):
        """student_id é opcional — None deve ser aceito."""
        body = auth_client.post(
            "/predict", json=_student_body(student_id=None)
        ).json()

        assert body["student_id"] is None

    def test_predict_prediction_id_é_uuid(self, client, auth_client):
        """prediction_id deve ser um UUID válido."""
        import uuid
        body = auth_client.post("/predict", json=_student_body()).json()

        # não deve levantar
        uuid.UUID(body["prediction_id"])

    def test_predict_timestamp_é_iso(self, client, auth_client):
        """timestamp deve ser parseable como ISO-8601."""
        from datetime import datetime
        body = auth_client.post("/predict", json=_student_body()).json()

        datetime.fromisoformat(body["timestamp"])

    # ── determinismo ──────────────────────────────────────
    def test_predict_determinismo(self, client, auth_client):
        """Mesmo input deve dar mesma probabilidade (modelo em eval)."""
        p1 = auth_client.post("/predict", json=_student_body()).json()["probability"]
        p2 = auth_client.post("/predict", json=_student_body()).json()["probability"]

        assert p1 == p2

    # ── validação de entrada ──────────────────────────────
    def test_predict_fase_fora_range(self, client, auth_client):
        """fase=99 deve ser rejeitado com 422."""
        feats = {**_VALID_FEATURES, "fase": 99}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_fase_negativa(self, client, auth_client):
        feats = {**_VALID_FEATURES, "fase": -1}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_idade_abaixo_min(self, client, auth_client):
        feats = {**_VALID_FEATURES, "idade": 5}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_nota_cg_zero(self, client, auth_client):
        """nota_cg ge=1, então 0 deve ser rejeitado."""
        feats = {**_VALID_FEATURES, "nota_cg": 0}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_defasagem_abaixo_min(self, client, auth_client):
        feats = {**_VALID_FEATURES, "defasagem": -10}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_turma_inválida(self, client, auth_client):
        """Letra não existente no enum deve dar 422."""
        feats = {**_VALID_FEATURES, "turma": "w"}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_genero_inválido(self, client, auth_client):
        feats = {**_VALID_FEATURES, "genero": "outro"}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_pedra_inválida(self, client, auth_client):
        feats = {**_VALID_FEATURES, "pedra_modal": "diamante"}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    def test_predict_campo_ausente(self, client, auth_client):
        """Omitir campo obrigatório deve dar 422."""
        feats = {k: v for k, v in _VALID_FEATURES.items() if k != "fase"}
        resp = auth_client.post("/predict", json=_student_body(features=feats))

        assert resp.status_code == 422

    # ── 503 sem artefatos ─────────────────────────────────
    def test_predict_503_sem_modelo(self, fake_context):
        from fastapi.testclient import TestClient
        from app.routes import User

        ctx = {**fake_context, "model": None}
        with patch("context.get_model_context", return_value=ctx), \
             patch("routes.get_model_context", return_value=ctx), \
             patch("routes.get_current_active_user", return_value=User(username="test")):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post("/predict", json=_student_body())

        assert resp.status_code == 503

    def test_predict_503_sem_preprocessor(self, fake_context):
        from fastapi.testclient import TestClient
        from app.routes import User

        ctx = {**fake_context, "preprocessor": None}
        with patch("context.get_model_context", return_value=ctx), \
             patch("routes.get_model_context", return_value=ctx), \
             patch("routes.get_current_active_user", return_value=User(username="test")):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post("/predict", json=_student_body())

        assert resp.status_code == 503

    def test_predict_503_sem_ambos(self, fake_context):
        from fastapi.testclient import TestClient
        from app.routes import User

        ctx = {**fake_context, "model": None, "preprocessor": None}
        with patch("context.get_model_context", return_value=ctx), \
             patch("routes.get_model_context", return_value=ctx), \
             patch("routes.get_current_active_user", return_value=User(username="test")):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post("/predict", json=_student_body())

        assert resp.status_code == 503


# ════════════════════════════════════════════════════════════
# TestPredictBatchEndpoint
# ════════════════════════════════════════════════════════════

class TestPredictBatchEndpoint:
    """POST /predict/batch"""

    def _batch_body(self, n: int = 3) -> dict:
        return {
            "students": [
                _student_body(student_id=f"RA-{i}")
                for i in range(n)
            ]
        }

    def test_batch_200(self, client, auth_client):
        # unauthorized should be rejected
        assert client.post("/predict/batch", json=self._batch_body()).status_code == 401
        resp = auth_client.post("/predict/batch", json=self._batch_body())

        assert resp.status_code == 200
        # check logging count
        assert auth_client.mongo_logger.log_prediction.call_count == 3

    def test_batch_campos_resposta(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body()).json()

        assert "total" in body
        assert "predictions" in body
        assert "timestamp" in body

    def test_batch_total_correto(self, client, auth_client):
        n = 5
        body = auth_client.post(
            "/predict/batch", json=self._batch_body(n)
        ).json()

        assert body["total"] == n
        assert len(body["predictions"]) == n

    def test_batch_cada_predição_completa(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body(3)).json()

        for pred in body["predictions"]:
            assert "student_id" in pred
            assert "probability" in pred
            assert "prediction" in pred
            assert "confidence" in pred
            assert "prediction_id" in pred

    def test_batch_student_ids_preservados(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body(3)).json()
        ids = [p["student_id"] for p in body["predictions"]]

        assert ids == ["RA-0", "RA-1", "RA-2"]

    def test_batch_probabilidades_válidas(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body(4)).json()

        for pred in body["predictions"]:
            assert 0.0 <= pred["probability"] <= 1.0

    # ── consistência com predict individual ───────────────
    def test_batch_consistente_com_individual(self, client, auth_client):
        """Mesmo input via batch e individual deve dar mesma probabilidade."""
        single = auth_client.post("/predict", json=_student_body(student_id="X")).json()

        batch_body = {"students": [_student_body(student_id="X")]}
        batch = auth_client.post("/predict/batch", json=batch_body).json()

        assert single["probability"] == batch["predictions"][0]["probability"]

    # ── determinismo ──────────────────────────────────────
    def test_batch_determinismo(self, client, auth_client):
        b1 = auth_client.post("/predict/batch", json=self._batch_body(2)).json()
        b2 = auth_client.post("/predict/batch", json=self._batch_body(2)).json()

        probs1 = [p["probability"] for p in b1["predictions"]]
        probs2 = [p["probability"] for p in b2["predictions"]]

        assert probs1 == probs2

    # ── limites de tamanho ────────────────────────────────
    def test_batch_vazio_422(self, client, auth_client):
        resp = auth_client.post("/predict/batch", json={"students": []})

        assert resp.status_code == 422

    def test_batch_501_alunos_422(self, client, auth_client):
        """Mais que 500 deve ser rejeitado."""
        body = {"students": [_student_body(student_id=f"R-{i}") for i in range(501)]}
        resp = client.post("/predict/batch", json=body)

        assert resp.status_code == 422

    # ── 503 sem artefatos ─────────────────────────────────
    def test_batch_503_sem_modelo(self, fake_context):
        from fastapi.testclient import TestClient

        ctx = {**fake_context, "model": None}
        with patch("context.get_model_context", return_value=ctx), \
             patch("routes.get_model_context", return_value=ctx):
            from app.main import app
            c = TestClient(app, raise_server_exceptions=False)
            resp = c.post("/predict/batch", json={"students": [_student_body()]})

        assert resp.status_code == 503

    # ── inputs mistos válidos ─────────────────────────────
    def test_batch_inputs_diferentes(self, client):
        """Alunos com dados diferentes devem produzir predições."""
        students = [
            _student_body(
                student_id=f"S-{i}",
                features={**_VALID_FEATURES, "nota_cg": 100 * (i + 1)},
            )
            for i in range(3)
        ]
        body = client.post(
            "/predict/batch", json={"students": students}
        ).json()

        assert body["total"] == 3
        # todas as probabilidades devem ser válidas
        for pred in body["predictions"]:
            assert 0.0 <= pred["probability"] <= 1.0


# ════════════════════════════════════════════════════════════
# EXECUÇÃO DIRETA
# ════════════════════════════════════════════════════════════
#
#   python test_model.py          → rodan todos
#   python test_model.py -v       → verbose
#   python test_model.py -k batch → filtrar por nome
#
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))