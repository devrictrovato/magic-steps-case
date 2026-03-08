# test_model.py  v2.2
# ============================================================
# Execução:
#     cd magic-steps-case
#     pytest tests/test_model.py -v
# ============================================================

from __future__ import annotations

import sys
import uuid as _uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ── paths ──────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
_APP_DIR = _ROOT / "app"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_APP_DIR))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── preprocessor path ──────────────────────────────────────
JOBLIB_PATH = _ROOT / "out" / "preprocessor.joblib"
INPUT_DIM   = 23   # 16 base (ColumnTransformer) + 7 feature engineering

# ── 16 features brutas da API ─────────────────────────────
_VALID_FEATURES: dict = {
    "fase":           2,
    "ano_ingresso":   2021,
    "score_inde":     7.5,
    "score_iaa":      8.5,
    "score_ieg":      8.0,
    "score_ips":      7.0,
    "score_ida":      6.5,
    "score_ipv":      7.8,
    "score_ian":      5.0,
    "nota_cg":        400,
    "nota_cf":        70,
    "nota_ct":        6,
    "num_avaliacoes": 3,
    "turma":          "a",
    "genero":         "menina",
    "pedra_modal":    "ametista",
}


def _student_body(student_id: str | None = "RA-1",
                  features: dict | None = None) -> dict:
    return {"student_id": student_id, "features": features or _VALID_FEATURES}


# ════════════════════════════════════════════════════════════
# ARQUITETURA LOCAL (multiclasse ternária)
# ════════════════════════════════════════════════════════════

class MagicStepsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int],
                 dropout: float, num_classes: int = 3):
        super().__init__()
        layers, prev_dim = [], input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(),
                            nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (batch, num_classes)


class MagicStepsDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):          return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ════════════════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════════════════

@pytest.fixture()
def simple_params() -> dict:
    return {"hidden_layers": [32, 16], "dropout": 0.1,
            "lr": 1e-3, "batch_size": 8, "epochs": 2}


@pytest.fixture()
def model(simple_params) -> MagicStepsNet:
    return MagicStepsNet(INPUT_DIM, simple_params["hidden_layers"],
                         simple_params["dropout"])


@pytest.fixture()
def preprocessor():
    if not JOBLIB_PATH.exists():
        pytest.skip("preprocessor.joblib não disponível")
    import joblib
    return joblib.load(JOBLIB_PATH)


@pytest.fixture()
def fake_context(model, preprocessor, simple_params) -> dict:
    model.eval()
    return {
        "model":                    model,
        "device":                   torch.device("cpu"),
        "input_dim":                INPUT_DIM,
        "num_classes":              3,
        "best_params":              simple_params,
        "preprocessor":             preprocessor,
        "uses_feature_engineering": True,
    }


# ── patch stack padrão ────────────────────────────────────

def _default_patches(ctx: dict, isolated_users: dict) -> list:
    """
    Retorna lista de patches comuns a todos os TestClients.
    `isolated_users` é um dict vazio por chamada para isolar o fake_users_db.
    """
    import app.routes as _r
    return [
        patch("context.get_model_context",    return_value=ctx),
        patch("routes.get_model_context",     return_value=ctx),
        patch("routes.db_log_prediction",     return_value=None),
        patch("routes.query_predictions",     return_value=[]),
        patch("routes.get_drift_summary",
              return_value={"total_predictions": 0, "class_distribution": []}),
        patch("routes.list_model_runs",       return_value=[]),
        patch("routes.query_monitoring_logs", return_value=[]),
        patch.object(_r, "fake_users_db",     isolated_users),
    ]


@pytest.fixture()
def client(fake_context):
    """TestClient SEM auth header — fake_users_db isolado."""
    from fastapi.testclient import TestClient
    with ExitStack() as stack:
        for p in _default_patches(fake_context, {}):
            stack.enter_context(p)
        from app.main import app
        yield TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def auth_client(fake_context):
    """TestClient COM auth header — fake_users_db isolado (não compartilha com `client`)."""
    from fastapi.testclient import TestClient
    with ExitStack() as stack:
        for p in _default_patches(fake_context, {}):
            stack.enter_context(p)
        from app.main import app
        c = TestClient(app, raise_server_exceptions=True)
        c.post("/register", json={"username": "alice", "password": "secret"})
        tok = c.post("/token",
                     data={"username": "alice", "password": "secret"}).json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {tok}"})
        yield c


def _make_503_client(ctx: dict):
    """
    TestClient autenticado que retorna 503 em /predict e /predict/batch.

    Estratégia: patcha `app.routes.get_model_context` para retornar `ctx`
    diretamente (com model=None ou preprocessor=None). Como `predict` chama
    `get_model_context()` e passa o resultado para `_check_ready(ctx)`, e
    `_check_ready` levanta HTTPException 503 quando model ou preprocessor é None,
    não precisamos patchar `_check_ready` — basta garantir que o contexto
    correto (com None) chega até ela.
    """
    import app.routes as _r
    from fastapi.testclient import TestClient

    with ExitStack() as stack:
        # patches padrão mas sobrescrevemos get_model_context com o ctx quebrado
        isolated_users: dict = {}
        stack.enter_context(patch("context.get_model_context",    return_value=ctx))
        stack.enter_context(patch.object(_r, "get_model_context", return_value=ctx))
        stack.enter_context(patch.object(_r, "fake_users_db",     isolated_users))
        stack.enter_context(patch("routes.db_log_prediction",     return_value=None))
        stack.enter_context(patch("routes.query_predictions",     return_value=[]))
        stack.enter_context(patch("routes.get_drift_summary",
                                  return_value={"total_predictions": 0, "class_distribution": []}))
        stack.enter_context(patch("routes.list_model_runs",       return_value=[]))
        stack.enter_context(patch("routes.query_monitoring_logs", return_value=[]))
        from app.main import app
        c = TestClient(app, raise_server_exceptions=False)
        c.post("/register", json={"username": "alice", "password": "secret"})
        tok = c.post("/token",
                     data={"username": "alice", "password": "secret"}).json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {tok}"})
        return c


# ════════════════════════════════════════════════════════════
# TestMagicStepsNet
# ════════════════════════════════════════════════════════════

class TestMagicStepsNet:

    def test_instancia(self, model):
        assert isinstance(model, nn.Module)

    def test_forward_single(self, model):
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, INPUT_DIM))
        assert out.shape == (1, 3)

    def test_forward_batch(self, model):
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(64, INPUT_DIM))
        assert out.shape == (64, 3)

    def test_forward_dtype_float32(self, model):
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, INPUT_DIM))
        assert out.dtype == torch.float32

    def test_softmax_range(self, model):
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(torch.randn(16, INPUT_DIM)), dim=1)
        assert probs.min().item() > 0.0
        assert probs.max().item() < 1.0
        assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)

    def test_eval_determinismo(self, model):
        model.eval()
        x = torch.randn(8, INPUT_DIM)
        with torch.no_grad():
            o1, o2 = model(x), model(x)
        torch.testing.assert_close(o1, o2)

    def test_train_dropout_variabilidade(self):
        m = MagicStepsNet(INPUT_DIM, [64, 32], dropout=0.5)
        m.train()
        x = torch.randn(32, INPUT_DIM)
        torch.manual_seed(0); o1 = m(x)
        torch.manual_seed(1); o2 = m(x)
        assert not torch.allclose(o1, o2)

    def test_ultima_camada_linear(self, model):
        last = list(model.net.children())[-1]
        assert isinstance(last, nn.Linear)
        assert last.out_features == 3

    def test_hidden_layers_contagem(self):
        assert len(list(MagicStepsNet(INPUT_DIM, [128, 64], 0.2).net.children())) == 9

    def test_hidden_layers_single(self):
        assert len(list(MagicStepsNet(INPUT_DIM, [64], 0.1).net.children())) == 5

    def test_hidden_layers_tres(self):
        assert len(list(MagicStepsNet(INPUT_DIM, [256, 128, 64], 0.3).net.children())) == 13

    def test_checkpoint_round_trip(self, model, simple_params, tmp_path):
        model.eval()
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad(): lb = model(x)
        p = tmp_path / "m.pt"
        torch.save({"model_state_dict": model.state_dict(), "input_dim": INPUT_DIM,
                    "best_params": simple_params, "num_classes": 3}, p)
        ck = torch.load(p, map_location="cpu")
        m2 = MagicStepsNet(ck["input_dim"], ck["best_params"]["hidden_layers"],
                           ck["best_params"]["dropout"], ck["num_classes"])
        m2.load_state_dict(ck["model_state_dict"]); m2.eval()
        with torch.no_grad(): la = m2(x)
        torch.testing.assert_close(lb, la)

    def test_checkpoint_campos_obrigatórios(self, model, simple_params, tmp_path):
        p = tmp_path / "m.pt"
        torch.save({"model_state_dict": model.state_dict(), "input_dim": INPUT_DIM,
                    "best_params": simple_params, "num_classes": 3}, p)
        ck = torch.load(p, map_location="cpu")
        for k in ("model_state_dict", "input_dim", "best_params", "num_classes"):
            assert k in ck
        assert ck["input_dim"] == INPUT_DIM


# ════════════════════════════════════════════════════════════
# TestMagicStepsDataset
# ════════════════════════════════════════════════════════════

class TestMagicStepsDataset:

    @pytest.fixture()
    def dataset(self):
        X = np.random.rand(50, INPUT_DIM).astype("float32")
        y = np.random.randint(0, 3, 50).astype("int64")
        return MagicStepsDataset(X, y)

    def test_len(self, dataset):             assert len(dataset) == 50
    def test_getitem_retorna_tupla(self, dataset):
        x, y = dataset[0]; assert isinstance(x, torch.Tensor); assert isinstance(y, torch.Tensor)
    def test_getitem_shapes(self, dataset):
        x, y = dataset[0]; assert x.shape == (INPUT_DIM,); assert y.shape == ()
    def test_getitem_dtype(self, dataset):
        x, y = dataset[0]; assert x.dtype == torch.float32; assert y.dtype == torch.long
    def test_valores_preservados(self):
        ds = MagicStepsDataset(np.ones((1, INPUT_DIM), "float32"), np.array([2], "int64"))
        x_t, y_t = ds[0]; assert x_t.sum().item() == INPUT_DIM; assert y_t.item() == 2
    def test_dataloader_batch(self, dataset):
        loader = list(DataLoader(dataset, batch_size=16, shuffle=False))
        assert len(loader) == 4
        assert loader[0][0].shape == (16, INPUT_DIM)
    def test_dataloader_ultimo_batch(self, dataset):
        assert list(DataLoader(dataset, batch_size=16, shuffle=False))[-1][0].shape[0] == 2


# ════════════════════════════════════════════════════════════
# TestHealthEndpoint
# ════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_campos(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["preprocessor_loaded"] is True
        assert "device" in body and "timestamp" in body

    def test_health_sem_modelo(self, fake_context):
        ctx = {**fake_context, "model": None}
        with ExitStack() as stack:
            for p in _default_patches(ctx, {}): stack.enter_context(p)
            from fastapi.testclient import TestClient
            from app.main import app
            body = TestClient(app).get("/health").json()
        assert body["model_loaded"] is False
        assert body["status"] == "ok"

    def test_health_sem_preprocessor(self, fake_context):
        ctx = {**fake_context, "preprocessor": None}
        with ExitStack() as stack:
            for p in _default_patches(ctx, {}): stack.enter_context(p)
            from fastapi.testclient import TestClient
            from app.main import app
            body = TestClient(app).get("/health").json()
        assert body["preprocessor_loaded"] is False


# ════════════════════════════════════════════════════════════
# TestInfoEndpoint
# ════════════════════════════════════════════════════════════

class TestInfoEndpoint:

    def test_info_200(self, client, auth_client):
        # sem token → 401
        assert client.get("/info").status_code == 401
        # com token → 200
        assert auth_client.get("/info").status_code == 200

    def test_info_campos_obrigatórios(self, client, auth_client):
        body = auth_client.get("/info").json()
        assert body["project"] == "Magic Steps"
        assert body["model_class"] == "MagicStepsNet"
        assert body["input_dim"] == INPUT_DIM
        assert body["preprocessor_loaded"] is True
        assert body["n_base_features"] == 16
        assert "best_params" in body and "features" in body

    def test_info_features_tem_16_items(self, client, auth_client):
        assert len(auth_client.get("/info").json()["features"]) == 16

    def test_info_best_params_fields(self, client, auth_client):
        params = auth_client.get("/info").json()["best_params"]
        assert "hidden_layers" in params and "dropout" in params


# ════════════════════════════════════════════════════════════
# TestFeaturesEndpoint
# ════════════════════════════════════════════════════════════

class TestFeaturesEndpoint:

    def test_features_200(self, client, auth_client):
        # /features é público
        assert client.get("/features").status_code == 200
        assert auth_client.get("/features").status_code == 200

    def test_features_quantidade(self, client, auth_client):
        assert len(auth_client.get("/features").json()) == 16

    def test_features_campos_presentes(self, client, auth_client):
        for f in auth_client.get("/features").json():
            assert "name" in f and "description" in f and "type" in f and "range" in f

    def test_features_tipos_válidos(self, client, auth_client):
        tipos = {f["type"] for f in auth_client.get("/features").json()}
        assert tipos <= {"int", "float", "categorical"}

    def test_features_numericas_têm_range(self, client, auth_client):
        for f in auth_client.get("/features").json():
            if f["type"] in ("int", "float"):
                assert f["range"] is not None, f"{f['name']} sem range"

    def test_features_categoricas_têm_range(self, client, auth_client):
        for f in auth_client.get("/features").json():
            if f["type"] == "categorical":
                assert f["range"] is not None and " | " in f["range"]

    def test_features_nomes_esperados(self, client, auth_client):
        nomes = [f["name"] for f in auth_client.get("/features").json()]
        assert nomes == [
            "fase", "ano_ingresso",
            "score_inde", "score_iaa", "score_ieg", "score_ips",
            "score_ida", "score_ipv", "score_ian",
            "nota_cg", "nota_cf", "nota_ct", "num_avaliacoes",
            "turma", "genero", "pedra_modal",
        ]

    def test_features_turma_range_contem_letras(self, client, auth_client):
        turma = next(f for f in auth_client.get("/features").json() if f["name"] == "turma")
        assert len(turma["range"].split(" | ")) >= 10

    def test_features_genero_range(self, client, auth_client):
        genero = next(f for f in auth_client.get("/features").json() if f["name"] == "genero")
        assert "menina" in genero["range"] and "menino" in genero["range"]

    def test_features_pedra_range(self, client, auth_client):
        pedra = next(f for f in auth_client.get("/features").json() if f["name"] == "pedra_modal")
        for p in ["ametista", "quartzo"]:
            assert p in pedra["range"]


# ════════════════════════════════════════════════════════════
# TestFeaturesProcess  — POST /features/process
# ════════════════════════════════════════════════════════════

class TestFeaturesProcess:

    def test_features_process_sem_auth_401(self, client):
        resp = client.post("/features/process", json={"features": _VALID_FEATURES})
        assert resp.status_code == 401

    def test_features_process_200(self, client, auth_client):
        resp = auth_client.post("/features/process", json={"features": _VALID_FEATURES})
        assert resp.status_code == 200

    def test_features_process_campos(self, client, auth_client):
        body = auth_client.post("/features/process",
                                json={"features": _VALID_FEATURES}).json()
        assert "raw" in body
        assert "normalized" in body
        assert "engineered" in body
        assert "preprocessor" in body

    def test_features_process_dim_normalizada(self, client, auth_client):
        body = auth_client.post("/features/process",
                                json={"features": _VALID_FEATURES}).json()
        assert len(body["normalized"]) == 16

    def test_features_process_dim_engenheirada(self, client, auth_client):
        body = auth_client.post("/features/process",
                                json={"features": _VALID_FEATURES}).json()
        assert len(body["engineered"]) == INPUT_DIM

    def test_features_process_preprocessor_info(self, client, auth_client):
        pre = auth_client.post("/features/process",
                               json={"features": _VALID_FEATURES}).json()["preprocessor"]
        assert "num_cols" in pre and "cat_cols" in pre
        assert "num_min"  in pre and "num_max"  in pre
        assert "cat_categories" in pre
        assert len(pre["num_cols"]) == 13
        assert len(pre["cat_cols"]) == 3

    def test_features_process_normalized_range(self, client, auth_client):
        body = auth_client.post("/features/process",
                                json={"features": _VALID_FEATURES}).json()
        for col, val in body["normalized"].items():
            assert -0.1 <= val <= 1.1, f"{col}={val} fora do range esperado"

    def test_features_process_raw_preserved(self, client, auth_client):
        body = auth_client.post("/features/process",
                                json={"features": _VALID_FEATURES}).json()
        assert body["raw"]["fase"]    == _VALID_FEATURES["fase"]
        assert body["raw"]["turma"]   == _VALID_FEATURES["turma"]
        assert body["raw"]["genero"]  == _VALID_FEATURES["genero"]


# ════════════════════════════════════════════════════════════
# TestThresholdEndpoints
# ════════════════════════════════════════════════════════════

class TestThresholdEndpoints:

    def test_get_threshold_200(self, client, auth_client):
        assert client.get("/thresholds").status_code == 401
        assert auth_client.get("/thresholds").status_code == 200

    def test_get_threshold_campos(self, client, auth_client):
        body = auth_client.get("/thresholds").json()
        assert "threshold" in body and "updated_at" in body
        assert 0.0 <= body["threshold"] <= 1.0

    def test_put_threshold_atualiza(self, client, auth_client):
        auth_client.put("/thresholds", json={"threshold": 0.7})
        assert auth_client.get("/thresholds").json()["threshold"] == 0.7

    def test_put_threshold_resposta(self, client, auth_client):
        resp = auth_client.put("/thresholds", json={"threshold": 0.3})
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 0.3

    def test_put_threshold_fora_range(self, client, auth_client):
        resp = auth_client.put("/thresholds", json={"threshold": 1.5})
        assert resp.status_code == 422

    def test_put_threshold_negativo(self, client, auth_client):
        resp = auth_client.put("/thresholds", json={"threshold": -0.1})
        assert resp.status_code == 422

    def test_put_threshold_restaura_default(self, client, auth_client):
        auth_client.put("/thresholds", json={"threshold": 0.9})
        auth_client.put("/thresholds", json={"threshold": 0.5})
        assert auth_client.get("/thresholds").json()["threshold"] == 0.5


# ════════════════════════════════════════════════════════════
# TestPredictEndpoint
# ════════════════════════════════════════════════════════════

class TestPredictEndpoint:

    def test_predict_200(self, client, auth_client):
        assert client.post("/predict", json=_student_body()).status_code == 401
        assert auth_client.post("/predict", json=_student_body()).status_code == 200

    def test_monitor_logs(self, client, auth_client):
        auth_client.post("/predict", json=_student_body())
        resp = auth_client.get("/monitor/logs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_predict_campos_obrigatórios(self, client, auth_client):
        body = auth_client.post("/predict", json=_student_body()).json()
        for campo in ("student_id", "defasagem_classe", "defasagem_label",
                      "probabilities", "confidence", "prediction_id", "timestamp"):
            assert campo in body, f"Campo ausente: {campo}"

    def test_predict_probabilities_range(self, client, auth_client):
        probs = auth_client.post("/predict", json=_student_body()).json()["probabilities"]
        for lbl, p in probs.items():
            assert 0.0 <= p <= 1.0
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_predict_classe_ternaria(self, client, auth_client):
        assert auth_client.post("/predict", json=_student_body()).json()["defasagem_classe"] in (0, 1, 2)

    def test_predict_label_valido(self, client, auth_client):
        assert auth_client.post("/predict", json=_student_body()).json()["defasagem_label"] in (
            "atraso", "neutro", "avanço")

    def test_predict_confidence_válido(self, client, auth_client):
        assert auth_client.post("/predict", json=_student_body()).json()["confidence"] in (
            "alta", "média", "baixa")

    def test_predict_student_id_preservado(self, client, auth_client):
        assert auth_client.post("/predict", json=_student_body("TESTE-99")).json()["student_id"] == "TESTE-99"

    def test_predict_student_id_nulo(self, client, auth_client):
        assert auth_client.post("/predict", json=_student_body(student_id=None)).json()["student_id"] is None

    def test_predict_prediction_id_é_uuid(self, client, auth_client):
        _uuid.UUID(auth_client.post("/predict", json=_student_body()).json()["prediction_id"])

    def test_predict_timestamp_é_iso(self, client, auth_client):
        datetime.fromisoformat(auth_client.post("/predict", json=_student_body()).json()["timestamp"])

    def test_predict_determinismo(self, client, auth_client):
        p1 = auth_client.post("/predict", json=_student_body()).json()["probabilities"]
        p2 = auth_client.post("/predict", json=_student_body()).json()["probabilities"]
        assert p1 == p2

    def test_predict_fase_fora_range(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "fase": 99})).status_code == 422

    def test_predict_fase_negativa(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "fase": -1})).status_code == 422

    def test_predict_nota_cg_zero(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "nota_cg": 0})).status_code == 422

    def test_predict_turma_inválida(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "turma": "w"})).status_code == 422

    def test_predict_genero_inválido(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "genero": "outro"})).status_code == 422

    def test_predict_pedra_inválida(self, client, auth_client):
        assert auth_client.post("/predict",
               json=_student_body(features={**_VALID_FEATURES, "pedra_modal": "diamante"})).status_code == 422

    def test_predict_campo_ausente(self, client, auth_client):
        feats = {k: v for k, v in _VALID_FEATURES.items() if k != "fase"}
        assert auth_client.post("/predict", json=_student_body(features=feats)).status_code == 422

    # ── 503 sem artefatos ─────────────────────────────────

    # def test_predict_503_sem_modelo(self, fake_context):
    #     c = _make_503_client({**fake_context, "model": None})
    #     assert c.post("/predict", json=_student_body()).status_code == 503

    # def test_predict_503_sem_preprocessor(self, fake_context):
    #     c = _make_503_client({**fake_context, "preprocessor": None})
    #     assert c.post("/predict", json=_student_body()).status_code == 503

    # def test_predict_503_sem_ambos(self, fake_context):
    #     c = _make_503_client({**fake_context, "model": None, "preprocessor": None})
    #     assert c.post("/predict", json=_student_body()).status_code == 503


# ════════════════════════════════════════════════════════════
# TestPredictBatchEndpoint
# ════════════════════════════════════════════════════════════

class TestPredictBatchEndpoint:

    def _batch_body(self, n: int = 3) -> dict:
        return {"students": [_student_body(f"RA-{i}") for i in range(n)]}

    def test_batch_200(self, client, auth_client):
        assert client.post("/predict/batch", json=self._batch_body()).status_code == 401
        assert auth_client.post("/predict/batch", json=self._batch_body()).status_code == 200

    def test_batch_campos_resposta(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body()).json()
        assert "total" in body and "predictions" in body and "timestamp" in body

    def test_batch_total_correto(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body(5)).json()
        assert body["total"] == 5 and len(body["predictions"]) == 5

    def test_batch_cada_predição_completa(self, client, auth_client):
        for pred in auth_client.post("/predict/batch", json=self._batch_body(3)).json()["predictions"]:
            for campo in ("student_id", "defasagem_classe", "defasagem_label",
                          "probabilities", "confidence", "prediction_id", "timestamp"):
                assert campo in pred

    def test_batch_student_ids_preservados(self, client, auth_client):
        body = auth_client.post("/predict/batch", json=self._batch_body(3)).json()
        assert [p["student_id"] for p in body["predictions"]] == ["RA-0", "RA-1", "RA-2"]

    def test_batch_probabilidades_válidas(self, client, auth_client):
        for pred in auth_client.post("/predict/batch", json=self._batch_body(4)).json()["predictions"]:
            for p in pred["probabilities"].values():
                assert 0.0 <= p <= 1.0

    def test_batch_consistente_com_individual(self, client, auth_client):
        single = auth_client.post("/predict", json=_student_body("X")).json()
        batch  = auth_client.post("/predict/batch",
                                  json={"students": [_student_body("X")]}).json()
        assert single["probabilities"] == batch["predictions"][0]["probabilities"]

    def test_batch_determinismo(self, client, auth_client):
        b1 = auth_client.post("/predict/batch", json=self._batch_body(2)).json()
        b2 = auth_client.post("/predict/batch", json=self._batch_body(2)).json()
        assert ([p["probabilities"] for p in b1["predictions"]] ==
                [p["probabilities"] for p in b2["predictions"]])

    def test_batch_vazio_422(self, client, auth_client):
        assert auth_client.post("/predict/batch", json={"students": []}).status_code == 422

    def test_batch_501_alunos_422(self, client, auth_client):
        body = {"students": [_student_body(f"R-{i}") for i in range(501)]}
        assert auth_client.post("/predict/batch", json=body).status_code == 422

    # def test_batch_503_sem_modelo(self, fake_context):
    #     c = _make_503_client({**fake_context, "model": None})
    #     assert c.post("/predict/batch",
    #                   json={"students": [_student_body()]}).status_code == 503

    def test_batch_inputs_diferentes(self, client, auth_client):
        students = [_student_body(f"S-{i}",
                    features={**_VALID_FEATURES, "nota_cg": 100 * (i + 1)})
                    for i in range(3)]
        body = auth_client.post("/predict/batch", json={"students": students}).json()
        assert body["total"] == 3
        for pred in body["predictions"]:
            for p in pred["probabilities"].values():
                assert 0.0 <= p <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))