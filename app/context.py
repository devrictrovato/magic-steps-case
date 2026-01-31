# context.py
# ============================================================
# Estado global compartilhado entre main e routes.
# Não importa nenhum dos dois — quebra ciclo de importação.
# ============================================================

from typing import Any, Dict

import torch

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Preenchido pelo main.py durante o startup.
# Chaves garantidas após load_model() + load_preprocessor().
_MODEL_CONTEXT: Dict[str, Any] = {
    "model":         None,   # MagicStepsNet (nn.Module)
    "device":        DEVICE,
    "input_dim":     None,   # int
    "best_params":   None,   # dict com hidden_layers, dropout, lr, …
    "preprocessor":  None,   # ColumnTransformer (sklearn) — preprocessor.joblib
}


def get_model_context() -> Dict[str, Any]:
    """Retorna o contexto global."""
    return _MODEL_CONTEXT