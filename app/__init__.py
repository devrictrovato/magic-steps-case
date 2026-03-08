"""
App package initializer.

Importa apenas os símbolos que vivem dentro do próprio pacote `app/`.
Os módulos de `src/` (db, settings, utils…) usam imports absolutos e
dependem do sys.path configurado em main.py/routes.py — não os
reexporte daqui para evitar importações circulares e erros de path.
"""

from .context import DEVICE, _MODEL_CONTEXT, get_model_context
from .routes import router

__all__ = [
    "DEVICE",
    "_MODEL_CONTEXT",
    "get_model_context",
    "router",
]