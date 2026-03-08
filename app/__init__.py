"""
App package initializer
Permite imports consistentes em diferentes ambientes (local, docker, render)
"""

from .context import DEVICE, _MODEL_CONTEXT, get_model_context
from .routes import router

__all__ = [
    "DEVICE",
    "_MODEL_CONTEXT",
    "get_model_context",
    "router",
]