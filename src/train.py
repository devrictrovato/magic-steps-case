"""
Pipeline de treinamento do modelo com Deep Learning e Grid Search.
Runs de treinamento são registrados no PostgreSQL (tabela model_runs).
"""

from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import model_config, data_config, settings, MODELS_DIR
from utils import setup_logger, ModelRegistry
from feature_engineering import load_features_for_training
from db import ensure_schema, create_model_run, update_model_run, log_monitoring_event


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "training.log")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(settings.random_state)
np.random.seed(settings.random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(settings.random_state)


# ============================================================
# DATASET
# ============================================================

class MagicStepsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataLoader_:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def load_dataset(self, use_feast: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        self.logger.info("Carregando dataset do PostgreSQL…")
        df = load_features_for_training(use_db=True)
        target_col = data_config.target_column
        X = df.drop(columns=[target_col]).values.astype("float32")
        y = df[target_col].values.astype("int64")
        num_classes = 3
        self.logger.info(f"Dataset: X={X.shape}, y={y.shape}")
        dist = dict(zip(*np.unique(y, return_counts=True)))
        self.logger.info(f"Distribuição: {dist} | 0=atraso 1=neutro 2=avanço")
        return X, y, num_classes


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MagicStepsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int],
                 dropout: float = 0.2, num_classes: int = 3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device = DEVICE):
        self.model = model.to(device)
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)

    def train_epoch(self, loader, criterion, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def validate_epoch(self, loader, criterion) -> float:
        self.model.eval()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            total_loss += criterion(self.model(X), y).item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs, patience=10):
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate_epoch(val_loader, criterion)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                self.logger.info(f"Early stopping na época {epoch+1}")
                break
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Época {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        return history


# ============================================================
# GRID SEARCH
# ============================================================

class GridSearchCV:
    def __init__(self, param_grid: Dict[str, List[Any]], device: torch.device = DEVICE):
        self.param_grid = param_grid
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
        self.best_params = None
        self.best_score = float("inf")
        self.results = []

    def fit(self, X_train, y_train, X_val, y_val, num_classes: int = 3) -> Dict[str, Any]:
        keys, values = zip(*self.param_grid.items())
        combinations = list(product(*values))
        self.logger.info(f"Total de combinações: {len(combinations)}")
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            self.logger.info(f"\n[{i}/{len(combinations)}] Testando: {params}")
            val_loss = self._fit_with_params(X_train, y_train, X_val, y_val, params, num_classes)
            self.results.append({**params, "val_loss": val_loss})
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_params = params
                self.logger.info(f"✨ Novo melhor modelo! Val Loss: {val_loss:.4f}")
        self.logger.info(f"\n🏆 Melhores parâmetros: {self.best_params}")
        return self.best_params

    def _fit_with_params(self, X_train, y_train, X_val, y_val, params, num_classes=3) -> float:
        model = MagicStepsNet(X_train.shape[1], params["hidden_layers"], params["dropout"], num_classes)
        trainer = Trainer(model, self.device)
        train_loader = DataLoader(MagicStepsDataset(X_train, y_train), batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(MagicStepsDataset(X_val, y_val), batch_size=params["batch_size"], shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        history = trainer.fit(train_loader, val_loader, criterion, optimizer,
                              epochs=params["epochs"], patience=model_config.patience)
        return min(history["val_loss"])


# ============================================================
# TRAINING PIPELINE
# ============================================================

class TrainingPipeline:
    def __init__(self, use_mlflow: bool = True):
        self.use_mlflow = use_mlflow
        self.logger = setup_logger(self.__class__.__name__)
        self.model_registry = ModelRegistry()
        if use_mlflow:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)

    def run(self, use_feast: bool = False, perform_grid_search: bool = True) -> Dict[str, Any]:
        run_id = str(uuid.uuid4())
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE DE TREINAMENTO")
        self.logger.info(f"Run ID: {run_id} | Device: {DEVICE}")
        self.logger.info("=" * 60)

        # Registrar run no PostgreSQL
        try:
            ensure_schema(settings.database_url)
            create_model_run(run_id, "model_magic_steps_dl", settings.database_url)
        except Exception as e:
            self.logger.warning(f"Não foi possível registrar run no PostgreSQL: {e}")

        try:
            data_loader = DataLoader_()
            X, y, num_classes = data_loader.load_dataset()
            self.logger.info(f"Número de classes: {num_classes}")

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=model_config.val_size,
                stratify=y, random_state=settings.random_state,
            )
            self.logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

            if perform_grid_search:
                best_params = self._perform_grid_search(X_train, y_train, X_val, y_val, num_classes)
            else:
                best_params = {"hidden_layers": [128, 64], "dropout": 0.2, "lr": 1e-3, "batch_size": 32, "epochs": 100}

            model, history = self._train_final_model(X, y, best_params, num_classes)
            self._save_model(model, best_params, history, num_classes)

            # Atualizar run no PostgreSQL
            final_loss = history["train_loss"][-1] if history["train_loss"] else None
            try:
                update_model_run(
                    run_id, settings.database_url,
                    best_params=best_params,
                    metrics={"final_train_loss": final_loss},
                    status="completed",
                )
                log_monitoring_event(
                    "training_completed",
                    {"run_id": run_id, "best_params": best_params, "final_loss": final_loss},
                    settings.database_url,
                )
            except Exception as e:
                self.logger.warning(f"Não foi possível atualizar run no PostgreSQL: {e}")

        except Exception as e:
            try:
                update_model_run(run_id, settings.database_url, status="failed", notes=str(e))
            except Exception:
                pass
            raise

        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        self.logger.info("=" * 60)
        return {"model": model, "best_params": best_params, "history": history}

    def _perform_grid_search(self, X_train, y_train, X_val, y_val, num_classes=3):
        self.logger.info("\n🔍 Iniciando Grid Search...")
        gs = GridSearchCV(param_grid=model_config.param_grid, device=DEVICE)
        return gs.fit(X_train, y_train, X_val, y_val, num_classes=num_classes)

    def _train_final_model(self, X, y, best_params, num_classes=3):
        self.logger.info("\n🧠 Treinando modelo final...")
        model = MagicStepsNet(X.shape[1], best_params["hidden_layers"], best_params["dropout"], num_classes)
        trainer = Trainer(model, DEVICE)
        train_loader = DataLoader(MagicStepsDataset(X, y), batch_size=best_params["batch_size"], shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        history = {"train_loss": []}
        for epoch in range(best_params["epochs"]):
            train_loss = trainer.train_epoch(train_loader, criterion, optimizer)
            history["train_loss"].append(train_loss)
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Época {epoch+1}/{best_params['epochs']} - Loss: {train_loss:.4f}")
        return model, history

    def _save_model(self, model, best_params, history, num_classes=3):
        model_name = "model_magic_steps_dl"
        model_path = MODELS_DIR / f"{model_name}.pt"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": list(model.parameters())[0].shape[1],
            "best_params": best_params,
            "history": history,
            "num_classes": num_classes,
            "class_labels": {0: "atraso", 1: "neutro", 2: "avanço"},
        }
        torch.save(checkpoint, model_path)
        self.logger.info(f"💾 Modelo salvo em: {model_path}")
        metadata = {
            "model_name": model_name,
            "best_params": best_params,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self.model_registry.save_model_metadata(model_name, metadata)
        if self.use_mlflow:
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metric("final_train_loss", history["train_loss"][-1])
                mlflow.pytorch.log_model(model, "model")


# ============================================================
# MAIN
# ============================================================

def main():
    pipeline = TrainingPipeline(use_mlflow=False)
    pipeline.run(use_feast=False, perform_grid_search=True)
    print("\n✅ Treinamento finalizado!")


if __name__ == "__main__":
    print("🎯 Treinamento Deep Learning com Grid Search — Magic Steps")
    main()