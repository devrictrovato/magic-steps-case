"""
Pipeline de treinamento do modelo com Deep Learning e Grid Search.
"""

from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict, Any, Optional
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


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "training.log")


# ============================================================
# DEVICE SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibilidade
torch.manual_seed(settings.random_state)
np.random.seed(settings.random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(settings.random_state)


# ============================================================
# DATASET
# ============================================================

class MagicStepsDataset(Dataset):
    """Dataset customizado para PyTorch."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Inicializa o dataset.
        
        Args:
            X: Features
            y: Labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # CrossEntropyLoss exige long para labels de classe
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataLoader_:
    """Carregador de dados para treinamento."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def load_dataset(self, use_feast: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega dataset para treinamento.
        
        Args:
            use_feast: Se True, carrega do Feast
        
        Returns:
            Tupla (X, y)
        """
        self.logger.info("Carregando dataset...")
        
        df = load_features_for_training(use_feast=use_feast)
        
        target_col = data_config.target_column
        
        X = df.drop(columns=[target_col]).values.astype("float32")
        y_raw = df[target_col].values
        
        # Classes já são 0/1/2 (bucket_defasagem no preprocessing)
        y = y_raw.astype("int64")
        num_classes = 3  # 0=atraso, 1=neutro, 2=avanço

        self.logger.info(f"Dataset carregado. X shape: {X.shape}, y shape: {y.shape}")
        dist = dict(zip(*np.unique(y, return_counts=True)))
        self.logger.info(f"Distribuição: {dist} | 0=atraso 1=neutro 2=avanço")

        return X, y, num_classes


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MagicStepsNet(nn.Module):
    """Rede Neural para classificação multiclasse (defasagem)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        """
        Inicializa a rede neural.
        
        Args:
            input_dim: Dimensão de entrada
            hidden_layers: Lista com dimensões das camadas ocultas
            dropout: Taxa de dropout
            num_classes: Número de classes de saída (len(label_encoder.classes_))
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        
        # Camada de saída com num_classes neurônios
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — retorna logits (sem softmax).
        
        Args:
            x: Input tensor
        
        Returns:
            Logits shape (batch, num_classes)
        """
        return self.net(x)


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """Treinador do modelo."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
    ):
        """
        Inicializa o treinador.
        
        Args:
            model: Modelo a ser treinado
            device: Device (CPU ou GPU)
        """
        self.model = model.to(device)
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
    
    def train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Treina uma época.
        
        Args:
            loader: DataLoader de treinamento
            criterion: Função de perda
            optimizer: Otimizador
        
        Returns:
            Perda média da época
        """
        self.model.train()
        total_loss = 0.0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    @torch.no_grad()
    def validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """
        Valida uma época.
        
        Args:
            loader: DataLoader de validação
            criterion: Função de perda
        
        Returns:
            Perda média de validação
        """
        self.model.eval()
        total_loss = 0.0
        
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            outputs = self.model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        patience: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Treina o modelo.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação
            criterion: Função de perda
            optimizer: Otimizador
            epochs: Número de épocas
            patience: Paciência para early stopping
        
        Returns:
            Histórico de treinamento
        """
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate_epoch(val_loader, criterion)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                self.logger.info(f"Early stopping na época {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Época {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
        
        return history


# ============================================================
# GRID SEARCH
# ============================================================

class GridSearchCV:
    """Grid Search para hiperparâmetros."""
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        device: torch.device = DEVICE,
    ):
        """
        Inicializa o Grid Search.
        
        Args:
            param_grid: Grid de parâmetros
            device: Device (CPU ou GPU)
        """
        self.param_grid = param_grid
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)
        
        self.best_params = None
        self.best_score = float("inf")
        self.results = []
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int = 2,
    ) -> Dict[str, Any]:
        """
        Executa o Grid Search.
        
        Args:
            X_train: Features de treinamento
            y_train: Labels de treinamento
            X_val: Features de validação
            y_val: Labels de validação
        
        Returns:
            Melhores parâmetros
        """
        keys, values = zip(*self.param_grid.items())
        combinations = list(product(*values))
        
        self.logger.info(f"Total de combinações: {len(combinations)}")
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            self.logger.info(f"\n[{i}/{len(combinations)}] Testando: {params}")
            
            # Treinar modelo com esses parâmetros
            val_loss = self._fit_with_params(
                X_train, y_train, X_val, y_val, params, num_classes=num_classes
            )
            
            # Armazenar resultados
            result = {**params, "val_loss": val_loss}
            self.results.append(result)
            
            # Atualizar melhor modelo
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_params = params
                self.logger.info(f"✨ Novo melhor modelo! Val Loss: {val_loss:.4f}")
        
        self.logger.info(f"\n🏆 Melhores parâmetros: {self.best_params}")
        self.logger.info(f"📉 Melhor Val Loss: {self.best_score:.4f}")
        
        return self.best_params
    
    def _fit_with_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Dict[str, Any],
        num_classes: int = 2,
    ) -> float:
        """
        Treina modelo com parâmetros específicos.
        
        Args:
            X_train: Features de treinamento
            y_train: Labels de treinamento
            X_val: Features de validação
            y_val: Labels de validação
            params: Parâmetros do modelo
        
        Returns:
            Perda de validação
        """
        # Criar modelo
        model = MagicStepsNet(
            input_dim=X_train.shape[1],
            hidden_layers=params["hidden_layers"],
            dropout=params["dropout"],
            num_classes=num_classes,
        )
        
        # Criar trainer
        trainer = Trainer(model, self.device)
        
        # Criar data loaders
        train_loader = DataLoader(
            MagicStepsDataset(X_train, y_train),
            batch_size=params["batch_size"],
            shuffle=True,
        )
        
        val_loader = DataLoader(
            MagicStepsDataset(X_val, y_val),
            batch_size=params["batch_size"],
            shuffle=False,
        )
        
        # CrossEntropyLoss para multiclasse
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        # Treinar
        history = trainer.fit(
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=params["epochs"],
            patience=model_config.patience,
        )
        
        # Retornar melhor perda de validação
        return min(history["val_loss"])


# ============================================================
# TRAINING PIPELINE
# ============================================================

class TrainingPipeline:
    """Pipeline completo de treinamento."""
    
    def __init__(self, use_mlflow: bool = True):
        """
        Inicializa o pipeline de treinamento.
        
        Args:
            use_mlflow: Se True, usa MLflow para tracking
        """
        self.use_mlflow = use_mlflow
        self.logger = setup_logger(self.__class__.__name__)
        self.model_registry = ModelRegistry()
        
        if use_mlflow:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
    
    def run(
        self,
        use_feast: bool = False,
        perform_grid_search: bool = True,
    ) -> Dict[str, Any]:
        """
        Executa o pipeline de treinamento.
        
        Args:
            use_feast: Se True, carrega features do Feast
            perform_grid_search: Se True, realiza grid search
        
        Returns:
            Dicionário com resultados do treinamento
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE DE TREINAMENTO")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {DEVICE}")
        
        # 1. Carregar dados
        data_loader = DataLoader_()
        X, y, num_classes = data_loader.load_dataset(use_feast=use_feast)
        self.logger.info(f"Número de classes (defasagem): {num_classes}")
        
        # 2. Split dos dados
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=model_config.val_size,
            stratify=y,
            random_state=settings.random_state,
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        
        # 3. Grid Search (opcional)
        if perform_grid_search:
            best_params = self._perform_grid_search(X_train, y_train, X_val, y_val, num_classes=num_classes)
        else:
            # Usar parâmetros padrão
            best_params = {
                "hidden_layers": [128, 64],
                "dropout": 0.2,
                "lr": 1e-3,
                "batch_size": 32,
                "epochs": 100,
            }
        
        # 4. Treinar modelo final
        model, history = self._train_final_model(X, y, best_params, num_classes=num_classes)
        
        # 5. Salvar modelo
        # 5. Salvar modelo (inclui num_classes e label_encoder para inferência)
        self._save_model(model, best_params, history, num_classes=num_classes)
        
        self.logger.info("=" * 60)
        self.logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        self.logger.info("=" * 60)
        
        return {
            "model": model,
            "best_params": best_params,
            "history": history,
        }
    
    def _perform_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int = 2,
    ) -> Dict[str, Any]:
        """Realiza grid search."""
        self.logger.info("\n🔍 Iniciando Grid Search...")
        
        grid_search = GridSearchCV(
            param_grid=model_config.param_grid,
            device=DEVICE,
        )
        
        best_params = grid_search.fit(X_train, y_train, X_val, y_val, num_classes=num_classes)
        
        return best_params
    
    def _train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        best_params: Dict[str, Any],
        num_classes: int = 2,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Treina modelo final com todos os dados."""
        self.logger.info("\n🧠 Treinando modelo final...")
        
        # Criar modelo
        model = MagicStepsNet(
            input_dim=X.shape[1],
            hidden_layers=best_params["hidden_layers"],
            dropout=best_params["dropout"],
            num_classes=num_classes,
        )
        
        # Criar trainer
        trainer = Trainer(model, DEVICE)
        
        # Criar data loader
        train_loader = DataLoader(
            MagicStepsDataset(X, y),
            batch_size=best_params["batch_size"],
            shuffle=True,
        )
        
        # CrossEntropyLoss para multiclasse
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
        
        # Treinar
        history = {"train_loss": []}
        
        for epoch in range(best_params["epochs"]):
            train_loss = trainer.train_epoch(train_loader, criterion, optimizer)
            history["train_loss"].append(train_loss)
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Época {epoch+1}/{best_params['epochs']} - Loss: {train_loss:.4f}")
        
        return model, history
    
    def _save_model(
        self,
        model: nn.Module,
        best_params: Dict[str, Any],
        history: Dict[str, List[float]],
        num_classes: int = 3,
    ) -> None:
        """Salva modelo e metadados."""
        model_name = "model_magic_steps_dl"
        model_path = MODELS_DIR / f"{model_name}.pt"
        
        # Salvar checkpoint
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
        
        # Salvar metadados
        metadata = {
            "model_name": model_name,
            "best_params": best_params,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        
        self.model_registry.save_model_metadata(model_name, metadata)
        
        # Log no MLflow (opcional)
        if self.use_mlflow:
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metric("final_train_loss", history["train_loss"][-1])
                mlflow.pytorch.log_model(model, "model")


# ============================================================
# MAIN
# ============================================================

def main():
    """Função principal."""
    pipeline = TrainingPipeline(use_mlflow=False)
    results = pipeline.run(
        use_feast=False,
        perform_grid_search=True,
    )
    
    print("\n✅ Treinamento finalizado!")


if __name__ == "__main__":
    print("🎯 Treinamento Deep Learning com Grid Search - Magic Steps")
    main()