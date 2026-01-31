"""
Pipeline de avaliação do modelo treinado.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, MODELS_DIR, ARTIFACTS_DIR
from utils import setup_logger, ModelRegistry
from feature_engineering import load_features_for_training


# ============================================================
# LOGGER
# ============================================================

logger = setup_logger(__name__, "evaluation.log")


# ============================================================
# DEVICE SETUP
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL (MESMA ARQUITETURA DO TREINO)
# ============================================================

class MagicStepsNet(nn.Module):
    """Rede Neural para classificação binária."""
    
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float):
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
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    """Carregador de modelos treinados."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.model_registry = ModelRegistry()
    
    def load_model(
        self, model_name: str = "model_magic_steps_dl"
    ) -> tuple[nn.Module, Dict[str, Any]]:
        """
        Carrega modelo treinado.
        
        Args:
            model_name: Nome do modelo
        
        Returns:
            Tupla (modelo, checkpoint)
        """
        model_path = MODELS_DIR / f"{model_name}.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        self.logger.info(f"Carregando modelo de: {model_path}")
        
        # Carregar checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Criar modelo
        model = MagicStepsNet(
            input_dim=checkpoint["input_dim"],
            hidden_layers=checkpoint["best_params"]["hidden_layers"],
            dropout=checkpoint["best_params"]["dropout"],
        ).to(DEVICE)
        
        # Carregar pesos
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        self.logger.info("Modelo carregado com sucesso")
        
        return model, checkpoint


# ============================================================
# DATA LOADER
# ============================================================

class EvaluationDataLoader:
    """Carregador de dados para avaliação."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def load_dataset(
        self, use_feast: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Carrega dataset para avaliação.
        
        Args:
            use_feast: Se True, carrega do Feast
        
        Returns:
            Tupla (X, y)
        """
        self.logger.info("Carregando dataset para avaliação...")
        
        df = load_features_for_training(use_feast=use_feast)
        
        target_col = data_config.target_column
        
        X = df.drop(columns=[target_col]).values.astype("float32")
        y = df[target_col].values.astype("int32")
        
        self.logger.info(f"Dataset carregado. Shape: X={X.shape}, y={y.shape}")
        
        return X, y


# ============================================================
# METRICS CALCULATOR
# ============================================================

class MetricsCalculator:
    """Calculador de métricas de avaliação."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula métricas de avaliação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições (0 ou 1)
            y_proba: Probabilidades
        
        Returns:
            Dicionário com métricas
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Imprime métricas formatadas.
        
        Args:
            metrics: Dicionário com métricas
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 MÉTRICAS DE AVALIAÇÃO")
        self.logger.info("=" * 60)
        
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name.upper():<15}: {value:.4f}")
        
        self.logger.info("=" * 60)
    
    def get_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calcula matriz de confusão.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
        
        Returns:
            Matriz de confusão
        """
        return confusion_matrix(y_true, y_pred)
    
    def print_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Imprime matriz de confusão.
        
        Args:
            cm: Matriz de confusão
        """
        self.logger.info("\n🧮 MATRIZ DE CONFUSÃO")
        self.logger.info(f"\n{cm}")
        
        tn, fp, fn, tp = cm.ravel()
        self.logger.info(f"\nTrue Negatives:  {tn}")
        self.logger.info(f"False Positives: {fp}")
        self.logger.info(f"False Negatives: {fn}")
        self.logger.info(f"True Positives:  {tp}")
    
    def get_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """
        Gera relatório de classificação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
        
        Returns:
            Relatório de classificação
        """
        return classification_report(
            y_true, y_pred, target_names=["Não Atingiu", "Atingiu"]
        )


# ============================================================
# VISUALIZER
# ============================================================

class Visualizer:
    """Gerador de visualizações."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Inicializa o visualizador.
        
        Args:
            save_dir: Diretório para salvar figuras
        """
        self.save_dir = save_dir or ARTIFACTS_DIR / "imgs"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)
        
        # Configurar estilo
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")
    
    def plot_confusion_matrix(
        self, cm: np.ndarray, save: bool = True
    ) -> None:
        """
        Plota matriz de confusão.
        
        Args:
            cm: Matriz de confusão
            save: Se True, salva a figura
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Não Atingiu", "Atingiu"],
            yticklabels=["Não Atingiu", "Atingiu"],
            ax=ax,
        )
        
        ax.set_xlabel("Predição")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de Confusão")
        
        if save:
            save_path = self.save_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Matriz de confusão salva em: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray, save: bool = True
    ) -> None:
        """
        Plota curva ROC.
        
        Args:
            y_true: Labels verdadeiros
            y_proba: Probabilidades
            save: Se True, salva a figura
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Taxa de Falsos Positivos")
        ax.set_ylabel("Taxa de Verdadeiros Positivos")
        ax.set_title("Curva ROC")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save:
            save_path = self.save_dir / "roc_curve.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Curva ROC salva em: {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(
        self, metrics: Dict[str, float], save: bool = True
    ) -> None:
        """
        Plota comparação de métricas.
        
        Args:
            metrics: Dicionário com métricas
            save: Se True, salva a figura
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color="skyblue", edgecolor="black")
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel("Valor")
        ax.set_title("Comparação de Métricas")
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.xticks(rotation=45, ha="right")
        
        if save:
            save_path = self.save_dir / "metrics_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Comparação de métricas salva em: {save_path}")
        
        plt.close()


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:
    """Avaliador completo do modelo."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.model_loader = ModelLoader()
        self.data_loader = EvaluationDataLoader()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
    
    @torch.no_grad()
    def evaluate(
        self,
        model_name: str = "model_magic_steps_dl",
        use_feast: bool = False,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Avalia o modelo treinado.
        
        Args:
            model_name: Nome do modelo
            use_feast: Se True, carrega dados do Feast
            generate_plots: Se True, gera visualizações
        
        Returns:
            Dicionário com resultados da avaliação
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO AVALIAÇÃO DO MODELO")
        self.logger.info("=" * 60)
        
        # 1. Carregar modelo
        model, checkpoint = self.model_loader.load_model(model_name)
        
        # 2. Carregar dados
        X, y_true = self.data_loader.load_dataset(use_feast=use_feast)
        
        # 3. Fazer predições
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor)
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba >= 0.5).astype(int)
        
        # 4. Calcular métricas
        metrics = self.metrics_calculator.calculate_metrics(y_true, y_pred, y_proba)
        self.metrics_calculator.print_metrics(metrics)
        
        # 5. Matriz de confusão
        cm = self.metrics_calculator.get_confusion_matrix(y_true, y_pred)
        self.metrics_calculator.print_confusion_matrix(cm)
        
        # 6. Relatório de classificação
        report = self.metrics_calculator.get_classification_report(y_true, y_pred)
        self.logger.info("\n📋 RELATÓRIO DE CLASSIFICAÇÃO")
        self.logger.info(f"\n{report}")
        
        # 7. Gerar visualizações
        if generate_plots:
            self.visualizer.plot_confusion_matrix(cm)
            self.visualizer.plot_roc_curve(y_true, y_proba)
            self.visualizer.plot_metrics_comparison(metrics)
        
        # 8. Salvar resultados
        results = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "model_params": checkpoint["best_params"],
        }
        
        self._save_results(results, model_name)
        
        self.logger.info("=" * 60)
        self.logger.info("AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
        self.logger.info("=" * 60)
        
        return results
    
    def _save_results(self, results: Dict[str, Any], model_name: str) -> None:
        """
        Salva resultados da avaliação.
        
        Args:
            results: Resultados da avaliação
            model_name: Nome do modelo
        """
        import json
        
        results_path = ARTIFACTS_DIR / f"{model_name}_evaluation_results.json"
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Resultados salvos em: {results_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Função principal."""
    evaluator = Evaluator()
    results = evaluator.evaluate(
        model_name="model_magic_steps_dl",
        use_feast=False,
        generate_plots=True,
    )
    
    print("\n✅ Avaliação finalizada!")


if __name__ == "__main__":
    print("📈 Avaliando modelo treinado - Magic Steps")
    main()