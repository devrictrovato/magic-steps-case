"""
Pipeline de avaliação do modelo treinado.

Gera as seguintes figuras em out/imgs/:
  01_confusion_matrix.png         — Matriz de confusão normalizada + absoluta
  02_roc_curves_ovr.png           — Curvas ROC One-vs-Rest para cada classe
  03_precision_recall_curves.png  — Curvas Precision-Recall por classe
  04_metrics_radar.png            — Radar com as principais métricas
  05_class_probabilities.png      — Distribuição de probabilidade por classe real
  06_calibration_curve.png        — Calibração do modelo por classe
  07_per_class_metrics.png        — Precision / Recall / F1 por classe (barras)
  08_prediction_confidence.png    — Distribuição da confiança (max prob) predita
  09_error_analysis.png           — Onde o modelo erra (heatmap erros por classe)
  10_summary_dashboard.png        — Dashboard resumo com KPIs + 3 gráficos
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
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
    precision_recall_curve,
    average_precision_score,
    top_k_accuracy_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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

# Paleta consistente para as 3 classes
CLASS_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#27AE60"}
CLASS_LABELS  = ["atraso", "neutro", "avanco"]
N_CLASSES     = 3


# ============================================================
# MODEL (MESMA ARQUITETURA DO TREINO)
# ============================================================

class MagicStepsNet(nn.Module):
    """Rede Neural para classificacao multiclasse (defasagem)."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        dropout: float,
        num_classes: int = 3,
    ):
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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.model_registry = ModelRegistry()

    def load_model(
        self, model_name: str = "model_magic_steps_dl"
    ) -> tuple[nn.Module, Dict[str, Any]]:
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

        self.logger.info(f"Carregando modelo de: {model_path}")
        checkpoint  = torch.load(model_path, map_location=DEVICE)
        num_classes = checkpoint.get("num_classes", N_CLASSES)

        model = MagicStepsNet(
            input_dim=checkpoint["input_dim"],
            hidden_layers=checkpoint["best_params"]["hidden_layers"],
            dropout=checkpoint["best_params"]["dropout"],
            num_classes=num_classes,
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.logger.info("Modelo carregado com sucesso")
        return model, checkpoint


# ============================================================
# DATA LOADER
# ============================================================

class EvaluationDataLoader:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def load_dataset(
        self, use_feast: bool = False
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        self.logger.info("Carregando dataset para avaliacao...")
        df = load_features_for_training(use_feast=use_feast)
        target_col = data_config.target_column

        X = df.drop(columns=[target_col]).values.astype("float32")
        y = df[target_col].values.astype("int64")

        self.logger.info(f"Dataset: X={X.shape}, y={y.shape}")
        dist = dict(zip(*np.unique(y, return_counts=True)))
        self.logger.info(f"Distribuicao: {dist}")
        return X, y, CLASS_LABELS


# ============================================================
# METRICS CALCULATOR
# ============================================================

class MetricsCalculator:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Conjunto completo de metricas para classificacao ternaria.
        """
        m: Dict[str, float] = {}

        m["accuracy"]           = accuracy_score(y_true, y_pred)
        m["balanced_accuracy"]  = balanced_accuracy_score(y_true, y_pred)
        m["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        m["recall_weighted"]    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        m["f1_weighted"]        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        m["f1_macro"]           = f1_score(y_true, y_pred, average="macro", zero_division=0)
        m["cohen_kappa"]        = cohen_kappa_score(y_true, y_pred)
        m["mcc"]                = matthews_corrcoef(y_true, y_pred)
        m["log_loss"]           = log_loss(y_true, y_proba)
        m["top2_accuracy"]      = top_k_accuracy_score(y_true, y_proba, k=2)

        try:
            m["roc_auc_ovr_weighted"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
            m["roc_auc_ovr_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError as e:
            self.logger.warning(f"ROC-AUC nao calculado: {e}")
            m["roc_auc_ovr_weighted"] = float("nan")
            m["roc_auc_ovr_macro"]    = float("nan")

        p_per = precision_score(y_true, y_pred, average=None, zero_division=0)
        r_per = recall_score(y_true, y_pred, average=None, zero_division=0)
        f_per = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, lbl in enumerate(CLASS_LABELS):
            if i < len(p_per):
                m[f"precision_{lbl}"] = float(p_per[i])
                m[f"recall_{lbl}"]    = float(r_per[i])
                m[f"f1_{lbl}"]        = float(f_per[i])

        return m

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("METRICAS DE AVALIACAO")
        self.logger.info("=" * 60)
        for name, val in metrics.items():
            self.logger.info(f"  {name:<32}: {val:.4f}")
        self.logger.info("=" * 60)

    def get_confusion_matrix(self, y_true, y_pred) -> np.ndarray:
        return confusion_matrix(y_true, y_pred)

    def print_confusion_matrix(self, cm: np.ndarray) -> None:
        self.logger.info("\nMATRIZ DE CONFUSAO")
        self.logger.info(f"\n{cm}")
        for i, row in enumerate(cm):
            self.logger.info(f"  {CLASS_LABELS[i]:>6}: {row.tolist()}")

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> str:
        return classification_report(
            y_true, y_pred,
            target_names=class_names or CLASS_LABELS,
            zero_division=0,
        )


# ============================================================
# VISUALIZER  — 10 figuras exportadas para out/imgs/
# ============================================================

class Visualizer:
    """Gera todas as figuras de avaliacao e salva em out/imgs/."""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or (ARTIFACTS_DIR / "imgs")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__)
        plt.style.use("seaborn-v0_8-whitegrid")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"  salvo: {filename}")

    # ── 01 ── Confusion matrix (absoluta + normalizada) ─────

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Matriz de Confusao", fontsize=14, fontweight="bold")

        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2f"],
            ["Contagens Absolutas", "Normalizada por Linha (Recall)"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax,
            )
            ax.set_xlabel("Predicao", fontsize=11)
            ax.set_ylabel("Real", fontsize=11)
            ax.set_title(title, fontsize=11)

        plt.tight_layout()
        self._save(fig, "01_confusion_matrix.png")

    # ── 02 ── ROC One-vs-Rest por classe ────────────────────

    def plot_roc_curves_ovr(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curvas ROC - One-vs-Rest por Classe",
                     fontsize=14, fontweight="bold")

        for i, (ax, name) in enumerate(zip(axes, class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            color = CLASS_COLORS[i]
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_xlabel("FPR", fontsize=10)
            if i == 0:
                ax.set_ylabel("TPR", fontsize=10)
            ax.set_title(f"Classe: {name}", fontsize=11)
            ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        self._save(fig, "02_roc_curves_ovr.png")

    # ── 03 ── Precision-Recall por classe ───────────────────

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curvas Precision-Recall por Classe",
                     fontsize=14, fontweight="bold")

        for i, (ax, name) in enumerate(zip(axes, class_names)):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap       = average_precision_score(y_bin[:, i], y_proba[:, i])
            baseline = float(y_bin[:, i].mean())
            color = CLASS_COLORS[i]

            ax.plot(rec, prec, color=color, lw=2.5, label=f"AP = {ap:.3f}")
            ax.axhline(baseline, color="gray", lw=1.2, linestyle="--",
                       label=f"Baseline = {baseline:.2f}")
            ax.fill_between(rec, prec, alpha=0.08, color=color)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.set_xlabel("Recall", fontsize=10)
            if i == 0:
                ax.set_ylabel("Precision", fontsize=10)
            ax.set_title(f"Classe: {name}", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        self._save(fig, "03_precision_recall_curves.png")

    # ── 04 ── Radar de metricas globais ─────────────────────

    def plot_metrics_radar(self, metrics: Dict[str, float]) -> None:
        keys   = ["accuracy", "balanced_accuracy", "precision_weighted",
                  "recall_weighted", "f1_weighted", "roc_auc_ovr_weighted",
                  "top2_accuracy"]
        labels = ["Accuracy", "Bal.Acc", "Precision", "Recall",
                  "F1", "ROC-AUC", "Top-2"]
        values = [max(0.0, metrics.get(k, 0.0)) for k in keys]
        N      = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        vals   = values + [values[0]]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.plot(angles, vals, "o-", lw=2, color="#2980B9")
        ax.fill(angles, vals, alpha=0.18, color="#2980B9")

        for angle, val in zip(angles[:-1], values):
            ax.annotate(f"{val:.2f}", xy=(angle, val),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=9, color="#2C3E50")

        ax.set_title("Visao Geral das Metricas", fontsize=13,
                     fontweight="bold", pad=22)
        self._save(fig, "04_metrics_radar.png")

    # ── 05 ── Distribuicao de probabilidades por classe real ─

    def plot_class_probabilities(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        fig, axes = plt.subplots(N_CLASSES, N_CLASSES, figsize=(13, 10))
        fig.suptitle(
            "P(classe predita) agrupada por Classe Real",
            fontsize=13, fontweight="bold",
        )

        for true_cls in range(N_CLASSES):
            mask = y_true == true_cls
            for pred_cls in range(N_CLASSES):
                ax = axes[true_cls][pred_cls]
                vals  = y_proba[mask, pred_cls]
                color = CLASS_COLORS[pred_cls]
                ax.hist(vals, bins=25, color=color, alpha=0.75, edgecolor="white")
                ax.axvline(vals.mean(), color="black", lw=1.5, linestyle="--",
                           label=f"u={vals.mean():.2f}")
                ax.set_xlim(0, 1)
                ax.legend(fontsize=8)
                if true_cls == 0:
                    ax.set_title(f"P({class_names[pred_cls]})", fontsize=10)
                if pred_cls == 0:
                    ax.set_ylabel(f"Real: {class_names[true_cls]}", fontsize=10)

        plt.tight_layout()
        self._save(fig, "05_class_probabilities.png")

    # ── 06 ── Calibration curve (Reliability Diagram) ───────

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
        n_bins: int = 10,
    ) -> None:
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curva de Calibracao (Reliability Diagram) por Classe",
                     fontsize=13, fontweight="bold")

        for i, (ax, name) in enumerate(zip(axes, class_names)):
            prob_cls = y_proba[:, i]
            true_cls = y_bin[:, i]
            color    = CLASS_COLORS[i]

            bins     = np.linspace(0, 1, n_bins + 1)
            bin_idx  = np.clip(np.digitize(prob_cls, bins) - 1, 0, n_bins - 1)
            frac_pos = np.array([
                true_cls[bin_idx == b].mean() if (bin_idx == b).sum() > 0 else np.nan
                for b in range(n_bins)
            ])
            mean_pred = np.array([
                prob_cls[bin_idx == b].mean() if (bin_idx == b).sum() > 0 else np.nan
                for b in range(n_bins)
            ])
            counts = np.array([(bin_idx == b).sum() for b in range(n_bins)])
            valid  = ~np.isnan(frac_pos)

            ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfeita")
            ax.plot(mean_pred[valid], frac_pos[valid], "o-",
                    color=color, lw=2, ms=6, label=name)

            ax2 = ax.twinx()
            ax2.bar((bins[:-1] + bins[1:]) / 2, counts,
                    width=0.08, alpha=0.18, color=color)
            ax2.set_ylabel("Contagem", fontsize=9)
            ax2.set_ylim(0, max(counts.max() * 4, 1))

            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Probabilidade predita", fontsize=10)
            if i == 0:
                ax.set_ylabel("Fracao de positivos", fontsize=10)
            ax.set_title(f"Classe: {name}", fontsize=11)
            ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        self._save(fig, "06_calibration_curve.png")

    # ── 07 ── Per-class Precision / Recall / F1 ─────────────

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, float],
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        metric_types = ["precision", "recall", "f1"]
        bar_colors   = ["#3498DB", "#E67E22", "#2ECC71"]
        x     = np.arange(N_CLASSES)
        width = 0.25

        fig, ax = plt.subplots(figsize=(9, 5))

        for j, (mt, bc) in enumerate(zip(metric_types, bar_colors)):
            vals = [metrics.get(f"{mt}_{lbl}", 0.0) for lbl in class_names]
            bars = ax.bar(x + j * width, vals, width,
                          label=mt.capitalize(), color=bc,
                          alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title("Precision / Recall / F1 por Classe",
                     fontsize=13, fontweight="bold")

        f1_macro = metrics.get("f1_macro", 0.0)
        ax.axhline(f1_macro, color="red", lw=1.5, linestyle="--",
                   label=f"F1 macro = {f1_macro:.2f}")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

        plt.tight_layout()
        self._save(fig, "07_per_class_metrics.png")

    # ── 08 ── Distribuicao de confianca ─────────────────────

    def plot_prediction_confidence(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        max_prob = y_proba.max(axis=1)
        correct  = y_pred == y_true

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Distribuicao da Confianca da Predicao",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.hist(max_prob[correct],  bins=30, alpha=0.65, color="#27AE60",
                label=f"Corretos (n={correct.sum()})", edgecolor="white")
        ax.hist(max_prob[~correct], bins=30, alpha=0.65, color="#E74C3C",
                label=f"Erros (n={(~correct).sum()})", edgecolor="white")
        ax.set_xlabel("Max P(classe)", fontsize=11)
        ax.set_ylabel("Frequencia", fontsize=11)
        ax.set_title("Confianca: acertos vs erros", fontsize=11)
        ax.legend(fontsize=10)
        ax.axvline(0.5, color="gray", lw=1.2, linestyle="--")

        ax2 = axes[1]
        for cls, name in enumerate(class_names):
            mask = y_pred == cls
            if mask.sum() > 0:
                ax2.hist(max_prob[mask], bins=25, alpha=0.55,
                         color=CLASS_COLORS[cls], label=name, edgecolor="white")
        ax2.set_xlabel("Max P(classe)", fontsize=11)
        ax2.set_ylabel("Frequencia", fontsize=11)
        ax2.set_title("Confianca por Classe Predita", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.axvline(0.5, color="gray", lw=1.2, linestyle="--")

        plt.tight_layout()
        self._save(fig, "08_prediction_confidence.png")

    # ── 09 ── Error analysis heatmap ────────────────────────

    def plot_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        error_matrix = cm.astype(float)
        np.fill_diagonal(error_matrix, 0)

        row_sums   = cm.sum(axis=1, keepdims=True)
        error_norm = np.where(row_sums > 0, error_matrix / row_sums, 0.0)

        cmap = LinearSegmentedColormap.from_list("wr", ["#FFFFFF", "#E74C3C"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Analise de Erros (diagonal zerada)",
                     fontsize=13, fontweight="bold")

        for ax, data, fmt, title in zip(
            axes,
            [error_matrix, error_norm],
            [".0f", ".2%"],
            ["Erros Absolutos", "Taxa de Erro por Classe Real"],
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, annot_kws={"size": 11},
            )
            ax.set_xlabel("Predicao (errada)", fontsize=11)
            ax.set_ylabel("Real", fontsize=11)
            ax.set_title(title, fontsize=11)

        plt.tight_layout()
        self._save(fig, "09_error_analysis.png")

    # ── 10 ── Summary dashboard ─────────────────────────────

    def plot_summary_dashboard(
        self,
        metrics: Dict[str, float],
        cm: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Magic Steps - Avaliacao do Modelo de Defasagem",
                     fontsize=15, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.42)

        # KPI boxes (linha 0)
        kpi_items = [
            ("Accuracy",        metrics.get("accuracy", 0),         "#2980B9"),
            ("Balanced Acc.",   metrics.get("balanced_accuracy", 0), "#8E44AD"),
            ("F1 Weighted",     metrics.get("f1_weighted", 0),       "#27AE60"),
            ("Cohen Kappa",     metrics.get("cohen_kappa", 0),       "#E67E22"),
        ]
        for col, (label, value, color) in enumerate(kpi_items):
            ax = fig.add_subplot(gs[0, col])
            ax.set_facecolor(color)
            ax.text(0.5, 0.56, f"{value:.3f}", ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white",
                    transform=ax.transAxes)
            ax.text(0.5, 0.16, label, ha="center", va="center",
                    fontsize=11, color="white", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Matriz de confusao normalizada (linha 1, col 0-1)
        ax_cm = fig.add_subplot(gs[1, :2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax_cm, cbar=False,
        )
        ax_cm.set_xlabel("Predicao", fontsize=10)
        ax_cm.set_ylabel("Real", fontsize=10)
        ax_cm.set_title("Matriz de Confusao (normalizada)", fontsize=11)

        # F1 por classe (linha 1, col 2)
        ax_f1 = fig.add_subplot(gs[1, 2])
        f1_vals = [metrics.get(f"f1_{lbl}", 0.0) for lbl in class_names]
        bars = ax_f1.barh(
            class_names, f1_vals,
            color=[CLASS_COLORS[i] for i in range(N_CLASSES)],
            edgecolor="white",
        )
        for bar, val in zip(bars, f1_vals):
            ax_f1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                       f"{val:.2f}", va="center", fontsize=10)
        ax_f1.set_xlim(0, 1.12)
        ax_f1.set_xlabel("F1 Score", fontsize=10)
        ax_f1.set_title("F1 por Classe", fontsize=11)
        ax_f1.grid(axis="x", alpha=0.3)

        # Real vs Predito (linha 1, col 3)
        ax_dist = fig.add_subplot(gs[1, 3])
        _, real_counts = np.unique(y_true, return_counts=True)
        pred_full = np.zeros(N_CLASSES, dtype=int)
        for u, c in zip(*np.unique(y_pred, return_counts=True)):
            pred_full[int(u)] = c

        x = np.arange(N_CLASSES)
        w = 0.35
        ax_dist.bar(x - w / 2, real_counts, w, label="Real",
                    color="#3498DB", alpha=0.82, edgecolor="white")
        ax_dist.bar(x + w / 2, pred_full, w, label="Predito",
                    color="#E67E22", alpha=0.82, edgecolor="white")
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(class_names, fontsize=10)
        ax_dist.set_ylabel("Amostras", fontsize=10)
        ax_dist.set_title("Real vs Predito", fontsize=11)
        ax_dist.legend(fontsize=9)
        ax_dist.grid(axis="y", alpha=0.3)

        self._save(fig, "10_summary_dashboard.png")

    # ── orquestrador ────────────────────────────────────────

    def plot_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        cm: np.ndarray,
        metrics: Dict[str, float],
        class_names: List[str] = CLASS_LABELS,
    ) -> None:
        self.logger.info(f"\nSalvando figuras em: {self.save_dir}")
        self.plot_confusion_matrix(cm, class_names)
        self.plot_roc_curves_ovr(y_true, y_proba, class_names)
        self.plot_precision_recall_curves(y_true, y_proba, class_names)
        self.plot_metrics_radar(metrics)
        self.plot_class_probabilities(y_true, y_proba, class_names)
        self.plot_calibration_curve(y_true, y_proba, class_names)
        self.plot_per_class_metrics(metrics, class_names)
        self.plot_prediction_confidence(y_true, y_pred, y_proba, class_names)
        self.plot_error_analysis(y_true, y_pred, class_names)
        self.plot_summary_dashboard(metrics, cm, y_true, y_pred, class_names)
        self.logger.info("Todas as figuras salvas.")


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:
    """Avaliador completo do modelo."""

    def __init__(self):
        self.logger       = setup_logger(self.__class__.__name__)
        self.model_loader = ModelLoader()
        self.data_loader  = EvaluationDataLoader()
        self.metrics_calc = MetricsCalculator()
        self.visualizer   = Visualizer()

    @torch.no_grad()
    def evaluate(
        self,
        model_name: str = "model_magic_steps_dl",
        use_feast: bool = False,
        generate_plots: bool = True,
    ) -> Dict[str, Any]:
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO AVALIACAO DO MODELO")
        self.logger.info("=" * 60)

        # 1. Modelo
        model, checkpoint = self.model_loader.load_model(model_name)

        # 2. Dados
        X, y_true, class_names = self.data_loader.load_dataset(use_feast=use_feast)

        # 3. Inferencia
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits   = model(X_tensor)
        y_proba  = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred   = np.argmax(y_proba, axis=1)

        # 4. Metricas
        metrics = self.metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
        self.metrics_calc.print_metrics(metrics)

        # 5. Matriz de confusao
        cm = self.metrics_calc.get_confusion_matrix(y_true, y_pred)
        self.metrics_calc.print_confusion_matrix(cm)

        # 6. Relatorio de classificacao
        report = self.metrics_calc.get_classification_report(
            y_true, y_pred, class_names=class_names
        )
        self.logger.info("\nRELATORIO DE CLASSIFICACAO")
        self.logger.info(f"\n{report}")

        # 7. Visualizacoes
        if generate_plots:
            self.visualizer.plot_all(
                y_true, y_pred, y_proba, cm, metrics, class_names
            )

        # 8. Salvar JSON
        results = {
            "metrics": {
                k: (v if not np.isnan(v) else None)
                for k, v in metrics.items()
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "model_params": checkpoint["best_params"],
            "class_labels": class_names,
        }
        self._save_results(results, model_name)

        self.logger.info("=" * 60)
        self.logger.info("AVALIACAO CONCLUIDA COM SUCESSO!")
        self.logger.info("=" * 60)

        return results

    def _save_results(self, results: Dict[str, Any], model_name: str) -> None:
        import json
        path = ARTIFACTS_DIR / f"{model_name}_evaluation_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Resultados salvos em: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    evaluator = Evaluator()
    evaluator.evaluate(
        model_name="model_magic_steps_dl",
        use_feast=False,
        generate_plots=True,
    )
    print("\nAvaliacao finalizada!")


if __name__ == "__main__":
    print("Avaliando modelo treinado - Magic Steps")
    main()