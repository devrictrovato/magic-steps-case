"""
Pipeline de avaliação do modelo — Magic Steps MLOps.

Métricas geradas:
  • Conjunto completo de métricas de classificação ternária
  • Cross-validation estratificada (K-Fold) com análise de overfitting
  • Gap train/val por fold para detectar overfitting
  • Figuras salvas em out/imgs/:
      01_confusion_matrix.png
      02_roc_curves_ovr.png
      03_precision_recall_curves.png
      04_metrics_radar.png
      05_class_probabilities.png
      06_calibration_curve.png
      07_per_class_metrics.png
      08_prediction_confidence.png
      09_error_analysis.png
      10_summary_dashboard.png
      11_cv_scores.png          ← cross-validation scores por fold
      12_train_val_gap.png      ← análise de overfitting (gap treino/val)
      13_learning_curves.png    ← curva de aprendizado (bias/variance)
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    top_k_accuracy_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from settings import data_config, MODELS_DIR, ARTIFACTS_DIR, settings
from utils import setup_logger, ModelRegistry
from feature_engineering import load_features_for_training
from db import ensure_schema, update_model_run, log_monitoring_event, list_model_runs


# ============================================================
# CONSTANTES
# ============================================================

logger = setup_logger(__name__, "evaluation.log")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_COLORS = {0: "#E74C3C", 1: "#F39C12", 2: "#27AE60"}
CLASS_LABELS = ["atraso", "neutro", "avanco"]
N_CLASSES    = 3


# ============================================================
# ARQUITETURA (espelho do train.py)
# ============================================================

class MagicStepsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list,
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
        return self.net(x)


# ============================================================
# MODEL LOADER
# ============================================================

class ModelLoader:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.model_registry = ModelRegistry()

    def load_model(self, model_name: str = "model_magic_steps_dl"):
        model_path = MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
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
        self.logger.info("Modelo carregado.")
        return model, checkpoint


# ============================================================
# DATA LOADER
# ============================================================

class EvaluationDataLoader:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def load_dataset(self, use_feast: bool = False):
        self.logger.info("Carregando dataset…")
        df = load_features_for_training(use_db=True)
        target_col = data_config.target_column
        X = df.drop(columns=[target_col]).values.astype("float32")
        y = df[target_col].values.astype("int64")
        self.logger.info(f"Dataset: X={X.shape}, y={y.shape}")
        return X, y, CLASS_LABELS


# ============================================================
# METRICS CALCULATOR
# ============================================================

class MetricsCalculator:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def calculate_metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        m: Dict[str, float] = {}
        m["accuracy"]            = accuracy_score(y_true, y_pred)
        m["balanced_accuracy"]   = balanced_accuracy_score(y_true, y_pred)
        m["precision_weighted"]  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        m["recall_weighted"]     = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        m["f1_weighted"]         = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        m["f1_macro"]            = f1_score(y_true, y_pred, average="macro", zero_division=0)
        m["cohen_kappa"]         = cohen_kappa_score(y_true, y_pred)
        m["mcc"]                 = matthews_corrcoef(y_true, y_pred)
        m["log_loss"]            = log_loss(y_true, y_proba)
        m["top2_accuracy"]       = top_k_accuracy_score(y_true, y_proba, k=2)
        try:
            m["roc_auc_ovr_weighted"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            m["roc_auc_ovr_macro"]    = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError as e:
            self.logger.warning(f"ROC-AUC não calculado: {e}")
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
        self.logger.info("MÉTRICAS DE AVALIAÇÃO")
        self.logger.info("=" * 60)
        for name, val in metrics.items():
            self.logger.info(f"  {name:<32}: {val:.4f}")

    def get_confusion_matrix(self, y_true, y_pred): return confusion_matrix(y_true, y_pred)

    def get_classification_report(self, y_true, y_pred, class_names=None) -> str:
        return classification_report(y_true, y_pred,
                                     target_names=class_names or CLASS_LABELS, zero_division=0)


# ============================================================
# CROSS-VALIDATION + OVERFIT ANALYSIS
# ============================================================

class CrossValidator:
    """
    Executa K-Fold estratificado e mede:
      • Métricas por fold (val)
      • Gap treino/val por fold  → detecta overfitting
      • Bias-variance summary
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.logger   = setup_logger(self.__class__.__name__)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Retorna dict com:
          fold_results  → lista de métricas por fold
          cv_summary    → média ± std das métricas
          overfit_analysis → gap treino/val, conclusão de overfitting
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        best_params = checkpoint["best_params"]
        num_classes = checkpoint.get("num_classes", N_CLASSES)

        fold_results = []

        self.logger.info(f"\n🔁 Cross-validation {self.n_splits}-fold estratificado…")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # ── treinar modelo do zero neste fold ──────────────────────────
            model = MagicStepsNet(
                input_dim=X_tr.shape[1],
                hidden_layers=best_params["hidden_layers"],
                dropout=best_params["dropout"],
                num_classes=num_classes,
            ).to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

            from torch.utils.data import DataLoader, TensorDataset
            train_ds = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)

            # treinar por epochs / 2 (avaliação rápida)
            n_epochs = max(10, best_params["epochs"] // 2)
            train_losses = []
            model.train()
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_losses.append(epoch_loss / len(train_loader))

            # ── métricas de treino (último epoch) ─────────────────────────
            model.eval()
            with torch.no_grad():
                X_tr_t  = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
                logits_tr = model(X_tr_t)
                proba_tr  = torch.softmax(logits_tr, dim=1).cpu().numpy()
                pred_tr   = np.argmax(proba_tr, axis=1)
                train_acc  = accuracy_score(y_tr, pred_tr)
                train_f1   = f1_score(y_tr, pred_tr, average="weighted", zero_division=0)
                train_loss = train_losses[-1]

                # ── métricas de validação ──────────────────────────────────
                X_val_t   = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                logits_val = model(X_val_t)
                proba_val  = torch.softmax(logits_val, dim=1).cpu().numpy()
                pred_val   = np.argmax(proba_val, axis=1)

                val_acc = accuracy_score(y_val, pred_val)
                val_f1  = f1_score(y_val, pred_val, average="weighted", zero_division=0)
                val_loss_val = criterion(
                    torch.tensor(logits_val.cpu().numpy()),
                    torch.tensor(y_val, dtype=torch.long),
                ).item()

                try:
                    val_auc = roc_auc_score(y_val, proba_val, multi_class="ovr", average="weighted")
                except ValueError:
                    val_auc = float("nan")

            gap_acc  = train_acc  - val_acc
            gap_f1   = train_f1   - val_f1
            gap_loss = train_loss - val_loss_val   # negativo = underfitting

            fold_results.append({
                "fold":       fold_idx,
                "train_acc":  round(train_acc, 4),
                "val_acc":    round(val_acc, 4),
                "train_f1":   round(train_f1, 4),
                "val_f1":     round(val_f1, 4),
                "train_loss": round(train_loss, 4),
                "val_loss":   round(val_loss_val, 4),
                "val_auc":    round(val_auc, 4) if not math.isnan(val_auc) else None,
                "gap_acc":    round(gap_acc, 4),
                "gap_f1":     round(gap_f1, 4),
                "gap_loss":   round(gap_loss, 4),
            })

            self.logger.info(
                f"  Fold {fold_idx}/{self.n_splits} — "
                f"val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
                f"gap_acc={gap_acc:+.3f}  gap_f1={gap_f1:+.3f}"
            )

        # ── summary ───────────────────────────────────────────────────────
        df_folds = pd.DataFrame(fold_results)
        cv_summary: Dict[str, Any] = {}
        for col in ["val_acc", "val_f1", "val_auc", "gap_acc", "gap_f1"]:
            vals = df_folds[col].dropna().values
            cv_summary[f"{col}_mean"] = float(np.mean(vals))
            cv_summary[f"{col}_std"]  = float(np.std(vals))

        # ── análise de overfitting ────────────────────────────────────────
        avg_gap_acc = cv_summary["gap_acc_mean"]
        avg_gap_f1  = cv_summary["gap_f1_mean"]
        std_val_acc = cv_summary["val_acc_std"]

        if avg_gap_acc > 0.12 or avg_gap_f1 > 0.12:
            overfit_level = "alto"
            overfit_msg   = (
                "O modelo apresenta overfitting significativo: o gap treino/val "
                f"médio é {avg_gap_acc:.2%} em accuracy e {avg_gap_f1:.2%} em F1. "
                "Considere aumentar dropout, reduzir hidden_layers, "
                "adicionar regularização L2 ou coletar mais dados."
            )
        elif avg_gap_acc > 0.06 or avg_gap_f1 > 0.06:
            overfit_level = "moderado"
            overfit_msg   = (
                "Overfitting moderado detectado "
                f"(gap acc={avg_gap_acc:.2%}, gap F1={avg_gap_f1:.2%}). "
                "Monitorar com atenção; ajustes menores podem ser benéficos."
            )
        elif cv_summary["val_acc_mean"] < 0.55:
            overfit_level = "underfitting"
            overfit_msg   = (
                f"Possível underfitting: val_acc médio={cv_summary['val_acc_mean']:.2%}. "
                "Considere aumentar a capacidade do modelo ou reduzir dropout."
            )
        else:
            overfit_level = "saudável"
            overfit_msg   = (
                f"Sem evidências de overfitting severo "
                f"(gap acc={avg_gap_acc:.2%}, gap F1={avg_gap_f1:.2%}). "
                "O modelo generaliza de forma adequada."
            )

        overfit_analysis = {
            "level":              overfit_level,
            "message":            overfit_msg,
            "avg_gap_accuracy":   round(avg_gap_acc, 4),
            "avg_gap_f1":         round(avg_gap_f1, 4),
            "val_acc_stability":  round(std_val_acc, 4),
            "recommendation":     overfit_msg,
        }

        self.logger.info(f"\n📊 CV Summary: {cv_summary}")
        self.logger.info(f"⚖️  Overfitting: [{overfit_level.upper()}] {overfit_msg}")

        return {
            "fold_results":      fold_results,
            "cv_summary":        cv_summary,
            "overfit_analysis":  overfit_analysis,
        }


# ============================================================
# VISUALIZER
# ============================================================

class Visualizer:
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

    # ── 01 confusion matrix ──────────────────────────────────────────────

    def plot_confusion_matrix(self, cm, class_names=CLASS_LABELS):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Matriz de Confusão", fontsize=14, fontweight="bold")
        for ax, data, fmt, title in zip(axes, [cm, cm_norm], ["d", ".2f"],
                                        ["Contagens Absolutas", "Normalizada (Recall)"]):
            sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.5, ax=ax)
            ax.set_xlabel("Predição"); ax.set_ylabel("Real"); ax.set_title(title)
        plt.tight_layout()
        self._save(fig, "01_confusion_matrix.png")

    # ── 02 ROC OvR ───────────────────────────────────────────────────────

    def plot_roc_curves_ovr(self, y_true, y_proba, class_names=CLASS_LABELS):
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curvas ROC — One-vs-Rest", fontsize=14, fontweight="bold")
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            color = CLASS_COLORS[i]
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_xlabel("FPR"); ax.set_title(f"Classe: {name}"); ax.legend(loc="lower right")
            if i == 0: ax.set_ylabel("TPR")
        plt.tight_layout()
        self._save(fig, "02_roc_curves_ovr.png")

    # ── 03 Precision-Recall ───────────────────────────────────────────────

    def plot_precision_recall_curves(self, y_true, y_proba, class_names=CLASS_LABELS):
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curvas Precision-Recall", fontsize=14, fontweight="bold")
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_bin[:, i], y_proba[:, i])
            baseline = float(y_bin[:, i].mean())
            color = CLASS_COLORS[i]
            ax.plot(rec, prec, color=color, lw=2.5, label=f"AP = {ap:.3f}")
            ax.axhline(baseline, color="gray", lw=1.2, linestyle="--",
                       label=f"Baseline = {baseline:.2f}")
            ax.fill_between(rec, prec, alpha=0.08, color=color)
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
            ax.set_xlabel("Recall"); ax.set_title(f"Classe: {name}"); ax.legend(loc="upper right")
            if i == 0: ax.set_ylabel("Precision")
        plt.tight_layout()
        self._save(fig, "03_precision_recall_curves.png")

    # ── 04 Radar ─────────────────────────────────────────────────────────

    def plot_metrics_radar(self, metrics: Dict[str, float]):
        keys   = ["accuracy","balanced_accuracy","precision_weighted",
                  "recall_weighted","f1_weighted","roc_auc_ovr_weighted","top2_accuracy"]
        labels = ["Accuracy","Bal.Acc","Precision","Recall","F1","ROC-AUC","Top-2"]
        values = [max(0.0, metrics.get(k, 0.0)) for k in keys]
        N      = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        vals   = values + [values[0]]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.plot(angles, vals, "o-", lw=2, color="#2980B9")
        ax.fill(angles, vals, alpha=0.18, color="#2980B9")
        for angle, val in zip(angles[:-1], values):
            ax.annotate(f"{val:.2f}", xy=(angle, val), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_title("Visão Geral das Métricas", fontsize=13, fontweight="bold", pad=22)
        self._save(fig, "04_metrics_radar.png")

    # ── 05 Class probabilities ────────────────────────────────────────────

    def plot_class_probabilities(self, y_true, y_proba, class_names=CLASS_LABELS):
        fig, axes = plt.subplots(N_CLASSES, N_CLASSES, figsize=(13, 10))
        fig.suptitle("P(classe predita) por Classe Real", fontsize=13, fontweight="bold")
        for true_cls in range(N_CLASSES):
            mask = y_true == true_cls
            for pred_cls in range(N_CLASSES):
                ax = axes[true_cls][pred_cls]
                vals  = y_proba[mask, pred_cls]
                color = CLASS_COLORS[pred_cls]
                ax.hist(vals, bins=25, color=color, alpha=0.75, edgecolor="white")
                ax.axvline(vals.mean(), color="black", lw=1.5, linestyle="--",
                           label=f"μ={vals.mean():.2f}")
                ax.set_xlim(0, 1); ax.legend(fontsize=8)
                if true_cls == 0: ax.set_title(f"P({class_names[pred_cls]})", fontsize=10)
                if pred_cls == 0: ax.set_ylabel(f"Real: {class_names[true_cls]}", fontsize=10)
        plt.tight_layout()
        self._save(fig, "05_class_probabilities.png")

    # ── 06 Calibration ───────────────────────────────────────────────────

    def plot_calibration_curve(self, y_true, y_proba, class_names=CLASS_LABELS, n_bins=10):
        y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
        fig, axes = plt.subplots(1, N_CLASSES, figsize=(15, 5), sharey=True)
        fig.suptitle("Curva de Calibração (Reliability Diagram)", fontsize=13, fontweight="bold")
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            prob_cls = y_proba[:, i]; true_cls = y_bin[:, i]
            color    = CLASS_COLORS[i]
            bins     = np.linspace(0, 1, n_bins + 1)
            bin_idx  = np.clip(np.digitize(prob_cls, bins) - 1, 0, n_bins - 1)
            frac_pos = np.array([true_cls[bin_idx == b].mean() if (bin_idx == b).sum() > 0
                                 else np.nan for b in range(n_bins)])
            mean_pred = np.array([prob_cls[bin_idx == b].mean() if (bin_idx == b).sum() > 0
                                  else np.nan for b in range(n_bins)])
            counts = np.array([(bin_idx == b).sum() for b in range(n_bins)])
            valid  = ~np.isnan(frac_pos)
            ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfeita")
            ax.plot(mean_pred[valid], frac_pos[valid], "o-", color=color, lw=2, ms=6, label=name)
            ax2 = ax.twinx()
            ax2.bar((bins[:-1] + bins[1:]) / 2, counts, width=0.08, alpha=0.18, color=color)
            ax2.set_ylabel("Contagem", fontsize=9)
            ax2.set_ylim(0, max(counts.max() * 4, 1))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Probabilidade predita"); ax.set_title(f"Classe: {name}"); ax.legend(loc="upper left")
            if i == 0: ax.set_ylabel("Fração de positivos")
        plt.tight_layout()
        self._save(fig, "06_calibration_curve.png")

    # ── 07 Per-class bars ─────────────────────────────────────────────────

    def plot_per_class_metrics(self, metrics, class_names=CLASS_LABELS):
        metric_types = ["precision", "recall", "f1"]
        bar_colors   = ["#3498DB", "#E67E22", "#2ECC71"]
        x = np.arange(N_CLASSES); width = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, (mt, bc) in enumerate(zip(metric_types, bar_colors)):
            vals = [metrics.get(f"{mt}_{lbl}", 0.0) for lbl in class_names]
            bars = ax.bar(x + j * width, vals, width, label=mt.capitalize(),
                          color=bc, alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x + width); ax.set_xticklabels(class_names, fontsize=11)
        ax.set_ylim(0, 1.18); ax.set_ylabel("Score"); ax.legend(fontsize=10)
        f1_macro = metrics.get("f1_macro", 0.0)
        ax.axhline(f1_macro, color="red", lw=1.5, linestyle="--",
                   label=f"F1 macro = {f1_macro:.2f}")
        ax.set_title("Precision / Recall / F1 por Classe", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        self._save(fig, "07_per_class_metrics.png")

    # ── 08 Prediction confidence ─────────────────────────────────────────

    def plot_prediction_confidence(self, y_true, y_pred, y_proba, class_names=CLASS_LABELS):
        max_prob = y_proba.max(axis=1)
        correct  = y_pred == y_true
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Distribuição da Confiança da Predição", fontsize=13, fontweight="bold")
        ax = axes[0]
        ax.hist(max_prob[correct],  bins=30, alpha=0.65, color="#27AE60",
                label=f"Corretos (n={correct.sum()})", edgecolor="white")
        ax.hist(max_prob[~correct], bins=30, alpha=0.65, color="#E74C3C",
                label=f"Erros (n={(~correct).sum()})", edgecolor="white")
        ax.set_xlabel("Max P(classe)"); ax.set_ylabel("Frequência")
        ax.set_title("Confiança: acertos vs erros"); ax.legend(fontsize=10)
        ax.axvline(0.5, color="gray", lw=1.2, linestyle="--")
        ax2 = axes[1]
        for cls, name in enumerate(class_names):
            mask = y_pred == cls
            if mask.sum() > 0:
                ax2.hist(max_prob[mask], bins=25, alpha=0.55,
                         color=CLASS_COLORS[cls], label=name, edgecolor="white")
        ax2.set_xlabel("Max P(classe)"); ax2.set_ylabel("Frequência")
        ax2.set_title("Confiança por Classe Predita"); ax2.legend(fontsize=10)
        ax2.axvline(0.5, color="gray", lw=1.2, linestyle="--")
        plt.tight_layout()
        self._save(fig, "08_prediction_confidence.png")

    # ── 09 Error analysis ────────────────────────────────────────────────

    def plot_error_analysis(self, y_true, y_pred, class_names=CLASS_LABELS):
        cm = confusion_matrix(y_true, y_pred)
        error_matrix = cm.astype(float)
        np.fill_diagonal(error_matrix, 0)
        row_sums   = cm.sum(axis=1, keepdims=True)
        error_norm = np.where(row_sums > 0, error_matrix / row_sums, 0.0)
        cmap = LinearSegmentedColormap.from_list("wr", ["#FFFFFF", "#E74C3C"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Análise de Erros (diagonal zerada)", fontsize=13, fontweight="bold")
        for ax, data, fmt, title in zip(axes, [error_matrix, error_norm], [".0f", ".2%"],
                                        ["Erros Absolutos", "Taxa de Erro por Classe Real"]):
            sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.5, ax=ax, annot_kws={"size": 11})
            ax.set_xlabel("Predição (errada)"); ax.set_ylabel("Real"); ax.set_title(title)
        plt.tight_layout()
        self._save(fig, "09_error_analysis.png")

    # ── 10 Summary dashboard ─────────────────────────────────────────────

    def plot_summary_dashboard(self, metrics, cm, y_true, y_pred, class_names=CLASS_LABELS):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Magic Steps — Avaliação do Modelo de Defasagem",
                     fontsize=15, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.48, wspace=0.42)
        kpi_items = [
            ("Accuracy",       metrics.get("accuracy", 0),         "#2980B9"),
            ("Balanced Acc.",  metrics.get("balanced_accuracy", 0), "#8E44AD"),
            ("F1 Weighted",    metrics.get("f1_weighted", 0),       "#27AE60"),
            ("Cohen Kappa",    metrics.get("cohen_kappa", 0),       "#E67E22"),
        ]
        for col, (label, value, color) in enumerate(kpi_items):
            ax = fig.add_subplot(gs[0, col])
            ax.set_facecolor(color)
            ax.text(0.5, 0.56, f"{value:.3f}", ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white", transform=ax.transAxes)
            ax.text(0.5, 0.16, label, ha="center", va="center",
                    fontsize=11, color="white", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
        ax_cm = fig.add_subplot(gs[1, :2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, ax=ax_cm, cbar=False)
        ax_cm.set_xlabel("Predição"); ax_cm.set_ylabel("Real")
        ax_cm.set_title("Matriz de Confusão (normalizada)")
        ax_f1 = fig.add_subplot(gs[1, 2])
        f1_vals = [metrics.get(f"f1_{lbl}", 0.0) for lbl in class_names]
        bars = ax_f1.barh(class_names, f1_vals,
                          color=[CLASS_COLORS[i] for i in range(N_CLASSES)], edgecolor="white")
        for bar, val in zip(bars, f1_vals):
            ax_f1.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                       f"{val:.2f}", va="center", fontsize=10)
        ax_f1.set_xlim(0, 1.12); ax_f1.set_xlabel("F1 Score"); ax_f1.set_title("F1 por Classe")
        ax_f1.grid(axis="x", alpha=0.3)
        ax_dist = fig.add_subplot(gs[1, 3])
        _, real_counts = np.unique(y_true, return_counts=True)
        pred_full = np.zeros(N_CLASSES, dtype=int)
        for u, c in zip(*np.unique(y_pred, return_counts=True)):
            pred_full[int(u)] = c
        x = np.arange(N_CLASSES); w = 0.35
        ax_dist.bar(x - w/2, real_counts, w, label="Real",    color="#3498DB", alpha=0.82, edgecolor="white")
        ax_dist.bar(x + w/2, pred_full,   w, label="Predito", color="#E67E22", alpha=0.82, edgecolor="white")
        ax_dist.set_xticks(x); ax_dist.set_xticklabels(class_names)
        ax_dist.set_ylabel("Amostras"); ax_dist.set_title("Real vs Predito"); ax_dist.legend()
        ax_dist.grid(axis="y", alpha=0.3)
        self._save(fig, "10_summary_dashboard.png")

    # ── 11 CV scores por fold ─────────────────────────────────────────────

    def plot_cv_scores(self, cv_results: Dict[str, Any]):
        fold_results = cv_results["fold_results"]
        cv_summary   = cv_results["cv_summary"]
        df = pd.DataFrame(fold_results)
        folds = df["fold"].values

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Cross-Validation {len(fold_results)}-Fold — Scores por Fold",
            fontsize=13, fontweight="bold"
        )

        # accuracy
        ax = axes[0]
        ax.plot(folds, df["train_acc"], "o-", color="#3498DB", lw=2, ms=7, label="Treino")
        ax.plot(folds, df["val_acc"],   "s--", color="#E74C3C", lw=2, ms=7, label="Validação")
        ax.axhline(cv_summary["val_acc_mean"], color="#E74C3C", lw=1, linestyle=":", alpha=0.7,
                   label=f"Média val = {cv_summary['val_acc_mean']:.3f}")
        ax.fill_between(folds,
                        cv_summary["val_acc_mean"] - cv_summary["val_acc_std"],
                        cv_summary["val_acc_mean"] + cv_summary["val_acc_std"],
                        alpha=0.12, color="#E74C3C")
        ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy por Fold")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # F1
        ax = axes[1]
        ax.plot(folds, df["train_f1"], "o-", color="#3498DB", lw=2, ms=7, label="Treino")
        ax.plot(folds, df["val_f1"],   "s--", color="#E74C3C", lw=2, ms=7, label="Validação")
        ax.axhline(cv_summary["val_f1_mean"], color="#E74C3C", lw=1, linestyle=":", alpha=0.7,
                   label=f"Média val = {cv_summary['val_f1_mean']:.3f}")
        ax.fill_between(folds,
                        cv_summary["val_f1_mean"] - cv_summary["val_f1_std"],
                        cv_summary["val_f1_mean"] + cv_summary["val_f1_std"],
                        alpha=0.12, color="#E74C3C")
        ax.set_xlabel("Fold"); ax.set_ylabel("F1 Weighted"); ax.set_title("F1 por Fold")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # AUC
        ax = axes[2]
        auc_vals = df["val_auc"].fillna(0).values
        ax.bar(folds, auc_vals, color="#8E44AD", alpha=0.8, edgecolor="white")
        ax.axhline(cv_summary.get("val_auc_mean", 0), color="black", lw=1.5, linestyle="--",
                   label=f"Média = {cv_summary.get('val_auc_mean', 0):.3f}")
        ax.set_xlabel("Fold"); ax.set_ylabel("ROC-AUC (weighted)"); ax.set_title("AUC por Fold")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        self._save(fig, "11_cv_scores.png")

    # ── 12 Train/Val gap — overfitting ────────────────────────────────────

    def plot_train_val_gap(self, cv_results: Dict[str, Any]):
        fold_results    = cv_results["fold_results"]
        overfit_analysis = cv_results["overfit_analysis"]
        df = pd.DataFrame(fold_results)
        folds = df["fold"].values

        # cores por nível de overfitting
        level_color = {
            "alto":          "#E74C3C",
            "moderado":      "#E67E22",
            "saudável":      "#27AE60",
            "underfitting":  "#3498DB",
        }
        level = overfit_analysis["level"]
        color = level_color.get(level, "#95A5A6")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Análise de Overfitting — Nível: {level.upper()}",
            fontsize=13, fontweight="bold", color=color,
        )

        # gap accuracy
        ax = axes[0]
        gaps_acc = df["gap_acc"].values
        bars = ax.bar(folds, gaps_acc, color=[color if g > 0.06 else "#27AE60" for g in gaps_acc],
                      alpha=0.85, edgecolor="white")
        ax.axhline(0.06, color="orange", lw=1.5, linestyle="--", label="Limiar moderado (6%)")
        ax.axhline(0.12, color="red",    lw=1.5, linestyle="--", label="Limiar alto (12%)")
        ax.axhline(df["gap_acc"].mean(), color="black", lw=1.5, linestyle=":",
                   label=f"Média = {df['gap_acc'].mean():.3f}")
        for bar, val in zip(bars, gaps_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:+.3f}", ha="center", fontsize=9)
        ax.set_xlabel("Fold"); ax.set_ylabel("Gap (Treino − Val) Accuracy")
        ax.set_title("Gap de Accuracy por Fold"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

        # gap F1
        ax = axes[1]
        gaps_f1 = df["gap_f1"].values
        bars = ax.bar(folds, gaps_f1, color=[color if g > 0.06 else "#27AE60" for g in gaps_f1],
                      alpha=0.85, edgecolor="white")
        ax.axhline(0.06, color="orange", lw=1.5, linestyle="--", label="Limiar moderado (6%)")
        ax.axhline(0.12, color="red",    lw=1.5, linestyle="--", label="Limiar alto (12%)")
        ax.axhline(df["gap_f1"].mean(), color="black", lw=1.5, linestyle=":",
                   label=f"Média = {df['gap_f1'].mean():.3f}")
        for bar, val in zip(bars, gaps_f1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:+.3f}", ha="center", fontsize=9)
        ax.set_xlabel("Fold"); ax.set_ylabel("Gap (Treino − Val) F1 Weighted")
        ax.set_title("Gap de F1 por Fold"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

        # anotação com a mensagem de overfitting
        fig.text(0.5, 0.01, overfit_analysis["message"],
                 ha="center", fontsize=9, color=color,
                 wrap=True, style="italic")

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        self._save(fig, "12_train_val_gap.png")

    # ── 13 Learning curves ────────────────────────────────────────────────

    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray, checkpoint: Dict[str, Any]):
        """
        Curva de aprendizado: treina com subconjuntos crescentes e plota
        train vs val score.  Permite detectar bias (underfitting) e variance.
        """
        from torch.utils.data import DataLoader, TensorDataset
        best_params = checkpoint["best_params"]
        num_classes = checkpoint.get("num_classes", N_CLASSES)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(X, y))
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        fractions = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
        train_scores, val_scores = [], []

        for frac in fractions:
            n = max(int(len(X_tr) * frac), N_CLASSES * 2)
            idx = np.random.choice(len(X_tr), n, replace=False)
            Xs, ys = X_tr[idx], y_tr[idx]

            model = MagicStepsNet(Xs.shape[1], best_params["hidden_layers"],
                                  best_params["dropout"], num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
            criterion = nn.CrossEntropyLoss()
            ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32),
                               torch.tensor(ys, dtype=torch.long))
            loader = DataLoader(ds, batch_size=best_params["batch_size"], shuffle=True)

            model.train()
            for _ in range(max(10, best_params["epochs"] // 2)):
                for Xb, yb in loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    nn.CrossEntropyLoss()(model(Xb), yb).backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                tr_pred = np.argmax(torch.softmax(
                    model(torch.tensor(Xs, dtype=torch.float32).to(DEVICE)), dim=1
                ).cpu().numpy(), axis=1)
                val_pred = np.argmax(torch.softmax(
                    model(torch.tensor(X_val, dtype=torch.float32).to(DEVICE)), dim=1
                ).cpu().numpy(), axis=1)

            train_scores.append(f1_score(ys, tr_pred,  average="weighted", zero_division=0))
            val_scores.append(  f1_score(y_val, val_pred, average="weighted", zero_division=0))

        sizes = [int(len(X_tr) * f) for f in fractions]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sizes, train_scores, "o-", color="#3498DB", lw=2, ms=7, label="Treino")
        ax.plot(sizes, val_scores,   "s--", color="#E74C3C", lw=2, ms=7, label="Validação")
        ax.fill_between(sizes, train_scores, val_scores, alpha=0.08, color="#95A5A6")
        ax.set_xlabel("Tamanho do conjunto de treino"); ax.set_ylabel("F1 Weighted")
        ax.set_title("Curva de Aprendizado — Bias vs Variance", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        self._save(fig, "13_learning_curves.png")

    # ── orquestrador ─────────────────────────────────────────────────────

    def plot_all(self, y_true, y_pred, y_proba, cm, metrics,
                 cv_results: Optional[Dict] = None,
                 X: Optional[np.ndarray] = None,
                 y_full: Optional[np.ndarray] = None,
                 checkpoint: Optional[Dict] = None,
                 class_names=CLASS_LABELS):
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
        if cv_results:
            self.plot_cv_scores(cv_results)
            self.plot_train_val_gap(cv_results)
        if X is not None and checkpoint is not None:
            self.logger.info("  Gerando curva de aprendizado (pode demorar alguns minutos)…")
            self.plot_learning_curves(X, y_full if y_full is not None else y_true, checkpoint)
        self.logger.info("Todas as figuras salvas.")


# ============================================================
# EVALUATOR
# ============================================================

class Evaluator:
    def __init__(self):
        self.logger        = setup_logger(self.__class__.__name__)
        self.model_loader  = ModelLoader()
        self.data_loader   = EvaluationDataLoader()
        self.metrics_calc  = MetricsCalculator()
        self.cross_val     = CrossValidator(n_splits=5)
        self.visualizer    = Visualizer()

    # @torch.no_grad()
    def evaluate(
        self,
        model_name:     str  = "model_magic_steps_dl",
        use_feast:      bool = False,
        generate_plots: bool = True,
        run_cv:         bool = True,
    ) -> Dict[str, Any]:
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO AVALIAÇÃO DO MODELO")
        self.logger.info("=" * 60)

        # 1. Modelo
        model, checkpoint = self.model_loader.load_model(model_name)

        # 2. Dados
        X, y_true, class_names = self.data_loader.load_dataset()

        # 3. Inferência no conjunto completo
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits   = model(X_tensor)
        y_proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
        y_pred   = np.argmax(y_proba, axis=1)

        # 4. Métricas de hold-out
        metrics = self.metrics_calc.calculate_metrics(y_true, y_pred, y_proba)
        self.metrics_calc.print_metrics(metrics)

        # 5. Matriz de confusão
        cm = self.metrics_calc.get_confusion_matrix(y_true, y_pred)
        report = self.metrics_calc.get_classification_report(y_true, y_pred, class_names)
        self.logger.info(f"\nRELATÓRIO DE CLASSIFICAÇÃO\n{report}")

        # 6. Cross-validation + análise de overfitting
        cv_results = None
        if run_cv:
            self.logger.info("\n" + "─" * 60)
            self.logger.info("CROSS-VALIDATION + ANÁLISE DE OVERFITTING")
            self.logger.info("─" * 60)
            cv_results = self.cross_val.run(X, y_true, checkpoint)

        # 7. Visualizações
        if generate_plots:
            self.visualizer.plot_all(
                y_true, y_pred, y_proba, cm, metrics,
                cv_results=cv_results,
                X=X, y_full=y_true, checkpoint=checkpoint,
                class_names=class_names,
            )

        # 8. Resultado consolidado
        results = {
            "metrics": {
                k: (v if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else None)
                for k, v in metrics.items()
            },
            "confusion_matrix":       cm.tolist(),
            "classification_report":  report,
            "model_params":           checkpoint["best_params"],
            "class_labels":           class_names,
            "cross_validation":       cv_results,
        }

        # 9. Salvar JSON + PostgreSQL
        self._save_results(results, model_name)

        self.logger.info("=" * 60)
        self.logger.info("AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
        self.logger.info("=" * 60)

        # Imprimir resumo amigável no terminal
        self._print_cv_summary(cv_results)

        return results

    def _save_results(self, results: Dict[str, Any], model_name: str) -> None:
        import json, math

        # ── JSON local ───────────────────────────────────────────────────
        path = ARTIFACTS_DIR / f"{model_name}_evaluation_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Resultados salvos em: {path}")

        # ── PostgreSQL ───────────────────────────────────────────────────
        try:
            ensure_schema(settings.database_url)
            runs = list_model_runs(settings.database_url)
            if runs:
                latest_run_id = runs[0]["run_id"]
                metrics_safe = {
                    k: (None if v is None or (isinstance(v, float) and
                        (math.isnan(v) or math.isinf(v))) else v)
                    for k, v in results.get("metrics", {}).items()
                }
                cv = results.get("cross_validation") or {}
                update_model_run(
                    latest_run_id, settings.database_url,
                    metrics={**metrics_safe, "cv_summary": cv.get("cv_summary", {}),
                             "overfit_level": cv.get("overfit_analysis", {}).get("level")},
                    confusion_matrix=results.get("confusion_matrix"),
                    classification_report=results.get("classification_report"),
                    status="evaluated",
                )
                log_monitoring_event(
                    "model_evaluated",
                    {
                        "run_id":      latest_run_id,
                        "accuracy":    metrics_safe.get("accuracy"),
                        "f1_weighted": metrics_safe.get("f1_weighted"),
                        "overfit":     cv.get("overfit_analysis", {}).get("level"),
                        "cv_val_acc":  cv.get("cv_summary", {}).get("val_acc_mean"),
                    },
                    settings.database_url,
                )
                self.logger.info("✅ Métricas de avaliação persistidas no PostgreSQL.")
        except Exception as e:
            self.logger.warning(f"Não foi possível persistir no PostgreSQL: {e}")

    def _print_cv_summary(self, cv_results: Optional[Dict]):
        if not cv_results:
            return
        s  = cv_results["cv_summary"]
        oa = cv_results["overfit_analysis"]
        print("\n" + "=" * 60)
        print("  CROSS-VALIDATION SUMMARY")
        print("=" * 60)
        print(f"  val_accuracy  : {s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f}")
        print(f"  val_f1        : {s['val_f1_mean']:.4f} ± {s['val_f1_std']:.4f}")
        if not math.isnan(s.get("val_auc_mean", float("nan"))):
            print(f"  val_auc       : {s['val_auc_mean']:.4f} ± {s['val_auc_std']:.4f}")
        print(f"  gap_accuracy  : {s['gap_acc_mean']:+.4f} ± {s['gap_acc_std']:.4f}")
        print(f"  gap_f1        : {s['gap_f1_mean']:+.4f} ± {s['gap_f1_std']:.4f}")
        print(f"\n  ⚖️  Overfitting [{oa['level'].upper()}]: {oa['message']}")
        print("=" * 60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    evaluator = Evaluator()
    evaluator.evaluate(
        model_name="model_magic_steps_dl",
        use_feast=False,
        generate_plots=True,
        run_cv=True,
    )
    print("\n✅ Avaliação finalizada!")


if __name__ == "__main__":
    print("🔍 Avaliando modelo — Magic Steps")
    main()