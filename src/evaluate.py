# evaluate.py

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARTIFACTS_DIR = Path("out")

DATASET_PATH = ARTIFACTS_DIR / "dataset_transformed_magic_steps.parquet"
MODEL_PATH = ARTIFACTS_DIR / "model_magic_steps_dl.pt"

TARGET_COLUMN = "flag_atingiu_pv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET
# ============================================================

class MagicStepsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    X = df.drop(columns=[TARGET_COLUMN]).values.astype("float32")
    y = df[TARGET_COLUMN].values.astype("float32")
    return X, y


# ============================================================
# MODEL
# ============================================================

class MagicStepsNet(nn.Module):
    def __init__(self, input_dim: int, hidden_layers, dropout: float):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(dropout),
                ]
            )
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_model(path: Path, input_dim: int):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = MagicStepsNet(
        input_dim=input_dim,
        hidden_layers=checkpoint["best_params"]["hidden_layers"],
        dropout=checkpoint["best_params"]["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, checkpoint["best_params"]


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_proba(model, loader):
    y_true, y_prob = [], []

    for X, y in loader:
        X = X.to(DEVICE)
        logits = model(X)
        probs = torch.sigmoid(logits)

        y_true.extend(y.numpy())
        y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_prob)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(y_true, y_prob, metric="f1"):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, y_prob
    )

    thresholds = np.append(thresholds, 1.0)

    scores = []
    for p, r in zip(precisions, recalls):
        if metric == "recall":
            scores.append(r)
        else:
            scores.append(2 * (p * r) / (p + r + 1e-8))

    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"🖥️ Device: {DEVICE}")

    print("📥 Carregando dataset...")
    X, y = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    test_loader = DataLoader(
        MagicStepsDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("🧠 Carregando modelo...")
    model, best_params = load_model(
        MODEL_PATH, input_dim=X.shape[1]
    )

    print("📊 Inferência...")
    y_true, y_prob = predict_proba(model, test_loader)

    auc = roc_auc_score(y_true, y_prob)
    print(f"\n🔥 ROC AUC: {auc:.4f}")

    print("\n🔎 Buscando melhor threshold (F1)...")
    best_thr, best_score = find_best_threshold(
        y_true, y_prob, metric="f1"
    )

    print(f"🎯 Melhor Threshold: {best_thr:.3f}")
    print(f"🏆 Melhor F1: {best_score:.4f}")

    y_pred = (y_prob >= best_thr).astype(int)

    print("\n📈 Classification Report (threshold ótimo):")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n🧮 Confusion Matrix:")
    print(cm)

    print(
        f"\nResumo:\n"
        f"TP: {tp} | FP: {fp}\n"
        f"FN: {fn} | TN: {tn}"
    )

    print("\n🧠 Hiperparâmetros usados:")
    for k, v in best_params.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    print("🎯 Avaliação Final do Modelo - Magic Steps")
    main()
