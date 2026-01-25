# train.py

from pathlib import Path
from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARTIFACTS_DIR = Path("out")

DATASET_PATH = ARTIFACTS_DIR / "dataset_transformed_magic_steps.parquet"
MODEL_PATH = ARTIFACTS_DIR / "model_magic_steps_dl.pt"

TARGET_COLUMN = "flag_atingiu_pv"

RANDOM_STATE = 42
VAL_SIZE = 0.2
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

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
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        dropout: float,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ============================================================
# TRAIN / VALID
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(X), y)
        total_loss += loss.item()

    return total_loss / len(loader)


def fit_model(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
) -> float:
    model = MagicStepsNet(
        input_dim=X_train.shape[1],
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"]
    )
    criterion = nn.BCEWithLogitsLoss()

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

    best_val = float("inf")
    epochs_no_improve = 0

    for _ in range(params["epochs"]):
        # 🔧 AQUI estava o erro
        train_epoch(model, train_loader, criterion, optimizer)

        val_loss = val_epoch(model, val_loader, criterion)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    return best_val


# ============================================================
# GRID SEARCH
# ============================================================

def grid_search(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "hidden_layers": [
            [64],
            [128],
            [128, 64],
            [256, 128],
            [256, 128, 64],
        ],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "lr": [1e-2, 5e-3, 1e-3, 5e-4],
        "batch_size": [16, 32, 64],
        "epochs": [100,],
    }

    keys, values = zip(*param_grid.items())
    combinations = list(product(*values))

    print(f"🔎 Total de combinações: {len(combinations)}")

    best_score = float("inf")
    best_params = None

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))

        val_loss = fit_model(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
        )

        print(
            f"[{i}/{len(combinations)}] "
            f"Val Loss: {val_loss:.4f} | Params: {params}"
        )

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

    print("\n🏆 Melhores hiperparâmetros:")
    print(best_params)
    print(f"📉 Melhor Val Loss: {best_score:.4f}")

    return best_params


# ============================================================
# FINAL TRAIN
# ============================================================

def train_final_model(X, y, best_params):
    model = MagicStepsNet(
        input_dim=X.shape[1],
        hidden_layers=best_params["hidden_layers"],
        dropout=best_params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=best_params["lr"]
    )
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        MagicStepsDataset(X, y),
        batch_size=best_params["batch_size"],
        shuffle=True,
    )

    for epoch in range(best_params["epochs"]):
        loss = train_epoch(model, loader, criterion, optimizer)
        print(f"Final Epoch {epoch+1:03d} | Loss: {loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "best_params": best_params,
        },
        MODEL_PATH,
    )

    print(f"\n💾 Modelo final salvo em: {MODEL_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"🖥️ Device: {DEVICE}")
    print("📥 Carregando dataset...")

    X, y = load_dataset(DATASET_PATH)

    print("🔍 Iniciando Grid Search...")
    best_params = grid_search(X, y)

    print("\n🧠 Treinando modelo final com melhores hiperparâmetros...")
    train_final_model(X, y, best_params)


if __name__ == "__main__":
    print("🎯 Treinamento Deep Learning com Grid Search - Magic Steps")
    main()
