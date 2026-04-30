# ============================================================
# train.py — End-to-end LSTM training pipeline
# ============================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional, Tuple, List

import config
from model import StockLSTM
from utils import (
    prepare_data,
    save_scaler,
    compute_metrics,
    inverse_scale,
)


# ── Device Selection ─────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset Builder ──────────────────────────────────────────

def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """Wrap numpy arrays in PyTorch DataLoaders."""

    def to_tensor(*arrays):
        return [torch.FloatTensor(a) for a in arrays]

    Xt, yt = to_tensor(X_train, y_train)
    Xv, yv = to_tensor(X_val,   y_val)

    train_loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(Xv, yv),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


# ── Training Loop ────────────────────────────────────────────

def train_model(
    ticker:      str       = config.DEFAULT_TICKER,
    start_date:  str       = config.DEFAULT_START_DATE,
    end_date:    str       = config.DEFAULT_END_DATE,
    epochs:      int       = config.EPOCHS,
    lr:          float     = config.LEARNING_RATE,
    batch_size:  int       = config.BATCH_SIZE,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """
    Orchestrates the full training workflow.

    Parameters
    ----------
    ticker       : Yahoo Finance ticker symbol
    start_date   : Training data start (YYYY-MM-DD)
    end_date     : Training data end   (YYYY-MM-DD)
    epochs       : Number of training epochs
    lr           : Adam learning rate
    batch_size   : Mini-batch size
    progress_cb  : Optional callback(message: str, pct: float)
                   called at each epoch for progress reporting.

    Returns
    -------
    dict with keys: train_losses, val_losses, test_metrics,
                    y_test_actual, y_pred_actual, test_dates
    """

    device = get_device()

    def _cb(msg: str, pct: float):
        if progress_cb:
            progress_cb(msg, pct)

    # ── 1. Data Preparation ──────────────────────────────────
    _cb("Fetching stock data…", 0.05)
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, seq_dates,
    ) = prepare_data(ticker, start_date, end_date)

    _cb("Building data loaders…", 0.10)
    train_loader, val_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, batch_size
    )

    # ── 2. Model, Loss, Optimiser ────────────────────────────
    model = StockLSTM(
        input_size  = 1,
        hidden_size = config.HIDDEN_SIZE,
        num_layers  = config.NUM_LAYERS,
        dropout     = config.DROPOUT,
        output_size = config.OUTPUT_SIZE,
    ).to(device)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    # ── 3. Epoch Loop ────────────────────────────────────────
    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # — Training —
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimiser.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_train_loss += loss.item() * X_batch.size(0)

        epoch_train_loss /= len(train_loader.dataset)

        # — Validation —
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                preds   = model(X_batch)
                epoch_val_loss += criterion(preds, y_batch).item() * X_batch.size(0)
        epoch_val_loss /= max(len(val_loader.dataset), 1)

        scheduler.step(epoch_val_loss)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # Keep best checkpoint in memory
        if epoch_val_loss < best_val_loss:
            best_val_loss  = epoch_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pct = 0.10 + 0.75 * (epoch / epochs)
        _cb(
            f"Epoch {epoch}/{epochs}  "
            f"Train Loss: {epoch_train_loss:.6f}  "
            f"Val Loss: {epoch_val_loss:.6f}",
            pct,
        )

    # ── 4. Restore Best Weights ──────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ── 5. Test-Set Evaluation ───────────────────────────────
    _cb("Evaluating on test set…", 0.88)
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()

    y_test_actual = inverse_scale(scaler, y_test)
    y_pred_actual = inverse_scale(scaler, y_pred_scaled)

    metrics = compute_metrics(y_test_actual, y_pred_actual)

    # Align test dates
    n_train = len(X_train)
    n_val   = len(X_val)
    test_dates = seq_dates[n_train + n_val :]

    # ── 6. Save Artefacts ────────────────────────────────────
    _cb("Saving model and scaler…", 0.95)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_size":  1,
                "hidden_size": config.HIDDEN_SIZE,
                "num_layers":  config.NUM_LAYERS,
                "dropout":     config.DROPOUT,
                "output_size": config.OUTPUT_SIZE,
            },
            "ticker":     ticker,
            "start_date": start_date,
            "end_date":   end_date,
            "metrics":    metrics,
        },
        config.MODEL_PATH,
    )
    save_scaler(scaler, config.SCALER_PATH)

    _cb("Training complete!", 1.0)

    return {
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "test_metrics":   metrics,
        "y_test_actual":  y_test_actual,
        "y_pred_actual":  y_pred_actual,
        "test_dates":     test_dates,
    }


# ── CLI Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_TICKER

    def cli_callback(msg: str, pct: float):
        bar_len  = 30
        filled   = int(bar_len * pct)
        bar      = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct*100:5.1f}%  {msg}", end="", flush=True)

    print(f"Training LSTM for {ticker}…\n")
    t0      = time.time()
    results = train_model(ticker=ticker, progress_cb=cli_callback)
    elapsed = time.time() - t0

    print(f"\n\nTraining finished in {elapsed:.1f}s")
    m = results["test_metrics"]
    print(f"  RMSE : {m['RMSE']:.4f}")
    print(f"  MAE  : {m['MAE']:.4f}")
    print(f"  R²   : {m['R2']:.4f}")
    print(f"\nModel saved to: {config.MODEL_PATH}")
