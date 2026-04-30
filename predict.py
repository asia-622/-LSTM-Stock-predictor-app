# ============================================================
# predict.py — Inference pipeline using a saved LSTM model
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional

import config
from model import StockLSTM
from utils import (
    fetch_stock_data,
    load_scaler,
    scale_data,
    inverse_scale,
    create_sequences,
    compute_metrics,
    directional_accuracy,
)


# ── Model Loader ─────────────────────────────────────────────

def load_model(
    model_path: str = config.MODEL_PATH,
    device:     Optional[torch.device] = None,
) -> Tuple[StockLSTM, dict]:
    """
    Load a saved LSTM checkpoint.

    Returns
    -------
    (model, checkpoint_dict)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Please train the model first via train.py or the 'Train Model' button."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    cfg        = checkpoint["config"]

    model = StockLSTM(
        input_size  = cfg["input_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        output_size = cfg["output_size"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ── Full Prediction Run ──────────────────────────────────────

def run_prediction(
    ticker:     str,
    start_date: str,
    end_date:   str,
    model_path: str = config.MODEL_PATH,
    scaler_path:str = config.SCALER_PATH,
) -> dict:
    """
    Run inference over the full date window and return results.

    The function:
      1. Fetches fresh data from Yahoo Finance
      2. Scales with the saved scaler
      3. Builds sequences
      4. Passes them through the LSTM
      5. Inverse-transforms predictions
      6. Computes evaluation metrics

    Returns
    -------
    dict with:
        dates_all       : DatetimeIndex aligned to sequences
        actual_all      : np.ndarray — full actual prices
        predicted_all   : np.ndarray — full model predictions
        metrics         : dict {RMSE, MAE, R2}
        dir_accuracy    : float — directional accuracy %
        raw_df          : original DataFrame from yfinance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load artefacts ───────────────────────────────────────
    model, _ = load_model(model_path, device)
    scaler   = load_scaler(scaler_path)

    # ── Fetch & preprocess ───────────────────────────────────
    df     = fetch_stock_data(ticker, start_date, end_date)
    prices = df[config.FEATURE_COLUMN].values.astype(np.float32)
    dates  = df.index

    scaled = scale_data(scaler, prices)
    X, y   = create_sequences(scaled, config.SEQUENCE_LENGTH)

    # Align dates (sequences start SEQUENCE_LENGTH steps in)
    seq_dates = dates[config.SEQUENCE_LENGTH :]

    # ── Inference ────────────────────────────────────────────
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    actual_all    = inverse_scale(scaler, y)
    predicted_all = inverse_scale(scaler, y_pred_scaled)

    # ── Metrics ──────────────────────────────────────────────
    metrics      = compute_metrics(actual_all, predicted_all)
    dir_accuracy = directional_accuracy(actual_all, predicted_all)

    return {
        "dates_all":      seq_dates,
        "actual_all":     actual_all,
        "predicted_all":  predicted_all,
        "metrics":        metrics,
        "dir_accuracy":   dir_accuracy,
        "raw_df":         df,
    }


# ── Single Next-Day Forecast ─────────────────────────────────

def predict_next_day(
    ticker:      str,
    start_date:  str,
    end_date:    str,
    model_path:  str = config.MODEL_PATH,
    scaler_path: str = config.SCALER_PATH,
) -> float:
    """
    Return a single next-day closing price forecast.

    Uses the last SEQUENCE_LENGTH trading days ending on end_date
    as the input window.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = load_model(model_path, device)
    scaler   = load_scaler(scaler_path)

    df     = fetch_stock_data(ticker, start_date, end_date)
    prices = df[config.FEATURE_COLUMN].values.astype(np.float32)

    if len(prices) < config.SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {config.SEQUENCE_LENGTH} data points for a forecast "
            f"(got {len(prices)})."
        )

    scaled   = scale_data(scaler, prices)
    window   = scaled[-config.SEQUENCE_LENGTH :]            # last N days
    X_input  = torch.FloatTensor(window).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        pred_scaled = model(X_input).cpu().numpy().flatten()

    return float(inverse_scale(scaler, pred_scaled)[0])


# ── CLI Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ticker     = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_TICKER
    start_date = sys.argv[2] if len(sys.argv) > 2 else config.DEFAULT_START_DATE
    end_date   = sys.argv[3] if len(sys.argv) > 3 else config.DEFAULT_END_DATE

    print(f"\nRunning predictions for {ticker} ({start_date} → {end_date})…\n")

    results = run_prediction(ticker, start_date, end_date)
    m       = results["metrics"]

    print(f"  RMSE               : {m['RMSE']:.4f}")
    print(f"  MAE                : {m['MAE']:.4f}")
    print(f"  R²                 : {m['R2']:.4f}")
    print(f"  Directional Acc    : {results['dir_accuracy']:.1f}%")

    next_price = predict_next_day(ticker, start_date, end_date)
    print(f"\n  Next-day forecast  : ${next_price:.2f}")
