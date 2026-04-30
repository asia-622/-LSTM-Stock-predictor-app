# ============================================================
# utils.py — Data utilities, sequence builder, metrics
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Optional
import os

import config


# ── Data Fetching ────────────────────────────────────────────

def fetch_stock_data(
    ticker:     str,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Returns
    -------
    DataFrame with DatetimeIndex and at minimum a 'Close' column.
    Raises ValueError on unknown ticker or empty result.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as exc:
        raise ValueError(f"Failed to download data for '{ticker}': {exc}") from exc

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}. "
            "Check the symbol and date range."
        )

    # Flatten multi-index columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# ── Scaling ──────────────────────────────────────────────────

def fit_scaler(data: np.ndarray) -> MinMaxScaler:
    """Fit and return a MinMaxScaler on 1-D price array."""
    scaler = MinMaxScaler(feature_range=config.SCALE_RANGE)
    scaler.fit(data.reshape(-1, 1))
    return scaler


def scale_data(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray:
    """Transform 1-D array to scaled values."""
    return scaler.transform(data.reshape(-1, 1)).flatten()


def inverse_scale(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled predictions back to price space."""
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def save_scaler(scaler: MinMaxScaler, path: str = config.SCALER_PATH) -> None:
    """Persist scaler to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str = config.SCALER_PATH) -> MinMaxScaler:
    """Load scaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at '{path}'. Train the model first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Sequence Construction ────────────────────────────────────

def create_sequences(
    data:            np.ndarray,
    sequence_length: int = config.SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window sequence builder.

    Parameters
    ----------
    data            : 1-D scaled price array
    sequence_length : look-back window size

    Returns
    -------
    X : (n_samples, sequence_length, 1)  — input sequences
    y : (n_samples,)                     — next-step targets
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i])
        y.append(data[i])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]   # add feature dim
    y = np.array(y, dtype=np.float32)
    return X, y


# ── Train / Val / Test Split ─────────────────────────────────

def train_val_test_split(
    X:          np.ndarray,
    y:          np.ndarray,
    train_frac: float = config.TRAIN_SPLIT,
    val_frac:   float = config.VAL_SPLIT,
) -> Tuple[np.ndarray, ...]:
    """
    Chronological (non-shuffled) split into train / validation / test.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    n        = len(X)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)

    X_train  = X[:n_train]
    y_train  = y[:n_train]

    X_val    = X[n_train : n_train + n_val]
    y_val    = y[n_train : n_train + n_val]

    X_test   = X[n_train + n_val :]
    y_test   = y[n_train + n_val :]

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Evaluation Metrics ───────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute RMSE, MAE, and R² for regression evaluation.

    Parameters
    ----------
    y_true : actual prices (inverse-scaled)
    y_pred : predicted prices (inverse-scaled)

    Returns
    -------
    dict with keys 'RMSE', 'MAE', 'R2'
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentage of time the model correctly predicts price direction
    (up or down vs previous day).
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)


# ── Convenience Wrapper ──────────────────────────────────────

def prepare_data(
    ticker:     str,
    start_date: str,
    end_date:   str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.DatetimeIndex]:
    """
    Full pipeline: fetch → scale → build sequences → split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, dates
    """
    df     = fetch_stock_data(ticker, start_date, end_date)
    prices = df[config.FEATURE_COLUMN].values.astype(np.float32)
    dates  = df.index

    scaler        = fit_scaler(prices)
    scaled_prices = scale_data(scaler, prices)

    X, y = create_sequences(scaled_prices, config.SEQUENCE_LENGTH)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Align date index: sequences start SEQUENCE_LENGTH steps in
    seq_dates = dates[config.SEQUENCE_LENGTH :]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, seq_dates
