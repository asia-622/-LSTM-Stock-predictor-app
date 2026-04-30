# ============================================================
# config.py — Centralized hyperparameters and app settings
# ============================================================

# ── Model Hyperparameters ────────────────────────────────────
SEQUENCE_LENGTH = 60        # Number of past days used as input window
HIDDEN_SIZE     = 128       # LSTM hidden units per layer
NUM_LAYERS      = 2         # Number of stacked LSTM layers
DROPOUT         = 0.2       # Dropout rate between LSTM layers
OUTPUT_SIZE     = 1         # Single next-day close price prediction

# ── Training Settings ────────────────────────────────────────
EPOCHS          = 50        # Training epochs (reduced for cloud deployment)
BATCH_SIZE      = 32        # Mini-batch size
LEARNING_RATE   = 0.001     # Adam optimizer learning rate
TRAIN_SPLIT     = 0.80      # Fraction of data used for training
VAL_SPLIT       = 0.10      # Fraction of data used for validation

# ── Data Settings ────────────────────────────────────────────
DEFAULT_TICKER      = "AAPL"
DEFAULT_START_DATE  = "2020-01-01"
DEFAULT_END_DATE    = "2024-12-31"
FEATURE_COLUMN      = "Close"       # Target column to predict
SCALE_RANGE         = (0, 1)        # MinMaxScaler range

# ── Paths ────────────────────────────────────────────────────
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")

MODEL_PATH  = os.path.join(MODELS_DIR, "lstm_model.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Ensure directories exist at import time
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── UI / Display ─────────────────────────────────────────────
APP_TITLE       = "📈 Stock Price Predictor"
APP_SUBTITLE    = "Deep Learning · LSTM · Real-Time Predictions"
CHART_HEIGHT    = 500
PLOTLY_THEME    = "plotly_dark"
