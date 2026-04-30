# 📈 Stock Price Predictor — LSTM Deep Learning

> **Predict stock prices with a multi-layer LSTM neural network, interactive Plotly charts, and a sleek Streamlit dashboard.**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **LSTM Model** | Multi-layer PyTorch LSTM with dropout & gradient clipping |
| 📡 **Live Data** | Fetches OHLCV data from Yahoo Finance via `yfinance` |
| 📊 **Interactive Charts** | Actual vs. Predicted overlay, error bars, candlestick + volume |
| 🔮 **Next-Day Forecast** | Predicts tomorrow's closing price with % change |
| 📏 **Evaluation Metrics** | RMSE, MAE, R² Score, Directional Accuracy |
| 🎛️ **Tunable Hyperparameters** | Epochs, sequence length, hidden units, learning rate via UI |
| ⚡ **Model Caching** | Saved `.pth` checkpoint — no retraining needed on reload |
| 🌑 **Dark Mode UI** | Glassmorphism-inspired Streamlit theme |

---

## 🗂️ Project Structure

```
stock-predictor-app/
│
├── app.py           ← Streamlit dashboard (entry point)
├── train.py         ← Full training pipeline
├── predict.py       ← Inference & next-day forecasting
├── model.py         ← LSTM architecture (PyTorch)
├── utils.py         ← Data fetch, scaling, sequence builder, metrics
├── config.py        ← All hyperparameters & paths
├── requirements.txt
├── README.md
│
├── data/            ← (auto-created) cached CSVs
├── models/
│   ├── lstm_model.pth   ← saved model checkpoint
│   └── scaler.pkl       ← fitted MinMaxScaler
└── notebooks/
    └── exploration.ipynb
```

---

## 🚀 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/stock-predictor-app.git
cd stock-predictor-app
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU support** (optional): install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before installing other requirements.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 🖥️ Usage Guide

1. **Select a ticker** — choose from the popular list (AAPL, TSLA, NVDA …) or type any Yahoo Finance symbol.
2. **Set a date range** — broader ranges give the model more training data.
3. **Tune hyperparameters** (optional) — expand the ⚙️ panel in the sidebar.
4. **Click "Train Model"** — watch the live loss curves and epoch log.
5. **Click "Run Prediction"** — view the Actual vs. Predicted chart, metrics, and next-day forecast.

---

## ⚙️ Configuration

All hyperparameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | 60 | Look-back window (trading days) |
| `HIDDEN_SIZE` | 128 | LSTM hidden units per layer |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.2 | Dropout between LSTM layers |
| `EPOCHS` | 50 | Training epochs |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `TRAIN_SPLIT` | 0.80 | Training data fraction |
| `VAL_SPLIT` | 0.10 | Validation data fraction |

---

## 🧪 Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root Mean Squared Error — punishes large errors |
| **MAE** | Mean Absolute Error — average price deviation |
| **R² Score** | Coefficient of determination (1.0 = perfect fit) |
| **Directional Accuracy** | % of days the model correctly predicted up/down movement |

---

## 🌐 Deployment

### Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file** to `app.py`.
4. Click **Deploy** — that's it!

### Hugging Face Spaces

1. Create a new Space → choose **Streamlit** SDK.
2. Upload all project files (or connect your GitHub repo).
3. Add a `packages.txt` if any system dependencies are needed (usually not required).
4. The Space will install `requirements.txt` automatically.

```
# packages.txt (only if needed)
# libgomp1
```

**Live demo:** `https://huggingface.co/spaces/your-username/stock-predictor`

---

## 🛠️ CLI Usage

Train from the command line:

```bash
python train.py TSLA
```

Run predictions:

```bash
python predict.py TSLA 2020-01-01 2024-12-31
```

---

## 📸 Screenshots

> *Add screenshots here after running the app.*

| Training | Prediction |
|---|---|
| ![train](docs/train.png) | ![predict](docs/predict.png) |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance of stock prices does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
