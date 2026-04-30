# ============================================================
# app.py — Streamlit dashboard for Stock Price Predictor
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

import config
from train   import train_model
from predict import run_prediction, predict_next_day, load_model


# ── Page Configuration ───────────────────────────────────────

st.set_page_config(
    page_title = "Stock Price Predictor",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── Custom CSS ───────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid #1e2a3a;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(14,165,233,0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(14,165,233,0.45);
    }

    /* ── Input widgets ── */
    .stTextInput input,
    .stSelectbox select,
    .stDateInput input {
        background-color: #1a2332 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Section header ── */
    .section-header {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #0ea5e9;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.3rem;
    }

    /* ── Hero header ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.3rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    /* ── Info / Warning boxes ── */
    .info-box {
        background: linear-gradient(135deg, #0c1929 0%, #0d1f35 100%);
        border: 1px solid #1e3a5f;
        border-left: 3px solid #0ea5e9;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: linear-gradient(135deg, #1a1000 0%, #1f1500 100%);
        border: 1px solid #78350f;
        border-left: 3px solid #f59e0b;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #d97706;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #1e2a3a;
        margin: 1.5rem 0;
    }

    /* ── Plotly chart borders ── */
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        overflow: hidden;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #818cf8) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Initialisation ─────────────────────────────

for key in ["trained", "predict_results", "train_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "model_meta" not in st.session_state:
    st.session_state["model_meta"] = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; margin-bottom:1.5rem;">'
        '<span style="font-size:2.5rem">📈</span>'
        '<h2 style="color:#38bdf8; margin:0.3rem 0 0; font-weight:700; font-size:1.1rem; '
        'letter-spacing:0.05em;">STOCK PREDICTOR</h2>'
        '<p style="color:#475569; font-size:0.7rem; font-family:JetBrains Mono,monospace; '
        'letter-spacing:0.1em; margin:0;">LSTM · DEEP LEARNING</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Ticker ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Ticker Symbol</div>', unsafe_allow_html=True)

    popular = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX", "SPY", "Custom…"]
    chosen  = st.selectbox("Popular symbols", popular, label_visibility="collapsed")

    if chosen == "Custom…":
        ticker = st.text_input("Enter ticker", value="BTC-USD", label_visibility="collapsed").upper().strip()
    else:
        ticker = chosen

    # ── Date Range ───────────────────────────────────────────
    st.markdown('<div class="section-header">Date Range</div>', unsafe_allow_html=True)

    today      = date.today()
    max_start  = today - timedelta(days=30)
    default_s  = date(2020, 1, 1)
    default_e  = today

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=default_s, max_value=max_start)
    with col2:
        end_date = st.date_input("To", value=default_e, min_value=start_date + timedelta(days=31))

    # ── Hyperparameters ──────────────────────────────────────
    with st.expander("⚙️ Hyperparameters", expanded=False):
        epochs     = st.slider("Epochs",        min_value=5,   max_value=200, value=config.EPOCHS,        step=5)
        seq_len    = st.slider("Sequence Length", min_value=10, max_value=120, value=config.SEQUENCE_LENGTH, step=5)
        hidden     = st.select_slider("Hidden Units", options=[32, 64, 128, 256], value=config.HIDDEN_SIZE)
        lr         = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=config.LEARNING_RATE,
            format_func=lambda x: f"{x:.4f}",
        )

    st.markdown("---")

    # ── Train Button ─────────────────────────────────────────
    train_btn = st.button("🧠 Train Model", use_container_width=True)

    # ── Predict Button ────────────────────────────────────────
    predict_btn = st.button("🔮 Run Prediction", use_container_width=True)

    # ── Model Status ─────────────────────────────────────────
    st.markdown("---")
    if os.path.exists(config.MODEL_PATH):
        st.markdown(
            '<div class="info-box">✅ <strong>Model found</strong><br>'
            'Ready to predict without retraining.</div>',
            unsafe_allow_html=True,
        )
        meta = st.session_state.get("model_meta", {})
        if meta.get("ticker"):
            st.caption(f"Last trained: **{meta['ticker']}** | "
                       f"{meta.get('start_date','?')} → {meta.get('end_date','?')}")
    else:
        st.markdown(
            '<div class="warning-box">⚠️ No model found.<br>'
            'Click <strong>Train Model</strong> first.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#334155; font-size:0.65rem; text-align:center; '
        'margin-top:2rem; font-family:JetBrains Mono,monospace;">'
        'LSTM · PyTorch · Streamlit<br>Not financial advice.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN PANEL — Header
# ============================================================

st.markdown(
    f'<div class="hero-title">{config.APP_TITLE}</div>'
    f'<div class="hero-sub">{config.APP_SUBTITLE}</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Live Price Strip ─────────────────────────────────────────

@st.cache_data(ttl=60)
def get_live_price(tick: str) -> dict:
    """Fetch today's OHLCV for a quick live price snapshot."""
    try:
        import yfinance as yf
        info   = yf.Ticker(tick).fast_info
        return {
            "price":  getattr(info, "last_price", None),
            "change": getattr(info, "regular_market_change_percent", None),
        }
    except Exception:
        return {"price": None, "change": None}


live = get_live_price(ticker)
c1, c2, c3, c4 = st.columns(4)
with c1:
    price_val  = f"${live['price']:.2f}"  if live["price"]  else "—"
    delta_val  = f"{live['change']:+.2f}%" if live["change"] else None
    st.metric("💹 Live Price", price_val, delta=delta_val)
with c2:
    st.metric("📅 Start Date", str(start_date))
with c3:
    st.metric("📅 End Date",   str(end_date))
with c4:
    trained_str = "✅ Ready" if os.path.exists(config.MODEL_PATH) else "❌ Not trained"
    st.metric("🤖 Model Status", trained_str)

st.markdown("---")


# ============================================================
# TRAIN ACTION
# ============================================================

if train_btn:
    st.markdown("### 🧠 Training LSTM Model")
    status_text  = st.empty()
    progress_bar = st.progress(0)
    log_area     = st.empty()
    log_lines    = []

    def training_callback(msg: str, pct: float):
        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(
            f'<span style="color:#38bdf8; font-family:JetBrains Mono,monospace;">'
            f'{msg}</span>',
            unsafe_allow_html=True,
        )
        log_lines.append(msg)
        if len(log_lines) > 6:
            log_lines.pop(0)
        log_area.code("\n".join(log_lines), language="")

    try:
        # Patch config values with sidebar overrides at runtime
        config.EPOCHS          = epochs
        config.SEQUENCE_LENGTH = seq_len
        config.HIDDEN_SIZE     = hidden
        config.LEARNING_RATE   = lr

        results = train_model(
            ticker      = ticker,
            start_date  = str(start_date),
            end_date    = str(end_date),
            epochs      = epochs,
            lr          = lr,
            progress_cb = training_callback,
        )

        progress_bar.progress(1.0)
        status_text.markdown(
            '<span style="color:#4ade80; font-weight:700;">✅ Training complete!</span>',
            unsafe_allow_html=True,
        )

        st.session_state["train_results"] = results
        st.session_state["trained"]       = True
        st.session_state["model_meta"]    = {
            "ticker":     ticker,
            "start_date": str(start_date),
            "end_date":   str(end_date),
        }

        # ── Training curves ──
        st.markdown("#### 📉 Loss Curves")
        fig_loss = go.Figure()
        epochs_x = list(range(1, len(results["train_losses"]) + 1))
        fig_loss.add_trace(go.Scatter(
            x=epochs_x, y=results["train_losses"],
            name="Train Loss", line=dict(color="#0ea5e9", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs_x, y=results["val_losses"],
            name="Val Loss", line=dict(color="#a78bfa", width=2, dash="dash"),
        ))
        fig_loss.update_layout(
            template    = config.PLOTLY_THEME,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(13,17,23,0.8)",
            height      = 300,
            margin      = dict(t=20, b=20, l=20, r=20),
            legend      = dict(orientation="h", y=1.05),
            xaxis_title = "Epoch",
            yaxis_title = "MSE Loss",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # ── Test metrics ──
        m = results["test_metrics"]
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("RMSE", f"{m['RMSE']:.4f}")
        mc2.metric("MAE",  f"{m['MAE']:.4f}")
        mc3.metric("R²",   f"{m['R2']:.4f}")

    except ValueError as e:
        st.error(f"⚠️ {e}")
    except Exception as e:
        st.error(f"Unexpected error during training: {e}")
        st.exception(e)


# ============================================================
# PREDICT ACTION
# ============================================================

if predict_btn:
    if not os.path.exists(config.MODEL_PATH):
        st.error("❌ No trained model found. Please train the model first.")
    else:
        with st.spinner("🔮 Running predictions…"):
            try:
                results = run_prediction(
                    ticker     = ticker,
                    start_date = str(start_date),
                    end_date   = str(end_date),
                )
                st.session_state["predict_results"] = results

            except ValueError as e:
                st.error(f"⚠️ {e}")
                results = None
            except Exception as e:
                st.error(f"Unexpected error during prediction: {e}")
                st.exception(e)
                results = None

        if results:
            # ── Metrics Row ──────────────────────────────────
            st.markdown("### 📊 Evaluation Metrics")
            m   = results["metrics"]
            d   = results["dir_accuracy"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("📉 RMSE",              f"{m['RMSE']:.4f}")
            mc2.metric("📏 MAE",               f"{m['MAE']:.4f}")
            mc3.metric("📈 R² Score",          f"{m['R2']:.4f}")
            mc4.metric("🎯 Directional Acc.",  f"{d:.1f}%")

            st.markdown("---")

            # ── Main Chart ────────────────────────────────────
            st.markdown("### 📈 Actual vs Predicted Prices")

            dates  = results["dates_all"]
            actual = results["actual_all"]
            pred   = results["predicted_all"]
            error  = actual - pred

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.04,
                subplot_titles=("Close Price: Actual vs Predicted", "Prediction Error"),
            )

            # Actual price
            fig.add_trace(
                go.Scatter(
                    x=dates, y=actual,
                    name="Actual",
                    line=dict(color="#38bdf8", width=1.5),
                    mode="lines",
                ),
                row=1, col=1,
            )

            # Predicted price
            fig.add_trace(
                go.Scatter(
                    x=dates, y=pred,
                    name="Predicted",
                    line=dict(color="#f472b6", width=1.5, dash="dot"),
                    mode="lines",
                ),
                row=1, col=1,
            )

            # Filled band between actual & predicted
            fig.add_trace(
                go.Scatter(
                    x=list(dates) + list(dates)[::-1],
                    y=list(actual) + list(pred)[::-1],
                    fill="toself",
                    fillcolor="rgba(248,113,185,0.05)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=1,
            )

            # Error bars
            fig.add_trace(
                go.Bar(
                    x=dates, y=error,
                    name="Error",
                    marker_color=np.where(error >= 0, "#4ade80", "#f87171").tolist(),
                    opacity=0.7,
                ),
                row=2, col=1,
            )

            fig.update_layout(
                template      = config.PLOTLY_THEME,
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(13,17,23,0.8)",
                height        = config.CHART_HEIGHT,
                margin        = dict(t=50, b=20, l=20, r=20),
                legend        = dict(orientation="h", y=1.02, x=0),
                hovermode     = "x unified",
                font          = dict(family="Space Grotesk, sans-serif", size=12),
            )
            fig.update_xaxes(gridcolor="#1e2a3a", zeroline=False)
            fig.update_yaxes(gridcolor="#1e2a3a", zeroline=False)

            st.plotly_chart(fig, use_container_width=True)

            # ── Next-Day Forecast ─────────────────────────────
            st.markdown("### 🔮 Next-Day Price Forecast")
            try:
                next_price = predict_next_day(
                    ticker,
                    str(start_date),
                    str(end_date),
                )
                last_actual = float(actual[-1])
                chg         = next_price - last_actual
                chg_pct     = (chg / last_actual) * 100

                nf1, nf2, nf3 = st.columns(3)
                nf1.metric("Today's Close",     f"${last_actual:.2f}")
                nf2.metric("Forecast (Tomorrow)", f"${next_price:.2f}",
                           delta=f"{chg:+.2f} ({chg_pct:+.2f}%)")
                nf3.metric("Direction", "📈 UP" if chg >= 0 else "📉 DOWN")

            except Exception as e:
                st.warning(f"Could not compute next-day forecast: {e}")

            st.markdown("---")

            # ── Candlestick (raw data) ────────────────────────
            st.markdown("### 🕯️ Historical OHLCV Candlestick")
            df_raw = results["raw_df"]
            if all(c in df_raw.columns for c in ["Open", "High", "Low", "Close"]):
                fig_candle = go.Figure(
                    go.Candlestick(
                        x     = df_raw.index,
                        open  = df_raw["Open"],
                        high  = df_raw["High"],
                        low   = df_raw["Low"],
                        close = df_raw["Close"],
                        increasing_line_color = "#4ade80",
                        decreasing_line_color = "#f87171",
                        name  = ticker,
                    )
                )
                if "Volume" in df_raw.columns:
                    fig_candle = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.8, 0.2],
                        vertical_spacing=0.03,
                    )
                    fig_candle.add_trace(
                        go.Candlestick(
                            x     = df_raw.index,
                            open  = df_raw["Open"],
                            high  = df_raw["High"],
                            low   = df_raw["Low"],
                            close = df_raw["Close"],
                            increasing_line_color = "#4ade80",
                            decreasing_line_color = "#f87171",
                            name  = ticker,
                        ),
                        row=1, col=1,
                    )
                    fig_candle.add_trace(
                        go.Bar(
                            x    = df_raw.index,
                            y    = df_raw["Volume"],
                            name = "Volume",
                            marker_color = "#334155",
                        ),
                        row=2, col=1,
                    )

                fig_candle.update_layout(
                    template      = config.PLOTLY_THEME,
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(13,17,23,0.8)",
                    height        = 450,
                    margin        = dict(t=20, b=20, l=20, r=20),
                    xaxis_rangeslider_visible = False,
                    font = dict(family="Space Grotesk, sans-serif", size=12),
                )
                fig_candle.update_xaxes(gridcolor="#1e2a3a")
                fig_candle.update_yaxes(gridcolor="#1e2a3a")
                st.plotly_chart(fig_candle, use_container_width=True)

            # ── Data Table ────────────────────────────────────
            with st.expander("📋 View Prediction Data Table"):
                out_df = pd.DataFrame({
                    "Date":      dates,
                    "Actual":    np.round(actual, 4),
                    "Predicted": np.round(pred,   4),
                    "Error":     np.round(error,  4),
                }).set_index("Date")
                st.dataframe(out_df.tail(50), use_container_width=True)


# ============================================================
# WELCOME SCREEN (no action yet)
# ============================================================

if not train_btn and not predict_btn:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0d1117,#111827);
                        border:1px solid #1e3a5f; border-radius:14px;
                        padding:2rem; height:100%;">
                <h3 style="color:#38bdf8; margin-top:0;">🚀 Quick Start</h3>
                <ol style="color:#94a3b8; line-height:2.0; font-size:0.9rem;">
                    <li>Select a <strong style="color:#e2e8f0;">ticker symbol</strong> from the sidebar</li>
                    <li>Choose a <strong style="color:#e2e8f0;">date range</strong></li>
                    <li>Click <strong style="color:#38bdf8;">Train Model</strong> to fit the LSTM</li>
                    <li>Click <strong style="color:#f472b6;">Run Prediction</strong> to see results</li>
                </ol>
                <p style="color:#475569; font-size:0.8rem; margin-bottom:0;">
                    Pre-trained models are cached — retrain only when you change the ticker or date range.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#0d1117,#111827);
                        border:1px solid #1e3a5f; border-radius:14px;
                        padding:2rem; height:100%;">
                <h3 style="color:#a78bfa; margin-top:0;">🧠 Model Architecture</h3>
                <table style="width:100%; border-collapse:collapse; font-size:0.85rem; color:#94a3b8;">
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Type</td>
                        <td style="color:#e2e8f0;">Multi-layer LSTM</td></tr>
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Framework</td>
                        <td style="color:#e2e8f0;">PyTorch</td></tr>
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Sequence</td>
                        <td style="color:#e2e8f0;">60-day look-back window</td></tr>
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Target</td>
                        <td style="color:#e2e8f0;">Next-day closing price</td></tr>
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Metrics</td>
                        <td style="color:#e2e8f0;">RMSE · MAE · R²</td></tr>
                    <tr><td style="padding:0.4rem 0; color:#64748b; font-family:JetBrains Mono,monospace;">Optimiser</td>
                        <td style="color:#e2e8f0;">Adam + LR scheduler</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#334155; font-size:0.75rem; '
        'font-family:JetBrains Mono,monospace; letter-spacing:0.05em;">'
        '⚠️ This tool is for educational purposes only. '
        'Past performance does not guarantee future results. '
        'Not financial advice.</p>',
        unsafe_allow_html=True,
    )
