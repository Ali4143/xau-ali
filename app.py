"""Standalone Bloomberg-Dark Dashboard for XAUUSD & Multi-Forex MTF 1-Minute Forecast Engine (v2 Pro UI).

Run on an isolated port (e.g. 8503):
    streamlit run xau_mtf/app.py --server.port 8503
"""
from __future__ import annotations

# Preload lightgbm BEFORE streamlit/scipy/sklearn on Windows to prevent OpenMP DLL access violations
import lightgbm as _lgb_preload  # noqa: F401

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Auto-detect Streamlit Cloud (read-only mode: hides training/optuna/full-fetch so web viewers cannot disturb models)
IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit"
    or os.environ.get("XAU_CLOUD_MODE") == "1"
    or os.path.exists("/mount/src")
)

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if _THIS_DIR.name != "xau_mtf":
    # Support deploying xau_mtf/ directly as the root of a standalone GitHub repo (e.g. /mount/src/xau-ali/)
    import types

    _pkg = types.ModuleType("xau_mtf")
    _pkg.__path__ = [str(_THIS_DIR)]
    sys.modules["xau_mtf"] = _pkg
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
elif str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from xau_mtf import config
import xau_mtf.charts as _xau_charts
import xau_mtf.fetcher.fetch_adapter as _xau_fetch
import xau_mtf.pipeline as _xau_pipe

importlib.reload(_xau_charts)
importlib.reload(_xau_fetch)
importlib.reload(_xau_pipe)

from xau_mtf.charts import build_mtf_forecast_chart
from xau_mtf.dataset import build_chronological_splits
from xau_mtf.eval_v2 import run_v2_evaluation
from xau_mtf.fetcher.fetch_adapter import data_health_report, fetch_and_update, fetch_live_tail, load_m1, load_m15
from xau_mtf.fetcher.vendor.dukascopy_vendor import INSTRUMENTS
from xau_mtf.pipeline import (
    CORE_MODEL_CLASSES,
    compute_live_scorecard,
    evaluate_logged_forecasts_accuracy,
    load_forecast_log,
    run_forecast_isolated,
    run_taaza_if_due_isolated,
    train_all,
)
from xau_mtf.train_monitor import read_training_status
from xau_mtf.tuning.optuna_tuner import load_best_params, tune_lightgbm, tune_tcn

st.set_page_config(
    page_title="XAUUSD & Forex MTF Forecast Terminal",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-contrast Bloomberg Dark CSS for Streamlit 1.67 (Zero white-on-white widgets!)
st.markdown(
    """
    <style>
    :root {
        --background-color: #000000;
        --secondary-background-color: #0d1321;
        --text-color: #F8FAFC;
        --primary-color: #F97316;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #000000 !important;
        color: #F8FAFC !important;
    }
    header[data-testid="stHeader"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #1e293b !important;
    }
    header[data-testid="stHeader"] * { color: #F8FAFC !important; }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b !important;
        color: #F8FAFC !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #F1F5F9 !important;
    }
    /* Inline code pills */
    code {
        background-color: #1e293b !important;
        color: #38BDF8 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid #334155 !important;
    }
    /* Selectboxes, Inputs & Dropdown Menus */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #162033 !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg,
    div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    div[data-baseweb="popover"],
    ul[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
    }
    ul[role="listbox"] li,
    li[role="option"] {
        background-color: #111827 !important;
        color: #F8FAFC !important;
    }
    ul[role="listbox"] li:hover,
    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38BDF8 !important;
    }
    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px !important;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: #F8FAFC !important;
    }
    /* Buttons (Secondary & Primary) */
    .stButton > button,
    button[data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important;
        color: #FFFFFF !important;
        border: 1px solid #475569 !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #273549 !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #F97316 !important;
        color: #000000 !important;
        border: 1px solid #F97316 !important;
        font-weight: 800 !important;
    }
    button[data-testid="stBaseButton-primary"] * {
        color: #000000 !important;
    }
    /* Tabs */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
    }
    .block-container { padding-top: 3.4rem !important; padding-bottom: 1.0rem !important; }
    .hud-card {
        background: #080c14; border: 1px solid #1e293b; border-radius: 8px;
        padding: 10px 14px; font-family: 'Consolas', monospace; margin-bottom: 6px;
    }
    .hud-title { font-size: 0.70rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
    .hud-val { font-size: 1.28rem; font-weight: 700; margin-top: 2px; }
    .hud-sub { font-size: 0.70rem; color: #cbd5e1; margin-top: 2px; }
    .signal-strip {
        background: #090f1c; border: 1px solid #26354d; border-radius: 8px;
        padding: 10px 16px; margin-bottom: 8px; font-family: 'Consolas', monospace;
        display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;
    }
    .sig-badge-long {
        background: rgba(0, 210, 106, 0.18); color: #00D26A; border: 1px solid #00D26A;
        padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem;
    }
    .sig-badge-short {
        background: rgba(248, 113, 113, 0.18); color: #F87171; border: 1px solid #F87171;
        padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem;
    }
    .sig-badge-wait {
        background: rgba(251, 191, 36, 0.16); color: #FBBF24; border: 1px solid #FBBF24;
        padding: 4px 12px; border-radius: 6px; font-weight: 800; font-size: 0.95rem;
    }
    .model-card {
        background: #0b1120; border: 1px solid #1e293b; border-radius: 8px;
        padding: 10px 12px; margin-bottom: 8px; font-family: 'Consolas', monospace;
    }
    .badge-active {
        background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid #34D399;
        border-radius: 4px; padding: 1px 6px; font-size: 0.68rem; font-weight: bold;
    }
    .badge-excl {
        background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid #F87171;
        border-radius: 4px; padding: 1px 6px; font-size: 0.68rem; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.title("🥇 XAUUSD / Forex MTF")
st.sidebar.caption(
    "☁️ Cloud Read-Only Terminal · 24/5 NY 17:00 Rollover"
    if IS_CLOUD
    else "🖥️ Local Workstation (`xau_mtf/`) · 24/5 NY 17:00 Rollover"
)

if "custom_pairs" not in st.session_state:
    st.session_state.custom_pairs = list(INSTRUMENTS.keys())

symbol = st.sidebar.selectbox("Instrument / Forex Pair", st.session_state.custom_pairs, index=0)

auto_live_fetch = st.sidebar.toggle("🟢 Auto 1-Min Live Candles (60s)", value=True)

if not IS_CLOUD:
    with st.sidebar.expander("➕ Add Custom Forex Pair", expanded=False):
        new_pair = st.text_input("6-Letter Pair (e.g. CADJPY, CHFJPY)", value="").upper().strip()
        if st.button("Add Pair", use_container_width=True) and len(new_pair) >= 6:
            if new_pair not in st.session_state.custom_pairs:
                st.session_state.custom_pairs.append(new_pair)
                st.success(f"Added {new_pair}!")

    st.sidebar.subheader("1️⃣ Smart Data Fetcher (Dukascopy + FIFO)")
    fetch_days = st.sidebar.selectbox(
        "Full History Fetch Window",
        [60, 180, 365],
        index=2,
        format_func=lambda d: f"{d} Days (1M → 15M Resampled)",
    )
    col_f1, col_f2 = st.sidebar.columns(2)
    with col_f1:
        if st.button("⚡ Live Sync", use_container_width=True):
            with st.spinner("Pulling latest live 1M bars..."):
                res = fetch_live_tail(symbol=symbol, hours=4)
            st.sidebar.success(f"Synced! Last: {str(res.get('end_utc'))[:16]} UTC")
    with col_f2:
        if st.button("📥 Full Fetch", use_container_width=True):
            with st.spinner(f"Fetching {fetch_days}d {symbol} 1M bars..."):
                prog = st.sidebar.progress(0.0)
                res = fetch_and_update(symbol=symbol, days=fetch_days, progress_cb=lambda p, m: prog.progress(p))
            st.sidebar.success(f"{res['m1_rows']:,} 1M bars saved!")

    st.sidebar.subheader("2️⃣ Model Training & Optuna")
    train_epochs = st.sidebar.slider("Neural Epochs", 10, 60, 20, step=5)
    train_stride = st.sidebar.selectbox("Origin Stride (bars)", [15, 30, 60], index=1)

    col_tr1, col_tr2 = st.sidebar.columns(2)
    with col_tr1:
        if st.button("⚡ Normal Train", use_container_width=True):
            import subprocess

            cmd = [
                sys.executable,
                "-m",
                "xau_mtf.pipeline",
                "--action",
                "train",
                "--symbol",
                symbol,
                "--stride",
                str(train_stride),
                "--epochs",
                str(train_epochs),
            ]
            subprocess.Popen(cmd, cwd=str(_ROOT))
            st.sidebar.success("🚀 Training started! Live progress monitor upar aur VS Code terminal mein chal raha hai.")

    with col_tr2:
        if st.button("🎯 Optuna Tune", use_container_width=True):
            import subprocess

            cmd = [
                sys.executable,
                "-m",
                "xau_mtf.pipeline",
                "--action",
                "optuna",
                "--symbol",
                symbol,
                "--stride",
                str(max(30, train_stride)),
            ]
            subprocess.Popen(cmd, cwd=str(_ROOT))
            st.sidebar.success("🎯 Optuna started! Live progress monitor upar chal raha hai.")

    st.sidebar.subheader("☁️ GitHub Cloud Sync")
    if st.sidebar.button("☁️ Cloud Sync Now", use_container_width=True):
        from xau_mtf.cloud_sync import cloud_sync_now

        with st.spinner("Pushing code, models & data to GitHub..."):
            ok_s, msg_s = cloud_sync_now()
        if ok_s:
            st.sidebar.success(msg_s)
        else:
            st.sidebar.warning(msg_s)

st.sidebar.subheader("3️⃣ Chart, Forecaster & Timezone")
forecaster_choice = st.sidebar.selectbox(
    "Active Forecaster Engine",
    ["tft", "ensemble"],
    index=0,
    format_func=lambda k: "TFT (1-Year Best · 56.5% 3h DA)" if k == "tft" else "Ensemble (v2 Weights)",
)
display_tz = st.sidebar.selectbox(
    "Chart Timezone (Default: UTC)",
    ["UTC", "Asia/Karachi", "America/New_York"],
    index=0,
    format_func=lambda z: "UTC (Session Standard)" if z == "UTC" else ("PKT (UTC+5)" if z == "Asia/Karachi" else z),
)
show_m15_bg = st.sidebar.checkbox("Overlay 15M Candles", value=False)
show_rsi_pane = st.sidebar.checkbox("Show Lower RSI(14) Pane", value=False)
st.sidebar.caption("Individual Model Overlays (Default: Ensemble only):")
show_members = [m_key for m_key in CORE_MODEL_CLASSES if st.sidebar.checkbox(m_key.upper(), value=False, key=f"chk_{m_key}")]

# ── LIVE 3-SECOND TRAINING & OPTUNA MONITOR (Shows Epoch, Loss, Val DA 1h/3h/EOS, Done vs Remaining) ──
@st.fragment(run_every="3s")
def _render_live_training_monitor() -> None:
    st_tr = read_training_status()
    if not st_tr or st_tr.get("state") not in ("running", "error"):
        return
    is_err = st_tr.get("state") == "error"
    title_col = "#F87171" if is_err else "#38BDF8"
    pct = float(st_tr.get("progress_pct", 0.0))
    el_s = int(st_tr.get("elapsed_sec", 0))
    el_str = f"{el_s // 60}m {el_s % 60:02d}s"
    done_str = ", ".join(m.upper() for m in st_tr.get("models_done", [])) or "None yet"
    rem_str = ", ".join(m.upper() for m in st_tr.get("models_remaining", [])) or "Finishing..."

    with st.container(border=True):
        m_c1, m_c2 = st.columns([4, 1])
        with m_c1:
            st.markdown(
                f"<b style='color:{title_col}; font-family:monospace;'>"
                f"{'❌ TRAINING ERROR' if is_err else '⚡ LIVE TRAINING MONITOR'} — "
                f"{st_tr.get('symbol')} ({st_tr.get('action', '').upper()}) · "
                f"Stage: {str(st_tr.get('current_stage', '')).upper()} · {st_tr.get('step_label', '')} · Elapsed {el_str}"
                f"</b>",
                unsafe_allow_html=True,
            )
        with m_c2:
            if not is_err and st.button("🛑 Stop Training", key="btn_stop_tr", use_container_width=True):
                import os
                import signal

                try:
                    os.kill(int(st_tr["pid"]), signal.SIGTERM)
                except Exception:
                    pass
                st.warning("Stop signal sent.")

        st.progress(min(1.0, max(0.0, pct / 100.0)), text=f"Overall Progress: {pct:.1f}%")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Val Loss", f"{st_tr['val_loss']:.4f}" if st_tr.get("val_loss") is not None else "—")
        k2.metric("Best Val Loss", f"{st_tr['best_val_loss']:.4f}" if st_tr.get("best_val_loss") is not None else "—")
        k3.metric("Val DA 1h", f"{st_tr['val_da_1h']:.1f}%" if st_tr.get("val_da_1h") is not None else "—")
        k4.metric("Val DA 3h", f"{st_tr['val_da_3h']:.1f}%" if st_tr.get("val_da_3h") is not None else "—")
        k5.metric("Val DA EOS", f"{st_tr['val_da_eos']:.1f}%" if st_tr.get("val_da_eos") is not None else "—")

        st.caption(f"✅ **Completed:** {done_str}  |  ⏳ **Remaining:** {rem_str}")
        logs = st_tr.get("recent_logs") or []
        if logs:
            st.code("\n".join(logs[-12:]), language="text")


_render_live_training_monitor()


# ── LIVE 60-SECOND FRAGMENT (Auto-fetches new 1M candle every 60s; White Taaza updates only on 15M close) ──
@st.fragment(run_every="60s" if auto_live_fetch else None)
def _render_live_terminal_section() -> None:
    if auto_live_fetch:
        try:
            fetch_live_tail(symbol=symbol, hours=3)
        except Exception:
            pass

    df_m1_l = load_m1(symbol)
    df_m15_l = load_m15(symbol)
    all_fcs_l = [r for r in load_forecast_log(limit=25) if r.get("symbol") == symbol.upper()]
    locked_fcs_l = [r for r in all_fcs_l if not str(r.get("mode", "")).startswith("live")]
    live_fcs_l = [r for r in all_fcs_l if str(r.get("mode", "")).startswith("live")]
    active_fc_l = locked_fcs_l[-1] if locked_fcs_l else (all_fcs_l[-1] if all_fcs_l else None)
    live_refresh_fc_l = (
        live_fcs_l[-1]
        if (live_fcs_l and active_fc_l and live_fcs_l[-1]["origin_utc"] >= active_fc_l["origin_utc"])
        else None
    )

    # White Dashed Taaza refreshes ONLY on completed 15m candle close
    taaza_st = {"due": False, "reason": ""}
    if active_fc_l and not df_m1_l.empty:
        try:
            live_refresh_fc_l, taaza_st = run_taaza_if_due_isolated(
                symbol, df_m1_l, active_fc_l, live_refresh_fc_l, forecaster=forecaster_choice
            )
        except Exception as _tz_err:
            taaza_st = {"due": False, "reason": str(_tz_err)}

    evaluate_logged_forecasts_accuracy(df_m1_l, locked_fcs_l or all_fcs_l, save_file=True)

    # Top Control Bar
    top1, top2, top3, top4 = st.columns([2.2, 1.0, 0.9, 1.3])
    with top1:
        tz_tag = "UTC" if display_tz == "UTC" else ("UTC+5" if display_tz == "Asia/Karachi" else "NY")
        st.markdown(f"#### 🥇 {symbol} · 1-Min 6-Hour Forecast (`{tz_tag}`)")
        if not df_m1_l.empty:
            last_utc = df_m1_l.index.max()
            last_disp = last_utc.tz_convert(display_tz)
            lag_min = max(0, int((datetime.now(timezone.utc) - last_utc.to_pydatetime()).total_seconds() // 60))
            auto_badge = "🟢 Auto 1m ON" if auto_live_fetch else "⚪ Auto OFF"
            lag_txt = f"{auto_badge} · Live ({lag_min}m lag · TradingView)" if lag_min <= 3 else f"{auto_badge} · Lag: {lag_min}m"
            st.caption(
                f"Live: **${float(df_m1_l['close'].iloc[-1]):,.2f}** · Last Bar: `{last_disp:%Y-%m-%d %H:%M} {tz_tag}` · {lag_txt}"
            )
    with top2:
        hist_opts = {
            "Forecast Focus (30m)": 30,
            "Last 1 Hour": 60,
            "Last 4 Hours": 240,
            "Last 12 Hours": 720,
            "Last 24 Hours": 1440,
        }
        hist_lbl = st.selectbox("Past Candles View", list(hist_opts.keys()), index=0, label_visibility="collapsed")
        history_minutes = hist_opts[hist_lbl]
    with top3:
        if st.button("🔄 Sync Live", use_container_width=True):
            with st.spinner("Syncing live TradingView candles..."):
                fetch_live_tail(symbol=symbol)
            st.rerun()
    with top4:
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🔮 Lock 6H", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"Locking new 6H forecast ({forecaster_choice.upper()})..."):
                        run_forecast_isolated(symbol=symbol, mode="manual", forecaster=forecaster_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
        with b_col2:
            if st.button("🔄 Taaza", use_container_width=True):
                try:
                    with st.spinner(f"Generating white dashed Taaza ({forecaster_choice.upper()})..."):
                        run_forecast_isolated(symbol=symbol, mode="live_refresh", forecaster=forecaster_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"Taaza failed: {e}")

    if taaza_st.get("due"):
        st.caption(f"🔄 **White Taaza Auto-Refreshed:** {taaza_st.get('why')} (Orange Locked Forecast unchanged).")
    elif taaza_st.get("reason"):
        st.caption(f"🔄 **White Taaza Status:** Active · {taaza_st.get('reason')}")

    # Directional Signal Strip (uses latest Taaza if present, else Locked Forecast)
    sig_fc = live_refresh_fc_l or active_fc_l
    if sig_fc and sig_fc.get("q50"):
        P0 = float(sig_fc["P0"])
        dm = sig_fc.get("direction_meta") or {}
        p1h = float((dm.get("1h") or {}).get("p_up", 0.5))
        p3h = float((dm.get("3h") or {}).get("p_up", 0.5))
        peos = float((dm.get("EOS") or {}).get("p_up", 0.5))
        p_avg = float(np.mean([p1h, p3h, peos]))
        fc_end = float(sig_fc["q50"][-1])
        delta_end = fc_end - P0

        if p_avg >= 0.55 and delta_end > 0:
            sig_cls, sig_txt = "sig-badge-long", f"🟢 LONG BIAS · {p_avg*100:.1f}% UP"
        elif p_avg <= 0.45 and delta_end < 0:
            sig_cls, sig_txt = "sig-badge-short", f"🔴 SHORT BIAS · {(1-p_avg)*100:.1f}% DOWN"
        else:
            sig_cls, sig_txt = "sig-badge-wait", f"🟡 NEUTRAL / WAIT · {max(p_avg, 1-p_avg)*100:.1f}%"

        th = float(sig_fc.get("target_high", P0))
        tl = float(sig_fc.get("target_low", P0))
        zz = sig_fc.get("zigzag_path") or sig_fc["q50"]
        f_times = pd.to_datetime(sig_fc["times"], utc=True).tz_convert(display_tz)
        t_pk = f_times[int(np.argmax(zz))]
        t_dp = f_times[int(np.argmin(zz))]
        tz_tag = "UTC" if display_tz == "UTC" else ("UTC+5" if display_tz == "Asia/Karachi" else "NY")

        st.markdown(
            f"""<div class="signal-strip">
                <span class="{sig_cls}">{sig_txt}</span>
                <span><b>1H Up:</b> <b style="color:{'#34D399' if p1h>=0.5 else '#F87171'};">{p1h*100:.1f}%</b></span>
                <span><b>3H Up:</b> <b style="color:{'#34D399' if p3h>=0.5 else '#F87171'};">{p3h*100:.1f}%</b></span>
                <span><b>6H Up:</b> <b style="color:{'#34D399' if peos>=0.5 else '#F87171'};">{peos*100:.1f}%</b></span>
                <span><b>▲ Peak:</b> <b style="color:#34D399;">${th:,.2f}</b> ({t_pk:%H:%M} {tz_tag})</span>
                <span><b>▼ Dip:</b> <b style="color:#F87171;">${tl:,.2f}</b> ({t_dp:%H:%M} {tz_tag})</span>
                <span><b>Close q50:</b> <b style="color:#F97316;">${fc_end:,.2f}</b> ({delta_end:+.2f})</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Compact HUD (tracks live 1M candles against latest Taaza or Locked forecast)
    v2_f = config.REPORTS_DIR / "v2" / "metrics_v2.json"
    v1_f = config.REPORTS_DIR / "metrics.json"
    rep_l = (
        json.loads(v2_f.read_text(encoding="utf-8"))
        if v2_f.exists()
        else (json.loads(v1_f.read_text(encoding="utf-8")) if v1_f.exists() else None)
    )
    sc = compute_live_scorecard(df_m1_l, live_refresh_fc_l or active_fc_l) if not df_m1_l.empty else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mp = sc.get("match_pct", 0.0)
        mcol = "#34D399" if mp >= 78 else ("#38BDF8" if mp >= 62 else "#FBBF24")
        st.markdown(
            f"""<div class="hud-card">
            <div class="hud-title">🎯 Causal Path Match %</div>
            <div class="hud-val" style="color:{mcol};">{mp:.1f}%</div>
            <div class="hud-sub">{sc.get('match_status', 'WAITING')} · Live MAE ${sc.get('mae_live_usd', 0):.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        hs = sc.get("heuristic_odds_score", 50.0)
        hcol = "#34D399" if hs >= 58 else ("#F87171" if hs <= 42 else "#FBBF24")
        st.markdown(
            f"""<div class="hud-card">
            <div class="hud-title">⚡ Odds% (HEURISTIC, not validated)</div>
            <div class="hud-val" style="color:{hcol};">{hs:.1f} / 100</div>
            <div class="hud-sub">Tick-Flow Proxy: {sc.get('bid_proxy_pct', 50):.1f}% · RSI(14): {sc.get('rsi_curr', 50):.1f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        tp = sc.get("confluence_trust_pct", 50.0)
        tcol = "#34D399" if tp >= 70 else ("#F87171" if tp < 50 else "#FBBF24")
        st.markdown(
            f"""<div class="hud-card">
            <div class="hud-title">⚖️ Confluence% (HEURISTIC, not validated)</div>
            <div class="hud-val" style="color:{tcol};">{tp:.1f}%</div>
            <div class="hud-sub">{sc.get('confluence_verdict', 'WAITING')}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        gate = (rep_l or {}).get("deployment_gate", {})
        verdict = gate.get("verdict", "UNTRAINED")
        dash_lbl = gate.get("dashboard_label", "Research output, not verified.")
        gcol = "#34D399" if verdict == "DEPLOYABLE" else "#FBBF24"
        ens_row = next((r for r in (rep_l or {}).get("models", []) if r["model"] == "ensemble"), {})
        st.markdown(
            f"""<div class="hud-card">
            <div class="hud-title">🏆 Strict v2 Gate</div>
            <div class="hud-val" style="color:{gcol};">{verdict}</div>
            <div class="hud-sub">{dash_lbl} · MAE ${ens_row.get('MAE_USD', 0):.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Full-Screen Interactive Forecast Chart
    if not df_m1_l.empty:
        fig = build_mtf_forecast_chart(
            df_m1=df_m1_l,
            df_m15=df_m15_l,
            active_fc=active_fc_l,
            live_refresh_fc=live_refresh_fc_l,
            past_forecasts=all_fcs_l,
            show_members=show_members,
            show_m15_bg=show_m15_bg,
            show_rsi_pane=show_rsi_pane,
            history_minutes=history_minutes,
            display_tz=display_tz,
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No local data found for {symbol}. Open the left sidebar (>) and click **📥 Full Fetch**.")


_render_live_terminal_section()

# Load static reports for bottom analytics tabs
v2_file = config.REPORTS_DIR / "v2" / "metrics_v2.json"
v1_file = config.REPORTS_DIR / "metrics.json"
rep_data = (
    json.loads(v2_file.read_text(encoding="utf-8"))
    if v2_file.exists()
    else (json.loads(v1_file.read_text(encoding="utf-8")) if v1_file.exists() else None)
)
df_m1_hist = load_m1(symbol)
all_fcs_hist = [r for r in load_forecast_log(limit=25) if r.get("symbol") == symbol.upper()]
accuracy_history = evaluate_logged_forecasts_accuracy(df_m1_hist, all_fcs_hist, save_file=False)

# ── PROFESSIONAL BOTTOM ANALYTICS TABS ────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 v2 TEST Scorecard",
        "🔁 Walk-Forward Folds",
        "📜 Forecast Accuracy History",
        "🎛️ Ensemble Weights & Optuna",
        "🩺 Data Health & Coverage",
    ]
)

with tab1:
    if rep_data:
        gate = rep_data["deployment_gate"]
        if gate.get("deployable"):
            st.success(f"**Gate Verdict: {gate['verdict']}** — {gate['reason']}")
        else:
            st.warning(f"**Gate Verdict: {gate['verdict']}** — `{gate.get('dashboard_label', 'Research output, not verified.')}`")
            for fc_cond in gate.get("failed_conditions", []):
                st.caption(f"❌ {fc_cond}")

        wf_n = (rep_data.get("walk_forward") or {}).get("n_folds", 0)
        rows = []
        for r in rep_data["models"]:
            da1 = r.get("DA_1h") or {}
            da3 = r.get("DA_3h") or {}
            dae = r.get("DA_EOS") or {}
            hac = r.get("dm_hac") or {}
            nov = r.get("dm_non_overlapping") or {}
            dp = r.get("directional_proxy_1h") or {}
            rows.append(
                {
                    "Model": r["model"].upper(),
                    "MAE ($)": r["MAE_USD"],
                    "RMSE ($)": r["RMSE_USD"],
                    "80% Cov %": r["coverage_80_pct"],
                    "DA 1h": da1.get("label", r.get("DA_1h_pct")),
                    "Excl Flat": da1.get("excluded_flat", 0),
                    "DA 3h": da3.get("label", r.get("DA_3h_pct")),
                    "DA EOS": dae.get("label", r.get("DA_EOS_pct")),
                    "Skill %": r.get("skill_edge_vs_nochange_pct"),
                    "HAC p (Holm)": hac.get("p_holm"),
                    "Non-Ovlp p (Holm)": nov.get("p_holm"),
                    "Non-Ovlp n": nov.get("n"),
                    "Folds": wf_n,
                    "1h Gross ($)": dp.get("gross_usd_per_trade", r.get("trading_proxy_1h_gross_usd")),
                    "1h Net ($)*": dp.get("net_usd_per_trade", r.get("trading_proxy_1h_net_usd")),
                    "Trades": dp.get("n_trades", 0),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("*`1h Net ($)`: Directional proxy, not a trading backtest. Cost applied ONLY on non-flat positions.")
    else:
        st.info("No test evaluation yet. Open sidebar and click **⚡ Normal Train**.")

with tab2:
    wf = (rep_data or {}).get("walk_forward") or {}
    if wf.get("folds"):
        wf_c1, wf_c2, wf_c3 = st.columns(3)
        wf_c1.metric("Positive Folds", f"{wf.get('positive_folds')}/{wf.get('n_folds')} ({wf.get('positive_fold_pct')}%)")
        wf_c2.metric("Mean Fold MAE", f"${wf.get('mean_mae_usd')}", f"±${wf.get('std_mae_usd')}")
        wf_c3.metric("Mean Fold Skill", f"{wf.get('mean_skill_pct')}%", f"±{wf.get('std_skill_pct')}%")
        st.dataframe(pd.DataFrame(wf["folds"]), use_container_width=True, hide_index=True)
    else:
        st.info("No walk-forward folds computed yet.")

with tab3:
    if accuracy_history:
        st.caption("Har past forecast ki real candles aane ke baad realized accuracy automatically calculate aur save hoti hai.")
        st.dataframe(pd.DataFrame(reversed(accuracy_history)), use_container_width=True, hide_index=True)
    else:
        st.info("No forecasts logged yet. Click **🔮 Generate 6H Forecast**.")

with tab4:
    weights_d = (rep_data or {}).get("ensemble_weights") or {}
    w_diag = (rep_data or {}).get("ensemble_weight_diagnostics") or {}
    cands = w_diag.get("candidates") or {}

    col_w1, col_w2 = st.columns([1.3, 1.0])
    with col_w1:
        st.markdown("##### 📊 Frozen Validation Ensemble Weights (v2)")
        if weights_d:
            names = [k.upper() for k in weights_d.keys()]
            vals = [round(float(v) * 100.0, 2) for v in weights_d.values()]
            colors = ["#34D399" if v > 0 else "#475569" for v in vals]
            w_fig = go.Figure(
                go.Bar(
                    x=vals,
                    y=names,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in vals],
                    textposition="outside",
                )
            )
            w_fig.update_layout(
                height=260,
                margin=dict(l=10, r=45, t=10, b=10),
                paper_bgcolor="#080c14",
                plot_bgcolor="#080c14",
                font=dict(color="#E2E8F0", size=12),
                xaxis=dict(range=[0, max(100, max(vals) + 15)], title="Weight %", gridcolor="#1e293b"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(w_fig, use_container_width=True)

            for k, w_val in weights_d.items():
                ci = cands.get(k, {})
                is_excl = bool(ci.get("excluded", False)) or w_val <= 0
                badge = '<span class="badge-excl">EXCLUDED (0%)</span>' if is_excl else f'<span class="badge-active">ACTIVE ({w_val*100:.1f}%)</span>'
                val_mae = ci.get("val_mae_usd")
                val_pb = ci.get("val_pinball")
                mae_txt = f"Val MAE: ${val_mae:.2f} · Pinball: {val_pb:.4f}" if val_mae is not None else "Trained Member"
                st.markdown(
                    f"""<div class="model-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#F8FAFC;">{k.upper()}</b> {badge}
                    </div>
                    <div style="font-size:0.72rem; color:#94a3b8; margin-top:4px;">{mae_txt}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No ensemble weights computed yet.")

    with col_w2:
        st.markdown("##### 🎯 Optuna Hyperparameter Tuning Status")
        best_p = load_best_params()
        if best_p:
            for m_k, info in best_p.items():
                p_items = " · ".join(f"{pk}={pv}" for pk, pv in (info.get("params") or {}).items())
                st.markdown(
                    f"""<div class="model-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#38BDF8;">{m_k.upper()}</b>
                        <span class="badge-active">TUNED ({info.get('n_trials', 0)} trials)</span>
                    </div>
                    <div style="font-size:0.72rem; color:#34D399; margin-top:4px;">Best Val Pinball: {info.get('val_pinball')}</div>
                    <div style="font-size:0.68rem; color:#94a3b8; margin-top:4px;">{p_items}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """<div class="model-card" style="border-color:#334155;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b style="color:#E2E8F0;">OPTUNA BAYESIAN TUNER</b>
                    <span class="badge-active" style="border-color:#38BDF8; color:#38BDF8; background:rgba(56,189,248,0.12);">DEFAULT PARAMS ACTIVE</span>
                </div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:6px;">
                    Abhi models verified default parameters par chal rahe hain.<br>
                    Optuna tuning chalane ke liye sidebar (<b>&gt;</b>) mein <b>🎯 Optuna Tune</b> dabayein.
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

with tab5:
    df1_h = load_m1(symbol)
    df15_h = load_m15(symbol)
    dh = data_health_report(df1_h, df15_h, symbol=symbol)
    if dh["m1_rows"] > 0:
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Health Status", dh["status"])
        h2.metric("1M Bars", f"{dh['m1_rows']:,}", f"{dh['span_days']} Days")
        h3.metric("15M Bars (Derived)", f"{dh['m15_rows']:,}", "100% Synced")
        h4.metric("Missing Intraday 1M", f"{dh['missing_1m_bars_intraday']:,}", f"Max Gap: {dh['largest_unexpected_gap_min']}m")
        h5.metric("Duplicates / NaNs", f"{dh['duplicates']} / {dh['nans']}", f"Bad OHLC: {dh['bad_ohlc']}")

        fifo_pct = min(1.0, dh["span_days"] / max(1.0, float(dh["fifo_limit_days"])))
        st.caption(
            f"**Range (UTC):** `{dh['start_utc'][:19]}` → `{dh['end_utc'][:19]}` | "
            f"**FIFO Window:** `{dh['span_days']} / {dh['fifo_limit_days']} days` (1.5Y Max) | "
            f"**Weekend Gaps:** `{dh['weekend_gaps']}` | **17:00 NY Maint Gaps:** `{dh['maintenance_gaps']}` | "
            f"**Real Volume Non-Zero:** `{dh['real_volume_nonzero']}` (Tick-volume proxy)"
        )
        st.progress(fifo_pct)
        if dh["daily_recent"]:
            st.markdown("##### 📅 Recent 30 Days Bar Coverage (1M Bars / Day)")
            st.dataframe(pd.DataFrame(reversed(dh["daily_recent"])), use_container_width=True, hide_index=True)
    else:
        st.warning("No local data stored yet.")
