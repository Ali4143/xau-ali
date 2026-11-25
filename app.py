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


def _resolve_auto_best_forecaster() -> tuple[str, str]:
    """Pick #1 best forecaster automatically from latest metrics_v2.json (Ensemble vs individual models)."""
    v2_p = config.REPORTS_DIR / "v2" / "metrics_v2.json"
    if v2_p.exists():
        try:
            rep = json.loads(v2_p.read_text(encoding="utf-8"))
            cands = [
                r
                for r in (rep.get("models") or [])
                if str(r.get("model", "")).lower() in ("ensemble", "tft", "lightgbm", "tcn", "wavelet_bilstm", "timemixer")
                and r.get("MAE_USD") is not None
            ]
            if cands:
                best = min(cands, key=lambda r: float(r["MAE_USD"]))
                m_key = str(best["model"]).lower()
                da3 = best.get("DA_3h_pct")
                da_txt = f"{float(da3):.1f}% 3h DA" if da3 is not None else f"MAE ${float(best['MAE_USD']):.2f}"
                return m_key, f"Auto Best → {m_key.upper()} ({da_txt})"
        except Exception:
            pass
    return "tft", "Auto Best → TFT (Default)"


_auto_best_key, _auto_best_lbl = _resolve_auto_best_forecaster()
_fc_sel = st.sidebar.selectbox(
    "Active Forecaster Engine",
    ["auto", "tft", "ensemble", "lightgbm", "tcn", "wavelet_bilstm"],
    index=0,
    format_func=lambda k: (
        f"🏆 {_auto_best_lbl}"
        if k == "auto"
        else (
            "TFT (Neural Transformer)"
            if k == "tft"
            else ("Ensemble (Multi-Model Blend)" if k == "ensemble" else k.upper())
        )
    ),
)
forecaster_choice = _auto_best_key if _fc_sel == "auto" else _fc_sel
display_tz = st.sidebar.selectbox(
    "Chart Timezone (Default: UTC)",
    ["UTC", "Asia/Karachi", "America/New_York"],
    index=0,
    format_func=lambda z: "UTC (Session Standard)" if z == "UTC" else ("PKT (UTC+5)" if z == "Asia/Karachi" else z),
)
num_targets = st.sidebar.selectbox(
    "🎯 Target High / Low Levels",
    [1, 2],
    index=0,
    format_func=lambda n: "1 Target (Default · T1)" if n == 1 else "2 Targets (T1 + T2 Extended)",
)
show_m15_bg = st.sidebar.checkbox("Overlay 15M Candles", value=False)
show_rsi_pane = st.sidebar.checkbox("Show Lower RSI(14) Pane", value=False)
show_liquidity = st.sidebar.checkbox("🏦 Show Liquidity Pools & 70%+ Signals", value=True)
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
            if is_err:
                if st.button("✖ Dismiss Error", key="btn_dismiss_tr_err", use_container_width=True):
                    from xau_mtf.train_monitor import clear_training_status

                    clear_training_status("Dismissed by user — best completed checkpoints preserved.")
                    st.rerun()
            elif st.button("🛑 Stop Training", key="btn_stop_tr", use_container_width=True):
                import os
                import signal
                from xau_mtf.train_monitor import clear_training_status

                try:
                    os.kill(int(st_tr["pid"]), signal.SIGTERM)
                except Exception:
                    pass
                clear_training_status("Stopped by user — best completed checkpoints saved!")
                st.rerun()

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

    # Live Model Comparison & "Kis Par Bharosa Karein?" Strip (above chart — tracks latest Taaza or Locked)
    track_fc_l = live_refresh_fc_l or active_fc_l
    if track_fc_l and not df_m1_l.empty and track_fc_l.get("times"):
        fc_eng_lbl = str(track_fc_l.get("forecaster") or track_fc_l.get("forecaster_engine") or forecaster_choice or "tft").upper()
        p0_l = float(track_fc_l.get("P0", df_m1_l["close"].iloc[-1]))
        p_now = float(df_m1_l["close"].iloc[-1])
        t_now_utc = pd.Timestamp(df_m1_l.index[-1])
        if t_now_utc.tzinfo is None:
            t_now_utc = t_now_utc.tz_localize("UTC")
        fc_ts_utc = pd.to_datetime(track_fc_l["times"], utc=True)
        # Find nearest elapsed forecast index to current live candle
        diffs_sec = np.abs((fc_ts_utc - t_now_utc).total_seconds().to_numpy())
        cur_idx = int(np.argmin(diffs_sec)) if len(diffs_sec) else 0
        actual_dir = 1 if (p_now - p0_l) > 0.15 else (-1 if (p_now - p0_l) < -0.15 else 0)

        rep_models = {str(r.get("model", "")).lower(): r for r in (rep_l or {}).get("models", [])}
        nc_mae = float((rep_models.get("no_change") or {}).get("MAE_USD") or 0.0)
        mem_map = track_fc_l.get("member_q50") or {}
        pills_html = []
        best_m_name, best_m_skill, best_m_da = "TFT", -999.0, 50.0
        live_status_by_m: dict[str, tuple[str, float]] = {}

        for m_k in ("tft", "lightgbm", "tcn", "wavelet_bilstm", "timemixer"):
            rm = rep_models.get(m_k) or (rep_models.get("lgb") if m_k == "lightgbm" else None) or {}
            da3_raw = rm.get("DA_3h_pct")
            if da3_raw is None and isinstance(rm.get("DA_3h"), dict):
                da3_raw = rm["DA_3h"].get("da_pct")
            elif da3_raw is None and isinstance(rm.get("DA_3h"), (int, float)):
                da3_raw = float(rm["DA_3h"]) * (100.0 if float(rm["DA_3h"]) <= 1.0 else 1.0)
            mae_m = rm.get("MAE_USD")
            sk_val = (
                (1.0 - float(mae_m) / nc_mae) * 100.0
                if (mae_m is not None and nc_mae > 0)
                else rm.get("Skill_MAE_pct")
            )
            da_s = f"{float(da3_raw):.1f}%" if da3_raw is not None else "N/A"
            sk_s = f"{float(sk_val):+.2f}%" if sk_val is not None else "N/A"
            if sk_val is not None and float(sk_val) > best_m_skill:
                best_m_skill = float(sk_val)
                best_m_name = m_k.upper()
                best_m_da = float(da3_raw) if da3_raw is not None else 50.0

            arr = mem_map.get(m_k) or (mem_map.get("lgb") if m_k == "lightgbm" else None)
            if arr and cur_idx < len(arr):
                p_pred = float(arr[cur_idx])
                err_now = abs(p_pred - p_now)
                p_dir = 1 if (p_pred - p0_l) > 0.15 else (-1 if (p_pred - p0_l) < -0.15 else 0)
                if actual_dir == 0 or p_dir == 0:
                    st_ico, st_col = "➖", "#94A3B8"
                elif p_dir == actual_dir:
                    st_ico, st_col = "✅", "#34D399"
                else:
                    st_ico, st_col = "⚠️", "#F87171"
                live_status_by_m[m_k.upper()] = (st_ico, err_now)
                err_txt = f"Live err ${err_now:.2f}"
            else:
                st_ico, st_col = "•", "#94A3B8"
                err_txt = "—"

            is_best = m_k.upper() == best_m_name
            bdr = "1px solid #34D399" if is_best else "1px solid #1E293B"
            bg = "rgba(16,185,129,0.12)" if is_best else "#0D1526"
            disp_lbl = "LGB" if m_k == "lightgbm" else m_k.upper()
            pills_html.append(
                f'<span style="background:{bg};border:{bdr};border-radius:6px;padding:3px 8px;font-size:11px;color:#E2E8F0;">'
                f'<b style="color:{st_col};">{st_ico} {disp_lbl}</b> '
                f'<span style="color:#94A3B8;">1Y DA:</span><b>{da_s}</b> '
                f'<span style="color:#94A3B8;">Skill:</span><b>{sk_s}</b> '
                f'<span style="color:#64748B;">· {err_txt}</span></span>'
            )

        b_ico, b_err = live_status_by_m.get(best_m_name, ("•", 0.0))
        if b_ico == "✅":
            trust_note = f"🟢 {best_m_name} Live Candles ke saath ALIGNED hai (err ${b_err:.2f}) — Primary Bias & Targets isi par dekhein"
        elif b_ico == "⚠️":
            trust_note = f"🟡 {best_m_name} abhi live candle se thoda diverging hai (err ${b_err:.2f}) — Candle confirmation ka intezar karein"
        else:
            trust_note = f"🔵 {best_m_name} 1-Year Best Model hai (3h DA {best_m_da:.1f}%, Skill {best_m_skill:+.2f}%) — Live move monitor ho rahi hai"

        st.markdown(
            f"""<div style="background:#0B1220;border:1px solid #1E293B;border-radius:8px;padding:8px 12px;margin:6px 0 8px 0;">
            <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;">
                <span style="font-size:12px;color:#F8FAFC;">
                    🟠 <b>Active Orange/White Line:</b> <b style="color:#F97316;">{fc_eng_lbl} (Real Zig-Zag)</b>
                    &nbsp;|&nbsp; 🏆 <b>Kis Par Bharosa Karein?</b> <b style="color:#34D399;">{best_m_name}</b>
                    <span style="color:#CBD5E1;">({trust_note})</span>
                </span>
                <span style="font-size:11px;color:#94A3B8;">Research Output · 1Y Best DA ~{best_m_da:.1f}% (confirmation zaroori)</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:6px;">{"".join(pills_html)}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # 🏦 Institutional Footprint, Cross-Market Macro (DXY/Silver/US10Y) & 1:2 RR Radar
    inst_sum = None
    if not df_m1_l.empty:
        from xau_mtf.institutional import build_institutional_summary, fetch_cross_market_snapshot

        cross_snap = fetch_cross_market_snapshot(df_xau_m1=df_m1_l, max_age_sec=55, allow_network=auto_live_fetch)
        inst_sum = build_institutional_summary(df_m1=df_m1_l, active_fc=live_refresh_fc_l or active_fc_l, cross_snap=cross_snap)
        if inst_sum:
            c_assets = (cross_snap or {}).get("assets") or {}
            macro_pills = []
            for m_sym, m_lbl in (("DXY", "💵 DXY"), ("XAGUSD", "🥈 Silver"), ("US10Y", "🏛️ US10Y")):
                a = c_assets.get(m_sym)
                if a:
                    imp = a.get("gold_impact", "NEUTRAL")
                    icol = "#34D399" if imp == "BULLISH" else ("#F87171" if imp == "BEARISH" else "#94A3B8")
                    macro_pills.append(
                        f'<span style="background:#0D1526;border:1px solid #1E293B;border-radius:6px;padding:2px 8px;font-size:11px;color:#E2E8F0;">'
                        f'<b>{m_lbl}:</b> {a["price"]:,.3f} <span style="color:{icol};">({a["chg15_pct"]:+.2f}% 15m → {imp})</span></span>'
                    )
            if not macro_pills:
                macro_pills.append('<span style="font-size:11px;color:#94A3B8;">Cross-market live feed loading...</span>')

            rr_txt = (
                f"Entry <b>${inst_sum['p_now']:,.2f}</b> · SL <b style='color:#F87171;'>${inst_sum['sl_price']:,.2f}</b> (-${inst_sum['sl_dist']:.2f}) · "
                f"TP <b style='color:#34D399;'>${inst_sum['tp_price']:,.2f}</b> (+${inst_sum['tp_dist']:.2f} · 1:2 RR)"
                if inst_sum.get("sl_price") is not None
                else f"No Active Setup (ATR15: ${inst_sum['atr15']:.2f} · Suggested 1:2 RR = ±${inst_sum['sl_dist']:.2f} / ±${inst_sum['tp_dist']:.2f})"
            )
            lvls = inst_sum.get("levels") or {}
            pat_d = inst_sum.get("patterns") or {}
            last_pat = pat_d.get("last_candle_pattern", "Normal")
            lad = pat_d.get("active_ladder")
            if lad:
                t_lvls = lad.get("targets") or {}
                lad_html = (
                    f"<b style='color:{'#34D399' if lad['dir']==1 else '#F87171'};'>{lad['side_label']} {lad['pattern']} ({lad['confidence']:.0f}%)</b> "
                    f"· <b style='color:#FACC15;'>{lad['status_text']}</b> "
                    f"<span style='color:#94A3B8;'>[T1: ${t_lvls.get(1,0):,.2f} · T2: ${t_lvls.get(2,0):,.2f} · T3: ${t_lvls.get(3,0):,.2f} · T4: ${t_lvls.get(4,0):,.2f} · T5: ${t_lvls.get(5,0):,.2f}]</span>"
                )
            else:
                lad_html = "<span style='color:#94A3B8;'>No ≥70% Pattern Active — waiting for clean candle setup (Doji / Hammer / Engulfing + Sweep)</span>"

            st.markdown(
                f"""<div style="background:#0A101E;border:1px solid #1E293B;border-radius:8px;padding:8px 12px;margin:0 0 8px 0;">
                <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-bottom:5px;">
                    <span style="font-size:12px;color:#F8FAFC;">
                        🏦 <b>Institutional Radar:</b>
                        <b style="color:{inst_sum['verdict_color']};">{inst_sum['verdict']}</b>
                        <span style="color:#94A3B8;">(Score: {inst_sum['inst_score']:+.1f})</span>
                        &nbsp;|&nbsp; <b>Regime:</b> <b style="color:{inst_sum['regime_color']};">{inst_sum['regime_label']}</b>
                        <span style="color:#94A3B8;">(ER {inst_sum['efficiency_ratio']:.2f} · VolZ {inst_sum['vol_z_60']:+.1f})</span>
                    </span>
                    <span style="font-size:11px;color:#CBD5E1;"><b>VPIN Toxicity:</b> {inst_sum['vpin_proxy']*100:.1f}% · <b>CVD(20):</b> {inst_sum['cvd_20']:+.2f} · <b>{inst_sum['sweep_label']}</b></span>
                </div>
                <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin-bottom:5px;">
                    {"".join(macro_pills)}
                    <span style="background:#0D1526;border:1px solid #334155;border-radius:6px;padding:2px 8px;font-size:11px;color:#E2E8F0;"><b>SMT:</b> {inst_sum['smt_status']}</span>
                </div>
                <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;font-size:11px;color:#E2E8F0;margin-bottom:5px;padding-top:4px;border-top:1px solid #162033;">
                    <span>🕯️ <b>Last Candle:</b> <b style="color:#38BDF8;">{last_pat}</b> &nbsp;|&nbsp; ⚡ <b>70%+ Pattern Ladder:</b> {lad_html}</span>
                </div>
                <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;font-size:11px;color:#94A3B8;">
                    <span>🎯 <b>1:2 RR Plan:</b> <span style="color:#E2E8F0;">{rr_txt}</span></span>
                    <span>🏦 <b>Pools:</b> PDH ${lvls.get('PDH',0):,.2f} / PDL ${lvls.get('PDL',0):,.2f} · Asian ${lvls.get('Asian_L',0):,.2f}–${lvls.get('Asian_H',0):,.2f} · London ${lvls.get('London_L',0):,.2f}–${lvls.get('London_H',0):,.2f}</span>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # 🤖 Auto-Tick Paper Trading Bot every 60s (manages open trades + opens new trades when RUNNING)
    from xau_mtf.paper_bot import load_bot_state, tick_paper_bot

    if not df_m1_l.empty and not IS_CLOUD:
        bot_state_l = tick_paper_bot(df_m1_l, active_fc_l, live_refresh_fc_l, inst_sum)
    else:
        bot_state_l = load_bot_state()

    if bot_state_l.get("running") or bot_state_l.get("open_trades"):
        b_run = bool(bot_state_l.get("running"))
        b_bal = float(bot_state_l.get("balance", 5000.0))
        b_eq = float(bot_state_l.get("equity", b_bal))
        b_fl = b_eq - b_bal
        b_open = bot_state_l.get("open_trades") or []
        b_badge = "<b style='color:#34D399;'>🟢 BOT RUNNING</b>" if b_run else "<b style='color:#FBBF24;'>⏸️ BOT PAUSED</b>"
        open_txt = " | ".join(
            f"{t['track']} {t['side']} {t['lot_size']}L @ ${t['entry_price']:,.2f} (Fl: {float(t.get('floating_pnl_usd', 0)):+.2f}$)"
            for t in b_open
        ) or "No open trades (waiting for ≥60% setup)"
        st.markdown(
            f"""<div style="background:#07131D;border:1px solid {'#10B981' if b_run else '#334155'};border-radius:8px;padding:6px 12px;margin:0 0 8px 0;font-size:11.5px;color:#E2E8F0;display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;">
                <span>🤖 {b_badge} · <b>Bal:</b> ${b_bal:,.2f} · <b>Eq:</b> ${b_eq:,.2f} (<b style="color:{'#34D399' if b_fl>=0 else '#F87171'};">{b_fl:+.2f}$</b>) · <b>Open ({len(b_open)}/2):</b> {open_txt}</span>
                <span style="color:#94A3B8;">{bot_state_l.get('last_event', '')}</span>
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
            show_liquidity=show_liquidity,
            num_targets=num_targets,
            inst_summary=inst_sum,
            paper_open_trades=bot_state_l.get("open_trades") or [],
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
tab_bot, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🤖 Paper Bot & Diagnostics",
        "📊 v2 TEST Scorecard",
        "🔁 Walk-Forward Folds",
        "📜 Forecast Accuracy History",
        "🎛️ Ensemble Weights & Optuna",
        "🩺 Data Health & Coverage",
    ]
)

with tab_bot:
    from xau_mtf.paper_bot import (
        close_all_open_trades,
        compute_diagnostics,
        load_bot_state,
        load_trade_history,
        reset_bot_account,
        save_bot_state,
    )

    bst = load_bot_state()
    hist_trades = load_trade_history(limit=500)
    diag = compute_diagnostics(hist_trades)

    st.subheader("🤖 Dual-Track Paper Trading Bot (🟠 6H Fix + ⚪ 15m Taaza)")
    if bst.get("daily_paused_reason"):
        st.error(f"⛔ **Bot Auto-Paused:** {bst['daily_paused_reason']}")

    # Control buttons (local workstation only; cloud is read-only)
    if not IS_CLOUD:
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            if not bst.get("running"):
                if st.button("▶️ Start Bot", type="primary", use_container_width=True, key="pb_start"):
                    bst["running"] = True
                    bst["daily_paused_reason"] = None
                    bst["last_event"] = "Bot STARTED by user — monitoring 6H Fix & 15m Taaza every 60s."
                    save_bot_state(bst)
                    st.rerun()
            else:
                if st.button("⏸️ Pause Bot", use_container_width=True, key="pb_pause"):
                    bst["running"] = False
                    bst["last_event"] = "Bot PAUSED by user (open trades still monitored)."
                    save_bot_state(bst)
                    st.rerun()
        with bc2:
            if st.button("🛑 Close All & Stop", use_container_width=True, key="pb_close_all"):
                close_all_open_trades(df_m1_hist, reason="MANUAL_STOP")
                st.rerun()
        with bc3:
            if st.button("🔄 Reset Credit Only", use_container_width=True, key="pb_reset_bal"):
                reset_bot_account(starting_credit=float(bst.get("starting_credit", 5000.0)), clear_history=False)
                st.rerun()
        with bc4:
            if st.button("🗑️ Reset + Clear History", use_container_width=True, key="pb_reset_all"):
                reset_bot_account(starting_credit=float(bst.get("starting_credit", 5000.0)), clear_history=True)
                st.rerun()

        with st.expander("⚙️ Bot Credit, Lot Sizing (0.01 / 0.05 / 0.10) & Risk Settings", expanded=False):
            sc1, sc2, sc3, sc4 = st.columns(4)
            new_cred = sc1.number_input("Starting Credit ($)", min_value=100.0, max_value=1000000.0, value=float(bst.get("starting_credit", 5000.0)), step=500.0)
            new_ln = sc2.number_input("Normal Lot (60-74%)", min_value=0.01, max_value=1.0, value=float(bst.get("lot_normal", 0.01)), step=0.01, format="%.2f")
            new_lh = sc3.number_input("High Lot (75-84%)", min_value=0.01, max_value=2.0, value=float(bst.get("lot_high", 0.05)), step=0.01, format="%.2f")
            new_lv = sc4.number_input("Heavy Lot (85%+)", min_value=0.01, max_value=5.0, value=float(bst.get("lot_heavy", 0.10)), step=0.01, format="%.2f")

            rc1, rc2, rc3, rc4 = st.columns(4)
            new_sp = rc1.number_input("Spread ($/oz)", min_value=0.0, max_value=2.0, value=float(bst.get("spread_usd", 0.30)), step=0.05, format="%.2f")
            new_cm = rc2.number_input("Commission ($/lot)", min_value=0.0, max_value=30.0, value=float(bst.get("commission_per_lot", 7.0)), step=1.0)
            new_mc = rc3.slider("Min Confidence %", 55, 85, int(bst.get("min_confidence", 60.0)))
            new_dd = rc4.slider("Daily Max DD %", 1, 20, int(bst.get("daily_dd_limit_pct", 5.0)))

            tc1, tc2, tc3 = st.columns(3)
            en_6h = tc1.checkbox("🟠 Enable Track A (6H Fix)", value=bool(bst.get("enable_6h_fix", True)))
            en_15m = tc2.checkbox("⚪ Enable Track B (15m Taaza)", value=bool(bst.get("enable_15m_taaza", True)))
            if tc3.button("💾 Save Bot Settings", use_container_width=True, key="pb_save_cfg"):
                old_start = float(bst.get("starting_credit", 5000.0))
                bst["starting_credit"] = float(new_cred)
                if abs(float(new_cred) - old_start) > 1e-6 and not hist_trades and not bst.get("open_trades"):
                    bst["balance"] = float(new_cred)
                    bst["peak_equity"] = float(new_cred)
                    bst["daily_start_balance"] = float(new_cred)
                bst["lot_normal"] = float(new_ln)
                bst["lot_high"] = float(new_lh)
                bst["lot_heavy"] = float(new_lv)
                bst["spread_usd"] = float(new_sp)
                bst["commission_per_lot"] = float(new_cm)
                bst["min_confidence"] = float(new_mc)
                bst["daily_dd_limit_pct"] = float(new_dd)
                bst["enable_6h_fix"] = bool(en_6h)
                bst["enable_15m_taaza"] = bool(en_15m)
                save_bot_state(bst)
                st.success("Bot settings saved!")
                st.rerun()

    # Account & Performance KPI Strip
    bal = float(bst.get("balance", 5000.0))
    eq = float(bst.get("equity", bal))
    st_cr = max(float(bst.get("starting_credit", 5000.0)), 1.0)
    tot_pnl = eq - st_cr
    tot_pnl_pct = tot_pnl / st_cr * 100.0
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("💰 Balance", f"${bal:,.2f}", f"{tot_pnl:+.2f}$ ({tot_pnl_pct:+.2f}%)")
    k2.metric("📈 Equity", f"${eq:,.2f}", f"Max DD {bst.get('max_drawdown_pct', 0.0):.2f}%")
    k3.metric("🎯 Win Rate", f"{diag.get('win_rate_pct', 0.0):.1f}%", f"{diag.get('wins', 0)}W / {diag.get('losses', 0)}L")
    k4.metric("✅ TP vs 🛑 SL", f"{diag.get('tp_hits', 0)} / {diag.get('sl_hits', 0)}", f"Total {diag.get('total_trades', 0)} closed")
    k5.metric("⬆️ Long / ⬇️ Short", f"{diag.get('longs', 0)} / {diag.get('shorts', 0)}", f"Avg R: {diag.get('avg_r', 0.0):+.2f}R")
    k6.metric("⚖️ Profit Factor", f"{diag.get('profit_factor', 0.0):.2f}", f"Costs: ${diag.get('total_costs_usd', 0.0):.2f}")

    # Open Trades Table
    open_list = bst.get("open_trades") or []
    st.markdown(f"**🟢 Live Open Trades ({len(open_list)} / 2)** — Status: `{'RUNNING' if bst.get('running') else 'PAUSED'}` · _{bst.get('last_event', '')}_")
    if open_list:
        open_rows = [
            {
                "ID": t["trade_id"],
                "Track": t["track"],
                "Side": t["side"],
                "Lot": t["lot_size"],
                "Conf%": t["confidence_pct"],
                "Entry ($)": t["entry_price"],
                "Current ($)": t.get("current_price", t["entry_price"]),
                "TP ($)": t["tp_price"],
                "SL ($)": t["sl_price"],
                "BE Armed": "✅" if t.get("breakeven_armed") else "—",
                "Floating ($)": t.get("floating_pnl_usd", 0.0),
                "Engine": t.get("forecaster_engine", "TFT"),
                "Regime": t.get("regime_at_entry", ""),
                "Pattern": t.get("candle_pattern", ""),
            }
            for t in open_list
        ]
        st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Koi open trade nahi hai. Bot Start karein — jab 6H Fix ya 15m Taaza par ≥60% setup milega to khud trade open hogi.")

    # Self-Diagnostic Insights ("Kahan-Kahan Masla Hai?")
    st.markdown("### 🔍 Self-Diagnostic Report (Kahan-Kahan Masla Hai · Code Improvement Guide)")
    for ins in diag.get("insights", []):
        st.markdown(f"- {ins}")

    if diag.get("total_trades", 0) > 0:
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.caption("**By Track (6H Fix vs 15m Taaza)**")
            st.dataframe(pd.DataFrame(diag["by_track"]), use_container_width=True, hide_index=True)
        with dc2:
            st.caption("**By Market Regime (Trend vs Chop)**")
            st.dataframe(pd.DataFrame(diag["by_regime"]), use_container_width=True, hide_index=True)
        with dc3:
            st.caption("**By Lot Tier (0.01 / 0.05 / 0.10)**")
            st.dataframe(pd.DataFrame(diag["by_lot"]), use_container_width=True, hide_index=True)

        dc4, dc5, dc6 = st.columns(3)
        with dc4:
            st.caption("**By Session (Asian / London / NY)**")
            st.dataframe(pd.DataFrame(diag["by_session"]), use_container_width=True, hide_index=True)
        with dc5:
            st.caption("**By Forecaster Engine**")
            st.dataframe(pd.DataFrame(diag["by_engine"]), use_container_width=True, hide_index=True)
        with dc6:
            st.caption("**By Exit Reason**")
            st.dataframe(pd.DataFrame(diag["by_exit"]), use_container_width=True, hide_index=True)

        # Complete Forensic Closed Trade History Table + CSV Export
        st.markdown("### 📜 Complete Closed Trade History (With Model & Market Context)")
        hist_df = pd.DataFrame(list(reversed(hist_trades)))
        show_cols = [
            c
            for c in [
                "trade_id", "track", "side", "lot_size", "confidence_pct",
                "entry_time_utc", "entry_price", "exit_time_utc", "exit_price",
                "exit_reason", "net_pnl_usd", "r_multiple", "balance_after",
                "forecaster_engine", "forecast_id", "regime_at_entry", "candle_pattern",
                "nearest_sr", "session",
            ]
            if c in hist_df.columns
        ]
        st.dataframe(hist_df[show_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Full Trade Journal (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="xau_paper_bot_journal.csv",
            mime="text/csv",
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
