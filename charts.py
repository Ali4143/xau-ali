"""Plotly Bloomberg-Dark Full-Screen Forecast Chart (UTC+5 / PKT display, exact KSE-100 visual match)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN_UP = "#00D26A"
RED_DOWN = "#F87171"
ORANGE_LOCK = "#F97316"
WHITE_LIVE = "#FFFFFF"
CYAN_HI = "#34D399"
RED_LO = "#F87171"
PIVOT_P = "#ED8936"
PIVOT_R = "#FC8181"
PIVOT_S = "#68D391"
PURPLE_BADGE = "#8B5CF6"
GRID_COL = "#181C24"

MEMBER_COLORS = {
    "tft": "#38BDF8",
    "lightgbm": "#A855F7",
    "tcn": "#FACC15",
    "wavelet_bilstm": "#EC4899",
    "timemixer": "#2DD4BF",
}


def _to_display_tz(idx: pd.DatetimeIndex, tz_name: str = "Asia/Karachi") -> pd.DatetimeIndex:
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz_name).tz_localize(None)


def _ts_to_display(ts: pd.Timestamp, tz_name: str = "Asia/Karachi") -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(tz_name).tz_localize(None)


def build_mtf_forecast_chart(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame,
    active_fc: Optional[dict] = None,
    live_refresh_fc: Optional[dict] = None,
    past_forecasts: Optional[list[dict]] = None,
    show_members: Optional[list[str]] = None,
    show_m15_bg: bool = False,
    show_rsi_pane: bool = False,
    show_liquidity: bool = True,
    inst_summary: Optional[dict] = None,
    history_minutes: int = 30,
    display_tz: str = "UTC",
    height: int = 640,
) -> go.Figure:
    """Render full-screen pitch-black forecast chart matching the exact KSE-100 layout.

    - Default view: Focuses on the locked 6-hour forecast window (only `history_minutes` before origin)
      so actual 1M candles and the zig-zag plausible path fill the screen cleanly.
    - Timestamps converted cleanly to `display_tz` (default Asia/Karachi = UTC+5).
    - 80% band placed in legend (`legendonly`) by default so the canvas stays razor-sharp.
    """
    tz_label = "UTC+5" if display_tz == "Asia/Karachi" else display_tz
    rows = 2 if show_rsi_pane else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.78, 0.22] if show_rsi_pane else [1.0],
    )

    # Determine clean X-axis window: anchor around active_fc origin - history_minutes
    if active_fc and active_fc.get("origin_utc"):
        origin_utc = pd.Timestamp(active_fc["origin_utc"])
        if origin_utc.tzinfo is None:
            origin_utc = origin_utc.tz_localize("UTC")
        win_start_utc = origin_utc - pd.Timedelta(minutes=history_minutes)
        m1_view = df_m1[df_m1.index >= win_start_utc].copy()
        m15_view = df_m15[df_m15.index >= win_start_utc - pd.Timedelta(minutes=15)].copy()
    else:
        m1_view = df_m1.tail(max(60, history_minutes)).copy()
        m15_view = df_m15.tail(max(16, history_minutes // 15 + 4)).copy()

    ys: list[float] = []

    # 1. Optional Background 15M Candles
    if show_m15_bg and not m15_view.empty:
        x15 = _to_display_tz(m15_view.index, display_tz)
        fig.add_trace(
            go.Candlestick(
                x=x15,
                open=m15_view["open"],
                high=m15_view["high"],
                low=m15_view["low"],
                close=m15_view["close"],
                name="15M Context",
                increasing_line_color="rgba(0, 210, 106, 0.28)",
                increasing_fillcolor="rgba(0, 210, 106, 0.08)",
                decreasing_line_color="rgba(248, 113, 113, 0.28)",
                decreasing_fillcolor="rgba(248, 113, 113, 0.08)",
            ),
            row=1,
            col=1,
        )

    # 2. Foreground: Actual 1-Minute Candles (Exact Green/Red Styling)
    x1 = _to_display_tz(m1_view.index, display_tz) if not m1_view.empty else pd.DatetimeIndex([])
    if not m1_view.empty:
        fig.add_trace(
            go.Candlestick(
                x=x1,
                open=m1_view["open"],
                high=m1_view["high"],
                low=m1_view["low"],
                close=m1_view["close"],
                name="Asal (1-min)",
                increasing_line_color=GREEN_UP,
                increasing_fillcolor=GREEN_UP,
                decreasing_line_color=RED_DOWN,
                decreasing_fillcolor=RED_DOWN,
            ),
            row=1,
            col=1,
        )
        ys += [float(m1_view["low"].min()), float(m1_view["high"].max())]

    # 3. Past Logged Forecasts (hidden by default in legend to keep screen spotless)
    if past_forecasts:
        for pf in past_forecasts[-4:]:
            if active_fc and pf.get("id") == active_fc.get("id"):
                continue
            pt_raw = [pd.Timestamp(pf["origin_utc"])] + list(pd.to_datetime(pf["times"], utc=True))
            pt = [_ts_to_display(t, display_tz) for t in pt_raw]
            py = [float(pf["P0"])] + list(pf.get("zigzag_path") or pf["q50"])
            fig.add_trace(
                go.Scatter(
                    x=pt,
                    y=py,
                    mode="lines",
                    name=f"Past FC {pf.get('id')} ({pt[0]:%H:%M})",
                    visible="legendonly",
                    line=dict(color="rgba(255, 255, 255, 0.30)", width=1.2, dash="dot"),
                    hovertemplate="%{x|%H:%M}<br>Past FC: $%{y:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    x_start = x1.min() if len(x1) else pd.Timestamp.now()
    x_end = x1.max() if len(x1) else x_start

    # 4. Primary Locked 6-Hour Forecast (Solid Orange Zig-Zag)
    if active_fc and active_fc.get("times"):
        origin_disp = _ts_to_display(pd.Timestamp(active_fc["origin_utc"]), display_tz)
        fut_disp = list(_to_display_tz(pd.to_datetime(active_fc["times"], utc=True), display_tz))
        t_fc = [origin_disp] + fut_disp
        x_start = min(x_start, origin_disp) if history_minutes <= 5 else x_start
        x_end = max(x_end, t_fc[-1])

        P0 = float(active_fc["P0"])
        q10 = [P0] + list(active_fc["q10"])
        q50 = [P0] + list(active_fc["q50"])
        q90 = [P0] + list(active_fc["q90"])
        zz = [P0] + list(active_fc.get("zigzag_path") or active_fc["q50"])
        ys += [float(min(zz)), float(max(zz))]

        # 80% Band (legendonly by default for spotless view, click in legend to toggle)
        fig.add_trace(
            go.Scatter(
                x=t_fc + t_fc[::-1],
                y=q90 + q10[::-1],
                fill="toself",
                fillcolor="rgba(251, 139, 30, 0.12)",
                line=dict(width=0),
                name="80% Band (q10–q90)",
                visible="legendonly",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        fc_engine = str(active_fc.get("forecaster") or active_fc.get("forecaster_engine") or "ensemble").upper()

        # Solid Orange Locked Zig-Zag Plausible Path
        fig.add_trace(
            go.Scatter(
                x=t_fc,
                y=zz,
                mode="lines",
                name=f"✅ Locked 6H Zig-Zag [{fc_engine}] — fix {origin_disp:%H:%M} ({tz_label})",
                line=dict(color=ORANGE_LOCK, width=2.5),
                hovertemplate=f"%{{x|%H:%M}}<br>Locked [{fc_engine}] Zig-Zag: $%{{y:,.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Mark Expected Swing Peak (▲) and Swing Dip (▼) directly on the forecast curve
        if len(zz) > 2:
            zz_arr = np.asarray(zz[1:], dtype=float)
            i_pk = int(np.argmax(zz_arr)) + 1
            i_dp = int(np.argmin(zz_arr)) + 1
            fig.add_trace(
                go.Scatter(
                    x=[t_fc[i_pk], t_fc[i_dp]],
                    y=[zz[i_pk], zz[i_dp]],
                    mode="markers+text",
                    marker=dict(
                        symbol=["triangle-up", "triangle-down"],
                        size=11,
                        color=[GREEN_UP, RED_DOWN],
                        line=dict(color="#FFFFFF", width=0.8),
                    ),
                    text=[
                        f"▲ Peak {t_fc[i_pk]:%H:%M} (${zz[i_pk]:,.2f})",
                        f"▼ Dip {t_fc[i_dp]:%H:%M} (${zz[i_dp]:,.2f})",
                    ],
                    textposition=["top center", "bottom center"],
                    textfont=dict(size=10, color="#E2E8F0", family="monospace"),
                    name="Expected Swing Peak/Dip",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        # Smooth Median (legendonly)
        fig.add_trace(
            go.Scatter(
                x=t_fc,
                y=q50,
                mode="lines",
                name="Median (q50 smooth)",
                visible="legendonly",
                line=dict(color="#FBBF24", width=1.4, dash="dot"),
                hovertemplate="%{x|%H:%M}<br>Median q50: $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Optional Individual Member Overlays
        members = active_fc.get("member_q50") or {}
        for m_name in (show_members or []):
            if m_name in members:
                fig.add_trace(
                    go.Scatter(
                        x=t_fc,
                        y=[P0] + list(members[m_name]),
                        mode="lines",
                        name=f"Model: {m_name.upper()} (raw q50)",
                        line=dict(color=MEMBER_COLORS.get(m_name, "#94A3B8"), width=1.4, dash="dot"),
                        hovertemplate=f"{m_name.upper()} raw q50: $%{{y:,.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Optional White Dashed Live Refresh Overlay (if distinct from locked forecast)
        if live_refresh_fc and live_refresh_fc.get("id") != active_fc.get("id") and live_refresh_fc.get("times"):
            o_lv = _ts_to_display(pd.Timestamp(live_refresh_fc["origin_utc"]), display_tz)
            t_lv = [o_lv] + list(_to_display_tz(pd.to_datetime(live_refresh_fc["times"], utc=True), display_tz))
            y_lv = [float(live_refresh_fc["P0"])] + list(live_refresh_fc.get("zigzag_path") or live_refresh_fc["q50"])
            lv_engine = str(live_refresh_fc.get("forecaster") or live_refresh_fc.get("forecaster_engine") or "ensemble").upper()
            fig.add_trace(
                go.Scatter(
                    x=t_lv,
                    y=y_lv,
                    mode="lines",
                    name=f"🔄 Taaza live [{lv_engine}] ({o_lv:%H:%M} → End)",
                    line=dict(color=WHITE_LIVE, width=2.0, dash="dash"),
                    hovertemplate=f"%{{x|%H:%M}}<br>Taaza [{lv_engine}]: $%{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            ys += [float(min(y_lv)), float(max(y_lv))]

        # Target High & Target Low Lines (use live Taaza's fresh targets when Taaza is active, else Locked targets)
        tgt_src = live_refresh_fc if (live_refresh_fc and live_refresh_fc.get("target_high")) else active_fc
        th = tgt_src.get("target_high")
        tl = tgt_src.get("target_low")
        if th is not None:
            ys.append(float(th))
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[th, th],
                    mode="lines",
                    name=f"🎯 Target High: {th:,.2f}",
                    line=dict(color=CYAN_HI, width=1.2, dash="dashdot"),
                    hovertemplate=f"Target High: ${th:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=x_start, y=th, xref="x", yref="y",
                text=f"🎯 Target High {th:,.2f}", showarrow=False,
                xanchor="left", yanchor="bottom", font=dict(size=10, color=CYAN_HI),
                row=1, col=1,
            )
        if tl is not None:
            ys.append(float(tl))
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[tl, tl],
                    mode="lines",
                    name=f"🎯 Target Low: {tl:,.2f}",
                    line=dict(color=RED_LO, width=1.2, dash="dashdot"),
                    hovertemplate=f"Target Low: ${tl:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=x_start, y=tl, xref="x", yref="y",
                text=f"🎯 Target Low {tl:,.2f}", showarrow=False,
                xanchor="left", yanchor="bottom", font=dict(size=10, color=RED_LO),
                row=1, col=1,
            )

        # 17:00 NY Floor Pivots (only annotate those near visible price range so chart stays clean)
        y_lo_ref = min(ys) if ys else P0 - 20
        y_hi_ref = max(ys) if ys else P0 + 20
        pivs = active_fc.get("pivots_ny17") or {}
        for pk in ("R2", "R1", "P", "S1", "S2"):
            val = pivs.get(pk)
            if val is None:
                continue
            col = PIVOT_P if pk == "P" else (PIVOT_R if pk.startswith("R") else PIVOT_S)
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[val, val],
                    mode="lines",
                    name="Pivots (17:00 NY)" if pk == "P" else pk,
                    showlegend=(pk == "P"),
                    legendgroup="pivots",
                    line=dict(color=col, width=1.0, dash="dot"),
                    hovertemplate=f"{pk}: ${val:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if (y_lo_ref - 15.0) <= val <= (y_hi_ref + 15.0):
                fig.add_annotation(
                    x=x_end, y=val, xref="x", yref="y",
                    text=f"{pk} {val:,.2f}", showarrow=False,
                    xanchor="right", yanchor="bottom", font=dict(size=10, color=col),
                    row=1, col=1,
                )

        # Institutional Liquidity Pools (PDH/PDL, Asian H/L, London H/L) & Stop-Hunt Sweeps
        if show_liquidity and inst_summary:
            liq_lvls = inst_summary.get("levels") or {}
            liq_colors = {
                "PDH": "#F472B6",
                "PDL": "#F472B6",
                "Asian_H": "#A78BFA",
                "Asian_L": "#A78BFA",
                "London_H": "#38BDF8",
                "London_L": "#38BDF8",
            }
            for lk, lval in liq_lvls.items():
                if lval is None or not np.isfinite(lval):
                    continue
                if (y_lo_ref - 22.0) <= float(lval) <= (y_hi_ref + 22.0):
                    lcol = liq_colors.get(lk, "#94A3B8")
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[lval, lval],
                            mode="lines",
                            name=f"🏦 {lk}: {lval:,.2f}",
                            line=dict(color=lcol, width=1.0, dash="longdash"),
                            opacity=0.65,
                            hovertemplate=f"Liquidity {lk}: ${lval:,.2f}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_annotation(
                        x=x_end, y=lval, xref="x", yref="y",
                        text=f"🏦 {lk} {lval:,.2f}", showarrow=False,
                        xanchor="right", yanchor="top", font=dict(size=9, color=lcol),
                        row=1, col=1,
                    )

            inst_tail = inst_summary.get("inst_tail")
            if isinstance(inst_tail, pd.DataFrame) and not inst_tail.empty and not m1_view.empty:
                sw_view = inst_tail.reindex(m1_view.index)["liq_sweep"].fillna(0.0)
                bull_m = sw_view > 0.5
                bear_m = sw_view < -0.5
                if bull_m.any():
                    fig.add_trace(
                        go.Scatter(
                            x=_to_display_tz(m1_view.index[bull_m], display_tz),
                            y=m1_view.loc[bull_m, "low"] - 0.6,
                            mode="markers",
                            name="⚡ Bull Stop-Hunt Sweep",
                            marker=dict(symbol="triangle-up", size=9, color="#34D399", line=dict(color="#FFFFFF", width=0.7)),
                            hovertemplate="%{x|%H:%M}<br>⚡ Bull Liquidity Sweep<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                if bear_m.any():
                    fig.add_trace(
                        go.Scatter(
                            x=_to_display_tz(m1_view.index[bear_m], display_tz),
                            y=m1_view.loc[bear_m, "high"] + 0.6,
                            mode="markers",
                            name="⚡ Bear Stop-Hunt Sweep",
                            marker=dict(symbol="triangle-down", size=9, color="#F87171", line=dict(color="#FFFFFF", width=0.7)),
                            hovertemplate="%{x|%H:%M}<br>⚡ Bear Liquidity Sweep<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )

        # Vertical Origin Lock Line & Top Annotation (Exact KSE-100 style)
        fig.add_vline(x=origin_disp, line=dict(color="#5A6270", width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(
            x=origin_disp, y=1.0, yref="y domain", yanchor="bottom", xanchor="left",
            text=f"✅ Forecast locked {origin_disp:%H:%M} ({tz_label}) · [{fc_engine}]",
            showarrow=False, font=dict(size=10, color="#E2E8F0"),
        )
        if active_fc.get("stopped_at_market_close"):
            fig.add_vline(x=t_fc[-1], line=dict(color="#F87171", width=1.2, dash="dot"), row=1, col=1)
            fig.add_annotation(
                x=t_fc[-1], y=0.96, yref="y domain", yanchor="top", xanchor="right",
                text="⏸ Market Closed (17:00 NY)", showarrow=False,
                font=dict(size=10, color="#F87171"),
            )

    # 5. Purple Right-Axis Live Price Tag Badge (Exact KSE-100 Style)
    if not m1_view.empty:
        last_p = float(m1_view["close"].iloc[-1])
        fig.add_annotation(
            x=1.0, xref="paper", y=last_p, yref="y",
            text=f" {last_p:,.2f} ", showarrow=False, xanchor="left",
            bgcolor=PURPLE_BADGE, font=dict(size=11, color="#000000", family="monospace"),
            bordercolor="#FFFFFF", borderwidth=1,
        )

    # 6. Optional Lower RSI(14) Pane
    if show_rsi_pane and not m1_view.empty:
        c_s = m1_view["close"].astype(float)
        d = c_s.diff()
        ag = d.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        al = (-d).clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rsi = 100.0 * ag / (ag + al + 1e-9)
        fig.add_trace(
            go.Scatter(
                x=x1, y=rsi, mode="lines", name="RSI(14)",
                line=dict(color="#A855F7", width=1.8),
                hovertemplate="%{x|%H:%M}<br>RSI(14): %{y:.1f}<extra></extra>",
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line=dict(color="rgba(248,113,113,0.4)", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="rgba(52,211,153,0.4)", width=1, dash="dot"), row=2, col=1)
        fig.update_yaxes(gridcolor=GRID_COL, zeroline=False, side="right", row=2, col=1, range=[0, 100])

    # Auto-fit Y-axis tightly around candles + forecast + targets
    y_range = None
    if ys:
        lo_y, hi_y = min(ys), max(ys)
        pad = max((hi_y - lo_y) * 0.08, hi_y * 0.0005)
        y_range = [lo_y - pad, hi_y + pad]

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=78, t=30, b=10),
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#E2E8F0", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.09, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
    )
    if show_rsi_pane:
        fig.update_layout(xaxis2_rangeslider_visible=False)
    fig.update_xaxes(
        range=[x_start, x_end],
        tickformat="%H:%M",
        gridcolor=GRID_COL,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=GRID_COL,
        zeroline=False,
        side="right",
        row=1,
        col=1,
        tickformat=",.2f",
        **({"range": y_range} if y_range else {}),
    )
    return fig
