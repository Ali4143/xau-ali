"""Dual-Track Paper Trading Bot (6H Fix + 15m Taaza) & Self-Diagnostic Engine for XAUUSD.

Strictly isolated inside `xau_mtf/`. Uses virtual credit only with realistic Gold execution
math (1.00 lot = 100 oz -> $1 move = $100; 0.01 = $1/$1, 0.05 = $5/$1, 0.10 = $10/$1),
spread + commission deduction, Breakeven/Momentum trailing, hybrid 15m smart exit,
persistent JSONL forensic history, and automated Self-Diagnostic analytics.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

PAPER_DIR = config.ROOT / "paper_trading"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
BOT_STATE_FILE = PAPER_DIR / "bot_state.json"
TRADE_HISTORY_FILE = PAPER_DIR / "trade_history.jsonl"

OZ_PER_LOT = 100.0


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_session(ts: pd.Timestamp) -> str:
    hr = int(ts.hour)
    if 0 <= hr < 7:
        return "ASIAN"
    if 7 <= hr < 13:
        return "LONDON"
    if 13 <= hr < 21:
        return "NEW_YORK"
    return "ROLLOVER"


def default_bot_state(starting_credit: float = 5000.0) -> dict[str, Any]:
    today = _now_utc().strftime("%Y-%m-%d")
    return {
        "running": False,
        "starting_credit": float(starting_credit),
        "balance": float(starting_credit),
        "peak_equity": float(starting_credit),
        "max_drawdown_pct": 0.0,
        "daily_date_utc": today,
        "daily_start_balance": float(starting_credit),
        "daily_dd_limit_pct": 5.0,
        "daily_paused_reason": None,
        "lot_normal": 0.01,
        "lot_high": 0.05,
        "lot_heavy": 0.10,
        "spread_usd": 0.30,
        "commission_per_lot": 7.0,
        "min_confidence": 60.0,
        "enable_6h_fix": True,
        "enable_15m_taaza": True,
        "open_trades": [],
        "last_closed_by_track": {},
        "last_tick_utc": None,
        "last_event": "Bot initialized (Stopped). Set credit & click Start Bot.",
    }


def load_bot_state() -> dict[str, Any]:
    if not BOT_STATE_FILE.exists():
        st = default_bot_state()
        save_bot_state(st)
        return st
    try:
        st = json.loads(BOT_STATE_FILE.read_text(encoding="utf-8"))
        base = default_bot_state(float(st.get("starting_credit", 5000.0)))
        base.update(st)
        return base
    except Exception:
        st = default_bot_state()
        save_bot_state(st)
        return st


def save_bot_state(state: dict[str, Any]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = BOT_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        tmp.replace(BOT_STATE_FILE)
    except Exception:
        BOT_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_trade_history(record: dict[str, Any]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRADE_HISTORY_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")


def load_trade_history(limit: int = 500) -> list[dict[str, Any]]:
    if not TRADE_HISTORY_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in TRADE_HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-limit:] if limit else rows


def reset_bot_account(starting_credit: float = 5000.0, clear_history: bool = False) -> dict[str, Any]:
    st = default_bot_state(starting_credit=starting_credit)
    save_bot_state(st)
    if clear_history and TRADE_HISTORY_FILE.exists():
        TRADE_HISTORY_FILE.write_text("", encoding="utf-8")
    return st


def _calc_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    lot: float,
    spread_usd: float,
    commission_per_lot: float,
) -> tuple[float, float, float, float]:
    """Return (gross_pnl, spread_cost, commission_cost, net_pnl) in USD."""
    d = 1.0 if side == "LONG" else -1.0
    gross = d * (float(exit_price) - float(entry_price)) * float(lot) * OZ_PER_LOT
    sp_cost = float(spread_usd) * float(lot) * OZ_PER_LOT
    comm = float(commission_per_lot) * float(lot)
    net = gross - sp_cost - comm
    return round(gross, 2), round(sp_cost, 2), round(comm, 2), round(net, 2)


def _select_lot_size(conf: float, state: dict[str, Any]) -> tuple[float, str]:
    if conf >= 85.0:
        return float(state.get("lot_heavy", 0.10)), "HEAVY (85%+)"
    if conf >= 75.0:
        return float(state.get("lot_high", 0.05)), "HIGH (75-84%)"
    return float(state.get("lot_normal", 0.01)), "NORMAL (60-74%)"


def _nearest_sr_level(p_now: float, active_fc: dict | None, inst_sum: dict | None) -> tuple[str, float, float]:
    """Find nearest Support/Resistance or Liquidity Pool level & distance in USD."""
    cands: list[tuple[str, float]] = []
    if inst_sum:
        for k, v in (inst_sum.get("levels") or {}).items():
            if v is not None and np.isfinite(v):
                cands.append((k, float(v)))
    if active_fc:
        for k, v in (active_fc.get("pivots_ny17") or {}).items():
            if v is not None and np.isfinite(v):
                cands.append((f"Pivot_{k}", float(v)))
    if not cands:
        return "NONE", p_now, 999.0
    best_k, best_v = min(cands, key=lambda kv: abs(kv[1] - p_now))
    return best_k, round(best_v, 2), round(abs(best_v - p_now), 2)


def _score_entry_signal(
    track: str,
    fc: dict[str, Any],
    p_now: float,
    inst_sum: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Evaluate forecast + S/R + candle pattern + institutional confluence for a trade setup."""
    if not fc or not fc.get("q50"):
        return None

    q50 = fc["q50"]
    p0 = float(fc.get("P0", p_now))
    atr15 = max(float(fc.get("atr15_usd", (inst_sum or {}).get("atr15", 8.0))), 2.5)
    # Track A looks at 1h..end forecast; Track B looks at 15m..45m short-term horizon
    horizon_idx = min(len(q50) - 1, 59 if track == "6H_FIX" else 14)
    q50_tgt = float(q50[horizon_idx])
    delta = q50_tgt - p_now if track == "15M_TAAZA" else (q50_tgt - p0)

    # Require meaningful directional tilt (>= 0.12 * ATR15 or >= $1.20)
    min_move = max(0.12 * atr15, 1.20)
    if abs(delta) < min_move:
        return None
    side = "LONG" if delta > 0 else "SHORT"
    d = 1 if side == "LONG" else -1

    # Base confidence from forecast tilt strength (50..64)
    tilt_ratio = min(abs(delta) / max(atr15, 1.0), 1.5)
    conf = 52.0 + 10.0 * tilt_ratio

    reasons: list[str] = [f"{track} {str(fc.get('forecaster') or 'tft').upper()} {side} ({delta:+.2f})"]

    # 1) DirectionMeta probability alignment
    dm = fc.get("direction_meta") or {}
    p_up = float((dm.get("1h") or {}).get("p_up", 0.5))
    p_side = p_up if d == 1 else (1.0 - p_up)
    if p_side >= 0.55:
        conf += 8.0
        reasons.append(f"DirMeta {p_side*100:.0f}%")

    # 2) Institutional Radar & Regime alignment
    reg_lbl = (inst_sum or {}).get("regime_label", "UNKNOWN")
    inst_sc = float((inst_sum or {}).get("inst_score", 0.0))
    if inst_sc * d >= 18.0:
        conf += 12.0
        reasons.append(f"InstScore {inst_sc:+.0f}")
    elif inst_sc * d >= 6.0:
        conf += 6.0
    elif inst_sc * d <= -20.0:
        conf -= 12.0  # Strong institutional opposition!

    if reg_lbl == "CLEAN TREND":
        conf += 6.0
        reasons.append("Clean Trend")

    # 3) Candle Pattern & >=70% Ladder alignment
    pat_d = (inst_sum or {}).get("patterns") or {}
    lad = pat_d.get("active_ladder")
    if lad and int(lad.get("dir", 0)) == d:
        conf += 14.0
        reasons.append(f"Pattern {lad['pattern']} ({lad['confidence']:.0f}%)")
    elif lad and int(lad.get("dir", 0)) == -d:
        conf -= 8.0

    # 4) Support/Resistance / Liquidity Pool proximity bonus
    sr_name, sr_px, sr_dist = _nearest_sr_level(p_now, fc, inst_sum)
    if sr_dist <= max(0.8 * atr15, 4.0):
        conf += 8.0
        reasons.append(f"Near {sr_name} (${sr_px:,.2f})")

    conf = float(np.clip(round(conf, 1), 0.0, 96.0))

    # Build TP & SL with realistic RR
    if track == "6H_FIX":
        raw_tp = float(fc.get("target_high", p_now + 2.0 * atr15)) if d == 1 else float(fc.get("target_low", p_now - 2.0 * atr15))
        tp_dist = max(abs(raw_tp - p_now), 1.8 * atr15)
        sl_dist = max(0.55 * tp_dist, 1.2 * atr15)
        sl_dist = min(sl_dist, 2.2 * atr15)
        tp_px = round(p_now + d * tp_dist, 2)
        sl_px = round(p_now - d * sl_dist, 2)
    else:
        # 15m Taaza fast scalp: use Pattern T1 if aligned, else quick 0.85 * ATR15 target
        if lad and int(lad.get("dir", 0)) == d and lad.get("active_target_price"):
            tp_dist = max(abs(float(lad["active_target_price"]) - p_now), 2.50)
        else:
            tp_dist = max(min(abs(delta) * 1.15, 1.2 * atr15), 2.80)
        sl_dist = max(0.65 * tp_dist, 2.20)
        sl_dist = min(sl_dist, 1.3 * atr15)
        tp_px = round(p_now + d * tp_dist, 2)
        sl_px = round(p_now - d * sl_dist, 2)

    return {
        "track": track,
        "side": side,
        "confidence": conf,
        "entry_price": round(p_now, 2),
        "tp_price": tp_px,
        "sl_price": sl_px,
        "initial_sl": sl_px,
        "atr15": round(atr15, 2),
        "reasons": reasons,
        "sr_name": sr_name,
        "sr_price": sr_px,
        "forecast_id": fc.get("id"),
        "forecaster_engine": str(fc.get("forecaster") or "tft").upper(),
        "forecast_q50_target": round(q50_tgt, 2),
        "target_high": fc.get("target_high"),
        "target_low": fc.get("target_low"),
    }


def close_trade_now(
    state: dict[str, Any],
    trade: dict[str, Any],
    exit_price: float,
    exit_reason: str,
    exit_time_utc: str | None = None,
) -> dict[str, Any]:
    """Close an open trade, update balance & drawdown, and append to persistent JSONL journal."""
    gross, sp_cost, comm, net = _calc_pnl(
        side=trade["side"],
        entry_price=trade["entry_price"],
        exit_price=exit_price,
        lot=trade["lot_size"],
        spread_usd=float(state.get("spread_usd", 0.30)),
        commission_per_lot=float(state.get("commission_per_lot", 7.0)),
    )
    risk_usd = abs(float(trade["entry_price"]) - float(trade["initial_sl"])) * float(trade["lot_size"]) * OZ_PER_LOT
    r_mult = round(net / max(risk_usd, 0.01), 2)

    state["balance"] = round(float(state["balance"]) + net, 2)
    closed_rec = {
        **trade,
        "status": "CLOSED",
        "exit_time_utc": exit_time_utc or _now_utc().isoformat(),
        "exit_price": round(float(exit_price), 2),
        "exit_reason": exit_reason,
        "gross_pnl_usd": gross,
        "spread_cost_usd": sp_cost,
        "commission_usd": comm,
        "net_pnl_usd": net,
        "r_multiple": r_mult,
        "balance_after": state["balance"],
    }
    append_trade_history(closed_rec)
    state["open_trades"] = [t for t in state.get("open_trades", []) if t.get("trade_id") != trade.get("trade_id")]
    state.setdefault("last_closed_by_track", {})[trade["track"]] = {
        "time_utc": closed_rec["exit_time_utc"],
        "forecast_id": trade.get("forecast_id"),
        "exit_reason": exit_reason,
        "net_pnl_usd": net,
    }
    state["last_event"] = f"Closed {trade['track']} {trade['side']} ({trade['lot_size']} lot) via {exit_reason}: {net:+.2f}$"
    return closed_rec


def close_all_open_trades(df_m1: pd.DataFrame | None, reason: str = "MANUAL_STOP") -> dict[str, Any]:
    state = load_bot_state()
    px = float(df_m1["close"].iloc[-1]) if (df_m1 is not None and not df_m1.empty) else None
    for tr in list(state.get("open_trades", [])):
        close_trade_now(state, tr, exit_price=px if px is not None else float(tr["entry_price"]), exit_reason=reason)
    state["running"] = False
    state["last_event"] = f"All trades closed ({reason}) & Bot stopped."
    save_bot_state(state)
    return state


def tick_paper_bot(
    df_m1: pd.DataFrame,
    active_fc: dict[str, Any] | None,
    live_refresh_fc: dict[str, Any] | None,
    inst_sum: dict[str, Any] | None,
) -> dict[str, Any]:
    """Advance the Paper Trading Bot by 1 live cycle (safe to call every 60s).

    - Always updates floating PnL & manages open trades (SL/TP/Breakeven/15m smart exit).
    - Opens new trades ONLY when `state["running"] == True` and safety checks pass.
    """
    state = load_bot_state()
    if df_m1 is None or df_m1.empty:
        return state

    last_bar = df_m1.iloc[-1]
    t_now = pd.Timestamp(df_m1.index[-1])
    if t_now.tzinfo is None:
        t_now = t_now.tz_localize("UTC")
    p_now = float(last_bar["close"])
    bar_hi = float(last_bar["high"])
    bar_lo = float(last_bar["low"])
    t_now_iso = t_now.isoformat()

    # Reset daily drawdown anchor on new UTC day
    today_str = t_now.strftime("%Y-%m-%d")
    if state.get("daily_date_utc") != today_str:
        state["daily_date_utc"] = today_str
        state["daily_start_balance"] = float(state["balance"])
        state["daily_paused_reason"] = None

    taaza_fc = live_refresh_fc or active_fc
    pat_lad = ((inst_sum or {}).get("patterns") or {}).get("active_ladder")

    # ── 1. MANAGE EXISTING OPEN TRADES ────────────────────────────────────────
    for tr in list(state.get("open_trades", [])):
        side = tr["side"]
        d = 1 if side == "LONG" else -1
        entry_px = float(tr["entry_price"])
        sl_px = float(tr["sl_price"])
        tp_px = float(tr["tp_price"])

        # Check all 1M bars strictly AFTER entry (or since last tick) so no intra-bar hit is missed
        ref_ts = pd.Timestamp(tr.get("last_checked_utc") or tr["entry_time_utc"])
        if ref_ts.tzinfo is None:
            ref_ts = ref_ts.tz_localize("UTC")
        bars_since = df_m1[df_m1.index > ref_ts]
        hi_win = float(bars_since["high"].max()) if not bars_since.empty else bar_hi
        lo_win = float(bars_since["low"].min()) if not bars_since.empty else bar_lo

        # Conservative check: SL checked first if both hit inside same minute
        sl_hit = (lo_win <= sl_px) if d == 1 else (hi_win >= sl_px)
        tp_hit = (hi_win >= tp_px) if d == 1 else (lo_win <= tp_px)

        if sl_hit:
            reason = "BREAKEVEN_STOP" if tr.get("breakeven_armed") else "SL_HIT"
            close_trade_now(state, tr, exit_price=sl_px, exit_reason=reason, exit_time_utc=t_now_iso)
            continue
        if tp_hit:
            close_trade_now(state, tr, exit_price=tp_px, exit_reason="TP_HIT", exit_time_utc=t_now_iso)
            continue

        # Update floating PnL
        _, _, _, fl_net = _calc_pnl(
            side, entry_px, p_now, tr["lot_size"], float(state["spread_usd"]), float(state["commission_per_lot"])
        )
        tr["floating_pnl_usd"] = fl_net
        tr["current_price"] = round(p_now, 2)
        tr["last_checked_utc"] = t_now_iso

        # Track A (6H_FIX): arm Breakeven once >= 50% of TP distance achieved
        if tr["track"] == "6H_FIX" and not tr.get("breakeven_armed"):
            tp_dist = abs(tp_px - entry_px)
            prog = d * (p_now - entry_px)
            if tp_dist > 0 and prog >= 0.50 * tp_dist:
                tr["sl_price"] = round(entry_px + d * float(state["spread_usd"]), 2)
                tr["breakeven_armed"] = True
                state["last_event"] = f"6H_FIX {side} armed Breakeven SL @ ${tr['sl_price']:,.2f}"

        # Track B (15M_TAAZA): Fast Momentum Auto-Extend + Smart 15m Boundary Exit
        if tr["track"] == "15M_TAAZA":
            # Fast momentum extension: if accelerating toward TP, arm BE & extend to Pattern T2/T3
            if pat_lad and int(pat_lad.get("dir", 0)) == d and pat_lad.get("fast_momentum"):
                new_tgt = float(pat_lad["active_target_price"])
                if (d == 1 and new_tgt > tp_px) or (d == -1 and new_tgt < tp_px):
                    tr["tp_price"] = round(new_tgt, 2)
                    if fl_net > 0 and not tr.get("breakeven_armed"):
                        tr["sl_price"] = round(entry_px + d * float(state["spread_usd"]), 2)
                        tr["breakeven_armed"] = True
                    state["last_event"] = f"🚀 15M_TAAZA {side} Fast Momentum -> Extended TP to ${tr['tp_price']:,.2f}"

            # Smart 15m Boundary Rule (minute :14, :29, :44, :59 -> lock profit cleanly!)
            if (t_now.minute % 15) == 14 and fl_net >= 0.50:
                close_trade_now(state, tr, exit_price=p_now, exit_reason="15M_PROFIT_LOCK", exit_time_utc=t_now_iso)
                continue

            # If a brand-new 15m Taaza forecast arrived and FLIPPED opposite direction -> exit cleanly!
            if taaza_fc and taaza_fc.get("id") and taaza_fc.get("id") != tr.get("forecast_id") and taaza_fc.get("q50"):
                q_short = float(taaza_fc["q50"][min(14, len(taaza_fc["q50"]) - 1)])
                new_dir = 1 if (q_short - float(taaza_fc.get("P0", p_now))) > 0.8 else (-1 if (q_short - float(taaza_fc.get("P0", p_now))) < -0.8 else 0)
                if new_dir == -d:
                    close_trade_now(state, tr, exit_price=p_now, exit_reason="TAAZA_FLIP", exit_time_utc=t_now_iso)
                    continue

    # ── 2. UPDATE EQUITY, MAX DRAWDOWN & DAILY CIRCUIT BREAKER ────────────────
    float_total = sum(float(t.get("floating_pnl_usd", 0.0)) for t in state.get("open_trades", []))
    equity = round(float(state["balance"]) + float_total, 2)
    state["equity"] = equity
    if equity > float(state.get("peak_equity", equity)):
        state["peak_equity"] = equity
    peak = max(float(state.get("peak_equity", equity)), 1.0)
    dd_pct = max(0.0, (peak - equity) / peak * 100.0)
    state["max_drawdown_pct"] = round(max(float(state.get("max_drawdown_pct", 0.0)), dd_pct), 2)

    day_start = max(float(state.get("daily_start_balance", state["starting_credit"])), 1.0)
    day_dd_pct = max(0.0, (day_start - equity) / day_start * 100.0)
    if state.get("running") and day_dd_pct >= float(state.get("daily_dd_limit_pct", 5.0)):
        state["running"] = False
        state["daily_paused_reason"] = f"Daily drawdown limit hit (-{day_dd_pct:.2f}% >= -{state['daily_dd_limit_pct']:.1f}%)"
        state["last_event"] = f"⛔ Auto-Paused: {state['daily_paused_reason']}"

    # ── 3. EVALUATE NEW ENTRIES (ONLY WHEN RUNNING, MARKET OPEN & NOT HIGH-VOL SHOCK) ─
    reg_code = int((inst_sum or {}).get("regime_state", 0))
    bar_age_min = (_now_utc() - t_now.to_pydatetime()).total_seconds() / 60.0
    market_is_live = bar_age_min <= 6.0
    if state.get("running") and not market_is_live:
        state["last_event"] = f"⏸ Market closed / rollover (last bar {int(bar_age_min)}m ago) — auto-resumes on open."
    if state.get("running") and market_is_live and reg_code != 2:
        open_tracks = {t["track"] for t in state.get("open_trades", [])}
        min_conf = float(state.get("min_confidence", 60.0))

        candidates = []
        if state.get("enable_6h_fix", True) and "6H_FIX" not in open_tracks and active_fc:
            candidates.append(("6H_FIX", active_fc, 10))
        if state.get("enable_15m_taaza", True) and "15M_TAAZA" not in open_tracks and taaza_fc:
            candidates.append(("15M_TAAZA", taaza_fc, 2))

        for track, fc_src, cool_min in candidates:
            last_cl = (state.get("last_closed_by_track") or {}).get(track)
            if last_cl and last_cl.get("time_utc"):
                try:
                    mins_since = (t_now - pd.Timestamp(last_cl["time_utc"])).total_seconds() / 60.0
                    if mins_since < cool_min:
                        continue
                    # Don't re-enter the exact same 6H locked forecast immediately after closing it
                    if track == "6H_FIX" and last_cl.get("forecast_id") == fc_src.get("id"):
                        continue
                except Exception:
                    pass

            sig = _score_entry_signal(track, fc_src, p_now, inst_sum)
            if sig and sig["confidence"] >= min_conf:
                lot, lot_tier = _select_lot_size(sig["confidence"], state)
                pat_d = (inst_sum or {}).get("patterns") or {}
                trade_obj = {
                    "trade_id": uuid.uuid4().hex[:8],
                    "track": track,
                    "side": sig["side"],
                    "lot_size": lot,
                    "lot_tier": lot_tier,
                    "confidence_pct": sig["confidence"],
                    "status": "OPEN",
                    "entry_time_utc": t_now_iso,
                    "last_checked_utc": t_now_iso,
                    "entry_price": sig["entry_price"],
                    "current_price": sig["entry_price"],
                    "tp_price": sig["tp_price"],
                    "sl_price": sig["sl_price"],
                    "initial_sl": sig["initial_sl"],
                    "breakeven_armed": False,
                    "floating_pnl_usd": round(
                        -(float(state["spread_usd"]) * lot * OZ_PER_LOT + float(state["commission_per_lot"]) * lot), 2
                    ),
                    "reasons": sig["reasons"],
                    # Full forensic model context for Self-Diagnostics
                    "forecast_id": sig["forecast_id"],
                    "forecaster_engine": sig["forecaster_engine"],
                    "forecast_q50_target": sig["forecast_q50_target"],
                    "target_high": sig["target_high"],
                    "target_low": sig["target_low"],
                    "nearest_sr": f"{sig['sr_name']} (${sig['sr_price']:,.2f})",
                    "regime_at_entry": (inst_sum or {}).get("regime_label", "UNKNOWN"),
                    "candle_pattern": pat_d.get("last_candle_pattern", "Normal"),
                    "vpin_proxy": (inst_sum or {}).get("vpin_proxy"),
                    "cvd_20": (inst_sum or {}).get("cvd_20"),
                    "macro_bias_score": (inst_sum or {}).get("macro_bias_score"),
                    "smt_status": (inst_sum or {}).get("smt_status", "N/A"),
                    "session": _utc_session(t_now),
                }
                state.setdefault("open_trades", []).append(trade_obj)
                state["last_event"] = (
                    f"Opened {track} {sig['side']} {lot} lot @ ${sig['entry_price']:,.2f} "
                    f"(Conf {sig['confidence']:.0f}% · TP ${sig['tp_price']:,.2f} · SL ${sig['sl_price']:,.2f})"
                )

    state["last_tick_utc"] = t_now_iso
    save_bot_state(state)
    return state


def compute_diagnostics(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze all closed paper trades & generate actionable Self-Diagnostic insights."""
    if not history:
        return {"total_trades": 0, "insights": ["Abhi tak koi closed trade nahi hai — Bot Start karein aur trades collect hone dein."]}

    df = pd.DataFrame(history)
    n = len(df)
    wins = int((df["net_pnl_usd"] > 0).sum())
    losses = int((df["net_pnl_usd"] <= 0).sum())
    win_rate = round(wins / n * 100.0, 1)
    tp_hits = int((df["exit_reason"] == "TP_HIT").sum())
    sl_hits = int((df["exit_reason"] == "SL_HIT").sum())
    longs = int((df["side"] == "LONG").sum())
    shorts = int((df["side"] == "SHORT").sum())
    gross_pos = float(df.loc[df["net_pnl_usd"] > 0, "net_pnl_usd"].sum())
    gross_neg = abs(float(df.loc[df["net_pnl_usd"] <= 0, "net_pnl_usd"].sum()))
    pf = round(gross_pos / gross_neg, 2) if gross_neg > 0 else (99.0 if gross_pos > 0 else 0.0)

    def _group_stats(col: str) -> list[dict[str, Any]]:
        if col not in df.columns:
            return []
        out = []
        for val, g in df.groupby(col, dropna=False):
            gn = len(g)
            gw = int((g["net_pnl_usd"] > 0).sum())
            out.append(
                {
                    "Group": str(val),
                    "Trades": gn,
                    "Wins": gw,
                    "Losses": gn - gw,
                    "Win%": round(gw / gn * 100.0, 1),
                    "Net PnL ($)": round(float(g["net_pnl_usd"].sum()), 2),
                    "Avg R": round(float(g["r_multiple"].mean()), 2) if "r_multiple" in g else 0.0,
                }
            )
        return sorted(out, key=lambda r: r["Net PnL ($)"], reverse=True)

    by_track = _group_stats("track")
    by_regime = _group_stats("regime_at_entry")
    by_lot = _group_stats("lot_tier")
    by_session = _group_stats("session")
    by_engine = _group_stats("forecaster_engine")
    by_exit = _group_stats("exit_reason")

    # Auto-generated diagnostic findings ("Kahan-Kahan Masla Hai?")
    insights: list[str] = []
    for row in by_regime:
        if row["Trades"] >= 3 and row["Net PnL ($)"] < 0 and "CHOP" in row["Group"]:
            insights.append(
                f"⚠️ **CHOP Regime Bleed:** `CHOP / RANGE` mein {row['Trades']} trades par `${row['Net PnL ($)']:.2f}` loss hua (Win {row['Win%']}%) — CHOP mein minimum confidence 75%+ karein ya skip karein."
            )
    for row in by_track:
        if row["Trades"] >= 3:
            ico = "✅" if row["Net PnL ($)"] > 0 else "🔴"
            insights.append(
                f"{ico} **Track `{row['Group']}`:** {row['Trades']} trades · Win `{row['Win%']}%` · Net **`${row['Net PnL ($)']:+.2f}`** (Avg `{row['Avg R']:+.2f}R`)."
            )
    for row in by_lot:
        if "HEAVY" in row["Group"] or "HIGH" in row["Group"]:
            if row["Trades"] >= 2 and row["Net PnL ($)"] < 0:
                insights.append(
                    f"⚠️ **Heavy Lot Warning ({row['Group']}):** {row['Trades']} trades par `${row['Net PnL ($)']:.2f}` — heavy lot threshold ko +5% tight karein."
                )
    if sl_hits > tp_hits and n >= 5:
        insights.append(
            f"🔍 **SL vs TP Ratio:** SL hits ({sl_hits}) > TP hits ({tp_hits}) — check karein ke SL bohat tight to nahi (`1.5×ATR` se `1.8×ATR` test karein)."
        )
    if not insights:
        insights.append("✅ Data jama ho raha hai — kam az kam 10+ closed trades ke baad deeper pattern insights dikhengi.")

    return {
        "total_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "tp_hits": tp_hits,
        "sl_hits": sl_hits,
        "longs": longs,
        "shorts": shorts,
        "gross_pnl_usd": round(float(df["gross_pnl_usd"].sum()), 2),
        "total_costs_usd": round(float((df["spread_cost_usd"] + df["commission_usd"]).sum()), 2),
        "net_pnl_usd": round(float(df["net_pnl_usd"].sum()), 2),
        "profit_factor": pf,
        "avg_r": round(float(df["r_multiple"].mean()), 2),
        "best_trade_usd": round(float(df["net_pnl_usd"].max()), 2),
        "worst_trade_usd": round(float(df["net_pnl_usd"].min()), 2),
        "by_track": by_track,
        "by_regime": by_regime,
        "by_lot": by_lot,
        "by_session": by_session,
        "by_engine": by_engine,
        "by_exit": by_exit,
        "insights": insights,
    }
