"""Automated tests for Dual-Track Paper Trading Bot (6H Fix + 15m Taaza) & Diagnostics."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import xau_mtf.paper_bot as pb


def _make_bars(prices: list[float]) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=end, periods=len(prices), freq="1min", tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p + 0.6, "low": p - 0.6, "close": p, "volume": np.full(len(p), 50.0)},
        index=idx,
    )


def test_paper_bot_full_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as td:
        pb.PAPER_DIR = Path(td)
        pb.BOT_STATE_FILE = pb.PAPER_DIR / "bot_state.json"
        pb.TRADE_HISTORY_FILE = pb.PAPER_DIR / "trade_history.jsonl"

        # 1. PnL & Lot sizing math check (0.01=$1/$1, 0.05=$5/$1, 0.10=$10/$1)
        g, sp, cm, net = pb._calc_pnl("LONG", 3500.0, 3510.0, 0.10, 0.30, 7.0)
        assert g == 100.0 and sp == 3.0 and cm == 0.70 and net == 96.30, (g, sp, cm, net)

        st = pb.reset_bot_account(5000.0, clear_history=True)
        st["running"] = True
        pb.save_bot_state(st)

        # 2. Entry on both 6H Fix and 15m Taaza (high confidence -> 0.05 or 0.10 lot)
        df = _make_bars([3500.0, 3500.2, 3500.5])
        fc_6h = {
            "id": "lock1",
            "forecaster": "tft",
            "P0": 3500.0,
            "atr15_usd": 6.0,
            "q50": [3500.5 + 0.25 * i for i in range(60)],
            "target_high": 3516.0,
            "target_low": 3492.0,
            "direction_meta": {"1h": {"p_up": 0.68}},
            "pivots_ny17": {"P": 3500.0},
        }
        fc_15m = {
            "id": "tz1",
            "forecaster": "tft",
            "P0": 3500.5,
            "atr15_usd": 6.0,
            "q50": [3500.5 + 0.45 * i for i in range(60)],
            "target_high": 3512.0,
            "target_low": 3496.0,
            "direction_meta": {"1h": {"p_up": 0.72}},
            "pivots_ny17": {"P": 3500.0},
        }
        inst = {
            "regime_state": 1,
            "regime_label": "CLEAN TREND",
            "inst_score": 45.0,
            "atr15": 6.0,
            "vpin_proxy": 0.55,
            "cvd_20": 0.35,
            "macro_bias_score": 30.0,
            "smt_status": "🟢 BULLISH SMT",
            "levels": {"PDL": 3500.2},
            "patterns": {
                "last_candle_pattern": "Bull Engulfing",
                "active_ladder": {
                    "dir": 1,
                    "pattern": "Bull Engulfing + Sweep",
                    "confidence": 82.0,
                    "active_target_price": 3506.0,
                    "fast_momentum": False,
                },
            },
        }
        st = pb.tick_paper_bot(df, fc_6h, fc_15m, inst)
        assert len(st["open_trades"]) == 2, f"Expected 2 open trades, got {len(st['open_trades'])}"
        lots = {t["track"]: t["lot_size"] for t in st["open_trades"]}
        assert lots["6H_FIX"] in (0.05, 0.10) and lots["15M_TAAZA"] in (0.05, 0.10), lots

        # 3. Price rallies to hit 15M_TAAZA TP ($3506) & arm 6H_FIX Breakeven, then hits 6H_FIX TP ($3516)
        df2 = _make_bars([3500.0, 3500.2, 3500.5, 3507.0, 3510.0, 3517.0])
        st = pb.tick_paper_bot(df2, fc_6h, fc_15m, inst)
        assert len(st["open_trades"]) == 0, f"Both trades should have hit TP: {st['open_trades']}"
        hist = pb.load_trade_history()
        assert len(hist) == 2 and all(h["exit_reason"] == "TP_HIT" for h in hist), hist
        assert st["balance"] > 5000.0, st["balance"]

        # 4. Self-Diagnostic report check
        diag = pb.compute_diagnostics(hist)
        assert diag["total_trades"] == 2 and diag["tp_hits"] == 2 and diag["win_rate_pct"] == 100.0
        print(
            f"[PASS] test_paper_bot_full_lifecycle: 2/2 TP_HIT, Balance=${st['balance']:,.2f}, "
            f"Lots={lots}, ProfitFactor={diag['profit_factor']}"
        )


if __name__ == "__main__":
    test_paper_bot_full_lifecycle()
    print("PAPER TRADING BOT TESTS PASSED 100%!")
