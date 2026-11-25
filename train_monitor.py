"""Real-time Training & Optuna Telemetry Tracker (shared by CLI, VS Code terminal, and Streamlit UI)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

STATUS_FILE = config.ARTIFACTS_DIR / "training_status.json"
LIVE_LOG_FILE = config.ARTIFACTS_DIR / "training_live.log"


class TrainTracker:
    """Writes structured live progress to training_status.json + training_live.log + stdout (VS Code)."""

    def __init__(self, action: str, symbol: str, all_stages: list[str]):
        self.action = action
        self.symbol = symbol.upper()
        self.all_stages = list(all_stages)
        self.done_stages: list[str] = []
        self.t0 = time.time()
        self.logs: list[str] = []
        self.state = {
            "state": "running",
            "action": action,
            "symbol": self.symbol,
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": 0,
            "progress_pct": 0.0,
            "current_stage": "initializing",
            "step_label": "",
            "train_loss": None,
            "val_loss": None,
            "best_val_loss": None,
            "val_da_1h": None,
            "val_da_3h": None,
            "val_da_eos": None,
            "models_done": [],
            "models_remaining": list(all_stages),
            "recent_logs": [],
            "error": None,
        }
        LIVE_LOG_FILE.write_text("", encoding="utf-8")
        self._flush()

    def _flush(self) -> None:
        self.state["elapsed_sec"] = int(time.time() - self.t0)
        self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.state["models_done"] = list(self.done_stages)
        self.state["models_remaining"] = [s for s in self.all_stages if s not in self.done_stages]
        self.state["recent_logs"] = self.logs[-35:]
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        try:
            tmp.replace(STATUS_FILE)
        except Exception:
            STATUS_FILE.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts} UTC] {msg}"
        print(line, flush=True)
        self.logs.append(line)
        try:
            with open(LIVE_LOG_FILE, "a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except Exception:
            pass
        self._flush()

    def start_stage(self, stage: str) -> None:
        self.state["current_stage"] = stage
        self.state["step_label"] = "starting..."
        self.state["train_loss"] = None
        self.state["val_loss"] = None
        self.state["best_val_loss"] = None
        self.state["val_da_1h"] = None
        self.state["val_da_3h"] = None
        self.state["val_da_eos"] = None
        base_pct = (len(self.done_stages) / max(1, len(self.all_stages))) * 100.0
        self.state["progress_pct"] = round(base_pct, 1)
        self.log(f"▶ STAGE START: {stage.upper()} ({len(self.done_stages)+1}/{len(self.all_stages)})")

    def update_step(
        self,
        step_label: str,
        frac_in_stage: float = 0.0,
        train_loss: float | None = None,
        val_loss: float | None = None,
        best_val_loss: float | None = None,
        val_da_1h: float | None = None,
        val_da_3h: float | None = None,
        val_da_eos: float | None = None,
        log_msg: str | None = None,
    ) -> None:
        self.state["step_label"] = step_label
        if train_loss is not None:
            self.state["train_loss"] = round(float(train_loss), 5)
        if val_loss is not None:
            self.state["val_loss"] = round(float(val_loss), 5)
        if best_val_loss is not None:
            self.state["best_val_loss"] = round(float(best_val_loss), 5)
        if val_da_1h is not None:
            self.state["val_da_1h"] = round(float(val_da_1h), 2)
        if val_da_3h is not None:
            self.state["val_da_3h"] = round(float(val_da_3h), 2)
        if val_da_eos is not None:
            self.state["val_da_eos"] = round(float(val_da_eos), 2)
        n = max(1, len(self.all_stages))
        pct = ((len(self.done_stages) + max(0.0, min(1.0, frac_in_stage))) / n) * 100.0
        self.state["progress_pct"] = round(min(99.5, pct), 1)
        if log_msg:
            self.log(log_msg)
        else:
            self._flush()

    def finish_stage(self, stage: str, summary: str = "") -> None:
        if stage not in self.done_stages:
            self.done_stages.append(stage)
        self.state["progress_pct"] = round((len(self.done_stages) / max(1, len(self.all_stages))) * 100.0, 1)
        self.log(f"✅ STAGE DONE: {stage.upper()} {summary}".strip())

    def finish(self, summary: str = "Completed") -> None:
        self.state["state"] = "done"
        self.state["progress_pct"] = 100.0
        self.state["current_stage"] = "completed"
        self.state["step_label"] = summary
        self.log(f"🏁 ALL DONE: {summary}")

    def fail(self, err: str) -> None:
        self.state["state"] = "error"
        self.state["error"] = str(err)
        self.log(f"❌ FAILED: {err}")


def read_training_status() -> dict | None:
    if not STATUS_FILE.exists():
        return None
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
