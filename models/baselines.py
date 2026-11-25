"""Model 8 (Baselines): NoChange (flat persistence) and HistoricalProfile (same UTC hour profile)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..dataset import Samples


class NoChangeBaseline:
    name = "no_change"

    def __init__(self):
        self.band: np.ndarray = np.zeros((config.HORIZON_BARS, 3), dtype=np.float32)

    def fit(self, tr: Samples) -> "NoChangeBaseline":
        H = tr.y.shape[1]
        self.band = np.zeros((H, 3), dtype=np.float32)
        for k in range(H):
            v = tr.y[tr.m[:, k] > 0, k]
            if len(v) >= 10:
                self.band[k] = np.quantile(v, config.QUANTILES)
        self.band[:, 1] = 0.0
        self.band = np.sort(self.band, axis=-1)
        return self

    def predict(self, S: Samples) -> np.ndarray:
        return np.repeat(self.band[None, :, :], len(S), axis=0)

    def save(self, d: Path) -> None:
        d = Path(d) / "baselines"
        d.mkdir(parents=True, exist_ok=True)
        (d / "no_change.json").write_text(json.dumps({"band": self.band.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "NoChangeBaseline":
        raw = json.loads((Path(d) / "baselines" / "no_change.json").read_text(encoding="utf-8"))
        obj = cls()
        obj.band = np.asarray(raw["band"], dtype=np.float32)
        return obj


class HistoricalProfileBaseline:
    name = "hist_profile"

    def __init__(self):
        self.tab: dict[int, np.ndarray] = {}
        self.fallback: np.ndarray = np.zeros((config.HORIZON_BARS, 3), dtype=np.float32)

    @staticmethod
    def _hours(S: Samples) -> np.ndarray:
        return pd.to_datetime(S.times, utc=True).hour.to_numpy(dtype=int)

    def fit(self, tr: Samples) -> "HistoricalProfileBaseline":
        H = tr.y.shape[1]
        hrs = self._hours(tr)
        self.fallback = np.zeros((H, 3), dtype=np.float32)
        for k in range(H):
            v = tr.y[tr.m[:, k] > 0, k]
            if len(v) >= 10:
                self.fallback[k] = [np.quantile(v, 0.1), float(np.mean(v)), np.quantile(v, 0.9)]
        self.fallback = np.sort(self.fallback, axis=-1)

        self.tab = {}
        for hr in range(24):
            sel = hrs == hr
            if sel.sum() < 10:
                continue
            prof = np.zeros((H, 3), dtype=np.float32)
            for k in range(H):
                v = tr.y[sel & (tr.m[:, k] > 0), k]
                if len(v) >= 5:
                    prof[k] = [np.quantile(v, 0.1), float(np.mean(v)), np.quantile(v, 0.9)]
            self.tab[hr] = np.sort(prof, axis=-1)
        return self

    def predict(self, S: Samples) -> np.ndarray:
        hrs = self._hours(S)
        H = self.fallback.shape[0]
        out = np.empty((len(S), H, 3), dtype=np.float32)
        for i, hr in enumerate(hrs):
            out[i] = self.tab.get(int(hr), self.fallback)
        return out

    def save(self, d: Path) -> None:
        d = Path(d) / "baselines"
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "fallback": self.fallback.tolist(),
            "tab": {str(k): v.tolist() for k, v in self.tab.items()},
        }
        (d / "hist_profile.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "HistoricalProfileBaseline":
        raw = json.loads((Path(d) / "baselines" / "hist_profile.json").read_text(encoding="utf-8"))
        obj = cls()
        obj.fallback = np.asarray(raw["fallback"], dtype=np.float32)
        obj.tab = {int(k): np.asarray(v, dtype=np.float32) for k, v in raw["tab"].items()}
        return obj
