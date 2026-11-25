"""Shared Keras/PyTorch utilities, masked pinball loss, ordered quantile head, and AMD ROCm-safe layers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from .. import config
from ..dataset import Samples

Q_ARR = np.array(config.QUANTILES, dtype=np.float32)


def pinball_loss_np(y: np.ndarray, m: np.ndarray, pred: np.ndarray) -> float:
    """Masked pinball loss: y, m [N, H], pred [N, H, 3]."""
    e = y[..., None] - pred
    loss = np.maximum(Q_ARR * e, (Q_ARR - 1.0) * e) * m[..., None]
    return float(loss.sum() / max(1.0, float(m.sum()) * len(Q_ARR)))


def make_keras_pinball_loss(direction_penalty: float = 0.0):
    import keras
    from keras import ops

    q = ops.convert_to_tensor(Q_ARR)

    def masked_pinball(y_true, y_pred):
        y, m = y_true[..., 0], y_true[..., 1]
        e = ops.expand_dims(y, -1) - y_pred
        pl = ops.maximum(q * e, (q - 1.0) * e) * ops.expand_dims(m, -1)
        out = ops.sum(pl) / ops.maximum(1.0, ops.sum(m) * 3.0)
        if direction_penalty > 0.0:
            wrong = ops.relu(-y * y_pred[..., 1]) * m
            out = out + direction_penalty * ops.sum(wrong) / ops.maximum(1.0, ops.sum(m))
        return out

    return masked_pinball


def ordered_quantile_head(x):
    """Guarantees q10 <= q50 <= q90 via softplus offsets."""
    from keras import ops

    mid = x[..., 1:2]
    lo = mid - ops.softplus(x[..., 0:1])
    hi = mid + ops.softplus(x[..., 2:3])
    return ops.concatenate([lo, mid, hi], axis=-1)


def safe_lstm(units: int, **kw):
    """LSTM with use_cudnn=False to stay 100% stable on AMD ROCm / Windows."""
    from keras import layers

    try:
        return layers.LSTM(units, use_cudnn=False, **kw)
    except TypeError:
        return layers.LSTM(units, **kw)


def last_step(t):
    return t[:, -1, :]


def cpu_scope(on: bool = True):
    """Run Keras-torch strictly on CPU (avoids AMD ROCm Windows MIOpen Conv1D/RNN driver hangs)."""
    import contextlib

    if not on:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _cm():
        import keras

        core = None
        old = None
        try:
            from keras.src.backend.torch import core as _core

            core, old = _core, _core.DEFAULT_DEVICE
            _core.DEFAULT_DEVICE = "cpu"
        except Exception:
            pass
        try:
            with keras.device("cpu"):
                yield
        finally:
            if core is not None:
                core.DEFAULT_DEVICE = old

    return _cm()


class BaseNeuralForecaster:
    """Base fit / predict / save / load wrapper for Keras-torch neural members."""

    name: str = "neural"
    default_hp: dict = {}

    def __init__(self, hp: dict | None = None, seeds=config.SEEDS, epochs: int = config.MAX_EPOCHS):
        self.hp = {**self.default_hp, **(hp or {})}
        self.seeds = tuple(seeds)
        self.epochs = int(epochs)
        self.nets: list = []
        self.history: list[dict] = []
        self.L: int = config.LOOKBACK_M1
        self.F: int = 0
        self.C_dim: int = 10
        self.H: int = config.HORIZON_BARS
        # Use AMD RX 9070 XT GPU by default (models like TCN override to cpu_only=True to avoid MIOpen Conv1D hang)
        self.cpu_only: bool = bool(getattr(self, "force_cpu", False))

    def build_net(self, L: int, F: int, C_dim: int, H: int):
        raise NotImplementedError

    @staticmethod
    def _ytrue(S: Samples) -> np.ndarray:
        return np.stack([S.y, S.m], axis=-1).astype(np.float32)

    def fit(self, tr: Samples, va: Samples, workdir: Path, progress_cb=None, tracker=None) -> "BaseNeuralForecaster":
        import keras
        from ..evaluation import compute_directional_accuracy

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.L, self.F = tr.X.shape[1], tr.X.shape[2]
        self.C_dim = tr.C.shape[1]
        self.H = tr.y.shape[1]
        self.nets, self.history = [], []

        # Subsample validation for fast per-epoch DA tracking (up to 400 rows)
        n_va_sub = min(len(va), 400)
        va_idx = np.linspace(0, len(va) - 1, n_va_sub, dtype=int) if n_va_sub else np.array([], dtype=int)
        Xv_sub, Cv_sub = va.X[va_idx], va.C[va_idx]
        yv_sub, mv_sub, eos_sub = va.y[va_idx], va.m[va_idx], va.eos[va_idx]

        dev_tag = "CPU" if self.cpu_only else "AMD GPU (ROCm)"
        if tracker is not None:
            tracker.log(f"[{self.name.upper()}] Running on {dev_tag}")
        elif progress_cb:
            progress_cb(f"[{self.name.upper()}] Running on {dev_tag}")

        try:
            self._fit_inner(tr, va, workdir, progress_cb, tracker, n_va_sub, Xv_sub, Cv_sub, yv_sub, mv_sub, eos_sub)
        except Exception as e:
            err_s = f"{type(e).__name__}: {e}".lower()
            is_gpu_err = any(k in err_s for k in ("miopen", "hip", "cuda", "rocm", "oom", "out of memory", "device"))
            if not self.cpu_only and is_gpu_err:
                if tracker is not None:
                    tracker.log(f"⚠️ [{self.name.upper()}] GPU error ({str(e)[:120]}) — auto-retrying on CPU")
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.cpu_only = True
                self.nets, self.history = [], []
                self._fit_inner(tr, va, workdir, progress_cb, tracker, n_va_sub, Xv_sub, Cv_sub, yv_sub, mv_sub, eos_sub)
            else:
                raise
        return self

    def _fit_inner(self, tr, va, workdir, progress_cb, tracker, n_va_sub, Xv_sub, Cv_sub, yv_sub, mv_sub, eos_sub):
        import keras
        from ..evaluation import compute_directional_accuracy

        with cpu_scope(self.cpu_only):
            for seed in self.seeds:
                import random
                import torch

                random.seed(int(seed))
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))
                net = self.build_net(self.L, self.F, self.C_dim, self.H)
                opt = keras.optimizers.AdamW(
                    learning_rate=float(self.hp["lr"]),
                    weight_decay=float(self.hp.get("weight_decay", 1e-4)),
                    clipnorm=1.0,
                )
                net.compile(optimizer=opt, loss=make_keras_pinball_loss(float(self.hp.get("dir_penalty", 0.0))))
                ck = workdir / f"{self.name}_seed{seed}.weights.h5"

                model_name = self.name
                tot_ep = self.epochs

                class _LiveEpochCB(keras.callbacks.Callback):
                    def __init__(self):
                        super().__init__()
                        self.best_vl = float("inf")

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        tl = float(logs.get("loss", float("nan")))
                        vl = float(logs.get("val_loss", float("nan")))
                        if np.isfinite(vl):
                            self.best_vl = min(self.best_vl, vl)
                        da1 = da3 = dae = None
                        if n_va_sub >= 10:
                            q50 = self.model.predict({"seq": Xv_sub, "ctx": Cv_sub}, batch_size=256, verbose=0)[..., 1]
                            r = np.arange(n_va_sub)
                            k1 = min(config.H_1H, q50.shape[1]) - 1
                            k3 = min(config.H_3H, q50.shape[1]) - 1
                            ke = np.clip(eos_sub - 1, 0, q50.shape[1] - 1)
                            da1 = compute_directional_accuracy(yv_sub[:, k1], q50[:, k1], mv_sub[:, k1] > 0)["da_pct"]
                            da3 = compute_directional_accuracy(yv_sub[:, k3], q50[:, k3], mv_sub[:, k3] > 0)["da_pct"]
                            dae = compute_directional_accuracy(yv_sub[r, ke], q50[r, ke], mv_sub[r, ke] > 0)["da_pct"]
                        msg = (
                            f"[{model_name.upper()}] Epoch {epoch+1}/{tot_ep} | "
                            f"loss={tl:.4f} val_loss={vl:.4f} best={self.best_vl:.4f} | "
                            f"DA 1h={da1}% 3h={da3}% EOS={dae}%"
                        )
                        if tracker is not None:
                            tracker.update_step(
                                step_label=f"Epoch {epoch+1}/{tot_ep}",
                                frac_in_stage=(epoch + 1) / max(1, tot_ep),
                                train_loss=tl,
                                val_loss=vl,
                                best_val_loss=self.best_vl,
                                val_da_1h=da1,
                                val_da_3h=da3,
                                val_da_eos=dae,
                                log_msg=msg,
                            )
                        elif progress_cb:
                            progress_cb(msg)

                cbs = [
                    keras.callbacks.ModelCheckpoint(
                        str(ck), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min"
                    ),
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=config.ES_PATIENCE, mode="min", restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", patience=config.LR_PATIENCE, factor=0.5, mode="min", min_lr=1e-5
                    ),
                    _LiveEpochCB(),
                ]
                h = net.fit(
                    {"seq": tr.X, "ctx": tr.C},
                    self._ytrue(tr),
                    validation_data=({"seq": va.X, "ctx": va.C}, self._ytrue(va)),
                    epochs=self.epochs,
                    batch_size=int(self.hp["batch"]),
                    verbose=0,
                    callbacks=cbs,
                    shuffle=True,
                )
                if ck.exists():
                    net.load_weights(str(ck))
                self.nets.append(net)
                vl = h.history.get("val_loss", [float("nan")])
                self.history.append(
                    {
                        "seed": int(seed),
                        "epochs_run": len(vl),
                        "best_epoch": int(np.argmin(vl)) + 1,
                        "best_val_loss": float(np.min(vl)),
                    }
                )
                if progress_cb:
                    progress_cb(f"{self.name} seed {seed}: best_val_loss={np.min(vl):.5f} (ep {np.argmin(vl)+1})")

    def predict(self, S: Samples) -> np.ndarray:
        if not len(S) or not self.nets:
            return np.zeros((len(S), self.H, 3), dtype=np.float32)
        with cpu_scope(self.cpu_only):
            preds = [n.predict({"seq": S.X, "ctx": S.C}, batch_size=256, verbose=0) for n in self.nets]
        return np.mean(preds, axis=0).astype(np.float32)

    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for net, h in zip(self.nets, self.history):
            net.save_weights(str(d / f"{self.name}_seed{h['seed']}.weights.h5"))
        meta = {
            "name": self.name,
            "L": self.L,
            "F": self.F,
            "C_dim": self.C_dim,
            "H": self.H,
            "seeds": [h["seed"] for h in self.history],
            "hp": self.hp,
            "history": self.history,
            "cpu_only": self.cpu_only,
        }
        (d / f"{self.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> "BaseNeuralForecaster":
        d = Path(d)
        meta = json.loads((d / f"{cls.name}.json").read_text(encoding="utf-8"))
        obj = cls(hp=meta["hp"], seeds=meta["seeds"])
        obj.L, obj.F, obj.C_dim, obj.H = meta["L"], meta["F"], meta["C_dim"], meta["H"]
        obj.history = meta["history"]
        obj.cpu_only = bool(meta.get("cpu_only", True))
        with cpu_scope(obj.cpu_only):
            for s in meta["seeds"]:
                net = obj.build_net(obj.L, obj.F, obj.C_dim, obj.H)
                net.load_weights(str(d / f"{cls.name}_seed{s}.weights.h5"))
                obj.nets.append(net)
        return obj
