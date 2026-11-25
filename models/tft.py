"""Model 1: Temporal Fusion Transformer (lite, AMD ROCm-safe) with VSN, GRN, and manual MHA."""
from __future__ import annotations

import numpy as np

from .common import BaseNeuralForecaster, last_step, ordered_quantile_head, safe_lstm


def _glu(x, d: int):
    from keras import layers

    return layers.Multiply()([layers.Dense(d)(x), layers.Dense(d, activation="sigmoid")(x)])


def _grn(x, d: int, dropout: float, ctx=None):
    from keras import layers

    skip = x if x.shape[-1] == d else layers.Dense(d)(x)
    h = layers.Dense(d)(x)
    if ctx is not None:
        h = layers.Add()([h, layers.Dense(d, use_bias=False)(ctx)])
    h = layers.Activation("elu")(h)
    h = layers.Dropout(dropout)(layers.Dense(d)(h))
    return layers.LayerNormalization()(layers.Add()([skip, _glu(h, d)]))


def _manual_mha(x, d: int, heads: int, dr: float):
    """Multi-head self-attention via pure matmul (avoids fused SDPA crash on AMD ROCm Windows)."""
    from keras import layers, ops

    dk = max(4, d // heads)
    q = layers.Dense(dk * heads)(x)
    k = layers.Dense(dk * heads)(x)
    v = layers.Dense(dk * heads)(x)

    def att(t, heads=heads, dk=dk):
        q_, k_, v_ = t
        b_, n_ = ops.shape(q_)[0], ops.shape(q_)[1]
        sh = lambda z: ops.transpose(ops.reshape(z, (b_, n_, heads, dk)), (0, 2, 1, 3))  # noqa: E731
        scores = ops.matmul(sh(q_), ops.transpose(sh(k_), (0, 1, 3, 2))) / float(np.sqrt(dk))
        o_ = ops.matmul(ops.softmax(scores, axis=-1), sh(v_))
        return ops.reshape(ops.transpose(o_, (0, 2, 1, 3)), (b_, n_, heads * dk))

    o = layers.Lambda(att, output_shape=(None, dk * heads))([q, k, v])
    return layers.Dropout(dr)(layers.Dense(d)(o))


class TFTForecaster(BaseNeuralForecaster):
    name = "tft"
    default_hp = {
        "d": 32,
        "heads": 2,
        "dense": 48,
        "dropout": 0.10,
        "lr": 2.0e-3,
        "batch": 64,
        "weight_decay": 1.0e-5,
    }

    def build_net(self, L: int, F: int, C_dim: int, H: int):
        import keras
        from keras import layers

        d = int(self.hp["d"])
        dr = float(self.hp["dropout"])
        heads = max(1, int(self.hp["heads"]))

        seq = keras.Input((L, F), name="seq")
        ctx = keras.Input((C_dim,), name="ctx")

        c = _grn(ctx, d, dr)
        c_h, c_c, c_e, c_v = (_grn(c, d, dr) for _ in range(4))
        rep = lambda t: layers.RepeatVector(L)(t)  # noqa: E731

        # Variable Selection Network over all MTF (15M + 1M) features
        w = layers.Dense(F, activation="softmax", name="var_select")(_grn(seq, d, dr, ctx=rep(c_v)))
        xs = layers.Dense(d)(layers.Multiply()([seq, w]))

        enc_in = layers.Dense(d)(layers.Concatenate()([xs, rep(c_h), rep(c_c)]))
        enc = safe_lstm(d, return_sequences=True, dropout=dr)(enc_in)
        t = layers.LayerNormalization()(layers.Add()([xs, _glu(layers.Dropout(dr)(enc), d)]))

        e = _grn(t, d, dr, ctx=rep(c_e))
        att = _manual_mha(e, d, heads, dr)
        a = layers.LayerNormalization()(layers.Add()([e, _glu(layers.Dropout(dr)(att), d)]))
        o = _grn(a, d, dr)
        o = layers.LayerNormalization()(layers.Add()([t, _glu(o, d)]))

        last = layers.Lambda(last_step, output_shape=(d,))(o)
        pool = layers.GlobalAveragePooling1D()(o)
        h = layers.Concatenate()([last, pool, ctx])
        h = layers.Dropout(dr)(layers.Dense(int(self.hp["dense"]), activation="relu")(h))
        out = layers.Reshape((H, 3))(layers.Dense(H * 3)(h))
        out = layers.Lambda(ordered_quantile_head, output_shape=(H, 3), name="quantiles")(out)
        return keras.Model([seq, ctx], out, name="tft")
