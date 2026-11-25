"""Model 5: TimeMixer (ICLR 2024 multi-scale PDM + FMM pure Dense/matmul MLP mixing)."""
from __future__ import annotations

import numpy as np

from .common import BaseNeuralForecaster, ordered_quantile_head


def _ma_matrix(n: int, k: int) -> np.ndarray:
    k = max(1, min(int(k), n))
    A = np.zeros((n, n), dtype="float32")
    for i in range(n):
        lo, hi = max(0, i - k // 2), min(n, i + k // 2 + 1)
        A[i, lo:hi] = 1.0 / (hi - lo)
    return A


class TimeMixerForecaster(BaseNeuralForecaster):
    name = "timemixer"
    default_hp = {
        "d": 32,
        "levels": 3,
        "blocks": 1,
        "dense": 64,
        "dropout": 0.25,
        "ma": 5,
        "lr": 1.8e-3,
        "batch": 64,
        "weight_decay": 3.0e-4,
    }

    def build_net(self, L: int, F: int, C_dim: int, H: int):
        import keras
        from keras import layers, ops

        d = int(self.hp["d"])
        dr = float(self.hp["dropout"])
        nb = max(1, int(self.hp["blocks"]))
        levels = max(1, int(self.hp["levels"]))
        while levels > 1 and L // (2 ** (levels - 1)) < 4:
            levels -= 1
        Lc = (L // (2 ** (levels - 1))) * (2 ** (levels - 1))
        lens = [Lc // (2**i) for i in range(levels)]

        seq = keras.Input((L, F), name="seq")
        ctx = keras.Input((C_dim,), name="ctx")

        x = layers.Lambda(lambda t, Lc=Lc: t[:, -Lc:, :], output_shape=(Lc, F))(seq)
        x = layers.Dense(d)(x)

        def down(t, n):
            return ops.mean(ops.reshape(t, (-1, n // 2, 2, d)), axis=2)

        xs = [x]
        for i in range(1, levels):
            xs.append(layers.Lambda(down, arguments={"n": lens[i - 1]}, output_shape=(lens[i], d))(xs[-1]))

        def tmix(t, n_out):
            y = layers.Permute((2, 1))(t)
            y = layers.Dense(n_out, activation="gelu")(y)
            y = layers.Dense(n_out)(y)
            return layers.Permute((2, 1))(y)

        def trend(t, n):
            A = _ma_matrix(n, self.hp["ma"])
            return layers.Lambda(
                lambda z, A=A: ops.einsum("ij,bjd->bid", ops.convert_to_tensor(A), z),
                output_shape=(n, d),
            )(t)

        for _ in range(nb):
            tr = [trend(xs[i], lens[i]) for i in range(levels)]
            se = [layers.Subtract()([xs[i], tr[i]]) for i in range(levels)]
            for i in range(1, levels):
                se[i] = layers.Add()([se[i], tmix(se[i - 1], lens[i])])
            for i in range(levels - 2, -1, -1):
                tr[i] = layers.Add()([tr[i], tmix(tr[i + 1], lens[i])])
            new = []
            for i in range(levels):
                m = layers.Add()([se[i], tr[i]])
                m = layers.Dense(2 * d, activation="gelu")(m)
                m = layers.Dropout(dr)(layers.Dense(d)(m))
                new.append(layers.LayerNormalization()(layers.Add()([xs[i], m])))
            xs = new

        heads = []
        for i in range(levels):
            h = layers.Concatenate()([layers.Flatten()(xs[i]), ctx])
            h = layers.Dropout(dr)(layers.Dense(int(self.hp["dense"]), activation="gelu")(h))
            heads.append(layers.Dense(H * 3)(h))

        o = heads[0] if levels == 1 else layers.Add()(heads)
        out = layers.Reshape((H, 3))(o)
        out = layers.Lambda(ordered_quantile_head, output_shape=(H, 3), name="quantiles")(out)
        return keras.Model([seq, ctx], out, name="timemixer")
