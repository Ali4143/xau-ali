"""Model 3: Causal Dilated Temporal Convolutional Network (dilations 1, 2, 4, 8, 16, 32)."""
from __future__ import annotations

from .common import BaseNeuralForecaster, last_step, ordered_quantile_head


class TCNForecaster(BaseNeuralForecaster):
    name = "tcn"
    force_cpu = True  # Conv1D hits MIOpen bug on AMD ROCm Windows; fast (~18s/ep) on 16-core CPU
    default_hp = {
        "filters": 32,
        "kernel": 2,
        "levels": 6,
        "dense": 64,
        "dropout": 0.25,
        "lr": 2.0e-3,
        "batch": 64,
        "weight_decay": 1.0e-5,
    }

    def build_net(self, L: int, F: int, C_dim: int, H: int):
        import keras
        from keras import layers

        filters = int(self.hp["filters"])
        kernel = int(self.hp["kernel"])
        dr = float(self.hp["dropout"])
        dilations = [2**i for i in range(int(self.hp["levels"]))]  # [1, 2, 4, 8, 16, 32]

        seq = keras.Input((L, F), name="seq")
        ctx = keras.Input((C_dim,), name="ctx")

        x = layers.Conv1D(filters, 1, padding="causal")(seq)
        for d in dilations:
            r = layers.Conv1D(filters, kernel, dilation_rate=d, padding="causal", activation="relu")(x)
            r = layers.Dropout(dr)(r)
            r = layers.Conv1D(filters, kernel, dilation_rate=d, padding="causal", activation="relu")(r)
            r = layers.Dropout(dr)(r)
            x = layers.Add()([x, r])

        last = layers.Lambda(last_step, output_shape=(filters,))(x)
        h = layers.Concatenate()([last, ctx])
        h = layers.Dense(int(self.hp["dense"]), activation="relu")(h)
        h = layers.Dropout(dr)(h)
        out = layers.Reshape((H, 3))(layers.Dense(H * 3)(h))
        out = layers.Lambda(ordered_quantile_head, output_shape=(H, 3), name="quantiles")(out)
        return keras.Model([seq, ctx], out, name="tcn")
