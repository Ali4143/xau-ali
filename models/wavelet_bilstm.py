"""Model 4: Wavelet-LSTM (2-layer stacked LSTM 96 -> 32 units with causal wavelet feature input)."""
from __future__ import annotations

from .common import BaseNeuralForecaster, ordered_quantile_head, safe_lstm


class WaveletBiLSTMForecaster(BaseNeuralForecaster):
    name = "wavelet_bilstm"
    default_hp = {
        "units1": 96,
        "units2": 32,
        "dense": 48,
        "dropout": 0.15,
        "rdrop": 0.10,
        "lr": 1.0e-3,
        "batch": 64,
        "weight_decay": 1.0e-4,
    }

    def build_net(self, L: int, F: int, C_dim: int, H: int):
        import keras
        from keras import layers

        dr = float(self.hp["dropout"])
        rdr = float(self.hp["rdrop"])

        seq = keras.Input((L, F), name="seq")
        ctx = keras.Input((C_dim,), name="ctx")

        x = safe_lstm(int(self.hp["units1"]), return_sequences=True, dropout=dr, recurrent_dropout=rdr)(seq)
        x = safe_lstm(int(self.hp["units2"]), dropout=dr, recurrent_dropout=rdr)(x)
        h = layers.Concatenate()([x, ctx])
        h = layers.Dense(int(self.hp["dense"]), activation="relu")(h)
        h = layers.Dropout(dr)(h)
        out = layers.Reshape((H, 3))(layers.Dense(H * 3)(h))
        out = layers.Lambda(ordered_quantile_head, output_shape=(H, 3), name="quantiles")(out)
        return keras.Model([seq, ctx], out, name="wavelet_bilstm")
