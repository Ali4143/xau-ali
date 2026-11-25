"""Unit tests for strict Directional Accuracy (flat pred=0 and zero y=0 excluded)."""
from __future__ import annotations

import numpy as np

from xau_mtf.evaluation import compute_directional_accuracy


def test_flat_forecast_never_scores_down_bars_as_correct() -> None:
    y = np.array([-5.0, -2.0, -1.0, 3.0, -4.0])
    pred_flat = np.zeros_like(y)
    res = compute_directional_accuracy(y, pred_flat)
    assert res["da_pct"] is None, f"Expected None for flat forecast, got {res}"
    assert res["label"] == "N/A (flat forecast)"
    assert res["excluded_flat"] == 5
    assert res["n_eval"] == 0
    print("[PASS] test_flat_forecast_never_scores_down_bars_as_correct")


def test_mixed_directions_and_zero_truth_exclusion() -> None:
    # 5 samples:
    # 0: pred +1, y +2 -> CORRECT
    # 1: pred -1, y -3 -> CORRECT
    # 2: pred +1, y -1 -> WRONG
    # 3: pred  0, y -5 -> EXCLUDED (flat pred)
    # 4: pred +2, y  0 -> EXCLUDED (zero truth)
    y = np.array([2.0, -3.0, -1.0, -5.0, 0.0])
    pred = np.array([1.0, -1.0, 1.0, 0.0, 2.0])
    res = compute_directional_accuracy(y, pred)
    assert res["n_eval"] == 3
    assert res["excluded_flat"] == 1
    assert res["excluded_zero_y"] == 1
    assert abs(res["da_pct"] - 66.67) < 0.02, f"Expected 66.67%, got {res['da_pct']}"
    print(f"[PASS] test_mixed_directions_and_zero_truth_exclusion: DA={res['da_pct']}% (excl_flat=1, excl_zero_y=1)")


if __name__ == "__main__":
    test_flat_forecast_never_scores_down_bars_as_correct()
    test_mixed_directions_and_zero_truth_exclusion()
    print("ALL DA TESTS PASSED 100%!")
