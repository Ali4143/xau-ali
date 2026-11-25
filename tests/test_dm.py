"""Unit tests for HAC Newey-West DM test, Holm-Bonferroni correction, and autocorrelation variance inflation."""
from __future__ import annotations

import numpy as np

from xau_mtf.evaluation import hac_dm_test, holm_bonferroni


def test_dm_direction_and_hac_autocorrelation_variance() -> None:
    rng = np.random.default_rng(7)
    n = 400
    # Create strongly positively autocorrelated loss differential d = AR(1) with phi=0.9 + positive edge
    eps = rng.normal(0.0, 1.0, n)
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 0.90 * d[i - 1] + eps[i]
    d = d - d.mean()
    loss_bench = 10.0 + d + 0.8
    loss_model = np.full(n, 10.0)  # model strictly lower loss by ~0.8 on average

    res_hac = hac_dm_test(loss_model, loss_bench, lag=20)
    res_naive = hac_dm_test(loss_model, loss_bench, lag=0)

    assert res_hac["dm_stat"] > 0, "Positive edge must give positive dm_stat"
    assert res_hac["var_hac"] > res_naive["var_naive"] * 2.0, (
        f"HAC variance ({res_hac['var_hac']}) must be larger than naive variance ({res_naive['var_naive']}) under AR(1)"
    )
    assert res_hac["p_value"] > res_naive["p_value"], "HAC p-value must be more conservative than naive p-value"

    # Reversed direction (model worse than benchmark) must give p > 0.5
    res_worse = hac_dm_test(loss_bench, loss_model, lag=20)
    assert res_worse["dm_stat"] < 0 and res_worse["p_value"] > 0.5

    # Insufficient sample check (n < 30)
    res_small = hac_dm_test(loss_model[:15], loss_bench[:15], lag=2)
    assert not res_small["sufficient_sample"] and "INSUFFICIENT" in res_small["status"]

    # Holm-Bonferroni ordering check
    adj = holm_bonferroni({"m1": 0.01, "m2": 0.03, "m3": 0.20})
    assert adj["m1"] == 0.03 and adj["m2"] == 0.06 and adj["m3"] == 0.20
    print(
        f"[PASS] test_dm_direction_and_hac_autocorrelation_variance: "
        f"var_naive={res_naive['var_naive']:.3f} < var_hac={res_hac['var_hac']:.3f}, Holm={adj}"
    )


if __name__ == "__main__":
    test_dm_direction_and_hac_autocorrelation_variance()
    print("ALL DM & HOLM TESTS PASSED 100%!")
