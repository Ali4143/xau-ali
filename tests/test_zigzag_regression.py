"""Regression guard: PathSimulator must produce IDENTICAL zig-zag output for same seed & target q50."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xau_mtf import config
from xau_mtf.fetcher.fetch_adapter import load_m1
from xau_mtf.models.path_simulator import PathSimulator


def test_zigzag_matches_reference_identically() -> None:
    ref_path = config.REPORTS_DIR / "v2" / "zigzag_reference.json"
    assert ref_path.exists(), f"Missing reference file: {ref_path}"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    df1 = load_m1("XAUUSD")
    df_upto = df1[df1.index <= ref["origin_utc"]]
    sim = PathSimulator.load(config.MODELS_DIR / "xauusd")

    out = sim.simulate(
        df_upto,
        horizon=len(ref["path"]),
        n_paths=300,
        target_price_q50=ref["target_price_q50"],
        seed=int(ref["seed"]),
    )
    max_diff = float(np.max(np.abs(np.asarray(out["path"]) - np.asarray(ref["path"]))))
    assert max_diff < 1e-9, f"Zig-zag path drifted from reference! max_diff={max_diff}"
    assert abs(out["targets"]["day_high"][1] - ref["target_high"]) < 1e-9
    assert abs(out["targets"]["day_low"][1] - ref["target_low"]) < 1e-9
    print(f"[PASS] test_zigzag_matches_reference_identically: max_diff={max_diff:.2e}, TH={ref['target_high']:.2f}, TL={ref['target_low']:.2f}")


if __name__ == "__main__":
    test_zigzag_matches_reference_identically()
    print("ZIG-ZAG REGRESSION TEST PASSED 100%!")
