
# =====================================================
# TSPC OPTIMIZER V2
# main_v2.py
# =====================================================

import numpy as np

from config import *
from transient import simulate
from visualization import plot_transient


def main():

    print("=" * 65)
    print("TSPC OPTIMIZER V2.0")
    print("Dynamic Transient Simulation")
    print("=" * 65)

    widths = {
        "P1": 826,
        "P2": 613,
        "P3": 1361,
        "P4": 759,
        "P5": 412,

        "N1": 88,
        "N2": 740,
        "N3": 1599,
        "N4": 630,
        "N5": 448,
        "N6": 279
    }

    frequency = 100e6

    result = simulate(
        widths,
        frequency,
        cycles=10,
        samples_per_cycle=500
    )

    print("\nNODE VOLTAGE RANGES")
    print("-" * 65)

    for name in NODES:

        vmin = np.min(result[name])
        vmax = np.max(result[name])

        print(
            f"{name:3s} | "
            f"Min={vmin:9.4f} V | "
            f"Max={vmax:9.4f} V"
        )

    print("\nCLOCK CYCLE OUTPUT CHECK")
    print("-" * 65)

    period = result["period"]

    passed = 0

    for cycle in range(10):

        edge_time = (
            cycle + RISE_FALL_RATIO / 2
        ) * period

        sample_time = (
            cycle + 0.45
        ) * period

        d_edge = np.interp(
            edge_time,
            result["time"],
            result["D"]
        )

        q = np.interp(
            sample_time,
            result["time"],
            result["Q"]
        )

        expected = int(d_edge >= VDD / 2)

        if expected == 1:
            success = q >= 0.9 * VDD
        else:
            success = q <= 0.1 * VDD

        if success:
            passed += 1

        status = "PASS" if success else "FAIL"

        print(
            f"Cycle {cycle + 1:2d} | "
            f"D_edge={d_edge:.4f} V | "
            f"Expected={expected} | "
            f"Q={q:.4f} V | "
            f"{status}"
        )

    print(
        f"\nFunctional check: {passed}/10"
    )
    plot_transient(result)

    print("\nSIMULATION COMPLETED")


if __name__ == "__main__":
    main()
