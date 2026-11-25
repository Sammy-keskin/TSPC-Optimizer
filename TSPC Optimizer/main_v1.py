
# =====================================================
# TSPC OPTIMIZER V1.0
# main.py
# =====================================================

import time

from config import *
from timing import *
from layout import *
from optimizer import *
from visualization import *


def main():

    start_time = time.perf_counter()

    print("=" * 65)
    print("TSPC OPTIMIZER V1.0")
    print("Analytical Differential Evolution")
    print("=" * 65)

    table = build_delay_tables()

    print("\nDELAY COEFFICIENT VALIDATION")

    print(f"KN 50% : {table['KN']:.6f}")
    print(f"KP 50% : {table['KP']:.6f}")
    print(f"KN 90% : {table['KN90']:.6f}")
    print(f"KP 90% : {table['KP90']:.6f}")

    # Initial reference design
    baseline = {
        "P1": 40,
        "P2": 40,
        "P3": 40,
        "P4": 80,
        "P5": 160,
        "N1": 20,
        "N2": 20,
        "N3": 20,
        "N4": 40,
        "N5": 40,
        "N6": 60
    }

    baseline_timing = estimate_timing(baseline)

    print("\nBASELINE DESIGN")

    print(
        f"Estimated frequency: "
        f"{baseline_timing['F_EST_MHZ']:.3f} MHz"
    )

    print(
        f"Estimated area: "
        f"{estimate_layout_area(baseline):.2f} um^2"
    )

    result, widths, history = run_optimization()

    timing = estimate_timing(widths)
    area = estimate_layout_area(widths)

    print("\n" + "=" * 65)
    print("OPTIMIZATION RESULTS")
    print("=" * 65)

    print(f"Converged: {result.success}")
    print(f"Status: {result.message}")
    print(f"Evaluations: {result.nfev}")
    print(f"Generations: {result.nit}")

    print("\nTRANSISTOR WIDTHS")
    print("-" * 65)

    for name, width in widths.items():

        g = transistor_geometry(width)

        print(
            f"{name:3s} | "
            f"W={width:8.1f} um | "
            f"NF={g['NF']:3d} | "
            f"WF={g['WF']:7.2f} um"
        )

    print("\nTIMING RESULTS")

    for key, value in timing.items():

        if key == "CAPACITANCES":
            continue

        if key.startswith("F_"):
            unit = "MHz"
        else:
            unit = "ns"

        print(f"{key:20s}: {value:10.4f} {unit}")

    print("\nNODE CAPACITANCES")

    for node, cap in timing["CAPACITANCES"].items():
        print(f"{node}: {cap:.6f} pF")

    print("\nAREA RESULTS")

    print(f"Estimated area : {area:.2f} um^2")
    print(f"Estimated area : {area / 1e6:.6f} mm^2")
    print(f"Area usage     : {area / MAX_LAYOUT_AREA * 100:.2f}%")
    print(f"Estimated pass : {area <= MAX_LAYOUT_AREA}")

    improvement = 100 * (
        timing["F_EST_MHZ"]
        / baseline_timing["F_EST_MHZ"] - 1
    )

    print("\nPERFORMANCE IMPROVEMENT")
    print(f"Frequency improvement: {improvement:.2f}%")

    elapsed = time.perf_counter() - start_time

    print(f"Execution time: {elapsed:.2f} seconds")

    save_results(
        widths, timing, area, history
    )

    print("\nOPTIMIZATION COMPLETED")


if __name__ == "__main__":
    main()
