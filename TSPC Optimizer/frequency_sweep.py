
# =====================================================
# TSPC OPTIMIZER V2.3
# frequency_sweep.py
# =====================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from transient import simulate
from validation import validate_transient
from visualization import plot_transient


RESULTS_DIR = Path("results/V2_3_sweep")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


WIDTHS = {
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


FREQUENCIES_MHZ = [
    100, 200, 300, 400, 500,
    600, 700, 800, 900, 990, 1000
]


def main():

    start = time.perf_counter()

    summary = []

    best_frequency = None
    best_result = None

    print("=" * 65)
    print("TSPC V2.3 - FREQUENCY SWEEP")
    print("=" * 65)

    for frequency_mhz in FREQUENCIES_MHZ:

        frequency = frequency_mhz * 1e6

        try:

            result = simulate(
                WIDTHS,
                frequency,
                cycles=10,
                samples_per_cycle=1000,
                timeout_s=60.0
            )

            validation = validate_transient(
                result
            )

            passed = validation["passed"]

            row = {
                "Frequency_MHz": frequency_mhz,
                "PASS": passed,
                "Status": "PASS" if passed else "FAIL",
                "Passed_cycles": (
                    validation["passed_cycles"]
                ),
                "Total_cycles": (
                    validation["total_cycles"]
                ),
                "Q_min_V": np.min(result["Q"]),
                "Q_max_V": np.max(result["Q"]),
                "Simulation_time_s": (
                    result["elapsed"]
                )
            }

            summary.append(row)

            pd.DataFrame(
                validation["checks"]
            ).to_csv(
                RESULTS_DIR
                / f"validation_{frequency_mhz}MHz.csv",
                index=False
            )

            pd.DataFrame({
                name: result[name]
                for name in [
                    "time", "CLK", "D",
                    "X1", "Y1", "X2",
                    "Y2", "X3", "Y3", "Q"
                ]
            }).to_csv(
                RESULTS_DIR
                / f"waveforms_{frequency_mhz}MHz.csv",
                index=False
            )

            if passed:

                best_frequency = frequency_mhz
                best_result = result

            status = "PASS" if passed else "FAIL"

            print(
                f"\n{frequency_mhz} MHz | "
                f"{status} | "
                f"{validation['passed_cycles']}/10"
            )

        except (RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:

            print(
                f"\n{frequency_mhz} MHz | "
                f"SIMULATION ERROR: {error}"
            )

            summary.append({
                "Frequency_MHz": frequency_mhz,
                "PASS": None,
                "Status": "SIMULATION ERROR",
                "Error": str(error)
            })

        # Save progress after each simulation

        pd.DataFrame(summary).to_csv(
            RESULTS_DIR / "frequency_summary.csv",
            index=False
        )

    print("\n" + "=" * 65)
    print("FREQUENCY SWEEP RESULTS")
    print("=" * 65)

    df = pd.DataFrame(summary)

    print(df.to_string(index=False))

    if best_frequency is not None:

        print(
            f"\nHighest tested passing frequency: "
            f"{best_frequency} MHz"
        )

        plot_transient(best_result)

    else:

        print(
            "\nNo frequency passed all checks."
        )

    elapsed = time.perf_counter() - start

    print(
        f"\nTotal execution time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
