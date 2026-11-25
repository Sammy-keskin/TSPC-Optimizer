
# =====================================================
# TSPC OPTIMIZER V2.4
# diagnostic_timing.py
# =====================================================

from pathlib import Path

import pandas as pd

import config
import tspc
import transient

from frequency_sweep import WIDTHS
from validation import validate_transient


OUTPUT_DIR = Path("results/V2_4_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCIES = [100, 200, 990]

DELAYS = [1.0, 1.10, 1.25]


def main():

    summary = []

    original = (
        config.DATA_DELAY_CYCLES,
        tspc.DATA_DELAY_CYCLES,
        transient.DATA_DELAY_CYCLES
    )

    try:

        for freq in FREQUENCIES:

            for delay in DELAYS:

                print("\n" + "=" * 65)
                print(
                    f"{freq} MHz | "
                    f"Data delay = {delay:.2f}T"
                )

                # Temporary diagnostic override
                config.DATA_DELAY_CYCLES = delay
                tspc.DATA_DELAY_CYCLES = delay
                transient.DATA_DELAY_CYCLES = delay

                tag = f"{freq}MHz_delay_{delay:.2f}"

                try:

                    result = transient.simulate(
                        WIDTHS,
                        freq * 1e6,
                        cycles=10,
                        samples_per_cycle=1000,
                        timeout_s=60.0
                    )

                    validation = validate_transient(
                        result
                    )

                    checks = pd.DataFrame(
                        validation["checks"]
                    )

                    checks.to_csv(
                        OUTPUT_DIR / f"{tag}_checks.csv",
                        index=False
                    )

                    pd.DataFrame({
                        name: result[name]
                        for name in [
                            "time", "CLK", "D",
                            "X1", "Y1", "X2", "Y2",
                            "X3", "Y3", "Q"
                        ]
                    }).to_csv(
                        OUTPUT_DIR / f"{tag}_waveforms.csv",
                        index=False
                    )

                    row = {
                        "Frequency_MHz": freq,
                        "Delay_cycles": delay,
                        "Sample_pass": int(
                            checks["Sample_pass"].sum()
                        ),
                        "Hold_pass": int(
                            checks["Hold_pass"].sum()
                        ),
                        "Glitch_pass": int(
                            checks["Glitch_pass"].sum()
                        ),
                        "Passed_cycles":
                            validation["passed_cycles"],
                        "PASS": validation["passed"]
                    }

                    summary.append(row)

                    print(pd.Series(row).to_string())

                except RuntimeError as error:

                    summary.append({
                        "Frequency_MHz": freq,
                        "Delay_cycles": delay,
                        "Error": str(error)
                    })

                    print(f"SIMULATION ERROR: {error}")

                pd.DataFrame(summary).to_csv(
                    OUTPUT_DIR / "timing_summary.csv",
                    index=False
                )

    finally:

        # Restore assignment settings
        (
            config.DATA_DELAY_CYCLES,
            tspc.DATA_DELAY_CYCLES,
            transient.DATA_DELAY_CYCLES
        ) = original

    print("\n" + "=" * 65)
    print("DIAGNOSTIC TIMING RESULTS")
    print("=" * 65)

    print(
        pd.DataFrame(summary).to_string(index=False)
    )


if __name__ == "__main__":
    main()
