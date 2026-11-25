
# =====================================================
# TSPC OPTIMIZER V2.5
# scaling_experiment.py
# =====================================================

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transient import simulate
from validation import validate_transient
from frequency_sweep import WIDTHS
from layout import estimate_layout_area
from config import DATA_DELAY_CYCLES


OUTPUT_DIR = Path("results/V2_5_scaling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCIES = [
    100, 200, 300, 500, 750, 990
]

SCALES = {
    "V1_OPT": 1.00,
    "S75": 0.75,
    "S50": 0.50,
    "S25": 0.25,
    "S10": 0.10
}

ORIGINAL_WIDTHS = {
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


def scale_widths(widths, scale):

    return {
        name: max(4, int(np.floor(w * scale + 0.5)))
        for name, w in widths.items()
    }


def create_designs():

    designs = {
        name: scale_widths(WIDTHS, scale)
        for name, scale in SCALES.items()
    }

    designs["ORIGINAL"] = ORIGINAL_WIDTHS.copy()

    return designs


def plot_results(summary):

    df = pd.DataFrame(summary)

    matrix = df.pivot(
        index="Design",
        columns="Frequency_MHz",
        values="Passed_cycles"
    )

    matrix = matrix.reindex(
        list(SCALES.keys()) + ["ORIGINAL"]
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=0,
        vmax=10,
        aspect="auto"
    )

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):

            value = matrix.iloc[i, j]

            label = (
                "ERR" if pd.isna(value)
                else f"{int(value)}/10"
            )

            ax.text(
                j, i, label,
                ha="center",
                va="center"
            )

    ax.set_xlabel("Clock Frequency [MHz]")
    ax.set_ylabel("Design")

    ax.set_title(
        "TSPC V2.5 - Transistor Scaling Experiment"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Passed Clock Cycles"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "scaling_heatmap.png",
        dpi=180
    )

    plt.close(fig)


def main():

    start = time.perf_counter()

    assert DATA_DELAY_CYCLES == 1, (
        "Assignment data delay must be 1T."
    )

    designs = create_designs()
    summary = []

    print("=" * 65)
    print("TSPC V2.5 - TRANSISTOR SCALING")
    print("=" * 65)

    design_rows = []

    for design, widths in designs.items():

        area = estimate_layout_area(widths)

        design_rows.append({
            "Design": design,
            **widths,
            "Area_um2": area
        })

    pd.DataFrame(design_rows).to_csv(
        OUTPUT_DIR / "design_widths.csv",
        index=False
    )

    for design, widths in designs.items():

        area = estimate_layout_area(widths)

        print("\n" + "=" * 65)
        print(f"DESIGN: {design}")
        print(f"Estimated area: {area:.2f} um2")
        print("=" * 65)

        for freq in FREQUENCIES:

            tag = f"{design}_{freq}MHz"

            try:

                result = simulate(
                    widths,
                    freq * 1e6,
                    cycles=10,
                    samples_per_cycle=1000,
                    timeout_s=60.0
                )

                validation = validate_transient(result)

                checks = pd.DataFrame(
                    validation["checks"]
                )

                passed = validation["passed"]

                row = {
                    "Design": design,
                    "Frequency_MHz": freq,
                    "Area_um2": area,
                    "Passed_cycles":
                        validation["passed_cycles"],
                    "Sample_pass":
                        int(checks["Sample_pass"].sum()),
                    "Hold_pass":
                        int(checks["Hold_pass"].sum()),
                    "Glitch_pass":
                        int(checks["Glitch_pass"].sum()),
                    "PASS": passed,
                    "Status": "PASS" if passed else "FAIL",
                    "Simulation_time_s": result["elapsed"]
                }

                checks.to_csv(
                    OUTPUT_DIR / f"{tag}_checks.csv",
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
                    OUTPUT_DIR / f"{tag}_waveforms.csv",
                    index=False
                )

                print(
                    f"\n{design} | {freq} MHz | "
                    f"{row['Status']} | "
                    f"{row['Passed_cycles']}/10"
                )

            except (
                RuntimeError,
                np.linalg.LinAlgError,
                FloatingPointError
            ) as error:

                row = {
                    "Design": design,
                    "Frequency_MHz": freq,
                    "Area_um2": area,
                    "Passed_cycles": np.nan,
                    "PASS": None,
                    "Status": "SIMULATION ERROR",
                    "Error": str(error)
                }

                print(f"SIMULATION ERROR: {error}")

            summary.append(row)

            pd.DataFrame(summary).to_csv(
                OUTPUT_DIR / "scaling_summary.csv",
                index=False
            )

    df = pd.DataFrame(summary)

    print("\n" + "=" * 65)
    print("SCALING EXPERIMENT RESULTS")
    print("=" * 65)

    print(df.to_string(index=False))

    print("\nHIGHEST TESTED PASSING FREQUENCIES")
    print("-" * 65)

    for design in designs:

        passed = df[
            (df["Design"] == design)
            & (df["PASS"] == True)
        ]

        if passed.empty:
            print(f"{design:10s} | NO PASS")
        else:
            best = passed["Frequency_MHz"].max()
            print(f"{design:10s} | {best} MHz")

    plot_results(summary)

    print(
        f"\nTotal execution time: "
        f"{time.perf_counter() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
