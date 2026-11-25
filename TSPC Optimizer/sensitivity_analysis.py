
# =====================================================
# TSPC OPTIMIZER V2.6
# sensitivity_analysis.py
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
from config import MAX_LAYOUT_AREA, DATA_DELAY_CYCLES


OUTPUT_DIR = Path("results/V2_6_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCY = 200

BASE_SCALE = 0.25

FACTORS = [0.5, 2.0]


def round_width(value):

    return max(4, int(np.floor(value + 0.5)))


BASE_WIDTHS = {
    name: round_width(width * BASE_SCALE)
    for name, width in WIDTHS.items()
}


def create_designs():

    designs = [
        ("BASE", "BASE", 1.0, BASE_WIDTHS.copy())
    ]

    for transistor in BASE_WIDTHS:

        for factor in FACTORS:

            widths = BASE_WIDTHS.copy()

            widths[transistor] = round_width(
                widths[transistor] * factor
            )

            name = f"{transistor}_x{factor:.2f}"

            designs.append(
                (name, transistor, factor, widths)
            )

    return designs


def plot_sensitivity(df):

    baseline = df.loc[
        df["Design"] == "BASE",
        "Passed_cycles"
    ].iloc[0]

    matrix = df[df["Design"] != "BASE"].pivot(
        index="Transistor",
        columns="Factor",
        values="Passed_cycles"
    )

    matrix = matrix.reindex(
        list(BASE_WIDTHS.keys())
    )

    delta = matrix - baseline

    fig, ax = plt.subplots(figsize=(8, 8))

    image = ax.imshow(
        delta.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=-10,
        vmax=10,
        aspect="auto"
    )

    ax.set_xticks(range(len(matrix.columns)))

    ax.set_xticklabels([
        f"{factor:.2f}x"
        for factor in matrix.columns
    ])

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

    ax.set_title(
        f"TSPC V2.6 - Sensitivity at "
        f"{FREQUENCY} MHz\n"
        f"Baseline: {baseline:.0f}/10"
    )

    ax.set_xlabel("Width scaling factor")

    fig.colorbar(
        image,
        ax=ax,
        label="Change in passed cycles"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "sensitivity_heatmap.png",
        dpi=180
    )

    plt.close(fig)


def main():

    start = time.perf_counter()

    assert DATA_DELAY_CYCLES == 1

    designs = create_designs()
    summary = []

    print("=" * 65)
    print("TSPC V2.6 - TRANSISTOR SENSITIVITY")
    print("=" * 65)

    print("\nBASE WIDTHS")

    for name, width in BASE_WIDTHS.items():
        print(f"{name:5s} | {width:5d} um")

    for index, design in enumerate(designs, 1):

        name, transistor, factor, widths = design

        area = estimate_layout_area(widths)

        print("\n" + "=" * 65)

        print(
            f"Experiment {index}/{len(designs)}"
            f" | {name}"
        )

        print(f"Estimated area: {area:.2f} um2")

        row = {
            "Design": name,
            "Transistor": transistor,
            "Factor": factor,
            "Area_um2": area,
            **widths
        }

        if area > MAX_LAYOUT_AREA:

            row.update({
                "Passed_cycles": np.nan,
                "PASS": False,
                "Status": "AREA LIMIT"
            })

        else:

            try:

                result = simulate(
                    widths,
                    FREQUENCY * 1e6,
                    cycles=10,
                    samples_per_cycle=1000,
                    timeout_s=60.0
                )

                validation = validate_transient(result)

                checks = pd.DataFrame(
                    validation["checks"]
                )

                passed = validation["passed"]

                row.update({
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
                })

                checks.to_csv(
                    OUTPUT_DIR / f"{name}_checks.csv",
                    index=False
                )

                pd.DataFrame({
                    node: result[node]
                    for node in [
                        "time", "CLK", "D",
                        "X1", "Y1", "X2", "Y2",
                        "X3", "Y3", "Q"
                    ]
                }).to_csv(
                    OUTPUT_DIR / f"{name}_waveforms.csv",
                    index=False
                )

                print(
                    f"\n{name} | "
                    f"{row['Status']} | "
                    f"{row['Passed_cycles']}/10"
                )

            except (
                RuntimeError,
                np.linalg.LinAlgError,
                FloatingPointError
            ) as error:

                row.update({
                    "Passed_cycles": np.nan,
                    "PASS": None,
                    "Status": "SIMULATION ERROR",
                    "Error": str(error)
                })

                print(f"ERROR: {error}")

        summary.append(row)

        pd.DataFrame(summary).to_csv(
            OUTPUT_DIR / "sensitivity_summary.csv",
            index=False
        )

    df = pd.DataFrame(summary)

    baseline = df.loc[
        df["Design"] == "BASE",
        "Passed_cycles"
    ].iloc[0]

    df["Delta_cycles"] = (
        df["Passed_cycles"] - baseline
    )

    ranked = df.sort_values(
        by=[
            "Passed_cycles",
            "Sample_pass",
            "Hold_pass",
            "Glitch_pass",
            "Area_um2"
        ],
        ascending=[
            False, False, False, False, True
        ]
    )

    ranked.to_csv(
        OUTPUT_DIR / "sensitivity_ranked.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("SENSITIVITY RANKING")
    print("=" * 65)

    columns = [
        "Design",
        "Area_um2",
        "Passed_cycles",
        "Delta_cycles",
        "Sample_pass",
        "Hold_pass",
        "Glitch_pass",
        "Status"
    ]

    print(
        ranked[columns].to_string(index=False)
    )

    plot_sensitivity(df)

    print(
        f"\nTotal execution time: "
        f"{time.perf_counter() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
