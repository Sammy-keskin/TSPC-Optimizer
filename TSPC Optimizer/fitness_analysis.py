
# =====================================================
# TSPC OPTIMIZER V2.8
# fitness_analysis.py
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import VDD, RISE_FALL_RATIO


INPUT_DIR = Path("results/V2_7_combined")
OUTPUT_DIR = Path("results/V2_8_fitness")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCY = 200e6
PERIOD = 1 / FREQUENCY
CYCLES = 10

DESIGNS = [
    "BASE", "A", "B", "C",
    "AB", "AC", "BC", "ABC"
]


def calculate_fitness(df):

    t = df["time"].to_numpy()
    d = df["D"].to_numpy()
    q = df["Q"].to_numpy()

    rows = []

    for cycle in range(CYCLES):

        edge_time = (
            cycle + RISE_FALL_RATIO / 2
        ) * PERIOD

        sample_time = (
            cycle + 0.45
        ) * PERIOD

        d_edge = np.interp(edge_time, t, d)

        expected = int(d_edge >= VDD / 2)

        q_ideal = expected * VDD

        q_sample = np.interp(
            sample_time, t, q
        )

        error = abs(q_sample - q_ideal) / VDD

        squared_error = error ** 2

        if expected == 1:
            margin = q_sample - 0.9 * VDD
        else:
            margin = 0.1 * VDD - q_sample

        rows.append({
            "Cycle": cycle + 1,
            "Expected": expected,
            "Q_ideal_V": q_ideal,
            "Q_sample_V": q_sample,
            "Normalized_error": error,
            "Squared_error": squared_error,
            "Voltage_margin_V": margin,
            "Sample_pass": margin >= 0
        })

    checks = pd.DataFrame(rows)

    metrics = {
        "Sample_MSE":
            checks["Squared_error"].mean(),

        "Sample_RMSE":
            np.sqrt(
                checks["Squared_error"].mean()
            ),

        "Mean_error":
            checks["Normalized_error"].mean(),

        "Worst_error":
            checks["Normalized_error"].max(),

        "Worst_margin_V":
            checks["Voltage_margin_V"].min(),

        "Sample_pass":
            int(checks["Sample_pass"].sum())
    }

    return metrics, checks


def main():

    summary = []
    all_checks = []

    print("=" * 65)
    print("TSPC V2.8 - CONTINUOUS FITNESS")
    print("=" * 65)

    for design in DESIGNS:

        path = INPUT_DIR / f"{design}_waveforms.csv"

        df = pd.read_csv(path)

        metrics, checks = calculate_fitness(df)

        summary.append({
            "Design": design,
            **metrics
        })

        checks.insert(0, "Design", design)

        all_checks.append(checks)

        checks.to_csv(
            OUTPUT_DIR / f"{design}_fitness.csv",
            index=False
        )

    ranked = pd.DataFrame(summary)

    ranked = ranked.sort_values(
        "Sample_MSE",
        ascending=True
    )

    ranked.to_csv(
        OUTPUT_DIR / "fitness_ranked.csv",
        index=False
    )

    details = pd.concat(
        all_checks,
        ignore_index=True
    )

    details.to_csv(
        OUTPUT_DIR / "cycle_details.csv",
        index=False
    )

    print("\nFITNESS RANKING")
    print("-" * 65)

    print(ranked.to_string(index=False))

    print("\nFAILED CYCLE DETAILS")
    print("-" * 65)

    failed = details[
        ~details["Sample_pass"]
    ]

    print(
        failed[
            [
                "Design",
                "Cycle",
                "Expected",
                "Q_sample_V",
                "Normalized_error",
                "Voltage_margin_V"
            ]
        ].to_string(index=False)
    )

    matrix = details.pivot(
        index="Design",
        columns="Cycle",
        values="Normalized_error"
    ).reindex(DESIGNS)

    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(
        matrix.to_numpy(),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(
            1.0,
            matrix.to_numpy().max()
        ),
        aspect="auto"
    )

    ax.set_xticks(range(CYCLES))
    ax.set_xticklabels(range(1, CYCLES + 1))

    ax.set_yticks(range(len(DESIGNS)))
    ax.set_yticklabels(DESIGNS)

    for i in range(len(DESIGNS)):
        for j in range(CYCLES):

            value = matrix.iloc[i, j]

            ax.text(
                j, i,
                f"{value:.3f}",
                ha="center",
                va="center"
            )

    ax.set_xlabel("Clock cycle")
    ax.set_ylabel("Design")

    ax.set_title(
        "TSPC V2.8 - Normalized Output Error"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Normalized voltage error"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "fitness_heatmap.png",
        dpi=180
    )

    plt.close(fig)

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
