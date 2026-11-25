
# =====================================================
# TSPC OPTIMIZER V2.4
# failure_analysis.py
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import VDD, RISE_FALL_RATIO


INPUT_DIR = Path("results/V2_3_sweep")
OUTPUT_DIR = Path("results/V2_4_analysis")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FREQUENCIES = [100, 200, 990]

SIGNALS = [
    "CLK", "D",
    "X1", "Y1",
    "X2", "Y2",
    "X3", "Y3",
    "Q"
]


def analyze_cycle(df, frequency, cycle):

    period = 1.0 / (frequency * 1e6)

    start = (cycle - 1) * period

    edge = start + RISE_FALL_RATIO * period / 2

    sample = start + 0.45 * period

    t = df["time"].to_numpy()

    print(f"\nCycle {cycle}")
    print("-" * 55)

    print(f"{'Node':5s} {'At edge [V]':>15s} "
          f"{'At sample [V]':>15s}")

    for name in SIGNALS:

        values = df[name].to_numpy()

        at_edge = np.interp(edge, t, values)
        at_sample = np.interp(sample, t, values)

        print(
            f"{name:5s} "
            f"{at_edge:15.5f} "
            f"{at_sample:15.5f}"
        )


def plot_cycle(df, frequency, cycle):

    period = 1.0 / (frequency * 1e6)

    start = (cycle - 1) * period

    mask = (
        (df["time"] >= start - 0.25 * period)
        & (df["time"] <= start + period)
    )

    local = df.loc[mask]

    time_ns = (
        local["time"] - start
    ) * 1e9

    fig, axes = plt.subplots(
        len(SIGNALS),
        1,
        figsize=(13, 14),
        sharex=True
    )

    edge_ns = (
        RISE_FALL_RATIO * period / 2
    ) * 1e9

    sample_ns = 0.45 * period * 1e9

    for ax, name in zip(axes, SIGNALS):

        ax.plot(
            time_ns,
            local[name],
            linewidth=1.5
        )

        ax.axhline(
            VDD / 2,
            color="gray",
            linestyle="--",
            linewidth=0.7
        )

        ax.axvline(
            edge_ns,
            color="red",
            linestyle="--"
        )

        ax.axvline(
            sample_ns,
            color="green",
            linestyle="--"
        )

        ax.set_ylabel(f"{name} [V]")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(
        "Time relative to cycle start [ns]"
    )

    fig.suptitle(
        f"TSPC Failure Analysis | "
        f"{frequency} MHz | Cycle {cycle}"
    )

    fig.tight_layout()

    filename = OUTPUT_DIR / (
        f"cycle_{cycle}_{frequency}MHz.png"
    )

    fig.savefig(filename, dpi=180)

    plt.close(fig)

    print(f"Saved: {filename}")


def main():

    print("=" * 65)
    print("TSPC V2.4 - FAILURE ANALYSIS")
    print("=" * 65)

    for frequency in FREQUENCIES:

        waveform_file = INPUT_DIR / (
            f"waveforms_{frequency}MHz.csv"
        )

        validation_file = INPUT_DIR / (
            f"validation_{frequency}MHz.csv"
        )

        df = pd.read_csv(waveform_file)

        validation = pd.read_csv(validation_file)

        print("\n" + "=" * 65)
        print(f"FREQUENCY: {frequency} MHz")
        print("=" * 65)

        print(validation.to_string(index=False))

        # Cycle 2: simultaneous CLK/D rise
        # Cycle 3: stable HIGH input
        # Cycle 5: stable LOW input

        for cycle in [2, 3, 5]:

            analyze_cycle(df, frequency, cycle)

            plot_cycle(df, frequency, cycle)

    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
