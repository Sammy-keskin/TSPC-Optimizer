
# =====================================================
# TSPC OPTIMIZER
# visualization.py
# =====================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def save_results(widths, timing, area, history):

    sizing = pd.DataFrame({
        "Transistor": list(widths.keys()),
        "Width_um": list(widths.values())
    })

    sizing.to_csv(
        RESULTS_DIR / "optimal_widths.csv",
        index=False
    )

    hist = pd.DataFrame(history)

    hist.to_csv(
        RESULTS_DIR / "optimization_history.csv",
        index=False
    )

    summary = {
        key: value
        for key, value in timing.items()
        if key != "CAPACITANCES"
    }

    summary["Area_um2"] = area

    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "design_summary.csv",
        index=False
    )

    if not hist.empty:

        plt.figure(figsize=(10, 5))

        plt.plot(
            hist["Generation"],
            hist["Frequency_MHz"],
            linewidth=2
        )

        plt.xlabel("Generation")
        plt.ylabel("Estimated frequency (MHz)")
        plt.title("Differential Evolution Convergence")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / "convergence.png",
            dpi=200
        )

        plt.close()

    plt.figure(figsize=(10, 5))

    colors = [
        "indianred" if name.startswith("P")
        else "steelblue"
        for name in widths
    ]

    plt.bar(
        list(widths.keys()),
        list(widths.values()),
        color=colors
    )

    plt.ylabel("Transistor width (um)")
    plt.title("Optimized TSPC Transistor Widths")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "transistor_widths.png",
        dpi=200
    )

    plt.close()

    print("\nResults saved in results/ directory.")

def plot_transient(result):

    frequency_mhz = result["frequency"] / 1e6

    time_ns = result["time"] * 1e9

    signals = [
        "CLK", "D",
        "X1", "Y1",
        "X2", "Y2",
        "X3", "Y3",
        "Q"
    ]

    fig, axes = plt.subplots(
        len(signals),
        1,
        figsize=(15, 15),
        sharex=True
    )

    for ax, name in zip(axes, signals):

        ax.plot(
            time_ns,
            result[name],
            linewidth=1.4
        )

        ax.axhline(
            1.5,
            color="gray",
            linestyle="--",
            linewidth=0.6
        )

        ax.set_ylabel(name + " [V]")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time [ns]")

    fig.suptitle(
        f"TSPC Transient Simulation - "
        f"{frequency_mhz:.0f} MHz",
        fontsize=15
    )

    plt.tight_layout()

    filename = (
        RESULTS_DIR
        / f"transient_{frequency_mhz:.0f}MHz.png"
    )

    plt.savefig(filename, dpi=200)

    plt.close(fig)

    data = {
        "Time_ns": time_ns
    }

    for name in signals:

        data[name] = result[name]

    pd.DataFrame(data).to_csv(
        RESULTS_DIR
        / f"transient_{frequency_mhz:.0f}MHz.csv",
        index=False
    )

    print(f"Waveforms saved: {filename}")
