
# =====================================================
# TSPC OPTIMIZER V2
# transient.py
# =====================================================

import time
import numpy as np

from scipy.integrate import solve_ivp

from config import *
from tspc import *


def switching_times(frequency, cycles):

    period = 1.0 / frequency
    data_period = period / DATA_FREQ_RATIO

    stop = cycles * period

    points = [0.0, stop]

    # Clock transition boundaries
    for k in range(cycles):

        start = k * period
        tr = RISE_FALL_RATIO * period

        points.extend([
            start,
            start + tr,
            start + CLOCK_DUTY_CYCLE * period,
            start + CLOCK_DUTY_CYCLE * period + tr
        ])

    # Data transition boundaries
    delay = DATA_DELAY_CYCLES * period

    k = 0

    while delay + k * data_period < stop:

        start = delay + k * data_period
        tr = RISE_FALL_RATIO * data_period

        points.extend([
            start,
            start + tr,
            start + CLOCK_DUTY_CYCLE * data_period,
            start + CLOCK_DUTY_CYCLE * data_period + tr
        ])

        k += 1

    points = sorted(
        p for p in points if 0 <= p <= stop
    )

    # Remove nearly identical boundaries
    unique = []

    for p in points:

        if not unique or p - unique[-1] > 1e-8 * period:
            unique.append(p)

    return unique


def simulate(
    widths,
    frequency,
    cycles=10,
    samples_per_cycle=500,
    timeout_s=60.0,
    smoothing_v=0.01,
    rtol=1e-5,
    atol=1e-7
):

    circuit = TSPCCircuit(widths, smoothing_v=smoothing_v)

    period = 1.0 / frequency

    # Assumed initial logic state
    state = np.array([
        VDD,    # X1
        VDD,    # Y1
        VDD,    # X2
        0.0,    # Y2
        VDD,    # X3
        0.0,    # Y3
        0.0     # Q
    ])

    boundaries = switching_times(
        frequency, cycles
    )

    time_parts = []
    voltage_parts = []

    total_nfev = 0

    start_time = time.perf_counter()

    print(
        f"\nSimulating {frequency / 1e6:.1f} MHz"
    )

    print(
        f"Clock period: {period * 1e9:.4f} ns"
    )

    print(
        f"Integration segments: {len(boundaries) - 1}"
    )

    for i in range(len(boundaries) - 1):

        ta = boundaries[i]
        tb = boundaries[i + 1]

        count = max(
            3,
            int(
                (tb - ta) / period
                * samples_per_cycle
            ) + 1
        )

        evaluation_times = np.linspace(
            ta, tb, count
        )

        # Local time in clock periods; freeze the input slopes on this segment.
        span = (tb - ta) / period
        mid = (ta + tb) / 2
        sig = input_signals(mid, frequency)
        local_eval = np.linspace(0.0, span, count)

        def rhs(u, y):
            if timeout_s is not None and time.perf_counter() - start_time > timeout_s:
                raise RuntimeError(f"Timeout at segment {i+1}/{len(boundaries)-1}")
            t = ta + u * period
            signals = dict(sig)
            for name in ("CLK", "D"):
                signals[name] = sig[name] + (t-mid) * sig["d"+name]
            return period * circuit.rhs(t, y, frequency, signals=signals)

        errors = []
        for method in ("Radau", "BDF"):
            sol = solve_ivp(rhs, (0.0, span), state.copy(), method=method,
                            t_eval=local_eval, rtol=rtol, atol=atol,
                            max_step=min(0.01, span))
            total_nfev += sol.nfev
            if sol.success and np.all(np.isfinite(sol.y)):
                break
            errors.append(f"{method}: {sol.message}")
            print(f"Segment {i+1}: {errors[-1]}", flush=True)
        else:
            raise RuntimeError(f"Segment {i+1}, {ta*1e9:.6g}-{tb*1e9:.6g} ns: " + "; ".join(errors))
        sol.t = evaluation_times

        state = sol.y[:, -1]

        if i == 0:

            time_parts.append(sol.t)
            voltage_parts.append(sol.y)

        else:

            time_parts.append(sol.t[1:])
            voltage_parts.append(sol.y[:, 1:])

        if (i + 1) % 10 == 0:

            print(
                f"Segment {i + 1}/"
                f"{len(boundaries) - 1}",
                flush=True
            )

    t = np.concatenate(time_parts)

    y = np.concatenate(
        voltage_parts, axis=1
    )

    result = {
        "time": t,
        "frequency": frequency,
        "period": period
    }

    for i, name in enumerate(NODES):

        result[name] = y[i]

    result["CLK"] = np.array([
        input_signals(ti, frequency)["CLK"]
        for ti in t
    ])

    result["D"] = np.array([
        input_signals(ti, frequency)["D"]
        for ti in t
    ])

    result["nfev"] = total_nfev

    result["elapsed"] = (
        time.perf_counter() - start_time
    )

    print(
        f"Function evaluations: {total_nfev}"
    )

    print(
        f"Simulation time: "
        f"{result['elapsed']:.2f} seconds"
    )

    return result
