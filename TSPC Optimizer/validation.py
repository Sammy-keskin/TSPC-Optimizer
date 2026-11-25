
# =====================================================
# TSPC OPTIMIZER V2.2
# validation.py
# =====================================================

import numpy as np

from config import *


def validate_transient(result):

    t = result["time"]
    q = result["Q"]
    d = result["D"]

    period = result["period"]

    cycles = int(round(t[-1] / period))

    low_limit = 0.1 * VDD
    high_limit = 0.9 * VDD

    checks = []

    for cycle in range(cycles):

        edge_time = (
            cycle + RISE_FALL_RATIO / 2
        ) * period

        sample_time = (
            cycle + 0.45
        ) * period

        d_edge = np.interp(
            edge_time, t, d
        )

        q_sample = np.interp(
            sample_time, t, q
        )

        expected = int(
            d_edge >= VDD / 2
        )

        # Check output at the end of evaluation

        if expected:
            sample_pass = q_sample >= high_limit
        else:
            sample_pass = q_sample <= low_limit

        # Check output retention until the next edge

        hold_end = (cycle + 1) * period

        mask = (
            (t >= sample_time)
            & (t < hold_end)
        )

        q_hold = np.append(
            q[mask], q_sample
        )

        hold_min = np.min(q_hold)
        hold_max = np.max(q_hold)

        if expected:
            hold_pass = hold_min >= high_limit
        else:
            hold_pass = hold_max <= low_limit

        # Count logic threshold crossings

        cycle_mask = (
            (t >= cycle * period)
            & (t < hold_end)
        )

        q_cycle = q[cycle_mask]

        logic = q_cycle >= VDD / 2

        crossings = np.count_nonzero(
            logic[1:] != logic[:-1]
        )

        if cycle == 0:
            expected_crossings = int(
                expected != int(q[0] >= VDD / 2)
            )
        else:
            expected_crossings = int(
                expected != checks[-1]["Expected"]
            )

        glitch_pass = (
            crossings == expected_crossings
        )

        passed = bool(
            sample_pass
            and hold_pass
            and glitch_pass
        )

        checks.append({
            "Cycle": cycle + 1,
            "D_edge_V": d_edge,
            "Expected": expected,
            "Q_sample_V": q_sample,
            "Q_hold_min_V": hold_min,
            "Q_hold_max_V": hold_max,
            "Crossings": crossings,
            "Expected_crossings": expected_crossings,
            "Sample_pass": bool(sample_pass),
            "Hold_pass": bool(hold_pass),
            "Glitch_pass": bool(glitch_pass),
            "PASS": passed
        })

    passed_cycles = sum(
        row["PASS"] for row in checks
    )

    return {
        "passed": passed_cycles == cycles,
        "passed_cycles": passed_cycles,
        "total_cycles": cycles,
        "checks": checks
    }
