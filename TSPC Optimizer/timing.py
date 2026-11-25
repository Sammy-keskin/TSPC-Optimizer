
# =====================================================
# TSPC OPTIMIZER
# timing.py
# =====================================================

import numpy as np

from functools import lru_cache
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

from config import *
from mosfet import *
from capacitance import *
from layout import *


def single_delay_coefficient(model, length, target=1.5):
    """
    Delay coefficient [ns*um/pF].
    target: remaining voltage during discharge
    or remaining voltage to VDD during charging.
    """

    def integrand(v):
        current = drain_current(
            model, 1.0, length, VDD, v
        )
        return 1.0 / current

    return 1e-3 * quad(
        integrand, target, VDD
    )[0]


def series_delay_coefficient(model, length, ratio):
    """
    Quasi-static series MOSFET delay coefficient.

    ratio = rail-side width / output-side width

    PMOS uses complementary voltage coordinates.
    """

    def series_current(x):

        def current_difference(y):

            i_output = drain_current(
                model, 1.0, length,
                VDD - y, x - y, y
            )

            i_rail = drain_current(
                model, ratio, length,
                VDD, y, 0.0
            )

            return i_output - i_rail

        y = brentq(
            current_difference, 0.0, x
        )

        return drain_current(
            model, 1.0, length,
            VDD - y, x - y, y
        )

    return 1e-3 * quad(
        lambda x: 1.0 / series_current(x),
        1.5, VDD
    )[0]


@lru_cache(maxsize=1)
def build_delay_tables():

    print("Building analytical delay tables...")

    ratios = np.geomspace(
        1 / 500, 500, 250
    )

    kn_values = [
        series_delay_coefficient(NMOS, LN, r)
        for r in ratios
    ]

    kp_values = [
        series_delay_coefficient(PMOS, LP, r)
        for r in ratios
    ]

    kn_interp = PchipInterpolator(
        np.log(ratios),
        np.log(kn_values)
    )

    kp_interp = PchipInterpolator(
        np.log(ratios),
        np.log(kp_values)
    )

    print("Delay tables ready.")

    return {
        "KN": single_delay_coefficient(NMOS, LN),
        "KP": single_delay_coefficient(PMOS, LP),

        "KP90": single_delay_coefficient(
            PMOS, LP, target=0.3
        ),

        "KN90": single_delay_coefficient(
            NMOS, LN, target=0.3
        ),

        "KN_SERIES": kn_interp,
        "KP_SERIES": kp_interp
    }


def node_capacitances(widths):
    """
    Approximate node capacitances [pF].

    Junction capacitances evaluated at
    representative reverse bias of 1.5 V.
    """

    caps = gate_loads(widths)

    caps["Y1"] = 0.0
    caps["Y2"] = 0.0
    caps["Y3"] = 0.0

    # transistor: drain node, source node
    connections = {
        "P1": ("Y1", "VDD"),
        "P2": ("X1", "Y1"),
        "P3": ("X2", "VDD"),
        "P4": ("X3", "VDD"),
        "P5": ("Q", "VDD"),

        "N1": ("X1", "GND"),
        "N2": ("X2", "Y2"),
        "N3": ("Y2", "GND"),
        "N4": ("X3", "Y3"),
        "N5": ("Y3", "GND"),
        "N6": ("Q", "GND")
    }

    for name, (drain, source) in connections.items():

        model = PMOS if name.startswith("P") else NMOS
        g = transistor_geometry(widths[name])

        if drain in caps:

            caps[drain] += junction_capacitance(
                model,
                g["AD"],
                g["PD"],
                reverse_bias=1.5
            )

        if source in caps:

            caps[source] += junction_capacitance(
                model,
                g["AS"],
                g["PS"],
                reverse_bias=1.5
            )

    return {
        node: cap * 1e12
        for node, cap in caps.items()
    }


def estimate_timing(w):

    table = build_delay_tables()
    c = node_capacitances(w)

    kn = table["KN"]
    kp = table["KP"]

    rp = w["P1"] / w["P2"]
    rn2 = w["N3"] / w["N2"]
    rn3 = w["N5"] / w["N4"]

    kp_series = np.exp(
        table["KP_SERIES"](np.log(rp))
    )

    kn_series2 = np.exp(
        table["KN_SERIES"](np.log(rn2))
    )

    kn_series3 = np.exp(
        table["KN_SERIES"](np.log(rn3))
    )

    # Stage 1
    tx1_lh = kp_series * c["X1"] / w["P2"]
    tx1_hl = kn * c["X1"] / w["N1"]

    # Stage 2
    tx2_precharge = (
        table["KP90"] * c["X2"] / w["P3"]
    )

    tx2_hl = (
        kn_series2 * c["X2"] / w["N2"]
    )

    # Stage 3
    tx3_lh = kp * c["X3"] / w["P4"]

    tx3_hl = (
        kn_series3 * c["X3"] / w["N4"]
    )

    # Stage 4
    tq_lh = kp * c["Q"] / w["P5"]
    tq_hl = kn * c["Q"] / w["N6"]

    # Approximate clock-to-Q paths
    tcq_hl = tx2_hl + tx3_lh + tq_hl
    tcq_lh = tx3_hl + tq_lh

    # Approximate phase budgets
    t_low = max(
        tx1_lh,
        tx1_hl,
        tx2_precharge
    )

    t_high = max(tcq_hl, tcq_lh)

    # Continuous analytical frequency score [MHz]
    f_est = 500.0 / max(t_low, t_high)

    # HW2 frequency step
    f_step = 10 * np.floor(f_est / 10)

    return {
        "X1_LH": tx1_lh,
        "X1_HL": tx1_hl,
        "X2_PRECHARGE": tx2_precharge,
        "X2_HL": tx2_hl,
        "X3_LH": tx3_lh,
        "X3_HL": tx3_hl,
        "Q_LH": tq_lh,
        "Q_HL": tq_hl,
        "CLK_Q_LH": tcq_lh,
        "CLK_Q_HL": tcq_hl,
        "T_LOW": t_low,
        "T_HIGH": t_high,
        "F_EST_MHZ": f_est,
        "F_STEP_MHZ": f_step,
        "CAPACITANCES": c
    }
