
# =====================================================
# TSPC OPTIMIZER
# capacitance.py
# =====================================================

from config import NMOS, PMOS, LN, LP, CL
from mosfet import oxide_capacitance, effective_length


def intrinsic_gate_capacitance(model, width_um, length):
    """Approximate full-channel oxide capacitance [F]."""

    width = width_um * 1e-6
    leff = effective_length(model, length)

    return oxide_capacitance(model) * width * leff


def overlap_capacitance(model, width_um):
    """Gate-drain and gate-source overlap capacitances [F]."""

    width = width_um * 1e-6

    cgd = model["CGDO"] * width
    cgs = model["CGSO"] * width

    return cgd, cgs


def total_gate_capacitance(model, width_um, length):
    """Approximate gate load for analytical optimization."""

    cox = intrinsic_gate_capacitance(
        model, width_um, length
    )

    cgd, cgs = overlap_capacitance(model, width_um)

    return cox + cgd + cgs


def junction_capacitance(model, area_um2,
                         perimeter_um, reverse_bias=0.0):
    """
    Reverse-biased junction capacitance [F].

    area_um2: diffusion area [um^2]
    perimeter_um: diffusion perimeter [um]
    reverse_bias: junction reverse voltage [V]
    """

    area = area_um2 * 1e-12
    perimeter = perimeter_um * 1e-6

    vr = max(reverse_bias, 0.0)

    pb = model["PB"]
    mj = model["MJ"]
    mjsw = model["MJSW"]

    c_bottom = (
        model["CJ"] * area
        / (1 + vr / pb)**mj
    )

    c_sidewall = (
        model["CJSW"] * perimeter
        / (1 + vr / pb)**mjsw
    )

    return c_bottom + c_sidewall


def nmos_gate_capacitance(width_um):
    return total_gate_capacitance(
        NMOS, width_um, LN
    )


def pmos_gate_capacitance(width_um):
    return total_gate_capacitance(
        PMOS, width_um, LP
    )


def gate_loads(widths):
    """
    Approximate gate loads of TSPC nodes.

    widths: dictionary of transistor widths [um]
    Returns capacitances in farads.
    """

    return {
        "X1": nmos_gate_capacitance(widths["N2"]),

        "X2": (
            pmos_gate_capacitance(widths["P4"])
            + nmos_gate_capacitance(widths["N5"])
        ),

        "X3": (
            pmos_gate_capacitance(widths["P5"])
            + nmos_gate_capacitance(widths["N6"])
        ),

        "Q": CL
    }
