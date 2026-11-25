
# =====================================================
# MOSFET LEVEL-1 PHYSICAL MODEL
# mosfet.py
# =====================================================

import numpy as np

from config import NMOS, PMOS, EPS_SIO2, LN, LP


def oxide_capacitance(model):
    """Oxide capacitance per unit area [F/m^2]."""
    return EPS_SIO2 / model["TOX"]


def effective_length(model, length):
    """Effective channel length [m]."""
    return length - 2 * model["LD"]


def process_transconductance(model):
    """Process transconductance parameter [A/V^2]."""
    mobility = model["UO"] * 1e-4
    return mobility * oxide_capacitance(model)


def threshold_voltage(model, v_body):
    """Threshold magnitude including body effect [V]."""

    v_body = max(v_body, 0.0)

    vt0 = abs(model["VTO"])
    gamma = model["GAMMA"]
    phi = model["PHI"]

    return vt0 + gamma * (
        np.sqrt(phi + v_body) - np.sqrt(phi)
    )


def drain_current(model, width_um, length, v_gate,
                  v_drain, v_body=0.0):
    """
    MOSFET drain current magnitude [A].

    NMOS inputs: VGS, VDS, VSB
    PMOS inputs: VSG, VSD, VBS

    width_um: transistor width in micrometers
    length: channel length in meters
    """

    width = width_um * 1e-6

    if width <= 0:
        raise ValueError("Width must be positive.")

    leff = effective_length(model, length)

    if leff <= 0:
        raise ValueError("Invalid channel length.")

    vt = threshold_voltage(model, v_body)

    vov = v_gate - vt
    vds = max(v_drain, 0.0)

    beta = process_transconductance(model) * width / leff
    lam = model["LAMBDA"]

    # Cutoff region
    if vov <= 0:
        return 0.0

    # Linear / triode region
    if vds < vov:
        current = beta * (
            vov * vds - 0.5 * vds**2
        )

    # Saturation region
    else:
        current = 0.5 * beta * vov**2

    # Channel length modulation
    current *= 1 + lam * vds

    return current


def nmos_current(width_um, vgs, vds, vsb=0.0):
    return drain_current(
        NMOS, width_um, LN, vgs, vds, vsb
    )


def pmos_current(width_um, vsg, vsd, vbs=0.0):
    return drain_current(
        PMOS, width_um, LP, vsg, vsd, vbs
    )


def operating_region(model, v_gate, v_drain, v_body=0):
    vt = threshold_voltage(model, v_body)
    vov = v_gate - vt

    if vov <= 0:
        return "Cutoff"

    if v_drain < vov:
        return "Linear"

    return "Saturation"

def junction_diode_current(
    model,
    area_um2,
    forward_voltage
):
    """
    Junction diode current.

    Positive current flows from
    anode to cathode.
    """

    from config import (
        THERMAL_VOLTAGE,
        DIODE_IDEALITY
    )

    import math

    area_m2 = area_um2 * 1e-12

    saturation_current = (
        model["JS"] * area_m2
    )

    exponent = (
        forward_voltage
        / (
            DIODE_IDEALITY
            * THERMAL_VOLTAGE
        )
    )

    # Numerical overflow protection
    exponent = min(exponent, 60.0)

    return (
        saturation_current
        * math.expm1(exponent)
    )