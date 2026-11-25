
# =====================================================
# TSPC OPTIMIZER V2
# tspc.py
# =====================================================

import numpy as np

from config import *
from mosfet import *
from capacitance import *
from layout import *


NODE_INDEX = {
    name: i for i, name in enumerate(NODES)
}


# MOSFET connections:
# Drain, Gate, Source, Bulk

CONNECTIONS = {
    "P1": ("Y1", "D",   "VDD", "VDD"),
    "P2": ("X1", "CLK", "Y1",  "VDD"),
    "P3": ("X2", "CLK", "VDD", "VDD"),
    "P4": ("X3", "X2",  "VDD", "VDD"),
    "P5": ("Q",  "X3",  "VDD", "VDD"),

    "N1": ("X1", "D",   "GND", "GND"),
    "N2": ("X2", "X1",  "Y2",  "GND"),
    "N3": ("Y2", "CLK", "GND", "GND"),
    "N4": ("X3", "CLK", "Y3",  "GND"),
    "N5": ("Y3", "X2",  "GND", "GND"),
    "N6": ("Q",  "X3",  "GND", "GND")
}


def pulse(t, period, delay=0.0):
    """
    Periodic trapezoidal waveform.

    Returns:
        voltage [V]
        voltage derivative [V/s]
    """

    if t < delay:
        return 0.0, 0.0

    phase = (t - delay) % period

    tr = RISE_FALL_RATIO * period
    tf = RISE_FALL_RATIO * period

    fall_start = CLOCK_DUTY_CYCLE * period

    if phase < tr:
        return VDD * phase / tr, VDD / tr

    if phase < fall_start:
        return VDD, 0.0

    if phase < fall_start + tf:

        voltage = VDD * (
            1 - (phase - fall_start) / tf
        )

        return voltage, -VDD / tf

    return 0.0, 0.0


def input_signals(t, frequency):

    period = 1.0 / frequency

    clk, dclk = pulse(
        t, period
    )

    data, ddata = pulse(
        t,
        period / DATA_FREQ_RATIO,
        delay=DATA_DELAY_CYCLES * period
    )

    return {
        "CLK": clk,
        "D": data,
        "dCLK": dclk,
        "dD": ddata
    }


class TSPCCircuit:

    def __init__(self, widths, smoothing_v=0.01):

        self.widths = widths.copy()
        if smoothing_v <= 0:
            raise ValueError("smoothing_v must be positive")
        self.smoothing_v = smoothing_v

        self.geometry = {
            name: transistor_geometry(width)
            for name, width in widths.items()
        }

        self.size = len(NODES)


    def voltages(self, t, state, frequency, signals=None):

        if signals is None:
            signals = input_signals(t, frequency)

        voltages = {
            name: state[i]
            for i, name in enumerate(NODES)
        }

        voltages.update({
            "VDD": VDD,
            "GND": 0.0,
            "CLK": signals["CLK"],
            "D": signals["D"]
        })

        derivatives = {
            "VDD": 0.0,
            "GND": 0.0,
            "CLK": signals["dCLK"],
            "D": signals["dD"]
        }

        return voltages, derivatives


    def transistor_current(self, name, v):
        """
        Signed conventional current from
        the named drain to the named source.

        Includes source-drain reversal.
        """

        d, g, s, b = CONNECTIONS[name]

        vd = v[d]
        vg = v[g]
        vs = v[s]
        vb = v[b]

        width = self.widths[name]

        if name.startswith("N"):

            if vd >= vs:

                return nmos_current(
                    width,
                    vg - vs,
                    vd - vs,
                    vs - vb
                )

            return -nmos_current(
                width,
                vg - vd,
                vs - vd,
                vd - vb
            )

        else:

            if vs >= vd:

                return -pmos_current(
                    width,
                    vs - vg,
                    vs - vd,
                    vb - vs
                )

            return pmos_current(
                width,
                vd - vg,
                vd - vs,
                vb - vd
            )


    def current_vector(self, v):

        currents = np.zeros(self.size)

        for name, (d, g, s, b) in CONNECTIONS.items():

            # MOSFET channel current

            current = self.transistor_current(
                name, v
            )

            if d in NODE_INDEX:
                currents[NODE_INDEX[d]] -= current

            if s in NODE_INDEX:
                currents[NODE_INDEX[s]] += current

            # Junction diode currents

            is_pmos = name.startswith("P")

            model = PMOS if is_pmos else NMOS

            geo = self.geometry[name]

            junctions = [
                (d, geo["AD"]),
                (s, geo["AS"])
            ]

            for node, area in junctions:

                if node not in NODE_INDEX:
                    continue

                if is_pmos:

                    forward_voltage = (
                        v[node] - v[b]
                    )

                    diode_current = (
                        junction_diode_current(
                            model,
                            area,
                            forward_voltage
                        )
                    )

                    currents[NODE_INDEX[node]] -= (
                        diode_current
                    )

                else:

                    forward_voltage = (
                        v[b] - v[node]
                    )

                    diode_current = (
                        junction_diode_current(
                            model,
                            area,
                            forward_voltage
                        )
                    )

                    currents[NODE_INDEX[node]] += (
                        diode_current
                    )

        return currents


    def stamp_capacitor(
        self, matrix, coupling,
        a, b, cap, derivatives
    ):
        """
        Add a capacitor between two nodes
        to the nodal capacitance matrix.
        """

        ia = NODE_INDEX.get(a)
        ib = NODE_INDEX.get(b)

        if ia is not None:

            matrix[ia, ia] += cap

            if ib is not None:
                matrix[ia, ib] -= cap

            else:
                coupling[ia] += (
                    cap * derivatives.get(b, 0.0)
                )

        if ib is not None:

            matrix[ib, ib] += cap

            if ia is not None:
                matrix[ib, ia] -= cap

            else:
                coupling[ib] += (
                    cap * derivatives.get(a, 0.0)
                )


    def capacitance_matrix(self, v, derivatives):

        matrix = np.zeros(
            (self.size, self.size)
        )

        coupling = np.zeros(self.size)

        # External output load
        self.stamp_capacitor(
            matrix, coupling,
            "Q", "GND", CL, derivatives
        )

        for name, (d, g, s, b) in CONNECTIONS.items():

            is_pmos = name.startswith("P")

            model = PMOS if is_pmos else NMOS
            length = LP if is_pmos else LN

            width = self.widths[name]
            geo = self.geometry[name]

            # Intrinsic gate capacitance
            cint = intrinsic_gate_capacitance(
                model, width, length
            )

            # Overlap capacitances
            cgd_ov, cgs_ov = overlap_capacitance(
                model, width
            )

            # Approximate intrinsic capacitance
            # partition by operating region

            if is_pmos:

                source = max(v[s], v[d])
                drain = min(v[s], v[d])

                vgs = source - v[g]
                vds = source - drain
                vbody = v[b] - source

                reversed_sd = v[d] > v[s]

            else:

                source = min(v[s], v[d])
                drain = max(v[s], v[d])

                vgs = v[g] - source
                vds = drain - source
                vbody = source - v[b]

                reversed_sd = v[d] < v[s]

            vt = threshold_voltage(
                model, vbody
            )

            vov = vgs - vt

            # Smooth interpolation of the original region approximations.
            # This is a differential-capacitance approximation, not a charge model.
            w = self.smoothing_v
            on = 0.5 * (1.0 + np.tanh(vov / w))
            sat = 0.5 * (1.0 + np.tanh((vds - vov) / w))
            # Fade source/drain asymmetry to zero at Vds = 0.
            asym = sat * np.tanh(vds / w)
            cgs_int = on * cint * (0.5 - sat / 6 + asym / 3)
            cgd_int = on * cint * (0.5 - sat / 6 - asym / 3)
            cgb_int = (1.0 - on) * cint
            if reversed_sd:
                cgs_int, cgd_int = cgd_int, cgs_int

            self.stamp_capacitor(
                matrix, coupling,
                g, s,
                cgs_ov + cgs_int,
                derivatives
            )

            self.stamp_capacitor(
                matrix, coupling,
                g, d,
                cgd_ov + cgd_int,
                derivatives
            )

            self.stamp_capacitor(
                matrix, coupling,
                g, b, cgb_int,
                derivatives
            )

            # Voltage-dependent junction capacitance

            if is_pmos:
                vrd = v[b] - v[d]
                vrs = v[b] - v[s]

            else:
                vrd = v[d] - v[b]
                vrs = v[s] - v[b]

            cbd = junction_capacitance(
                model,
                geo["AD"],
                geo["PD"],
                vrd
            )

            cbs = junction_capacitance(
                model,
                geo["AS"],
                geo["PS"],
                vrs
            )

            self.stamp_capacitor(
                matrix, coupling,
                d, b, cbd, derivatives
            )

            self.stamp_capacitor(
                matrix, coupling,
                s, b, cbs, derivatives
            )

        return matrix, coupling


    def rhs(self, t, state, frequency, signals=None):

        v, derivatives = self.voltages(
            t, state, frequency, signals
        )

        currents = self.current_vector(v)

        matrix, coupling = (
            self.capacitance_matrix(
                v, derivatives
            )
        )

        dvdt = np.linalg.solve(
            matrix,
            currents + coupling
        )

        return dvdt
