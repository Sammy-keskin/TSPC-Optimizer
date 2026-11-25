
# =====================================================
# TSPC OPTIMIZER V2
# test_v2.py
# =====================================================

import numpy as np

from config import *
from tspc import *


# Best design from V1 analytical optimization

widths = {
    "P1": 826,
    "P2": 613,
    "P3": 1361,
    "P4": 759,
    "P5": 412,

    "N1": 88,
    "N2": 740,
    "N3": 1599,
    "N4": 630,
    "N5": 448,
    "N6": 279
}

circuit = TSPCCircuit(widths)

frequency = 990e6

print("=" * 65)
print("TSPC V2 - PHYSICAL CIRCUIT TEST")
print("=" * 65)

print("\nTRANSISTOR CONNECTIONS")

for name, connection in CONNECTIONS.items():

    print(f"{name}: {connection}")


print("\nINPUT SIGNAL TEST")

period = 1.0 / frequency

for cycle in [0, 0.005, 0.25, 0.5, 0.75, 1.25, 2.25]:

    t = cycle * period

    signals = input_signals(t, frequency)

    print(
        f"t={t * 1e9:8.4f} ns | "
        f"CLK={signals['CLK']:.4f} V | "
        f"D={signals['D']:.4f} V"
    )


print("\nINITIAL NODE VOLTAGES")

# Arbitrary test state, not a steady-state solution

state = np.array([
    3.0,   # X1
    3.0,   # Y1
    3.0,   # X2
    1.5,   # Y2
    0.0,   # X3
    0.0,   # Y3
    3.0    # Q
])

for name, value in zip(NODES, state):

    print(f"{name}: {value:.4f} V")


t = 0.25 * period

v, derivatives = circuit.voltages(
    t, state, frequency
)


print("\nTRANSISTOR CURRENTS")

for name in TRANSISTORS:

    current = circuit.transistor_current(
        name, v
    )

    print(
        f"{name}: {current * 1e3:10.4f} mA"
    )


print("\nCAPACITANCE MATRIX [pF]")

matrix, coupling = circuit.capacitance_matrix(
    v, derivatives
)

np.set_printoptions(
    precision=4,
    suppress=True,
    linewidth=120
)

print(matrix * 1e12)


print("\nMATRIX VALIDATION")

symmetric = np.allclose(
    matrix,
    matrix.T,
    atol=1e-24,
    rtol=1e-10
)

eigenvalues = np.linalg.eigvalsh(matrix)

positive_definite = np.all(
    eigenvalues > 0
)

print(f"Symmetric: {symmetric}")

print(
    f"Positive definite: "
    f"{positive_definite}"
)

print(
    f"Condition number: "
    f"{np.linalg.cond(matrix):.4f}"
)

print(
    f"Eigenvalues [pF]: "
    f"{eigenvalues * 1e12}"
)


print("\nNODE VOLTAGE DERIVATIVES")

dvdt = circuit.rhs(
    t, state, frequency
)

for name, value in zip(NODES, dvdt):

    print(
        f"d{name}/dt: "
        f"{value * 1e-9:10.4f} V/ns"
    )


print("\nTEST COMPLETED")
