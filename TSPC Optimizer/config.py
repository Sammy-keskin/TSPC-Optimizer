
# =====================================================
# TSPC OPTIMIZER
# EEE411 - Digital Integrated Circuits - HW2
# config.py
# =====================================================

# HW2 specifications
VDD = 3.0
CL = 5e-12

LAMBDA_LAYOUT = 1e-6
LN = 2e-6
LP = 2e-6

MAX_LAYOUT_AREA = 100_000  # um^2

FREQ_STEP = 10e6
DATA_FREQ_RATIO = 0.2
DATA_DELAY_CYCLES = 1
RISE_FALL_RATIO = 0.01
CLOCK_DUTY_CYCLE = 0.5
SIMULATION_CYCLES = 10

# Physical constants
EPS0 = 8.8541878128e-12
EPS_SIO2 = 3.9 * EPS0

# NMOS model
NMOS = {
    "LEVEL": 1,
    "TOX": 9e-9,
    "LD": 0.08e-6,
    "GAMMA": 0.45,
    "NSUB": 9e14,
    "UO": 350,
    "VTO": 0.7,
    "CGDO": 0.4e-9,
    "CGSO": 0.4e-9,
    "CJ": 0.56e-3,
    "CJSW": 0.35e-11,
    "PB": 0.9,
    "MJ": 0.45,
    "MJSW": 0.2,
    "PHI": 0.9,
    "LAMBDA": 0.1,
    "JS": 1e-8
}

# PMOS model
PMOS = {
    "LEVEL": 1,
    "TOX": 9e-9,
    "LD": 0.09e-6,
    "GAMMA": 0.4,
    "NSUB": 5e14,
    "UO": 100,
    "VTO": -0.8,
    "CGDO": 0.3e-9,
    "CGSO": 0.3e-9,
    "CJ": 0.94e-3,
    "CJSW": 0.32e-11,
    "PB": 0.9,
    "MJ": 0.5,
    "MJSW": 0.3,
    "PHI": 0.8,
    "LAMBDA": 0.2,
    "JS": 0.5e-8
}

# Optimization variables
TRANSISTORS = [
    "P1", "P2", "P3", "P4", "P5",
    "N1", "N2", "N3", "N4", "N5", "N6"
]

# Internal circuit nodes and output
NODES = [
    "X1", "Y1",
    "X2", "Y2",
    "X3", "Y3",
    "Q"
]

# Differential Evolution settings
DE_SEED = 42
DE_POPSIZE = 20
DE_MAXITER = 300
DE_MUTATION = (0.5, 1.0)
DE_RECOMBINATION = 0.7
DE_TOL = 1e-6

# Junction diode model

TEMPERATURE = 300.0

BOLTZMANN = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19

THERMAL_VOLTAGE = (
    BOLTZMANN * TEMPERATURE
    / ELECTRON_CHARGE
)

DIODE_IDEALITY = 1.0