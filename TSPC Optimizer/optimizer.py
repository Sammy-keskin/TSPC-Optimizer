
# =====================================================
# TSPC OPTIMIZER
# optimizer.py
# =====================================================

import numpy as np

from scipy.optimize import (
    differential_evolution,
    NonlinearConstraint
)

from config import *
from layout import *
from timing import *


MIN_WIDTH = 4
MAX_WIDTH = 2000

BOUNDS = [
    (MIN_WIDTH, MAX_WIDTH)
    for _ in TRANSISTORS
]


def vector_to_widths(x):

    return {
        name: float(value)
        for name, value in zip(TRANSISTORS, x)
    }


def area_function(x):

    widths = vector_to_widths(x)

    return estimate_layout_area(widths)


def objective(x):

    widths = vector_to_widths(x)

    timing = estimate_timing(widths)

    return -timing["F_EST_MHZ"]


def run_optimization(seed=DE_SEED):

    build_delay_tables()

    history = []
    generation = 0

    area_constraint = NonlinearConstraint(
        area_function,
        0,
        MAX_LAYOUT_AREA
    )

    def callback(xk, convergence):

        nonlocal generation

        generation += 1

        widths = vector_to_widths(xk)
        timing = estimate_timing(widths)
        area = estimate_layout_area(widths)

        history.append({
            "Generation": generation,
            "Frequency_MHz": timing["F_EST_MHZ"],
            "Area_um2": area,
            "Convergence": convergence
        })

        if generation == 1 or generation % 10 == 0:

            print(
                f"Gen {generation:4d} | "
                f"Freq={timing['F_EST_MHZ']:9.3f} MHz | "
                f"Area={area:10.2f} um^2",
                flush=True
            )

        return False

    print("\nStarting Differential Evolution...")
    print(f"Seed: {seed}")
    print(f"Population: {DE_POPSIZE * len(TRANSISTORS)}")
    print(f"Max generations: {DE_MAXITER}")
    print("-" * 65)

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        constraints=(area_constraint,),
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        tol=DE_TOL,
        seed=seed,
        integrality=[True] * len(TRANSISTORS),
        callback=callback,
        polish=False,
        workers=1
    )

    widths = vector_to_widths(result.x)

    return result, widths, history
