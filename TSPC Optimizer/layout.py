
# =====================================================
# TSPC OPTIMIZER
# layout.py
# =====================================================

import math

from config import MAX_LAYOUT_AREA


# Preliminary geometry parameters [um]
GATE_LENGTH = 2.0
DIFFUSION_EXTENSION = 5.0
POLY_EXTENSION = 2.0
DEVICE_SPACING = 6.0
MAX_FINGER_WIDTH = 80.0

# Additional area for routing, wells and contacts
ROUTING_FACTOR = 1.40


def transistor_geometry(width_um):
    """
    Approximate multi-finger MOSFET geometry.

    All dimensions are in micrometers.
    All areas are in square micrometers.
    """

    if width_um <= 0:
        raise ValueError("Width must be positive.")

    nf = math.ceil(width_um / MAX_FINGER_WIDTH)
    wf = width_um / nf

    # Shared source/drain diffusion between fingers
    diffusion_count = nf + 1

    active_length = (
        nf * GATE_LENGTH
        + diffusion_count * DIFFUSION_EXTENSION
    )

    active_area = active_length * wf

    source_regions = math.ceil(diffusion_count / 2)
    drain_regions = math.floor(diffusion_count / 2)

    region_area = wf * DIFFUSION_EXTENSION

    region_perimeter = 2 * (
        wf + DIFFUSION_EXTENSION
    )

    source_area = source_regions * region_area
    drain_area = drain_regions * region_area

    source_perimeter = source_regions * region_perimeter
    drain_perimeter = drain_regions * region_perimeter

    cell_width = active_length + DEVICE_SPACING

    cell_height = (
        wf
        + 2 * POLY_EXTENSION
        + DEVICE_SPACING
    )

    cell_area = cell_width * cell_height

    return {
        "W": width_um,
        "L": GATE_LENGTH,
        "NF": nf,
        "WF": wf,
        "AD": drain_area,
        "AS": source_area,
        "PD": drain_perimeter,
        "PS": source_perimeter,
        "active_area": active_area,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "cell_area": cell_area
    }


def estimate_layout_area(widths):
    """Preliminary total layout area estimate [um^2]."""

    total = sum(
        transistor_geometry(w)["cell_area"]
        for w in widths.values()
    )

    return total * ROUTING_FACTOR


def area_utilization(widths):
    area = estimate_layout_area(widths)

    return 100 * area / MAX_LAYOUT_AREA


def satisfies_estimated_area_limit(widths):
    return estimate_layout_area(widths) <= MAX_LAYOUT_AREA
