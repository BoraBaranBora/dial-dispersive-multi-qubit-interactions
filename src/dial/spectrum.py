"""Register-resolved mediator spectra and candidate-tone construction."""

from __future__ import annotations

import numpy as np

from .basis import z_configs


def transition_offsets_from_couplings(
    couplings: np.ndarray,
) -> np.ndarray:
    """Register-resolved mediator transition offsets."""
    couplings = np.asarray(
        couplings,
        dtype=float,
    )

    return z_configs(len(couplings)) @ couplings


def candidate_tones(
    offsets: np.ndarray,
    *,
    num_grid: int = 301,
    span: float = 1.35,
    min_detuning: float = 0.04,
) -> np.ndarray:
    """
    Construct a uniform candidate-tone grid with transition exclusion zones.

    ``min_detuning`` follows the historical normalized convention and is
    multiplied by the spectral radius.
    """
    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    spectral_radius = float(
        np.max(np.abs(offsets))
    )

    grid = np.linspace(
        -span * spectral_radius,
        +span * spectral_radius,
        num_grid,
    )

    keep = np.ones_like(
        grid,
        dtype=bool,
    )

    for transition in offsets:
        keep &= (
            np.abs(grid - transition)
            >= min_detuning * spectral_radius
        )

    return grid[keep]


def mean_transition_spacing(
    offsets: np.ndarray,
) -> float:
    """Mean spacing defined from the full transition bandwidth."""
    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    if len(offsets) < 2:
        raise ValueError(
            "Need at least two transition frequencies."
        )

    return float(
        (
            np.max(offsets)
            - np.min(offsets)
        )
        / (len(offsets) - 1)
    )


def minimum_transition_spacing(
    offsets: np.ndarray,
) -> float:
    """Minimum nearest-neighbor spacing of the ordered spectrum."""
    offsets = np.sort(
        np.asarray(
            offsets,
            dtype=float,
        )
    )

    if len(offsets) < 2:
        raise ValueError(
            "Need at least two transition frequencies."
        )

    return float(
        np.min(
            np.diff(offsets)
        )
    )
