"""Deterministic benchmark-ensemble sampling for DIAL."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import qmc


def offsets_from_couplings(
    A: np.ndarray,
) -> np.ndarray:
    """
    Normalized n-spin extension of the benchmark spectrum

        delta_alpha =
            sum_i (-1)^alpha_i A_i / sum_i A_i.

    Delta_Lambda is set to unity, as in the normalized spectral
    benchmark.
    """
    A = np.asarray(
        A,
        dtype=float,
    )

    if A.ndim != 1:
        raise ValueError(
            "A must be one-dimensional."
        )

    if np.any(A <= 0.0):
        raise ValueError(
            "All A_i must be positive."
        )

    n = len(A)

    states = np.asarray(
        list(
            itertools.product(
                (0, 1),
                repeat=n,
            )
        ),
        dtype=int,
    )

    signs = 1.0 - 2.0 * states

    return (
        signs @ A
        / float(np.sum(A))
    )


def sample_realizations(
    *,
    n: int,
    n_systems: int,
    seed: int,
    ratio_max: float,
    sobol_power: int,
) -> list[dict]:
    """
    Sample register realizations in logarithmic coupling-ratio
    coordinates.

    A1 = 1 and A2,...,An are sampled in [1, ratio_max], then
    ordered so that

        1 = A1 <= A2 <= ... <= An <= ratio_max.

    Only accidental spectral degeneracies are rejected, using
    the same 1e-10 tolerance as the existing three-spin benchmark.
    """
    if n < 2:
        raise ValueError(
            "n must be at least 2."
        )

    if n_systems < 1:
        raise ValueError(
            "n_systems must be positive."
        )

    if ratio_max <= 1.0:
        raise ValueError(
            "ratio_max must exceed 1."
        )

    sampler = qmc.Sobol(
        d=n - 1,
        scramble=True,
        seed=int(seed) + 1009 * int(n),
    )

    unit_points = sampler.random_base2(
        m=sobol_power
    )

    log_max = math.log10(
        ratio_max
    )

    realizations = []

    for point in unit_points:
        ratios = 10.0 ** (
            log_max
            * np.asarray(
                point,
                dtype=float,
            )
        )

        ratios.sort()

        A = np.concatenate(
            (
                np.array(
                    [1.0],
                    dtype=float,
                ),
                ratios,
            )
        )

        offsets = offsets_from_couplings(
            A
        )

        ordered = np.sort(
            offsets
        )

        min_spacing = float(
            np.min(
                np.diff(ordered)
            )
        )

        # Match the original benchmark philosophy:
        # reject accidental degeneracies only.
        if min_spacing <= 1e-10:
            continue

        realizations.append(
            {
                "n": int(n),
                "realization_index":
                    len(realizations) + 1,
                "A": A,
                "offsets": offsets,
                "minimum_transition_spacing":
                    min_spacing,
            }
        )

        if len(realizations) >= n_systems:
            break

    if len(realizations) != n_systems:
        raise RuntimeError(
            f"Generated only "
            f"{len(realizations)} "
            f"admissible realizations "
            f"for n={n}; expected "
            f"{n_systems}."
        )

    return realizations
