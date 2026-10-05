"""Linear DIAL transfer map from tone intensities to Pauli-Z rates."""

from __future__ import annotations

import numpy as np

from .basis import pauli_eigenvalue_matrix


def transfer_matrix(
    offsets: np.ndarray,
    tones: np.ndarray,
    *,
    n: int,
    include_identity: bool = False,
) -> tuple[list[str], np.ndarray]:
    r"""
    Construct the DIAL transfer matrix ``G``.

    For fixed tone frequencies,

        K = G x,

    where ``x_k = |Omega_k|^2 / 2``.

    Each column is obtained from the configuration-dependent dispersive
    response ``1 / (omega_k - Lambda_alpha)`` and transformed into the
    Pauli-Z-string basis.
    """
    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    tones = np.asarray(
        tones,
        dtype=float,
    )

    labels, eig = pauli_eigenvalue_matrix(
        n,
        include_identity=include_identity,
    )

    G = np.empty(
        (len(labels), len(tones)),
        dtype=float,
    )

    for k, omega in enumerate(tones):
        detuning = omega - offsets
        response_alpha = 1.0 / detuning

        G[:, k] = (
            eig.T @ response_alpha
        ) / float(2**n)

    return labels, G
