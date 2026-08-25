"""Computational-basis and Pauli-Z bookkeeping."""

from __future__ import annotations

import numpy as np


def bit_configs(n: int) -> np.ndarray:
    """Computational-basis bit strings in integer order."""
    return np.array(
        [
            [
                (a >> (n - 1 - i)) & 1
                for i in range(n)
            ]
            for a in range(2**n)
        ],
        dtype=int,
    )


def z_configs(n: int) -> np.ndarray:
    """Pauli-Z eigenvalues associated with computational basis states."""
    return (-1.0) ** bit_configs(n)


def pauli_masks_and_labels(
    n: int,
    *,
    include_identity: bool = False,
) -> tuple[np.ndarray, list[str]]:
    masks: list[int] = []
    labels: list[str] = []

    start = 0 if include_identity else 1

    for mask in range(start, 2**n):
        inds = [
            i + 1
            for i in range(n)
            if (mask >> (n - 1 - i)) & 1
        ]

        label = (
            "I"
            if not inds
            else "".join(f"Z{i}" for i in inds)
        )

        masks.append(mask)
        labels.append(label)

    return np.asarray(masks, dtype=int), labels


def pauli_eigenvalue_matrix(
    n: int,
    *,
    include_identity: bool = False,
) -> tuple[list[str], np.ndarray]:
    """
    Matrix of Pauli-Z-string eigenvalues over register configurations.

    Rows correspond to computational-basis configurations and columns
    to Pauli-Z strings.
    """
    masks, labels = pauli_masks_and_labels(
        n,
        include_identity=include_identity,
    )

    bits = bit_configs(n)

    eig = np.ones(
        (2**n, len(labels)),
        dtype=float,
    )

    for j, mask in enumerate(masks):
        values = np.ones(2**n, dtype=float)

        for i in range(n):
            if (mask >> (n - 1 - i)) & 1:
                values *= (-1.0) ** bits[:, i]

        eig[:, j] = values

    return labels, eig


def pauli_order(label: str) -> int:
    """Number of nonidentity Z factors in a Pauli-Z-string label."""
    if label == "I":
        return 0

    return label.count("Z")
