"""Fidelity and mediator-excitation metrics for DIAL validation."""

from __future__ import annotations

import math

import numpy as np

from . import dynamics
from .synthesis import local_intensity_caps


def mediator_z_optimized_fidelity(
    blocks: np.ndarray,
    *,
    target_label: str,
    signed_target_phase: float,
) -> float:
    """
    Process fidelity maximized analytically over a free mediator-only
    Z phase.

    The register dimension is inferred from the propagated aligned-RWA
    blocks, so the same objective applies to arbitrary register size.
    For the target interaction

        U_target = exp[-i theta sigma_z Z_S],

    the mediator sigma_z eigenvalues are -1 in |g> and +1 in |e>.
    """
    blocks = np.asarray(
        blocks,
        dtype=np.complex128,
    )

    if (
        blocks.ndim != 3
        or blocks.shape[1:] != (2, 2)
    ):
        raise ValueError(
            "Expected blocks with shape "
            "(2**n, 2, 2), got "
            f"{blocks.shape}."
        )

    n_reg = int(blocks.shape[0])

    if n_reg < 1:
        raise ValueError(
            "At least one register block is required."
        )

    n = int(round(math.log2(n_reg)))

    if 2**n != n_reg:
        raise ValueError(
            "Number of register blocks must be "
            f"a power of two, got {n_reg}."
        )

    support = [
        q
        for q in range(1, n + 1)
        if f"Z{q}" in target_label
    ]

    if not support:
        raise ValueError(
            "Could not identify a Pauli-Z support "
            f"from target label {target_label!r}."
        )

    A_g = 0.0 + 0.0j
    A_e = 0.0 + 0.0j

    for alpha in range(n_reg):
        z_target = 1

        for q in support:
            bit_position = n - q
            bit = (
                (int(alpha) >> bit_position)
                & 1
            )

            z_target *= (
                +1
                if bit == 0
                else -1
            )

        phase_g = np.exp(
            +1j
            * signed_target_phase
            * z_target
        )

        phase_e = np.exp(
            -1j
            * signed_target_phase
            * z_target
        )

        A_g += (
            np.conjugate(phase_g)
            * blocks[
                alpha,
                0,
                0,
            ]
        )

        A_e += (
            np.conjugate(phase_e)
            * blocks[
                alpha,
                1,
                1,
            ]
        )

    trace_max = (
        abs(A_g)
        + abs(A_e)
    )

    dim_total = 2 * n_reg

    return float(
        trace_max**2
        / float(dim_total**2)
    )


def final_max_ground_to_excited_flip(
    blocks: np.ndarray,
) -> float:
    return float(
        np.max(
            np.abs(
                blocks[:, 1, 0]
            ) ** 2
        )
    )


def decode_parameters(
    z: np.ndarray,
    *,
    M: int,
    offsets: np.ndarray,
    r_disp: float,
    omega_hw_max: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tones = np.asarray(
        z[:M],
        dtype=float,
    )

    fractions = np.asarray(
        z[M:],
        dtype=float,
    )

    xmax, detunings = (
        local_intensity_caps(
            tones,
            offsets,
            r_disp=r_disp,
            omega_hw_max=omega_hw_max,
        )
    )

    intensities = (
        xmax * fractions
    )

    return (
        tones,
        intensities,
        xmax,
        detunings,
    )


def ground_manifold_register_fidelity(
    blocks: np.ndarray,
    *,
    target_label: str,
    signed_target_phase: float,
) -> float:
    """
    Register process fidelity for a mediator initialized and
    projected in |g>.

    With sigma_z^(med)|g> = -|g>, the register target is

        exp(+i signed_target_phase Z_S).

    Final population transfer out of |g> is automatically penalized
    because only the |g> -> |g> block enters the overlap.
    """
    blocks = np.asarray(
        blocks,
        dtype=np.complex128,
    )

    if (
        blocks.ndim != 3
        or blocks.shape[1:] != (2, 2)
    ):
        raise ValueError(
            "Expected blocks with shape (2^n, 2, 2)."
        )

    n_reg = int(blocks.shape[0])
    n = n_reg.bit_length() - 1

    if 2**n != n_reg:
        raise ValueError(
            "Register dimension must be a power of two."
        )

    support = [
        q
        for q in range(1, n + 1)
        if f"Z{q}" in target_label
    ]

    if not support:
        raise ValueError(
            "Could not identify a Pauli-Z support "
            f"from target label {target_label!r}."
        )

    overlap = 0.0 + 0.0j

    for alpha in range(n_reg):
        z_target = 1

        for q in support:
            bit_position = n - q
            bit = (
                (int(alpha) >> bit_position)
                & 1
            )
            z_target *= (
                +1 if bit == 0 else -1
            )

        target_phase = np.exp(
            +1j
            * signed_target_phase
            * z_target
        )

        overlap += (
            np.conjugate(target_phase)
            * blocks[alpha, 0, 0]
        )

    return float(
        abs(overlap) ** 2
        / n_reg**2
    )


def final_fidelity(
    z: np.ndarray,
    *,
    offsets: np.ndarray,
    M: int,
    T_gate: float,
    n_steps: int,
    target_label: str,
    signed_target_phase: float,
    r_disp: float,
    omega_hw_max: float,
    fidelity_mode: str = "mediator_z_optimized",
) -> tuple[float, dict]:
    (
        tones,
        intensities,
        xmax,
        detunings,
    ) = decode_parameters(
        z,
        M=M,
        offsets=offsets,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
    )

    blocks = (
        dynamics.propagate_final_blocks_jit(
            offsets,
            tones,
            intensities,
            T_gate,
            n_steps,
        )
    )

    if fidelity_mode == "mediator_z_optimized":
        F = (
            mediator_z_optimized_fidelity(
                blocks,
                target_label=target_label,
                signed_target_phase=(
                    signed_target_phase
                ),
            )
        )
    elif fidelity_mode == "ground_manifold":
        F = ground_manifold_register_fidelity(
            blocks,
            target_label=target_label,
            signed_target_phase=(
                signed_target_phase
            ),
        )
    else:
        raise ValueError(
            f"Unknown fidelity_mode={fidelity_mode!r}"
        )

    final_flip = (
        final_max_ground_to_excited_flip(
            blocks
        )
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    ratios = np.divide(
        amplitudes,
        detunings,
        out=np.zeros_like(amplitudes),
        where=detunings > 0.0,
    )

    details = {
        "tones": tones,
        "intensities": intensities,
        "xmax": xmax,
        "detunings": detunings,
        "amplitudes": amplitudes,
        "dispersive_ratios": ratios,
        "total_intensity": float(
            np.sum(intensities)
        ),
        "minimum_spacing": float(
            np.min(np.diff(tones))
        ),
        "final_max_flip": float(
            final_flip
        ),
    }

    return float(F), details


def target_support(
    target_label: str,
    n: int,
) -> list[int]:
    support = [
        q
        for q in range(1, n + 1)
        if f"Z{q}" in target_label
    ]

    if not support:
        raise ValueError(
            "Could not identify Pauli-Z support "
            f"from target label {target_label!r}."
        )

    return support


def target_unitary_ground(
    *,
    n: int,
    target_label: str,
    signed_target_phase: float,
) -> np.ndarray:
    d = 2**n
    support = target_support(
        target_label,
        n,
    )

    diag = np.empty(
        d,
        dtype=np.complex128,
    )

    for alpha in range(d):
        z_target = 1

        for q in support:
            bit_position = n - q
            bit = (
                (alpha >> bit_position)
                & 1
            )

            z_target *= (
                +1 if bit == 0 else -1
            )

        # sigma_z^(med)|g> = -|g>,
        # hence U_tar^(g) = exp(+i theta Z_S).
        diag[alpha] = np.exp(
            +1j
            * signed_target_phase
            * z_target
        )

    return np.diag(diag)


def ground_manifold_fidelity(
    U_final: np.ndarray,
    *,
    n: int,
    target_label: str,
    signed_target_phase: float,
) -> float:
    d = 2**n

    if U_final.shape != (2 * d, 2 * d):
        raise ValueError(
            "Unexpected full propagator shape "
            f"{U_final.shape}; expected "
            f"{(2*d, 2*d)}."
        )

    K_g = U_final[:d, :d]

    U_target = target_unitary_ground(
        n=n,
        target_label=target_label,
        signed_target_phase=signed_target_phase,
    )

    overlap = np.trace(
        U_target.conj().T
        @ K_g
    )

    return float(
        abs(overlap) ** 2
        / d**2
    )


def maximum_mediator_excitation(
    U_t: np.ndarray,
    *,
    n: int,
) -> float:
    """
    Maximum mediator excitation for computational
    ground-manifold inputs.

    At finite misalignment the excited-state population
    is summed over every excited-manifold register state beta.
    """
    d = 2**n

    pmax = 0.0

    for U in U_t:
        # Rows: all final excited-manifold states beta.
        # Cols: initial |g,alpha>.
        excited_from_ground = U[
            d:(2 * d),
            0:d,
        ]

        p_by_input = np.sum(
            np.abs(excited_from_ground) ** 2,
            axis=0,
        )

        pmax = max(
            pmax,
            float(np.max(p_by_input)),
        )

    return pmax
