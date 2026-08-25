from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit

from . import full_dynamics as stage04

N_REG = stage04.DIM_REG
DIM_TOTAL = stage04.DIM_TOTAL


@njit(cache=True)
def _derivative(
    U,
    t,
    offsets,
    tones,
    amplitudes,
):
    """
    Eight aligned 2x2 mediator blocks.

    U shape: (8, 2, 2).
    """
    n_reg = offsets.shape[0]
    n_tones = tones.shape[0]

    dU = np.empty_like(U)

    for alpha in range(n_reg):
        c = 0.0 + 0.0j

        for k in range(n_tones):
            phase = -(
                tones[k]
                - offsets[alpha]
            ) * t

            c += (
                0.5
                * amplitudes[k]
                * complex(
                    math.cos(phase),
                    math.sin(phase),
                )
            )

        cc = np.conjugate(c)

        dU[alpha, 0, 0] = (
            -1j
            * c
            * U[alpha, 1, 0]
        )
        dU[alpha, 0, 1] = (
            -1j
            * c
            * U[alpha, 1, 1]
        )

        dU[alpha, 1, 0] = (
            -1j
            * cc
            * U[alpha, 0, 0]
        )
        dU[alpha, 1, 1] = (
            -1j
            * cc
            * U[alpha, 0, 1]
        )

    return dU


@njit(cache=True)
def propagate_final_blocks_jit(
    offsets,
    tones,
    intensities,
    T_gate,
    n_steps,
):
    """
    Final-only aligned-RWA propagation using the same RK4
    equations as Stage 04.

    Returns shape (8, 2, 2).
    """
    amplitudes = np.sqrt(
        2.0
        * np.maximum(
            intensities,
            0.0,
        )
    )

    U = np.zeros(
        (offsets.shape[0], 2, 2),
        dtype=np.complex128,
    )

    for alpha in range(
        offsets.shape[0]
    ):
        U[alpha, 0, 0] = 1.0
        U[alpha, 1, 1] = 1.0

    dt = T_gate / n_steps
    t = 0.0

    for _ in range(n_steps):
        k1 = _derivative(
            U,
            t,
            offsets,
            tones,
            amplitudes,
        )

        k2 = _derivative(
            U + 0.5 * dt * k1,
            t + 0.5 * dt,
            offsets,
            tones,
            amplitudes,
        )

        k3 = _derivative(
            U + 0.5 * dt * k2,
            t + 0.5 * dt,
            offsets,
            tones,
            amplitudes,
        )

        k4 = _derivative(
            U + dt * k3,
            t + dt,
            offsets,
            tones,
            amplitudes,
        )

        U = (
            U
            + (dt / 6.0)
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        t += dt

    return U


@njit(cache=True)
def propagate_final_blocks_with_transient_flip_jit(
    offsets,
    tones,
    intensities,
    T_gate,
    n_steps,
):
    """
    Aligned-RWA RK4 propagation with a transient mediator-excitation
    diagnostic.

    Returns
    -------
    U_final : ndarray, shape (n_reg, 2, 2)
        Final mediator propagator blocks.

    max_transient_flip : float
        Maximum sampled ground-to-excited mediator transition
        probability over all register configurations and accepted
        integration times,

            max_{t_m, alpha} |U_alpha(t_m)[1, 0]|^2.

        Here t_m denotes the RK4 integration grid.
    """
    amplitudes = np.sqrt(
        2.0
        * np.maximum(
            intensities,
            0.0,
        )
    )

    U = np.zeros(
        (offsets.shape[0], 2, 2),
        dtype=np.complex128,
    )

    for alpha in range(offsets.shape[0]):
        U[alpha, 0, 0] = 1.0
        U[alpha, 1, 1] = 1.0

    dt = T_gate / n_steps
    t = 0.0

    max_transient_flip = 0.0

    for _ in range(n_steps):
        k1 = _derivative(
            U,
            t,
            offsets,
            tones,
            amplitudes,
        )

        k2 = _derivative(
            U + 0.5 * dt * k1,
            t + 0.5 * dt,
            offsets,
            tones,
            amplitudes,
        )

        k3 = _derivative(
            U + 0.5 * dt * k2,
            t + 0.5 * dt,
            offsets,
            tones,
            amplitudes,
        )

        k4 = _derivative(
            U + dt * k3,
            t + dt,
            offsets,
            tones,
            amplitudes,
        )

        U = (
            U
            + (dt / 6.0)
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        t += dt

        for alpha in range(offsets.shape[0]):
            flip = (
                U[alpha, 1, 0].real ** 2
                + U[alpha, 1, 0].imag ** 2
            )

            if flip > max_transient_flip:
                max_transient_flip = flip

    return U, max_transient_flip


def blocks_to_full_unitary(
    blocks: np.ndarray,
) -> np.ndarray:
    U = np.zeros(
        (DIM_TOTAL, DIM_TOTAL),
        dtype=np.complex128,
    )

    for alpha in range(N_REG):
        g = alpha
        e = N_REG + alpha

        U[g, g] = blocks[alpha, 0, 0]
        U[g, e] = blocks[alpha, 0, 1]
        U[e, g] = blocks[alpha, 1, 0]
        U[e, e] = blocks[alpha, 1, 1]

    return U


def choose_n_steps(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    intensities: np.ndarray,
    T_gate: float,
    steps_per_period: int,
    max_steps: int,
) -> int:
    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    nus = (
        tones[None, :]
        - offsets[:, None]
    )

    max_frequency = (
        float(
            np.max(
                np.abs(nus)
            )
        )
        if nus.size
        else 1.0
    )

    max_amplitude = (
        float(
            np.max(
                np.abs(amplitudes)
            )
        )
        if amplitudes.size
        else 0.0
    )

    max_rate = max(
        max_frequency,
        max_amplitude,
        1e-12,
    )

    dt_max = (
        2.0
        * math.pi
        / (
            steps_per_period
            * max_rate
        )
    )

    n_steps = int(
        math.ceil(
            T_gate / dt_max
        )
    )

    return min(
        max(n_steps, 1),
        max_steps,
    )


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


def load_control(
    path: Path,
) -> dict:
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        "offsets":
            np.asarray(
                data["inputs"][
                    "offsets"
                ],
                dtype=np.float64,
            ),
        "tones":
            np.asarray(
                data[
                    "refined_solution"
                ]["tones"],
                dtype=np.float64,
            ),
        "intensities":
            np.asarray(
                data[
                    "refined_solution"
                ]["intensities"],
                dtype=np.float64,
            ),
        "T_gate":
            float(
                data[
                    "refined_solution"
                ]["T_effective"]
            ),
        "target_label":
            str(
                data["inputs"][
                    "target"
                ]
            ),
        "target_phase":
            float(
                data["inputs"][
                    "target_phase"
                ]
            ),
    }


def final_metrics_from_stage04(
    *,
    U_full: np.ndarray,
    target_label: str,
    signed_target_phase: float,
) -> dict:
    U_t = U_full[None, :, :]

    coeffs_t, _ = (
        stage04.extract_phase_coefficients(
            U_t
        )
    )

    U_target = (
        stage04.build_target_unitary(
            target_label,
            signed_target_phase,
        )
    )

    raw, corrected = (
        stage04.compute_fidelity_trajectories(
            U_t,
            coeffs_t,
            U_target,
        )
    )

    return {
        "raw":
            float(raw[-1]),
        "stage04_final_only_corrected":
            float(
                corrected[-1]
            ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--control-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--steps-per-period",
        nargs="+",
        type=int,
        default=[6, 8, 12, 16, 24],
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
    )

    args = parser.parse_args()

    control = load_control(
        args.control_json
    )

    offsets = control["offsets"]
    tones = control["tones"]
    intensities = (
        control["intensities"]
    )
    T_gate = control["T_gate"]

    labels, G = (
        stage04.response_matrix(
            offsets,
            tones,
            include_identity=False,
        )
    )

    target_index = {
        label: i
        for i, label
        in enumerate(labels)
    }[
        control["target_label"]
    ]

    K = G @ intensities
    target_rate = float(
        K[target_index]
    )

    signed_target_phase = (
        math.copysign(
            abs(
                control[
                    "target_phase"
                ]
            ),
            target_rate,
        )
    )

    print()
    print("[JIT aligned-RWA final-only audit]")
    print(
        f"target       = "
        f"{control['target_label']}"
    )
    print(
        f"tones        = "
        f"{len(tones)}"
    )
    print(
        f"T_gate       = "
        f"{T_gate:.10e}"
    )

    # ----------------------------------------------------------
    # Warm-up compilation. Do not include compile time in timings.
    # ----------------------------------------------------------
    warm_steps = 2

    _ = propagate_final_blocks_jit(
        offsets,
        tones,
        intensities,
        T_gate,
        warm_steps,
    )

    results = {}

    print()
    print(
        "spp   steps      seconds       "
        "F_medZopt      finalPflip"
    )

    for spp in args.steps_per_period:
        n_steps = choose_n_steps(
            offsets=offsets,
            tones=tones,
            intensities=intensities,
            T_gate=T_gate,
            steps_per_period=spp,
            max_steps=args.max_steps,
        )

        t0 = time.perf_counter()

        blocks = (
            propagate_final_blocks_jit(
                offsets,
                tones,
                intensities,
                T_gate,
                n_steps,
            )
        )

        elapsed = (
            time.perf_counter()
            - t0
        )

        F_opt = (
            mediator_z_optimized_fidelity(
                blocks,
                target_label=(
                    control[
                        "target_label"
                    ]
                ),
                signed_target_phase=(
                    signed_target_phase
                ),
            )
        )

        max_flip = (
            final_max_ground_to_excited_flip(
                blocks
            )
        )

        U_full = (
            blocks_to_full_unitary(
                blocks
            )
        )

        stage04_final = (
            final_metrics_from_stage04(
                U_full=U_full,
                target_label=(
                    control[
                        "target_label"
                    ]
                ),
                signed_target_phase=(
                    signed_target_phase
                ),
            )
        )

        results[spp] = {
            "n_steps":
                n_steps,
            "seconds":
                elapsed,
            "blocks":
                blocks.copy(),
            "F_opt":
                F_opt,
            "max_flip":
                max_flip,
            "F_raw":
                stage04_final["raw"],
            "F_stage04_final_only":
                stage04_final[
                    "stage04_final_only_corrected"
                ],
        }

        print(
            f"{spp:>3d} "
            f"{n_steps:>7d} "
            f"{elapsed:>11.6f} "
            f"{F_opt:>13.10f} "
            f"{max_flip:>12.3e}"
        )

    reference_spp = max(
        args.steps_per_period
    )

    ref = results[
        reference_spp
    ]

    print()
    print(
        f"[errors relative to "
        f"spp={reference_spp}]"
    )
    print(
        "spp   max|dU|       "
        "|dF_medZopt|   "
        "|dPflip|"
    )

    for spp in args.steps_per_period:
        row = results[spp]

        max_u_error = float(
            np.max(
                np.abs(
                    row["blocks"]
                    - ref["blocks"]
                )
            )
        )

        dF = abs(
            row["F_opt"]
            - ref["F_opt"]
        )

        dP = abs(
            row["max_flip"]
            - ref["max_flip"]
        )

        print(
            f"{spp:>3d} "
            f"{max_u_error:>12.3e} "
            f"{dF:>14.3e} "
            f"{dP:>12.3e}"
        )

    print()
    print("[24-step final-only metric audit]")
    print(
        f"analytic mediator-Z optimum = "
        f"{ref['F_opt']:.10f}"
    )
    print(
        f"Stage04 final-only correction= "
        f"{ref['F_stage04_final_only']:.10f}"
    )
    print(
        f"difference                  = "
        f"{abs(ref['F_opt'] - ref['F_stage04_final_only']):.3e}"
    )

    print()
    print(
        "Note: Stage 04's published validation uses "
        "phase unwrapping along the full trajectory. "
        "The final-only Stage04 number above is only an audit "
        "of the candidate optimization metric."
    )


if __name__ == "__main__":
    main()
