"""Driven mediator-register dynamics used to validate DIAL controls."""

from __future__ import annotations

import math

import numpy as np
from numba import njit


# The full basis-mismatch audit in the paper is the n=3 benchmark.
N_QUBITS = 3
DIM_REG = 2**N_QUBITS
DIM_TOTAL = 2 * DIM_REG


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


def conservative_n_steps(
    *,
    offsets: np.ndarray,
    span: float,
    total_intensity_max: float,
    T_gate: float,
    steps_per_period: int,
    max_steps: int,
) -> int:
    """
    Use one fixed integration grid for the complete optimization.

    This avoids changing numerical resolution as the optimizer moves
    around parameter space.
    """
    edge_detunings = []

    for offset in offsets:
        edge_detunings.append(
            abs(-span - offset)
        )
        edge_detunings.append(
            abs(+span - offset)
        )

    max_detuning = max(
        edge_detunings
    )

    # Since sum x_k <= X_tot,
    # every individual amplitude obeys
    #
    #   Omega_k = sqrt(2 x_k)
    #           <= sqrt(2 X_tot).
    max_amplitude = math.sqrt(
        2.0 * total_intensity_max
    )

    max_rate = max(
        max_detuning,
        max_amplitude,
        1e-12,
    )

    n_steps = int(
        math.ceil(
            T_gate
            * steps_per_period
            * max_rate
            / (2.0 * math.pi)
        )
    )

    if n_steps > max_steps:
        raise ValueError(
            f"Required n_steps={n_steps} exceeds "
            f"--max-steps={max_steps}. "
            "Do not silently cap the optimization resolution."
        )

    return max(n_steps, 1)


def single_qubit_ry(theta: float) -> np.ndarray:
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)

    return np.asarray(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.complex128,
    )


def kron_all(mats: list[np.ndarray]) -> np.ndarray:
    out = np.asarray([[1.0]], dtype=np.complex128)

    for mat in mats:
        out = np.kron(out, mat)

    return out


def local_register_rotation(eta: float) -> np.ndarray:
    single = single_qubit_ry(eta)
    return kron_all([single for _ in range(N_QUBITS)])


def overlap_matrix_from_eta(eta: float) -> np.ndarray:
    if abs(eta) <= 0.0:
        return np.eye(DIM_REG, dtype=np.complex128)

    rotation = local_register_rotation(eta)

    # If |beta_e(eta)> = R(eta)|beta_g>, then
    # M_{beta,alpha}(eta) = <beta_e(eta)|alpha_g>
    #                     = <beta_g|R^\dagger(eta)|alpha_g>.
    return rotation.conj().T


def alignment_score_from_overlap(overlap: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diag(overlap)) ** 2))


def expected_local_alignment_score(eta: float) -> float:
    return float(math.cos(0.5 * eta) ** (2 * N_QUBITS))


def build_full_hamiltonian(
    *,
    t: float,
    offsets: np.ndarray,
    tones: np.ndarray,
    amplitudes: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    coupling_ge = np.zeros((DIM_REG, DIM_REG), dtype=np.complex128)

    for omega, amp in zip(tones, amplitudes):
        phase_beta = np.exp(-1j * (omega - offsets) * t)

        # Rows: ground alpha. Columns: excited beta.
        #
        # The overlap is M_{beta,alpha}. Therefore overlap.T has
        # indices [alpha,beta].
        #
        # At eta=0, overlap=I and this reduces exactly to the old
        # independent 2x2 blocks.
        coupling_ge += 0.5 * amp * overlap.T * phase_beta[None, :]

    H = np.zeros((DIM_TOTAL, DIM_TOTAL), dtype=np.complex128)

    g = slice(0, DIM_REG)
    e = slice(DIM_REG, DIM_TOTAL)

    H[g, e] = coupling_ge
    H[e, g] = coupling_ge.conj().T

    return H


def full_derivative(
    U: np.ndarray,
    *,
    t: float,
    offsets: np.ndarray,
    tones: np.ndarray,
    amplitudes: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    H = build_full_hamiltonian(
        t=t,
        offsets=offsets,
        tones=tones,
        amplitudes=amplitudes,
        overlap=overlap,
    )

    return -1j * H @ U


def rk4_step(
    U: np.ndarray,
    *,
    t: float,
    dt: float,
    offsets: np.ndarray,
    tones: np.ndarray,
    amplitudes: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    k1 = full_derivative(
        U,
        t=t,
        offsets=offsets,
        tones=tones,
        amplitudes=amplitudes,
        overlap=overlap,
    )

    k2 = full_derivative(
        U + 0.5 * dt * k1,
        t=t + 0.5 * dt,
        offsets=offsets,
        tones=tones,
        amplitudes=amplitudes,
        overlap=overlap,
    )

    k3 = full_derivative(
        U + 0.5 * dt * k2,
        t=t + 0.5 * dt,
        offsets=offsets,
        tones=tones,
        amplitudes=amplitudes,
        overlap=overlap,
    )

    k4 = full_derivative(
        U + dt * k3,
        t=t + dt,
        offsets=offsets,
        tones=tones,
        amplitudes=amplitudes,
        overlap=overlap,
    )

    return U + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def choose_time_steps(
    *,
    T_gate: float,
    nus: np.ndarray,
    amplitudes: np.ndarray,
    steps_per_period: int,
    max_steps: int,
    num_samples: int,
) -> tuple[int, np.ndarray]:
    max_frequency = float(np.max(np.abs(nus))) if nus.size else 1.0
    max_amplitude = float(np.max(np.abs(amplitudes))) if amplitudes.size else 0.0

    max_rate = max(max_frequency, max_amplitude, 1e-12)
    dt_max = 2.0 * math.pi / (steps_per_period * max_rate)

    n_steps = max(num_samples - 1, int(math.ceil(T_gate / dt_max)))

    if n_steps > max_steps:
        print(
            f"[warning] requested {n_steps} integration steps; "
            f"capping to --max-steps={max_steps}",
            flush=True,
        )
        n_steps = max_steps

    sample_indices = np.unique(
        np.round(np.linspace(0, n_steps, num_samples)).astype(int)
    )

    return n_steps, sample_indices


def flip_probabilities_from_full_unitary(U: np.ndarray) -> np.ndarray:
    g = slice(0, DIM_REG)
    e = slice(DIM_REG, DIM_TOTAL)

    Ueg = U[e, g]

    return np.sum(np.abs(Ueg) ** 2, axis=0)


def off_diagonal_weight(U: np.ndarray) -> float:
    total = float(np.sum(np.abs(U) ** 2))
    diagonal = float(np.sum(np.abs(np.diag(U)) ** 2))

    return float((total - diagonal) / U.shape[0])


def register_configuration_leakage(
    U: np.ndarray,
    *,
    initial_mediator: int | None = None,
) -> float:
    """
    Average probability that the register configuration changes.

    This captures Raman-like population transfer between register
    configurations in the full rotating-wave propagator.

    Basis convention follows idx(e_state, alpha):
        |g, alpha> -> alpha
        |e, alpha> -> DIM_REG + alpha

    If initial_mediator is None, average over both initial mediator states.
    If initial_mediator is 0, average only over initial |g, alpha>.
    If initial_mediator is 1, average only over initial |e, alpha>.
    """
    if U.shape != (DIM_TOTAL, DIM_TOTAL):
        raise ValueError(f"Expected U shape {(DIM_TOTAL, DIM_TOTAL)}, got {U.shape}.")

    mediator_inputs = [0, 1] if initial_mediator is None else [int(initial_mediator)]

    probs = np.abs(U) ** 2
    leakage = 0.0
    count = 0

    for e_in in mediator_inputs:
        for alpha_in in range(DIM_REG):
            col = idx(e_in, alpha_in)
            count += 1

            for e_out in range(2):
                for alpha_out in range(DIM_REG):
                    if alpha_out == alpha_in:
                        continue

                    row = idx(e_out, alpha_out)
                    leakage += probs[row, col]

    return float(leakage / count)


def simulate_case(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    intensities: np.ndarray,
    T_gate: float,
    eta: float,
    num_samples: int,
    steps_per_period: int,
    max_steps: int,
) -> dict:
    amplitudes = np.sqrt(np.maximum(2.0 * intensities, 0.0))

    nus = tones[None, :] - offsets[:, None]

    overlap = overlap_matrix_from_eta(eta)
    alignment_score = alignment_score_from_overlap(overlap)

    n_steps, sample_indices = choose_time_steps(
        T_gate=T_gate,
        nus=nus,
        amplitudes=amplitudes,
        steps_per_period=steps_per_period,
        max_steps=max_steps,
        num_samples=num_samples,
    )

    dt = T_gate / float(n_steps)

    U = np.eye(DIM_TOTAL, dtype=np.complex128)

    sample_set = set(int(i) for i in sample_indices)

    U_t = []
    time = []
    max_flip = []
    mean_flip = []
    offdiag = []
    reg_leak = []
    reg_leak_g_initial = []

    def record(step: int, t: float) -> None:
        U_t.append(U.copy())
        time.append(t)

        flips = flip_probabilities_from_full_unitary(U)

        max_flip.append(float(np.max(flips)))
        mean_flip.append(float(np.mean(flips)))
        offdiag.append(float(off_diagonal_weight(U)))
        reg_leak.append(float(register_configuration_leakage(U)))
        reg_leak_g_initial.append(
            float(register_configuration_leakage(U, initial_mediator=0))
        )

    record(0, 0.0)

    t = 0.0

    for step in range(1, n_steps + 1):
        U = rk4_step(
            U,
            t=t,
            dt=dt,
            offsets=offsets,
            tones=tones,
            amplitudes=amplitudes,
            overlap=overlap,
        )

        t += dt

        if step in sample_set:
            record(step, t)

    return {
        "U_t": np.asarray(U_t),
        "time": np.asarray(time),
        "max_flip": np.asarray(max_flip),
        "mean_flip": np.asarray(mean_flip),
        "off_diagonal_weight": np.asarray(offdiag),
        "register_configuration_leakage": np.asarray(reg_leak),
        "register_configuration_leakage_g_initial": np.asarray(reg_leak_g_initial),
        "n_steps": int(n_steps),
        "dt": float(dt),
        "max_amplitude": float(np.max(amplitudes)) if len(amplitudes) else 0.0,
        "max_detuning_frequency": float(np.max(np.abs(nus))) if nus.size else 0.0,
        "alignment_score": float(alignment_score),
        "alignment_leakage": float(1.0 - alignment_score),
        "expected_local_alignment_score": expected_local_alignment_score(eta),
    }
