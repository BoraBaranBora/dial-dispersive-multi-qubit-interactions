from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .core import (
    DATA_DIR,
    FIG_DIR,
    TARGET_REGISTER_PHASE,
    ensure_dirs,
    response_matrix,
    transition_offsets_from_couplings,
    write_csv,
)


N_QUBITS = 3
DIM_REG = 2**N_QUBITS
DIM_TOTAL = 2 * DIM_REG

PAULI_LABEL_ORDER = [
    "Z3",
    "Z2",
    "Z2Z3",
    "Z1",
    "Z1Z3",
    "Z1Z2",
    "Z1Z2Z3",
]


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_float_list(text: str) -> np.ndarray:
    text = str(text).strip()
    if not text:
        return np.array([], dtype=float)

    return np.array([float(x) for x in text.split(";") if x.strip()], dtype=float)


def to_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def idx(e_state: int, alpha: int) -> int:
    return int(e_state) * DIM_REG + int(alpha)


def sigma_z_eigenvalue(e_state: int) -> int:
    return -1 if int(e_state) == 0 else +1


def z_eigenvalue(alpha: int, qubit: int) -> int:
    bit_position = N_QUBITS - qubit
    bit = (int(alpha) >> bit_position) & 1
    return +1 if bit == 0 else -1


def pauli_label_eigenvalue(alpha: int, label: str) -> int:
    value = 1

    for q in (1, 2, 3):
        if f"Z{q}" in label:
            value *= z_eigenvalue(alpha, q)

    return value


def pretty_pauli_label(label: str) -> str:
    digits = [c for c in label if c.isdigit()]
    return "$" + "".join([rf"Z_{d}" for d in digits]) + "$"


def phase_tick_formatter(x, pos):
    if abs(x) < 1e-12:
        return "0"

    value = x / math.pi
    return f"{value:.2f}$\\pi$"


def eta_tag(eta: float) -> str:
    return f"eta_{eta:.4f}".replace(".", "p").replace("-", "m")


def build_feature_matrix() -> tuple[list[str], np.ndarray]:
    feature_names = ["I", "sigma_z"]

    for label in PAULI_LABEL_ORDER:
        feature_names.append(label)

    for label in PAULI_LABEL_ORDER:
        feature_names.append("sigma_z " + label)

    rows = []

    for e_state in range(2):
        sz = sigma_z_eigenvalue(e_state)

        for alpha in range(DIM_REG):
            row = [1.0, float(sz)]

            for label in PAULI_LABEL_ORDER:
                row.append(float(pauli_label_eigenvalue(alpha, label)))

            for label in PAULI_LABEL_ORDER:
                row.append(float(sz * pauli_label_eigenvalue(alpha, label)))

            rows.append(row)

    return feature_names, np.asarray(rows, dtype=float)


FEATURE_NAMES, FEATURE_MATRIX = build_feature_matrix()
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def target_coefficients(target_label: str, signed_register_phase: float) -> np.ndarray:
    coeffs = np.zeros(len(FEATURE_NAMES), dtype=float)
    coeffs[FEATURE_INDEX["sigma_z " + target_label]] = -signed_register_phase
    return coeffs


def build_diagonal_unitary_from_coeffs(coeffs: np.ndarray) -> np.ndarray:
    phases = FEATURE_MATRIX @ coeffs
    return np.diag(np.exp(1j * phases))


def build_target_unitary(target_label: str, signed_register_phase: float) -> np.ndarray:
    coeffs = target_coefficients(target_label, signed_register_phase)
    return build_diagonal_unitary_from_coeffs(coeffs)


def build_mediator_phase_unitary(theta_sigma_z: float) -> np.ndarray:
    coeffs = np.zeros(len(FEATURE_NAMES), dtype=float)
    coeffs[FEATURE_INDEX["sigma_z"]] = theta_sigma_z
    return build_diagonal_unitary_from_coeffs(coeffs)


def unitary_fidelity(U: np.ndarray, U_target: np.ndarray) -> float:
    d = U.shape[0]
    return float(abs(np.trace(U_target.conj().T @ U)) ** 2 / (d * d))


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


def time_average(y: np.ndarray, time: np.ndarray) -> float:
    if len(y) == 0 or len(time) == 0:
        return 0.0

    if float(time[-1]) <= 0.0:
        return float(y[-1])

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, time) / float(time[-1]))

    return float(np.trapz(y, time) / float(time[-1]))


def extract_phase_coefficients(U_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diag = np.diagonal(U_t, axis1=1, axis2=2)
    phases = np.unwrap(np.angle(diag), axis=0)

    coeffs_t = np.zeros((U_t.shape[0], len(FEATURE_NAMES)), dtype=float)
    fit_rms = np.zeros(U_t.shape[0], dtype=float)

    for i in range(U_t.shape[0]):
        coeffs, *_ = np.linalg.lstsq(FEATURE_MATRIX, phases[i], rcond=None)
        fit = FEATURE_MATRIX @ coeffs
        err = phases[i] - fit

        coeffs_t[i] = coeffs
        fit_rms[i] = float(np.sqrt(np.mean(err**2)))

    return coeffs_t, fit_rms


def register_phase_trajectories(coeffs_t: np.ndarray) -> dict[str, np.ndarray]:
    out = {}

    for label in PAULI_LABEL_ORDER:
        feature = "sigma_z " + label
        out[label] = -coeffs_t[:, FEATURE_INDEX[feature]]

    return out


def predicted_register_phases(
    *,
    labels: list[str],
    K: np.ndarray,
    time: np.ndarray,
) -> dict[str, np.ndarray]:
    out = {}

    for i, label in enumerate(labels):
        out[label] = 0.5 * K[i] * time

    return out


def compute_fidelity_trajectories(
    U_t: np.ndarray,
    coeffs_t: np.ndarray,
    U_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros(U_t.shape[0], dtype=float)
    mediator_corrected = np.zeros(U_t.shape[0], dtype=float)

    for i, U in enumerate(U_t):
        raw[i] = unitary_fidelity(U, U_target)

        theta_sigma_z = coeffs_t[i, FEATURE_INDEX["sigma_z"]]
        U_med = build_mediator_phase_unitary(theta_sigma_z)
        U_corr = U_med.conj().T @ U

        mediator_corrected[i] = unitary_fidelity(U_corr, U_target)

    return raw, mediator_corrected


def geometry_short_label(label: str) -> str:
    if label.startswith("high"):
        return "high"
    if label.startswith("intermediate"):
        return "intermediate"
    if label.startswith("low"):
        return "low"
    return label.replace(" ", "_")


def plot_case(
    *,
    outdir: Path,
    case: dict,
    time: np.ndarray,
    phase_t: dict[str, np.ndarray],
    phase_pred_t: dict[str, np.ndarray],
    target_label: str,
    signed_target_phase: float,
    raw_fidelity: np.ndarray,
    corrected_fidelity: np.ndarray,
    max_flip: np.ndarray,
    mean_flip: np.ndarray,
    eta: float,
    alignment_score: float,
) -> Path:
    case_id = int(float(case["case_id"]))
    geom = geometry_short_label(case["geometry_label"])
    target_type = case["target_type"]

    fig_path = outdir / (
        f"fig05_case_{case_id:02d}_{geom}_{target_type}_{target_label}_{eta_tag(eta)}.png"
    )

    tau = time / time[-1] if time[-1] > 0 else time

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.75), constrained_layout=True)

    ax = axes[0]

    for label in PAULI_LABEL_ORDER:
        if label == target_label:
            ax.plot(
                tau,
                phase_t[label],
                linewidth=2.2,
                label=pretty_pauli_label(label) + " full RWA",
            )
            ax.plot(
                tau,
                phase_pred_t[label],
                linestyle="--",
                linewidth=1.3,
                label=pretty_pauli_label(label) + " effective",
            )
        else:
            ax.plot(
                tau,
                phase_t[label],
                linewidth=0.9,
                alpha=0.55,
                label=pretty_pauli_label(label),
            )

    ax.axhline(
        signed_target_phase,
        linestyle=":",
        linewidth=1.4,
        color="black",
        label=r"target $\phi$",
    )
    ax.axhline(0.0, linestyle="-", linewidth=0.6, color="0.75")

    ax.set_xlabel(r"normalized time $t/T$")
    ax.set_ylabel(r"register phase $\phi_S(t)$")
    ax.set_title("(a) extracted Pauli-Z phases", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(phase_tick_formatter))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, ncol=2, frameon=True)

    ax_pop = axes[1]
    ax_fid = ax_pop.twinx()

    ax_pop.plot(
        tau,
        max_flip,
        linewidth=1.8,
        label=r"max $P_{\rm flip}$",
    )
    ax_pop.plot(
        tau,
        mean_flip,
        linestyle="--",
        linewidth=1.3,
        label=r"mean $P_{\rm flip}$",
    )

    ax_fid.plot(
        tau,
        raw_fidelity,
        linewidth=1.8,
        label=r"$F_{\rm raw}$",
    )
    ax_fid.plot(
        tau,
        corrected_fidelity,
        linestyle="--",
        linewidth=1.8,
        label=r"$F_{\rm med.corr.}$",
    )

    ax_pop.set_xlabel(r"normalized time $t/T$")
    ax_pop.set_ylabel("mediator population transfer")
    ax_fid.set_ylabel("process fidelity")

    ax_pop.set_ylim(bottom=-0.02)
    ax_fid.set_ylim(-0.02, 1.02)

    ax_pop.set_title("(b) mediator motion and fidelity", fontsize=9)
    ax_pop.grid(alpha=0.25)

    lines0, labels0 = ax_pop.get_legend_handles_labels()
    lines1, labels1 = ax_fid.get_legend_handles_labels()
    ax_fid.legend(
        lines0 + lines1,
        labels0 + labels1,
        fontsize=7,
        loc="center right",
        frameon=True,
    )

    fig.suptitle(
        f"{case['geometry_label']} | target {target_label} | "
        rf"$\rho={float(case['rho']):.3f}$, "
        rf"$\sigma={float(case['sigma']):.3f}$, "
        rf"$\eta={eta:.3f}$, "
        rf"$A_{{\rm align}}={alignment_score:.3f}$",
        fontsize=10,
    )

    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path


def validate_case(
    case: dict,
    *,
    drive_scale: float,
    eta: float,
    num_samples: int,
    steps_per_period: int,
    max_steps: int,
    outdir: Path,
) -> tuple[dict, list[dict]]:
    case_id = int(float(case["case_id"]))
    target_label = case["target_label"]

    rho = to_float(case, "rho")
    sigma = to_float(case, "sigma")

    A = np.array(
        [
            to_float(case, "A1"),
            to_float(case, "A2"),
            to_float(case, "A3"),
        ],
        dtype=float,
    )

    offsets = transition_offsets_from_couplings(A)

    tones = parse_float_list(case["selected_tones"])
    x_unit = parse_float_list(case["selected_intensities_unit"])

    if len(tones) != len(x_unit):
        raise ValueError(
            f"Case {case_id}: selected_tones and selected_intensities_unit "
            f"have different lengths."
        )

    intensities = drive_scale * x_unit

    labels, G_selected = response_matrix(
        offsets,
        tones,
        include_identity=False,
    )

    K_unit = G_selected @ x_unit
    K = drive_scale * K_unit

    label_to_index = {label: i for i, label in enumerate(labels)}
    target_index = label_to_index[target_label]

    target_rate = float(K[target_index])

    target_register_phase = to_float(
        case,
        "target_register_phase",
        TARGET_REGISTER_PHASE,
    )

    if abs(target_rate) <= 1e-14:
        raise ValueError(f"Case {case_id}: target rate is zero.")

    T_gate = 2.0 * abs(target_register_phase) / abs(target_rate)
    signed_target_phase = math.copysign(target_register_phase, target_rate)

    print(
        f"[full RWA] case {case_id:02d}: "
        f"{case['geometry_label']} target={target_label} "
        f"T={T_gate:.3e}, drive_scale={drive_scale:.2e}, eta={eta:.4f}",
        flush=True,
    )

    sim = simulate_case(
        offsets=offsets,
        tones=tones,
        intensities=intensities,
        T_gate=T_gate,
        eta=eta,
        num_samples=num_samples,
        steps_per_period=steps_per_period,
        max_steps=max_steps,
    )

    U_t = sim["U_t"]
    time = sim["time"]

    coeffs_t, fit_rms = extract_phase_coefficients(U_t)
    phase_t = register_phase_trajectories(coeffs_t)
    phase_pred_t = predicted_register_phases(labels=labels, K=K, time=time)

    U_target = build_target_unitary(target_label, signed_target_phase)

    raw_fid, corr_fid = compute_fidelity_trajectories(
        U_t,
        coeffs_t,
        U_target,
    )

    fig_path = plot_case(
        outdir=outdir,
        case=case,
        time=time,
        phase_t=phase_t,
        phase_pred_t=phase_pred_t,
        target_label=target_label,
        signed_target_phase=signed_target_phase,
        raw_fidelity=raw_fid,
        corrected_fidelity=corr_fid,
        max_flip=sim["max_flip"],
        mean_flip=sim["mean_flip"],
        eta=eta,
        alignment_score=float(sim["alignment_score"]),
    )

    final_phases = {label: float(phase_t[label][-1]) for label in PAULI_LABEL_ORDER}

    target_error = final_phases[target_label] - signed_target_phase

    spectator_only = [
        final_phases[label]
        for label in PAULI_LABEL_ORDER
        if label != target_label
    ]
    spectator_rms = float(np.sqrt(np.mean(np.asarray(spectator_only) ** 2)))

    max_flip_population = float(np.max(sim["max_flip"]))
    mean_flip_population_max = float(np.max(sim["mean_flip"]))

    time_averaged_max_flip_population = time_average(sim["max_flip"], time)
    time_averaged_mean_flip_population = time_average(sim["mean_flip"], time)

    final_max_flip_population = float(sim["max_flip"][-1])
    final_mean_flip_population = float(sim["mean_flip"][-1])

    final_off_diagonal_weight = float(sim["off_diagonal_weight"][-1])
    max_off_diagonal_weight = float(np.max(sim["off_diagonal_weight"]))
    
    final_register_configuration_leakage = float(
        sim["register_configuration_leakage"][-1]
    )
    max_register_configuration_leakage = float(
        np.max(sim["register_configuration_leakage"])
    )
    time_averaged_register_configuration_leakage = time_average(
        sim["register_configuration_leakage"],
        time,
    )

    final_register_configuration_leakage_g_initial = float(
        sim["register_configuration_leakage_g_initial"][-1]
    )
    max_register_configuration_leakage_g_initial = float(
        np.max(sim["register_configuration_leakage_g_initial"])
    )
    time_averaged_register_configuration_leakage_g_initial = time_average(
        sim["register_configuration_leakage_g_initial"],
        time,
    )

    max_amplitude = float(sim["max_amplitude"])

    if max_amplitude > 0.0:
        rabi_cycles_at_max_amplitude = float(
            T_gate * max_amplitude / (2.0 * math.pi)
        )
        target_rate_over_max_amplitude = float(abs(target_rate) / max_amplitude)
    else:
        rabi_cycles_at_max_amplitude = float("nan")
        target_rate_over_max_amplitude = float("nan")

    summary = {
        "case_id": case_id,
        "case_label": case["case_label"],
        "geometry_label": case["geometry_label"],
        "target_type": case["target_type"],
        "target_label": target_label,
        "rho": rho,
        "sigma": sigma,
        "eta": float(eta),
        "eta_degrees": float(eta * 180.0 / math.pi),
        "alignment_score": float(sim["alignment_score"]),
        "alignment_leakage": float(sim["alignment_leakage"]),
        "expected_local_alignment_score": float(sim["expected_local_alignment_score"]),
        "drive_scale": float(drive_scale),
        "T_gate": float(T_gate),
        "signed_target_phase": float(signed_target_phase),
        "target_rate": float(target_rate),
        "num_samples": int(len(time)),
        "n_steps": int(sim["n_steps"]),
        "dt": float(sim["dt"]),
        "max_amplitude": max_amplitude,
        "rabi_cycles_at_max_amplitude": rabi_cycles_at_max_amplitude,
        "target_rate_over_max_amplitude": target_rate_over_max_amplitude,
        "max_detuning_frequency": float(sim["max_detuning_frequency"]),
        "final_raw_fidelity": float(raw_fid[-1]),
        "final_mediator_corrected_fidelity": float(corr_fid[-1]),
        "max_raw_fidelity": float(np.max(raw_fid)),
        "max_mediator_corrected_fidelity": float(np.max(corr_fid)),
        "max_flip_population": max_flip_population,
        "mean_flip_population_max": mean_flip_population_max,
        "time_averaged_max_flip_population": time_averaged_max_flip_population,
        "time_averaged_mean_flip_population": time_averaged_mean_flip_population,
        "final_max_flip_population": final_max_flip_population,
        "final_mean_flip_population": final_mean_flip_population,
        "final_off_diagonal_weight": final_off_diagonal_weight,
        "max_off_diagonal_weight": max_off_diagonal_weight,
        "final_target_phase": float(final_phases[target_label]),
        "target_phase_error": float(target_error),
        "spectator_phase_rms": float(spectator_rms),
        "max_phase_fit_rms": float(np.max(fit_rms)),
        "figure_path": str(fig_path),
        "final_register_configuration_leakage": final_register_configuration_leakage,
        "max_register_configuration_leakage": max_register_configuration_leakage,
        "time_averaged_register_configuration_leakage": (
            time_averaged_register_configuration_leakage
        ),
        "final_register_configuration_leakage_g_initial": (
            final_register_configuration_leakage_g_initial
        ),
        "max_register_configuration_leakage_g_initial": (
            max_register_configuration_leakage_g_initial
        ),
        "time_averaged_register_configuration_leakage_g_initial": (
            time_averaged_register_configuration_leakage_g_initial
        ),
    }

    for label in PAULI_LABEL_ORDER:
        summary[f"final_phi_{label}"] = final_phases[label]
        summary[f"predicted_final_phi_{label}"] = float(phase_pred_t[label][-1])

    trajectory_rows = []

    for i, t in enumerate(time):
        row = {
            "case_id": case_id,
            "case_label": case["case_label"],
            "eta": float(eta),
            "eta_degrees": float(eta * 180.0 / math.pi),
            "alignment_score": float(sim["alignment_score"]),
            "alignment_leakage": float(sim["alignment_leakage"]),
            "time": float(t),
            "normalized_time": float(t / T_gate) if T_gate > 0 else 0.0,
            "raw_fidelity": float(raw_fid[i]),
            "mediator_corrected_fidelity": float(corr_fid[i]),
            "max_flip_population": float(sim["max_flip"][i]),
            "mean_flip_population": float(sim["mean_flip"][i]),
            "off_diagonal_weight": float(sim["off_diagonal_weight"][i]),
            "register_configuration_leakage": float(
                sim["register_configuration_leakage"][i]
            ),
            "register_configuration_leakage_g_initial": float(
                sim["register_configuration_leakage_g_initial"][i]
            ),
            "phase_fit_rms": float(fit_rms[i]),
        }

        for label in PAULI_LABEL_ORDER:
            row[f"phi_{label}"] = float(phase_t[label][i])
            row[f"phi_pred_{label}"] = float(phase_pred_t[label][i])

        trajectory_rows.append(row)

    print(
        f"[case {case_id:02d}] "
        f"T={T_gate:.6g}, "
        f"F_raw={raw_fid[-1]:.6f}, "
        f"F_corr={corr_fid[-1]:.6f}, "
        f"A_align={float(sim['alignment_score']):.3f}, "
        f"avg mean P_flip={time_averaged_mean_flip_population:.3e}, "
        f"final mean P_flip={final_mean_flip_population:.3e}, "
        f"cycles@Omax={rabi_cycles_at_max_amplitude:.3f}, "
        f"target err={target_error:.3e}, "
        f"reg leak={final_register_configuration_leakage:.3e}",
        flush=True,
    )

    return summary, trajectory_rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        type=Path,
        default=DATA_DIR / "validation_cases.csv",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of case IDs to run.",
    )
    parser.add_argument("--drive-scale", type=float, default=5e-4)
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Single-qubit register-basis misalignment angle in radians.",
    )
    parser.add_argument(
        "--eta-seed",
        type=int,
        default=None,
        help="Ignored; kept only for compatibility with older wrapper scripts.",
    )
    parser.add_argument("--num-samples", type=int, default=501)
    parser.add_argument("--steps-per-period", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=FIG_DIR / "full_rwa_validation",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.drive_scale <= 0:
        raise ValueError("--drive-scale must be positive.")

    if args.eta < 0:
        raise ValueError("--eta must be nonnegative.")

    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2.")

    ensure_dirs()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.cases.exists():
        raise FileNotFoundError(
            f"Could not find {args.cases}. "
            "Run 03_validation_case_design.py first."
        )

    cases = load_csv_rows(args.cases)

    if args.case_ids is not None and len(args.case_ids) > 0:
        wanted = set(args.case_ids)
        cases = [c for c in cases if int(float(c["case_id"])) in wanted]

    if not cases:
        raise ValueError("No validation cases selected.")

    summary_rows = []
    trajectory_rows = []

    for case in cases:
        summary, traj = validate_case(
            case,
            drive_scale=args.drive_scale,
            eta=args.eta,
            num_samples=args.num_samples,
            steps_per_period=args.steps_per_period,
            max_steps=args.max_steps,
            outdir=args.outdir,
        )

        summary_rows.append(summary)
        trajectory_rows.extend(traj)

    summary_csv = (
        args.summary_csv
        if args.summary_csv is not None
        else DATA_DIR / "full_rwa_validation_summary.csv"
    )
    trajectory_csv = (
        args.trajectory_csv
        if args.trajectory_csv is not None
        else DATA_DIR / "full_rwa_validation_trajectories.csv"
    )

    write_csv(summary_csv, summary_rows)
    print(f"[write] {summary_csv}")

    write_csv(trajectory_csv, trajectory_rows)
    print(f"[write] {trajectory_csv}")

    print()
    print("[full RWA validation summary]")
    
    print(
        "case geometry                  target      eta      A_align   F_raw     F_corr    "
        "avgMeanP   finalMeanP   maxPflip   regLeak    cycles@Omax   target_err"
    )

    for r in summary_rows:
        print(
            f"{int(r['case_id']):>2d}   "
            f"{r['geometry_label']:<25} "
            f"{r['target_label']:<10} "
            f"{r['eta']:>6.3f} "
            f"{r['alignment_score']:>8.3f} "
            f"{r['final_raw_fidelity']:>8.5f} "
            f"{r['final_mediator_corrected_fidelity']:>8.5f} "
            f"{r['time_averaged_mean_flip_population']:>9.2e} "
            f"{r['final_mean_flip_population']:>10.2e} "
            f"{r['max_flip_population']:>9.2e} "
            f"{r['final_register_configuration_leakage']:>9.2e} "
            f"{r['rabi_cycles_at_max_amplitude']:>12.3f} "
            f"{r['target_phase_error']:>10.2e}"
        )

    print()
    print(f"[figures] {args.outdir}")


if __name__ == "__main__":
    main()