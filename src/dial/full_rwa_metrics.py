from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import aligned_dynamics as jit


def load_control(path: Path) -> dict:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    inputs = data["inputs"]
    solution = data["refined_solution"]

    tones = np.asarray(
        solution["tones"],
        dtype=float,
    )
    intensities = np.asarray(
        solution["intensities"],
        dtype=float,
    )

    order = np.argsort(tones)

    return {
        "data": data,
        "offsets": np.asarray(
            inputs["offsets"],
            dtype=float,
        ),
        "tones": tones[order],
        "intensities": intensities[order],
        "target_label": str(inputs["target"]),
        "target_phase": float(inputs["target_phase"]),
        "T_gate": float(solution["T_effective"]),
        "span": float(inputs["span"]),
        "r_disp": float(inputs["r_disp"]),
        "total_intensity_max": float(
            inputs["total_intensity_max"]
        ),
        "min_tone_spacing": float(
            inputs["min_tone_spacing"]
        ),
        "target_sign": float(
            data["coarse_selection"]["target_sign"]
        ),
    }


def local_intensity_caps(
    tones: np.ndarray,
    offsets: np.ndarray,
    *,
    r_disp: float,
    omega_hw_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    detunings = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    omega_caps = (
        r_disp * detunings
    )

    if math.isfinite(omega_hw_max):
        omega_caps = np.minimum(
            omega_caps,
            omega_hw_max,
        )

    xmax = (
        0.5 * omega_caps**2
    )

    return xmax, detunings


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


def random_spaced_tones(
    rng: np.random.Generator,
    *,
    M: int,
    span: float,
    min_spacing: float,
) -> np.ndarray:
    """
    Uniform ordered seed coordinates with guaranteed spacing.

    If y_1 <= ... <= y_M are sampled in the compressed interval,
    then

        omega_i = y_i + i * delta

    guarantees omega_{i+1} - omega_i >= delta.
    """
    compressed_hi = (
        span
        - (M - 1) * min_spacing
    )

    compressed_lo = -span

    if compressed_hi <= compressed_lo:
        raise ValueError(
            "Frequency window is too small "
            "for the requested number of tones "
            "and minimum spacing."
        )

    y = np.sort(
        rng.uniform(
            compressed_lo,
            compressed_hi,
            size=M,
        )
    )

    tones = (
        y
        + min_spacing
        * np.arange(M)
    )

    return tones


def random_feasible_fractions(
    rng: np.random.Generator,
    *,
    xmax: np.ndarray,
    total_intensity_max: float,
) -> np.ndarray:
    """
    Construct a nontrivial feasible random power allocation.

    Random positive weights are scaled until approximately 90% of the
    smaller of the global power budget and available local-cap budget
    is occupied.
    """
    available = float(
        np.sum(xmax)
    )

    if available <= 0.0:
        return np.zeros_like(xmax)

    target = 0.90 * min(
        total_intensity_max,
        available,
    )

    weights = rng.exponential(
        scale=1.0,
        size=len(xmax),
    )

    weights = np.maximum(
        weights,
        1e-12,
    )

    # Find alpha such that
    #
    #   sum min(alpha*w_k, xmax_k) = target.
    lo = 0.0
    hi = 1.0

    def allocated(alpha: float) -> float:
        return float(
            np.sum(
                np.minimum(
                    alpha * weights,
                    xmax,
                )
            )
        )

    while allocated(hi) < target:
        hi *= 2.0

        if hi > 1e20:
            break

    for _ in range(80):
        mid = 0.5 * (lo + hi)

        if allocated(mid) < target:
            lo = mid
        else:
            hi = mid

    x = np.minimum(
        hi * weights,
        xmax,
    )

    fractions = np.divide(
        x,
        xmax,
        out=np.zeros_like(x),
        where=xmax > 0.0,
    )

    return np.clip(
        fractions,
        0.0,
        1.0,
    )


class FullRWAObjective:
    def __init__(
        self,
        *,
        offsets: np.ndarray,
        M: int,
        T_gate: float,
        n_steps: int,
        target_label: str,
        signed_target_phase: float,
        r_disp: float,
        omega_hw_max: float,
        total_intensity_max: float,
        min_tone_spacing: float,
    ):
        self.offsets = offsets
        self.M = M
        self.T_gate = T_gate
        self.n_steps = n_steps
        self.target_label = target_label
        self.signed_target_phase = (
            signed_target_phase
        )
        self.r_disp = r_disp
        self.omega_hw_max = (
            omega_hw_max
        )
        self.total_intensity_max = (
            total_intensity_max
        )
        self.min_tone_spacing = (
            min_tone_spacing
        )

        self.n_evaluations = 0

        # Best point seen regardless of feasibility.
        self.best_fidelity = -np.inf
        self.best_z = None

        # Best physically feasible point seen.
        self.best_feasible_fidelity = -np.inf
        self.best_feasible_z = None

    def decode(
        self,
        z: np.ndarray,
    ):
        return decode_parameters(
            z,
            M=self.M,
            offsets=self.offsets,
            r_disp=self.r_disp,
            omega_hw_max=(
                self.omega_hw_max
            ),
        )

    def fidelity(
        self,
        z: np.ndarray,
    ) -> float:
        (
            tones,
            intensities,
            _,
            _,
        ) = self.decode(z)

        blocks = (
            jit.propagate_final_blocks_jit(
                self.offsets,
                tones,
                intensities,
                self.T_gate,
                self.n_steps,
            )
        )

        F = (
            jit.mediator_z_optimized_fidelity(
                blocks,
                target_label=(
                    self.target_label
                ),
                signed_target_phase=(
                    self.signed_target_phase
                ),
            )
        )

        self.n_evaluations += 1

        z_array = np.asarray(
            z,
            dtype=float,
        )

        if F > self.best_fidelity:
            self.best_fidelity = F
            self.best_z = z_array.copy()

        (
            tones,
            intensities,
            _xmax,
            _detunings,
        ) = self.decode(z_array)

        spacing_feasible = bool(
            np.all(
                np.diff(tones)
                >= self.min_tone_spacing - 1e-8
            )
        )

        power_feasible = bool(
            np.sum(intensities)
            <= self.total_intensity_max + 1e-10
        )

        feasible = (
            spacing_feasible
            and power_feasible
        )

        if (
            feasible
            and F > self.best_feasible_fidelity
        ):
            self.best_feasible_fidelity = F
            self.best_feasible_z = z_array.copy()

        if self.n_evaluations % 25 == 0:
            feasible_text = (
                f"{self.best_feasible_fidelity:.10f}"
                if np.isfinite(
                    self.best_feasible_fidelity
                )
                else "none"
            )

            print(
                f"    [eval {self.n_evaluations:>5d}] "
                f"F={F:.10f} "
                f"bestAny={self.best_fidelity:.10f} "
                f"bestFeas={feasible_text}",
                flush=True,
            )

        return float(F)

    def __call__(
        self,
        z: np.ndarray,
    ) -> float:
        return (
            1.0
            - self.fidelity(z)
        )


def spacing_constraint(
    z: np.ndarray,
    *,
    M: int,
    min_spacing: float,
) -> np.ndarray:
    tones = z[:M]

    return (
        np.diff(tones)
        - min_spacing
    )


def power_constraint(
    z: np.ndarray,
    *,
    M: int,
    offsets: np.ndarray,
    r_disp: float,
    omega_hw_max: float,
    total_intensity_max: float,
) -> float:
    (
        _,
        intensities,
        _,
        _,
    ) = decode_parameters(
        z,
        M=M,
        offsets=offsets,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
    )

    return float(
        total_intensity_max
        - np.sum(intensities)
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
        jit.propagate_final_blocks_jit(
            offsets,
            tones,
            intensities,
            T_gate,
            n_steps,
        )
    )

    if fidelity_mode == "mediator_z_optimized":
        F = (
            jit.mediator_z_optimized_fidelity(
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
        jit.final_max_ground_to_excited_flip(
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


def run_one_start(
    *,
    name: str,
    z0: np.ndarray,
    offsets: np.ndarray,
    M: int,
    T_gate: float,
    n_steps_search: int,
    n_steps_final: int,
    target_label: str,
    signed_target_phase: float,
    r_disp: float,
    omega_hw_max: float,
    total_intensity_max: float,
    min_tone_spacing: float,
    span: float,
    maxiter: int,
    ftol: float,
) -> dict:
    objective = FullRWAObjective(
        offsets=offsets,
        M=M,
        T_gate=T_gate,
        n_steps=n_steps_search,
        target_label=target_label,
        signed_target_phase=(
            signed_target_phase
        ),
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
        total_intensity_max=(
            total_intensity_max
        ),
        min_tone_spacing=(
            min_tone_spacing
        ),
    )

    F0_search = objective.fidelity(
        z0
    )

    F0_final, initial_details = (
        final_fidelity(
            z0,
            offsets=offsets,
            M=M,
            T_gate=T_gate,
            n_steps=n_steps_final,
            target_label=target_label,
            signed_target_phase=(
                signed_target_phase
            ),
            r_disp=r_disp,
            omega_hw_max=omega_hw_max,
        )
    )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z: spacing_constraint(
                z,
                M=M,
                min_spacing=(
                    min_tone_spacing
                ),
            ),
        },
        {
            "type": "ineq",
            "fun": lambda z: power_constraint(
                z,
                M=M,
                offsets=offsets,
                r_disp=r_disp,
                omega_hw_max=(
                    omega_hw_max
                ),
                total_intensity_max=(
                    total_intensity_max
                ),
            ),
        },
    ]

    bounds = (
        [(-span, +span)] * M
        + [(0.0, 1.0)] * M
    )

    # Frequency derivatives need a much smaller relative perturbation
    # because T is long; power fractions can use a larger perturbation.
    finite_diff_rel_step = np.concatenate(
        [
            np.full(M, 1e-7),
            np.full(M, 1e-4),
        ]
    )

    print()
    print(f"[start: {name}]")
    print(
        f"  initial F_search = "
        f"{F0_search:.10f}"
    )
    print(
        f"  initial F_24     = "
        f"{F0_final:.10f}"
    )
    print(
        f"  initial power    = "
        f"{initial_details['total_intensity']:.8e}"
    )
    print(
        f"  initial spacing  = "
        f"{initial_details['minimum_spacing']:.8e}"
    )

    iteration_state = {
        "iteration": 0,
    }

    def progress_callback(z_current):
        iteration_state["iteration"] += 1

        (
            _tones,
            _intensities,
            _xmax,
            _detunings,
        ) = decode_parameters(
            z_current,
            M=M,
            offsets=offsets,
            r_disp=r_disp,
            omega_hw_max=omega_hw_max,
        )

        current_power = float(
            np.sum(_intensities)
        )

        current_spacing = float(
            np.min(
                np.diff(
                    z_current[:M]
                )
            )
        )

        feasible_text = (
            f"{objective.best_feasible_fidelity:.10f}"
            if np.isfinite(
                objective.best_feasible_fidelity
            )
            else "none"
        )

        print(
            f"  [iter {iteration_state['iteration']:>3d}] "
            f"evals={objective.n_evaluations:>5d} "
            f"bestAny={objective.best_fidelity:.10f} "
            f"bestFeas={feasible_text} "
            f"power={current_power:.3e} "
            f"spacing={current_spacing:.5f}",
            flush=True,
        )

    t0 = time.perf_counter()

    result = minimize(
        objective,
        z0,
        method="SLSQP",
        jac="2-point",
        bounds=bounds,
        constraints=constraints,
        callback=progress_callback,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "disp": False,
            "finite_diff_rel_step":
                finite_diff_rel_step,
        },
    )

    elapsed = (
        time.perf_counter()
        - t0
    )

    # Use the best point actually seen by the expensive objective,
    # not blindly result.x.
    candidates = [
        np.asarray(
            z0,
            dtype=float,
        ),
        np.asarray(
            result.x,
            dtype=float,
        ),
    ]

    if objective.best_feasible_z is not None:
        candidates.append(
            objective.best_feasible_z
        )

    best_z = None
    best_F24 = -np.inf
    best_details = None

    for candidate in candidates:
        # Only accept physically feasible final points.
        spacing_ok = bool(
            np.all(
                spacing_constraint(
                    candidate,
                    M=M,
                    min_spacing=(
                        min_tone_spacing
                    ),
                )
                >= -1e-8
            )
        )

        power_ok = (
            power_constraint(
                candidate,
                M=M,
                offsets=offsets,
                r_disp=r_disp,
                omega_hw_max=(
                    omega_hw_max
                ),
                total_intensity_max=(
                    total_intensity_max
                ),
            )
            >= -1e-10
        )

        if not (
            spacing_ok
            and power_ok
        ):
            continue

        F24, details = final_fidelity(
            candidate,
            offsets=offsets,
            M=M,
            T_gate=T_gate,
            n_steps=n_steps_final,
            target_label=target_label,
            signed_target_phase=(
                signed_target_phase
            ),
            r_disp=r_disp,
            omega_hw_max=(
                omega_hw_max
            ),
        )

        if F24 > best_F24:
            best_F24 = F24
            best_z = candidate.copy()
            best_details = details

    if best_z is None:
        raise RuntimeError(
            f"{name}: optimizer returned no "
            "feasible final candidate."
        )

    print(
        f"  final F_24       = "
        f"{best_F24:.10f}"
    )
    print(
        f"  improvement      = "
        f"{best_F24 - F0_final:+.3e}"
    )
    print(
        f"  evaluations      = "
        f"{objective.n_evaluations}"
    )
    print(
        f"  iterations       = "
        f"{result.nit}"
    )
    print(
        f"  wall seconds     = "
        f"{elapsed:.2f}"
    )
    print(
        f"  solver success   = "
        f"{result.success}"
    )
    print(
        f"  solver message   = "
        f"{result.message}"
    )
    print(
        f"  final power      = "
        f"{best_details['total_intensity']:.8e}"
    )
    print(
        f"  final spacing    = "
        f"{best_details['minimum_spacing']:.8e}"
    )
    print(
        f"  max disp ratio   = "
        f"{np.max(best_details['dispersive_ratios']):.6f}"
    )

    return {
        "name": name,
        "initial_F_search": float(
            F0_search
        ),
        "initial_F_final": float(
            F0_final
        ),
        "final_F_final": float(
            best_F24
        ),
        "improvement": float(
            best_F24 - F0_final
        ),
        "n_evaluations": int(
            objective.n_evaluations
        ),
        "n_iterations": int(
            result.nit
        ),
        "wall_seconds": float(
            elapsed
        ),
        "solver_success": bool(
            result.success
        ),
        "solver_status": int(
            result.status
        ),
        "solver_message": str(
            result.message
        ),
        "z": [
            float(v)
            for v in best_z
        ],
        "tones": [
            float(v)
            for v in best_details[
                "tones"
            ]
        ],
        "intensities": [
            float(v)
            for v in best_details[
                "intensities"
            ]
        ],
        "amplitudes": [
            float(v)
            for v in best_details[
                "amplitudes"
            ]
        ],
        "dispersive_ratios": [
            float(v)
            for v in best_details[
                "dispersive_ratios"
            ]
        ],
        "total_intensity": float(
            best_details[
                "total_intensity"
            ]
        ),
        "minimum_spacing": float(
            best_details[
                "minimum_spacing"
            ]
        ),
        "final_max_flip": float(
            best_details[
                "final_max_flip"
            ]
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
        "--num-random-starts",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260818,
    )

    parser.add_argument(
        "--search-steps-per-period",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--final-steps-per-period",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300000,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--omega-hw-max",
        type=float,
        default=float("inf"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmarks/"
            "direct_full_rwa_equal_time.json"
        ),
    )

    args = parser.parse_args()

    control = load_control(
        args.control_json
    )

    offsets = control["offsets"]
    compiler_tones = control["tones"]
    compiler_x = control["intensities"]

    M = len(
        compiler_tones
    )

    T_gate = control["T_gate"]
    span = control["span"]
    r_disp = control["r_disp"]
    total_intensity_max = (
        control["total_intensity_max"]
    )
    min_tone_spacing = (
        control["min_tone_spacing"]
    )

    signed_target_phase = (
        math.copysign(
            abs(
                control[
                    "target_phase"
                ]
            ),
            control[
                "target_sign"
            ],
        )
    )

    compiler_xmax, _ = (
        local_intensity_caps(
            compiler_tones,
            offsets,
            r_disp=r_disp,
            omega_hw_max=(
                args.omega_hw_max
            ),
        )
    )

    compiler_fraction = np.divide(
        compiler_x,
        compiler_xmax,
        out=np.zeros_like(
            compiler_x
        ),
        where=compiler_xmax > 0.0,
    )

    compiler_fraction = np.clip(
        compiler_fraction,
        0.0,
        1.0,
    )

    z_compiler = np.concatenate(
        [
            compiler_tones,
            compiler_fraction,
        ]
    )

    n_steps_search = (
        conservative_n_steps(
            offsets=offsets,
            span=span,
            total_intensity_max=(
                total_intensity_max
            ),
            T_gate=T_gate,
            steps_per_period=(
                args.search_steps_per_period
            ),
            max_steps=args.max_steps,
        )
    )

    n_steps_final = (
        conservative_n_steps(
            offsets=offsets,
            span=span,
            total_intensity_max=(
                total_intensity_max
            ),
            T_gate=T_gate,
            steps_per_period=(
                args.final_steps_per_period
            ),
            max_steps=args.max_steps,
        )
    )

    # Warm up Numba outside all benchmark timings.
    _ = (
        jit.propagate_final_blocks_jit(
            offsets,
            compiler_tones,
            compiler_x,
            T_gate,
            2,
        )
    )

    print()
    print(
        "[direct full-RWA equal-time benchmark]"
    )
    print(
        f"target              = "
        f"{control['target_label']}"
    )
    print(
        f"M                   = {M}"
    )
    print(
        f"T                   = "
        f"{T_gate:.10e}"
    )
    print(
        f"X_tot               = "
        f"{total_intensity_max:.8e}"
    )
    print(
        f"r_disp              = "
        f"{r_disp:.6f}"
    )
    print(
        f"min tone spacing    = "
        f"{min_tone_spacing:.6f}"
    )
    print(
        f"frequency window    = "
        f"[{-span:.3f}, {span:.3f}]"
    )
    print(
        f"search steps        = "
        f"{n_steps_search}"
    )
    print(
        f"final-check steps   = "
        f"{n_steps_final}"
    )

    runs = []

    runs.append(
        run_one_start(
            name="compiler",
            z0=z_compiler,
            offsets=offsets,
            M=M,
            T_gate=T_gate,
            n_steps_search=(
                n_steps_search
            ),
            n_steps_final=(
                n_steps_final
            ),
            target_label=(
                control[
                    "target_label"
                ]
            ),
            signed_target_phase=(
                signed_target_phase
            ),
            r_disp=r_disp,
            omega_hw_max=(
                args.omega_hw_max
            ),
            total_intensity_max=(
                total_intensity_max
            ),
            min_tone_spacing=(
                min_tone_spacing
            ),
            span=span,
            maxiter=args.maxiter,
            ftol=args.ftol,
        )
    )

    rng = np.random.default_rng(
        args.seed
    )

    for j in range(
        args.num_random_starts
    ):
        tones0 = (
            random_spaced_tones(
                rng,
                M=M,
                span=span,
                min_spacing=(
                    min_tone_spacing
                ),
            )
        )

        xmax0, _ = (
            local_intensity_caps(
                tones0,
                offsets,
                r_disp=r_disp,
                omega_hw_max=(
                    args.omega_hw_max
                ),
            )
        )

        fractions0 = (
            random_feasible_fractions(
                rng,
                xmax=xmax0,
                total_intensity_max=(
                    total_intensity_max
                ),
            )
        )

        z0 = np.concatenate(
            [
                tones0,
                fractions0,
            ]
        )

        runs.append(
            run_one_start(
                name=f"random_{j + 1:02d}",
                z0=z0,
                offsets=offsets,
                M=M,
                T_gate=T_gate,
                n_steps_search=(
                    n_steps_search
                ),
                n_steps_final=(
                    n_steps_final
                ),
                target_label=(
                    control[
                        "target_label"
                    ]
                ),
                signed_target_phase=(
                    signed_target_phase
                ),
                r_disp=r_disp,
                omega_hw_max=(
                    args.omega_hw_max
                ),
                total_intensity_max=(
                    total_intensity_max
                ),
                min_tone_spacing=(
                    min_tone_spacing
                ),
                span=span,
                maxiter=args.maxiter,
                ftol=args.ftol,
            )
        )

    best = max(
        runs,
        key=lambda row:
            row[
                "final_F_final"
            ],
    )

    compiler_run = runs[0]

    print()
    print("[summary]")
    print(
        "start          initial_F24    "
        "final_F24      dF          "
        "evals    seconds"
    )

    for row in runs:
        print(
            f"{row['name']:<13} "
            f"{row['initial_F_final']:.10f} "
            f"{row['final_F_final']:.10f} "
            f"{row['improvement']:+.3e} "
            f"{row['n_evaluations']:>6d} "
            f"{row['wall_seconds']:>9.2f}"
        )

    print()
    print(
        f"compiler initial F24 = "
        f"{compiler_run['initial_F_final']:.10f}"
    )
    print(
        f"compiler-polished F24= "
        f"{compiler_run['final_F_final']:.10f}"
    )
    print(
        f"best direct F24      = "
        f"{best['final_F_final']:.10f}"
    )
    print(
        f"best start           = "
        f"{best['name']}"
    )

    output = {
        "control_json": str(
            args.control_json
        ),
        "problem": {
            "target_label":
                control[
                    "target_label"
                ],
            "M": int(M),
            "T_gate": float(
                T_gate
            ),
            "span": float(
                span
            ),
            "r_disp": float(
                r_disp
            ),
            "total_intensity_max":
                float(
                    total_intensity_max
                ),
            "min_tone_spacing":
                float(
                    min_tone_spacing
                ),
            "signed_target_phase":
                float(
                    signed_target_phase
                ),
            "search_steps_per_period":
                int(
                    args.search_steps_per_period
                ),
            "final_steps_per_period":
                int(
                    args.final_steps_per_period
                ),
            "n_steps_search":
                int(
                    n_steps_search
                ),
            "n_steps_final":
                int(
                    n_steps_final
                ),
        },
        "runs": runs,
        "best_run": best,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"[output] "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
