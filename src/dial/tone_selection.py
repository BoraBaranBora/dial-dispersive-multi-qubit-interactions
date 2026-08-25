from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix

from . import full_dynamics as stage04


def load_problem(path: Path) -> dict:
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    inputs = data["inputs"]

    offsets = np.asarray(
        inputs["offsets"],
        dtype=float,
    )

    return {
        "data": data,
        "offsets": offsets,
        "target": str(inputs["target"]),
        "target_phase": float(
            inputs["target_phase"]
        ),
        "span": float(inputs["span"]),
        "r_disp": float(inputs["r_disp"]),
        "min_spacing": float(
            inputs["min_tone_spacing"]
        ),
        "target_sign": float(
            data.get(
                "coarse_selection",
                {},
            ).get(
                "target_sign",
                -1.0,
            )
        ),
    }


def spectral_spacing(
    offsets: np.ndarray,
) -> float:
    values = np.sort(
        np.unique(offsets)
    )

    gaps = np.diff(values)
    gaps = gaps[gaps > 1e-12]

    if len(gaps) == 0:
        raise ValueError(
            "No nonzero spectral spacing."
        )

    return float(np.min(gaps))


def local_caps(
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

    if math.isfinite(
        omega_hw_max
    ):
        omega_caps = np.minimum(
            omega_caps,
            omega_hw_max,
        )

    xmax = 0.5 * omega_caps**2

    return xmax, detunings


def build_candidate_grid(
    *,
    span: float,
    grid_points: int,
    offsets: np.ndarray,
) -> np.ndarray:
    tones = np.linspace(
        -span,
        span,
        grid_points,
    )

    # Remove only exact/numerically singular resonances.
    d = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    tones = tones[
        d > 1e-10
    ]

    return tones


def solve_spacing_milp(
    *,
    offsets: np.ndarray,
    candidate_tones: np.ndarray,
    target: str,
    target_sign: float,
    max_tones: int,
    r_disp: float,
    min_spacing: float,
    omega_hw_max: float,
    time_limit: float,
) -> dict:
    labels, G = (
        stage04.response_matrix(
            offsets,
            candidate_tones,
            include_identity=False,
        )
    )

    labels = list(labels)

    if target not in labels:
        raise ValueError(
            f"Target {target!r} "
            f"not in {labels}"
        )

    target_index = labels.index(
        target
    )

    xmax, detunings = local_caps(
        candidate_tones,
        offsets,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
    )

    # H maps normalized intensity fractions y_k in [0,1]
    # directly to rates.
    H = (
        G
        * xmax[None, :]
    )

    scale = float(
        np.max(
            np.abs(H)
        )
    )

    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            "Invalid response scaling."
        )

    Hs = H / scale

    K = len(candidate_tones)
    n_channels = Hs.shape[0]

    # Variables:
    #   y[0:K]       continuous intensity fractions
    #   z[K:2K]      binary tone selectors
    #   q            nonnegative scaled target rate
    nvar = 2 * K + 1
    q_index = 2 * K

    c = np.zeros(
        nvar,
        dtype=float,
    )
    c[q_index] = -1.0

    # Equality:
    #
    #   H y = sign * kappa * e_target
    #
    # with q = kappa / scale.
    rows = []
    cols = []
    vals = []

    for i in range(n_channels):
        for k in range(K):
            value = Hs[i, k]

            if value != 0.0:
                rows.append(i)
                cols.append(k)
                vals.append(value)

    rows.append(target_index)
    cols.append(q_index)
    vals.append(-target_sign)

    Aeq = coo_matrix(
        (
            vals,
            (rows, cols),
        ),
        shape=(
            n_channels,
            nvar,
        ),
    ).tocsr()

    eq_constraint = LinearConstraint(
        Aeq,
        np.zeros(n_channels),
        np.zeros(n_channels),
    )

    # Inequalities:
    #
    #   y_k <= z_k
    #   sum z_k <= M
    #   z_i + z_j <= 1 for conflicting frequencies.
    ub_rows = []
    ub_cols = []
    ub_vals = []
    upper = []

    row = 0

    for k in range(K):
        ub_rows.extend(
            [row, row]
        )
        ub_cols.extend(
            [k, K + k]
        )
        ub_vals.extend(
            [1.0, -1.0]
        )
        upper.append(0.0)
        row += 1

    for k in range(K):
        ub_rows.append(row)
        ub_cols.append(K + k)
        ub_vals.append(1.0)

    upper.append(float(max_tones))
    row += 1

    # Candidate tones are sorted, so only inspect nearby pairs.
    for i in range(K):
        j = i + 1

        while (
            j < K
            and candidate_tones[j]
            - candidate_tones[i]
            < min_spacing - 1e-12
        ):
            ub_rows.extend(
                [row, row]
            )
            ub_cols.extend(
                [K + i, K + j]
            )
            ub_vals.extend(
                [1.0, 1.0]
            )
            upper.append(1.0)

            row += 1
            j += 1

    Aub = coo_matrix(
        (
            ub_vals,
            (
                ub_rows,
                ub_cols,
            ),
        ),
        shape=(
            row,
            nvar,
        ),
    ).tocsr()

    ineq_constraint = (
        LinearConstraint(
            Aub,
            -np.inf,
            np.asarray(
                upper,
                dtype=float,
            ),
        )
    )

    q_upper = max(
        1.0,
        float(
            np.sum(
                np.abs(
                    Hs[
                        target_index,
                        :
                    ]
                )
            )
        ),
    )

    lb = np.zeros(
        nvar,
        dtype=float,
    )

    ub = np.ones(
        nvar,
        dtype=float,
    )
    ub[q_index] = q_upper

    integrality = np.zeros(
        nvar,
        dtype=int,
    )
    integrality[K:2 * K] = 1

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=[
            eq_constraint,
            ineq_constraint,
        ],
        options={
            "time_limit":
                float(time_limit),
            "mip_rel_gap":
                1e-8,
            "disp":
                False,
        },
    )

    if result.x is None:
        raise RuntimeError(
            "MILP returned no feasible solution. "
            f"status={result.status}, "
            f"message={result.message}"
        )

    y = np.asarray(
        result.x[:K],
        dtype=float,
    )
    z = np.asarray(
        result.x[K:2 * K],
        dtype=float,
    )

    selected = np.where(
        z > 0.5
    )[0]

    tones = (
        candidate_tones[selected]
    )
    intensities = (
        xmax[selected]
        * y[selected]
    )

    kappa = (
        scale
        * float(
            result.x[q_index]
        )
    )

    return {
        "labels": labels,
        "target_index":
            target_index,
        "tones": tones,
        "intensities":
            intensities,
        "kappa": float(kappa),
        "status":
            int(result.status),
        "success":
            bool(result.success),
        "message":
            str(result.message),
        "mip_gap":
            float(
                getattr(
                    result,
                    "mip_gap",
                    float("nan"),
                )
            ),
        "mip_node_count":
            int(
                getattr(
                    result,
                    "mip_node_count",
                    -1,
                )
            ),
    }


def solve_fixed_tones(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    target: str,
    target_sign: float,
    r_disp: float,
    omega_hw_max: float,
) -> dict | None:
    labels, G = (
        stage04.response_matrix(
            offsets,
            tones,
            include_identity=False,
        )
    )

    labels = list(labels)
    target_index = labels.index(
        target
    )

    xmax, detunings = local_caps(
        tones,
        offsets,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
    )

    H = (
        G
        * xmax[None, :]
    )

    scale = float(
        np.max(
            np.abs(H)
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        return None

    Hs = H / scale

    M = len(tones)

    # Variables y_1...y_M,q.
    c = np.zeros(
        M + 1,
        dtype=float,
    )
    c[-1] = -1.0

    Aeq = np.zeros(
        (
            len(labels),
            M + 1,
        ),
        dtype=float,
    )

    Aeq[:, :M] = Hs
    Aeq[
        target_index,
        -1,
    ] = -target_sign

    q_upper = max(
        1.0,
        float(
            np.sum(
                np.abs(
                    Hs[
                        target_index,
                        :
                    ]
                )
            )
        ),
    )

    bounds = (
        [(0.0, 1.0)] * M
        + [(0.0, q_upper)]
    )

    result = linprog(
        c,
        A_eq=Aeq,
        b_eq=np.zeros(
            len(labels)
        ),
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return None

    y = np.asarray(
        result.x[:M],
        dtype=float,
    )

    intensities = (
        xmax * y
    )

    K_rates = G @ intensities

    return {
        "tones":
            tones.copy(),
        "intensities":
            intensities,
        "xmax":
            xmax,
        "detunings":
            detunings,
        "kappa":
            float(
                scale
                * result.x[-1]
            ),
        "rates":
            K_rates,
        "labels":
            labels,
        "target_index":
            target_index,
    }


def refine_frequencies(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    target: str,
    target_sign: float,
    r_disp: float,
    omega_hw_max: float,
    span: float,
    min_spacing: float,
    passes: int,
    points: int,
) -> dict:
    tones = np.sort(
        np.asarray(
            tones,
            dtype=float,
        )
    )

    best = solve_fixed_tones(
        offsets=offsets,
        tones=tones,
        target=target,
        target_sign=target_sign,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
    )

    if best is None:
        raise RuntimeError(
            "Initial fixed-tone LP failed."
        )

    for _ in range(passes):
        improved = False

        for j in range(
            len(tones)
        ):
            lo = -span
            hi = +span

            if j > 0:
                lo = max(
                    lo,
                    tones[j - 1]
                    + min_spacing,
                )

            if j + 1 < len(tones):
                hi = min(
                    hi,
                    tones[j + 1]
                    - min_spacing,
                )

            if hi <= lo:
                continue

            trial_values = np.linspace(
                lo,
                hi,
                points,
            )

            local_best = best
            local_tones = tones.copy()

            for omega in trial_values:
                trial = tones.copy()
                trial[j] = float(
                    omega
                )

                # Avoid exact resonances only.
                if (
                    np.min(
                        np.abs(
                            trial[j]
                            - offsets
                        )
                    )
                    <= 1e-10
                ):
                    continue

                solved = solve_fixed_tones(
                    offsets=offsets,
                    tones=trial,
                    target=target,
                    target_sign=target_sign,
                    r_disp=r_disp,
                    omega_hw_max=omega_hw_max,
                )

                if solved is None:
                    continue

                if (
                    solved["kappa"]
                    > local_best["kappa"]
                    * (
                        1.0
                        + 1e-10
                    )
                ):
                    local_best = solved
                    local_tones = (
                        trial.copy()
                    )

            if (
                local_best["kappa"]
                > best["kappa"]
                * (
                    1.0
                    + 1e-10
                )
            ):
                tones = local_tones
                best = local_best
                improved = True

        if not improved:
            break

    return best


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--control-json",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-tones",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--r-disp",
        type=float,
        default=None,
        help=(
            "Override the dispersive ratio stored "
            "in the source control JSON."
        ),
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=401,
    )

    parser.add_argument(
        "--refine-passes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--refine-points",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--time-limit",
        type=float,
        default=180.0,
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
            "minimum_gate_duration.json"
        ),
    )

    args = parser.parse_args()

    problem = load_problem(
        args.control_json
    )

    offsets = problem["offsets"]

    if args.r_disp is not None:
        problem["r_disp"] = float(
            args.r_disp
        )

    delta_min = spectral_spacing(
        offsets
    )

    candidates = build_candidate_grid(
        span=problem["span"],
        grid_points=(
            args.grid_points
        ),
        offsets=offsets,
    )

    print()
    print(
        "[minimum gate-duration diagnostic]"
    )
    print(
        f"target             = "
        f"{problem['target']}"
    )
    print(
        f"target phase       = "
        f"{problem['target_phase']:.10f}"
    )
    print(
        f"target sign        = "
        f"{problem['target_sign']:+.0f}"
    )
    print(
        f"M max              = "
        f"{args.max_tones}"
    )
    print(
        f"r_disp             = "
        f"{problem['r_disp']:.6f}"
    )
    print(
        f"frequency window   = "
        f"[{-problem['span']:.3f}, "
        f"{problem['span']:.3f}]"
    )
    print(
        f"min tone spacing   = "
        f"{problem['min_spacing']:.6f}"
    )
    print(
        f"candidate tones    = "
        f"{len(candidates)}"
    )
    print(
        f"Delta_min          = "
        f"{delta_min:.10e}"
    )
    print()
    print(
        "No global total-intensity cap "
        "is imposed in this diagnostic."
    )

    print()
    print("[1/2] grid-global MILP")

    coarse = solve_spacing_milp(
        offsets=offsets,
        candidate_tones=candidates,
        target=problem["target"],
        target_sign=(
            problem["target_sign"]
        ),
        max_tones=args.max_tones,
        r_disp=problem["r_disp"],
        min_spacing=(
            problem["min_spacing"]
        ),
        omega_hw_max=(
            args.omega_hw_max
        ),
        time_limit=args.time_limit,
    )

    print(
        f"status             = "
        f"{coarse['status']}"
    )
    print(
        f"success            = "
        f"{coarse['success']}"
    )
    print(
        f"message            = "
        f"{coarse['message']}"
    )
    print(
        f"MIP gap            = "
        f"{coarse['mip_gap']:.3e}"
    )
    print(
        f"selected tones     = "
        f"{len(coarse['tones'])}"
    )
    print(
        f"kappa_grid         = "
        f"{coarse['kappa']:.10e}"
    )

    print()
    print(
        "[2/2] continuous coordinate refinement"
    )

    refined = refine_frequencies(
        offsets=offsets,
        tones=coarse["tones"],
        target=problem["target"],
        target_sign=(
            problem["target_sign"]
        ),
        r_disp=problem["r_disp"],
        omega_hw_max=(
            args.omega_hw_max
        ),
        span=problem["span"],
        min_spacing=(
            problem["min_spacing"]
        ),
        passes=args.refine_passes,
        points=args.refine_points,
    )

    kappa = float(
        refined["kappa"]
    )

    T_min = (
        2.0
        * abs(
            problem["target_phase"]
        )
        / kappa
    )

    N_sys = (
        T_min
        * delta_min
        / (
            2.0
            * math.pi
        )
    )

    intensities = np.asarray(
        refined["intensities"],
        dtype=float,
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    omega_max = float(
        np.max(amplitudes)
    )

    N_rabi = (
        T_min
        * omega_max
        / (
            2.0
            * math.pi
        )
    )

    ratios = np.divide(
        amplitudes,
        refined["detunings"],
        out=np.zeros_like(
            amplitudes
        ),
        where=(
            refined["detunings"]
            > 0.0
        ),
    )

    target_index = (
        refined["target_index"]
    )

    rates = np.asarray(
        refined["rates"],
        dtype=float,
    )

    spectator = np.delete(
        rates,
        target_index,
    )

    spectator_rms = float(
        np.sqrt(
            np.mean(
                spectator**2
            )
        )
    )

    print()
    print("[result]")
    print(
        f"kappa_refined      = "
        f"{kappa:.10e}"
    )
    print(
        f"grid improvement   = "
        f"{100.0 * (kappa / coarse['kappa'] - 1.0):.3f}%"
    )
    print(
        f"T_min effective    = "
        f"{T_min:.10e}"
    )
    print(
        f"N_sys,min          = "
        f"{N_sys:.6f}"
    )
    print(
        f"N_Rabi,max-tone    = "
        f"{N_rabi:.6f}"
    )
    print(
        f"total intensity    = "
        f"{np.sum(intensities):.10e}"
    )
    print(
        f"max amplitude      = "
        f"{np.max(amplitudes):.10e}"
    )
    print(
        f"max disp ratio     = "
        f"{np.max(ratios):.6f}"
    )
    print(
        f"spectator rate RMS = "
        f"{spectator_rms:.3e}"
    )

    print()
    print("[reference requirements]")

    requirements = {}

    for N in [6.0, 10.0, 20.0]:
        T_required = (
            2.0
            * math.pi
            * N
            / delta_min
        )

        kappa_required = (
            2.0
            * abs(
                problem[
                    "target_phase"
                ]
            )
            / T_required
        )

        ratio = (
            kappa / kappa_required
        )

        requirements[str(int(N))] = {
            "T": float(
                T_required
            ),
            "kappa_required":
                float(
                    kappa_required
                ),
            "achieved_over_required":
                float(ratio),
        }

        print(
            f"N_sys={N:>4.0f}: "
            f"T={T_required:10.3f} "
            f"kappa_req="
            f"{kappa_required:.6e} "
            f"kappa_max/kappa_req="
            f"{ratio:.4f}"
        )

    print()
    print("[tones]")

    tone_rows = []

    for k, (
        omega,
        x,
        Omega,
        det,
        ratio,
    ) in enumerate(
        zip(
            refined["tones"],
            intensities,
            amplitudes,
            refined["detunings"],
            ratios,
        ),
        start=1,
    ):
        print(
            f"{k:>2d} "
            f"omega={omega:+.10f} "
            f"x={x:.8e} "
            f"Omega={Omega:.8e} "
            f"d={det:.8e} "
            f"Omega/d={ratio:.6f}"
        )

        tone_rows.append(
            {
                "omega":
                    float(omega),
                "intensity":
                    float(x),
                "amplitude":
                    float(Omega),
                "nearest_detuning":
                    float(det),
                "dispersive_ratio":
                    float(ratio),
            }
        )

    output = {
        "source_control":
            str(
                args.control_json
            ),
        "constraints": {
            "max_tones":
                int(
                    args.max_tones
                ),
            "span":
                float(
                    problem[
                        "span"
                    ]
                ),
            "r_disp":
                float(
                    problem[
                        "r_disp"
                    ]
                ),
            "min_tone_spacing":
                float(
                    problem[
                        "min_spacing"
                    ]
                ),
            "omega_hw_max":
                (
                    None
                    if not math.isfinite(
                        args.omega_hw_max
                    )
                    else float(
                        args.omega_hw_max
                    )
                ),
            "global_total_intensity_cap":
                None,
        },
        "spectrum": {
            "offsets": [
                float(v)
                for v in offsets
            ],
            "delta_min":
                float(
                    delta_min
                ),
        },
        "target": {
            "label":
                problem["target"],
            "phase":
                float(
                    problem[
                        "target_phase"
                    ]
                ),
            "sign":
                float(
                    problem[
                        "target_sign"
                    ]
                ),
        },
        "coarse": {
            "kappa":
                float(
                    coarse[
                        "kappa"
                    ]
                ),
            "status":
                int(
                    coarse[
                        "status"
                    ]
                ),
            "success":
                bool(
                    coarse[
                        "success"
                    ]
                ),
            "mip_gap":
                float(
                    coarse[
                        "mip_gap"
                    ]
                ),
        },
        "refined": {
            "kappa":
                float(
                    kappa
                ),
            "T_effective_min":
                float(
                    T_min
                ),
            "N_sys_min":
                float(
                    N_sys
                ),
            "N_rabi_max_tone":
                float(
                    N_rabi
                ),
            "total_intensity":
                float(
                    np.sum(
                        intensities
                    )
                ),
            "max_amplitude":
                float(
                    np.max(
                        amplitudes
                    )
                ),
            "max_dispersive_ratio":
                float(
                    np.max(
                        ratios
                    )
                ),
            "spectator_rate_rms":
                float(
                    spectator_rms
                ),
            "tones":
                tone_rows,
        },
        "reference_requirements":
            requirements,
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
