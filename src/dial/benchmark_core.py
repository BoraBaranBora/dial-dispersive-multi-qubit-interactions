from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import synthesis as inv
from . import control_encoding as comparison1
from . import full_rwa_metrics as base


DEFAULT_SEED = 20260818
DEFAULT_PHASE = math.pi / 4.0


# ----------------------------------------------------------------------
# JSON / checkpointing
# ----------------------------------------------------------------------


def jsonable(value):
    if isinstance(value, dict):
        return {
            str(k): jsonable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            jsonable(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, Path):
        return str(value)

    return value


def atomic_write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            jsonable(payload),
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(tmp, path)


# ----------------------------------------------------------------------
# n-spin benchmark ensemble
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# Full-body target label
# ----------------------------------------------------------------------


def resolve_full_z_target(
    labels,
    n: int,
):
    """
    Identify the label corresponding to Z_1 ... Z_n without
    assuming one particular string convention.
    """
    labels = list(labels)

    candidates = [
        "".join(
            f"Z{i}"
            for i in range(1, n + 1)
        ),
        "Z" * n,
        "".join(
            f"Z_{i}"
            for i in range(1, n + 1)
        ),
    ]

    for candidate in candidates:
        if candidate in labels:
            return candidate

    # Most label conventions contain one Z symbol per supported
    # spin. The n-body Z string should therefore be unique.
    z_count_matches = [
        label
        for label in labels
        if str(label).count("Z") == n
    ]

    if len(z_count_matches) == 1:
        return z_count_matches[0]

    # Additional fallback for labels whose support is represented
    # primarily by explicit spin indices.
    support_matches = []

    for label in labels:
        numbers = {
            int(x)
            for x in re.findall(
                r"\d+",
                str(label),
            )
        }

        if numbers == set(
            range(1, n + 1)
        ):
            support_matches.append(
                label
            )

    if len(support_matches) == 1:
        return support_matches[0]

    raise KeyError(
        "Could not uniquely identify "
        f"the full n-body Z target "
        f"for n={n}.\n"
        f"Available labels: {labels}"
    )


# ----------------------------------------------------------------------
# n-aware finite-tone inversion
# ----------------------------------------------------------------------


def finite_tone_control_inversion_n(
    *,
    offsets: np.ndarray,
    n: int,
    max_tones: int,
    min_tone_spacing: float,
    span: float,
    grid_points: int,
    candidate_pool: int,
    min_detuning: float,
) -> dict:
    """
    n-spin extension of the existing finite-tone construction.

    The procedure itself is unchanged:

        admissible tone pool
        -> zero-response filtering
        -> deterministic downsampling
        -> greedy joint tone selection
        -> signed NNLS.

    Only the response-matrix dimension n is generalized.
    """
    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    tones = inv.candidate_tones(
        offsets,
        num_grid=grid_points,
        span=span,
        min_detuning=min_detuning,
    )

    labels, G_pool = (
        inv.response_matrix(
            offsets,
            tones,
            n=n,
            include_identity=False,
        )
    )

    target = resolve_full_z_target(
        labels,
        n,
    )

    tones, G_pool = (
        inv.filter_nonzero_response_columns(
            tones,
            G_pool,
        )
    )

    tones_ds, G_ds = (
        inv.downsample_candidates(
            tones,
            G_pool,
            candidate_pool=candidate_pool,
        )
    )

    target_index = labels.index(
        target
    )

    (
        selected_indices,
        x_unit,
        metrics,
    ) = inv.greedy_sparse_target_synthesis(
        tones_ds,
        G_ds,
        target_index=target_index,
        max_tones=max_tones,
        min_tone_spacing=min_tone_spacing,
    )

    if not selected_indices:
        raise RuntimeError(
            "Finite-tone inversion "
            "selected no tones."
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    selected_tones = np.asarray(
        tones_ds[selected_indices],
        dtype=float,
    )

    x_unit = np.asarray(
        x_unit,
        dtype=float,
    )

    # Match the existing benchmark:
    # zero-NNLS-intensity tones are not physical control tones.
    active = x_unit > 1e-10

    selected_indices = (
        selected_indices[active]
    )

    selected_tones = (
        selected_tones[active]
    )

    x_unit = x_unit[active]

    if len(selected_tones) == 0:
        raise RuntimeError(
            "All selected tones received "
            "zero final NNLS intensity."
        )

    G_selected = G_ds[
        :,
        selected_indices,
    ]

    K_unit = (
        G_selected @ x_unit
    )

    target_rate_unit = float(
        K_unit[target_index]
    )

    if abs(target_rate_unit) <= 1e-15:
        raise RuntimeError(
            "Inversion produced zero "
            "target interaction rate."
        )

    return {
        "n": int(n),
        "labels": list(labels),
        "target": target,
        "target_index":
            int(target_index),
        "target_sign":
            float(metrics["target_sign"]),
        "tones": selected_tones,
        "intensities_unit": x_unit,
        "target_rate_unit":
            target_rate_unit,
        "relative_residual":
            float(
                metrics[
                    "relative_residual"
                ]
            ),
        "epsilon_spec":
            float(
                metrics[
                    "epsilon_spec"
                ]
            ),
        "total_intensity_unit":
            float(
                np.sum(x_unit)
            ),
        "active_count":
            int(len(selected_tones)),
        "K_unit": K_unit,
        "candidate_tones":
            np.asarray(
                tones_ds,
                dtype=float,
            ),
        "candidate_pool_size":
            int(len(tones_ds)),
    }


# ----------------------------------------------------------------------
# Tone budget
# ----------------------------------------------------------------------


def tone_budget_for_n(
    *,
    n: int,
    mode: str,
    fixed_max_tones: int,
    tone_cap: int | None,
) -> int:
    if mode == "channels":
        budget = 2**n - 1

    elif mode == "fixed":
        budget = int(
            fixed_max_tones
        )

    else:
        raise ValueError(
            f"Unknown tone-budget mode "
            f"{mode!r}."
        )

    if tone_cap is not None:
        budget = min(
            budget,
            int(tone_cap),
        )

    return max(
        1,
        int(budget),
    )


# ----------------------------------------------------------------------
# One realization
# ----------------------------------------------------------------------


def run_realization(
    *,
    realization: dict,
    args: argparse.Namespace,
    do_warmup: bool,
) -> dict:
    t_case = time.perf_counter()

    n = int(
        realization["n"]
    )

    offsets = np.asarray(
        realization["offsets"],
        dtype=float,
    )

    max_tones = tone_budget_for_n(
        n=n,
        mode=args.tone_budget_mode,
        fixed_max_tones=(
            args.fixed_max_tones
        ),
        tone_cap=args.tone_cap,
    )

    inversion = (
        finite_tone_control_inversion_n(
            offsets=offsets,
            n=n,
            max_tones=max_tones,
            min_tone_spacing=(
                args.min_tone_spacing
            ),
            span=args.span,
            grid_points=(
                args.grid_points
            ),
            candidate_pool=(
                args.candidate_pool
            ),
            min_detuning=(
                args.min_detuning
            ),
        )
    )

    tones = np.asarray(
        inversion["tones"],
        dtype=float,
    )

    x_unit = np.asarray(
        inversion[
            "intensities_unit"
        ],
        dtype=float,
    )

    scaling = (
        inv.dispersive_scale_from_unit_control(
            tones=tones,
            intensities_unit=x_unit,
            offsets=offsets,
            r_disp=args.r_disp,
        )
    )

    intensities = np.asarray(
        scaling["intensities"],
        dtype=float,
    )

    drive_scale = float(
        scaling["drive_scale"]
    )

    target_rate = (
        drive_scale
        * float(
            inversion[
                "target_rate_unit"
            ]
        )
    )

    if abs(target_rate) <= 1e-15:
        raise RuntimeError(
            "Physical target interaction "
            "rate is zero."
        )

    signed_target_phase = (
        math.copysign(
            abs(args.target_phase),
            target_rate,
        )
    )

    T_gate = (
        2.0
        * abs(args.target_phase)
        / abs(target_rate)
    )

    total_intensity_max = float(
        np.sum(intensities)
    )

    M = int(
        len(tones)
    )

    target = str(
        inversion["target"]
    )

    z_inv = comparison1.control_to_z(
        tones=tones,
        intensities=intensities,
        offsets=offsets,
        r_disp=args.r_disp,
        omega_hw_max=(
            args.omega_hw_max
        ),
    )

    n_steps_search = (
        base.conservative_n_steps(
            offsets=offsets,
            span=args.span,
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
        base.conservative_n_steps(
            offsets=offsets,
            span=args.span,
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

    # Keep JIT compilation outside the optimizer wall-time measurement.
    if do_warmup:
        _ = (
            base.jit.propagate_final_blocks_jit(
                offsets,
                tones,
                intensities,
                T_gate,
                2,
            )
        )

    run = base.run_one_start(
        name="inversion_seed",
        z0=z_inv,
        offsets=offsets,
        M=M,
        T_gate=T_gate,
        n_steps_search=n_steps_search,
        n_steps_final=n_steps_final,
        target_label=target,
        signed_target_phase=(
            signed_target_phase
        ),
        r_disp=args.r_disp,
        omega_hw_max=(
            args.omega_hw_max
        ),
        total_intensity_max=(
            total_intensity_max
        ),
        min_tone_spacing=(
            args.min_tone_spacing
        ),
        span=args.span,
        maxiter=args.maxiter,
        ftol=args.ftol,
    )

    elapsed = (
        time.perf_counter()
        - t_case
    )

    return {
        "n": n,
        "realization_index":
            int(
                realization[
                    "realization_index"
                ]
            ),
        "A": realization["A"],
        "offsets": offsets,
        "minimum_transition_spacing":
            float(
                realization[
                    "minimum_transition_spacing"
                ]
            ),
        "target": target,
        "target_phase_magnitude":
            abs(args.target_phase),
        "selected_target_sign":
            float(
                inversion[
                    "target_sign"
                ]
            ),
        "signed_target_phase":
            signed_target_phase,
        "tone_budget":
            int(max_tones),
        "active_tone_count":
            M,
        "gate_time":
            float(T_gate),
        "r_disp_target":
            float(args.r_disp),
        "r_disp_actual":
            float(
                scaling[
                    "max_dispersive_ratio"
                ]
            ),
        "total_intensity":
            total_intensity_max,
        "inversion": {
            **inversion,
            "drive_scale":
                drive_scale,
            "physical_intensities":
                intensities,
        },

        # The initial final-grid fidelity is precisely the
        # inversion control evaluated before optimization.
        "direct_inversion_fidelity":
            float(
                run[
                    "initial_F_final"
                ]
            ),

        "optimized_fidelity":
            float(
                run[
                    "final_F_final"
                ]
            ),

        "optimization_improvement":
            float(
                run[
                    "final_F_final"
                ]
                - run[
                    "initial_F_final"
                ]
            ),

        "optimizer": {
            "n_evaluations":
                int(
                    run[
                        "n_evaluations"
                    ]
                ),
            "n_iterations":
                int(
                    run[
                        "n_iterations"
                    ]
                ),
            "wall_seconds":
                float(
                    run[
                        "wall_seconds"
                    ]
                ),
            "solver_success":
                bool(
                    run[
                        "solver_success"
                    ]
                ),
            "solver_status":
                int(
                    run[
                        "solver_status"
                    ]
                ),
            "solver_message":
                str(
                    run[
                        "solver_message"
                    ]
                ),
        },

        # Important: this is the FINAL mediator flip returned by
        # the optimizer, not the maximum transient excitation.
        "optimized_final_max_flip":
            float(
                run[
                    "final_max_flip"
                ]
            ),

        "optimized_control": {
            "tones":
                run["tones"],
            "intensities":
                run["intensities"],
            "z":
                run["z"],
        },

        "n_steps_search":
            int(n_steps_search),
        "n_steps_final":
            int(n_steps_final),
        "case_elapsed_seconds":
            float(elapsed),

        # Retain the complete native result for reproducibility.
        "run_one_start":
            run,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scaling benchmark for inversion-seeded "
            "full-RWA optimization of Z_1...Z_n targets."
        )
    )

    parser.add_argument(
        "--n-min",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--n-max",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--n-systems",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--ratio-max",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--sobol-power",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--target-phase",
        type=float,
        default=DEFAULT_PHASE,
    )

    parser.add_argument(
        "--tone-budget-mode",
        choices=(
            "channels",
            "fixed",
        ),
        default="channels",
        help=(
            "'channels' uses 2^n-1 as the maximum "
            "tone budget, matching M=7 at n=3. "
            "'fixed' uses --fixed-max-tones."
        ),
    )

    parser.add_argument(
        "--fixed-max-tones",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--tone-cap",
        type=int,
        default=None,
        help=(
            "Optional upper cap on the tone budget "
            "for exploratory scans."
        ),
    )

    parser.add_argument(
        "--span",
        type=float,
        default=1.35,
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=401,
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=151,
    )

    parser.add_argument(
        "--min-detuning",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--min-tone-spacing",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--r-disp",
        type=float,
        default=0.10,
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
            "nbody_inversion_seeded_scaling.json"
        ),
    )

    args = parser.parse_args()

    if args.n_min < 2:
        raise ValueError(
            "--n-min must be >= 2."
        )

    if args.n_max < args.n_min:
        raise ValueError(
            "--n-max must be >= --n-min."
        )

    # Fail early if the expected reuse points are absent.
    required = [
        (
            inv,
            "candidate_tones",
        ),
        (
            inv,
            "response_matrix",
        ),
        (
            inv,
            "filter_nonzero_response_columns",
        ),
        (
            inv,
            "downsample_candidates",
        ),
        (
            inv,
            "greedy_sparse_target_synthesis",
        ),
        (
            inv,
            "dispersive_scale_from_unit_control",
        ),
        (
            comparison1,
            "control_to_z",
        ),
        (
            base,
            "conservative_n_steps",
        ),
        (
            base,
            "run_one_start",
        ),
    ]

    missing = [
        f"{module.__name__}.{name}"
        for module, name in required
        if not hasattr(
            module,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "Missing expected existing "
            "benchmark helpers:\n"
            + "\n".join(missing)
        )

    config = {
        "kind":
            "nbody_inversion_seeded_scaling",
        "n_min": args.n_min,
        "n_max": args.n_max,
        "n_systems": args.n_systems,
        "seed": args.seed,
        "ratio_max":
            args.ratio_max,
        "sobol_power":
            args.sobol_power,
        "target_phase":
            args.target_phase,
        "tone_budget_mode":
            args.tone_budget_mode,
        "fixed_max_tones":
            args.fixed_max_tones,
        "tone_cap":
            args.tone_cap,
        "span": args.span,
        "grid_points":
            args.grid_points,
        "candidate_pool":
            args.candidate_pool,
        "min_detuning":
            args.min_detuning,
        "min_tone_spacing":
            args.min_tone_spacing,
        "r_disp":
            args.r_disp,
        "search_steps_per_period":
            args.search_steps_per_period,
        "final_steps_per_period":
            args.final_steps_per_period,
        "max_steps":
            args.max_steps,
        "maxiter":
            args.maxiter,
        "ftol":
            args.ftol,
        "omega_hw_max":
            args.omega_hw_max,
    }

    if args.output.exists():
        output = json.loads(
            args.output.read_text(
                encoding="utf-8"
            )
        )

        if output.get(
            "config"
        ) != jsonable(config):
            raise RuntimeError(
                "Existing output has a "
                "different configuration. "
                "Use a new --output path."
            )

        print(
            f"[resume] {args.output}",
            flush=True,
        )

    else:
        output = {
            "schema_version": 1,
            "config": config,
            "cases": {},
        }

        atomic_write_json(
            args.output,
            output,
        )

    warmed_n = set()

    total_expected = (
        (args.n_max - args.n_min + 1)
        * args.n_systems
    )

    completed_before = sum(
        1
        for row in output[
            "cases"
        ].values()
        if not row.get(
            "failed",
            False,
        )
    )

    print()
    print(
        "Inversion-seeded n-body "
        "scaling benchmark"
    )
    print(
        f"  n range       : "
        f"{args.n_min}..{args.n_max}"
    )
    print(
        f"  systems / n   : "
        f"{args.n_systems}"
    )
    print(
        f"  total cases   : "
        f"{total_expected}"
    )
    print(
        f"  already done  : "
        f"{completed_before}"
    )
    print(
        f"  tone budget   : "
        f"{args.tone_budget_mode}"
    )
    print(
        f"  output        : "
        f"{args.output}"
    )

    for n in range(
        args.n_min,
        args.n_max + 1,
    ):
        realizations = (
            sample_realizations(
                n=n,
                n_systems=(
                    args.n_systems
                ),
                seed=args.seed,
                ratio_max=(
                    args.ratio_max
                ),
                sobol_power=(
                    args.sobol_power
                ),
            )
        )

        print()
        print(
            "=" * 72
        )
        print(
            f"n={n}: target "
            f"Z_1...Z_{n}"
        )
        print(
            "=" * 72
        )

        for realization in realizations:
            idx = int(
                realization[
                    "realization_index"
                ]
            )

            case_id = (
                f"n{n}_"
                f"realization_{idx:03d}"
            )

            old = output[
                "cases"
            ].get(case_id)

            if (
                old is not None
                and not old.get(
                    "failed",
                    False,
                )
            ):
                print(
                    f"[skip] {case_id}",
                    flush=True,
                )
                continue

            print()
            print(
                f"[run] {case_id}",
                flush=True,
            )
            print(
                f"  A = "
                f"{np.asarray(realization['A'])}",
                flush=True,
            )

            try:
                row = run_realization(
                    realization=realization,
                    args=args,
                    do_warmup=(
                        n not in warmed_n
                    ),
                )

                warmed_n.add(n)

                output[
                    "cases"
                ][case_id] = row

                atomic_write_json(
                    args.output,
                    output,
                )

                print(
                    f"  F_inv = "
                    f"{row['direct_inversion_fidelity']:.9f}",
                    flush=True,
                )
                print(
                    f"  F_opt = "
                    f"{row['optimized_fidelity']:.9f}",
                    flush=True,
                )
                print(
                    f"  dF    = "
                    f"{row['optimization_improvement']:+.3e}",
                    flush=True,
                )
                print(
                    f"  tones = "
                    f"{row['active_tone_count']}",
                    flush=True,
                )
                print(
                    f"  evals = "
                    f"{row['optimizer']['n_evaluations']}",
                    flush=True,
                )
                print(
                    f"  time  = "
                    f"{row['optimizer']['wall_seconds']:.2f} s",
                    flush=True,
                )
                print(
                    f"[checkpoint] "
                    f"{args.output}",
                    flush=True,
                )

            except Exception as exc:
                traceback.print_exc()

                output[
                    "cases"
                ][case_id] = {
                    "n": n,
                    "realization_index":
                        idx,
                    "A":
                        realization["A"],
                    "offsets":
                        realization[
                            "offsets"
                        ],
                    "failed": True,
                    "error_type":
                        type(exc).__name__,
                    "error": repr(exc),
                }

                atomic_write_json(
                    args.output,
                    output,
                )

                print(
                    f"[FAILED] "
                    f"{case_id}: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

    good = [
        row
        for row in output[
            "cases"
        ].values()
        if not row.get(
            "failed",
            False,
        )
    ]

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for n in range(
        args.n_min,
        args.n_max + 1,
    ):
        rows = [
            row
            for row in good
            if int(
                row["n"]
            ) == n
        ]

        if not rows:
            print(
                f"n={n}: no "
                f"successful cases"
            )
            continue

        F_inv = np.asarray(
            [
                row[
                    "direct_inversion_fidelity"
                ]
                for row in rows
            ],
            dtype=float,
        )

        F_opt = np.asarray(
            [
                row[
                    "optimized_fidelity"
                ]
                for row in rows
            ],
            dtype=float,
        )

        evals = np.asarray(
            [
                row[
                    "optimizer"
                ][
                    "n_evaluations"
                ]
                for row in rows
            ],
            dtype=float,
        )

        seconds = np.asarray(
            [
                row[
                    "optimizer"
                ][
                    "wall_seconds"
                ]
                for row in rows
            ],
            dtype=float,
        )

        tones = np.asarray(
            [
                row[
                    "active_tone_count"
                ]
                for row in rows
            ],
            dtype=float,
        )

        print(
            f"n={n}: "
            f"cases={len(rows):2d}  "
            f"median F_inv="
            f"{np.median(F_inv):.6f}  "
            f"median F_opt="
            f"{np.median(F_opt):.6f}  "
            f"median tones="
            f"{np.median(tones):.1f}  "
            f"median evals="
            f"{np.median(evals):.0f}  "
            f"median time="
            f"{np.median(seconds):.2f}s"
        )


if __name__ == "__main__":
    main()
