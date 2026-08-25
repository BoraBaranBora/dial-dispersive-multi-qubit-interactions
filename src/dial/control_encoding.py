from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path

import numpy as np


from . import full_rwa_metrics as base

from . import aligned_dynamics as jit


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def atomic_write_json(
    path: Path,
    data: dict,
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
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def wait_for_comparison0(
    path: Path,
    *,
    expected_cases: int,
    poll_seconds: float,
) -> dict:

    last_count = None

    while True:
        try:
            data = load_json(path)
            cases = data.get(
                "cases",
                [],
            )

            count = len(cases)

            if count != last_count:
                print(
                    f"[waiting for comparison 0] "
                    f"{count}/{expected_cases} "
                    f"case records available",
                    flush=True,
                )

                last_count = count

            if count >= expected_cases:
                print(
                    "[comparison 0 complete enough "
                    "for comparison 1]",
                    flush=True,
                )
                return data

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
        ):
            pass

        time.sleep(
            poll_seconds
        )


def sorted_control(
    tones,
    intensities,
):
    tones = np.asarray(
        tones,
        dtype=float,
    )

    intensities = np.asarray(
        intensities,
        dtype=float,
    )

    order = np.argsort(tones)

    return (
        tones[order],
        intensities[order],
    )


def control_to_z(
    *,
    tones: np.ndarray,
    intensities: np.ndarray,
    offsets: np.ndarray,
    r_disp: float,
    omega_hw_max: float,
) -> np.ndarray:

    tones, intensities = (
        sorted_control(
            tones,
            intensities,
        )
    )

    xmax, _ = (
        base.local_intensity_caps(
            tones,
            offsets,
            r_disp=r_disp,
            omega_hw_max=(
                omega_hw_max
            ),
        )
    )

    scale = max(
        1.0,
        float(
            np.max(
                np.abs(xmax)
            )
        ),
    )

    if np.any(
        intensities
        > xmax + 1e-10 * scale
    ):
        raise RuntimeError(
            "Saved Comparison-0 seed "
            "violates the optimizer's "
            "local intensity caps."
        )

    fractions = np.divide(
        intensities,
        xmax,
        out=np.zeros_like(
            intensities
        ),
        where=xmax > 0.0,
    )

    fractions = np.clip(
        fractions,
        0.0,
        1.0,
    )

    return np.concatenate(
        [
            tones,
            fractions,
        ]
    )


def find_best_random(
    case: dict,
) -> dict:

    sampling = case[
        "random_sampling"
    ]

    best_index = int(
        sampling[
            "fidelity_summary"
        ][
            "best_index"
        ]
    )

    for row in sampling[
        "controls"
    ]:
        if int(
            row["random_index"]
        ) == best_index:
            return row

    raise RuntimeError(
        f"Best random index "
        f"{best_index} not found."
    )


def threshold_evaluations(
    run: dict,
) -> dict:

    thresholds = (
        0.99,
        0.999,
        0.9999,
    )

    out = {}

    trace = run.get(
        "optimization_trace",
        [],
    )

    for threshold in thresholds:
        key = f"{threshold:.4f}"

        if (
            run[
                "initial_F_search"
            ]
            >= threshold
        ):
            out[key] = 0
            continue

        hit = None

        for row in trace:
            value = row.get(
                "best_feasible_fidelity"
            )

            if (
                value is not None
                and value >= threshold
            ):
                hit = int(
                    row["evaluations"]
                )
                break

        out[key] = hit

    return out


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Comparison 1: identical "
            "full-dynamics refinement from "
            "the inversion seed and the "
            "best-of-100 random seed saved "
            "by Comparison 0."
        )
    )

    parser.add_argument(
        "--comparison0-json",
        type=Path,
        default=Path(
            "results/benchmarks/"
            "geometry_control_comparison0.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmarks/"
            "geometry_control_comparison1.json"
        ),
    )

    parser.add_argument(
        "--expected-cases",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--wait",
        action="store_true",
    )

    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--span",
        type=float,
        default=1.35,
    )

    parser.add_argument(
        "--min-tone-spacing",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--expected-r-disp",
        type=float,
        default=0.15,
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
        default=20,
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

    args = parser.parse_args()

    if args.wait:
        comparison0 = (
            wait_for_comparison0(
                args.comparison0_json,
                expected_cases=(
                    args.expected_cases
                ),
                poll_seconds=(
                    args.poll_seconds
                ),
            )
        )
    else:
        comparison0 = load_json(
            args.comparison0_json
        )

        if (
            len(
                comparison0.get(
                    "cases",
                    [],
                )
            )
            < args.expected_cases
        ):
            raise RuntimeError(
                "Comparison 0 is not complete."
            )

    if args.output.exists():
        output = load_json(
            args.output
        )
        print(
            f"[resume] {args.output}",
            flush=True,
        )
    else:
        output = {
            "comparison0_json": str(
                args.comparison0_json
            ),
            "protocol": {
                "comparison": (
                    "inversion-seeded versus "
                    "best-of-100-random-seeded "
                    "full-dynamics refinement"
                ),
                "fixed_gate_time": True,
                "same_optimizer": True,
                "search_steps_per_period":
                    args.search_steps_per_period,
                "final_steps_per_period":
                    args.final_steps_per_period,
                "maxiter":
                    args.maxiter,
                "ftol":
                    args.ftol,
                "span":
                    args.span,
                "min_tone_spacing":
                    args.min_tone_spacing,
                "expected_r_disp":
                    args.expected_r_disp,
                "omega_hw_max": (
                    None
                    if not math.isfinite(
                        args.omega_hw_max
                    )
                    else args.omega_hw_max
                ),
            },
            "cases": {},
        }

    warmed_up = False

    cases = comparison0[
        "cases"
    ]

    for case_number, case in enumerate(
        cases,
        start=1,
    ):
        geometry_id = case.get(
            "geometry_id",
            f"case_{case_number:03d}",
        )

        print()
        print("=" * 72)
        print(
            f"COMPARISON 1 | "
            f"{geometry_id} | "
            f"{case_number}/"
            f"{len(cases)}"
        )
        print("=" * 72)

        case_out = output[
            "cases"
        ].setdefault(
            geometry_id,
            {
                "geometry_id":
                    geometry_id,
                "arms": {},
            },
        )

        if case.get(
            "failed",
            False,
        ):
            case_out[
                "skipped"
            ] = True
            case_out[
                "reason"
            ] = (
                "Comparison 0 case failed."
            )
            atomic_write_json(
                args.output,
                output,
            )
            continue

        try:
            target = str(
                case["target"]
            )

            offsets = np.asarray(
                case["offsets"],
                dtype=float,
            )

            T_gate = float(
                case["gate_time"]
            )

            signed_target_phase = float(
                case[
                    "signed_target_phase"
                ]
            )

            r_disp = float(
                case[
                    "r_disp_target"
                ]
            )

            if not math.isclose(
                r_disp,
                args.expected_r_disp,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{geometry_id}: "
                    f"r_disp={r_disp} "
                    f"does not match frozen "
                    f"value "
                    f"{args.expected_r_disp}."
                )

            n_random = int(
                case[
                    "random_sampling"
                ][
                    "n_random"
                ]
            )

            if n_random != 100:
                raise RuntimeError(
                    f"{geometry_id}: expected "
                    f"100 random controls, "
                    f"found {n_random}."
                )

            inv = case[
                "inversion"
            ]

            inv_tones, inv_x = (
                sorted_control(
                    inv["tones"],
                    inv["intensities"],
                )
            )

            random_best = (
                find_best_random(
                    case
                )
            )

            rand_tones, rand_x = (
                sorted_control(
                    random_best[
                        "tones"
                    ],
                    random_best[
                        "intensities"
                    ],
                )
            )

            M = len(
                inv_tones
            )

            if len(
                rand_tones
            ) != M:
                raise RuntimeError(
                    f"{geometry_id}: "
                    "seed tone counts differ."
                )

            total_intensity_max = float(
                np.sum(
                    inv_x
                )
            )

            if not np.isclose(
                np.sum(rand_x),
                total_intensity_max,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"{geometry_id}: "
                    "best-random seed does not "
                    "match inversion total "
                    "intensity."
                )

            z_inv = control_to_z(
                tones=inv_tones,
                intensities=inv_x,
                offsets=offsets,
                r_disp=r_disp,
                omega_hw_max=(
                    args.omega_hw_max
                ),
            )

            z_rand = control_to_z(
                tones=rand_tones,
                intensities=rand_x,
                offsets=offsets,
                r_disp=r_disp,
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
                    max_steps=(
                        args.max_steps
                    ),
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
                    max_steps=(
                        args.max_steps
                    ),
                )
            )

            if not warmed_up:
                _ = (
                    jit.propagate_final_blocks_jit(
                        offsets,
                        inv_tones,
                        inv_x,
                        T_gate,
                        2,
                    )
                )
                warmed_up = True

            case_out[
                "problem"
            ] = {
                "target": target,
                "M": int(M),
                "T_gate": T_gate,
                "r_disp": r_disp,
                "total_intensity_max":
                    total_intensity_max,
                "min_tone_spacing":
                    args.min_tone_spacing,
                "span": args.span,
                "n_steps_search":
                    int(n_steps_search),
                "n_steps_final":
                    int(n_steps_final),
                "comparison0_F_inversion":
                    float(
                        case[
                            "comparison0"
                        ][
                            "F_inversion"
                        ]
                    ),
                "comparison0_F_best_random":
                    float(
                        case[
                            "comparison0"
                        ][
                            "F_random_best"
                        ]
                    ),
                "best_random_index":
                    int(
                        case[
                            "random_sampling"
                        ][
                            "fidelity_summary"
                        ][
                            "best_index"
                        ]
                    ),
            }

            common = dict(
                offsets=offsets,
                M=M,
                T_gate=T_gate,
                n_steps_search=(
                    n_steps_search
                ),
                n_steps_final=(
                    n_steps_final
                ),
                target_label=target,
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
                    args.min_tone_spacing
                ),
                span=args.span,
                maxiter=args.maxiter,
                ftol=args.ftol,
            )

            arm_specs = (
                (
                    "inversion_seed",
                    z_inv,
                ),
                (
                    "best_random_seed",
                    z_rand,
                ),
            )

            for arm_name, z0 in arm_specs:

                if arm_name in case_out[
                    "arms"
                ]:
                    print(
                        f"[skip completed arm] "
                        f"{arm_name}",
                        flush=True,
                    )
                    continue

                print(
                    f"[running arm] "
                    f"{arm_name}",
                    flush=True,
                )

                try:
                    run = (
                        base.run_one_start(
                            name=arm_name,
                            z0=z0,
                            **common,
                        )
                    )

                    run[
                        "threshold_evaluations_search"
                    ] = (
                        threshold_evaluations(
                            run
                        )
                    )

                    case_out[
                        "arms"
                    ][
                        arm_name
                    ] = run

                except Exception as exc:
                    case_out[
                        "arms"
                    ][
                        arm_name
                    ] = {
                        "failed": True,
                        "error": repr(exc),
                    }

                    print(
                        f"[arm failed] "
                        f"{geometry_id} "
                        f"{arm_name}: "
                        f"{exc!r}",
                        flush=True,
                    )

                # Checkpoint after every arm.
                atomic_write_json(
                    args.output,
                    output,
                )

                print(
                    f"[checkpoint] "
                    f"{args.output}",
                    flush=True,
                )

            inv_run = case_out[
                "arms"
            ].get(
                "inversion_seed"
            )

            rand_run = case_out[
                "arms"
            ].get(
                "best_random_seed"
            )

            if (
                inv_run
                and rand_run
                and not inv_run.get(
                    "failed",
                    False,
                )
                and not rand_run.get(
                    "failed",
                    False,
                )
            ):
                case_out[
                    "comparison1"
                ] = {
                    "F_inversion_seed_final":
                        float(
                            inv_run[
                                "final_F_final"
                            ]
                        ),
                    "F_best_random_seed_final":
                        float(
                            rand_run[
                                "final_F_final"
                            ]
                        ),
                    "delta_F_final":
                        float(
                            inv_run[
                                "final_F_final"
                            ]
                            -
                            rand_run[
                                "final_F_final"
                            ]
                        ),
                    "evals_inversion_seed":
                        int(
                            inv_run[
                                "n_evaluations"
                            ]
                        ),
                    "evals_best_random_seed":
                        int(
                            rand_run[
                                "n_evaluations"
                            ]
                        ),
                }

                atomic_write_json(
                    args.output,
                    output,
                )

        except Exception as exc:
            case_out[
                "failed"
            ] = True

            case_out[
                "error"
            ] = repr(exc)

            atomic_write_json(
                args.output,
                output,
            )

            print(
                f"[case failed] "
                f"{geometry_id}: "
                f"{exc!r}",
                flush=True,
            )

    successful = [
        row
        for row in output[
            "cases"
        ].values()
        if "comparison1" in row
    ]

    print()
    print("=" * 72)
    print("COMPARISON 1 COMPLETE")
    print("=" * 72)
    print(
        f"paired completed cases : "
        f"{len(successful)}"
    )

    if successful:
        delta = np.asarray(
            [
                row[
                    "comparison1"
                ][
                    "delta_F_final"
                ]
                for row in successful
            ],
            dtype=float,
        )

        print(
            f"inversion-seed wins    : "
            f"{np.mean(delta > 0):.3f}"
        )

        print(
            f"median final delta F   : "
            f"{np.median(delta):+.9f}"
        )

    atomic_write_json(
        args.output,
        output,
    )


if __name__ == "__main__":
    main()
