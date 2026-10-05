from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from dialcontrol.dynamics import (
    conservative_n_steps,
    propagate_final_blocks_jit,
)
from dialcontrol.io import atomic_write_json
from dialcontrol.metrics import (
    final_max_ground_to_excited_flip,
    ground_manifold_register_fidelity,
)
from dialcontrol.sampling import sample_realizations
from dialcontrol.spectrum import mean_transition_spacing
from dialcontrol.synthesis import (
    dispersive_scale_from_unit_control,
    finite_tone_control_inversion_n,
)


DEFAULT_SEED = 20260818
DEFAULT_PHASE = math.pi / 4.0


def run_case(
    *,
    realization: dict,
    target_phase: float,
    chi_tone_spacing: float,
    r_disp: float,
    span: float,
    grid_points: int,
    steps_per_period: int,
    max_steps: int,
    omega_hw_max: float,
) -> dict:
    n = int(realization["n"])

    offsets = np.asarray(
        realization["offsets"],
        dtype=float,
    )

    ordered = np.sort(offsets)

    delta_min = float(
        np.min(np.diff(ordered))
    )

    bar_delta = mean_transition_spacing(
        offsets
    )

    spectral_radius = float(
        np.max(np.abs(offsets))
    )

    physical_min_detuning = (
        0.45 * delta_min
    )

    candidate_min_detuning = (
        physical_min_detuning
        / spectral_radius
    )

    min_tone_spacing = (
        chi_tone_spacing
        * bar_delta
    )

    inversion = (
        finite_tone_control_inversion_n(
            offsets=offsets,
            n=n,
            max_tones=2**n - 1,
            min_tone_spacing=min_tone_spacing,
            span=span,
            grid_points=grid_points,
            candidate_pool=grid_points,
            min_detuning=candidate_min_detuning,
        )
    )

    scaled = (
        dispersive_scale_from_unit_control(
            tones=np.asarray(
                inversion["tones"],
                dtype=float,
            ),
            intensities_unit=np.asarray(
                inversion["intensities_unit"],
                dtype=float,
            ),
            offsets=offsets,
            r_disp=r_disp,
        )
    )

    # Match the publication benchmark exactly: keep the
    # saturated tone infinitesimally inside the dispersive cap.
    safety = 1.0 - 1e-10

    drive_scale = float(
        scaled["drive_scale"]
        * safety
    )

    intensities = (
        np.asarray(
            inversion["intensities_unit"],
            dtype=float,
        )
        * drive_scale
    )

    target_rate = float(
        inversion["target_rate_unit"]
        * drive_scale
    )

    T_gate = float(
        2.0
        * abs(target_phase)
        / abs(target_rate)
    )

    signed_target_phase = float(
        math.copysign(
            abs(target_phase),
            target_rate,
        )
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    tones = np.asarray(
        inversion["tones"],
        dtype=float,
    )

    detunings = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    dispersive_ratios = (
        amplitudes / detunings
    )

    n_steps = conservative_n_steps(
        offsets=offsets,
        span=span,
        total_intensity_max=float(
            np.sum(intensities)
        ),
        T_gate=T_gate,
        steps_per_period=steps_per_period,
        max_steps=max_steps,
    )

    blocks = propagate_final_blocks_jit(
        offsets,
        tones,
        intensities,
        T_gate,
        n_steps,
    )

    F = ground_manifold_register_fidelity(
        blocks,
        target_label=str(
            inversion["target"]
        ),
        signed_target_phase=(
            signed_target_phase
        ),
    )

    terminal_flip = (
        final_max_ground_to_excited_flip(
            blocks
        )
    )

    return {
        "status": "complete",
        "n": n,
        "source_index": int(
            realization[
                "realization_index"
            ]
        ),
        "A": realization["A"],
        "offsets": offsets,
        "target": inversion["target"],
        "target_phase": float(
            target_phase
        ),
        "mean_transition_spacing":
            float(bar_delta),
        "minimum_transition_spacing":
            float(delta_min),
        "min_detuning":
            float(
                physical_min_detuning
            ),
        "min_tone_spacing_required":
            float(
                min_tone_spacing
            ),
        "max_tones":
            int(2**n - 1),
        "candidate_pool":
            int(grid_points),
        "relative_residual":
            float(
                inversion[
                    "relative_residual"
                ]
            ),
        "epsilon_spec":
            float(
                inversion[
                    "epsilon_spec"
                ]
            ),
        "active_tones":
            int(
                inversion[
                    "active_count"
                ]
            ),
        "tones": tones,
        "intensities": intensities,
        "target_rate":
            target_rate,
        "T_gate":
            T_gate,
        "max_dispersive_ratio":
            float(
                np.max(
                    dispersive_ratios
                )
            ),
        "n_steps":
            int(n_steps),
        "F_RWA":
            float(F),
        "terminal_max_flip":
            float(
                terminal_flip
            ),
        "actual_min_tone_spacing":
            (
                float(
                    np.min(
                        np.diff(
                            np.sort(tones)
                        )
                    )
                )
                if len(tones) > 1
                else None
            ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the main DIAL "
            "register-size benchmark."
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
        default=5,
    )

    parser.add_argument(
        "--n-systems",
        type=int,
        default=100,
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
        "--chi-tone-spacing",
        type=float,
        default=0.075,
    )

    parser.add_argument(
        "--r-disp",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--span",
        type=float,
        default=1.35,
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=6401,
    )

    parser.add_argument(
        "--steps-per-period",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000_000,
    )

    parser.add_argument(
        "--omega-hw-max",
        type=float,
        default=1.0e6,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/publication/"
            "main_benchmark_regenerated.json"
        ),
    )

    args = parser.parse_args()

    out = {
        "schema_version": 1,
        "config": vars(args) | {
            "output": str(
                args.output
            ),
            "d_min_factor": 0.45,
            "candidate_pool_mode":
                "full_admissible_grid",
        },
        "cases": {},
    }

    for n in range(
        args.n_min,
        args.n_max + 1,
    ):
        realizations = sample_realizations(
            n=n,
            n_systems=args.n_systems,
            seed=args.seed,
            ratio_max=args.ratio_max,
            sobol_power=args.sobol_power,
        )

        for realization in realizations:
            index = int(
                realization[
                    "realization_index"
                ]
            )

            case_id = (
                f"n{n}_realization_"
                f"{index:03d}"
            )

            print(
                f"[run] {case_id}",
                flush=True,
            )

            row = run_case(
                realization=realization,
                target_phase=(
                    args.target_phase
                ),
                chi_tone_spacing=(
                    args.chi_tone_spacing
                ),
                r_disp=args.r_disp,
                span=args.span,
                grid_points=(
                    args.grid_points
                ),
                steps_per_period=(
                    args.steps_per_period
                ),
                max_steps=(
                    args.max_steps
                ),
                omega_hw_max=(
                    args.omega_hw_max
                ),
            )

            out["cases"][case_id] = row

            atomic_write_json(
                args.output,
                out,
            )

            print(
                f"      F={row['F_RWA']:.8f}  "
                f"Pfinal="
                f"{row['terminal_max_flip']:.3e}  "
                f"T={row['T_gate']:.4e}",
                flush=True,
            )

    atomic_write_json(
        args.output,
        out,
    )


if __name__ == "__main__":
    main()
