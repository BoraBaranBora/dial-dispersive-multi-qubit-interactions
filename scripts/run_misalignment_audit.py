from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dial.dynamics import simulate_case
from dial.io import atomic_write_json
from dial.metrics import (
    ground_manifold_fidelity,
    maximum_mediator_excitation,
)


def completed_n3_cases(
    source: dict,
) -> list[tuple[str, dict]]:
    rows = []

    for case_id, case in source["cases"].items():
        if int(case["n"]) != 3:
            continue

        if (
            case.get("status") != "complete"
            or case.get("F_RWA") is None
        ):
            continue

        rows.append(
            (case_id, case)
        )

    rows.sort(
        key=lambda item:
            int(item[1]["source_index"])
    )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the fixed-control DIAL "
            "basis-mismatch audit for n=3."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/publication/"
            "main_benchmark.json"
        ),
    )

    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "results/publication/"
            "misalignment_n3.json"
        ),
        help=(
            "Frozen publication result used only "
            "to recover the numerical sweep protocol."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/publication/"
            "misalignment_n3_regenerated.json"
        ),
    )

    parser.add_argument(
        "--n-systems",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--etas-degrees",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional subset/override in degrees. "
            "By default use the publication sweep."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
    )

    args = parser.parse_args()

    source = json.loads(
        args.input.read_text(
            encoding="utf-8-sig"
        )
    )

    protocol = json.loads(
        args.protocol.read_text(
            encoding="utf-8-sig"
        )
    )

    if int(protocol["n"]) != 3:
        raise RuntimeError(
            "Expected an n=3 misalignment protocol."
        )

    if args.etas_degrees is None:
        etas = [
            float(x)
            for x in protocol["etas"]
        ]
    else:
        etas = [
            math.radians(float(x))
            for x in args.etas_degrees
        ]

    num_samples = int(
        protocol["num_samples"]
    )

    steps_per_period = int(
        protocol["steps_per_period"]
    )

    max_steps = int(
        protocol["max_steps"]
    )

    cases = completed_n3_cases(
        source
    )

    if args.n_systems is not None:
        cases = cases[:args.n_systems]

    config = {
        "kind":
            "dial_basis_mismatch_audit",
        "n":
            3,
        "etas":
            etas,
        "num_samples":
            num_samples,
        "steps_per_period":
            steps_per_period,
        "max_steps":
            max_steps,
        "model": (
            "R_y(eta) tensor-product rotation "
            "of the excited-manifold register basis; "
            "transition frequencies held fixed."
        ),
    }

    if (
        args.output.exists()
        and not args.fresh
    ):
        out = json.loads(
            args.output.read_text(
                encoding="utf-8-sig"
            )
        )

        if out.get("config") != config:
            raise RuntimeError(
                "Existing output uses a different "
                "protocol. Use --fresh or another "
                "output file."
            )
    else:
        out = {
            "schema_version": 1,
            "config": config,
            "cases": {},
        }

    total = (
        len(cases)
        * len(etas)
    )

    count = 0

    for case_id, case in cases:
        offsets = np.asarray(
            case["offsets"],
            dtype=float,
        )

        tones = np.asarray(
            case["tones"],
            dtype=float,
        )

        intensities = np.asarray(
            case["intensities"],
            dtype=float,
        )

        T_gate = float(
            case["T_gate"]
        )

        target = str(
            case["target"]
        )

        target_phase = float(
            case["target_phase"]
        )

        target_rate = float(
            case["target_rate"]
        )

        signed_phase = math.copysign(
            abs(target_phase),
            target_rate,
        )

        case_out = out["cases"].setdefault(
            case_id,
            {
                "n": 3,
                "source_index":
                    int(case["source_index"]),
                "target":
                    target,
                "signed_target_phase":
                    float(signed_phase),
                "T_gate":
                    T_gate,
                "tones":
                    tones,
                "intensities":
                    intensities,
                "results": {},
            },
        )

        for eta in etas:
            count += 1
            key = f"{eta:.8f}"

            if (
                key in case_out["results"]
                and not args.fresh
            ):
                print(
                    f"[skip {count:4d}/{total}] "
                    f"{case_id} eta={eta:.6f}",
                    flush=True,
                )
                continue

            print(
                f"[run  {count:4d}/{total}] "
                f"{case_id}  "
                f"eta={math.degrees(eta):.2f} deg",
                flush=True,
            )

            sim = simulate_case(
                offsets=offsets,
                tones=tones,
                intensities=intensities,
                T_gate=T_gate,
                eta=float(eta),
                num_samples=num_samples,
                steps_per_period=steps_per_period,
                max_steps=max_steps,
            )

            U_t = np.asarray(
                sim["U_t"],
                dtype=np.complex128,
            )

            F = ground_manifold_fidelity(
                U_t[-1],
                n=3,
                target_label=target,
                signed_target_phase=
                    signed_phase,
            )

            pmax = (
                maximum_mediator_excitation(
                    U_t,
                    n=3,
                )
            )

            result = {
                "eta":
                    float(eta),
                "eta_degrees":
                    float(
                        math.degrees(eta)
                    ),
                "alignment_score":
                    float(
                        sim[
                            "alignment_score"
                        ]
                    ),
                "F_ground":
                    float(F),
                "infidelity":
                    float(1.0 - F),
                "P_med_max":
                    float(pmax),
                "n_steps":
                    int(
                        sim["n_steps"]
                    ),
                "dt":
                    float(
                        sim["dt"]
                    ),
            }

            case_out[
                "results"
            ][key] = result

            atomic_write_json(
                args.output,
                out,
            )

            print(
                f"      F={F:.10f}  "
                f"Pmax={pmax:.4e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
