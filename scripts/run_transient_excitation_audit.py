from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dialcontrol.dynamics import (
    propagate_final_blocks_with_transient_flip_jit,
)
from dialcontrol.io import atomic_write_json
from dialcontrol.metrics import (
    final_max_ground_to_excited_flip,
    ground_manifold_register_fidelity,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute maximum transient mediator "
            "excitation for saved DIAL benchmark controls."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/publication/main_benchmark.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/publication/"
            "transient_excitation_regenerated.json"
        ),
    )

    args = parser.parse_args()

    source = json.loads(
        args.input.read_text(
            encoding="utf-8-sig"
        )
    )

    out = {
        "schema_version": 1,
        "source": str(args.input),
        "definition": (
            "Maximum over integration-grid times "
            "and computational register inputs of "
            "|U_alpha(t)[1,0]|^2."
        ),
        "cases": {},
    }

    rows = sorted(
        source["cases"].items(),
        key=lambda item: (
            int(item[1]["n"]),
            int(item[1]["source_index"]),
        ),
    )

    for case_id, row in rows:
        if (
            row.get("status") != "complete"
            or row.get("F_RWA") is None
        ):
            continue

        offsets = np.asarray(
            row["offsets"],
            dtype=float,
        )

        tones = np.asarray(
            row["tones"],
            dtype=float,
        )

        intensities = np.asarray(
            row["intensities"],
            dtype=float,
        )

        target_rate = float(
            row["target_rate"]
        )

        signed_phase = math.copysign(
            abs(
                float(
                    row["target_phase"]
                )
            ),
            target_rate,
        )

        (
            blocks,
            pmax,
        ) = (
            propagate_final_blocks_with_transient_flip_jit(
                offsets,
                tones,
                intensities,
                float(row["T_gate"]),
                int(row["n_steps"]),
            )
        )

        F_check = (
            ground_manifold_register_fidelity(
                blocks,
                target_label=str(
                    row["target"]
                ),
                signed_target_phase=signed_phase,
            )
        )

        terminal = (
            final_max_ground_to_excited_flip(
                blocks
            )
        )

        F_saved = float(
            row["F_RWA"]
        )

        delta_F = abs(
            F_check - F_saved
        )

        if delta_F > 1e-8:
            raise RuntimeError(
                f"{case_id}: repropagated "
                f"ground-manifold fidelity differs "
                f"from saved value by "
                f"{delta_F:.3e}."
            )

        out["cases"][case_id] = {
            "status": "complete",
            "n": int(row["n"]),
            "source_index":
                int(row["source_index"]),
            "max_mediator_excitation":
                float(pmax),
            "terminal_max_flip":
                float(terminal),
            "F_check":
                float(F_check),
        }

        atomic_write_json(
            args.output,
            out,
        )

        print(
            f"[done] {case_id}  "
            f"Pmax={pmax:.6e}  "
            f"F={F_check:.10f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
