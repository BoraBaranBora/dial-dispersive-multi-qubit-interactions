from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_COLORS = {
    2: "#1f77b4",  # blue
    3: "#ff7f0e",  # orange
    4: "#2ca02c",  # green
    5: "#d62728",  # red
}

plt.rcParams.update({
    "font.size": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
})


def scatter_with_median(
    ax,
    x,
    values,
    *,
    seed,
    color,
    jitter=0.11,
):
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)

    dx = rng.uniform(
        -jitter,
        +jitter,
        size=len(values),
    )

    ax.scatter(
        x + dx,
        values,
        s=18,
        alpha=0.58,
        linewidths=0.0,
        color=color,
        zorder=2,
    )

    median = float(np.median(values))

    ax.hlines(
        median,
        x - 0.20,
        x + 0.20,
        color="black",
        linewidth=2.2,
        zorder=4,
    )


def panel_label(ax, label):
    ax.text(
        0.03,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/publication/main_benchmark.json"
        ),
    )

    parser.add_argument(
        "--excitation-input",
        type=Path,
        default=Path(
            "results/publication/transient_excitation.json"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
    )

    args = parser.parse_args()

    benchmark = json.loads(
        args.input.read_text(
            encoding="utf-8-sig"
        )
    )

    excitation = json.loads(
        args.excitation_input.read_text(
            encoding="utf-8-sig"
        )
    )

    groups = {}

    for row in benchmark["cases"].values():
        if (
            row.get("status") == "complete"
            and row.get("F_RWA") is not None
        ):
            groups.setdefault(
                int(row["n"]),
                [],
            ).append(row)

    ns = sorted(groups)
    xpos = np.arange(1, len(ns) + 1)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 3.45),
        constrained_layout=True,
    )

    ax_f, ax_p, ax_t = axes

    # ------------------------------------------------------------
    # (a) full-dynamics infidelity
    # ------------------------------------------------------------

    for i, n in enumerate(ns):
        values = np.asarray(
            [
                1.0 - float(row["F_RWA"])
                for row in groups[n]
            ]
        )

        scatter_with_median(
            ax_f,
            xpos[i],
            values,
            seed=20260821 + n,
            color=N_COLORS[n],
        )

    ax_f.set_yscale("log")
    ax_f.set_xticks(xpos, ns)

    ax_f.set_xlabel(
        r"Register size $n$"
    )

    ax_f.set_ylabel(
        r"Infidelity $1-F$"
    )

    ax_f.grid(
        axis="y",
        which="major",
        alpha=0.22,
    )

    ax_f.grid(
        axis="y",
        which="minor",
        alpha=0.10,
    )

    panel_label(ax_f, "(a)")

    # ------------------------------------------------------------
    # (b) maximum transient mediator excitation
    # ------------------------------------------------------------

    for i, n in enumerate(ns):
        values = []

        for row in groups[n]:
            case_id = (
                f"n{n}_realization_"
                f"{int(row['source_index']):03d}"
            )

            audit = excitation[
                "cases"
            ].get(case_id)

            if (
                audit is not None
                and audit.get("status")
                == "complete"
            ):
                values.append(
                    100.0
                    * float(
                        audit[
                            "max_mediator_excitation"
                        ]
                    )
                )

        scatter_with_median(
            ax_p,
            xpos[i],
            np.asarray(values),
            seed=20260921 + n,
            color=N_COLORS[n],
        )

    ax_p.set_xticks(xpos, ns)

    ax_p.set_xlabel(
        r"Register size $n$"
    )

    ax_p.set_ylabel(
        r"$P_{\mathrm{med}}^{\max}$ (\%)"
    )

    ax_p.set_ylim(bottom=0.0)

    ax_p.grid(
        axis="y",
        alpha=0.22,
    )

    panel_label(ax_p, "(b)")

    # ------------------------------------------------------------
    # (c) normalized gate duration
    # ------------------------------------------------------------

    for i, n in enumerate(ns):
        values = []

        for row in groups[n]:
            x = np.asarray(
                row["intensities"],
                dtype=float,
            )

            omega_max = float(
                np.max(
                    np.sqrt(
                        np.maximum(
                            2.0 * x,
                            0.0,
                        )
                    )
                )
            )

            values.append(
                float(row["T_gate"])
                * omega_max
                / (2.0 * np.pi)
            )

        scatter_with_median(
            ax_t,
            xpos[i],
            np.asarray(values),
            seed=20261021 + n,
            color=N_COLORS[n],
        )

    ax_t.set_yscale("log")
    ax_t.set_xticks(xpos, ns)

    ax_t.set_xlabel(
        r"Register size $n$"
    )

    ax_t.set_ylabel(
        r"$T_{\mathrm{gate}}/"
        r"T_{\mathrm{Rabi}}^{\max}$"
    )

    ax_t.grid(
        axis="y",
        which="major",
        alpha=0.22,
    )

    ax_t.grid(
        axis="y",
        which="minor",
        alpha=0.10,
    )

    panel_label(ax_t, "(c)")

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 350}),
    ):
        path = (
            args.outdir
            / f"main_direct_inversion_scaling.{suffix}"
        )

        fig.savefig(
            path,
            bbox_inches="tight",
            **kwargs,
        )

        print("Wrote", path)

    plt.close(fig)


if __name__ == "__main__":
    main()
