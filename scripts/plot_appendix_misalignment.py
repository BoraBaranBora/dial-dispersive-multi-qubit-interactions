from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


BLUE = "#1f77b4"

plt.rcParams.update({
    "font.size": 12.0,
    "axes.labelsize": 12.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
})


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


def add_distribution(
    ax,
    x,
    values,
    *,
    seed,
    jitter=0.28,
    median_width=0.80,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    dx = rng.uniform(
        -jitter,
        +jitter,
        size=len(values),
    )

    ax.scatter(
        x + dx,
        values,
        s=18,
        alpha=0.55,
        linewidths=0.0,
        color=BLUE,
        zorder=2,
    )

    median = float(
        np.median(values)
    )

    # Independent median bar at each eta.
    # Deliberately no connecting line.
    ax.hlines(
        median,
        x - median_width / 2.0,
        x + median_width / 2.0,
        color="black",
        linewidth=2.3,
        zorder=4,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/publication/"
            "misalignment_n3.json"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
    )

    args = parser.parse_args()

    data = json.loads(
        args.input.read_text(
            encoding="utf-8-sig"
        )
    )

    grouped = {}

    for case in data["cases"].values():
        for result in (
            case["results"].values()
        ):
            eta = float(
                result["eta"]
            )

            grouped.setdefault(
                eta,
                {
                    "degrees":
                        float(
                            result[
                                "eta_degrees"
                            ]
                        ),
                    "F": [],
                    "P": [],
                },
            )

            grouped[eta]["F"].append(
                float(
                    result[
                        "F_ground"
                    ]
                )
            )

            grouped[eta]["P"].append(
                100.0
                * float(
                    result[
                        "P_med_max"
                    ]
                )
            )

    etas = sorted(grouped)

    # Display the intended round-number angle values.
    x = np.asarray(
        [
            round(
                grouped[eta]["degrees"],
                1,
            )
            for eta in etas
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 5.10),
        constrained_layout=True,
    )

    ax_f, ax_p = axes

    for i, eta in enumerate(etas):
        add_distribution(
            ax_f,
            x[i],
            grouped[eta]["F"],
            seed=20261100 + i,
        )

        add_distribution(
            ax_p,
            x[i],
            grouped[eta]["P"],
            seed=20261200 + i,
        )

    # ------------------------------------------------------------
    # (a) fidelity
    # ------------------------------------------------------------

    ax_f.set_xlabel(
        r"Basis mismatch $\eta$ (deg)"
    )

    ax_f.set_ylabel(
        r"Fidelity $F$"
    )

    ax_f.set_xticks(x)

    ax_f.grid(
        axis="y",
        alpha=0.20,
    )

    panel_label(
        ax_f,
        "(a)",
    )

    # ------------------------------------------------------------
    # (b) excitation
    # ------------------------------------------------------------

    ax_p.set_xlabel(
        r"Basis mismatch $\eta$ (deg)"
    )

    # Deliberately concise: symbol only.
    ax_p.set_ylabel(
        r"$P_{\mathrm{med}}^{\max}$"
    )

    ax_p.set_xticks(x)
    ax_p.set_ylim(bottom=0.0)

    # Values are plotted as percentages, while the axis label
    # remains only the physical symbol.
    ax_p.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100.0,
            decimals=1,
        )
    )

    ax_p.grid(
        axis="y",
        alpha=0.20,
    )

    panel_label(
        ax_p,
        "(b)",
    )

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
            / f"dial_misalignment_audit.{suffix}"
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
