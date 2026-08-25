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
    "legend.fontsize": 11.0,
})


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
        "--outdir",
        type=Path,
        default=Path("manuscript/figures"),
    )

    args = parser.parse_args()

    data = json.loads(
        args.input.read_text(
            encoding="utf-8-sig"
        )
    )

    rows = []

    for row in data["cases"].values():
        if (
            row.get("status") != "complete"
            or row.get("F_RWA") is None
        ):
            continue

        delta_min = float(
            row[
                "minimum_transition_spacing"
            ]
        )

        mean_delta = float(
            row[
                "mean_transition_spacing"
            ]
        )

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

        relative_spacing = (
            delta_min
            / mean_delta
        )

        normalized_time = (
            float(row["T_gate"])
            * omega_max
            / (2.0 * np.pi)
        )

        rows.append({
            "n": int(row["n"]),
            "x": relative_spacing,
            "y": normalized_time,
        })

    fig, ax = plt.subplots(
        figsize=(5.2, 4.33),
        constrained_layout=True,
    )

    markers = {
        2: "o",
        3: "s",
        4: "^",
        5: "D",
    }

    for n in sorted(
        {row["n"] for row in rows}
    ):
        rr = [
            row
            for row in rows
            if row["n"] == n
        ]

        ax.scatter(
            [row["x"] for row in rr],
            [row["y"] for row in rr],
            s=21,
            alpha=0.62,
            linewidths=0.0,
            color=N_COLORS[n],
            marker=markers[n],
            label=fr"$n={n}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$\Delta_{\min}/\bar{\Delta}$"
    )

    ax.set_ylabel(
        r"$T_{\mathrm{gate}}/"
        r"T_{\mathrm{Rabi}}^{\max}$"
    )

    ax.grid(
        which="major",
        alpha=0.20,
    )

    ax.legend(
        frameon=False,
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        "scaling_relative_minimum_spacing"
    )

    for suffix, kwargs in (
        ("pdf", {}),
        ("png", {"dpi": 350}),
    ):
        path = (
            args.outdir
            / f"{stem}.{suffix}"
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
