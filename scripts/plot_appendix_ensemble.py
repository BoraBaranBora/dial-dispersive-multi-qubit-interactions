from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#1f77b4"

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

    points = []

    for row in data["cases"].values():
        if (
            row.get("status") == "complete"
            and int(row["n"]) == 3
        ):
            A = np.asarray(
                row["A"],
                dtype=float,
            )

            points.append(
                [A[1], A[2]]
            )

    points = np.asarray(
        points,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(5.2, 4.44),
        constrained_layout=True,
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=22,
        alpha=0.68,
        linewidths=0.0,
        color=BLUE,
        label=fr"Benchmark systems ($N={len(points)}$)",
    )

    boundary = np.logspace(
        0.0,
        1.0,
        300,
    )

    ax.plot(
        boundary,
        boundary,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=r"$A_3=A_2$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1.0, 10.0)
    ax.set_ylim(1.0, 10.0)

    ax.set_xlabel(r"$A_2$")
    ax.set_ylabel(r"$A_3$")

    ticks = [1, 2, 3, 5, 10]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(
        [str(x) for x in ticks]
    )

    ax.set_yticklabels(
        [str(x) for x in ticks]
    )

    ax.grid(
        which="major",
        alpha=0.18,
    )

    ax.text(
        0.04,
        0.95,
        r"$A_1=1$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.text(
        4.6,
        1.7,
        "Permutation\nredundant",
        ha="center",
        va="center",
        fontsize=11.0,
        alpha=0.70,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        "benchmark_ensemble_n3_"
        "normalized_spectral_space"
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
