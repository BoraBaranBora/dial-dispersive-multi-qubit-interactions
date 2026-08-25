from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(name):
    path = SCRIPTS / name

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    subprocess.run(
        [
            sys.executable,
            str(path),
        ],
        cwd=ROOT,
        check=True,
    )


def main():
    for name in (
        "plot_main_benchmark.py",
        "plot_appendix_ensemble.py",
        "plot_appendix_spectral_crowding.py",
        "plot_appendix_misalignment.py",
    ):
        run(name)


if __name__ == "__main__":
    main()
