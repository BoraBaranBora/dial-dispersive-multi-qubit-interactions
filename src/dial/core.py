from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import json
import os

import numpy as np
from scipy.optimize import nnls

def _configured_path(environment_variable: str, default: str) -> Path:
    value = os.environ.get(environment_variable)
    return Path(value) if value else Path(default)


DATA_DIR = _configured_path("DPRC_DATA_DIR", "data/final_numerics")
FIG_DIR = _configured_path("DPRC_FIG_DIR", "media/final_numerics")

N_QUBITS = 3

# Manuscript convention:
# H_eff = (1/2) sigma_z sum_S K_S Z_S.
#
# G has no extra factor of 2.
# K = G x.
#
# Quoted target phase = desired register phase.
# Therefore T = 2 * target_register_phase / |K_target|.
TARGET_REGISTER_PHASE = np.pi / 4
TARGET_PHASE_PARAMETER = 2.0 * TARGET_REGISTER_PHASE


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))



def bit_configs(n: int = N_QUBITS) -> np.ndarray:
    return np.array(
        [[(a >> (n - 1 - i)) & 1 for i in range(n)] for a in range(2**n)],
        dtype=int,
    )


def z_configs(n: int = N_QUBITS) -> np.ndarray:
    return (-1.0) ** bit_configs(n)


def pauli_masks_and_labels(
    n: int = N_QUBITS,
    *,
    include_identity: bool = False,
) -> tuple[np.ndarray, list[str]]:
    masks: list[int] = []
    labels: list[str] = []

    start = 0 if include_identity else 1
    for mask in range(start, 2**n):
        inds = [i + 1 for i in range(n) if (mask >> (n - 1 - i)) & 1]
        label = "I" if not inds else "".join(f"Z{i}" for i in inds)
        masks.append(mask)
        labels.append(label)

    return np.array(masks, dtype=int), labels


def pauli_eigenvalue_matrix(
    n: int = N_QUBITS,
    *,
    include_identity: bool = False,
) -> tuple[list[str], np.ndarray]:
    masks, labels = pauli_masks_and_labels(n, include_identity=include_identity)
    bits = bit_configs(n)

    eig = np.ones((2**n, len(labels)), dtype=float)
    for j, mask in enumerate(masks):
        val = np.ones(2**n, dtype=float)
        for i in range(n):
            if (mask >> (n - 1 - i)) & 1:
                val *= (-1.0) ** bits[:, i]
        eig[:, j] = val

    return labels, eig


def pauli_order(label: str) -> int:
    if label == "I":
        return 0
    return label.count("Z")


def normalized_couplings_from_ratios(
    rho: float,
    sigma: float,
    *,
    B: float = 1.0,
) -> np.ndarray:
    raw = np.array([1.0, float(rho), float(sigma)], dtype=float)
    return B * raw / np.sum(np.abs(raw))


def transition_offsets_from_couplings(A: np.ndarray) -> np.ndarray:
    z = z_configs(len(A))
    return z @ A


def candidate_tones(
    offsets: np.ndarray,
    *,
    num_grid: int = 301,
    span: float = 1.35,
    min_detuning: float = 0.04,
) -> np.ndarray:
    B = float(np.max(np.abs(offsets)))
    grid = np.linspace(-span * B, span * B, num_grid)

    keep = np.ones_like(grid, dtype=bool)
    for lam in offsets:
        keep &= np.abs(grid - lam) >= min_detuning * B

    return grid[keep]


def response_matrix(
    offsets: np.ndarray,
    tones: np.ndarray,
    *,
    n: int = N_QUBITS,
    include_identity: bool = False,
) -> tuple[list[str], np.ndarray]:
    labels, eig = pauli_eigenvalue_matrix(n, include_identity=include_identity)

    G = np.empty((len(labels), len(tones)), dtype=float)
    for k, omega in enumerate(tones):
        det = omega - offsets
        response_alpha = 1.0 / det
        G[:, k] = (eig.T @ response_alpha) / float(2**n)

    return labels, G


def leakage_ratio(K: np.ndarray, target_index: int) -> float:
    target = abs(float(K[target_index]))
    if target == 0:
        return np.inf

    spectator = np.linalg.norm(np.delete(K, target_index))
    return float(spectator / target)


def solve_dense_target(G: np.ndarray, target_index: int) -> dict:
    target = np.zeros(G.shape[0], dtype=float)
    target[target_index] = 1.0

    x, residual = nnls(G, target)
    K = G @ x

    return {
        "x": x,
        "K": K,
        "residual": float(residual),
        "leakage": leakage_ratio(K, target_index),
        "target_rate": float(K[target_index]),
        "spectator_norm": float(np.linalg.norm(np.delete(K, target_index))),
    }


def best_targets_by_order(labels: list[str], G: np.ndarray) -> dict[int, dict]:
    out: dict[int, dict] = {}

    for order in (1, 2, 3):
        candidates = [
            i for i, label in enumerate(labels)
            if pauli_order(label) == order
        ]

        best = None
        for i in candidates:
            sol = solve_dense_target(G, i)
            row = {
                "order": order,
                "target": labels[i],
                "target_index": i,
                "leakage": sol["leakage"],
                "residual": sol["residual"],
                "target_rate": sol["target_rate"],
                "spectator_norm": sol["spectator_norm"],
            }
            if best is None or row["leakage"] < best["leakage"]:
                best = row

        if best is None:
            raise RuntimeError(f"No targets found for order {order}")

        out[order] = best

    return out


def applicability_score(leakages: Iterable[float], floor: float = 1e-16) -> float:
    worst = max(float(x) for x in leakages)
    return float(-np.log10(max(worst, floor)))


def gate_time_from_rate(
    K_target: float,
    *,
    target_register_phase: float = TARGET_REGISTER_PHASE,
) -> float:
    rate = abs(float(K_target))
    if rate == 0:
        return np.inf

    return float(2.0 * abs(target_register_phase) / rate)


@dataclass(frozen=True)
class PlatformFamily:
    family: str
    benchmark_label: str
    B_over_2pi_hz: float
    description: str


PLATFORM_FAMILIES = [
    PlatformFamily(
        family="narrow_cluster",
        benchmark_label="rare-earth-like narrow splittings",
        B_over_2pi_hz=45e3,
        description="small absolute bandwidth; slow but spectrally compact",
    ),
    PlatformFamily(
        family="intermediate_cluster",
        benchmark_label="weak color-center-like splittings",
        B_over_2pi_hz=450e3,
        description="intermediate absolute bandwidth",
    ),
    PlatformFamily(
        family="resolved_mixed",
        benchmark_label="resolved color-center-like splittings",
        B_over_2pi_hz=1.08e6,
        description="MHz-scale resolved manifold",
    ),
    PlatformFamily(
        family="broad_proximal",
        benchmark_label="proximal strongly coupled splittings",
        B_over_2pi_hz=10.8e6,
        description="large absolute bandwidth; faster phase accumulation",
    ),
]