"""Target-aware DIAL synthesis under physical dispersive constraints."""

from __future__ import annotations

import math
import re

import numpy as np
from scipy.optimize import nnls

from .spectrum import candidate_tones
from .transfer import transfer_matrix


def filter_nonzero_transfer_columns(
    tones: np.ndarray,
    G: np.ndarray,
    *,
    eps: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(G, axis=0)
    keep = norms > eps
    return tones[keep], G[:, keep]


def downsample_candidates(
    tones: np.ndarray,
    G: np.ndarray,
    *,
    candidate_pool: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(tones) <= candidate_pool:
        return tones, G

    indices = np.linspace(0, len(tones) - 1, candidate_pool)
    indices = np.unique(np.round(indices).astype(int))

    return tones[indices], G[:, indices]


def respects_tone_spacing(
    tones: np.ndarray,
    selected: list[int],
    candidate_index: int,
    *,
    min_tone_spacing: float,
) -> bool:
    omega = tones[candidate_index]

    for j in selected:
        if abs(omega - tones[j]) < min_tone_spacing:
            return False

    return True


def leakage_ratio(
    K: np.ndarray,
    target_index: int,
    *,
    eps: float = 1e-14,
) -> float:
    target_rate = float(K[target_index])
    off = np.delete(K, target_index)
    off_norm = float(np.linalg.norm(off))

    if abs(target_rate) <= eps:
        return float("inf")

    return float(off_norm / abs(target_rate))


def evaluate_solution(
    G_selected: np.ndarray,
    x: np.ndarray,
    *,
    target_index: int,
    target_sign: float,
    eps: float = 1e-14,
) -> dict:
    n_rows = G_selected.shape[0]

    target = np.zeros(n_rows)
    target[target_index] = target_sign

    K = G_selected @ x

    residual = float(np.linalg.norm(K - target))
    epsilon_spec = leakage_ratio(K, target_index, eps=eps)

    target_rate = float(K[target_index])
    total_intensity = float(np.sum(x))
    max_intensity = float(np.max(x)) if len(x) else 0.0
    active_count = int(np.sum(x > 1e-10))

    if total_intensity > eps:
        rate_per_total_intensity = abs(target_rate) / total_intensity
    else:
        rate_per_total_intensity = 0.0

    return {
        "target_sign": float(target_sign),
        "relative_residual": residual,
        "epsilon_spec": epsilon_spec,
        "target_rate": target_rate,
        "total_intensity": total_intensity,
        "max_intensity": max_intensity,
        "rate_per_total_intensity": float(rate_per_total_intensity),
        "active_count": active_count,
        "K": K,
    }


def solve_signed_nnls(
    G_selected: np.ndarray,
    *,
    target_index: int,
) -> tuple[np.ndarray, dict]:
    """
    Solve NNLS for both target signs and keep the better solution.

    The calibrated dispersive intensities are nonnegative,
    x_k = |Omega_k|^2 / 2 >= 0. We allow either sign of the synthesized
    target rate because the final target phase can be assigned with the
    corresponding sign convention.
    """
    n_rows = G_selected.shape[0]

    best_x = None
    best_metrics = None
    best_key = None

    for target_sign in (+1.0, -1.0):
        target = np.zeros(n_rows)
        target[target_index] = target_sign

        x, _ = nnls(G_selected, target)

        metrics = evaluate_solution(
            G_selected,
            x,
            target_index=target_index,
            target_sign=target_sign,
        )

        key = (
            metrics["epsilon_spec"]
            if math.isfinite(metrics["epsilon_spec"])
            else float("inf"),
            metrics["relative_residual"],
            metrics["total_intensity"],
        )

        if best_key is None or key < best_key:
            best_x = x
            best_metrics = metrics
            best_key = key

    return best_x, best_metrics


def greedy_sparse_target_synthesis(
    tones: np.ndarray,
    G: np.ndarray,
    *,
    target_index: int,
    max_tones: int,
    min_tone_spacing: float,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Greedily choose a sparse tone set. After each trial addition, solve NNLS.

    Objective:
        1. minimize spectator leakage ratio epsilon_spec
        2. minimize residual to signed unit target vector
        3. minimize total intensity
    """
    n_candidates = G.shape[1]

    if n_candidates == 0:
        return [], np.array([]), {
            "target_sign": 1.0,
            "relative_residual": float("inf"),
            "epsilon_spec": float("inf"),
            "target_rate": 0.0,
            "total_intensity": 0.0,
            "max_intensity": 0.0,
            "rate_per_total_intensity": 0.0,
            "active_count": 0,
            "K": np.zeros(G.shape[0]),
        }

    selected: list[int] = []
    remaining = list(range(n_candidates))

    best_x = np.array([])
    best_metrics = None

    while len(selected) < max_tones:
        best_candidate = None
        best_trial_selected = None
        best_trial_x = None
        best_trial_metrics = None
        best_key = None

        for j in remaining:
            if not respects_tone_spacing(
                tones,
                selected,
                j,
                min_tone_spacing=min_tone_spacing,
            ):
                continue

            trial_selected = selected + [j]
            G_trial = G[:, trial_selected]

            x_trial, metrics_trial = solve_signed_nnls(
                G_trial,
                target_index=target_index,
            )

            key = (
                metrics_trial["epsilon_spec"]
                if math.isfinite(metrics_trial["epsilon_spec"])
                else float("inf"),
                metrics_trial["relative_residual"],
                metrics_trial["total_intensity"],
            )

            if best_key is None or key < best_key:
                best_key = key
                best_candidate = j
                best_trial_selected = trial_selected
                best_trial_x = x_trial
                best_trial_metrics = metrics_trial

        if best_candidate is None:
            break

        selected = best_trial_selected
        remaining.remove(best_candidate)

        best_x = best_trial_x
        best_metrics = best_trial_metrics

    if best_metrics is None:
        best_x, best_metrics = solve_signed_nnls(
            G[:, selected],
            target_index=target_index,
        )

    return selected, best_x, best_metrics


def resolve_full_z_target(
    labels,
    n: int,
):
    """
    Identify the label corresponding to Z_1 ... Z_n without
    assuming one particular string convention.
    """
    labels = list(labels)

    candidates = [
        "".join(
            f"Z{i}"
            for i in range(1, n + 1)
        ),
        "Z" * n,
        "".join(
            f"Z_{i}"
            for i in range(1, n + 1)
        ),
    ]

    for candidate in candidates:
        if candidate in labels:
            return candidate

    # Most label conventions contain one Z symbol per supported
    # spin. The n-body Z string should therefore be unique.
    z_count_matches = [
        label
        for label in labels
        if str(label).count("Z") == n
    ]

    if len(z_count_matches) == 1:
        return z_count_matches[0]

    # Additional fallback for labels whose support is represented
    # primarily by explicit spin indices.
    support_matches = []

    for label in labels:
        numbers = {
            int(x)
            for x in re.findall(
                r"\d+",
                str(label),
            )
        }

        if numbers == set(
            range(1, n + 1)
        ):
            support_matches.append(
                label
            )

    if len(support_matches) == 1:
        return support_matches[0]

    raise KeyError(
        "Could not uniquely identify "
        f"the full n-body Z target "
        f"for n={n}.\n"
        f"Available labels: {labels}"
    )


def finite_tone_control_inversion_n(
    *,
    offsets: np.ndarray,
    n: int,
    max_tones: int,
    min_tone_spacing: float,
    span: float,
    grid_points: int,
    candidate_pool: int,
    min_detuning: float,
) -> dict:
    """
    n-spin extension of the existing finite-tone construction.

    The procedure itself is unchanged:

        admissible tone pool
        -> zero-response filtering
        -> deterministic downsampling
        -> greedy joint tone selection
        -> signed NNLS.

    Only the response-matrix dimension n is generalized.
    """
    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    tones = candidate_tones(
        offsets,
        num_grid=grid_points,
        span=span,
        min_detuning=min_detuning,
    )

    labels, G_pool = (
        transfer_matrix(
            offsets,
            tones,
            n=n,
            include_identity=False,
        )
    )

    target = resolve_full_z_target(
        labels,
        n,
    )

    tones, G_pool = (
        filter_nonzero_transfer_columns(
            tones,
            G_pool,
        )
    )

    tones_ds, G_ds = (
        downsample_candidates(
            tones,
            G_pool,
            candidate_pool=candidate_pool,
        )
    )

    target_index = labels.index(
        target
    )

    (
        selected_indices,
        x_unit,
        metrics,
    ) = greedy_sparse_target_synthesis(
        tones_ds,
        G_ds,
        target_index=target_index,
        max_tones=max_tones,
        min_tone_spacing=min_tone_spacing,
    )

    if not selected_indices:
        raise RuntimeError(
            "Finite-tone inversion "
            "selected no tones."
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    selected_tones = np.asarray(
        tones_ds[selected_indices],
        dtype=float,
    )

    x_unit = np.asarray(
        x_unit,
        dtype=float,
    )

    # Match the existing benchmark:
    # zero-NNLS-intensity tones are not physical control tones.
    active = x_unit > 1e-10

    selected_indices = (
        selected_indices[active]
    )

    selected_tones = (
        selected_tones[active]
    )

    x_unit = x_unit[active]

    if len(selected_tones) == 0:
        raise RuntimeError(
            "All selected tones received "
            "zero final NNLS intensity."
        )

    G_selected = G_ds[
        :,
        selected_indices,
    ]

    K_unit = (
        G_selected @ x_unit
    )

    target_rate_unit = float(
        K_unit[target_index]
    )

    if abs(target_rate_unit) <= 1e-15:
        raise RuntimeError(
            "Inversion produced zero "
            "target interaction rate."
        )

    return {
        "n": int(n),
        "labels": list(labels),
        "target": target,
        "target_index":
            int(target_index),
        "target_sign":
            float(metrics["target_sign"]),
        "tones": selected_tones,
        "intensities_unit": x_unit,
        "target_rate_unit":
            target_rate_unit,
        "relative_residual":
            float(
                metrics[
                    "relative_residual"
                ]
            ),
        "epsilon_spec":
            float(
                metrics[
                    "epsilon_spec"
                ]
            ),
        "total_intensity_unit":
            float(
                np.sum(x_unit)
            ),
        "active_count":
            int(len(selected_tones)),
        "K_unit": K_unit,
        "candidate_tones":
            np.asarray(
                tones_ds,
                dtype=float,
            ),
        "candidate_pool_size":
            int(len(tones_ds)),
    }


def local_intensity_caps(
    tones: np.ndarray,
    offsets: np.ndarray,
    *,
    r_disp: float,
    omega_hw_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    tones = np.asarray(tones, dtype=float)
    offsets = np.asarray(offsets, dtype=float)

    dmin = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    omega_cap = r_disp * dmin

    if math.isfinite(omega_hw_max):
        omega_cap = np.minimum(
            omega_cap,
            omega_hw_max,
        )

    xmax = 0.5 * omega_cap**2

    return xmax, dmin


def dispersive_scale_from_unit_control(
    *,
    tones: np.ndarray,
    intensities_unit: np.ndarray,
    offsets: np.ndarray,
    r_disp: float,
) -> dict:
    """
    Choose one global positive scale s such that

        x_k = s x_k^(unit)

    satisfies

        max_k sqrt(2 x_k) / d_k = r_disp,

    where

        d_k = min_alpha |omega_k - Lambda_alpha|.

    Relative NNLS intensities are left unchanged.
    """

    if r_disp <= 0.0:
        raise ValueError(
            "r_disp must be positive."
        )

    tones = np.asarray(
        tones,
        dtype=float,
    )

    x_unit = np.asarray(
        intensities_unit,
        dtype=float,
    )

    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    active = x_unit > 1e-14

    if not np.any(active):
        raise ValueError(
            "Control has no active intensities."
        )

    tones_active = tones[active]
    x_active = x_unit[active]

    dmin = np.min(
        np.abs(
            tones_active[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    if np.any(dmin <= 0.0):
        raise ValueError(
            "A selected tone lies exactly on a transition."
        )

    scale_caps = (
        (r_disp * dmin) ** 2
        / (2.0 * x_active)
    )

    drive_scale = float(
        np.min(scale_caps)
    )

    physical_intensities = (
        drive_scale * x_unit
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * physical_intensities,
            0.0,
        )
    )

    all_dmin = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    ratios = np.zeros_like(amplitudes)

    good = all_dmin > 0.0
    ratios[good] = (
        amplitudes[good]
        / all_dmin[good]
    )

    return {
        "drive_scale": drive_scale,
        "intensities": physical_intensities,
        "nearest_detunings": all_dmin,
        "local_dispersive_ratios": ratios,
        "max_dispersive_ratio": float(
            np.max(ratios)
        ),
    }
