from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from .core import candidate_tones
from .core import (
    response_matrix,
    transition_offsets_from_couplings,
)

from .tone_selection import (
    build_candidate_grid,
    solve_spacing_milp,
    refine_frequencies,
)

from .full_rwa_metrics import (
    conservative_n_steps,
)

from .aligned_dynamics import (
    blocks_to_full_unitary,
    final_max_ground_to_excited_flip,
    final_metrics_from_stage04,
    mediator_z_optimized_fidelity,
    propagate_final_blocks_jit,
)


from . import case_design as _case_design

filter_nonzero_response_columns = (
    _case_design.filter_nonzero_response_columns
)
downsample_candidates = (
    _case_design.downsample_candidates
)
greedy_sparse_target_synthesis = (
    _case_design.greedy_sparse_target_synthesis
)

DEFAULT_SEED = 20260818
DEFAULT_PHASE = math.pi / 4.0


# ----------------------------------------------------------------------
# JSON / IO
# ----------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return jsonable(value.item())

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(value), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def normalized_couplings(rho: float, sigma: float) -> np.ndarray:
    raw = np.array(
        [1.0, float(rho), float(sigma)],
        dtype=float,
    )
    return raw / np.sum(np.abs(raw))


def offsets_from_geometry(
    rho: float,
    sigma: float,
) -> np.ndarray:
    A = normalized_couplings(rho, sigma)
    return np.asarray(
        transition_offsets_from_couplings(A),
        dtype=float,
    )


def prepare_manifest(
    *,
    path: Path,
    n_geometries: int,
    seed: int,
    rho_min: float,
    rho_max: float,
    sigma_min: float,
    sigma_max: float,
    force: bool,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. "
            "Do not regenerate a frozen manifest. "
            "Use --force-manifest only if you intentionally want "
            "a new benchmark ensemble."
        )

    sampler = qmc.Sobol(
        d=2,
        scramble=True,
        seed=seed,
    )

    # Generate a large deterministic Sobol pool and then retain
    # the physical triangular domain sigma >= rho.
    m = 12
    unit_points = sampler.random_base2(m=m)

    if min(rho_min, rho_max, sigma_min, sigma_max) <= 0.0:
        raise ValueError(
            "Log-space geometry sampling requires positive ratio bounds."
        )

    # Sample uniformly in logarithmic ratio coordinates.  Equal areas in
    # this design correspond to comparable multiplicative changes in the
    # coupling hierarchy rather than comparable additive changes.
    log_scaled = qmc.scale(
        unit_points,
        [math.log10(rho_min), math.log10(sigma_min)],
        [math.log10(rho_max), math.log10(sigma_max)],
    )

    geometries = []

    for log_rho, log_sigma in log_scaled:
        rho = float(10.0 ** log_rho)
        sigma = float(10.0 ** log_sigma)

        if sigma < rho:
            continue

        offsets = offsets_from_geometry(rho, sigma)

        # Reject accidental degeneracies only; this is not
        # performance-based filtering.
        ordered = np.sort(offsets)
        min_spacing = float(np.min(np.diff(ordered)))

        if min_spacing <= 1e-10:
            continue

        geometries.append(
            {
                "geometry_id": (
                    f"geometry_{len(geometries) + 1:03d}"
                ),
                "rho": rho,
                "sigma": sigma,
                "couplings": normalized_couplings(
                    rho,
                    sigma,
                ).tolist(),
                "offsets": offsets.tolist(),
                "minimum_transition_spacing": min_spacing,
            }
        )

        if len(geometries) >= n_geometries:
            break

    if len(geometries) != n_geometries:
        raise RuntimeError(
            f"Only generated {len(geometries)} admissible "
            f"geometries, expected {n_geometries}."
        )

    manifest = {
        "schema_version": 1,
        "kind": "geometry_control_inversion_manifest",
        "selection_rule": (
            "Scrambled Sobol points prescribed independently of "
            "control synthesis, response diagnostics, or fidelity; "
            "sampled uniformly in logarithmic coupling-ratio coordinates "
            "and restricted only to sigma >= rho and nondegenerate "
            "transition spectra."
        ),
        "seed": int(seed),
        "domain": {
            "rho_min": rho_min,
            "rho_max": rho_max,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "constraint": "sigma >= rho",
            "sampling_coordinates": "log10(rho), log10(sigma)",
        },
        "n_geometries": n_geometries,
        "geometries": geometries,
    }

    write_json_atomic(path, manifest)

    print()
    print("[manifest prepared]")
    print(f"  path          : {path}")
    print(f"  geometries    : {n_geometries}")
    print(f"  seed          : {seed}")
    print(
        f"  rho range     : "
        f"[{rho_min:.3f}, {rho_max:.3f}]"
    )
    print(
        f"  sigma range   : "
        f"[{sigma_min:.3f}, {sigma_max:.3f}]"
    )
    print("  constraint    : sigma >= rho")


# ----------------------------------------------------------------------
# Physical control constraints
# ----------------------------------------------------------------------


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


def control_resource_metrics(
    *,
    tones: np.ndarray,
    intensities: np.ndarray,
    offsets: np.ndarray,
    T_gate: float,
) -> dict:
    tones = np.asarray(tones, dtype=float)
    intensities = np.asarray(
        intensities,
        dtype=float,
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    dmin = np.min(
        np.abs(
            tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    ratios = np.zeros_like(amplitudes)

    valid = dmin > 0.0
    ratios[valid] = (
        amplitudes[valid]
        / dmin[valid]
    )

    max_amplitude = (
        float(np.max(amplitudes))
        if amplitudes.size
        else 0.0
    )

    return {
        "tone_count": int(len(tones)),
        "active_tone_count": int(
            np.sum(
                intensities > 1e-14
            )
        ),
        "total_intensity": float(
            np.sum(intensities)
        ),
        "max_amplitude": max_amplitude,
        "max_local_dispersive_ratio": (
            float(np.max(ratios))
            if ratios.size
            else 0.0
        ),
        "N_Rabi": float(
            T_gate
            * max_amplitude
            / (2.0 * math.pi)
        ),
    }



# ----------------------------------------------------------------------
# Finite-tone control inversion
# ----------------------------------------------------------------------


def require_inversion_fields(
    result: Any,
    *,
    context: str,
) -> dict:
    if not isinstance(result, dict):
        raise TypeError(
            f"{context} returned "
            f"{type(result).__name__}, expected dict."
        )

    required = (
        "tones",
        "intensities",
        "kappa",
    )

    missing = [
        key
        for key in required
        if key not in result
    ]

    if missing:
        raise KeyError(
            f"{context} missing keys {missing}. "
            f"Available keys: {sorted(result.keys())}"
        )

    return result


def solve_one_sign(
    *,
    offsets: np.ndarray,
    target: str,
    target_sign: float,
    max_tones: int,
    r_disp: float,
    min_tone_spacing: float,
    omega_hw_max: float,
    span: float,
    grid_points: int,
    refine_passes: int,
    refine_points: int,
    milp_time_limit: float,
) -> dict | None:
    candidate_tones = build_candidate_grid(
        span=span,
        grid_points=grid_points,
        offsets=offsets,
    )

    coarse = solve_spacing_milp(
        offsets=offsets,
        candidate_tones=candidate_tones,
        target=target,
        target_sign=target_sign,
        max_tones=max_tones,
        r_disp=r_disp,
        min_spacing=min_tone_spacing,
        omega_hw_max=omega_hw_max,
        time_limit=milp_time_limit,
    )

    if not bool(coarse.get("success", False)):
        return None

    coarse = require_inversion_fields(
        coarse,
        context="solve_spacing_milp",
    )

    refined = refine_frequencies(
        offsets=offsets,
        tones=np.asarray(
            coarse["tones"],
            dtype=float,
        ),
        target=target,
        target_sign=target_sign,
        r_disp=r_disp,
        omega_hw_max=omega_hw_max,
        span=span,
        min_spacing=min_tone_spacing,
        passes=refine_passes,
        points=refine_points,
    )

    refined = require_inversion_fields(
        refined,
        context="refine_frequencies",
    )

    out = dict(refined)
    out["target_sign"] = float(target_sign)
    out["coarse_kappa"] = float(
        coarse["kappa"]
    )
    out["coarse_status"] = coarse.get("status")
    out["coarse_message"] = coarse.get("message")
    out["coarse_mip_gap"] = coarse.get("mip_gap")
    out["coarse_mip_node_count"] = coarse.get(
        "mip_node_count"
    )

    return out


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


def finite_tone_control_inversion(
    *,
    offsets: np.ndarray,
    target: str,
    max_tones: int,
    min_tone_spacing: float,
    span: float,
    grid_points: int,
    candidate_pool: int,
    min_detuning: float,
) -> dict:
    """
    Original finite-tone construction:

        admissible tone pool
        -> remove zero-response columns
        -> deterministic downsampling
        -> greedy joint tone selection
        -> signed NNLS.
    """

    tones = candidate_tones(
        offsets,
        num_grid=grid_points,
        span=span,
        min_detuning=min_detuning,
    )

    labels, G_pool = response_matrix(
        offsets,
        tones,
        n=3,
        include_identity=False,
    )

    tones, G_pool = (
        filter_nonzero_response_columns(
            tones,
            G_pool,
        )
    )

    tones_ds, G_ds = downsample_candidates(
        tones,
        G_pool,
        candidate_pool=candidate_pool,
    )

    if target not in labels:
        raise KeyError(
            f"Target {target!r} not available. "
            f"Labels: {labels}"
        )

    target_index = labels.index(target)

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
            "Greedy finite-tone inversion "
            "selected no tones."
        )

    selected_tones = np.asarray(
        tones_ds[selected_indices],
        dtype=float,
    )

    x_unit = np.asarray(
        x_unit,
        dtype=float,
    )

    # Greedy selection can retain tones to which the final
    # NNLS solution assigns zero intensity.  They are not
    # physical control tones and should not count as a
    # resource in the matched random baseline.
    selected_indices_array = np.asarray(
        selected_indices,
        dtype=int,
    )

    active = x_unit > 1e-10

    selected_tones = selected_tones[active]
    x_unit = x_unit[active]

    selected_indices_active = (
        selected_indices_array[active]
    )

    target_rate_unit = float(
        metrics["target_rate"]
    )

    if (
        not math.isfinite(
            target_rate_unit
        )
        or abs(target_rate_unit) <= 1e-14
    ):
        raise RuntimeError(
            "Greedy finite-tone inversion "
            "produced zero target rate."
        )

    total_intensity_unit = float(
        np.sum(x_unit)
    )

    if total_intensity_unit > 0.0:
        normalized_x = (
            x_unit
            / total_intensity_unit
        )
    else:
        normalized_x = (
            np.zeros_like(x_unit)
        )

    return {
        "labels": list(labels),
        "target_index": int(
            target_index
        ),
        "target_sign": float(
            metrics["target_sign"]
        ),
        "tones": selected_tones,
        "intensities_unit": x_unit,
        "normalized_intensities": (
            normalized_x
        ),
        "target_rate_unit": (
            target_rate_unit
        ),
        "total_intensity_unit": (
            total_intensity_unit
        ),
        "relative_residual": float(
            metrics[
                "relative_residual"
            ]
        ),
        "epsilon_spec": float(
            metrics["epsilon_spec"]
        ),
        "active_count": int(
            metrics["active_count"]
        ),
        "K_unit": np.asarray(
            metrics["K"],
            dtype=float,
        ),
        "num_admissible_tones": int(
            len(tones)
        ),
        "num_candidate_tones": int(
            len(tones_ds)
        ),
        "candidate_tones": np.asarray(
            tones_ds,
            dtype=float,
        ),
        "selected_indices": [
            int(i)
            for i in selected_indices_active
        ],
    }




def effective_control_metrics(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    intensities: np.ndarray,
    target: str,
) -> dict:
    labels, G = response_matrix(
        offsets,
        tones,
        n=3,
        include_identity=False,
    )

    rates = G @ intensities

    if target not in labels:
        raise KeyError(
            f"Target {target!r} not present in "
            f"response labels {labels}."
        )

    target_index = labels.index(target)
    target_rate = float(rates[target_index])

    spectators = np.delete(
        rates,
        target_index,
    )

    spectator_rms = (
        float(
            np.sqrt(
                np.mean(spectators**2)
            )
        )
        if spectators.size
        else 0.0
    )

    spectator_l2 = float(
        np.linalg.norm(spectators)
    )

    denom = max(
        abs(target_rate),
        1e-300,
    )

    return {
        "labels": list(labels),
        "rates": rates.tolist(),
        "target_rate": target_rate,
        "spectator_rate_rms": spectator_rms,
        "spectator_rate_l2": spectator_l2,
        "relative_spectator_l2": (
            spectator_l2 / denom
        ),
    }


# ----------------------------------------------------------------------
# Random multi-tone baseline
# ----------------------------------------------------------------------


def sample_random_tones(
    *,
    rng: np.random.Generator,
    candidate_tones_pool: np.ndarray,
    M: int,
    min_spacing: float,
    max_attempts: int = 100000,
) -> np.ndarray:
    pool = np.asarray(
        candidate_tones_pool,
        dtype=float,
    )

    if M > len(pool):
        raise ValueError(
            f"Cannot draw {M} tones "
            f"from pool of size {len(pool)}."
        )

    for _ in range(max_attempts):
        indices = rng.choice(
            len(pool),
            size=M,
            replace=False,
        )

        tones = np.sort(
            pool[indices]
        )

        if M <= 1:
            return tones

        if (
            np.min(np.diff(tones))
            >= min_spacing
        ):
            return tones

    raise RuntimeError(
        "Could not sample a "
        "spacing-feasible random "
        "tone subset."
    )



def sample_random_control(
    *,
    rng: np.random.Generator,
    candidate_tones_pool: np.ndarray,
    offsets: np.ndarray,
    M: int,
    min_spacing: float,
    total_intensity: float,
    r_disp: float,
    max_attempts: int = 100000,
) -> dict:
    if total_intensity <= 0.0:
        raise ValueError(
            "total_intensity must be positive."
        )

    if r_disp <= 0.0:
        raise ValueError(
            "r_disp must be positive."
        )

    offsets = np.asarray(
        offsets,
        dtype=float,
    )

    for _ in range(max_attempts):

        tones = sample_random_tones(
            rng=rng,
            candidate_tones_pool=(
                candidate_tones_pool
            ),
            M=M,
            min_spacing=min_spacing,
        )

        # Uniform draw on the nonnegative intensity simplex.
        fractions = rng.dirichlet(
            np.ones(M, dtype=float)
        )

        intensities = (
            total_intensity
            * fractions
        )

        amplitudes = np.sqrt(
            np.maximum(
                2.0 * intensities,
                0.0,
            )
        )

        dmin = np.min(
            np.abs(
                tones[:, None]
                - offsets[None, :]
            ),
            axis=1,
        )

        if np.any(dmin <= 0.0):
            continue

        ratios = amplitudes / dmin

        max_ratio = float(
            np.max(ratios)
        )

        # Matched random controls may use LESS than the
        # allowed dispersive strength, but never more.
        if max_ratio <= r_disp * (1.0 + 1e-12):
            return {
                "tones": tones,
                "intensity_fractions": (
                    fractions
                ),
                "intensities": intensities,
                "nearest_detunings": dmin,
                "local_dispersive_ratios": (
                    ratios
                ),
                "max_dispersive_ratio": (
                    max_ratio
                ),
            }

    raise RuntimeError(
        "Could not draw a matched-power random control "
        "within the dispersive-ratio bound."
    )




# ----------------------------------------------------------------------
# Full aligned-RWA evaluation
# ----------------------------------------------------------------------


def full_rwa_evaluate(
    *,
    offsets: np.ndarray,
    tones: np.ndarray,
    intensities: np.ndarray,
    T_gate: float,
    target_label: str,
    signed_target_phase: float,
    span: float,
    steps_per_period: int,
    max_steps: int,
) -> dict:
    total_intensity = float(
        np.sum(intensities)
    )

    n_steps = conservative_n_steps(
        offsets=offsets,
        span=span,
        total_intensity_max=max(
            total_intensity,
            1e-16,
        ),
        T_gate=T_gate,
        steps_per_period=steps_per_period,
        max_steps=max_steps,
    )

    blocks = propagate_final_blocks_jit(
        np.asarray(offsets, dtype=float),
        np.asarray(tones, dtype=float),
        np.asarray(intensities, dtype=float),
        float(T_gate),
        int(n_steps),
    )

    fidelity_corr = float(
        mediator_z_optimized_fidelity(
            blocks,
            target_label=target_label,
            signed_target_phase=(
                signed_target_phase
            ),
        )
    )

    U_full = blocks_to_full_unitary(blocks)

    metrics = final_metrics_from_stage04(
        U_full=U_full,
        target_label=target_label,
        signed_target_phase=(
            signed_target_phase
        ),
    )

    final_flip = float(
        final_max_ground_to_excited_flip(
            blocks
        )
    )

    return {
        "mediator_z_corrected_fidelity": (
            fidelity_corr
        ),
        "final_max_ground_to_excited_flip": (
            final_flip
        ),
        "n_steps": int(n_steps),
        "stage04_final_metrics": metrics,
    }


# ----------------------------------------------------------------------
# One geometry
# ----------------------------------------------------------------------


def run_geometry(
    *,
    geometry: dict,
    target: str,
    target_phase: float,
    n_random: int,
    max_tones: int,
    min_tone_spacing: float,
    span: float,
    grid_points: int,
    candidate_pool: int,
    min_detuning: float,
    r_disp: float,
    steps_per_period: int,
    max_steps: int,
    seed: int,
) -> dict:
    geometry_id = geometry[
        "geometry_id"
    ]

    rho = float(
        geometry["rho"]
    )

    sigma = float(
        geometry["sigma"]
    )

    offsets = np.asarray(
        geometry["offsets"],
        dtype=float,
    )

    print()
    print("=" * 72)
    print(
        f"{geometry_id} | "
        f"rho={rho:.6f} | "
        f"sigma={sigma:.6f} | "
        f"target={target}"
    )
    print("=" * 72)

    t0 = time.perf_counter()

    inversion = (
        finite_tone_control_inversion(
            offsets=offsets,
            target=target,
            max_tones=max_tones,
            min_tone_spacing=(
                min_tone_spacing
            ),
            span=span,
            grid_points=grid_points,
            candidate_pool=(
                candidate_pool
            ),
            min_detuning=min_detuning,
        )
    )

    tones_inv = np.asarray(
        inversion["tones"],
        dtype=float,
    )

    x_unit_inv = np.asarray(
        inversion["intensities_unit"],
        dtype=float,
    )

    dispersive_scaling = (
        dispersive_scale_from_unit_control(
            tones=tones_inv,
            intensities_unit=x_unit_inv,
            offsets=offsets,
            r_disp=r_disp,
        )
    )

    drive_scale = float(
        dispersive_scaling[
            "drive_scale"
        ]
    )

    intensities_inv = np.asarray(
        dispersive_scaling[
            "intensities"
        ],
        dtype=float,
    )

    actual_r_disp = float(
        dispersive_scaling[
            "max_dispersive_ratio"
        ]
    )

    target_rate_unit = float(
        inversion["target_rate_unit"]
    )

    target_rate = (
        drive_scale
        * target_rate_unit
    )

    if abs(target_rate) <= 1e-14:
        raise RuntimeError(
            "Physical inversion target "
            "rate is zero."
        )

    signed_target_phase = math.copysign(
        abs(target_phase),
        target_rate,
    )

    T_gate = (
        2.0
        * abs(target_phase)
        / abs(target_rate)
    )

    effective = effective_control_metrics(
        offsets=offsets,
        tones=tones_inv,
        intensities=intensities_inv,
        target=target,
    )

    inv_resource = (
        control_resource_metrics(
            tones=tones_inv,
            intensities=(
                intensities_inv
            ),
            offsets=offsets,
            T_gate=T_gate,
        )
    )

    inv_full = full_rwa_evaluate(
        offsets=offsets,
        tones=tones_inv,
        intensities=intensities_inv,
        T_gate=T_gate,
        target_label=target,
        signed_target_phase=(
            signed_target_phase
        ),
        span=span,
        steps_per_period=(
            steps_per_period
        ),
        max_steps=max_steps,
    )

    F_inv = float(
        inv_full[
            "mediator_z_corrected_fidelity"
        ]
    )

    print(
        f"[inversion] "
        f"eps={inversion['epsilon_spec']:.3e}  "
        f"res={inversion['relative_residual']:.3e}  "
        f"K={target_rate:.6e}  "
        f"T={T_gate:.6f}  "
        f"F={F_inv:.9f}  "
        f"tones={len(tones_inv)}  "
        f"active={inversion['active_count']}  "
        f"r_disp={actual_r_disp:.4f}"
    )

    geometry_number = int(
        geometry_id.split("_")[-1]
    )

    rng = np.random.default_rng(
        int(seed)
        + 1000003 * geometry_number
    )

    random_controls = []

    best_random_index = None
    best_random_fidelity = (
        -math.inf
    )

    M_random = int(
        len(tones_inv)
    )

    print(
        f"[random] evaluating "
        f"{n_random} matched controls..."
    )

    for random_index in range(
        n_random
    ):
        control = sample_random_control(
            rng=rng,
            candidate_tones_pool=(
                np.asarray(
                    inversion[
                        "candidate_tones"
                    ],
                    dtype=float,
                )
            ),
            offsets=offsets,
            M=M_random,
            min_spacing=(
                min_tone_spacing
            ),
            total_intensity=float(
                np.sum(
                    intensities_inv
                )
            ),
            r_disp=r_disp,
        )

        full = full_rwa_evaluate(
            offsets=offsets,
            tones=control["tones"],
            intensities=(
                control["intensities"]
            ),
            T_gate=T_gate,
            target_label=target,
            signed_target_phase=(
                signed_target_phase
            ),
            span=span,
            steps_per_period=(
                steps_per_period
            ),
            max_steps=max_steps,
        )

        fidelity = float(
            full[
                "mediator_z_corrected_fidelity"
            ]
        )

        resource = (
            control_resource_metrics(
                tones=control["tones"],
                intensities=control[
                    "intensities"
                ],
                offsets=offsets,
                T_gate=T_gate,
            )
        )

        effective_random = (
            effective_control_metrics(
                offsets=offsets,
                tones=control[
                    "tones"
                ],
                intensities=control[
                    "intensities"
                ],
                target=target,
            )
        )

        row = {
            "random_index": (
                random_index
            ),
            "tones": control[
                "tones"
            ],
            "intensity_fractions": (
                control[
                    "intensity_fractions"
                ]
            ),
            "intensities": control[
                "intensities"
            ],
            "effective_model": (
                effective_random
            ),
            "resource_metrics": (
                resource
            ),
            "full_rwa": full,
        }

        random_controls.append(row)

        if (
            fidelity
            > best_random_fidelity
        ):
            best_random_fidelity = (
                fidelity
            )
            best_random_index = (
                random_index
            )

        if (
            (random_index + 1) % 10
            == 0
            or random_index + 1
            == n_random
        ):
            print(
                f"  {random_index + 1:4d}/"
                f"{n_random}: "
                f"best F="
                f"{best_random_fidelity:.9f}"
            )

    random_fidelities = np.array(
        [
            row["full_rwa"][
                "mediator_z_corrected_fidelity"
            ]
            for row in random_controls
        ],
        dtype=float,
    )

    F_best = float(
        np.max(random_fidelities)
    )

    F_median = float(
        np.median(random_fidelities)
    )

    delta_f = F_inv - F_best

    inv_infidelity = max(
        1.0 - F_inv,
        1e-15,
    )

    random_best_infidelity = max(
        1.0 - F_best,
        0.0,
    )

    infidelity_ratio = (
        random_best_infidelity
        / inv_infidelity
    )

    elapsed = (
        time.perf_counter() - t0
    )

    print()
    print("[comparison 0]")
    print(
        f"  inversion F       : "
        f"{F_inv:.9f}"
    )
    print(
        f"  best random F     : "
        f"{F_best:.9f}"
    )
    print(
        f"  median random F   : "
        f"{F_median:.9f}"
    )
    print(
        f"  delta F           : "
        f"{delta_f:+.9f}"
    )
    print(
        f"  elapsed           : "
        f"{elapsed:.2f} s"
    )

    return {
        "geometry_id": geometry_id,
        "rho": rho,
        "sigma": sigma,
        "offsets": offsets,
        "target": target,
        "target_phase_magnitude": (
            abs(target_phase)
        ),
        "selected_target_sign": (
            float(
                inversion[
                    "target_sign"
                ]
            )
        ),
        "signed_target_phase": (
            signed_target_phase
        ),
        "gate_time": T_gate,
        "r_disp_target": r_disp,
        "drive_scale": drive_scale,
        "inversion": {
            "tones": tones_inv,
            "intensities_unit": (
                x_unit_inv
            ),
            "intensities": (
                intensities_inv
            ),
            "target_rate_unit": (
                target_rate_unit
            ),
            "target_rate": (
                target_rate
            ),
            "epsilon_spec": (
                inversion[
                    "epsilon_spec"
                ]
            ),
            "relative_residual": (
                inversion[
                    "relative_residual"
                ]
            ),
            "active_count": (
                inversion[
                    "active_count"
                ]
            ),
            "candidate_pool_size": (
                inversion[
                    "num_candidate_tones"
                ]
            ),
            "max_dispersive_ratio": (
                actual_r_disp
            ),
            "local_dispersive_ratios": (
                dispersive_scaling[
                    "local_dispersive_ratios"
                ]
            ),
            "nearest_detunings": (
                dispersive_scaling[
                    "nearest_detunings"
                ]
            ),
            "effective_model": effective,
            "resource_metrics": (
                inv_resource
            ),
            "full_rwa": inv_full,
        },
        "random_sampling": {
            "n_random": n_random,
            "matched_resources": {
                "tone_count": M_random,
                "total_intensity": float(
                    np.sum(
                        intensities_inv
                    )
                ),
                "r_disp_cap": r_disp,
                "gate_time": T_gate,
                "candidate_tone_pool": (
                    "same downsampled "
                    "admissible tone pool "
                    "as inversion"
                ),
            },
            "distribution": {
                "tones": (
                    "uniform random feasible "
                    "subset of the same "
                    "candidate-tone pool"
                ),
                "intensities": (
                    "uniform Dirichlet "
                    "partition of the "
                    "inversion control's "
                    "total intensity"
                ),
            },
            "controls": (
                random_controls
            ),
            "fidelity_summary": {
                "best": F_best,
                "best_index": (
                    best_random_index
                ),
                "median": F_median,
                "q10": float(
                    np.quantile(
                        random_fidelities,
                        0.10,
                    )
                ),
                "q90": float(
                    np.quantile(
                        random_fidelities,
                        0.90,
                    )
                ),
            },
        },
        "comparison0": {
            "F_inversion": F_inv,
            "F_random_best": F_best,
            "F_random_median": (
                F_median
            ),
            "delta_F_inversion_minus_best_random": (
                delta_f
            ),
            "random_to_inversion_infidelity_ratio": (
                infidelity_ratio
            ),
        },
        "elapsed_seconds": elapsed,
    }




# ----------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------


def run_comparison(
    args: argparse.Namespace,
) -> None:
    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run --mode prepare first."
        )

    manifest = load_json(manifest_path)

    geometries = list(
        manifest["geometries"]
    )

    if args.mode == "smoke":
        n_random = (
            args.n_random
            if args.n_random is not None
            else 10
        )

        max_cases = (
            args.max_cases
            if args.max_cases is not None
            else 3
        )

        output_path = Path(
            args.output
            or (
                "results/benchmarks/"
                "geometry_control_comparison0_smoke.json"
            )
        )

    else:
        n_random = (
            args.n_random
            if args.n_random is not None
            else 100
        )

        max_cases = (
            args.max_cases
            if args.max_cases is not None
            else len(geometries)
        )

        output_path = Path(
            args.output
            or (
                "results/benchmarks/"
                "geometry_control_comparison0.json"
            )
        )

    geometries = geometries[:max_cases]

    config = {
        "construction": (
            "greedy_signed_nnls"
        ),
        "target": args.target,
        "target_phase": args.target_phase,
        "n_random": n_random,
        "max_tones": args.max_tones,
        "span": args.span,
        "min_detuning": (
            args.min_detuning
        ),
        "min_tone_spacing": (
            args.min_tone_spacing
        ),
        "grid_points": (
            args.grid_points
        ),
        "candidate_pool": (
            args.candidate_pool
        ),
        "r_disp": (
            args.r_disp
        ),
        "steps_per_period": (
            args.steps_per_period
        ),
        "max_steps": args.max_steps,
        "random_seed": args.seed,
    }

    if output_path.exists():
        result = load_json(output_path)

        old_config = result.get(
            "config",
            {},
        )

        if old_config != jsonable(config):
            raise RuntimeError(
                "Existing output has a different "
                "benchmark configuration. "
                "Remove it or choose another "
                "--output path."
            )

        done = {
            row["geometry_id"]
            for row in result.get(
                "cases",
                [],
            )
        }

        print(
            f"[resume] {len(done)} completed "
            f"cases found in {output_path}"
        )

    else:
        result = {
            "schema_version": 2,
            "kind": (
                "geometry_control_inversion_"
                "comparison0"
            ),
            "taxonomy": {
                "finite_tone_control_inversion": (
                    "Construction of a finite "
                    "multi-tone control from the "
                    "dispersive response matrix by "
                    "greedy joint tone selection "
                    "followed by signed nonnegative "
                    "least squares."
                ),
                "inversion_derived_control": (
                    "The resulting multi-tone "
                    "control before optimization "
                    "against the full driven "
                    "dynamics."
                ),
                "randomly_sampled_multitone_control": (
                    "A matched-resource multi-tone "
                    "control whose tones and "
                    "relative intensities are "
                    "sampled without response-matrix "
                    "guided inversion."
                ),
            },
            "manifest": str(manifest_path),
            "config": config,
            "cases": [],
        }

        done = set()

    for geometry in geometries:
        geometry_id = geometry[
            "geometry_id"
        ]

        if geometry_id in done:
            print(
                f"[skip] {geometry_id} "
                f"already complete"
            )
            continue

        try:
            case = run_geometry(
                geometry=geometry,
                target=args.target,
                target_phase=(
                    args.target_phase
                ),
                n_random=n_random,
                max_tones=args.max_tones,
                min_tone_spacing=(
                    args.min_tone_spacing
                ),
                span=args.span,
                grid_points=(
                    args.grid_points
                ),
                candidate_pool=(
                    args.candidate_pool
                ),
                min_detuning=(
                    args.min_detuning
                ),
                r_disp=(
                    args.r_disp
                ),
                steps_per_period=(
                    args.steps_per_period
                ),
                max_steps=(
                    args.max_steps
                ),
                seed=args.seed,
            )

        except Exception as exc:
            case = {
                "geometry_id": (
                    geometry_id
                ),
                "rho": geometry["rho"],
                "sigma": geometry[
                    "sigma"
                ],
                "target": args.target,
                "failed": True,
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(exc),
            }

            result["cases"].append(
                case
            )

            write_json_atomic(
                output_path,
                result,
            )

            print()
            print(
                f"[FAILED] {geometry_id}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if args.mode == "smoke":
                raise

            continue

        result["cases"].append(case)

        write_json_atomic(
            output_path,
            result,
        )

        print(
            f"[checkpoint] "
            f"{output_path}"
        )

    print()
    print("[benchmark complete]")
    print(
        f"  output: {output_path}"
    )

    summarize_result(result)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def summarize_result(result: dict) -> None:
    cases = [
        row
        for row in result.get("cases", [])
        if not row.get("failed", False)
    ]

    if not cases:
        print("[summary] no completed cases")
        return

    F_inv = np.array(
        [
            row["comparison0"]["F_inversion"]
            for row in cases
        ],
        dtype=float,
    )

    F_best = np.array(
        [
            row["comparison0"]["F_random_best"]
            for row in cases
        ],
        dtype=float,
    )

    delta = F_inv - F_best

    # Use the actual number of random controls in this run.
    n_random = result.get(
        "config",
        {},
    ).get(
        "n_random",
        None,
    )

    if n_random is None:
        n_random = cases[0].get(
            "random_sampling",
            {},
        ).get(
            "n_random",
            "?",
        )

    print()
    print("=" * 72)
    print("COMPARISON 0 SUMMARY")
    print("=" * 72)

    print(
        f"completed geometries            : "
        f"{len(cases)}"
    )

    print(
        f"median inversion F             : "
        f"{np.median(F_inv):.9f}"
    )

    print(
        f"minimum inversion F            : "
        f"{np.min(F_inv):.9f}"
    )

    print(
        f"median best-of-{n_random} random F"
        f" : {np.median(F_best):.9f}"
    )

    print(
        f"median delta F                 : "
        f"{np.median(delta):+.9f}"
    )

    print(
        f"inversion > best-of-{n_random} random"
        f" : {np.mean(delta > 0.0):.3f}"
    )

    for threshold in (
        0.99,
        0.999,
        0.9999,
    ):
        print(
            f"P(F_inv >= {threshold:.4f})"
            f"               : "
            f"{np.mean(F_inv >= threshold):.3f}"
        )

        print(
            f"P(F_best-of-{n_random} >= {threshold:.4f})"
            f"        : "
            f"{np.mean(F_best >= threshold):.3f}"
        )



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comparison 0: original greedy signed-NNLS "
            "finite-tone control inversion versus "
            "matched randomly sampled multi-tone "
            "controls under full aligned-RWA "
            "propagation."
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "prepare",
            "smoke",
            "comparison0",
            "summarize",
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "results/benchmarks/"
            "geometry_control_manifest.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--target",
        default="Z1Z2Z3",
    )

    parser.add_argument(
        "--target-phase",
        type=float,
        default=DEFAULT_PHASE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--n-geometries",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rho-min",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--rho-max",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--sigma-min",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sigma-max",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--force-manifest",
        action="store_true",
    )

    parser.add_argument(
        "--n-random",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    # Original finite-tone synthesis defaults.
    parser.add_argument(
        "--max-tones",
        type=int,
        default=7,
        help=(
            "Maximum finite-tone synthesis budget. "
            "For the three-spin benchmark we use M=7, "
            "the worst-case support bound of the seven-dimensional "
            "nontrivial diagonal interaction space."
        ),
    )

    parser.add_argument(
        "--span",
        type=float,
        default=1.35,
    )

    parser.add_argument(
        "--min-detuning",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--min-tone-spacing",
        type=float,
        default=0.04,
    )

    parser.add_argument(
        "--grid-points",
        type=int,
        default=401,
    )

    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=151,
    )

    parser.add_argument(
        "--r-disp",
        type=float,
        default=0.10,
        help=(
            "Maximum local dispersive ratio "
            "max_k |Omega_k| / min_alpha |Delta_kalpha|."
        ),
    )

    parser.add_argument(
        "--steps-per-period",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=200000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "prepare":
        prepare_manifest(
            path=Path(args.manifest),
            n_geometries=args.n_geometries,
            seed=args.seed,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            force=args.force_manifest,
        )
        return

    if args.mode == "summarize":
        if args.output is None:
            path = Path(
                "results/benchmarks/"
                "geometry_control_comparison0.json"
            )
        else:
            path = Path(args.output)

        summarize_result(
            load_json(path)
        )
        return

    run_comparison(args)


if __name__ == "__main__":
    main()
