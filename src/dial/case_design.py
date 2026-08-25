from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from .core import (
    DATA_DIR,
    TARGET_REGISTER_PHASE,
    candidate_tones,
    ensure_dirs,
    normalized_couplings_from_ratios,
    response_matrix,
    transition_offsets_from_couplings,
    write_csv,
)


GEOMETRY_LABEL_ORDER = [
    "lower diversity",
    "intermediate diversity",
    "upper diversity",
]


DEFAULT_TARGETS = [
    "Z3",
    "Z2",
    "Z2Z3",
    "Z1",
    "Z1Z3",
    "Z1Z2",
    "Z1Z2Z3",
]


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def pauli_order(label: str) -> int:
    return label.count("Z")


def target_type(label: str) -> str:
    order = pauli_order(label)
    if order == 1:
        return "single"
    if order == 2:
        return "pair"
    if order == 3:
        return "triple"
    return f"order{order}"


def sort_representatives(rows: list[dict]) -> list[dict]:
    order = {label: i for i, label in enumerate(GEOMETRY_LABEL_ORDER)}
    return sorted(rows, key=lambda r: order.get(r.get("label", ""), 999))


def filter_nonzero_response_columns(
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


def selectivity_score(
    epsilon_spec: float,
    *,
    score_ceiling: float,
) -> float:
    if not math.isfinite(epsilon_spec):
        return 0.0

    if epsilon_spec <= 0:
        return float(score_ceiling)

    return float(min(max(-math.log10(epsilon_spec), 0.0), score_ceiling))


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


def design_validation_cases(
    reps: list[dict],
    *,
    targets: list[str],
    num_grid: int,
    candidate_pool: int,
    max_tones: int,
    span: float,
    min_detuning: float,
    min_tone_spacing: float,
    target_register_phase: float,
    score_ceiling: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    case_rows: list[dict] = []
    control_rows: list[dict] = []
    rate_rows: list[dict] = []

    reps = sort_representatives(reps)

    case_id = 0

    for rep in reps:
        geometry_label = rep["label"]
        rho = to_float(rep, "rho")
        sigma = to_float(rep, "sigma")

        print(
            f"[case design] {geometry_label}: rho={rho:.3f}, sigma={sigma:.3f}",
            flush=True,
        )

        A = normalized_couplings_from_ratios(rho, sigma, B=1.0)
        offsets = transition_offsets_from_couplings(A)

        tones = candidate_tones(
            offsets,
            num_grid=num_grid,
            span=span,
            min_detuning=min_detuning,
        )

        labels, G_pool = response_matrix(
            offsets,
            tones,
            include_identity=False,
        )

        tones, G_pool = filter_nonzero_response_columns(tones, G_pool)

        tones_ds, G_ds = downsample_candidates(
            tones,
            G_pool,
            candidate_pool=candidate_pool,
        )

        label_to_index = {label: i for i, label in enumerate(labels)}

        for target_label in targets:
            if target_label not in label_to_index:
                raise ValueError(
                    f"Target {target_label} is not available. "
                    f"Available labels: {labels}"
                )

            case_id += 1

            target_index = label_to_index[target_label]

            selected_indices, x_unit, metrics = greedy_sparse_target_synthesis(
                tones_ds,
                G_ds,
                target_index=target_index,
                max_tones=max_tones,
                min_tone_spacing=min_tone_spacing,
            )

            selected_tones = (
                tones_ds[selected_indices]
                if selected_indices
                else np.array([])
            )
            G_selected = (
                G_ds[:, selected_indices]
                if selected_indices
                else np.zeros((len(labels), 0))
            )

            K_unit = metrics["K"]

            target_rate_unit = float(metrics["target_rate"])
            if abs(target_rate_unit) > 1e-14 and math.isfinite(target_rate_unit):
                T_unit = 2.0 * abs(target_register_phase) / abs(target_rate_unit)
            else:
                T_unit = float("inf")

            epsilon_spec = metrics["epsilon_spec"]
            score = selectivity_score(epsilon_spec, score_ceiling=score_ceiling)

            # These are dimensionless calibrated-intensity synthesis units.
            # In full dynamics we choose an absolute calibrated-intensity scale
            # and keep the same relative x.
            total_intensity_unit = float(metrics["total_intensity"])

            if total_intensity_unit > 0:
                normalized_x = x_unit / total_intensity_unit
            else:
                normalized_x = np.zeros_like(x_unit)

            case_label = (
                f"{geometry_label.replace(' ', '_')}"
                f"__{target_type(target_label)}__{target_label}"
            )

            case_row = {
                "case_id": int(case_id),
                "case_label": case_label,
                "geometry_label": geometry_label,
                "target_type": target_type(target_label),
                "target_label": target_label,
                "target_order": int(pauli_order(target_label)),
                "rho": float(rho),
                "sigma": float(sigma),
                "A1": float(A[0]),
                "A2": float(A[1]),
                "A3": float(A[2]),
                "target_register_phase": float(target_register_phase),
                "target_phase_parameter": float(2.0 * target_register_phase),
                "target_sign": float(metrics["target_sign"]),
                "num_candidate_tones": int(len(tones_ds)),
                "max_tones": int(max_tones),
                "num_selected_tones": int(len(selected_indices)),
                "active_count": int(metrics["active_count"]),
                "relative_residual": float(metrics["relative_residual"]),
                "epsilon_spec": (
                    float(epsilon_spec)
                    if math.isfinite(epsilon_spec)
                    else "inf"
                ),
                "selectivity_score": float(score),
                "target_rate_unit": float(target_rate_unit),
                "gate_time_unit": (
                    float(T_unit)
                    if math.isfinite(T_unit)
                    else "inf"
                ),
                "total_intensity_unit": float(total_intensity_unit),
                "max_intensity_unit": float(metrics["max_intensity"]),
                "rate_per_total_intensity": float(
                    metrics["rate_per_total_intensity"]
                ),
                "selected_tones": ";".join(f"{w:.10f}" for w in selected_tones),
                "selected_intensities_unit": ";".join(
                    f"{v:.10e}" for v in x_unit
                ),
                "selected_intensities_normalized": ";".join(
                    f"{v:.10e}" for v in normalized_x
                ),
                "pauli_labels": ";".join(labels),
                "span": float(span),
                "min_detuning": float(min_detuning),
                "min_tone_spacing": float(min_tone_spacing),
                "num_grid": int(num_grid),
                "candidate_pool": int(candidate_pool),
            }

            case_rows.append(case_row)

            for local_index, global_index in enumerate(selected_indices):
                intensity_unit = float(x_unit[local_index])
                normalized_intensity = (
                    float(normalized_x[local_index])
                    if local_index < len(normalized_x)
                    else 0.0
                )

                control_rows.append(
                    {
                        "case_id": int(case_id),
                        "case_label": case_label,
                        "geometry_label": geometry_label,
                        "target_type": target_type(target_label),
                        "target_label": target_label,
                        "tone_number": int(local_index + 1),
                        "tone_index": int(global_index),
                        "omega": float(tones_ds[global_index]),
                        "intensity_unit": intensity_unit,
                        "normalized_intensity": normalized_intensity,
                    }
                )

            for i, label in enumerate(labels):
                rate_rows.append(
                    {
                        "case_id": int(case_id),
                        "case_label": case_label,
                        "geometry_label": geometry_label,
                        "target_type": target_type(target_label),
                        "target_label": target_label,
                        "pauli_label": label,
                        "is_target": int(label == target_label),
                        "K_unit": float(K_unit[i]),
                        "phi_at_gate_unit": (
                            float(0.5 * K_unit[i] * T_unit)
                            if math.isfinite(T_unit)
                            else "inf"
                        ),
                    }
                )

            print(
                f"  {target_label:<8} "
                f"epsilon={epsilon_spec:.3e} "
                f"A={score:.2f} "
                f"K_target={target_rate_unit:.3e} "
                f"T_unit={T_unit:.3e} "
                f"tones={len(selected_indices)}",
                flush=True,
            )

    return case_rows, control_rows, rate_rows


def print_case_summary(case_rows: list[dict]) -> None:
    print()
    print("[validation cases]")
    print(
        "case geometry                  target      eps_spec     A_score   "
        "K_unit      T_unit"
    )

    for r in case_rows:
        print(
            f"{int(r['case_id']):>2d}   "
            f"{r['geometry_label']:<25} "
            f"{r['target_label']:<10} "
            f"{float(r['epsilon_spec']):>9.2e} "
            f"{float(r['selectivity_score']):>8.3f} "
            f"{float(r['target_rate_unit']):>10.3e} "
            f"{float(r['gate_time_unit']):>10.3e}"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representatives",
        type=Path,
        default=DATA_DIR / "representative_geometries.csv",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Fixed target Pauli strings used for all representative geometries.",
    )
    parser.add_argument("--num-grid", type=int, default=401)
    parser.add_argument("--candidate-pool", type=int, default=151)
    parser.add_argument("--max-tones", type=int, default=8)
    parser.add_argument("--span", type=float, default=1.35)
    parser.add_argument("--min-detuning", type=float, default=0.04)
    parser.add_argument("--min-tone-spacing", type=float, default=0.04)
    parser.add_argument(
        "--target-register-phase",
        type=float,
        default=TARGET_REGISTER_PHASE,
    )
    parser.add_argument("--score-ceiling", type=float, default=4.0)

    args = parser.parse_args()

    if args.min_tone_spacing <= 0:
        raise ValueError("--min-tone-spacing must be positive.")

    if args.max_tones <= 0:
        raise ValueError("--max-tones must be positive.")

    ensure_dirs()

    if not args.representatives.exists():
        raise FileNotFoundError(
            f"Could not find {args.representatives}. "
            "Run 02_response_geometry_representatives.py first."
        )

    reps = load_csv_rows(args.representatives)

    case_rows, control_rows, rate_rows = design_validation_cases(
        reps,
        targets=args.targets,
        num_grid=args.num_grid,
        candidate_pool=args.candidate_pool,
        max_tones=args.max_tones,
        span=args.span,
        min_detuning=args.min_detuning,
        min_tone_spacing=args.min_tone_spacing,
        target_register_phase=args.target_register_phase,
        score_ceiling=args.score_ceiling,
    )

    cases_csv = DATA_DIR / "validation_cases.csv"
    controls_csv = DATA_DIR / "validation_control_tones.csv"
    rates_csv = DATA_DIR / "validation_predicted_rates.csv"

    write_csv(cases_csv, case_rows)
    print(f"[write] {cases_csv}")

    write_csv(controls_csv, control_rows)
    print(f"[write] {controls_csv}")

    write_csv(rates_csv, rate_rows)
    print(f"[write] {rates_csv}")

    print_case_summary(case_rows)


if __name__ == "__main__":
    main()