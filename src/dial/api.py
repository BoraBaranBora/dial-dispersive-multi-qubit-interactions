"""Public user-facing interface for Target-Aware DIAL."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re

import numpy as np

from .spectrum import (
    candidate_tones,
    mean_transition_spacing,
    minimum_transition_spacing,
)
from .synthesis import (
    dispersive_scale_from_unit_control,
    downsample_candidates,
    filter_nonzero_transfer_columns,
    greedy_sparse_target_synthesis,
)
from .transfer import transfer_matrix


@dataclass(frozen=True)
class DIALValidation:
    """
    Numerical aligned driven-RWA validation of a DIAL control.

    This validation includes the complete multitone mediator dynamics
    within the configuration-preserving RWA model. It does not include
    basis misalignment, decoherence, relaxation, or counter-rotating
    laboratory-frame terms.
    """

    phase: float
    signed_target_phase: float

    gate_time: float
    max_rabi_frequency: float
    max_rabi_period: float
    gate_time_over_max_rabi_period: float

    ground_manifold_fidelity: float

    max_transient_mediator_excitation: float
    terminal_mediator_excitation: float

    n_steps: int


@dataclass(frozen=True)
class DIALControl:
    """
    Dispersive multi-tone control returned by Target-Aware DIAL.

    Frequencies, amplitudes, and interaction rates use the same
    frequency convention and units supplied by the user.
    """

    n: int
    configurations: tuple[str, ...]
    transition_frequencies: np.ndarray
    reference_frequency: float

    target: str
    labels: tuple[str, ...]
    target_index: int

    tones: np.ndarray
    amplitudes: np.ndarray
    intensities: np.ndarray
    rates: np.ndarray

    spectral_error: float
    relative_residual: float
    max_dispersive_ratio: float

    min_detuning: float
    min_tone_spacing: float
    r_disp: float
    approx_mediator_excitation: float

    @property
    def target_rate(self) -> float:
        """Synthesized rate of the requested Pauli-Z interaction."""
        return float(
            self.rates[self.target_index]
        )

    @property
    def rates_by_label(self) -> dict[str, float]:
        """All synthesized nonidentity Pauli-Z interaction rates."""
        return {
            label: float(rate)
            for label, rate in zip(
                self.labels,
                self.rates,
            )
        }

    @property
    def spectator_rates(self) -> dict[str, float]:
        """Synthesized rates excluding the requested target."""
        return {
            label: float(rate)
            for i, (label, rate) in enumerate(
                zip(
                    self.labels,
                    self.rates,
                )
            )
            if i != self.target_index
        }

    def gate_time(
        self,
        phase: float = math.pi / 4.0,
    ) -> float:
        """
        Evolution time required to accumulate |phase| on the target.

        Uses the convention

            theta(T) = K_target T / 2.
        """
        rate = abs(self.target_rate)

        if rate <= 0.0:
            raise ZeroDivisionError(
                "The synthesized target rate is zero."
            )

        return float(
            2.0 * abs(phase) / rate
        )

    def phase_at_time(
        self,
        time: float,
    ) -> float:
        """Signed target phase accumulated after the supplied time."""
        return float(
            0.5
            * self.target_rate
            * float(time)
        )

    @property
    def max_rabi_frequency(self) -> float:
        """
        Largest applied tone amplitude.

        With the package convention, this is
        ``Omega_max = max_k |Omega_k|``.
        """
        if len(self.amplitudes) == 0:
            return 0.0

        return float(
            np.max(
                np.abs(
                    self.amplitudes
                )
            )
        )

    @property
    def max_rabi_period(self) -> float:
        """
        Rabi period associated with the largest applied amplitude,

            T_Rabi^max = 2 pi / Omega_max.
        """
        omega_max = (
            self.max_rabi_frequency
        )

        if omega_max <= 0.0:
            return float("inf")

        return float(
            2.0
            * math.pi
            / omega_max
        )

    def gate_time_over_max_rabi_period(
        self,
        phase: float = math.pi / 4.0,
    ) -> float:
        """
        Gate time normalized by the fastest applied Rabi period,

            T_gate / T_Rabi^max
            = T_gate Omega_max / (2 pi).
        """
        return float(
            self.gate_time(phase)
            * self.max_rabi_frequency
            / (
                2.0
                * math.pi
            )
        )

    def validate_full_dynamics(
        self,
        phase: float = math.pi / 4.0,
        *,
        steps_per_period: int = 12,
        max_steps: int = 20_000_000,
    ) -> DIALValidation:
        """
        Numerically validate the control with aligned driven-RWA dynamics.

        This propagates the complete multitone mediator dynamics for every
        register configuration. It is the same physical validation model
        used for the aligned benchmark in the accompanying paper.

        The returned fidelity is the ground-manifold register process
        fidelity. The transient mediator excitation is obtained directly
        from the numerical propagation rather than from the dispersive
        estimate.

        This remains a model-based validation: basis mismatch, decoherence,
        relaxation, and counter-rotating laboratory-frame terms are not
        included.
        """
        if steps_per_period <= 0:
            raise ValueError(
                "steps_per_period must be positive."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be positive."
            )

        from .dynamics import (
            propagate_final_blocks_with_transient_flip_jit,
        )
        from .metrics import (
            final_max_ground_to_excited_flip,
            ground_manifold_register_fidelity,
        )

        T_gate = self.gate_time(
            phase
        )

        offsets = (
            np.asarray(
                self.transition_frequencies,
                dtype=float,
            )
            - self.reference_frequency
        )

        tones = (
            np.asarray(
                self.tones,
                dtype=float,
            )
            - self.reference_frequency
        )

        intensities = np.asarray(
            self.intensities,
            dtype=float,
        )

        amplitudes = np.asarray(
            self.amplitudes,
            dtype=float,
        )

        max_detuning = float(
            np.max(
                np.abs(
                    tones[:, None]
                    - offsets[None, :]
                )
            )
        )

        max_rate = max(
            max_detuning,
            self.max_rabi_frequency,
            1e-12,
        )

        n_steps = max(
            1,
            int(
                math.ceil(
                    T_gate
                    * steps_per_period
                    * max_rate
                    / (
                        2.0
                        * math.pi
                    )
                )
            ),
        )

        if n_steps > max_steps:
            raise ValueError(
                f"Full-dynamics validation requires "
                f"n_steps={n_steps}, exceeding "
                f"max_steps={max_steps}. Increase max_steps "
                "explicitly rather than silently reducing "
                "the integration resolution."
            )

        (
            blocks,
            max_transient_flip,
        ) = (
            propagate_final_blocks_with_transient_flip_jit(
                offsets,
                tones,
                intensities,
                T_gate,
                n_steps,
            )
        )

        signed_target_phase = float(
            math.copysign(
                abs(float(phase)),
                self.target_rate,
            )
        )

        fidelity = (
            ground_manifold_register_fidelity(
                blocks,
                target_label=self.target,
                signed_target_phase=(
                    signed_target_phase
                ),
            )
        )

        terminal_excitation = (
            final_max_ground_to_excited_flip(
                blocks
            )
        )

        return DIALValidation(
            phase=float(
                abs(phase)
            ),
            signed_target_phase=(
                signed_target_phase
            ),
            gate_time=float(
                T_gate
            ),
            max_rabi_frequency=float(
                self.max_rabi_frequency
            ),
            max_rabi_period=float(
                self.max_rabi_period
            ),
            gate_time_over_max_rabi_period=float(
                self.gate_time_over_max_rabi_period(
                    phase
                )
            ),
            ground_manifold_fidelity=float(
                fidelity
            ),
            max_transient_mediator_excitation=float(
                max_transient_flip
            ),
            terminal_mediator_excitation=float(
                terminal_excitation
            ),
            n_steps=int(
                n_steps
            ),
        )


def _infer_n(
    count: int,
) -> int:
    if count < 2:
        raise ValueError(
            "A register spectrum must contain at least two transitions."
        )

    n = int(round(math.log2(count)))

    if 2**n != count:
        raise ValueError(
            "A register spectrum must contain exactly 2**n "
            "configuration-resolved transition frequencies."
        )

    return n


def _configuration_key(
    key,
    *,
    n: int,
) -> str:
    if isinstance(key, str):
        value = key.strip()

        if (
            value.startswith("|")
            and value.endswith(">")
        ):
            value = value[1:-1]

        value = value.replace(" ", "")

    elif isinstance(key, tuple):
        value = "".join(
            str(int(bit))
            for bit in key
        )

    else:
        raise TypeError(
            "Spectrum mapping keys must be bit strings such as "
            "'010', kets such as '|010>', or tuples such as "
            "(0, 1, 0)."
        )

    if (
        len(value) != n
        or any(bit not in "01" for bit in value)
    ):
        raise ValueError(
            f"Invalid {n}-spin register configuration: {key!r}"
        )

    return value


def _coerce_spectrum(
    spectrum,
) -> tuple[
    int,
    tuple[str, ...],
    np.ndarray,
]:
    count = len(spectrum)
    n = _infer_n(count)

    configurations = tuple(
        format(i, f"0{n}b")
        for i in range(2**n)
    )

    if isinstance(spectrum, Mapping):
        normalized = {}

        for key, value in spectrum.items():
            config = _configuration_key(
                key,
                n=n,
            )

            if config in normalized:
                raise ValueError(
                    f"Duplicate configuration {config!r}."
                )

            normalized[config] = float(value)

        missing = [
            config
            for config in configurations
            if config not in normalized
        ]

        if missing:
            raise ValueError(
                "Spectrum is missing register configurations: "
                + ", ".join(missing)
            )

        frequencies = np.asarray(
            [
                normalized[config]
                for config in configurations
            ],
            dtype=float,
        )

    else:
        frequencies = np.asarray(
            spectrum,
            dtype=float,
        )

        if frequencies.ndim != 1:
            raise ValueError(
                "transition frequencies must be one-dimensional."
            )

        if len(frequencies) != 2**n:
            raise ValueError(
                "Unexpected spectrum length."
            )

    if not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "All transition frequencies must be finite."
        )

    ordered = np.sort(frequencies)

    if np.any(
        np.diff(ordered) <= 0.0
    ):
        raise ValueError(
            "The simple public DIAL interface requires distinct "
            "configuration-resolved transition frequencies. "
            "For exactly degenerate spectra, use the lower-level "
            "dial modules directly."
        )

    return (
        n,
        configurations,
        frequencies,
    )


def _canonical_target(
    target,
    *,
    n: int,
    labels: Sequence[str],
) -> str:
    labels = tuple(labels)

    if isinstance(target, str):
        compact = (
            target
            .replace("_", "")
            .replace(" ", "")
        )

        if compact in labels:
            return compact

        if compact == "Z" * n:
            full = "".join(
                f"Z{i}"
                for i in range(1, n + 1)
            )

            if full in labels:
                return full

        indices = [
            int(value)
            for value in re.findall(
                r"Z(\d+)",
                compact,
            )
        ]

        reconstructed = "".join(
            f"Z{i}"
            for i in indices
        )

        if (
            indices
            and reconstructed == compact
        ):
            pass
        else:
            raise ValueError(
                f"Could not interpret target {target!r}. "
                "Use labels such as 'Z1', 'Z1Z3', or 'Z1Z2Z3'."
            )

    else:
        try:
            indices = [
                int(i)
                for i in target
            ]
        except TypeError as exc:
            raise TypeError(
                "target must be a Pauli-Z label or a sequence "
                "of register-spin indices."
            ) from exc

    if not indices:
        raise ValueError(
            "At least one target register spin is required."
        )

    if len(set(indices)) != len(indices):
        raise ValueError(
            "Target register-spin indices must be unique."
        )

    if any(
        i < 1 or i > n
        for i in indices
    ):
        raise ValueError(
            f"Target indices must lie between 1 and {n}."
        )

    canonical = "".join(
        f"Z{i}"
        for i in sorted(indices)
    )

    if canonical not in labels:
        raise ValueError(
            f"Target {canonical!r} is unavailable. "
            f"Available targets are: {', '.join(labels)}"
        )

    return canonical


def design_control(
    spectrum,
    *,
    target,
    r_disp: float | None = None,
    approx_mediator_excitation: float | None = None,
    max_tones: int | None = None,
    min_detuning_fraction: float = 0.45,
    min_tone_spacing_fraction: float = 0.075,
    span: float = 1.35,
    grid_points: int = 6401,
    candidate_pool: int | None = None,
    reference_frequency: float | None = None,
    safety: float = 1.0 - 1e-10,
) -> DIALControl:
    """
    Design a Target-Aware DIAL control from a register-resolved spectrum.

    Parameters
    ----------
    spectrum
        Either a mapping from computational-basis register configurations
        to mediator transition frequencies, for example

            {"000": ..., "001": ..., ..., "111": ...}

        or a one-dimensional array in binary integer order

            000, 001, 010, ..., 111.

        Absolute and relative transition frequencies are both accepted.

    target
        Desired nonidentity Pauli-Z string, for example ``"Z1Z2Z3"``
        or ``"Z1Z3"``. A sequence such as ``(1, 3)`` is also accepted.

    r_disp
        Maximum allowed dispersive ratio
        ``max_k |Omega_k| / d_k``. If neither ``r_disp`` nor
        ``approx_mediator_excitation`` is supplied, the default is
        ``r_disp=0.10``.

    approx_mediator_excitation
        Optional approximate mediator-excitation scale, supplied as a
        probability between 0 and 1. For example, ``0.01`` means an
        approximate 1% excitation scale. It is converted to a dispersive
        ratio using the isolated detuned two-level estimate

            P_max ~= r**2 / (1 + r**2),

        where ``r = |Omega| / |Delta|``.

        This is a convenient dispersive estimate, not a guarantee on the
        maximum mediator population under a multitone control. Do not
        supply this together with ``r_disp``.

    min_detuning_fraction
        Candidate tones are kept at least this fraction of the minimum
        transition spacing away from every register-resolved transition.

    min_tone_spacing_fraction
        Minimum separation between selected tones, expressed as a fraction
        of the mean transition spacing.

    reference_frequency
        Frequency reference subtracted before the DIAL calculation.
        By default the midpoint of the supplied spectrum is used.
        The reference is added back to the returned tone frequencies.

    Notes
    -----
    The present Target-Aware DIAL solver evaluates both signs of the
    requested target rate under nonnegative tone intensities and retains
    the better-conditioned solution, matching the validated paper
    implementation. Inspect ``control.target_rate`` for the resulting sign.
    """

    (
        n,
        configurations,
        frequencies,
    ) = _coerce_spectrum(
        spectrum
    )

    if (
        r_disp is not None
        and approx_mediator_excitation is not None
    ):
        raise ValueError(
            "Specify either r_disp or approx_mediator_excitation, "
            "not both."
        )

    if approx_mediator_excitation is not None:
        p_approx = float(
            approx_mediator_excitation
        )

        if not (
            0.0 < p_approx < 1.0
        ):
            raise ValueError(
                "approx_mediator_excitation must satisfy "
                "0 < p < 1."
            )

        r_disp = math.sqrt(
            p_approx
            / (1.0 - p_approx)
        )

    elif r_disp is None:
        # Preserve the original public-package default.
        r_disp = 0.10

    else:
        r_disp = float(r_disp)

    if r_disp <= 0.0:
        raise ValueError(
            "r_disp must be positive."
        )

    if not (
        0.0 < safety <= 1.0
    ):
        raise ValueError(
            "safety must satisfy 0 < safety <= 1."
        )

    if min_detuning_fraction <= 0.0:
        raise ValueError(
            "min_detuning_fraction must be positive."
        )

    if min_tone_spacing_fraction < 0.0:
        raise ValueError(
            "min_tone_spacing_fraction cannot be negative."
        )

    if grid_points < 3:
        raise ValueError(
            "grid_points must be at least 3."
        )

    if reference_frequency is None:
        reference_frequency = float(
            0.5
            * (
                np.min(frequencies)
                + np.max(frequencies)
            )
        )
    else:
        reference_frequency = float(
            reference_frequency
        )

    offsets = (
        frequencies
        - reference_frequency
    )

    spectral_radius = float(
        np.max(np.abs(offsets))
    )

    if spectral_radius <= 0.0:
        raise ValueError(
            "The supplied spectrum has zero bandwidth."
        )

    minimum_spacing = (
        minimum_transition_spacing(
            offsets
        )
    )

    mean_spacing = (
        mean_transition_spacing(
            offsets
        )
    )

    physical_min_detuning = float(
        min_detuning_fraction
        * minimum_spacing
    )

    candidate_min_detuning = float(
        physical_min_detuning
        / spectral_radius
    )

    min_tone_spacing = float(
        min_tone_spacing_fraction
        * mean_spacing
    )

    relative_tones = candidate_tones(
        offsets,
        num_grid=grid_points,
        span=span,
        min_detuning=(
            candidate_min_detuning
        ),
    )

    labels, G_pool = transfer_matrix(
        offsets,
        relative_tones,
        n=n,
        include_identity=False,
    )

    canonical_target = (
        _canonical_target(
            target,
            n=n,
            labels=labels,
        )
    )

    target_index = labels.index(
        canonical_target
    )

    relative_tones, G_pool = (
        filter_nonzero_transfer_columns(
            relative_tones,
            G_pool,
        )
    )

    if candidate_pool is None:
        candidate_pool = grid_points

    if candidate_pool <= 0:
        raise ValueError(
            "candidate_pool must be positive."
        )

    relative_tones, G_pool = (
        downsample_candidates(
            relative_tones,
            G_pool,
            candidate_pool=(
                candidate_pool
            ),
        )
    )

    if max_tones is None:
        max_tones = len(labels)

    if max_tones <= 0:
        raise ValueError(
            "max_tones must be positive."
        )

    (
        selected,
        intensities_unit,
        metrics,
    ) = greedy_sparse_target_synthesis(
        relative_tones,
        G_pool,
        target_index=target_index,
        max_tones=max_tones,
        min_tone_spacing=(
            min_tone_spacing
        ),
    )

    if not selected:
        raise RuntimeError(
            "Target-Aware DIAL selected no control tones."
        )

    selected = np.asarray(
        selected,
        dtype=int,
    )

    selected_tones = np.asarray(
        relative_tones[selected],
        dtype=float,
    )

    intensities_unit = np.asarray(
        intensities_unit,
        dtype=float,
    )

    active = (
        intensities_unit > 1e-10
    )

    selected = selected[active]
    selected_tones = (
        selected_tones[active]
    )
    intensities_unit = (
        intensities_unit[active]
    )

    if len(selected_tones) == 0:
        raise RuntimeError(
            "All selected DIAL tones received zero intensity."
        )

    G_selected = G_pool[
        :,
        selected,
    ]

    rates_unit = (
        G_selected
        @ intensities_unit
    )

    target_rate_unit = float(
        rates_unit[target_index]
    )

    if abs(target_rate_unit) <= 1e-15:
        raise RuntimeError(
            "DIAL produced zero target interaction rate."
        )

    scaled = (
        dispersive_scale_from_unit_control(
            tones=selected_tones,
            intensities_unit=(
                intensities_unit
            ),
            offsets=offsets,
            r_disp=r_disp,
        )
    )

    drive_scale = float(
        scaled["drive_scale"]
        * safety
    )

    intensities = (
        intensities_unit
        * drive_scale
    )

    amplitudes = np.sqrt(
        np.maximum(
            2.0 * intensities,
            0.0,
        )
    )

    rates = (
        rates_unit
        * drive_scale
    )

    detunings = np.min(
        np.abs(
            selected_tones[:, None]
            - offsets[None, :]
        ),
        axis=1,
    )

    dispersive_ratios = (
        amplitudes
        / detunings
    )

    max_dispersive_ratio = float(
        np.max(
            dispersive_ratios
        )
    )

    # Isolated detuned two-level estimate.
    #
    # This is deliberately reported as an approximation:
    # multitone interference can change the actual mediator
    # population in the complete driven dynamics.
    approx_excitation = float(
        max_dispersive_ratio**2
        / (
            1.0
            + max_dispersive_ratio**2
        )
    )

    absolute_tones = (
        selected_tones
        + reference_frequency
    )

    return DIALControl(
        n=n,
        configurations=(
            configurations
        ),
        transition_frequencies=(
            frequencies.copy()
        ),
        reference_frequency=(
            reference_frequency
        ),
        target=canonical_target,
        labels=tuple(labels),
        target_index=target_index,
        tones=np.asarray(
            absolute_tones,
            dtype=float,
        ),
        amplitudes=np.asarray(
            amplitudes,
            dtype=float,
        ),
        intensities=np.asarray(
            intensities,
            dtype=float,
        ),
        rates=np.asarray(
            rates,
            dtype=float,
        ),
        spectral_error=float(
            metrics["epsilon_spec"]
        ),
        relative_residual=float(
            metrics["relative_residual"]
        ),
        max_dispersive_ratio=(
            max_dispersive_ratio
        ),
        min_detuning=(
            physical_min_detuning
        ),
        min_tone_spacing=(
            min_tone_spacing
        ),
        r_disp=float(r_disp),
        approx_mediator_excitation=(
            approx_excitation
        ),
    )
