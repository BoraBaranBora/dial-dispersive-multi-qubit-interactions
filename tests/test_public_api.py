from pathlib import Path
import json
import unittest

import numpy as np

from dial import design_control


ROOT = Path(__file__).resolve().parents[1]


class PublicAPITest(unittest.TestCase):

    def test_matches_frozen_n2_publication_case(self):
        data = json.loads(
            (
                ROOT
                / "results"
                / "publication"
                / "main_benchmark.json"
            ).read_text(
                encoding="utf-8-sig"
            )
        )

        row = data["cases"][
            "n2_realization_001"
        ]
        config = data["config"]

        control = design_control(
            row["offsets"],
            target=row["target"],
            r_disp=float(
                config["r_disp"]
            ),
            max_tones=int(
                row["max_tones"]
            ),
            min_detuning_fraction=(
                float(
                    row["min_detuning"]
                )
                / float(
                    row[
                        "minimum_transition_spacing"
                    ]
                )
            ),
            min_tone_spacing_fraction=(
                float(
                    row[
                        "min_tone_spacing_required"
                    ]
                )
                / float(
                    row[
                        "mean_transition_spacing"
                    ]
                )
            ),
            span=float(
                config["span"]
            ),
            grid_points=int(
                config["grid_points"]
            ),
            candidate_pool=int(
                row["candidate_pool"]
            ),
            reference_frequency=0.0,
        )

        np.testing.assert_allclose(
            control.tones,
            np.asarray(
                row["tones"],
                dtype=float,
            ),
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            control.intensities,
            np.asarray(
                row["intensities"],
                dtype=float,
            ),
            rtol=1e-11,
            atol=1e-14,
        )

        self.assertAlmostEqual(
            control.target_rate,
            float(
                row["target_rate"]
            ),
            places=12,
        )

        self.assertAlmostEqual(
            control.spectral_error,
            float(
                row["epsilon_spec"]
            ),
            places=12,
        )

        self.assertAlmostEqual(
            control.max_dispersive_ratio,
            float(
                row[
                    "max_dispersive_ratio"
                ]
            ),
            places=10,
        )


    def test_excitation_tolerance_matches_explicit_r_disp(self):
        spectrum = np.array([
            6.9,
            -1.5,
            3.1,
            -5.3,
            5.3,
            -3.1,
            1.5,
            -6.9,
        ])

        p = 0.01

        equivalent_r = np.sqrt(
            p / (1.0 - p)
        )

        by_population = design_control(
            spectrum,
            target="Z1Z2Z3",
            approx_mediator_excitation=p,
            reference_frequency=0.0,
        )

        by_ratio = design_control(
            spectrum,
            target="Z1Z2Z3",
            r_disp=equivalent_r,
            reference_frequency=0.0,
        )

        np.testing.assert_allclose(
            by_population.tones,
            by_ratio.tones,
            rtol=0.0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            by_population.intensities,
            by_ratio.intensities,
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            by_population.rates,
            by_ratio.rates,
            rtol=1e-12,
            atol=1e-14,
        )

        self.assertAlmostEqual(
            by_population.r_disp,
            equivalent_r,
            places=12,
        )

        self.assertAlmostEqual(
            by_population.approx_mediator_excitation,
            p,
            places=9,
        )

    def test_default_r_disp_is_preserved(self):
        spectrum = np.array([
            3.0,
            1.0,
            -1.0,
            -3.0,
        ])

        control = design_control(
            spectrum,
            target="Z1Z2",
            reference_frequency=0.0,
        )

        self.assertAlmostEqual(
            control.r_disp,
            0.10,
            places=12,
        )

    def test_rejects_both_scale_inputs(self):
        spectrum = np.array([
            3.0,
            1.0,
            -1.0,
            -3.0,
        ])

        with self.assertRaises(ValueError):
            design_control(
                spectrum,
                target="Z1Z2",
                r_disp=0.10,
                approx_mediator_excitation=0.01,
                reference_frequency=0.0,
            )

    def test_rejects_invalid_excitation_probability(self):
        spectrum = np.array([
            3.0,
            1.0,
            -1.0,
            -3.0,
        ])

        for p in (
            -0.01,
            0.0,
            1.0,
            1.1,
        ):
            with self.assertRaises(ValueError):
                design_control(
                    spectrum,
                    target="Z1Z2",
                    approx_mediator_excitation=p,
                    reference_frequency=0.0,
                )


    def test_gate_time_rabi_normalization(self):
        spectrum = np.array([
            6.9,
            -1.5,
            3.1,
            -5.3,
            5.3,
            -3.1,
            1.5,
            -6.9,
        ])

        control = design_control(
            spectrum,
            target="Z1Z2Z3",
            r_disp=0.10,
            reference_frequency=0.0,
        )

        T = control.gate_time(
            np.pi / 4.0
        )

        expected = (
            T
            * np.max(
                control.amplitudes
            )
            / (
                2.0
                * np.pi
            )
        )

        self.assertAlmostEqual(
            control.gate_time_over_max_rabi_period(
                np.pi / 4.0
            ),
            expected,
            places=12,
        )

    def test_optional_full_dynamics_validation(self):
        spectrum = np.array([
            3.0,
            1.0,
            -1.0,
            -3.0,
        ])

        control = design_control(
            spectrum,
            target="Z1Z2",
            r_disp=0.05,
            reference_frequency=0.0,
            grid_points=1001,
        )

        validation = (
            control.validate_full_dynamics(
                np.pi / 4.0,
                steps_per_period=12,
            )
        )

        self.assertTrue(
            0.0
            <= validation.ground_manifold_fidelity
            <= 1.000001
        )

        self.assertTrue(
            0.0
            <= validation.max_transient_mediator_excitation
            <= 1.000001
        )

        self.assertTrue(
            0.0
            <= validation.terminal_mediator_excitation
            <= 1.000001
        )

        self.assertAlmostEqual(
            validation.gate_time,
            control.gate_time(
                np.pi / 4.0
            ),
            places=12,
        )

        self.assertAlmostEqual(
            validation.gate_time_over_max_rabi_period,
            control.gate_time_over_max_rabi_period(
                np.pi / 4.0
            ),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
