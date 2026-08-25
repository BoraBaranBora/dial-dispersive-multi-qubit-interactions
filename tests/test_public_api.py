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


if __name__ == "__main__":
    unittest.main()
