"""Apply Target-Aware DIAL to a user-supplied register spectrum."""

import numpy as np

from dialcontrol import design_control


# Example three-spin register-resolved mediator transition spectrum.
#
# Absolute or relative frequencies are accepted. The dictionary keys
# identify the computational-basis register configuration.
spectrum = {
    "000": 2876.9,
    "001": 2868.5,
    "010": 2873.1,
    "011": 2864.7,
    "100": 2875.3,
    "101": 2866.9,
    "110": 2871.5,
    "111": 2863.1,
}

control = design_control(
    spectrum,
    target="Z1Z2Z3",
    r_disp=0.10,
)

# Alternatively, specify an approximate tolerated mediator
# excitation instead of r_disp:
#
# control = design_control(
#     spectrum,
#     target="Z1Z2Z3",
#     approx_mediator_excitation=0.01,
# )

print("Target:", control.target)
print("Tone frequencies:", control.tones)
print("Tone amplitudes:", control.amplitudes)
print("Tone intensities:", control.intensities)
print("Target rate:", control.target_rate)
print("Spectral error:", control.spectral_error)
print(
    "Approx. mediator excitation:",
    control.approx_mediator_excitation,
)
phase = np.pi / 4.0

print(
    "Gate time for pi/4:",
    control.gate_time(phase),
)

print(
    "Gate time / max Rabi period:",
    control.gate_time_over_max_rabi_period(
        phase
    ),
)

# Optional numerical validation using the complete aligned
# driven-RWA model. This is more expensive than analytical
# DIAL synthesis and therefore runs only when explicitly requested.
validation = (
    control.validate_full_dynamics(
        phase
    )
)

print(
    "Full-dynamics fidelity:",
    validation.ground_manifold_fidelity,
)

print(
    "Simulated max mediator excitation:",
    validation.max_transient_mediator_excitation,
)

print(
    "Terminal mediator excitation:",
    validation.terminal_mediator_excitation,
)
