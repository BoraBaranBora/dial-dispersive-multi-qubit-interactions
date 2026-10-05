# DIAL: Dispersive Multi-Qubit Interaction Synthesis

This repository contains the numerical implementation and publication
reproduction workflow for **dispersive interaction via analytical linear
inversion (DIAL)**.

DIAL constructs off-resonant multi-tone controls for engineering diagonal
multi-qubit interactions in mediator-coupled quantum registers. In the
configuration-preserving dispersive regime, the induced Pauli-Z interaction
rates depend linearly on the tone intensities, allowing the control problem
to be formulated as a constrained linear inversion.

The accompanying paper is:

> **Analytical controls for dispersive multi-qubit interactions in
> mediator-coupled quantum registers**

[arXiv:2610.03064](https://arxiv.org/abs/2610.03064)

## Repository entry points

This repository has two complementary uses:

- **Use Target-Aware DIAL on your own register:** provide a
  configuration-resolved mediator transition spectrum and the desired
  Pauli-Z interaction, and obtain a dispersively admissible
  multi-tone control.
- **Reproduce the paper:** use the frozen publication datasets,
  benchmark scripts, and plotting workflow provided in this repository.

## Installation

Python 3.10 or later is required.

```bash
pip install -e .
```

The main numerical dependencies are NumPy, SciPy, Matplotlib, and Numba.

For reproduction of the publication numerics using the package versions
used in this work, install the pinned environment first:

```bash
pip install -r requirements.txt
pip install -e .
```

The pinned file records the Python-package versions used for the
publication calculations, while `pyproject.toml` keeps the general DIAL
package dependencies flexible.

## Use DIAL on your own register

The public interface accepts either absolute or relative
configuration-resolved mediator transition frequencies. For a
three-spin register, for example:

```python
import numpy as np
from dialcontrol import design_control

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

print(control.tones)
print(control.amplitudes)
print(control.target_rate)
print(control.spectator_rates)
phase = np.pi / 4

print(control.gate_time(phase))
print(
    control.gate_time_over_max_rabi_period(
        phase
    )
)

validation = control.validate_full_dynamics(
    phase
)

print(validation.ground_manifold_fidelity)
print(
    validation.max_transient_mediator_excitation
)
```

The dictionary keys specify which mediator transition belongs to each
computational-basis register configuration. An array can also be supplied,
in binary integer order $00\ldots0, 00\ldots1,\ldots,11\ldots1$.

The control strength can be specified in two equivalent ways. Users who
work directly with the dispersive approximation can set the maximum
allowed ratio

```python
control = design_control(
    spectrum,
    target="Z1Z2Z3",
    r_disp=0.10,
)
```

Alternatively, a tolerated approximate mediator-excitation scale can be
used:

```python
control = design_control(
    spectrum,
    target="Z1Z2Z3",
    approx_mediator_excitation=0.01,
)
```

Here `0.01` means an approximate 1% excitation scale. DIAL converts this
using the isolated detuned two-level estimate

```math
P_{\mathrm{med}}^{\mathrm{approx}}
=
\frac{r_{\mathrm{disp}}^2}{1+r_{\mathrm{disp}}^2}.
```

This is a convenient estimate for choosing the dispersive drive scale;
it is **not** a guarantee on the maximum mediator population under the
complete multitone dynamics. The returned control reports both
`control.max_dispersive_ratio` and
`control.approx_mediator_excitation`. Obtaining the actual transient
mediator population requires propagation of the complete driven
mediator-register model.

The returned tone frequencies use the same frequency coordinate as the
supplied spectrum. Internally, DIAL removes an arbitrary common frequency
reference before constructing the transfer matrix and restores it in the
returned tones, so absolute and zero-centered spectra are equivalent.

All frequencies and amplitudes must use one consistent convention and
unit; the package does not insert or remove factors of $2\pi$.

The public solver accepts any nonidentity Pauli-Z string such as
`Z1`, `Z1Z3`, or `Z1Z2Z3`. The present implementation evaluates both
signs of the target rate under nonnegative tone intensities, as in the
paper implementation. The realized sign is reported by
`control.target_rate`.

The gate time is accompanied by the dimensionless quantity

```math
\frac{T_{\mathrm{gate}}}{T_{\mathrm{Rabi}}^{\max}}
=
\frac{T_{\mathrm{gate}}\Omega_{\max}}{2\pi}.
```

which measures the gate duration in units of the fastest Rabi period
available in the designed control.

The analytical design itself does not require numerical propagation.
For users who want a model-based validation, the returned control provides
`validate_full_dynamics()`. This propagates the complete multitone dynamics
within the aligned, configuration-preserving rotating-wave model used for
the paper benchmark and reports the ground-manifold register fidelity,
maximum transient mediator excitation, terminal mediator excitation, and
integration resolution.

This numerical fidelity should not be interpreted as a complete device
fidelity: decoherence, relaxation, basis mismatch, calibration errors, and
counter-rotating laboratory-frame effects are not included.

A complete runnable example is provided in:

```text
examples/design_from_spectrum.py
```

## Reproducing the publication figures

The publication datasets are checked into:

```text
results/publication/
```

The numerical publication figures can be regenerated without rerunning
the full propagation calculations:

```bash
python scripts/reproduce_figures.py
```

This regenerates:

```text
figures/main_direct_inversion_scaling.pdf
figures/benchmark_ensemble_n3_normalized_spectral_space.pdf
figures/scaling_relative_minimum_spacing.pdf
figures/dial_misalignment_audit.pdf
```

PNG versions are generated alongside the PDF files.

## Publication datasets

`results/publication/main_benchmark.json`

: Target-Aware DIAL synthesis and full driven-dynamics validation for
  100 sampled register realizations at each
  n=2,3,4,5, targeting the highest-order interaction
  $Z_1\cdots Z_n$.

`results/publication/transient_excitation.json`

: Maximum transient mediator excitation for the controls in the main
  benchmark.

`results/publication/misalignment_n3.json`

: Fixed-control n=3 audit under inter-manifold register-basis mismatch.

The numerical protocol stored in these datasets is summarized in:

```text
configs/publication.json
```

## Regenerating the numerical data

The checked-in JSON files are the numerical data used for the manuscript.
The simulations can also be recomputed from the clean implementation.

Main DIAL benchmark:

```bash
python scripts/run_main_benchmark.py
```

Transient mediator-excitation audit:

```bash
python scripts/run_transient_excitation_audit.py
```

Basis-mismatch audit:

```bash
python scripts/run_misalignment_audit.py
```

These calculations are substantially more expensive than figure generation.
The default regenerated outputs use separate filenames so that the frozen
publication data are not overwritten.

## Scientific core

The reusable implementation is under `src/dialcontrol/`:

```text
basis.py       computational-basis and Pauli-Z bookkeeping
spectrum.py    register-resolved transition spectra and candidate tones
transfer.py    dispersive transfer matrix
sampling.py    benchmark ensemble generation
synthesis.py   Target-Aware DIAL inversion and dispersive scaling
dynamics.py    driven mediator-register propagation
metrics.py     fidelity and mediator-excitation metrics
io.py          numerical serialization utilities
```

The finite-basis-mismatch full-dynamics implementation in `dynamics.py`
is specialized to the three-register-spin audit used in the manuscript;
the aligned block propagators are generic in register size.

## Reproducibility

The clean scientific core and publication entry points were checked
numerically against the frozen publication implementation. The clean
workflows reproduce the main benchmark, transient mediator-excitation
audit, and complete n=3 basis-mismatch protocol to floating-point
precision.

## Paper

The manuscript is available as [arXiv:2610.03064](https://arxiv.org/abs/2610.03064).

## License

The software in this repository, including the source code, scripts,
examples, and tests, is licensed under the BSD 3-Clause License. See
`LICENSE`.


