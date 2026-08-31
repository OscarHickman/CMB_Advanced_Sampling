# diffcmb

[![Python Tests](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml/badge.svg)](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml)

`diffcmb` is a Python package for differentiable, full-sky, curved-sky (HEALPix) Bayesian Cosmic Microwave Background (CMB) analysis. It performs joint posterior sampling over the unlensed CMB temperature field ($a_{\ell m}$), the CMB angular power spectrum ($C_\ell$), the gravitational lensing potential field ($\phi$), and the lensing potential power spectrum ($C_L^{\phi\phi}$).

## Method Overview

The inference framework uses a four-block Gibbs sampler:

1. **$C_\ell \mid a_{\ell m}$**: Exact inverse-Gamma draw per multipole $\ell$.
2. **$a_{\ell m} \mid C_\ell, \phi$**: Hamiltonian Monte Carlo (HMC) sampling using matrix-free spherical harmonic transforms (`ducc0`).
3. **$\phi \mid a_{\ell m}, C_\ell$**: HMC sampling through a differentiable curved-sky forward lensing operator.
4. **$C_L^{\phi\phi} \mid \phi$**: Exact inverse-Gamma draw for the lensing power spectrum.

Both spectrum blocks draw from $\mathrm{InvGamma}(k_L/2 + a_0,\; b_0 + S_L/2)$, where $k_L$ is the number of *real* degrees of freedom the packed parameter vector carries at multipole $L$ and $a_0, b_0$ are the prior's shape and scale. $k_L$ is derived from the packing rather than assumed — see `CLAUDE.md`.

By default $C_L^{\phi\phi}$ carries a flat prior, which is improper: integrating it out leaves a $\phi$ marginal that is flat in $S_L$ and rising with amplitude, so the $\phi$ amplitude is constrained by the lensing likelihood alone. Pass `cl_phiphi_prior_nu` for a proper conjugate prior instead; this is required for any strict calibration statement about $C_L^{\phi\phi}$.

## Validation status

Simulation-based calibration at $\ell_{\max}=64$, 12 independent chains, with $C_L^{\phi\phi}$ held fixed (so the $\phi$ prior is proper and matches the generative process): pooled mean normalised ranks $\bar u_\phi = 0.453$ ($p=0.17$) and $\bar u_{a_{\ell m}} = 0.537$ ($p=0.33$), both consistent with uniformity.

Note that the rank statistic used for the *spectrum* rows is non-uniform even for an exact sampler — it ranks the truth against its own conditional's mode — so those rows must be read against the null from `scripts/validate_coverage_rank_nulls.py`, not against 0.5.

**Known limitation:** the packed parameterisation forces $\mathrm{Im}(a_{\ell,1}) = 0$, so one real degree of freedom per multipole is missing and the model cannot represent a completely general sky (~20% of the modes at $\ell=2$, falling to 0.8% at $\ell=63$). This is tracked in `ROADMAP.md` as the largest open defect.

## Installation

```bash
# Create virtual environment and install dependencies
make setup

# Optional: Build the Rust spherical-harmonic extension for dense SHT acceleration
make build-rust
```

## Quick Start

```python
from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain

# Initialize model
model = CosmologyAdvancedSampling(_lmax=30, _NSIDE=16, _noisesig=1.0, data_mode='synthetic')

# Run joint Gibbs sampling over (alm, C_l) -- Blocks 1-2 only (no phi block)
samples, logp, accepts, final_step_size = run_gibbs_chain(
    model,
    n_samples=500,
    alm_sampler='hmc',
    hmc_step_size=0.01,
    n_lfs=10,
)
# `samples` is (n_samples, len(model.x0)): each row packs [ln C_l, real alm, imag alm]
# per the parameter vector layout described in the module docstrings / CLAUDE.md.
```

## Running Tests

```bash
make test
# or directly with pytest:
PYTHONPATH=diffcmb pytest
```

