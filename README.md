# diffcmb

[![Python Tests](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml/badge.svg)](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml)

`diffcmb` is a Python package for differentiable, full-sky, curved-sky (HEALPix) Bayesian Cosmic Microwave Background (CMB) analysis. It performs joint posterior sampling over the unlensed CMB temperature field ($a_{\ell m}$), the CMB angular power spectrum ($C_\ell$), the gravitational lensing potential field ($\phi$), and the lensing potential power spectrum ($C_L^{\phi\phi}$).

## Method Overview

The inference framework uses a four-block Gibbs sampler:

1. **$C_\ell \mid a_{\ell m}$**: Exact inverse-Gamma draw per multipole $\ell$.
2. **$a_{\ell m} \mid C_\ell, \phi$**: Hamiltonian Monte Carlo (HMC) sampling using matrix-free spherical harmonic transforms (`ducc0`).
3. **$\phi \mid a_{\ell m}, C_\ell$**: HMC sampling through a differentiable curved-sky forward lensing operator.
4. **$C_L^{\phi\phi} \mid \phi$**: Exact inverse-Gamma draw for the lensing power spectrum.

## Installation

```bash
# Create virtual environment and install dependencies
make setup

# Optional: Build the Rust spherical-harmonic extension for dense SHT acceleration
make build-rust
```

## Quick Start

```python
import numpy as np
import tensorflow as tf
from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain

# Initialize model
model = CosmologyAdvancedSampling(_lmax=30, _NSIDE=16, _noisesig=1.0, data_mode='synthetic')

# Run joint Gibbs sampling over (alm, C_l)
traces, cl_samples, _ = run_gibbs_chain(
    model,
    num_samples=500,
    alm_sampler='hmc',
    alm_step_size=0.01,
    alm_n_lfs=10,
)
```

## Running Tests

```bash
make test
# or directly with pytest:
PYTHONPATH=diffcmb pytest
```

