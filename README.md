# diffcmb

[![Python Tests](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml/badge.svg)](https://github.com/OscarHickman/diffcmb/actions/workflows/test.yml)

Accurate CMB power spectrum sampling using TensorFlow Probability and advanced MCMC techniques (HMC and NUTS). The pipeline goes from ΛCDM cosmological parameters through CAMB → spherical harmonics → Bayesian posterior sampling over `{C_ℓ, a_ℓm}`.

healpy is Linux/macOS only. Windows users should use Google Colab or a VM.

## From MSci Dissertation to Now

This repo started as the codebase behind `docs/MSci_Report.pdf` ("Cosmology from the CMB with advanced sampling techniques", Imperial College London). That project sampled `{C_ℓ, a_ℓm}` **jointly in a single NUTS chain** (no Gibbs structure), at `N_side=128`, `2 ≤ ℓ < 60`, on a Google Colab CPU notebook with 25.5GB RAM — the report notes (§3.1) that full Planck resolution (`N_side=2048`, `ℓ_max=2500`) would need 1.8PB of RAM for the dense spherical-harmonic matrix, so it never ran on real full-resolution Planck data. Its stated future work (§5) was: run on a bigger machine, then infer ΛCDM parameters via MCMC from the recovered power spectrum.

The codebase has since moved well past that scope, not just scaled it up:

- **Different, more scalable algorithm.** Joint NUTS over `(C_ℓ, a_ℓm)` was replaced with a proper Gibbs sampler (Wandelt/Jewell/Larson, the same family of algorithm Commander uses): exact inverse-Gamma draws for `C_ℓ | a_ℓm`, plus either HMC or an exact conjugate-gradient Gaussian draw for `a_ℓm | C_ℓ` (IAT = 1 by construction — see `samplers.py`, ROADMAP Phase 0b).
- **Real Planck data at far higher resolution** — `ℓ_max=300`, `N_side=256` — via a custom multi-GPU/CPU matvec split (`model.py`), something the original 25.5GB Colab constraint made impossible.
- **A convergence failure mode the original diagnostics couldn't see.** float32 chains showed R-hat ≈ 1.0 for `C_ℓ` while `a_ℓm` R-hat hit 17,000–58,000 — false convergence invisible to the coarser R-hat/effective-sample-size checks used in the dissertation. Root-caused to float32 gradient noise; fixed by moving to float64 + the exact CG sampler.
- **A new research direction with no precedent in the dissertation at all**: a differentiable CMB lensing operator (`lensing.py`, Phase 1, validated against finite differences) and a four-block Gibbs sampler jointly inferring the unlensed CMB signal, its power spectrum, the lensing potential `φ`, *and* `φ`'s own power spectrum `C_L^φφ` (Phase 2) — a genuinely novel contribution beyond what Commander-style conjugate Gibbs sampling can do once the model stops being purely Gaussian. See `ROADMAP.md` for the full research case.
- The dissertation's own proposed next step (ΛCDM parameter inference from the recovered `C_ℓ`) still hasn't been done — it's Phase 2b on the roadmap. Effort instead went into fixing a scalability/correctness problem the original approach didn't know it had, and opening the lensing extension instead.

## Project Structure

```
diffcmb/diffcmb/                # Python package (see CLAUDE.md for module responsibilities)
├── power.py                    # CAMB power spectrum generation
├── alm.py                      # Noise map and single-pixel sph_harm
├── alm_utils.py                # All alm/map transforms (two index orderings)
├── tf_helpers.py                # TF weight tensor for psi3 term
├── model.py                    # CosmologyAdvancedSampling class + psi_tf
├── samplers.py                 # HMC/NUTS single-block wrappers + run_gibbs_chain (up to 4 blocks)
├── lensing.py                  # Differentiable lensing operator + φ|alm,C_ℓ and C_L^φφ|φ blocks
├── sht_ducc.py                 # Matrix-free spherical harmonic transforms (ducc0)
├── messenger.py                # Messenger-field sampler (closed-out route, kept for reference)
└── load_results.py             # Chain loading utilities

rust_sph/                       # Rust extension (optional, recommended)
├── spherical_harmonics.rs      # Holmes-Featherstone ALF recurrence (Rayon parallel)
├── Cargo.toml
└── pyproject.toml

scripts/                        # HPC entry points and diagnostics — not unit tests (see CLAUDE.md)
├── run_sampler.py               # CLI driver for HMC/NUTS/Gibbs chains
├── gate_*.py                    # One-off go/no-go checks before scaling a configuration
├── debug_*.py                   # Investigation scripts from specific past bugs
├── validate_*.py, smoke_*.py    # Production-scale validation and smoke runs
├── coverage_ensemble_chain.py, aggregate_coverage_ranks.py  # SBC/coverage test harness
├── analyze_*.py                 # R-hat / ESS / logp / correlation diagnostics
└── submit_*.slurm, *.slurm      # COSMA SLURM wrappers (paired with the matching .py script)

examples/                       # Getting-started and HPC-results notebooks

tests/                          # Fast, deterministic pytest coverage at small lmax
├── test_alm.py, test_alm_utils.py, test_power.py
├── test_model.py, test_samplers.py, test_messenger.py
├── test_lensing.py             # Lensing operator gradient validation
├── test_sht_ducc.py, test_sht_ducc_model_integration.py
└── test_cg_matvec.py           # CG matvec linearity/symmetry regression
```

## Installation

```bash
# Create venv and install Python dependencies
make setup

# Build the Rust spherical-harmonic extension (optional but strongly recommended)
# Requires: cargo (rustup) + maturin (pip install maturin)
make build-rust
```

The Rust extension (`cmb_sph`) parallelises spherical harmonic matrix construction using Rayon. Without it, `model._ensure_tf_tensors()` falls back to sequential scipy calls, which is significantly slower at large lmax.

## Quick Start

```python
from diffcmb import CosmologyAdvancedSampling, run_chain_hmc, run_chain_nut
import tensorflow as tf
import numpy as np

model = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)

initial_state = tf.constant(np.random.randn(len(model.x0)) * 0.1, dtype=tf.float64)

# HMC
samples, results = run_chain_hmc(model, initial_state, num_results=1000)

# NUTS
samples, results = run_chain_nut(model, initial_state, _step_size=0.01, num_results=1000)
```

See `examples/basic_usage.ipynb` for a full walkthrough.

## Performance Notes

| Component | Approach |
|-----------|----------|
| Spherical harmonic matrix | Rust + Rayon (Holmes-Featherstone recurrence), falls back to scipy |
| alm index reordering (`almmotho`/`almhotmo`) | Precomputed numpy fancy-index permutation |
| `splittosingularalm_tf` | `tf.scatter_nd` with precomputed indices, replaces O(lmax²) `tf.concat` loop |
| `psi_tf` | Compiled with `tf.function` on first call; graph reused for all subsequent HMC steps |

## Running Tests

```bash
make test
# or
PYTHONPATH=diffcmb pytest
```
