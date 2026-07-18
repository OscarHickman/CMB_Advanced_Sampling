# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project documents — read these before planning any nontrivial work

This is an active research project (differentiable, curved-sky, joint Bayesian CMB lensing analysis) with three living docs at the repo root, each with a distinct job — don't duplicate content across them:

- **`ROADMAP.md`** — forward-looking only. The todo list of all outstanding work, in priority order, with a "Standing discipline" section of hard-won rules (precision, validation-before-production, claims hygiene). Check here first for "what's next."
- **`achievements.md`** — condensed record of everything validated, closed-out, or fixed so far, including sampler routes that were tried and abandoned (with why, so they aren't retried without new evidence) and real bugs caught by validation. Check here before proposing an approach that might already be tried.
- **`literature.md`** — annotated bibliography and the novelty/positioning argument (what's been done in the field, what hasn't, why the claim still holds). Check here before any framing/motivation writing.

When asked to "continue the roadmap," read `ROADMAP.md`'s next unchecked item and `achievements.md`'s most recent entries for context on what's already been validated.

## Commands

```bash
# Run all tests
make test
# or
PYTHONPATH=diffcmb pytest

# Run a single test file
PYTHONPATH=diffcmb pytest tests/test_alm.py

# Lint (ruff via pre-commit)
make precommit
# or directly:
ruff check diffcmb/ tests/ --fix

# Set up virtualenv with all dependencies
make setup

# Build the Rust spherical-harmonic extension (optional but recommended)
make build-rust

# Run the minimal entry point
PYTHONPATH=diffcmb python Main.py
```

## Architecture

The package lives in `diffcmb/diffcmb/` and is structured as a pipeline from raw cosmological parameters to MCMC samples, culminating in a 3-block Gibbs sampler over (alm, C_ℓ, φ):

```
CAMB params → power.py → alm_utils.py → model.py ─┬─ samplers.py (C_ℓ|alm exact, alm|C_ℓ,φ HMC)
                                                    └─ lensing.py + sht_ducc.py (φ|alm,C_ℓ HMC)
```

The `diffcmb/rust_sph/` directory contains an optional Rust extension that parallelises spherical harmonic matrix construction using Rayon, providing significant speedups for large lmax. It is superseded for production-scale (lmax≳300) runs by the matrix-free ducc0 SHT path (`use_matrixfree_sht=True`, see below) — the dense matrix (Rust-accelerated or not) doesn't scale.

### Module responsibilities

- **`power.py`** — Calls CAMB to produce a CMB angular power spectrum (`C_l` array). Only works for `lmax ≤ 2551`. Default ΛCDM parameters are `[H0=67.74, ombh2=0.0486, omch2=0.2589, mnu=0.06, omk=0.0, tau=0.066]`.

- **`alm.py`** — Minimal utilities: adding Gaussian noise to a pixel map (`noisemapfunc`) and a single-pixel spherical harmonic evaluation (`sphharm`).

- **`alm_utils.py`** — All alm/map transforms. There are **two alm index orderings** in use:
  - *Author ordering* (`mo`): row-major by `(L, m)` — used internally in `psi`
  - *Healpy ordering* (`ho`): column-major by `m` — used by all `hp.*` functions
  - `almmotho` converts author→healpy; `almhotmo` converts healpy→author. Functions prefixed with `hp` use healpy ordering; bare names use author ordering.
  - TF variants of core transforms (`splittosingularalm_tf`, `almtomap_tf`) accept and return TensorFlow tensors.

- **`tf_helpers.py`** — Builds `shape`, the `(lmax × len_alm)` tensor of `1.0`/`2.0` weights used in the `psi3` term.

- **`sht_ducc.py`** — Matrix-free spherical harmonic transforms via ducc0, wrapped in `tf.custom_gradient` (a `tf.py_function` escape hatch, since ducc0 isn't TF-native). `HealpixSHT` (masked-sky synthesis, `nthreads` param) and `full_synthesis_tf` (full-sky synthesis — lensing needs this because deflected positions can fall outside the eventual mask). This is what makes lmax=300+ tractable at all: the dense SHT matrix doesn't fit in GPU memory and is ~500x slower per call.

- **`lensing.py`** — The differentiable weak-lensing forward operator. `apply_lensing_tf`/`lens_map_tf` deflect an unlensed alm through φ to a lensed map; `psi_lensed` is the corresponding negative-log-posterior for the φ|alm,C_ℓ HMC block. Both branch on `model.use_matrixfree_sht`: `True` routes through `sht_ducc.py::full_synthesis_tf`; `False` uses the dense `model.sph_parts` Y-matrix. Only the full-sky matrix-free path is currently validated (see `achievements.md`) — masked-sky matrix-free lensing is untested.

- **`model.py`** — `CosmologyAdvancedSampling` is the central class. Its `__init__` runs the full setup pipeline (CAMB → alms → prior map → initial parameter vector `x0`), and accepts `use_matrixfree_sht=False, sht_nthreads=0` to select the SHT backend. TensorFlow-dependent tensors (`self.sph`, `self.shape`) are created **lazily** on the first call via `_ensure_tf_tensors()`, to allow importing without TF. `psi_tf` is the negative log-posterior for the unlensed alm|C_ℓ block. `compute_sl_np` computes the exact S_l = Σ_m |a_lm|² (with correct packed real/imag m=0-vs-m>0 weighting) that `sample_cl_given_alm`'s exact inverse-Gamma C_ℓ|alm conditional (`C_l|alm ~ InvGamma(l-0.5, S_l/2)`) is built on.

- **`samplers.py`** — `run_chain_hmc`/`run_chain_nut` are thin wrappers around `tfp.mcmc.HamiltonianMonteCarlo`/`NoUTurnSampler` for single-block sampling. `run_gibbs_chain` is the production 3-block Gibbs driver: Block 1 (C_ℓ|alm) is an exact inverse-Gamma draw; Block 2 (alm|C_ℓ,φ) is HMC (or `alm_sampler='cg'`, closed as biased when a φ block is active — see `achievements.md`); Block 3 (φ|alm,C_ℓ), enabled by passing `cl_phiphi_full`, is HMC and requires `alm_sampler` in `('hmc','cg')`. Supports checkpointing (`checkpoint_path`/`checkpoint_every`) that resumes both alm- and phi-block state, needed because production lmax=300 chains run tens of seconds per sweep under a SLURM walltime.

### Dependency guards

All heavy dependencies (`healpy`, `scipy`, `tensorflow`, `tensorflow_probability`, `camb`) are imported with `try/except` at module level and set to `None` on failure. Functions that need them raise `ImportError` at call time. This keeps the package importable in restricted environments (e.g. for lightweight testing).

### Parameter vector layout (`x0` / `_params`)

The sampled parameter vector encodes:
1. `_lncl[2 : lmax]` — log power spectrum coefficients (length `lmax - 2`)
2. `_realalm` — real parts of alm coefficients for `L ≥ 2, m ≥ 0` (excluding monopole/dipole and low-`m` imaginary parts)
3. `_imagalm` — imaginary parts for `m ≥ 2`

### `tests/` vs `scripts/`

`tests/` holds fast, deterministic pytest coverage at small lmax (`test_lensing.py`, `test_samplers.py`, `test_sht_ducc.py`, etc.) — run these for any code change. `scripts/` is production/validation infrastructure, not unit tests: `gate_*.py` are one-off go/no-go checks before scaling a sampler configuration to lmax=300 (results recorded in `achievements.md`, not re-run routinely); `debug_*.py` are investigation scripts from specific past bugs (see `achievements.md`'s "Real bugs" list for which); `submit_*.slurm`/`*.slurm` are the SLURM wrappers for the matching `.py` script, following a consistent pattern (`-A durham -p dine2`, per-job `$TMPDIR`, `PYTHONPATH` set to `diffcmb/`). When adding a new production-scale validation, mirror an existing gate/smoke script's structure rather than inventing a new one.
