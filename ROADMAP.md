# diffcmb: Roadmap & Project Overview

`diffcmb` implements full-sky, curved-sky (HEALPix), differentiable joint Bayesian CMB lensing inference. It provides a joint Gibbs sampler over the unlensed temperature spherical harmonic coefficients ($a_{\ell m}$), the CMB angular power spectrum ($C_\ell$), the lensing potential field ($\phi$), and the lensing potential power spectrum ($C_L^{\phi\phi}$).

---

## Core Sampling Framework

The inference pipeline cycles through four Gibbs conditional blocks:

1. **$C_\ell \mid a_{\ell m}$** — Exact inverse-Gamma draw per multipole $\ell$.
2. **$a_{\ell m} \mid C_\ell, \phi$** — Hamiltonian Monte Carlo (HMC) sampling using matrix-free spherical harmonic transforms via `ducc0`.
3. **$\phi \mid a_{\ell m}, C_\ell$** — HMC sampling through the differentiable forward lensing deflection operator.
4. **$C_L^{\phi\phi} \mid \phi$** — Exact inverse-Gamma draw for the lensing power spectrum.

---

## Roadmap

### Phase 1: Sampler Validation & Coverage Testing
- **Rank & Coverage Ensemble**: Run multi-realisation simulation ensembles to verify exactness via Cook-Gelman-Rubin rank uniformity and credible interval coverage.
- **Equilibration & Mixing Validation**: Verify chain convergence across all angular scales for the full 4-block joint system.

### Phase 2: Scientific Deliverables & Posterior Characterization
- **Joint $(C_\ell, C_L^{\phi\phi})$ Posterior**: Extract and characterise the correlated joint posterior distribution of CMB temperature power and lensing potential power.
- **Uncertainty Propagation**: Quantify the error reduction and parameter accuracy compared to marginal and lensing-blind baselines.
- **Baseline Comparison**: Benchmark against published flat-sky and iterative quadratic estimator results.

### Phase 3: Real Data & Cosmological Parameters
- **Planck Data Application**: Apply the joint pipeline to Planck temperature data to evaluate full-sky curved-sky lensing consistency.
- **$\Lambda\text{CDM}$ Inference**: Derive cosmological parameter constraints from the sampled $C_\ell$ posteriors.

### Phase 4: Polarization & Future Extensions
- **Polarization (TQU)**: Extend the differentiable forward model and spherical harmonic transforms to spin-2 fields (TE, EE, BB).
- **Delensing Forecasts**: Evaluate delensing efficiency and primordial tensor-to-scalar ratio ($r$) recovery for next-generation experiments (LiteBIRD / CMB-S4).
