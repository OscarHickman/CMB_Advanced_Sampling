# Achievements — DiffCMB

*Condensed record of completed and closed-out work. Full derivation/debugging detail is in git history (the pre-restructure `ROADMAP.md`). The forward plan lives in `ROADMAP.md`.*

## Validated foundations

- **Phase 0 — unlensed Gibbs baseline on real Planck data (lmax=300, float64): converged and trusted.** C_ℓ R-hat median 1.026, 0% of multipoles above 1.1; ~48% sampling efficiency. This is the reference everything downstream is checked against.
- **The float32 false-convergence trap was caught and became a standing precision rule**: C_ℓ R-hat can look converged while the alm block's R-hat blows up by orders of magnitude from gradient noise. fp64 end-to-end unless a mixed scheme is explicitly validated.
- **Phase 1 — differentiable lensing operator**: custom TF implementation, gradients w.r.t. both alm and φ validated against finite differences and an independent FD-free gold-standard Jacobian (exact linear deflection-field via unit-impulse alm × analytic weight gradients).
- **Phase 1.5 — dense SHT matrix eliminated (the hard gate)**: matrix-free ducc0 SHT behind `tf.custom_gradient`, forward+backward 0.018s at lmax=300 (~500× under the gate; the dense path was 9.38s and didn't fit in GPU memory). CG symmetry/linearity/PD checks pass matrix-free; the lensing op is graph-traceable.

## Real bugs found and fixed (each caught by dense-reference validation, not by trusting derivations)

- **φ-block HMC gradient**: the FD backward pass could re-query a shifted angle into a different interpolation cell, producing wild step-size-dependent gradients — replaced with an analytic bilinear-interpolation gradient; the regression tests' own FD ground truth was also unstable and fixed. Block 3 gradients are production-trustworthy.
- **alm precision convention**: uniform 1/C_ℓ used where the real spherical-harmonic convention requires 2/C_ℓ for m>0 — a genuine variance error, though not the cause of any sampler stall.
- **Messenger conditional derivation**: the plausible-looking naive `t|s,d` precision was subtly wrong (needs the reduced noise N−T); caught against an exact dense reference before touching production.
- **SHT operator normalisation**: AᵀA is not a scalar multiple of identity — needs per-mode m=0/m>0 weights *and* empirical calibration (HEALPix quadrature is ~1–2% off the analytic constant, enough to bias specific modes by tens of posterior SEs).
- **HMC + matrix-free SHT JIT-compile incompatibility**: `jit_compile=True` cannot compile through `tf.py_function`, so the combination had silently never run — a one-line fix unlocked ~0.34s/sweep production HMC (see ROADMAP).

## Closed-out sampler routes (do not revisit without new evidence)

- **Phase 0b — diagonally-preconditioned CG on the masked sky: abandoned.** The Jacobi preconditioner degrades ~4 orders of magnitude under a realistic mask (f_sky≈0.74); extrapolated cost 10⁴–10⁵ iterations per Gibbs step. Diagnosed in miniature and quantified.
- **Phase 0c — messenger-field sampler: built, validated at small scale, and closed as non-viable at production scale.** The full arc: correct conditionals validated against dense references; the masked-sky divergence traced to off-diagonal AᵀA coupling; a block-diagonal-by-m correction built that matches the exact dense fix's accuracy (the off-diagonal energy is 99.7% same-m with a parity selection rule); production benchmark at m_group_size=5 = 354.68s/sweep. The production run completed but **failed validation — intrinsic critical slowing down** at n_alm≈90k: ESS 0.6%, C_ℓ biased low, monotonic drift. Systematically ruled out as fixes: more inner iterations (would need ~25–85 *hours* per sweep), τ² annealing (refuted by targeted toy tests — the current τ² is already optimal), mask topology, C_ℓ dynamic range, and the block approximation itself (the exact dense correction mixes equally slowly). The slowdown is dimensionality-intrinsic to plain messenger Gibbs — the known failure mode that motivated messenger-as-CG-preconditioner work in the literature.

## Positioning (settled)

- **Novelty claim scoped to survive review**: the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). CMBLensing.jl owns flat-sky joint sampling; Commander is full-sky but lensing-blind; MUSE is marginal, not a sampler. The window is finite (curved-sky MUSE announced) — Phases →2 are the critical path.
- **Prior-art audit done honestly** before any claim was drafted; every "first" carries scope qualifiers and nearest-prior-work citations.
