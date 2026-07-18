# Literature — DiffCMB

*Annotated bibliography, with narrative assessment folded in below. Narrative assessment dated 2026-07-13; action items updated 2026-07-18.*

## Narrative assessment (2026-07-13)

**Verdict:** the novelty window is still open — no curved-sky MUSE and no curved-sky joint (alm, C_ℓ, φ) sampler has appeared — but the A_L science hook has weakened materially (anomaly largely resolved in Planck PR4/NPIPE and ACT DR6). **Reframe the real-data hook; lean harder on LiteBIRD delensing** (citations in the sections below).

**⚠ The A_L hook — reframe, don't drop.** "Interrogate a live anomaly" is no longer an honest pitch. Two honest replacements:
1. **Post-mortem with proper joint uncertainties**: joint (alm, C_ℓ, φ) posterior on Planck 2018 vs PR4 maps, showing where in parameter space the anomaly lived and how it dissolves — a fully-propagated-uncertainty account of a systematic's resolution.
2. **Internal-consistency machinery as the product**: the same joint posterior as a principled internal lensing-consistency test for future data (SO, LiteBIRD) — sell the capability, demonstrated on Planck, rather than the anomaly.

Either way, the beam/pixel-window/anisotropic-noise realism pre-condition stands unchanged.

**Action items:**
- [x] Reframe the A_L item in `ROADMAP.md`: post-mortem/consistency-machinery framing, not live-anomaly interrogation.
- [x] Cite arXiv:2507.22618 as the Phase-3 target-experiment reference in `ROADMAP.md` (2026-07-18, Section 3 intro).

## The occupied cells of the 2×2 (full-sky × joint-sampling) — the claim's boundaries

- **Millea, Anderes & Wandelt 2020**, arXiv:2002.00965 — **flat-sky joint (f, φ, r) Bayesian sampling (CMBLensing.jl); the nearest prior work.** Their reparametrisation lesson (naive block alternation mixes catastrophically at high S/N) is the Phase-2 mixing-risk gate's motivation.
- **Millea et al. 2021** — CMBLensing.jl applied to real SPTpol data (flat patches).
- **Millea & Seljak — MUSE**, arXiv:2112.09354 — marginal score expansion; not a sampler, no joint posterior; their "curved-sky HMC slightly out of reach" statement defines our empty cell. **No curved-sky MUSE exists as of 2026-07.**
- **SPT-3G 2019–2020 MUSE analysis**, arXiv:2411.06000 — production flat-patch MUSE; evidence the field's best still isn't sampling.
- **Commander / Eriksen et al. 2004-line Gibbs sampling** — full-sky (alm, C_ℓ) but lensing-blind and conjugate-only; the other occupied cell.
- **Anderes, Wandelt & Lavaux 2015**, arXiv:1412.4079 — Bayesian CMB lensing inference ancestor.
- **Carron & Lewis 2017**, arXiv:1701.01712 — iterative MAP lensing reconstruction (point estimate, not posterior); also the lensing-operator reference implementation (lenspyx/delensalot).

## Samplers & numerical methods

- **Elsner & Wandelt 2013**, arXiv:1210.4931 — messenger-field method (Phase 0c's basis; now closed at production scale — see `achievements.md`).
- **Papež, Grigori & Stompor 2018** — messenger as CG preconditioner; the named fallback if the HMC pivot fails.
- **Huffenberger & Næss 2018** — messenger-preconditioned CG for map-making; same fallback family.
- **Neal 2011 / Duane et al. 1987** — HMC; **TFP DualAveragingStepSizeAdaptation** — the step-size scheme `run_chain_hmc` uses (the Gibbs branch's Robbins-Monro is cruder; burn-in-length lesson recorded).
- **ducc0 (Reinecke)** — the matrix-free SHT backend; **s2fft** (JAX differentiable SHT) — the custom-vjp pattern cribbed from.
- **BeyondPlanck collaboration**, arXiv:2303.04819 — polarization Gibbs blocks (inverse-Wishart TE structure) for Phase 3.

## Science targets

- **LiteBIRD lensing forecast**, arXiv:2507.22618 (July 2025) — Planck+LiteBIRD full-sky QE reconstruction (72–78σ forecast); **the target experiment's pipeline is still QE/iterative — the sampling-based delensing gap Phase 3 fills**; also the benchmark/motivation citation.
- **Planck 2018 lensing & likelihood papers** — the A_L anomaly's origin.
- **Planck PR4/NPIPE likelihoods (CamSpec, HiLLiPoP)** — report the A_L anomaly much weakened/ΛCDM-consistent; **the anomaly is largely resolved** — basis of the 2026-07-13 reframe (post-mortem, not live target).
- **ACT DR6 lensing** (Madhavacheril et al. 2024) + **ACT DR6 extended models**, arXiv:2503.14454 — no lensing-amplitude excess; corroborates the resolution.
- **Revisiting A_L in Planck 2018 temperature**, arXiv:2310.03127 — the anomaly's anatomy; useful for the post-mortem framing.
- **Iterative/QE advances** (arXiv:2407.00228 non-Gaussian deflections; arXiv:2506.20667 noise-bias-minimising iterative estimator) — the current non-sampling state of the art to compare against on the full sky.

## Standing claims-hygiene rule

Re-scan arXiv for curved-sky MUSE / curved-sky field-level lensing samplers **before every submission milestone** — the window is open (verified 2026-07-13) but finite.
