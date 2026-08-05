# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; full detail in git history.*

**The claim being built:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). LiteBIRD-class delensing is structurally inaccessible to flat-sky codes. The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures below are the critical path; everything else waits.

**Why the scope is what it is.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and explicitly discards the two things this project spent its budget building (a differentiable forward model, and a sampler; see `literature.md`). That sets the bar for what this project's deliverables have to be:

- **The product is *demonstrated* exactness, not asserted exactness.** "HMC targets the true posterior asymptotically" is worth nothing to a referee if the φ block's own autocorrelation is poor. A convincing coverage/rank test is the single highest-value remaining deliverable — higher than any additional scale. Run it where the chain can actually equilibrate (lmax≈100-150), not at lmax=300 where it currently cannot be shown to.
- **The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations** — an object neither MUSE (marginal), QE (point estimate), Commander (lensing-blind), nor the diffusion route (learned, no C_ℓ block) produces at all.
- **Position vs learned inference is offensive, not defensive.** An exact sampler is the natural reference standard against which amortised/learned posteriors are validated (the ANVIL/KARMA thesis: learned posteriors can be calibrated yet inaccurate). That framing is only as strong as the scale at which converged reference posteriors are actually produced — state that scale plainly rather than overclaiming.
- **Deprioritised:** the CMBLensing.jl benchmark is a related-work obligation (cite published numbers, don't install Julia), not a science result; lmax scaling is not the route to impact and shouldn't consume budget before the coverage test lands.

---

## 1. Phase 2 — four-block Gibbs over (alm, C_ℓ, φ, C_L^φφ): the methods paper

All four blocks are implemented and validated (`achievements.md`): C_ℓ|alm exact inverse-Gamma; alm|C_ℓ,φ HMC (the conditional is exactly Gaussian at fixed φ — HMC is a cost choice, and the paper must say so); φ|alm,C_ℓ HMC; C_L^φφ|φ exact inverse-Gamma (opt-in `sample_cl_phiphi=True`).

**Open point-validation caveat:** the lmax=300 simulation-validation run (`scripts/validate_sim_lmax300_lensing.py`, `phi_n_lfs=80`, post-memory-leak-fix) completed cleanly but recovered φ power systematically under-estimated by roughly half to seven-eighths across l=10-300 — not yet root-caused (see `achievements.md`). Per the standing discipline below, this is **not** being chased with further lmax=300 compute; it's carried as an open, demonstrated-scale caveat while effort focuses on the items below.

### Priority 1 — the exactness evidence (highest-value remaining work)

- [ ] **Multi-realization rank/coverage test at lmax≈100-150 — the headline validation.** ~O(10-20) independent simulated realizations, each a full 4-block chain warm-started away from truth, checking that the rank statistic of the true (alm, φ) fields is uniform within each posterior (Cook-Gelman-Rubin / simulation-based calibration), with interval coverage (not strict rank calibration — see `achievements.md` for why) for the spectra C_ℓ and C_L^φφ. Reuse `scripts/validate_sim_lmax300_lensing.py`'s structure (realized-power comparison via `compute_sl_np`, not the CAMB ensemble mean). Requires `sample_cl_phiphi=True`, which forces `phi_mass_matrix='prior'`. O(10-20) concurrent chains sits comfortably under the 200-job `durham`/`dine2` submission cap.
  - Harness is written and smoke-tested end-to-end at lmax=24: `scripts/coverage_ensemble_chain.py` (one realization per SLURM array task), `scripts/aggregate_coverage_ranks.py` (offline rank pooling + KS uniformity), `scripts/submit_coverage_ensemble.slurm` (16-task array).
  - **Blocked on the pilot chain returning GO and a `--thin` value from its measured lag-k autocorrelation.** A pilot at lmax=128 (data-driven MAP start, `phi_n_lfs=80`, 600 samples) came back NO-GO on the script's fixed lag-1 gate (worst 0.981 vs 0.90 threshold), but an offline extended lag-table analysis found the autocorrelation genuinely decays through zero by lag~75-100 in every l-bin — a slow-mixing-but-stationary signature, not a stuck one (contrast with the fully-stuck lmax=300 run's profile: lag-1/lag-5 >0.9999, no decay at any lag). Acting on that: a window-extension follow-up job resumed the same chain to 2500 samples via checkpoint. **Next step once that completes:** re-run the lag-k/drift analysis on the longer trace. If lag-1 still exceeds 0.90 but the decay-to-noise pattern holds and deepens, that's grounds for a judgment-call GO with `--thin` set from the measured decorrelation lag — document that decision explicitly here rather than silently overriding the script's verdict.
  - **Gate rationale**: a rank/coverage test on a non-equilibrated chain produces confidently wrong uniformity plots (this project has been burned by that failure mode once already, see `achievements.md`). Hence one pilot chain, gated on direct lag-k autocorrelation (not any IAT/ESS estimator, both of which have separately misreported near-perfect mixing on stuck chains here), before funding the full ensemble.

### Priority 2 — the differentiator figures (what the paper is *for*; unblocked by Block 4)

- [ ] **The joint (C_ℓ^TT, C_L^φφ) posterior with its full correlation structure — as a figure.** The single object no competing method produces. Comes free from the Priority-1 pilot/ensemble chains (`sample_cl_phiphi=True` is all it needs). A one-chain test (`scripts/analyze_joint_posterior_pilot.py`) was inconclusive — underpowered, not negative (see `achievements.md`). **Next step**: re-run the analysis once the pilot's window-extension lands; the eventual ensemble's pooled traces are the fallback if still underpowered.
- [ ] **Quantify what joint sampling buys over marginal methods (per-mode uncertainty propagation) — as a figure, not a sentence.**
- [ ] **Quantify C_ℓ^TT bias reduction vs a lensing-blind Commander-style analysis of the same sims** — the "this changes an answer" result that turns a methods demo into a measurement claim.
- [ ] **Write the position vs learned/amortised inference explicitly into the paper** (intro paragraph + a subsection): exactness, joint-posterior scope, no training set required — and the offensive framing that an exact sampler is the reference standard for validating learned posteriors (cite ANVIL/KARMA). Not optional; it's the most likely referee question. Basis: `literature.md`.

### Priority 3 — related-work obligation (scope down; not a science result)

- [ ] **CMBLensing.jl benchmark — cite published numbers, do not install Julia.** Design notes: `docs/notes/cmblensing_benchmark_notes.md`. diffcmb's already-completed patch-scale gate (f_sky=0.016 ≈ 660 deg²) already matches CMBLensing.jl's own "BIG" config (650 deg², their reported practical ceiling for flat-sky HMC) — no patch redesign needed; direction is diffcmb full-sky sim → cut flat patch → compare. Proposed comparison set: C_ℓ^TT/C_ℓ^EE posterior recovery, C_L^φφ posterior recovery (headline plot), per-mode ESS/accept-rate table, wall-clock-per-effective-sample (computable from CMBLensing.jl's own published Table II — 19-50h/GPU, autocorr lengths 5-33 — without rerunning). CMBLensing.jl's f'-block CG is architecturally the *same* pattern diffcmb uses (phi drawn separately via HMC), not the CG-shortcut diffcmb rejected — state this explicitly to avoid an apparent (but false) contradiction. MUSE is confirmed marginal-only (no joint posterior).
  - **Caveats to state plainly in the paper rather than paper over:** (a) T-only vs polarization-only — CMBLensing.jl's examples are all QU; diffcmb is T-only until Phase 3. (b) lmax mismatch — diffcmb's validated scale is lmax=300 vs their l<3500; frame as a methodology/mixing comparison, not an information-content one. Do not inflate diffcmb's lmax to force a match — scale is not the route to impact. (c) Their fiducial cosmology differs slightly from `LCDM_PARAMS` (tau/omega_b/omega_c) — note it as a small confound when citing.

## 2. The real-data run — end-to-end demonstration

Both pre-conditions (beam+pixel window, anisotropic per-pixel noise) are done and validated (`achievements.md`), so this is now cheap once Phase 2 validates, but it's supporting evidence, not the paper's headline — the A_L anomaly that originally motivated it is no longer live (Planck PR4/NPIPE and ACT DR6 report it resolved). The honest pitch is one of: (a) **the A_L post-mortem** — a joint (alm, C_ℓ, φ) posterior on Planck 2018 vs PR4 maps showing where in the joint space the anomaly lived and how it dissolves; or (b) **internal-consistency machinery as the product** — the joint posterior as a principled lensing-consistency test for SO/LiteBIRD data, demonstrated on Planck.

- [ ] Run the joint sampler on real Planck data; report the joint (C_ℓ, φ) posterior's internal lensing-consistency verdict.

## 3. Phase 3 — polarization / LiteBIRD delensing (the science paper)

Full TQU joint analysis; the reason curved-sky matters at all. After Phase 2 submits. Target-experiment reference: the LiteBIRD lensing forecast (arXiv:2507.22618) — its pipeline is QE/iterative, so a sampling-based delensing result fills a real gap (see `literature.md`).

- [ ] Spin-2 extension of alm utilities and the lensing operator (ducc0 spin-2 transforms — Phase 1.5 infrastructure carries over).
- [ ] TQU joint likelihood with (TT, TE, EE, BB) block; C_ℓ^TE breaks inverse-Gamma conjugacy → 2×2 inverse-Wishart (BeyondPlanck structure) or HMC.
- [ ] Simulated lensed TQU at LiteBIRD-like noise: delensing efficiency vs QE/iterative baselines; recovered r constraint.

## 4. Parked (do not start; recorded so the platform argument isn't lost)

- **Phase 2b — ΛCDM parameters from C_ℓ**: scientifically routine; a cheap robustness section *after* the Phase 2 paper submits.
- **Phase 4 — lmax≥1000 scaling**: tuning, not rearchitecture, now the dense matrix is gone; profile only when Phase 2/3 need it.
- **Phase 5 — non-Gaussian extensions** (fNL sampling, mask in-painting, learned priors, systematics blocks): each a separate paper after Phases 2-3.
- **Lensed-operator exact Block-2 draw**: see `achievements.md`'s closed-routes section for why the current shortcut is rejected and what an alternative would need. Not worth pursuing unless HMC-on-both-blocks becomes a proven throughput bottleneck.
- **Re-tune the matrix-free-HMC step-size adaptation**: the current adapted step-size regime (0.16, accept 0.55-0.65) mixes ~4x less efficiently per-sample than the old dense-SHT reference's (0.10-0.11, accept 0.6-0.82); recovering that would make the already-large (~31x) net throughput win even larger. Skip unless Phase 2 chains show it matters.

## Standing discipline

- **One critical path**: Phase 2 gates ✓ → **coverage/rank test at lmax≈128** → **joint-posterior differentiator figures** → paper. The lmax=300 point-validation run and the real-data demo are supporting evidence, not the critical path; scale is explicitly not the route to impact. Anything not on this waits.
- **Demonstrated beats asserted.** Against a learned-inference competitor, "asymptotically exact" is only worth what the convergence evidence backs. Prefer a converged result at a smaller scale over a non-converged result at a larger one — every time, and say which one you have.
- **Precision rule**: fp64 end-to-end unless a mixed scheme with fp64 accumulation is validated against fp64 chains (a float32 false-convergence trap is the standing counterexample, see `achievements.md`).
- **Dense-reference discipline**: every new sampler/operator is validated against an exact small-scale reference before production — this has caught real bugs repeatedly (see `achievements.md`); don't skip it under time pressure.
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations (Millea+2020, MUSE, delensalot); re-check arXiv for curved-sky MUSE *and* the diffusion/generative route before every submission milestone (`literature.md`).
- **R-hat on C_ℓ alone is not convergence** — chains drifting together from the same start fool it; check the alm block and tail-ESS trends too.
- On dine2/cosma8 reused nodes, scripts need a job-private `$TMPDIR` (autograph cache collision).
- **`tf.gather` on a `tf.custom_gradient` output produces `IndexedSlices`, not a dense tensor**, as the upstream cotangent — any downstream `tf.py_function`/`.numpy()` call in that custom gradient's backward pass must `tf.convert_to_tensor()` it first. Applies to any future custom-gradient op fed into `tf.gather`/`tf.boolean_mask`.
- **Eager-mode `tf.py_function` calls in a hot loop leak memory** (a documented TF footgun — each eager invocation registers a callback in TF's process-global eager py-function registry, never freed); wrap in `@tf.function` instead, tracing once and reusing the graph. Re-verify "X cannot be traced" assumptions against the current code before trusting them — infrastructure changes underneath them. If a value must reach a traced function and changes between calls (e.g. Block 4's resampled `cl_phiphi_full`), it must be a `tf.Variable` read via `tf.convert_to_tensor`/live reference, not a plain Python/numpy value closed over or passed raw — the latter gets silently baked into the trace as a constant.
