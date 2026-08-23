# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; positioning/novelty argument is in `literature.md`; full detail in git history.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures are the critical path; everything else waits.

**Why this scope.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and discards the two things this project built: a differentiable forward model and a sampler (`literature.md`). That sets the bar:

- The product is *demonstrated* exactness, not asserted exactness — a convincing coverage/rank test outranks any additional scale.
- The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations — an object no competing method (MUSE, QE, Commander, diffusion) produces.
- Position vs learned inference is offensive, not defensive: an exact sampler is the reference standard learned posteriors get validated against (ANVIL/KARMA thesis) — but only as strong as the scale actually demonstrated.
- Deprioritised: CMBLensing.jl benchmark is a citation, not a science result; lmax scaling is not the route to impact.

**Positioning, decided 2026-08-06 (full reasoning: `literature.md`; consequences already actioned: `achievements.md`).** Broader scope, accepting scoop risk: the real-data Planck run and Phase 2b ΛCDM-parameter section are in scope for this paper, sequenced *after* the critical path below. Never lead with the differentiable machinery (Flinch made it table stakes) — lead with the joint (C_ℓ, C_L^φφ) posterior. Re-run the named-author arXiv scan (Millea, Seljak, Bayer, Loureiro) and the citation-hygiene grep before every submission milestone.

---

## Currently doing

### ⟹ NEXT SESSION — read this first

**φ-block equilibration: lmax=128 scale-up is NO-GO (harvested 2026-08-22, job 11830353). Block 4 + `phi_mass_matrix='block'` unblocked 2026-08-23; the production-configuration gate (job 11836793, lmax=64 with Block 4 ON) is in flight — check it first.**

- Root cause + fix at lmax=64, both closed (`achievements.md`): the φ-block posterior has cross-L Hessian coupling no diagonal-in-L mass matrix can represent; `phi_mass_matrix='block'` (a per-m-block Nystrom correction) fixes it there. NUTS alone does not (job 11781382, NO-GO) — confirms it was never a trajectory-length problem.
- **lmax=64 (job 11781626): GO, harvested 2026-08-18.** Worst lag-1 autocorrelation 0.557, worst drift 0.20σ.
- **lmax=128 (job 11795998 → resumed 11808346 → resumed 11830353): NO-GO, harvested 2026-08-22.** Full 3300-sample chain completed. Worst lag-1 autocorrelation **0.996** in the lowest-L bin `[2,10)` only — every other bin passes or comes close. This is a *new, low-L-specific* failure mode, not a repeat of the previously-falsified band-edge/stuck-bin pattern. Phi accept rate 0.238 (healthy, so not a step-size problem). Full detail: `achievements.md`.

**Do not launch another φ-equilibration pilot without asking first** (standing rule below). Leading hypothesis for the low-L failure, with a cheap offline test that needs no pilot: **fixed Nystrom rank (`phi_block_n_probes=6`) against a block size that doubles with lmax** — full mechanism, corroborating evidence and the proposed diagnostic are in `achievements.md`.

### ⚠ Blocker found 2026-08-23: the coverage ensemble and the headline figure need mutually exclusive configurations

Item 2's first bullet claims the joint (C_ℓ, C_L^φφ) figure "falls out of the coverage chains for free (`sample_cl_phiphi=True`)". **It does not, as the code currently stands.** `samplers.py::run_gibbs_chain` raises on `sample_cl_phiphi=True` together with `phi_mass_matrix in ('fisher','block')`. But Block 4 is what produces C_L^φφ, and `phi_mass_matrix='block'` is the *only* configuration that has ever cleared the φ equilibration gate. So today you can have an equilibrated φ block **or** the paper's headline differentiator, not both.

- **The guard is conservative, not mathematically required.** Its stated reason is that a burn-in-frozen mass matrix "would be inconsistent with a spectrum that keeps changing every sweep" — an *efficiency* argument. HMC leaves its target invariant for **any** fixed SPD mass matrix; only a mass matrix adapted from the current state during sampling would break detailed balance. A frozen block mass matrix built at a fiducial C_L^φφ is exact, merely suboptimal. Relaxing the guard + a test asserting the invariant distribution is unchanged looks like the cheap unblock.
- **Real risk to check, not assume away:** the lmax=64 GO was measured with **Block 4 off**. Turning Block 4 on couples low-L φ amplitudes to a C_L^φφ drawn from an inverse-Gamma with very few modes (L=2 has 5) — a centred-parameterisation funnel, exactly the geometry most likely to hurt *low L*, which is already the weak spot at lmax=128. So the production configuration has never been gated.
- **Consequence, actioned 2026-08-23:** the guard was relaxed (`run_gibbs_chain` now allows `sample_cl_phiphi=True` + `phi_mass_matrix='block'`; `'fisher'` stays excluded, out of scope) by freezing only the expensive likelihood-curvature estimate and cheaply rebuilding the diagonal prior-precision term from the resampled spectrum every sweep. Full suite green (102 passed/1 skipped), committed (`4f6eb74`). **Job 11836793 launched** (`submit_pilot_coverage_lmax64_block_cl4.slurm`, lmax=64, same config as the GO job 11781626 but with Block 4 now ON) — the actual production configuration, gated for the first time. ~11h expected; check `sacct -j 11836793` / `logs/pilot_coverage_lmax64_block_cl4_11836793.out` next session.

**Also note:** job 11781626's own verdict line already recommended lmax=64 for the ensemble — "lmax=64 at ~12s/sweep is a defensible configuration for the O(10-20)-chain rank/coverage ensemble". That recommendation was not acted on; effort went to scaling to 128 instead. Per the standing "demonstrated beats asserted" rule, a fully-passing rank test at lmax=64 outranks a partially-passing one at lmax=128.

## Todo, priority order

### 1. Exactness evidence (highest value)
- [ ] Multi-realization rank/coverage test at lmax≈100-150 (~10-20 independent chains, Cook-Gelman-Rubin rank uniformity for alm/φ, interval coverage for C_ℓ/C_L^φφ). Harness is built and smoke-tested (`scripts/coverage_ensemble_chain.py`, `scripts/aggregate_coverage_ranks.py`, `scripts/submit_coverage_ensemble.slurm`). **Blocked on job 11795998** (above).

### 2. Differentiator figures (what the paper is *for*)
- [ ] Joint (C_ℓ^TT, C_L^φφ) posterior correlation figure — **does NOT currently fall out of the coverage chains for free**: `sample_cl_phiphi=True` is rejected by `run_gibbs_chain` together with `phi_mass_matrix='block'`, the only configuration that equilibrates (see the ⚠ blocker above). Unblocking this is a prerequisite for the paper's headline figure, not a detail. A single-chain test so far is underpowered, not negative; needs a longer/pooled trace.
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [~] C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis of the same sims. **Lensing-blind reference chain DONE** (`achievements.md`; output `results/analysis/lensing_blind_baseline_lmax128.npz`). The lensing-aware side still needs the equilibrated joint chain.
- [ ] Write the position vs learned/amortised inference into the paper explicitly (intro + subsection) — the most likely referee question.
- [ ] Write the position vs **Flinch and Almanac** explicitly too — curved-sky differentiable/HMC (map, C_ℓ) inference now exists and is lensing-blind. A *second*, separate referee question, answered by the φ / C_L^φφ block. Draft language and citations already in `docs/paper/main.tex` (`achievements.md`).

### 3. Related-work obligation (not a science result)
- [ ] CMBLensing.jl benchmark write-up — cite their published numbers (Table II: 19-50h/GPU, autocorr lengths 5-33), don't install Julia. Design notes: `docs/notes/cmblensing_benchmark_notes.md`. State plainly: T-only vs their QU, lmax=300 vs their l<3500, small cosmology mismatch.

## 2. Real-data run — end-to-end demonstration

In scope for this paper (2026-08-06 decision above). Supporting evidence, not the headline (the A_L anomaly that originally motivated it is no longer live per Planck PR4/ACT DR6). Pitch as either an A_L post-mortem on Planck 2018 vs PR4, or the joint posterior as a lensing-consistency test for SO/LiteBIRD-class data. Sequenced after the lmax≈128 coverage/rank test and joint-posterior figure — those stay gate 1 regardless of scope.

- [ ] Run the joint sampler on real Planck data; report the joint (C_ℓ, φ) posterior's lensing-consistency verdict.

## 2b. Phase 2b — ΛCDM parameters from C_ℓ

In scope for this paper (2026-08-06 decision above). Routine, a cheap robustness section — derive standard ΛCDM parameter constraints from the posterior C_ℓ chains once Phase 2's coverage/rank test and joint-posterior figure are done. Sequenced after those, not before.

- [ ] Parameter-inference pass on the posterior C_ℓ^TT chains from the lmax≈128 (and, if run, real-data) chains; report against Planck/ACT/SPT baselines.

## 3. Phase 3 — polarization / LiteBIRD delensing (the science paper)

Full TQU joint analysis, after Phase 2 submits. Target reference: LiteBIRD lensing forecast (arXiv:2507.22618, QE/iterative pipeline — a sampling-based result fills a real gap).

- [ ] Spin-2 extension of alm utilities and the lensing operator (ducc0 spin-2 transforms).
- [ ] TQU joint likelihood (TT, TE, EE, BB); C_ℓ^TE breaks inverse-Gamma conjugacy → 2×2 inverse-Wishart or HMC.
- [ ] Simulated lensed TQU at LiteBIRD-like noise: delensing efficiency vs QE/iterative baselines, recovered r constraint.

## Parked (not started; recorded so the platform argument isn't lost)

- Phase 4 — lmax≥1000 scaling: tuning, not rearchitecture; profile only when Phase 2/3 need it.
- Phase 5 — non-Gaussian extensions (fNL, mask in-painting, learned priors, systematics): separate papers after Phases 2-3.
- Lensed-operator exact Block-2 draw: rejected shortcut, alternative unexplored (`achievements.md`). Not worth it unless HMC-on-both-blocks becomes a proven bottleneck.
- Re-tune matrix-free-HMC step-size adaptation: current regime mixes ~4x less efficiently per-sample than the old dense-SHT reference. Skip unless Phase 2 chains show it matters.

## Standing discipline

- **One critical path**: Phase 2 gates ✓ → coverage/rank test at lmax≈128 → joint-posterior differentiator figures → paper. Anything not on this waits. The manuscript runs in parallel rather than at the end.
- **Scope**: broader scope, accepting scoop risk (2026-08-06 decision above). Real-data Planck run and Phase 2b are in scope, sequenced *after* the critical path, not instead of it.
- **Never lead with the differentiable machinery.** Every abstract, talk, and intro leads with the joint (C_ℓ, C_L^φφ) posterior.
- **Watch authors, not only keywords.** Millea, Seljak, Bayer and Loureiro are the highest-probability source of a scoop; any φ or lensing extension of Flinch, Almanac, or CMBLensing.jl changes the plan. Named-author arXiv check before every submission milestone (`literature.md`).
- **Demonstrated beats asserted.** Prefer a converged result at a smaller scale over a non-converged one at a larger scale — every time, and say which one you have.
- **Precision**: fp64 end-to-end unless a mixed scheme is validated against fp64 chains (a float32 false-convergence trap is the standing counterexample).
- **Dense-reference discipline**: validate every new sampler/operator against an exact small-scale reference before production — this has caught real bugs repeatedly.
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations; re-check arXiv for curved-sky MUSE and the diffusion/generative route before every submission milestone (`literature.md`).
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
