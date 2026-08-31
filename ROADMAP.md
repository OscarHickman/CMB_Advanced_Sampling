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

### 🛑 2026-08-24: alm ordering bug found — every φ-block result below is invalid

`lensing.py`'s `_alm_packed_to_hp`/`_alm_hp_to_packed` never called `almmotho`/`almhotmo`, so the packed φ vector was handed to healpy/ducc in author (L-major) ordering while they read m-major. Every coefficient sat at the wrong multipole; the per-ℓ prior in Blocks 1 and 4 was applied to modes the likelihood placed elsewhere. Fixed and covered by two new absolute-(L,m) tests (`achievements.md` has the full entry and why the round-trip test could never catch it).

**Consequences — do this before trusting anything in the sections below:**

1. **Every φ-block number recorded in this file and in `achievements.md` predates the fix and must be re-derived**, including the lag-1 autocorrelations quoted below (0.557, 0.945, 0.996), the `phi_mass_matrix='block'` GO at lmax=64, and the NUTS/MCLMC/Fisher NO-GOs. The *closed-out routes* list should be treated as provisional: those methods were judged against a scrambled target.
2. **FALSIFIED 2026-08-25 (job 11849969, harvested): the ordering fix alone does NOT fix φ-block mixing.** Post-fix `phi_mass_matrix='prior'`, Block 4 ON, lmax=64 baseline — same config as the pre-fix NO-GO job 11752452 (worst lag-1 0.975) — came back **worst lag-1 0.978** (bin `[2,10)`), essentially unchanged. Every ℓ-bin is bad (0.949-0.978), not just one. Full detail: `achievements.md`. **This means the Nystrom/Fisher/rescaling-move investigation was not chasing a pure ordering artifact** — the underlying φ-block geometry problem is real and still unsolved. Needs user direction on next step (see item 3 below).
3. **Actioned 2026-08-24: ensemble array `11848757` cancelled.** Tasks 0-5 had already completed against the broken path before cancellation (`results/analysis/coverage_ensemble_lmax64/`) — that output is invalid and must not be aggregated or cited; tasks 6-11 were cancelled before running. Do not resubmit the ensemble until a configuration clears the equilibration gate post-fix.
4. **The gate that produced most of those NO-GOs was itself a measurement artifact (2026-08-26).** Re-scoring the saved chains on τ_int/ESS/split-R̂ instead of raw lag-1 (`scripts/rescore_phi_equilibration.py`, job 11874935 — no new sampling) shows the lag-1<0.9 gate conflates *slowly-mixing but stationary* with *genuinely unequilibrated*. Under the corrected diagnostic the one post-fix run (11849969, plain `prior` mass matrix) has **3 of 4 ℓ-bins stationary and usable** (R̂ 1.000-1.050, ESS 37-47); the only real failure is the top band-edge bin `[60,64)` (R̂ 1.404). Full table + caveats: `achievements.md`.
5. **Fix-status is a timeline check, not a harvest date.** The ordering fix is commit `a16e9e6`, 2026-08-24 20:08. Jobs 11836793 and 11830353 both *ran to completion before it* despite being harvested after — their numbers are invalid regardless of how clean they look. **Only job 11849969 is post-fix.** Verify with `sacct -j <id> --format=Start,End -X` against `git log -- diffcmb/diffcmb/lensing.py` before citing any φ result.
6. **Harvested 2026-08-27: job 11874976 completed — NO-GO, and it survives the corrected diagnostic.** Re-ran job 11836793's exact config (lmax=64, `phi_mass_matrix='block'`, Block 4 ON) post-fix, then re-scored with `scripts/rescore_phi_equilibration.py` per the rule above. Result: **0/4 ℓ-bins pass** (R̂ 1.002-1.064, ESS 13.2-21.6) — unlike job 11849969 (plain `prior`, 2/4 bins clean + 1 marginal), this is not a gate artifact, it's a genuine failure across the whole spectrum. Current post-fix ranking is now **`prior` (no preconditioning) > `block` (Nystrom correction)** once Block 4 is on — the opposite of the pre-fix, Block-4-off finding that motivated building `block` in the first place. **Also found and fixed while re-scoring: `tau_int_geyer`'s `converged` flag was sign-inverted** (fixed in `scripts/rescore_phi_equilibration.py`; this revised 11849969's headline from "3/4 usable" to "2/4 clean + 1 marginal" — R̂/ESS/AR1-ratio numbers themselves were unaffected, only the derived pass/fail label). Full numbers: `achievements.md`.

   **User decision (2026-08-27): ship the coverage ensemble on `prior` + Block 4 (job 11849969's config), accepting the weak `[60,64)` bin.** Launched `scripts/submit_coverage_ensemble_lmax64_prior_cl4.slurm` → `results/analysis/coverage_ensemble_lmax64_prior_cl4/`. Do not reuse `results/analysis/coverage_ensemble_lmax64/` — that directory holds the invalid pre-fix chains from cancelled job 11848757. After all tasks complete, aggregate with `scripts/aggregate_coverage_ranks.py --indir results/analysis/coverage_ensemble_lmax64_prior_cl4 --thin 90` (thin chosen from the two clean bins' τ_int 70-78, rounded up; the `[60,64)` bin's own τ_int=326 is a known, accepted weak spot, not something this thin value fixes — report it separately in any coverage figure/table, not folded into an aggregate that hides it).

7. **Harvested, aggregated AND root-caused 2026-08-28: job 11887897 (12/12 COMPLETED) failed completely — cause found, fixed, validation running.** The aggregate rank test gave `phi_power` and `C_L^φφ` rank **0/8 in every ℓ-bin of all 12 realizations** (KS_p=0.0000 throughout), with φ sitting **1.9e3-2.5e5× above the true power** and frozen there. The CMB side was comparatively healthy (`alm`/`cl_TT` ratios 0.14-1.3), so this was φ-specific.

   **Root cause: `scripts/coverage_ensemble_chain.py` started the alm from a cold prior draw and never ran the MAP pre-solve that `scripts/pilot_coverage_equilibration.py` runs — so the equilibration gate and the production script were never running the same pipeline.** The mechanism, measured rather than assumed (job 11892269, `scripts/diagnose_phi_amplitude_identifiability.py`, forward evaluations only):
   - With `alm = truth`, the lensing likelihood minimises at **s=1**, the true φ amplitude (whole-map lensing S/N = 67). The model is sound and φ *is* identified.
   - With the ensemble's actual cold-start alm, that minimum moves to **s=30** — φ is the only remaining freedom to absorb a ~1000× larger residual, so the likelihood actively drives it up.
   - With Block 4 on, the refitted `C_L^φφ` makes the φ prior term **exactly scale-free**: substituting Block 4's conditional mean `S_L/(2(L-1.5))` gives `0.5·Σ_L S_L/C_L = Σ_L (L-1.5)`, independent of φ. Confirmed numerically to 8 s.f. — **1922.000 at every amplitude from s=0.1 to s=1000**, and `Σ_{L=2}^{63}(L-1.5) = 1922` exactly. Nothing pulls φ back down.
   - The chain then freezes (per-step scatter 1.6e-4 at a location 2.2e-2 ⇒ ~10⁴ sweeps to diffuse back), which is why it *looked* stationary.

   **Corroborating evidence:** chain starts are at the correct φ scale (so burn-in drove them up); φ power varies only 0.0-3% across the saved window where ~50% is expected; `logp` drifts monotonically across the entire "production" window with no plateau (burn-in never finished); alm-vs-truth cosine similarity is only 0.24-0.91 and *decreasing*; and across realizations **corr(alm cosine, log₁₀ φ inflation) = −0.78, p=0.003** — the worse the alm fit, the more φ inflates, exactly as the mechanism predicts. Every lmax=64 *pilot* (which has the MAP start) sits at the correct φ scale, ratio 0.09-2.5.

   **This exact failure had already been diagnosed once** — job 11663105 (2026-07-30) froze the φ block with `phi accept 0.004` for the same reason, which is why the pilot carries a mandatory MAP start and a warning comment. The ensemble script never inherited it. (Ensemble realization 002 shows accept 0.008, the same signature.)

   **Fixed 2026-08-28:** `coverage_ensemble_chain.py` now runs the data-driven MAP start (`--map_steps` default 2000, `--map_lr` 0.01) and `--n_burnin` defaults to 400, both copied from the pilot so gate and production match; `map_steps=0` reproduces the old behaviour for deliberate A/B only. Root cause locked down by two new tests in `tests/test_lensing.py` (`test_block4_refitted_prior_is_scale_free_in_phi_amplitude`, plus its fixed-spectrum contrast case) so the scale-free property can never again be mistaken for a prior that regularises the amplitude. **Single-realization validation job 11892308 running** (`scripts/submit_coverage_ensemble_lmax64_mapfix_validate.slurm` → `results/analysis/coverage_ensemble_lmax64_mapfix_val/`); PASS = φ/truth power ratio O(1), MAP-vs-truth alm cosine ≫0, `logp` plateauing. **Do not relaunch the full 12-chain ensemble until that validation passes**, and do not aggregate `results/analysis/coverage_ensemble_lmax64_prior_cl4/` — like the pre-fix directory before it, that output is invalid.

   **Standing lesson (added to discipline below): a gate must run the production script's own initialisation path, not a sibling script's.** Every φ equilibration verdict on record was measured through the pilot's MAP start; none of them ever tested how the ensemble actually starts.

   **PASSED 2026-08-30: validation job 11892308 (COMPLETED, 12:04:39, 2026-08-28) confirms the MAP-start fix.** Checked directly from `results/analysis/coverage_ensemble_lmax64_mapfix_val/chain_r000.npz`:
   - φ-power/truth ratio: **1.40-1.65** across the post-burn-in chain (healthy pilots land in 0.09-2.5; the broken ensemble was 1e3-1e5).
   - alm-vs-truth cosine similarity: **flat at 0.9998** for the entire 600-sample chain (broken run: 0.24-0.91 and decreasing).
   - `logp`: **plateaued** — linear-fit slopes of -0.20 and +0.07 over the two 300-sample halves, noise-level against a std of ~52 (broken run drifted monotonically with no plateau).

   `n_burnin=100` (not the new 400 default) was sufficient — the MAP start alone already gives alm cosine 0.9999, and this run confirms it end-to-end, so the full rerun below keeps `n_burnin=100` to match the validated config exactly rather than introduce an unvalidated change.

   **Actioned 2026-08-30: full 12-chain ensemble relaunched, job 11899585** (`scripts/submit_coverage_ensemble_lmax64_prior_cl4_mapfix.slurm` → fresh directory `results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix/`, same config as job 11849969 / the original `prior_cl4` ensemble — lmax=64, `phi_mass_matrix='prior'`, Block 4 ON — only the alm-init pipeline changed). ~4.8h/realization, 24h walltime, `durham` account at 22/200 jobs before submission.

   **HARVESTED AND FULLY CLEARED 2026-08-31: job 11899585, 12/12 COMPLETED, the ensemble is healthy and the sampler shows no evidence of incorrectness.** All `.err` files carry only benign TF/PTX warnings. Per-realization φ/truth power ratio **0.55-1.98** (broken run: 1e3-1e5); MAP-vs-truth alm cosine **0.9916-0.9998** on all 12. Aggregated with `--thin 90`.

   The raw aggregate looked alarming — `C_l^TT` FLAGged in all 4 ℓ-bins (mean_u 0.094-0.375, KS_p=0.0000) and `C_L^φφ` in 2. **All of those flags are artifacts of the rank statistic itself, now proven, not sampler bias.** The chain of evidence:

   1. **The spectrum "rank" ranks the truth against its own conditional's mode.** Both Block 1 and Block 4 draw from `InvGamma(α=L-0.5, β=S_L/2)`, whose mode is `β/(α+1) = S_L/(2L+1)` — which is exactly what `aggregate_coverage_ranks.py::realized_spectrum` calls the truth. InvGamma is right-skewed, so `P(draw < mode) < 0.5` **for a perfect sampler**, and bin-averaging over ℓ shrinks the spread while the offset stays, driving mean_u toward 0. The spectrum rows can therefore never be uniform; they are coverage, not calibration, exactly as the script header warns.
   2. **`C_l^TT` sits on the perfect-sampler null in every bin.** Simulating the null from the ensemble's own truth `S_l` (`InvGamma(l-0.5, S_l/2)`, 7 draws, 2000 reps): null 0.095/0.095/0.115/0.345 vs observed 0.094/0.104/0.094/0.375 — all four inside the 95% band. **Cleared.**
   3. **`C_L^φφ` sits on the null too, once the null includes φ's own scatter.** The frozen-φ null (0.107/0.141/0.147/0.346) is too narrow because it leaves only Block 4's InvGamma noise, which averages down across ℓ; the real chain's φ moves sweep-to-sweep, adding common-mode spread that does not average down. Redoing the null on the chain's *actual* φ trajectory gives 0.268/0.276/0.335/0.240 with 95% bands `[0.219,0.313]`, `[0.240,0.323]`, `[0.292,0.385]`, `[0.219,0.271]` — **observed 0.281/0.323/0.323/0.229: three comfortably inside, and `[10,30)` exactly on the 97.5% edge (0.323 vs 0.3229), i.e. no bin gives evidence of bias.** (`C_l^TT` needed no such correction because alm is pinned at cosine 0.9998, so freezing it is a good approximation; φ is not.) **Cleared.**
   4. **Block 4 is exact, tested directly rather than inferred.** Using the saved per-sweep `phi_samples`/`cl_phiphi_samples` pairs, `u = CDF_InvGamma(L-0.5, S_L(φ_i)/2)[C_{L,i}]` is uniform to **KS_p=0.53, mean_u=0.4970 over N=17856** (the lag-1 misalignment control degrades to KS_p=0.006, confirming the test has power). Block 4 draws precisely from its stated conditional.

   All four checks are reproducible via `scripts/validate_coverage_rank_nulls.py --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix --thin 90` (login-node, ~2 min) — the null bands above move by ~0.01 between seeds, so read them as bands, not thresholds.

   **What is left is genuine, and it is a target-definition issue, not a sampler bug.** The field-level ranks — the only rows this design supports as calibration — are `alm` mean_u=0.461 (pooled KS_p=0.026, mild) and **`phi` mean_u=0.367 (pooled KS_p=0.0040)**, i.e. posterior φ power runs high; measured per-bin chain/truth φ power is median 1.05-1.33 (range 0.53-2.28). **Analytically this is expected, because with Block 4 on the target's φ prior is improper.** Integrating the flat-improper-`C_L^φφ` joint over `C_L` gives marginal ∝ `S_L^{-(L-0.5)}`; against the `2L+1`-component radial measure `r^{2L}dr` that is `p(r) ∝ r¹`, i.e. **flat in `S_L` — improper, and rising with amplitude** (verified numerically; the identity `∫C^{-α-1}e^{-β/C}dC = Γ(α)β^{-α}` reproduces to 6 s.f.). This is the same fact as the already-recorded "Block 4's refitted prior is exactly scale-free in the φ amplitude", stated as a marginal. Consequences:
   - The φ amplitude is constrained **only by the lensing likelihood**, never by the prior.
   - The truth was drawn from `N(0, C_L^φφ,fid)`, a *proper* prior the sampler is not targeting — so **the φ field rank is not a valid calibration test while Block 4 is on.** Its non-uniformity is a prior mismatch by construction.
   - `alm` is much less affected (mean_u 0.461) because Block 1 has the same improper structure but the data pins alm far more tightly.

   **⟹ RESUME HERE NEXT SESSION — needs a user decision, do not launch unilaterally** (standing no-φ-pilot rule). The exactness evidence is as far as this configuration can take it: everything testable came back clean, and the one non-uniform row is explained by an improper prior rather than by the sampler. Options, in the author's recommended order:
   1. **Re-run the ensemble with Block 4 OFF** (`C_L^φφ` fixed at fiducial). Then the φ prior is proper, the SBC rank null *is* uniform, and the φ rank becomes a real calibration test — the strongest exactness claim available. Costs ~12 × 2.1h. Downside: no `C_L^φφ` samples, so it does not produce the headline differentiator on its own.
   2. **Give `C_L^φφ` a proper (weakly informative) prior** so the joint target is proper in the φ amplitude, then re-run. Best long-term answer and it protects the paper's headline object, but it is new sampler code plus a fresh gate.
   3. **Ship as-is and state the caveat** — defensible given (2)-(4) above, but a referee who checks the φ prior will find it improper, and the paper's differentiator *is* the `C_L^φφ` posterior.

   Do not aggregate `results/analysis/coverage_ensemble_lmax64_prior_cl4/` (job 11887897) or `results/analysis/coverage_ensemble_lmax64/` (job 11848757) — both invalid, cold-start/pre-fix respectively.

   **Real bug found and fixed while clearing this (2026-08-31): `run_gibbs_chain(seed=...)` never seeded the HMC.** `seed` created `np.random.default_rng(seed)` for the numpy-side draws (Blocks 1 and 4, mass-matrix probes, CG/messenger noise), but none of the five `one_step` calls pass a `seed=`, so TFP drew every HMC/NUTS momentum and Metropolis uniform from TensorFlow's **process-global** RNG. Measured: consuming 5 `tf.random.normal` draws before an otherwise identical `seed=7` chain moved the recovered `C_L^φφ` by 5× (4.12e-7 → 8.32e-8). This is what made `tests/test_samplers.py::test_gibbs_chain_sample_cl_phiphi_recovers_known_spectrum` "flaky" — it was never flaky, it was reading an unseeded stream, and the ROADMAP's guess that it was order-dependence was right for the wrong reason. Fixed by seeding the global stream in `run_gibbs_chain`; covered by `test_gibbs_chain_seed_is_immune_to_prior_global_tf_rng_state`, which asserts bit-identity and was confirmed to fail without the fix. Suite now **117 passed / 1 skipped / 0 failed**, ruff clean. **This does not invalidate job 11899585** — the 12 realizations still had properly seeded, independent *data* (numpy side), so the ranks remain valid draws; only run-to-run reproducibility of the HMC trajectories was affected.
4. The non-centred (φ, C_L^φφ) rescaling move (`phi_rescale_move`, commit `f606929`) is unit-tested and correct on its own terms, but the funnel diagnosis that motivated it rests on the invalid 0.557→0.945 comparison. Re-establish that comparison post-fix before spending pilot compute on the move — job 11849969's result may make it unnecessary.

**φ-block equilibration: the production configuration (lmax=64, Block 4 ON) is NO-GO, harvested 2026-08-24 (job 11836793).** Worst lag-1 autocorrelation 0.945 (gate <0.9), in bin `[30,60)` — up from 0.557 with Block 4 off (job 11781626, same lmax=64 setup). Turning Block 4 on materially degrades φ mixing even at the scale that previously passed. This confirms the risk flagged when the guard was relaxed: coupling low-L φ amplitudes to a resampled C_L^φφ hurts equilibration, though the worst bin is mid-L `[30,60)` here, not low-L as at lmax=128. Full detail: `achievements.md`. **Per the standing no-unilateral-tuning rule, do not launch another φ-equilibration pilot without asking first** — report this to the user and get direction (their preference order was: drop lmax further, lengthen the window, or raise `phi_n_lfs`).

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
- **A gate must run the production script's own initialisation path.** Every φ equilibration verdict on record was measured through `pilot_coverage_equilibration.py`'s MAP warm start, while `coverage_ensemble_chain.py` cold-started — so the gate never tested how production actually begins, and the ensemble failed 12/12 (2026-08-28, item 7 above). When a gate and the thing it gates are separate scripts, diff their setup before trusting the verdict.
- **Check calibration, not just mixing.** R̂/ESS/τ_int measure whether a chain is moving, not whether it is in the right place: the 2026-08-28 ensemble was frozen 10⁴-10⁵× away from the truth and still passed a mixing-based gate. Always compare a recovered quantity against its known truth scale as well.
- **A rank/coverage statistic needs its own null before any flag is read as bias.** The 2026-08-31 aggregate FLAGged `C_l^TT` in all four ℓ-bins at KS_p=0.0000; simulating what a *perfect* sampler produces under the same statistic reproduced those numbers to within the 95% band, because the statistic ranks the truth against its own conditional's **mode** (`InvGamma(α,β)` mode `= β/(α+1) = S_L/(2L+1)` = `realized_spectrum`). Build the null from the actual conditional, and build it with the same *conditioning-variable scatter* the real chain has — a null that freezes φ at truth was far too narrow and made a correct `C_L^φφ` look biased.
- **An intermittent test failure is a hypothesis, not a flake.** The "order-dependent or flaky" Block 4 test was a real unseeded-RNG bug in `run_gibbs_chain` (2026-08-31). It was carried in this file as a known-tolerated oddity for three days.
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations; re-check arXiv for curved-sky MUSE and the diffusion/generative route before every submission milestone (`literature.md`).
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
