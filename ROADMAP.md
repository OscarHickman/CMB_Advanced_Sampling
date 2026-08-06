# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; full detail in git history.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures are the critical path; everything else waits.

**Why this scope.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and discards the two things this project built: a differentiable forward model and a sampler (`literature.md`). That sets the bar:

- The product is *demonstrated* exactness, not asserted exactness — a convincing coverage/rank test outranks any additional scale.
- The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations — an object no competing method (MUSE, QE, Commander, diffusion) produces.
- Position vs learned inference is offensive, not defensive: an exact sampler is the reference standard learned posteriors get validated against (ANVIL/KARMA thesis) — but only as strong as the scale actually demonstrated.
- Deprioritised: CMBLensing.jl benchmark is a citation, not a science result; lmax scaling is not the route to impact.

> **2026-08-06 amendment:** the claim above still stands, but the *framing* around it has changed and the window has shortened. Read the next section before writing any draft text or spending compute.

---

## 2026-08-06 — Literature response (rescan of 2026-08-05, `literature.md` 61 → ~360 lines)

`literature.md` is the authority; this section is only what the rescan changes about the *plan*.

### What survived, and what did not

**Survived:** no curved-sky joint (a_ℓm, C_ℓ, φ, C_L^φφ) sampler exists, and no curved-sky MUSE exists. This was confirmed by two direct arXiv API enumerations, not keyword search alone. Scoop risk on the headline: low.

**Did not survive: differentiable curved-sky machinery as a contribution.** Flinch (arXiv:2510.26691, Oct 2025 — Crespi, Bonici, Loureiro, ..., **Millea & Seljak**) is a fully differentiable curved-sky field-level inference framework, validated on masked CMB temperature maps, recovering maps *and* angular power spectra. It has no φ block and no C_L^φφ block, so it occupies the *Commander cell* differentiably — the joint/lensed cell is still empty. But it makes differentiable SHTs under autodiff on the sphere **table stakes**, not a differentiator.

**The competitive landscape statement was incomplete.** Almanac (arXiv:2305.16134, Sellentin, Loureiro et al.) is an all-sky HMC over (map, C_ℓ) — full-sky, not Commander, not Gibbs, and previously uncited. "Commander is the other occupied cell" was wrong as written; Almanac and Flinch occupy it by different routes. Both must be cited and distinguished. Leaving Flinch uncited is now the single most likely "why didn't they cite X" referee complaint.

### The re-pitch (use this language; it supersedes any earlier framing)

**Lead with the object, not the machinery.**

- **What the paper is:** the first curved-sky, full-sky posterior over the *joint* (a_ℓm, C_ℓ, φ, C_L^φφ) — in particular the joint (C_ℓ^TT, C_L^φφ) posterior with propagated correlations, an object no existing method produces. It is offered as a **reference standard**: the exact posterior that fast, marginal, or learned methods (QE, MUSE, Commander/Almanac/Flinch, diffusion) get validated against.
- **What the paper must NOT claim as novel:** differentiable spherical-harmonic transforms, matrix-free SHTs under `tf.custom_gradient`, or "differentiable curved-sky inference" as such. Flinch published that. Present our differentiable infrastructure as *enabling machinery, consistent with Flinch*, in a methods subsection — never in the abstract's novelty sentence.
- **Concrete intro framing (draft language, to be sharpened at writing time):**
  > "Curved-sky differentiable field-level CMB inference is now established: Flinch (Crespi et al. 2025) demonstrates gradient propagation from masked full-sky temperature maps to cosmological parameters, and Almanac (Sellentin et al. 2023) samples all-sky maps jointly with their power spectra. What none of these frameworks includes is the lensing potential. Their posteriors are lensing-blind: they infer (map, C_ℓ) and, in Flinch's case, parameters, but carry no φ and no C_L^φφ block. Conversely, the methods that do treat lensing on the curved sky — quadratic estimators, iterative MAP reconstruction (Carron & Lewis 2017, arXiv:1704.08230), MUSE (Millea & Seljak 2022) — return point estimates or marginal constraints, not a joint posterior. This paper closes that cell: a full-sky, HEALPix, differentiable joint sampler over (a_ℓm, C_ℓ, φ, C_L^φφ), validated by simulation-based calibration, whose product is the joint (C_ℓ, C_L^φφ) posterior with its correlations propagated rather than assumed."
- **Keep the offensive positioning intact** (it strengthens, not weakens, under Flinch): an exact sampler is what learned/amortised posteriors get validated against (Doeser & Jasche arXiv:2606.10023; ANVIL/KARMA calibrated-but-not-accurate verdict). Cite the honest scale limit in the same breath.

### Time pressure and scope discipline

**Assessment.** The group with the strongest motive and the strongest infrastructure — Millea, Seljak, Bayer, Loureiro — is now **one block away**. Adding a φ block to Flinch is their natural next paper, and they have already published the sampler-efficiency tooling (MCLMC) that would make it work at scale. There is no evidence they are doing it, and no announced preprint; this is a motive-and-means assessment, not a report of a competing project.

**Decision made (2026-08-06): broader scope, accepting the scoop risk.** The narrow MPU below was the speed-optimised recommendation; the user has explicitly chosen to hold for broader scope instead. This does not relax anything on the critical path — lmax≈128 coverage/rank test and the joint-posterior figure are still gate 1, unchanged and still first — it means the items previously slated for deferral are back **in scope for this paper**, not a follow-up:

- The real-data Planck run (§2 below) — un-deferred.
- Phase 2b ΛCDM parameters from C_ℓ — un-deferred, moved from "Parked" into the main scope once Phase 2 submits.
- The per-mode uncertainty-propagation and C_ℓ^TT bias-vs-Commander figures (2.2/2.3 below) — no longer conditional on "falling out of the coverage chains cheaply"; back in as full scope items.
- lmax scaling stays parked (tuning, not a scoop-relevant result) unless a specific reviewer/venue reason emerges.

Consequence to watch, not yet acted on: broader scope means more wall-clock between now and submission, which is the exact axis the scoop-risk assessment above is about (Millea/Seljak/Bayer/Loureiro, "one block away"). No new mitigation is proposed here beyond the standing discipline (watch named authors on arXiv, re-run the claims-hygiene scan before submission) — worth revisiting if a competing preprint actually appears.

The original narrow-MPU recommendation is kept below for the record, since it's still the fallback if schedule pressure reasserts itself.

**Original recommendation — minimum publishable unit (MPU), superseded by the decision above.** Ship the *narrow* paper:

1. Converged joint sampler on simulations at lmax ≈ 128;
2. SBC rank/coverage evidence for (a_ℓm, φ) plus interval coverage for the spectra;
3. The joint (C_ℓ^TT, C_L^φφ) posterior correlation figure;
4. Related-work positioning vs Flinch / Almanac / MUSE / diffusion.

Everything else was proposed as a follow-up (real-data Planck run, Phase 2b ΛCDM parameters, lmax scaling, per-mode uncertainty-propagation and Commander-style-bias figures) if not falling out of the coverage chains for free. Superseded — see the decision above.

### Sampler-lever decision — re-opened (supersedes "raise `phi_n_lfs`")

The current plan names raising `phi_n_lfs` as the next lever if the pilot chain does not equilibrate. Two independent groups now report **MCLMC beating HMC by 1–3 orders of magnitude** at exactly this dimensionality, on exactly this symptom: Bayer, Seljak & Modi (arXiv:2307.09504, >1 order at ~2.6×10⁵ dimensions, gap *widening* with dimension) and Flinch (arXiv:2510.26691, ~3 orders on curved-sky CMB maps). The literature says the next lever is a **different integrator**, not more leapfrog steps.

| Option | Cost | Upside | Risk |
|---|---|---|---|
| Raise `phi_n_lfs` | Cheap, no new code; linear compute cost per sample | Known quantity; keeps the validated TFP/HMC stack and all dense-reference checks | May not fix it — if the pathology is geometry, more leapfrog steps buy proportionally little |
| Port φ block to MCLMC | TFP → MCLMC port is **not free**: new integrator, new tuning, and the dense-reference discipline must be re-run from scratch | 1–3 orders of magnitude if the reports transfer; also a citable methods point aligned with the field's direction | Schedule risk during a shortening window; **a converged HMC result beats a non-converged MCLMC one, every time** |

**Recommended decision rule (2026-08-06):**

1. **Do not port yet.** First finish the extended lmax≈128 pilot and the free diagnostic below.
2. **Try the cheap lever once, bounded.** If the pilot still fails its equilibration gate, raise `phi_n_lfs` by a factor of ~2–4 in *one* bounded run. Set the budget before launching.
3. **Trigger the port only on evidence of a geometry problem, not a compute problem.** Justify the MCLMC port if *either*: (a) autocorrelation time scales roughly linearly or worse with `phi_n_lfs` — i.e. cost per effectively-independent sample does not improve, so more leapfrog steps are buying nothing; or (b) the projected wall-clock for the ~10–20-chain coverage ensemble under the best HMC configuration exceeds the schedule the MPU decision above sets.
4. **If the port is triggered, scope it as a spike first** — φ block only, against the existing dense small-scale reference, with a hard abort if it does not clear that reference. Do not port the whole sampler.
5. **Standing discipline still governs**: demonstrated beats asserted. If HMC converges at lmax≈128, ship it and put MCLMC in future work.

**Explicit human decision required:** the `phi_n_lfs` budget in step 2 and the wall-clock threshold in step 3(b) are the user's to set — they depend on the submission target date.

### The free discriminating test (do this early, it costs nothing)

**Per-ℓ-bin φ-power deficit vs S/N plot**, from the existing lmax=300 outputs. The unexplained 51–86% φ-power deficit across ℓ=10–300 has no counterpart anywhere in the literature; what `literature.md` supplies is a ranked suspect list, and this one plot discriminates the top two:

- **Deficit largest at LOW S/N** → consistent with suspect (1), **under-mixing retaining a Wiener-suppressed start** (Millea, Anderes & Wandelt arXiv:2002.00965 make parameterisation-driven catastrophic mixing failure their central result; a φ block started from a MAP/Wiener-filtered φ that has not equilibrated keeps its starting amplitude, and Wiener filtering suppresses exactly the low-S/N modes). **Implication:** the lmax=300 deficit and the lmax=128 pilot's lag-1 autocorrelation of 0.981 are the same phenomenon at two scales, it is not a separate open problem, and it dissolves under work already on the critical path. No new compute justified.
- **Deficit largest at HIGH S/N** → consistent with the competing hypothesis, **sampler geometry**: Taylor, Ashdown & Hobson (arXiv:0708.2989) report HMC correlation lengths degrading specifically at the highest S/N. **Implication:** the mixing story is wrong, this is a real sampler-geometry problem, and it materially strengthens the case for the MCLMC port under rule 3 above.
- **Flat / no S/N trend** → neither story explains it; escalate, and do not write the deficit off as mixing in the draft.

**Ruled out by construction — say so in three sentences in the paper, don't spend compute:** N0/N1 and mean-field bias (a Gibbs sampler has no noise-bias subtraction step), non-Gaussian deflections (Gaussian-φ simulated input), foregrounds (foreground-free sims), and lensing-operator accuracy (validated to machine precision against the dense reference and `healpy`; standard set by Reinecke, Belkner & Carron arXiv:2304.10431).

### Re-ranked critical path (2026-08-06)

The shortened window does not change the critical path — it **hardens** it. Priority order:

0. Per-ℓ-bin deficit-vs-S/N plot (free, discriminating, feeds the sampler decision).
1. **lmax≈128 coverage/rank test** — unchanged as critical path, and now the highest-priority item in the project. It is the one result that cannot be pre-empted by someone adding a φ block to an existing framework, because it is evidence of *exactness*, not of capability.
2. Joint (C_ℓ, C_L^φφ) differentiator figure.
3. Manuscript (start now, in parallel — see below).

---

## Currently doing

**IN FLIGHT (check this first): job 11694912, `pilot_coverage_lmax128_phi240`, submitted 2026-08-06 ~14:32, dine2/durham, 24h walltime.** This is the bounded `phi_n_lfs` lever from the decision rule below (80 → 240, 3x), running 800 additional sweeps on top of the 2500-sample checkpoint. **Next AI agent: check `squeue -u dc-hick2` first — if it's finished (or died), re-run**
```
PYTHONPATH=diffcmb .venv/bin/python scripts/reanalyze_pilot_checkpoint.py \
    --checkpoint results/analysis/pilot_coverage_lmax128_v3_phi240_ckpt.npz --lmax 128
```
**against the resulting checkpoint and compare the l=[60,100) bin's lag-1/lag-k table to the phi_n_lfs=80 numbers below.** Three outcomes and what they mean:
- Lag-1 drops meaningfully (e.g. toward <0.9) and/or the bin now crosses |r_k|<0.2 by lag~150-200 like the other bins → the cheap lever worked; consider one more bounded raise or proceed to sizing the coverage ensemble's `--thin` from this chain's measured autocorrelation (item 1 below).
- No real improvement (lag-1 still ~0.99+, no decay by lag 200) → this is evidence *for* a geometry problem despite the deficit-vs-S/N plot pointing the other way; re-read decision rule 3(a) below (autocorrelation not improving with more leapfrog steps is itself the geometry trigger) and consider escalating toward the MCLMC spike.
- Partial improvement, ambiguous → use judgement per the "genuine decay vs stuck" distinction already established for the phi_n_lfs=80 chain (below); do not just re-run longer without deciding which case this is first.

**Background, for context:** The window-extension job before this one (11665429, `pilot_coverage_lmax128_v2_ckpt.npz`, phi_n_lfs=80) ran the chain out to 2500 samples; four of five l-bins decay cleanly, but the l=[60,100) bin got *worse* with more samples (lag-1 autocorrelation 0.996, up from 0.981 at 1800 samples; still doesn't drop under |r_k|<0.2 by lag 200, unlike every other bin which crosses by lag 150). Fixed lag-1 gate: NO-GO. More window alone doesn't fix that bin — hence the phi_n_lfs raise now running.

**Separately, the free per-ℓ-bin φ-deficit-vs-S/N plot is done** (`scripts/analyze_phi_deficit_vs_snr.py`, `results/analysis/phi_deficit_vs_snr_lmax300.png`, post-processing the existing `validate_sim_lmax300_phi80.npz` posterior, no new MCMC). Result: **deficit decreases smoothly and monotonically with S/N** (Spearman ρ=-0.78 across 27 clean l-bins spanning l=20-300; one l=[10,20) bin excluded as an ill-conditioned single-realization cosmic-variance fluke in the truth draw, not a sampler signal — see script docstring). Deficit runs ~75-90% at S/N≈1.3 (l≈20-100) down to ~35% at S/N≈8 (l≈295). This matches the **under-mixing / Wiener-suppressed-start** hypothesis (Millea, Anderes & Wandelt arXiv:2002.00965), not the sampler-geometry one (Taylor, Ashdown & Hobson arXiv:0708.2989) — worst-mixing is at *low* S/N, same direction as the l=[60,100) pilot-bin pathology. **Per the decision rule below, this plot alone does NOT trigger the MCLMC port** — it's evidence for a compute/mixing problem, which is exactly what job 11694912 is testing.

`scripts/pilot_coverage_equilibration.py` (run a chain), `scripts/reanalyze_pilot_checkpoint.py` (re-check diagnostics from a checkpoint without re-running).

## Todo, priority order

### 0. Added 2026-08-06 — do these first (cheap, unblocking, or window-driven)
- [x] **Per-ℓ-bin φ-deficit-vs-S/N plot** from the existing lmax=300 outputs — done, see "Currently doing" above. Verdict: under-mixing/Wiener-suppressed-start, not sampler geometry (ρ=-0.78). MCLMC not triggered by this evidence.
- [ ] **Start the manuscript file.** No manuscript exists for this paper. With the window shortening, the draft should be growing alongside the chains, not after them. Create it with the sections whose content is already fixed and does not depend on pending results: intro/positioning (the re-pitch above), related work (Flinch, Almanac, MUSE, Commander line, diffusion answers), methods (blocks, differentiable operator — framed as enabling machinery, *not* as the novelty), and the "what the deficit is not" paragraph. Leave results/figures as placeholders.
- [ ] **Cite Flinch (arXiv:2510.26691) and Almanac (arXiv:2305.16134)** in related work with the distinction stated explicitly: both are curved-sky (map, C_ℓ) samplers; neither has a φ or C_L^φφ block. Highest-priority citation item in `literature.md`.
- [x] **Citation-hygiene audit across this repo** for the two IDs corrected on 2026-08-05 — done 2026-08-06:
  - `arXiv:1701.01712` → `arXiv:1704.08230` fixed at `diffcmb/lensing.py:24`. No other occurrences outside `literature.md`.
  - `arXiv:2209.10512` — no occurrences outside `literature.md`; already correctly described there.
  - Re-run the grep before every submission milestone; both IDs are in `literature.md`'s claims-hygiene checklist.
- [ ] **Sampler-lever decision** (see the trade-off table and decision rule above). Needs a human call on the `phi_n_lfs` budget and the wall-clock threshold that would trigger an MCLMC port.
- [x] **Scope decision on the minimum publishable unit — decided 2026-08-06: broader scope, accepting scoop risk.** Real-data run and Phase 2b are back in scope for this paper (see the "Decision made" note under Time pressure and scope discipline above), not deferred.

### 1. Exactness evidence (highest value)
- [ ] Multi-realization rank/coverage test at lmax≈100-150 (~10-20 independent chains, Cook-Gelman-Rubin rank uniformity for alm/φ, interval coverage for C_ℓ/C_L^φφ). Harness is built and smoke-tested (`scripts/coverage_ensemble_chain.py`, `scripts/aggregate_coverage_ranks.py`, `scripts/submit_coverage_ensemble.slurm`). **Blocked on the pilot chain above returning GO** and a `--thin` value from its measured autocorrelation.

### 2. Differentiator figures (what the paper is *for*)
- [ ] Joint (C_ℓ^TT, C_L^φφ) posterior correlation figure — falls out of the coverage chains for free (`sample_cl_phiphi=True`). A single-chain test so far is underpowered, not negative; needs a longer/pooled trace.
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [ ] C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis of the same sims.
- [ ] Write the position vs learned/amortised inference into the paper explicitly (intro + subsection) — the most likely referee question.
- [ ] *(2026-08-06)* Write the position vs **Flinch and Almanac** explicitly too — curved-sky differentiable/HMC (map, C_ℓ) inference now exists and is lensing-blind. This is a *second*, separate referee question from the learned-inference one, and the answer is the φ / C_L^φφ block. Draft language is in the 2026-08-06 re-pitch.
- [ ] *(2026-08-06 scope note, superseded)* The original MPU recommendation made items 2.2/2.3 conditional on falling out of the coverage chains cheaply. Superseded by the broader-scope decision above — 2.2 and 2.3 are full scope items now, not conditional. Item 2.1 (the joint correlation figure) remains the paper's headline object regardless.

### 3. Related-work obligation (not a science result)
- [ ] CMBLensing.jl benchmark write-up — cite their published numbers (Table II: 19-50h/GPU, autocorr lengths 5-33), don't install Julia. Design notes: `docs/notes/cmblensing_benchmark_notes.md`. State plainly: T-only vs their QU, lmax=300 vs their l<3500, small cosmology mismatch.

## 2. Real-data run — end-to-end demonstration

**2026-08-06: in scope for this paper** (the broader-scope decision above un-defers this — it is no longer pushed to a follow-up). Supporting evidence, not the headline (the A_L anomaly that originally motivated it is no longer live per Planck PR4/ACT DR6, so it still isn't the *reason* to run it). Pitch as either an A_L post-mortem on Planck 2018 vs PR4, or the joint posterior as a lensing-consistency test for SO/LiteBIRD-class data. Sequenced after the lmax≈128 coverage/rank test and joint-posterior figure — those stay gate 1 regardless of scope.

- [ ] Run the joint sampler on real Planck data; report the joint (C_ℓ, φ) posterior's lensing-consistency verdict.

## 2b. Phase 2b — ΛCDM parameters from C_ℓ

**2026-08-06: in scope for this paper** (un-deferred from Parked by the broader-scope decision above). Routine, a cheap robustness section — derive standard ΛCDM parameter constraints from the posterior C_ℓ chains once Phase 2's coverage/rank test and joint-posterior figure are done. Sequenced after those, not before.

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
- Re-tune matrix-free-HMC step-size adaptation: current regime mixes ~4x less efficiently per-sample than the old dense-SHT reference; would enlarge the already-large (~31x) net throughput win. Skip unless Phase 2 chains show it matters.

## Standing discipline

- **One critical path**: Phase 2 gates ✓ → coverage/rank test at lmax≈128 → joint-posterior differentiator figures → paper. Anything not on this waits. *(2026-08-06: unchanged and hardened. The manuscript now runs in parallel rather than at the end — the window is short enough that "write it after" is itself a risk.)*
- *(2026-08-06)* **Scope decision made: broader scope, accepting scoop risk** (see "Time pressure and scope discipline" above). The real-data Planck run and Phase 2b are back in this paper's scope, sequenced *after* the critical path above, not instead of it. This does not change the critical path itself — it changes what ships alongside it once gate 1 clears.
- *(2026-08-06)* **Never lead with the differentiable machinery.** Flinch made curved-sky differentiability table stakes. Every abstract, talk, and intro leads with the joint (C_ℓ, C_L^φφ) posterior. Treat this as a hard drafting rule, not a preference.
- *(2026-08-06)* **Watch authors, not only keywords.** Millea, Seljak, Bayer and Loureiro are the highest-probability source of a scoop; any φ or lensing extension of Flinch, Almanac, or CMBLensing.jl changes the plan. Add a named-author arXiv check to the pre-submission scan (`literature.md`, standing claims-hygiene rule, item 3).
- **Demonstrated beats asserted.** Prefer a converged result at a smaller scale over a non-converged one at a larger scale — every time, and say which one you have.
- **Precision**: fp64 end-to-end unless a mixed scheme is validated against fp64 chains (a float32 false-convergence trap is the standing counterexample).
- **Dense-reference discipline**: validate every new sampler/operator against an exact small-scale reference before production — this has caught real bugs repeatedly.
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations; re-check arXiv for curved-sky MUSE and the diffusion/generative route before every submission milestone (`literature.md`).
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- On dine2/cosma8 reused nodes, scripts need a job-private `$TMPDIR` (autograph cache collision).
