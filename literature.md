# Literature — DiffCMB

*Annotated bibliography, with narrative assessment folded in below. Narrative assessment dated 2026-07-13; action items updated 2026-07-18; literature rescan and diffusion-route positioning 2026-07-29 (acted on in `ROADMAP.md`'s scope re-set); citations spot-checked and the CMB-diffusion action item closed in the 2026-07-29 doc audit.*

## 2026-07-29 rescan — novelty window confirmed open, but a competing paradigm is unaddressed

**The core claim survives, re-verified:** no curved-sky MUSE and no curved-sky joint
(a_ℓm, C_ℓ, φ) sampler has appeared. The 2×2 cell is still empty. The standing claims-hygiene
rule at the bottom of this file was run and passed.

**⚠ But the paper has no position on the diffusion/score-based route, and that is now the most
likely referee question.** Two papers define the threat, neither previously in this file:

- **"Denoising diffusion delensing"** (MNRAS 533, 423, 2024; arXiv:2405.05598) — score-based
  generative models reconstructing the CMB lensing convergence, pitched **explicitly as an
  alternative to HMC**, and claiming *uncorrelated* samples and therefore **more effective
  sampling of large angular scales than MCMC chains**.
- **JADE** (arXiv:2606.31988, 2026) — joint posterior over convergence maps *and* cosmological
  parameters from a single conditional diffusion model; amortised (~0.2 s/sample after training),
  and **explicitly requires neither a differentiable forward model nor inference-time MCMC**.

**Why this is sharper than a normal related-work gap.** DiffCMB's current blocker *is* HMC
autocorrelation in the φ block — the whole `phi_n_lfs` 20→80→200 investigation, and the reason
Phase 2 needs 3–4 sequential 24 h runs. The diffusion line markets the exact cure for the exact
symptom, and does so while discarding the two things DiffCMB spent its engineering budget
building (differentiability and a sampler). "Why fight HMC autocorrelation when diffusion gives
uncorrelated samples in 0.2 s?" is the first question a methods referee will ask, and the
manuscript currently has no answer on file.

**Three good answers exist — put them in the paper rather than discovering them at referee time:**
1. **Exactness.** HMC targets the true posterior asymptotically; a diffusion model samples a
   *learned* approximation whose error has no convergence guarantee and no diagnostic that
   certifies it. For a measurement intended to feed a tension debate, that difference is the
   product.
2. **Scope of the posterior.** Both cited works infer a lensing map (JADE adds cosmological
   parameters). DiffCMB's object is the **joint (a_ℓm, C_ℓ, φ)** posterior — the C_ℓ block, and
   its correlations with φ, is what makes the internal-consistency framing (assessment item 2)
   possible at all, and it is absent from the diffusion route.
3. **No training set required.** Diffusion priors must be trained on simulations, importing every
   bias of that suite; the Gibbs/HMC route needs only the forward model.

**Portfolio-level argument, and the strongest one available — use it.** The reason to prefer an
exact sampler over an amortised learned posterior for a tension-grade measurement is *precisely*
the thesis of ANVIL and KARMA: learned posteriors can pass their diagnostics and still be wrong
("calibrated but not accurate"), and in field-level cosmology specifically no known fix repairs
it (KARMA's verdict). **DiffCMB can cite our own trust-verdict triad as the methodological
justification for its architecture.** That converts a defensive answer into a coherent programme
and ties the fourth-front paper into the portfolio's central brand. Worth a paragraph in the
introduction, and worth naming in the thesis intro alongside the triad.

**Action items (2026-07-29):**
- [x] Carry this into the plan rather than leaving it as a note — `ROADMAP.md` was scope-re-set on
  2026-07-29 in direct response: the coverage/rank test was promoted to Priority 1 (exactness has
  to be *demonstrated*, since a learned competitor will show calibration plots), the joint
  (C_ℓ, C_L^φφ) posterior figure to Priority 2, and writing the position vs learned inference is
  now an explicit deliverable rather than an assumed intro paragraph.
- [x] Push the argument from defensive to offensive — the framing now on file is that an exact
  sampler is the **reference standard against which learned posteriors are validated**, which is
  the ANVIL/KARMA thesis applied to this problem class. Honest limit, recorded in `ROADMAP.md`:
  the framing is only as strong as the scale at which we can produce *converged* reference
  posteriors, so it must be stated with that scale attached, not in the abstract.
- [x] Check whether the diffusion line has published anything on **CMB** (not galaxy) joint C_ℓ+φ
  inference — **checked 2026-07-29 (second pass, same day): no.** JADE remains galaxy weak lensing,
  so it is a statement of the paradigm's ambition rather than a literal pre-emption, and that
  distinction stays load-bearing for how strongly the novelty claim can be worded. The one
  CMB-domain diffusion paper the second pass surfaced (arXiv:2512.22683, generative reconstruction
  of low-ℓ B-modes by reverse diffusion) *treats lensing as a contaminant to denoise away* rather
  than inferring φ — it does not produce a lensing posterior at all, let alone a joint one.
  Re-check at drafting time regardless; the paradigm is moving fast.
- Citation integrity spot-check (2026-07-29): all three diffusion-route arXiv IDs resolve to the
  papers described here (2405.05598 = MNRAS 533, 423 — the journal title drops the arXiv version's
  "Delight"; 2606.31988 = JADE; 2606.00803 = accepted to CVPR 2026). Independent re-scan for a
  curved-sky joint (a_ℓm, C_ℓ, φ) sampler or curved-sky MUSE again returned nothing — window open.

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

## The generative/diffusion route (added 2026-07-29 — the unaddressed competing paradigm)

See the rescan block at the top of this file for the positioning argument; these are the citations.

- **"Denoising diffusion delensing: reconstructing the non-Gaussian CMB lensing potential with
  diffusion models"**, MNRAS 533, 423 (2024), arXiv:2405.05598 — score-based generative
  reconstruction of the lensing convergence, framed as an **alternative to HMC**, claiming
  uncorrelated samples and better large-angular-scale sampling than MCMC. **The direct challenge
  to Phase 2's mixing work — must be cited and answered.**
- **JADE — "Joint inference of weak lensing convergence map and cosmology with diffusion models"**,
  arXiv:2606.31988 (2026) — single conditional diffusion model learning the joint convergence +
  cosmology posterior; amortised (~0.2 s/sample), **no differentiable forward model, no
  inference-time MCMC**. Galaxy weak lensing rather than CMB, so not a literal pre-emption, but it
  is the clearest statement of the paradigm's claim to obviate DiffCMB's two core design choices.
- **"Generative Diffusion Priors for 3D Mapping of the Dark Universe"**, arXiv:2606.00803 (2026) —
  same family, wider scope; cite as evidence the approach is a programme, not a one-off.
- *Positioning note:* the honest contrast is **exact-but-expensive vs learned-and-fast**, and the
  case for exactness is a calibration-trust case — which is exactly what ANVIL and KARMA
  establish. Cite the sibling repos rather than re-deriving the argument.

## Standing claims-hygiene rule

Re-scan arXiv for curved-sky MUSE / curved-sky field-level lensing samplers **before every submission milestone** — the window is open (verified 2026-07-13, re-verified 2026-07-18, **re-verified 2026-07-29**) but finite.

**Extended 2026-07-29:** scan the **generative/diffusion** route at the same time, not just the
sampler route. The 2026-07-29 pass found the competing paradigm had been developing for over a
year while every previous scan looked only for samplers and MUSE variants — the same failure mode
the dynamical-friction repo hit by searching only for merger-timescale papers. Search the *problem*
(joint CMB lensing posteriors), not only the *method family* we happen to use.
