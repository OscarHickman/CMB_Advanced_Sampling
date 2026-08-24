# Literature — DiffCMB (Paper 7)

*Annotated bibliography with the novelty/positioning argument folded in. Every arXiv ID below
was verified on its abstract page unless explicitly marked otherwise. Last full rescan:
2026-08-05.*

---

## ⚠ Verdict (last checked 2026-08-05)

**The core claim survives: no curved-sky joint (a_ℓm, C_ℓ, φ, C_L^φφ) sampler exists, and no
curved-sky MUSE exists.** Checked by keyword search *and* by direct arXiv API enumeration of
every `astro-ph.CO` submission with "CMB lensing" in the abstract back to 2026-03, and every
`astro-ph.CO` submission with "field-level" in the abstract back to 2025-12. The complete
post-2026-03 CMB-lensing list contains **no sampler paper at all** — it is cross-correlations,
parameter constraints, and estimator engineering (2605.18659 control variates, 2606.07745
CMBolic emulators, 2607.05784 SPT-3G summer-survey QE). The field-level list contains no CMB
lensing entry. **Scoop risk on the headline: low.**

**But the strategic picture changed materially this pass, and the cause is one paper.**

### 1. Flinch (arXiv:2510.26691) is the most important reference this project has not been citing

**Crespi, Bonici, Loureiro, Ruiz-Zapatero, Sladoljev, Li, Bayer, Millea & Seljak (30 Oct
2025)** — *Flinch: A Differentiable Framework for Field-Level Inference of Cosmological
parameters from curved sky data*. Fully differentiable, high-performance field-level inference
on **angular maps on the curved sky**; gradients propagate from individual map pixels to
cosmological parameters; validated on **masked CMB temperature maps**, reconstructing both maps
and angular power spectra; ~40% tighter than pseudo-C_ℓ.

Read the author list. **Millea and Seljak** — i.e. CMBLensing.jl and MUSE — are now co-authors
on a curved-sky differentiable field-level CMB framework. Three consequences, all of which
belong in the paper rather than in a referee report:

- **It does not scoop the claim.** Flinch's abstract contains no lensing potential, no φ, no
  C_L^φφ. It infers (map, C_ℓ, cosmological parameters) on the curved sky — i.e. it occupies the
  *Commander cell* of the 2×2, differentiably. The lensed/joint cell is still empty. The 2×2
  boundaries below are unchanged.
- **It removes "differentiable curved-sky machinery is hard" as a contribution.** DiffCMB's
  matrix-free-SHT-under-`tf.custom_gradient` engineering (`achievements.md`, Phase 1.5) is no
  longer a differentiator on its own — it is table stakes. The differentiator is and must be
  stated as **the joint posterior including φ and C_L^φφ**, exactly as `ROADMAP.md` already
  says. Do not let any draft lead with the differentiable-SHT engineering.
- **It shortens the window.** The group with the strongest motive to build a curved-sky joint
  lensing sampler now has published curved-sky differentiable infrastructure. Adding a φ block
  to Flinch is a natural next paper for them. This is the single strongest argument in the file
  for prioritising the lmax≈128 coverage test over anything else.

**It also hands over a concrete, tested fix for DiffCMB's actual bottleneck.** Flinch reports
**MicroCanonical Langevin Monte Carlo (MCLMC) beating HMC by nearly three orders of magnitude in
sampling efficiency** at its highest resolutions, corroborating **Bayer, Seljak & Modi
(arXiv:2307.09504)**, who report >1 order of magnitude over HMC at ~2.6×10⁵ dimensions with the
gap *widening* with dimension. DiffCMB's one live blocker is φ-block HMC autocorrelation at
lmax=128 (`ROADMAP.md`, "Currently doing"). Two independent groups now report that the standard
fix for exactly this symptom, in exactly this dimensionality regime, is not "more leapfrog
steps" but "a different integrator." See the Open items.

### 2. Almanac (arXiv:2305.16134) was missing and is the closest full-sky HMC neighbour

**Sellentin, Loureiro, Whiteway, Lafaurie, Balan, Olamaie, Jaffe & Heavens (2023)** — HMC
sampling of **all-sky** noiseless maps *and* their auto/cross power spectra, millions of
parameters, handles highly variable S/N, spin-2 E/B/EB without EB-leakage. This is a full-sky
HMC (map, C_ℓ) sampler that is *not* Commander and *not* Gibbs, and this file did not cite it.
A referee who knows the sphere-sampling literature will know it. It is lensing-blind, so the
2×2 is unaffected — but "Commander is the other occupied cell" is now an incomplete sentence;
Almanac and Flinch occupy it too, by different routes.

### 3. Two citation errors fixed this pass — both referee-visible

1. **Carron & Lewis 2017 (iterative MAP lensing) was cited as arXiv:1701.01712. That is wrong.**
   1701.01712 is **Carron, Lewis & Challinor, "Internal delensing of Planck CMB temperature and
   polarization"** — a different paper. The MAP-reconstruction paper is **arXiv:1704.08230, Phys.
   Rev. D 96, 063510 (2017)** (verified on the abstract page). Fix before this reaches a `.bib`.
2. **arXiv:2209.10512 was described here as "MUSE's own follow-up, still flat-sky/patch-based."**
   It is **Millea, "Improved Marginal Unbiased Score Expansion (MUSE) via Implicit
   Differentiation"** — a *methodological* paper (implicit differentiation; test cases are Neal's
   funnel, Bayesian NNs, probabilistic PCA), with no sky geometry at all. The correct statement
   is: *no curved-sky MUSE application has been published*; 2209.10512 is geometry-agnostic
   machinery, not evidence either way. The conclusion is unchanged; the wording was wrong.

### 4. Nothing found bears directly on the 51–86% φ-power deficit — but one lead is strong

No paper reports a comparable systematic φ-power deficit in a joint sampler. Nothing in the
literature diagnoses it. But **Millea, Anderes & Wandelt 2020 (arXiv:2002.00965)** is the right
place to look: their central methodological result is that the *naive* joint parameterisation
mixes catastrophically, and that the ancillary-vs-sufficient reparameterisation is what makes
the chain move — a slowly-mixing φ block warm-started from a MAP/Wiener-filtered estimate will
*retain the Wiener suppression of its starting point*, which is a low-amplitude φ, and a
51–86% deficit that grows with S/N-starved multipoles is the expected signature of that, not of
a wrong conditional. See the dedicated section below. **This is a hypothesis, not a literature
finding — no source states it about this configuration.**

---

## The 2×2: full-sky × joint-φ-sampling. The claim's boundaries

The claim is a cell in {flat-sky, curved-sky} × {marginal/point-estimate, joint sampling of φ}.
Cell (curved-sky, joint sampling) is empty. Everything below is a neighbour that occupies one of
the other three.

### Flat-sky, joint sampling — the nearest prior work

- **Millea, Anderes & Wandelt 2020**, arXiv:2002.00965, Phys. Rev. D 102, 123542 — *"Bayesian
  delensing delight: sampling-based inference of the primordial CMB and gravitational lensing."*
  Flat-sky joint (f, φ, r, A_φ) Bayesian sampling; CMBLensing.jl. **The nearest prior work, full
  stop.** Their reparametrisation lesson (naive block alternation mixes catastrophically at high
  S/N) motivated this project's Phase-2 mixing-risk gate — and is the leading suspect for the
  φ-deficit.
- **Millea et al. 2021**, arXiv:2012.01709, ApJ 922, 259 — *Optimal CMB Lensing Reconstruction
  and Parameter Estimation with SPTpol Data.* CMBLensing.jl on real SPTpol data, 100 deg²
  polarization; A_φ = 0.949 ± 0.122, 17% smaller errors than their own QE pipeline. The
  "sampling works on real data" existence proof, and the benchmark for the CMBLensing.jl
  related-work paragraph (`docs/notes/cmblensing_benchmark_notes.md`).
- **Anderes, Wandelt & Lavaux 2015**, arXiv:1412.4079, ApJ 808, 152 — Bayesian CMB lensing
  inference ancestor; the Gibbs-over-(CMB field, φ) idea in its original form.
- **"Bayesian delensing of CMB temperature and polarization"**, arXiv:1708.06753 — same lineage,
  flat-sky. *Author list unverified — check before citing.*

### Curved-sky, marginal or point-estimate — the other occupied cell

- **Millea & Seljak 2022 — MUSE**, arXiv:2112.09354, Phys. Rev. D 105, 103531. Marginal unbiased
  score expansion: asymptotically-unbiased marginal constraints on global parameters, ~6M latent
  dimensions, *not a sampler and no joint posterior*. Their own statement that curved-sky HMC is
  "slightly out of reach" is the sentence that defines this project's empty cell. **No curved-sky
  MUSE application has been published as of this scan.**
- **Millea 2022**, arXiv:2209.10512 — MUSE via implicit differentiation. Geometry-agnostic
  methodology (see correction above), not a curved-sky MUSE.
- **SPT-3G 2019–2020 MUSE analysis**, arXiv:2411.06000 — production MUSE on real data (lensing +
  delensed EE); evidence that the field's best lensing inference still is not sampling.
- **Carron & Lewis 2017**, arXiv:**1704.08230**, Phys. Rev. D 96, 063510 — iterative MAP lensing
  reconstruction (LensIt). A point estimate, not a posterior. ~2× improvement on σ(r) vs QE for
  S4. *Corrected ID — see Verdict.*
- **Carron, Lewis & Challinor 2017**, arXiv:1701.01712 — internal delensing of Planck T and P.
  Cite separately if the Planck real-data section needs it; do **not** use for the MAP claim.
- **Belkner, Carron et al. 2023 — CMB-S4 iterative internal delensing**, arXiv:2310.06729, ApJ
  (2024) — `delensalot`: the first lensing-reconstruction pipeline optimal for arbitrary sky
  coverage, 92–93% B-lensing power removed. **The curved-sky state of the art that is not a
  sampler** — the thing Phase 3's sampling-based delensing has to beat or match.
- **Darwish 2025**, arXiv:2503.03682 — optimal *joint* MAP reconstruction of multiple
  line-of-sight distortion fields (lensing + birefringence + patchy screening) accounting for
  mutual contamination. Note the word "joint" in the title: a referee may cite it as prior art.
  It is joint over *distortion fields*, is a MAP point estimate, and has no C_ℓ or C_L^φφ block.
  State that distinction explicitly.
- **Iterative/QE frontier**: arXiv:2407.00228 (non-Gaussian deflections in iterative optimal
  reconstruction, PRD 110, 103520); arXiv:2506.20667 (iterative estimator minimising
  instrumental-noise bias, PRD 2026). The non-sampling comparison baselines.
- **Namikawa & Sherwin 2026**, arXiv:2605.18659 — faster realisation-dependent bias via control
  variates (~5× on the total cost of an ACT/SO-like lensing power-spectrum measurement). Cite as
  evidence the QE pipeline is still being actively cost-engineered — i.e. the incumbent is not
  standing still.

### Curved-sky, joint sampling of (map, C_ℓ) but lensing-blind — the third cell, now crowded

- **Eriksen et al. 2004 / Jewell, Levin & Anderson 2004 / Wandelt, Larson & Lakshminarayanan
  2004 — the Commander line.** Full-sky Gibbs over (a_ℓm, C_ℓ), conjugate-only, lensing-blind.
  The direct structural ancestor of DiffCMB's Blocks 1–2. *Cite the original trio at
  journal-formatting time; IDs not pinned here.*
- **Eriksen et al. — "A self-contained guide to the CMB Gibbs sampler"**, arXiv:0905.3823 — the
  pedagogical reference for the conjugate (a_ℓm | C_ℓ) / (C_ℓ | a_ℓm) structure. Useful for the
  methods section's exposition.
- **Racine, Jewell, Eriksen & Wehus 2016**, arXiv:1512.06619 — *Cosmological Parameters from CMB
  Maps without Likelihood Approximation.* The joint-move step that fixes low-S/N Gibbs
  degeneracy between signal and spectrum. **Relevant to Block 1/2's mixing**, not just to
  related work.
- **"Improved Gibbs samplers for CMB power spectrum estimation"**, arXiv:2111.07664 — the
  successor line on the same conjugate-block mixing problem. *Author list unverified — check
  before citing.*
- **BeyondPlanck collaboration**, arXiv:2303.04819 — polarization Gibbs blocks, inverse-Wishart
  TE structure. The reference for Phase 3's TQU extension.
- **Cosmoglobe**, arXiv:2306.15511 — end-to-end CMB cosmological parameter estimation without
  likelihood approximations. The "global Bayesian analysis" framing DiffCMB's internal-
  consistency pitch lives adjacent to.
- **Sellentin, Loureiro et al. 2023 — Almanac**, arXiv:2305.16134 — **NEW this pass.** All-sky
  HMC over maps and auto/cross-spectra in multiple bins; millions of parameters; spin-2 E/B/EB
  without EB-leakage. Cite as the nearest full-sky HMC (not Gibbs) neighbour, and note it is
  model-independent by design (statistical isotropy only) — a different goal from DiffCMB's.
- **Crespi, Bonici, Loureiro, ..., Millea & Seljak 2025 — Flinch**, arXiv:2510.26691 — **NEW
  this pass, and the most consequential entry in this file.** See the Verdict. Differentiable
  curved-sky field-level inference from masked CMB temperature maps to cosmological parameters;
  MCLMC ≫ HMC; no φ. **Must be cited, must be distinguished, and its author list must inform the
  schedule.**
- **Taylor, Ashdown & Hobson — Hamiltonian sampling for CMB power spectra**, arXiv:0708.2989 —
  the HMC-instead-of-Gibbs ancestor; reports HMC correlation lengths comparable to or better than
  Gibbs *except at the highest S/N*. Directly relevant prior evidence for DiffCMB's own
  block-choice. *Author list unverified — check before citing.*

---

## The competing paradigm: diffusion / learned posteriors

**The most likely referee question is the diffusion/score-based route, and the paper needs an
answer on file.** Nothing new appeared this pass; the threat set is stable.

- **"Denoising diffusion delensing"**, arXiv:2405.05598, MNRAS 533, 423 (2024) — score-based
  generative reconstruction of the CMB lensing convergence, pitched **explicitly as an
  alternative to HMC**, claiming uncorrelated samples and better large-angular-scale behaviour
  than MCMC chains. The single most dangerous citation for this paper.
- **JADE**, arXiv:2606.31988 (2026) — *Joint inference of weak lensing convergence map and
  cosmology with diffusion models.* Joint posterior over convergence maps **and** cosmological
  parameters from one conditional diffusion model; amortised (~0.2 s/sample); explicitly requires
  **neither a differentiable forward model nor inference-time MCMC**. Galaxy weak lensing, not
  CMB — not a literal pre-emption, but the clearest statement of the paradigm's ambition to
  obviate DiffCMB's two core design choices.
- **Zhao, Scognamiglio, Doré & Bouman 2026 — "Generative Diffusion Priors for 3D Mapping of the
  Dark Universe"**, arXiv:2606.00803 — same family, wider scope; evidence the approach is a
  programme, not a one-off.
- **Likhit & Saha 2025**, arXiv:2512.22683 — generative reconstruction of low-ℓ CMB B-modes via
  reverse diffusion (VE-SDE), trained on r=0.001 spectra for ECHO. The one *CMB-domain* diffusion
  paper found: it treats lensing as a contaminant to denoise away rather than a field to infer.
  **No lensing posterior, joint or otherwise.**
- **arXiv:2511.04792** — blind strong-lensing inversion with score-based models. Galaxy lensing,
  off-topic; recorded so a future scan does not re-litigate it.

**Why this is sharper than a normal related-work gap:** DiffCMB's current bottleneck *is* HMC
autocorrelation in the φ block. The diffusion line markets the exact cure for the exact symptom,
while discarding the two things DiffCMB spent its engineering budget building. "Why fight HMC
autocorrelation when diffusion gives uncorrelated samples in 0.2 s?" is the first question a
methods referee will ask.

**Four answers, to state in the paper rather than discover at referee time:**

1. **Exactness.** HMC targets the true posterior asymptotically; a diffusion model samples a
   *learned* approximation whose error has no convergence guarantee and no diagnostic that
   certifies it. For a measurement intended to feed a tension debate, that difference is the
   product.
2. **Scope of the posterior.** Both cited works infer a lensing map (JADE adds cosmological
   parameters). DiffCMB's object is the joint **(a_ℓm, C_ℓ, φ, C_L^φφ)** posterior — the C_ℓ and
   C_L^φφ blocks, and their correlations with φ, are absent from the diffusion route and are what
   make the internal-consistency framing (`ROADMAP.md` §2) possible at all.
3. **No training set required.** Diffusion priors must be trained on simulations, importing every
   bias of that suite; the Gibbs/HMC route needs only the forward model.
4. **Someone has to be the reference.** See next section — this is the offensive version of the
   argument and the one to lead with.

**Portfolio-level argument — use it.** The reason to prefer an exact sampler over an amortised
learned posterior for a tension-grade measurement is precisely the ANVIL/KARMA thesis: learned
posteriors can pass their diagnostics and still be wrong ("calibrated but not accurate"), and in
field-level cosmology specifically no known fix repairs it. DiffCMB can cite that trust-verdict
work as the methodological justification for its architecture. Worth a paragraph in the
introduction.

**Honest limit:** this framing is only as strong as the scale at which converged reference
posteriors are actually produced. If that's lmax≈128 and the diffusion papers operate at
lmax≫1000, say so plainly rather than overclaiming (`ROADMAP.md` scope discussion).

---

## "Exact sampler as reference standard" — the supporting literature

The reference-standard positioning is not a bare assertion; there is a citable practice of
validating fast/learned inference against exact MCMC, and it is growing.

- **Doeser & Jasche 2026**, arXiv:2606.10023 (Learning the Universe) — in high-dimensional
  field-level inference, matching posterior means/marginals/cross-correlations does **not** imply
  correct uncertainty structure, established **by checking against HMC reference posteriors**.
  This is the argument DiffCMB is built to serve, made by someone else, in a neighbouring
  regime. **Cite prominently in the introduction.**
- **Mishra 2026**, arXiv:2606.16248 — *Benchmarking Exact, GP-Emulated, and Simulation-Based
  Inference for Late-Time Cosmology.* Treats exact MCMC as the gold standard and measures GP/SBI
  against it (agreement to 0.3σ on easy data, drifting to ~1.5σ on harder combinations).
  Low-dimensional and late-time, so it is a *precedent for the protocol*, not a competitor.
- **ANVIL / KARMA** (in-house, `../ANVIL/`, `../karma/literature.md`) — the calibrated-but-not-
  accurate verdict and the C2ST instrument. Own the general claim by citing them rather than
  re-deriving it.

---

## Validation, convergence and coverage — the critical path's own literature

The lmax≈128 rank/coverage test is the paper's headline evidence. Its methodology needs citing
properly, and one subtlety in `achievements.md` has a literature home.

- **Cook, Gelman & Rubin 2006**, *Validation of Software for Bayesian Models Using Posterior
  Quantiles*, JCGS 15, 675 — the original posterior-quantile validation scheme. **Note the
  published correction (Taylor & Francis, 2017)** to the distributional claim about posterior
  quantiles; cite the corrected form, not the 2006 statement, or a statistician referee will
  catch it.
- **Talts, Betancourt, Simpson, Vehtari & Gelman 2018 — SBC**, arXiv:1804.06788 — rank-statistic
  simulation-based calibration; the correct modern form of the above. **This is the protocol
  `scripts/aggregate_coverage_ranks.py` implements** and the one the paper should name.
- **Modrák et al. — "Simulation-Based Calibration Checking for Bayesian Computation: The Choice
  of Test Quantities Shapes Sensitivity"**, arXiv:2211.02383 — SBC's sensitivity depends
  entirely on which test quantities are ranked. Directly load-bearing: DiffCMB ranks per-ℓ-bin
  φ-power summaries, and this paper is the citation for why that choice is not innocent. **Read
  before finalising the coverage-test design.**
- **Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021**, arXiv:1903.08008, Bayesian Analysis
  16, 667 — rank-normalised, folded, split R̂ plus quantile-local ESS. The paper's convergence
  gates should use *this* R̂, not Gelman–Rubin 1992, and should show **rank plots rather than
  trace plots** — their explicit recommendation, and a cheap way to make the convergence section
  look current. Aligns with `ROADMAP.md`'s "R-hat on C_ℓ alone is not convergence."
- **`achievements.md`'s SBC-scope finding has no literature counterexample.** The observation
  that strict SBC applies to the fields (a_ℓm, φ) but not to the spectra — because Blocks 1 and 4
  have flat/improper implied priors, so there is no θ_true ~ p(θ) to rank — is correct and
  self-consistent with Talts et al.'s prior-sampling requirement. Nothing found contradicts it.
  **Label the spectrum result "interval coverage against realized power," never "calibration."**

---

## Bearing on the open 51–86% φ-power deficit

**⚠ 2026-08-24: `lensing.py` had an alm author/healpy ordering bug (fixed, commit `a16e9e6`,
`ROADMAP.md`'s stop-block) that scrambled every φ coefficient onto the wrong multipole, including
in every pilot run this section's autocorrelation numbers (e.g. "0.981" below) are drawn from.
Treat the deficit-vs-mixing argument below as a hypothesis motivated by pre-fix evidence, not a
conclusion — re-derive the lag-1/deficit numbers post-fix (job 11849969 in flight) before citing
them.**

**No paper found reports or explains a comparable deficit.** State that plainly. What the
literature does supply is a ranked list of suspects, all cheap to test:

1. **Under-mixing retaining a Wiener-suppressed start (most likely).** Millea, Anderes & Wandelt
   (arXiv:2002.00965) make the parameterisation of the joint (f, φ) chain their central
   methodological result: the naive alternation mixes catastrophically, and the fix is a
   reparameterisation, not more compute. A φ block that has not equilibrated, started from a
   MAP/Wiener-filtered φ, keeps its starting amplitude — and Wiener filtering suppresses exactly
   the low-S/N modes, which is where the deficit is largest. **The lmax=300 chain's φ-power
   deficit and the lmax=128 pilot's lag-1 autocorrelation of 0.981 are plausibly the same
   phenomenon at two scales.** Testable without new physics: does the deficit shrink monotonically
   with chain length / `phi_n_lfs`? If yes, it is mixing, not bias, and it is not a separate open
   problem at all.
2. **The competing hypothesis has a literature home too.** Taylor, Ashdown & Hobson
   (arXiv:0708.2989) report HMC correlation lengths degrading specifically **at the highest
   S/N** — the opposite regime. If the deficit turns out to be worst at *high*-S/N multipoles
   rather than low, suspicion should move from the Wiener-start story to the sampler geometry.
   The l=10–300 span reported in `achievements.md` is not resolved finely enough to distinguish
   these; a per-ℓ-bin deficit-vs-S/N plot is the discriminating diagnostic and costs nothing.
3. **Ruled out by construction, but say so.** The deficit is *not* the QE/iterative N0/N1 or
   mean-field bias family (arXiv:2506.20667, arXiv:2407.00228) — a Gibbs sampler has no
   noise-bias subtraction step. It is *not* the non-Gaussian-deflection effect (arXiv:2407.00228)
   at Gaussian-φ simulated input. It is *not* foreground-induced (arXiv:2406.15351,
   arXiv:2502.20801) in a foreground-free simulation. Naming these and dismissing them costs
   three sentences and pre-empts three referee questions.
4. **Lensing-operator accuracy is not a suspect.** The operator is validated to machine precision
   against the dense reference and against `healpy` (`achievements.md`), and the accuracy standard
   for curved-sky lensing operators is set by Reinecke, Belkner & Carron (arXiv:2304.10431), whose
   `ducc0`/`lenspyx` implementation DiffCMB uses. Do not spend compute here.

**Bottom line: nothing in the literature turns the deficit into a known failure mode, and
suspect (1) predicts it would dissolve under the lmax≈128 equilibration work already on the
critical path.** That supports `ROADMAP.md`'s decision not to chase it with lmax=300 compute.

---

## Samplers, numerics and differentiable infrastructure

- **Neal 2011 / Duane et al. 1987** — HMC. **TFP `DualAveragingStepSizeAdaptation`** — the
  step-size scheme `run_chain_hmc` uses.
- **Bayer, Seljak & Modi 2023 — MCLMC for field-level inference**, arXiv:2307.09504 — >1 order of
  magnitude over HMC at ~2.6×10⁵ dimensions, gap **widening** with dimension. **Corroborated by
  Flinch (arXiv:2510.26691) at ~3 orders of magnitude on curved-sky CMB maps.** Two independent
  reports, both in this project's dimensionality regime, both on the exact symptom that is
  blocking the critical path. See Open items.
- **Elsner & Wandelt 2013**, arXiv:1210.4931 — messenger-field method; this project's abandoned
  Phase 0c route (`achievements.md`).
- **Papež, Grigori & Stompor 2018** — messenger as CG preconditioner; the named fallback family
  if the HMC pivot ever needs revisiting. *ID not pinned — check before citing.*
- **Huffenberger & Næss 2018** — messenger-preconditioned CG for map-making; same family. *ID not
  pinned — check before citing.*
- **ducc0 (Reinecke)** — the matrix-free SHT backend. **Reinecke, Belkner & Carron 2023**,
  arXiv:2304.10431, A&A 678, A165 — *Improved CMB (de-)lensing using general spherical harmonic
  transforms*: the accuracy/performance standard for the curved-sky lensing operator and its
  adjoint (NUFFT-based, `lenspyx` + `ducc`). **The correct citation for DiffCMB's lensing
  operator**, alongside the plain ducc0 reference.
- **Belkner et al. 2024 — cunuSHT**, arXiv:2406.14542, RASTI 3, 711 — GPU non-uniform SHTs on
  arbitrary pixelisations, machine precision, up to 5× the fastest CPU algorithm at ℓ_max > 4000.
  The GPU route if Phase 4 (lmax ≥ 1000) is ever unparked — and the reason not to hand-roll one.
- **s2fft (Price & McEwen)** — JAX/PyTorch differentiable spherical harmonic and Wigner
  transforms, precompute and on-the-fly modes, hybrid auto/manual differentiation; the custom-vjp
  pattern DiffCMB's `tf.custom_gradient` wrapper was cribbed from. *arXiv ID not verified —
  check before citing; the JCP 2024 paper is "Differentiable and accelerated spherical harmonic
  and Wigner transforms."*
- **cuHPX**, arXiv:2510.01785 — GPU-accelerated differentiable SHTs specifically on **HEALPix**
  grids. Newer and more directly matched to DiffCMB's pixelisation than s2fft. *Abstract not
  fetched this pass — verify before citing.*
- **jax-cosmo**, arXiv:2302.05163, OJAp 6, 15 (2023) — the end-to-end differentiable cosmology
  library; the canonical "why differentiability" citation, and what Flinch is built on top of.
  Cite in the introduction's framing of the differentiable-cosmology programme.

---

## Science targets and real-data framing

- **LiteBIRD lensing forecast**, arXiv:2507.22618 (2025) — Planck+LiteBIRD full-sky QE
  reconstruction. The target experiment's pipeline is still QE/iterative — the gap Phase 3's
  sampling-based delensing fills. Benchmark and motivation citation.
- **LiteBIRD multitracer delensing**, arXiv:2312.05194, JCAP 06 (2024) 010 (Namikawa, Lonappan et
  al.) — external tracers (CIB, S4, Euclid/LSST) improve σ(r) by ~20%. The competing route to the
  same goal; Phase 3 must say why *internal* sampling-based delensing is complementary.
- **CMB-S4 iterative internal delensing**, arXiv:2310.06729 — the 92–93% B-lensing-removal
  benchmark any Phase 3 delensing-efficiency number gets compared to.
- **Planck 2018 lensing & likelihood papers** — the A_L anomaly's origin.
- **Planck PR4/NPIPE likelihoods (CamSpec, HiLLiPoP)** — report A_L much weakened / ΛCDM-
  consistent.
- **ACT DR6 lensing** (Madhavacheril et al. 2024) + **ACT DR6 extended models**,
  arXiv:2503.14454 — no lensing-amplitude excess; corroborates the resolution.
- **"Revisiting A_L in Planck 2018 temperature"**, arXiv:2310.03127 — the anomaly's anatomy;
  basis for the post-mortem framing.
- **Robustness of Bayesian lensing to foregrounds**: arXiv:2406.15351 (polarized extragalactic
  foregrounds on Bayesian CMB lensing, PRD 111, 023503) and arXiv:2502.20801 (non-Gaussian
  foreground bias in optimal reconstruction of lensing *and* temperature spectra). **The second
  is the closer analogue** — it asks what foregrounds do to a *joint* lensing+C_ℓ reconstruction,
  which is DiffCMB's object. Both are the honest "limitations" citations for the real-data run.

**The A_L anomaly is no longer a live hook.** Planck PR4/NPIPE (especially HiLLiPoP) report it
much weakened; ACT DR6 lensing shows no excess. The honest real-data pitch is the
post-mortem/internal-consistency framing in `ROADMAP.md` §2, not "interrogate a live anomaly."

---

## Standing claims-hygiene rule

Before every submission milestone, re-scan arXiv for **three** things — the third is new:

1. curved-sky MUSE / curved-sky field-level lensing samplers;
2. the generative/diffusion route for **CMB** (not just galaxy) joint lensing posteriors;
3. **any φ / lensing extension of Flinch, Almanac, or CMBLensing.jl.** The Flinch author list
   (Millea, Seljak, Bayer, Loureiro) is the highest-probability source of a scoop that now
   exists. Watch those authors by name, not only by keyword.

Search the *problem* (joint CMB lensing posteriors), not only the *method family* this project
happens to use — item (2) was missed for over a year of scans that only looked for samplers and
MUSE variants, and item (3) was missed until 2026-08-05 despite Flinch being nine months old.

---

## Open items

- [x] **Cite Flinch (arXiv:2510.26691) and Almanac (arXiv:2305.16134) in related work, with the
      distinction stated explicitly**: both are curved-sky (map, C_ℓ) samplers, neither has a φ or
      C_L^φφ block. **Done 2026-08-24** — `docs/paper/main.tex` carries both as `\bibitem`s and
      states the lensing-blind distinction in three places: the related-work opening, the
      "vs Commander / Almanac / Flinch" positioning entry, and the conclusion. Flinch is also
      cited as the precedent for the differentiable-SHT machinery (framed as enabling machinery,
      not a contribution — see the next item).
- [ ] **Stop leading with the differentiable-SHT engineering.** Flinch makes curved-sky
      differentiability table stakes. Lead with the joint (C_ℓ, C_L^φφ) posterior.
- [ ] **Evaluate MCLMC for the φ block — TRIGGERED 2026-08-07, spike implemented and mid-decision
      2026-08-08.** Job 11694912 (`phi_n_lfs` 80→240) came back NO-GO with near-zero improvement
      despite 3x compute — evidence for a geometry problem, not just under-mixing (contradicting
      the deficit-vs-S/N plot's direction), which triggered the port per `ROADMAP.md`'s decision
      rule. Hand-implemented directly in TF (`diffcmb/mclmc.py`, no JAX/blackjax dependency — see
      `ROADMAP.md` for why), unit-validated, and grid-tuned at small lmax=20: the best config beat
      HMC's ESS/wall-clock-second on 4/6 probed l-bins with a clean bias check, confirming the
      1–3-orders-of-magnitude reports transfer at least partially at this dimensionality. Deciding
      run at the actual lmax≈128 pilot scale/problem bin is in flight — see `ROADMAP.md`'s
      "Currently doing" for live status and the GO/NO-GO procedure.
- [x] **Fix the Carron & Lewis ID** — 1704.08230, not 1701.01712 — done 2026-08-06 at
      `diffcmb/lensing.py:24` (the only live occurrence outside this file). Re-check before it
      reaches a `.bib` or a draft.
- [x] **Fix the 2209.10512 description** — checked 2026-08-06, no occurrences outside this file;
      already correctly described here as geometry-agnostic MUSE methodology, not a flat-sky
      MUSE follow-up.
- [ ] **Read Modrák et al. (arXiv:2211.02383) before finalising the coverage-test design** — SBC
      sensitivity depends on the test quantity, and per-ℓ-bin φ-power is a non-innocent choice.
- [ ] **Use rank-normalised split-R̂ and rank plots** (arXiv:1903.08008), not Gelman–Rubin 1992
      and trace plots, in the convergence section.
- [ ] **Cite the corrected form of Cook, Gelman & Rubin 2006** (JCGS correction, 2017), not the
      original quantile statement.
- [x] **Produce the per-ℓ-bin φ-deficit-vs-S/N plot** — done 2026-08-06
      (`scripts/analyze_phi_deficit_vs_snr.py`, `results/analysis/phi_deficit_vs_snr_lmax300.png`).
      Result: deficit decreases with S/N (Spearman ρ=-0.78) — the Wiener-suppressed-start/
      under-mixing hypothesis, not high-S/N sampler geometry. See `ROADMAP.md`'s "Currently doing"
      for the full write-up and its bearing on the sampler-lever decision.
- [ ] **Add the three-sentence "what the deficit is not" paragraph** (not N0/N1, not mean-field,
      not non-Gaussian deflection, not foregrounds) to pre-empt referee questions.
- [ ] **Cite Doeser & Jasche (arXiv:2606.10023) in the introduction** as the external, independent
      statement of why an exact reference posterior is needed at all.
- [ ] Cite **Reinecke, Belkner & Carron (arXiv:2304.10431)** for the lensing operator, and
      **Racine et al. (arXiv:1512.06619)** / **arXiv:2111.07664** for the conjugate-block mixing
      lineage.
- [ ] Verify before citing: s2fft arXiv ID; cuHPX (arXiv:2510.01785) abstract; author lists for
      arXiv:1708.06753, arXiv:2111.07664, arXiv:0708.2989; Papež et al. 2018 and Huffenberger &
      Næss 2018 IDs; the Eriksen/Jewell/Wandelt 2004 Commander trio IDs.
