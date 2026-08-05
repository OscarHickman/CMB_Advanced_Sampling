# Literature — DiffCMB

*Annotated bibliography with the novelty/positioning argument folded in below. Last rescanned: see the standing claims-hygiene rule at the bottom.*

## Positioning: the novelty window and the competing paradigm

**The core claim holds:** no curved-sky MUSE and no curved-sky joint (a_ℓm, C_ℓ, φ) sampler has appeared. The full-sky × joint-sampling cell of the 2×2 below is still empty.

**The most likely referee question is the diffusion/score-based route, and the paper needs an answer on file.** Two papers define the threat:

- **"Denoising diffusion delensing"** (MNRAS 533, 423, 2024; arXiv:2405.05598) — score-based generative reconstruction of the CMB lensing convergence, pitched explicitly as an **alternative to HMC**, claiming uncorrelated samples and better large-angular-scale sampling than MCMC chains.
- **JADE** (arXiv:2606.31988, 2026) — joint posterior over convergence maps *and* cosmological parameters from a single conditional diffusion model; amortised (~0.2s/sample after training), explicitly requiring neither a differentiable forward model nor inference-time MCMC. Galaxy weak lensing, not CMB — not a literal pre-emption, but the clearest statement of the paradigm's ambition to obviate DiffCMB's two core design choices.
- **"Generative Diffusion Priors for 3D Mapping of the Dark Universe"** (arXiv:2606.00803, 2026) — same family, wider scope; evidence the approach is a programme, not a one-off.
- The one CMB-domain diffusion paper found in scanning (arXiv:2512.22683, generative reconstruction of low-ℓ B-modes) treats lensing as a contaminant to denoise away rather than inferring φ — no lensing posterior, joint or otherwise.

**Why this is sharper than a normal related-work gap:** DiffCMB's current bottleneck *is* HMC autocorrelation in the φ block (see `ROADMAP.md`, `achievements.md`). The diffusion line markets the exact cure for the exact symptom, while discarding the two things DiffCMB spent its engineering budget building (differentiability and a sampler). "Why fight HMC autocorrelation when diffusion gives uncorrelated samples in 0.2s?" is the first question a methods referee will ask.

**Three answers, to state in the paper rather than discover at referee time:**
1. **Exactness.** HMC targets the true posterior asymptotically; a diffusion model samples a *learned* approximation whose error has no convergence guarantee and no diagnostic that certifies it. For a measurement intended to feed a tension debate, that difference is the product.
2. **Scope of the posterior.** Both cited works infer a lensing map (JADE adds cosmological parameters). DiffCMB's object is the **joint (a_ℓm, C_ℓ, φ)** posterior — the C_ℓ block, and its correlations with φ, is absent from the diffusion route and is what makes the internal-consistency framing (Section 2 of `ROADMAP.md`) possible at all.
3. **No training set required.** Diffusion priors must be trained on simulations, importing every bias of that suite; the Gibbs/HMC route needs only the forward model.

**Portfolio-level argument — use it.** The reason to prefer an exact sampler over an amortised learned posterior for a tension-grade measurement is precisely the thesis of ANVIL and KARMA: learned posteriors can pass their diagnostics and still be wrong ("calibrated but not accurate"), and in field-level cosmology specifically no known fix repairs it. DiffCMB can cite that trust-verdict work as the methodological justification for its architecture — converts a defensive answer into a coherent programme. Worth a paragraph in the introduction.

**Honest limit:** this framing is only as strong as the scale at which converged reference posteriors are actually produced. If that's lmax≈128 and the diffusion papers operate at lmax≫1000, say so plainly rather than overclaiming (see `ROADMAP.md`'s scope discussion).

**The A_L anomaly is no longer a live hook.** Planck PR4/NPIPE likelihoods (especially HiLLiPoP) report it much weakened/consistent with ΛCDM; ACT DR6 lensing shows no excess. The honest real-data pitch is the post-mortem/internal-consistency framing in `ROADMAP.md` Section 2, not "interrogate a live anomaly."

## The occupied cells of the 2×2 (full-sky × joint-sampling) — the claim's boundaries

- **Millea, Anderes & Wandelt 2020**, arXiv:2002.00965 — flat-sky joint (f, φ, r) Bayesian sampling (CMBLensing.jl); the nearest prior work. Their reparametrisation lesson (naive block alternation mixes catastrophically at high S/N) motivated this project's Phase-2 mixing-risk gate.
- **Millea et al. 2021** — CMBLensing.jl applied to real SPTpol data (flat patches).
- **Millea & Seljak — MUSE**, arXiv:2112.09354 — marginal score expansion; not a sampler, no joint posterior. Their own "curved-sky HMC slightly out of reach" statement defines this project's empty cell. No curved-sky MUSE exists as of the last scan.
- **SPT-3G 2019-2020 MUSE analysis**, arXiv:2411.06000 — production flat-patch MUSE; evidence the field's best still isn't sampling.
- **Commander / Eriksen et al. 2004-line Gibbs sampling** — full-sky (alm, C_ℓ) but lensing-blind and conjugate-only; the other occupied cell.
- **Anderes, Wandelt & Lavaux 2015**, arXiv:1412.4079 — Bayesian CMB lensing inference ancestor.
- **Carron & Lewis 2017**, arXiv:1701.01712 — iterative MAP lensing reconstruction (point estimate, not posterior); also the lensing-operator reference implementation (lenspyx/delensalot).

## Samplers & numerical methods

- **Elsner & Wandelt 2013**, arXiv:1210.4931 — messenger-field method (this project's abandoned Phase 0c route — see `achievements.md`).
- **Papež, Grigori & Stompor 2018** — messenger as CG preconditioner; the named fallback family if the HMC pivot ever needs revisiting.
- **Huffenberger & Næss 2018** — messenger-preconditioned CG for map-making; same fallback family.
- **Neal 2011 / Duane et al. 1987** — HMC; **TFP DualAveragingStepSizeAdaptation** — the step-size scheme `run_chain_hmc` uses.
- **ducc0 (Reinecke)** — the matrix-free SHT backend; **s2fft** (JAX differentiable SHT) — the custom-vjp pattern cribbed from.
- **BeyondPlanck collaboration**, arXiv:2303.04819 — polarization Gibbs blocks (inverse-Wishart TE structure), the reference for Phase 3.

## Science targets

- **LiteBIRD lensing forecast**, arXiv:2507.22618 (2025) — Planck+LiteBIRD full-sky QE reconstruction (72-78σ forecast); the target experiment's pipeline is still QE/iterative, which is the gap Phase 3's sampling-based delensing fills. Also the benchmark/motivation citation.
- **Planck 2018 lensing & likelihood papers** — the A_L anomaly's origin.
- **Planck PR4/NPIPE likelihoods (CamSpec, HiLLiPoP)** — report the A_L anomaly much weakened/ΛCDM-consistent.
- **ACT DR6 lensing** (Madhavacheril et al. 2024) + **ACT DR6 extended models**, arXiv:2503.14454 — no lensing-amplitude excess; corroborates the resolution.
- **Revisiting A_L in Planck 2018 temperature**, arXiv:2310.03127 — the anomaly's anatomy; basis for the post-mortem framing.
- **Iterative/QE advances** (arXiv:2407.00228 non-Gaussian deflections; arXiv:2506.20667 noise-bias-minimising iterative estimator) — the current non-sampling state of the art to compare against on the full sky.

## Standing claims-hygiene rule

Before every submission milestone, re-scan arXiv for **both**: (1) curved-sky MUSE / curved-sky field-level lensing samplers, and (2) the generative/diffusion route for CMB (not just galaxy) joint lensing posteriors. The window has stayed open across every scan so far, but it's finite, and (2) was missed for over a year of scans that only looked for samplers and MUSE variants — search the *problem* (joint CMB lensing posteriors), not only the *method family* this project happens to use.
