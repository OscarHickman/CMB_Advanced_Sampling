# Literature review — DiffCMB, 2026-07-13

*Focus: last ~12 months. Verdict up front: **the novelty window is still open — no curved-sky MUSE and no curved-sky joint (alm, C_ℓ, φ) sampler has appeared** — but the A_L science hook has substantially weakened: the anomaly is largely resolved in Planck PR4/NPIPE likelihoods and ACT DR6 sees no excess. Reframe the real-data hook; lean harder on LiteBIRD delensing.*

## The claim check — still first, still a race

- **No curved-sky MUSE**: nothing published extends MUSE (Millea & Seljak, arXiv:2112.09354) beyond the flat sky; the SPT-3G production analyses (e.g. arXiv:2411.06000) remain flat-patch MUSE.
- **No curved-sky joint sampler**: recent lensing-reconstruction work is iterative/QE-class (e.g. non-Gaussian-deflection iterative reconstruction, arXiv:2407.00228; noise-bias-minimising iterative estimators, arXiv:2506.20667). Millea, Anderes & Wandelt 2020 (arXiv:2002.00965) remains the nearest prior work, flat-sky only — their own "conceptually straightforward, computationally the challenge" framing on curvature still stands unclaimed by anyone.
- **LiteBIRD's lensing pipeline is still QE/iterative** (arXiv:2507.22618, July 2025: Planck+LiteBIRD full-sky reconstruction forecast, 72–78σ) — i.e. the target experiment's own forecast literature confirms no sampling-based full-sky machinery exists. That paper is also the natural benchmark/motivation citation for Phase 3.
- **Conclusion: the "first full-sky curved-sky differentiable joint Gibbs sampler" claim survives as scoped.** Keep the standing rule: re-scan arXiv for curved-sky MUSE before every submission milestone — the window remains finite.

## ⚠ The A_L hook — materially weakened, reframe it

Status as of mid-2026: **Planck PR4/NPIPE likelihoods (CamSpec, and especially HiLLiPoP) report a much weaker A_L anomaly, consistent with ΛCDM; ACT DR6 lensing shows no amplitude excess and agrees with Planck lensing.** The community reading is that the 2018-era anomaly was substantially a data-processing artefact.

- **Consequence:** "interrogate a live anomaly" is no longer an honest pitch. Two honest replacements:
  1. **Post-mortem with proper joint uncertainties**: a joint (alm, C_ℓ, φ) posterior run on Planck 2018 vs PR4 maps can show *where in the joint parameter space* the anomaly lived and how it dissolves — the first fully-propagated-uncertainty account of a famous systematic's resolution. Smaller headline, but methodologically clean and referee-safe.
  2. **Internal-consistency machinery as the product**: the same joint posterior is the principled internal lensing-consistency test for *future* data (SO, LiteBIRD) — sell the capability, demonstrated on Planck, rather than the anomaly.
- Either way the beam/pixel-window/anisotropic-noise realism pre-condition stands unchanged.

## Phase-3 context (strengthening)

- LiteBIRD lensing/delensing forecast literature is growing (arXiv:2507.22618 and successors) and remains QE-based — the sampling-based-delensing gap Phase 3 targets is intact.

## Action items

- [ ] Reframe the A_L item in `ROADMAP.md` (done this pass): post-mortem/consistency-machinery framing, not live-anomaly interrogation.
- [ ] Cite arXiv:2507.22618 as the Phase-3 target-experiment reference and the "no full-sky sampler exists" evidence.
- [ ] Standing rule unchanged: arXiv re-scan for curved-sky MUSE/field-level lensing samplers before each milestone.
