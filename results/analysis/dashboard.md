# Sampling & Validation Dashboard
*Last updated: 2026-08-31*

Live status of the production chains. Forward plan: `ROADMAP.md`. Closed-out
results and the bug record: `achievements.md`.

---

## Current headline — simulation-based calibration

**lmax=64, nside=64, 12 independent chains, `phi_mass_matrix='prior'`, Block 4 OFF**
(job 11900600). Block 4 off pins `C_L^φφ` at the fiducial spectrum, so the φ
prior is proper *and identical to the process that generated the truth* — which
is what makes the φ rank a genuine calibration test.

| Field rank (pooled over 4 ℓ-bins, N=48) | mean_u | KS_p | verdict |
|---|---|---|---|
| φ | **0.4534** | **0.165** | consistent with uniform |
| alm | **0.5367** | **0.326** | consistent with uniform |

Reference: the same configuration with Block 4 **on** (flat improper prior on
`C_L^φφ`, job 11899585) gives φ mean_u = 0.367, KS_p = 0.0040 — a statement
about the prior, not about the sampler (see below).

Mixing, same runs: τ_int median 6–25 per bin with Block 4 off, vs 48–62 with it
on. R̂ ≤ 1.06 outside the lowest bin.

---

## ⚠ How to read the spectrum rows

`aggregate_coverage_ranks.py` FLAGs the `C_l^TT` and `C_L^φφ` rows in every run.
**Those flags are not evidence of bias.** The statistic ranks the truth's
realized power `S_L/(2L+1)` against posterior draws — and that is exactly the
*mode* of the inverse-Gamma conditional. An inverse-Gamma is right-skewed, so
`P(draw < mode) < 0.5` for a *correct* sampler, and bin-averaging shrinks the
spread while preserving the offset, driving the mean rank toward zero.

Always compare against the simulated null:

```bash
PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
    --indir results/analysis/<ensemble dir> --thin <matching thin>
```

Measured for job 11899585 — observed vs null, all inside the 95% band:

| bin | `C_l^TT` obs | null | `C_L^φφ` obs | null (φ-trajectory) |
|---|---|---|---|---|
| [2,10) | 0.094 | 0.095 | 0.281 | 0.268 |
| [10,30) | 0.104 | 0.095 | 0.323 | 0.276 |
| [30,60) | 0.094 | 0.115 | 0.323 | 0.335 |
| [60,64) | 0.375 | 0.345 | 0.229 | 0.240 |

The `C_L^φφ` null must retain the chain's own sweep-to-sweep φ scatter; a null
that freezes φ at truth is far too narrow and makes a correct sampler look
biased. `C_l^TT` needs no such correction because alm is pinned at cosine 0.9998.

---

## In flight

| Job | Configuration | Purpose |
|---|---|---|
| 11903181 | Block 4 OFF, corrected InvGamma shape | Should reproduce φ ≈ 0.453 / alm ≈ 0.537 |
| 11903182 | Proper prior ν=6, corrected shape | **Decisive test of the shape fix** — its strict `C_L^φφ` SBC rank was 0.25–0.28 (KS_p=0.0000) before the fix and must come back uniform |

Harvest checklist: `.err` for tracebacks (SLURM `COMPLETED` is not sufficient),
per-realization φ/truth power ratio O(1), re-derive `--thin` from each run's own
τ_int. In the flat-prior run, watch ℓ=2: the corrected shape is heavier-tailed
at low ℓ (α = 1 at ℓ=2, conditional mean undefined), and instability there is
the improper prior biting rather than a regression.

---

## Invalid output — do not aggregate or cite

| Directory | Job | Why |
|---|---|---|
| `coverage_ensemble_lmax64/` | 11848757 | Pre-dates the 2026-08-24 alm ordering fix |
| `coverage_ensemble_lmax64_prior_cl4/` | 11887897 | Cold-start alm; φ frozen 1e3–1e5× above truth |

Both `..._prior_cl4_mapfix/` (11899585) and `..._prior_nocl4/` (11900600) are
valid but were produced with the pre-2026-08-31 inverse-Gamma shape; their
field-rank conclusions stand, their spectrum values shift slightly at low ℓ.

---

## Historical — pre-pivot dense-SHT era (2026-06-27)

Kept because the float32 result is the project's standing counterexample for the
fp64 discipline rule, not because these runs are current.

| Run | Precision | Accept | R-hat C_l (med/max) | ESS C_l | Status |
|---|---|---|---|---|---|
| lmax300 Gibbs | **float64** | 71% | 1.026 / 1.085 | 385 | C_l converged |
| lmax300 Gibbs | float32 | 38% | 1.000 / 1.001 | 1553 | alm **frozen** |
| lmax200 Gibbs | float32 | 64% | 1.000 / 1.001 | 1551 | alm **frozen** |
| lmax200 HMC | float32 | 64% | 2985 / — | 5 | diverged |
| lmax64 NUTS | float32 | 100% | 1.180 / — | 12.5 | underlength |

**The float32 trap, worth re-reading before any mixed-precision proposal:**
gradient noise in the SHT matmul accumulated across ~607k unmasked pixels and
drove the HMC step size to ~1e-7. Acceptance looked healthy at 38–65%, but the
chains moved <2e-6 in whitened alm space per step — effectively frozen. `C_l`
R̂ then read a *perfect* 1.000, because each chain sat in its own frozen alm
realisation and converged rapidly within that stuck mode. R̂ on one block is not
convergence.
