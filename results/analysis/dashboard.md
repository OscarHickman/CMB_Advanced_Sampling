# Sampling & Validation Dashboard
*Last updated: 2026-08-31*

Live status of the production chains. Forward plan: `ROADMAP.md`. Closed-out
results and the bug record: `achievements.md`.

---

## Current headline — simulation-based calibration

**lmax=64, nside=64, 12 independent chains, `phi_mass_matrix='prior'`, Block 4 OFF**
(job **11903181**, corrected inverse-Gamma shape; supersedes job 11900600).
Block 4 off pins `C_L^φφ` at the fiducial spectrum, so the φ prior is proper
*and identical to the process that generated the truth* — which is what makes
the φ rank a genuine calibration test.

| Field rank (pooled over 4 ℓ-bins, N=48) | mean_u | KS_p | verdict |
|---|---|---|---|
| φ | **0.4688** | **0.124** | consistent with uniform |
| alm | **0.5312** | **0.235** | consistent with uniform |

No flagged bin in either field row (the pre-fix run had one, φ `[30,60)`).
Thin-robust: φ 0.475 (p=0.44) at `--thin 30`, 0.453 (p=0.24) at 45, 0.469
(p=0.124) at 90; τ_int max 42.5, so 90 is conservative. The pre-fix pair was
0.4534 / 0.5367 — the corrected shape moved both *closer* to 0.5.

Reference: the same configuration with Block 4 **on** (flat improper prior on
`C_L^φφ`, job 11899585) gives φ mean_u = 0.367, KS_p = 0.0040 — a statement
about the prior, not about the sampler (see below).

Mixing, same runs: τ_int median 4.7–27 per bin with Block 4 off, vs 24–56 with
it on. R̂ ≤ 1.07 outside the lowest and highest bins.

---

## ⚠ Open: the proper-prior configuration is NOT yet SBC-validated

**Job 11903182** (Block 4 ON, proper conjugate prior ν=6, corrected shape) is
the source for the paper's joint `(C_ℓ, C_L^φφ)` differentiator figure, and its
strict `C_L^φφ` SBC rank does **not** clear:

| Statistic | Result | Verdict |
|---|---|---|
| alm field rank | 0.5052 (KS_p 0.408) | clean |
| Block 4 PIT given φ | 0.4999 (KS_p 0.42) | ⚠ **vacuous** — lag-1 control also passes (0.183) |
| φ field rank | 0.4115 (KS_p 0.0264) | low, worst bin `[10,30)` = 0.292 |
| **strict `C_L^φφ` SBC rank** | **0.3802 (KS_p 0.0013)** | **improved from 0.25–0.28 pre-fix, still not uniform** |

The φ and `C_L^φφ` deficits are the same deficit: Block 4 is exact given φ, so
if `C_L^φφ` ranks low its conditioning `S_L(φ)` must be high — measured φ
power/truth median 1.08–1.14, concentrated in the same `[10,30)` bin. Not
burn-in (0.400 on the 2nd half, 0.396 on the last quarter; φ power *rises*
1.002 → 1.060 across the chain). Leading hypothesis is Block 3 mixing under the
Block-4-ON funnel (τ_int max 92.7 vs 42.5, split-R̂ max 1.64 vs 1.31), not a
wrong conditional. Any figure sourced from this job must carry the caveat.

---

## ⚠ How to read the spectrum rows

`aggregate_coverage_ranks.py` FLAGs the `C_l^TT` and `C_L^φφ` rows in every run.
**Those flags are not evidence of bias.** The statistic ranks the truth's
realized power `S_L/k_L` (`k_L = 2L`, the packed dof) against posterior draws — and that is exactly the
*mode* of the inverse-Gamma conditional. An inverse-Gamma is right-skewed, so
`P(draw < mode) < 0.5` for a *correct* sampler, and bin-averaging shrinks the
spread while preserving the offset, driving the mean rank toward zero.

Always compare against the simulated null:

```bash
PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
    --indir results/analysis/<ensemble dir> --thin <matching thin>
```

Measured under the **corrected shape** — observed vs null, all inside the 95%
band (`C_l^TT` from job 11903181, `C_L^φφ` from job 11903182):

| bin | `C_l^TT` obs | null | `C_L^φφ` obs | null (φ-trajectory) |
|---|---|---|---|---|
| [2,10) | 0.083 | 0.086 | 0.260 | 0.279 |
| [10,30) | 0.083 | 0.092 | 0.302 | 0.339 |
| [30,60) | 0.094 | 0.114 | 0.427 | 0.457 |
| [60,64) | 0.333 | 0.344 | 0.406 | 0.376 |

Note this row is *interval coverage*, distinct from the **strict** `C_L^φφ` SBC
rank in the open-issue section above — that one ranks `cl_phiphi_true` (drawn
from the sampler's own prior) among the Block 4 samples, is uniform under a
correct sampler with no null needed, and is the statistic that does not clear.

The `C_L^φφ` null must retain the chain's own sweep-to-sweep φ scatter; a null
that freezes φ at truth is far too narrow and makes a correct sampler look
biased. `C_l^TT` needs no such correction because alm is pinned at cosine 0.9998.

---

## In flight

**Nothing in flight** as of 2026-08-31. Jobs 11903181 and 11903182 are both
harvested (12/12 COMPLETED each, no tracebacks, φ/truth power ratios 0.77–1.39
and 0.71–1.60).

Standing harvest checklist for the next ensemble: `.err` for tracebacks (SLURM
`COMPLETED` is not sufficient), per-realization φ/truth power ratio O(1),
re-derive `--thin` from each run's own τ_int, and **check that the Block 4 PIT's
lag-1 control still fails** before quoting the aligned pass.

---

## Invalid output — do not aggregate or cite

| Directory | Job | Why |
|---|---|---|
| `coverage_ensemble_lmax64/` | 11848757 | Pre-dates the 2026-08-24 alm ordering fix |
| `coverage_ensemble_lmax64_prior_cl4/` | 11887897 | Cold-start alm; φ frozen 1e3–1e5× above truth |

Both `..._prior_cl4_mapfix/` (11899585) and `..._prior_nocl4/` (11900600) are
valid but were produced with the pre-2026-08-31 inverse-Gamma shape; their
field-rank conclusions stand, their spectrum values shift slightly at low ℓ.
**Superseded** by `..._prior_nocl4_doffix/` (11903181) and
`..._prior_cl4_properprior_doffix/` (11903182) — cite those.

⚠ Any aggregation run *before* 2026-08-31's analysis-script fix is also stale:
`aggregate_coverage_ranks.py` and `validate_coverage_rank_nulls.py` both still
hardcoded the `2L+1` dof assumption after the samplers were corrected, so a
harvest from that window compared corrected chains against a stale reference.
Re-run both scripts rather than quoting an older printout.

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
