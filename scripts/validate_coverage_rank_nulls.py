"""Null calibration for the coverage-ensemble rank statistics (2026-08-31).

WHY THIS EXISTS. `scripts/aggregate_coverage_ranks.py` FLAGs a bin when its
rank distribution is non-uniform. For the *spectrum* rows (C_l^TT, C_L^phiphi)
that flag fires even for a perfect sampler, because the statistic ranks the
truth against its own conditional's MODE:

    Blocks 1 and 4 both draw   C ~ InvGamma(alpha = L - 0.5, beta = S_L / 2)
    whose mode is             beta/(alpha+1) = S_L / (2L+1)
    which is exactly          aggregate_coverage_ranks.realized_spectrum(S)

InvGamma is right-skewed, so P(draw < mode) < 0.5 by construction, and
bin-averaging over l shrinks the spread while leaving the offset -- driving
mean_u toward 0. Reading those flags as sampler bias is a mistake; this script
computes what a correct sampler actually produces so observed can be compared
against the right null.

It runs three checks:

  1. NULL, conditioning variable frozen at truth. Cheap, and adequate for
     C_l^TT because alm is pinned tightly by the data (cosine ~0.9998).
  2. NULL, using the chain's OWN phi trajectory. Required for C_L^phiphi:
     phi moves sweep-to-sweep, adding common-mode spread across l that does
     NOT average down within a bin, so the frozen-phi null is far too narrow
     and makes a correct sampler look biased.
  3. BLOCK 4 EXACTNESS, directly on production chains. Using the saved
     per-sweep (phi_samples, cl_phiphi_samples) pairs,
     u = CDF_InvGamma(L-0.5, S_L(phi_i)/2)[C_{L,i}] must be Uniform(0,1).
     A lag-1 misalignment is run as a control so a pass can't be vacuous.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_mapfix --thin 90
"""
import argparse
import glob
import os

import numpy as np
from aggregate_coverage_ranks import _sl_from_packed, realized_spectrum
from scipy import stats
from scipy.special import gammaincc

from diffcmb.lensing import compute_sl_phi_np

BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]


def load(indir):
    files = sorted(
        f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
        if not f.endswith("_ckpt.npz")
    )
    if not files:
        raise SystemExit(f"no chains found in {indir}")
    return files


def _rank_u(post, tru, n_eff):
    return (int(np.sum(post < tru)) + 0.5) / (n_eff + 1.0)


def null_frozen(files, tag, n_eff, n_rep, rng):
    """Null with the conditioning field frozen at truth."""
    per_real = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S = (_sl_from_packed(d["alm_true_packed"], lmax) if tag == "cl_TT"
             else compute_sl_phi_np(d["phi_true_packed"], lmax))
        per_real.append((S, lmax))

    print(f"\n--- NULL [{tag}]: conditioning field FROZEN at truth ---")
    print(f"  {'bin':12s} {'null_mean':>10s} {'2.5%':>9s} {'97.5%':>9s}")
    for lo, hi in BINS:
        means = np.empty(n_rep)
        for r in range(n_rep):
            us = []
            for S, lmax in per_real:
                ells = np.arange(max(2, lo), min(lmax, hi))
                beta = S[ells] / 2.0
                draws = beta[None, :] / rng.gamma(
                    (ells - 0.5)[None, :], 1.0, size=(n_eff, len(ells))
                )
                us.append(_rank_u(draws.mean(axis=1),
                                  realized_spectrum(S, lmax)[ells].mean(), n_eff))
            means[r] = np.mean(us)
        print(f"  [{lo:3d},{hi:3d})   {means.mean():10.4f} "
              f"{np.percentile(means, 2.5):9.4f} {np.percentile(means, 97.5):9.4f}")


def null_with_phi_trajectory(files, thin, n_rep, rng):
    """Null for C_L^phiphi that keeps the chain's real sweep-to-sweep phi scatter."""
    per_real = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S_chain = np.array([compute_sl_phi_np(p, lmax)
                            for p in d["phi_samples"][::thin]])
        per_real.append((S_chain, compute_sl_phi_np(d["phi_true_packed"], lmax), lmax))
    n_eff = per_real[0][0].shape[0]

    print(f"\n--- NULL [cl_phiphi]: using the chain's OWN phi trajectory "
          f"(n_eff={n_eff}) ---")
    print(f"  {'bin':12s} {'null_mean':>10s} {'2.5%':>9s} {'97.5%':>9s}")
    for lo, hi in BINS:
        means = np.empty(n_rep)
        for r in range(n_rep):
            us = []
            for S_chain, S_true, lmax in per_real:
                ells = np.arange(max(2, lo), min(lmax, hi))
                beta = S_chain[:, ells] / 2.0
                draws = beta / rng.gamma((ells - 0.5)[None, :], 1.0, size=beta.shape)
                us.append(_rank_u(draws.mean(axis=1),
                                  realized_spectrum(S_true, lmax)[ells].mean(), n_eff))
            means[r] = np.mean(us)
        print(f"  [{lo:3d},{hi:3d})   {means.mean():10.4f} "
              f"{np.percentile(means, 2.5):9.4f} {np.percentile(means, 97.5):9.4f}")


def block4_exactness(files, stride):
    """Is Block 4 drawing from InvGamma(L-0.5, S_L(phi)/2) given its own phi?"""
    print("\n--- BLOCK 4 EXACTNESS on production chains ---")
    print("  u = CDF_InvGamma(L-0.5, S_L(phi_i)/2)[C_{L,i}]; uniform if exact.")
    for lag, label in ((0, "aligned"), (1, "lag-1 (control, should FAIL)")):
        allu = []
        for f in files:
            d = np.load(f)
            lmax = int(d["lmax"])
            phi, lnC = d["phi_samples"], d["cl_phiphi_samples"]
            if lag:
                phi, lnC = phi[:-1], lnC[1:]
            ells = np.arange(2, lmax)
            for i in range(0, len(phi), stride):
                S = compute_sl_phi_np(phi[i], lmax)[ells]
                allu.append(gammaincc(ells - 0.5, (S / 2.0) / np.exp(lnC[i])))
        u = np.concatenate(allu)
        ks = stats.kstest(u, "uniform")
        print(f"  {label:30s} N={u.size:6d}  mean_u={u.mean():.4f}  "
              f"KS_D={ks.statistic:.4f}  KS_p={ks.pvalue:.3g}")


def phi_power_bias(files, stride):
    """Per-bin <S_L(phi_chain)> / S_L(phi_true): is phi itself over-powered?"""
    print("\n--- phi power bias, per bin (>1 = posterior phi over-powered) ---")
    tab = {b: [] for b in BINS}
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        S_true = compute_sl_phi_np(d["phi_true_packed"], lmax)
        phi = d["phi_samples"]
        S_chain = np.mean([compute_sl_phi_np(phi[i], lmax)
                           for i in range(0, len(phi), stride)], axis=0)
        for lo, hi in BINS:
            ells = np.arange(lo, min(hi, lmax))
            tab[(lo, hi)].append(S_chain[ells].sum() / S_true[ells].sum())
    print(f"  {'bin':12s} {'median':>8s} {'min':>8s} {'max':>8s}")
    for b in BINS:
        v = np.array(tab[b])
        print(f"  [{b[0]:3d},{b[1]:3d})  {np.median(v):8.3f} {v.min():8.3f} "
              f"{v.max():8.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", required=True)
    p.add_argument("--thin", type=int, default=90,
                   help="must match the --thin passed to aggregate_coverage_ranks.py")
    p.add_argument("--n_rep", type=int, default=400, help="null replications")
    p.add_argument("--stride", type=int, default=25,
                   help="sweep subsampling for the Block 4 / phi-power checks")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = load(args.indir)
    rng = np.random.default_rng(args.seed)
    n_eff = len(np.load(files[0])["phi_samples"][::args.thin])
    print(f"=== Rank-null calibration: {len(files)} realizations, "
          f"thin={args.thin} (n_eff={n_eff}) ===")

    for tag in ("cl_TT", "cl_phiphi"):
        null_frozen(files, tag, n_eff, args.n_rep, rng)
    null_with_phi_trajectory(files, args.thin, args.n_rep, rng)
    block4_exactness(files, args.stride)
    phi_power_bias(files, args.stride)

    print("\nCompare each observed mean_u from aggregate_coverage_ranks.py against\n"
          "the matching null band above -- NOT against 0.5. For C_L^phiphi use the\n"
          "phi-trajectory null; the frozen-phi one is too narrow.")


if __name__ == "__main__":
    main()
