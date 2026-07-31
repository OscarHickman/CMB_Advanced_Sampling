"""
ROADMAP.md Section 1, Priority 1: one realization of the multi-realization
rank/coverage ensemble. Run as a SLURM array task (one task per realization);
`scripts/aggregate_coverage_ranks.py` pools the outputs into the uniformity
assessment.

Each task draws its OWN independent (alm_true, phi_true, noise) triple from the
fiducial spectra, seeded off `--realization`, runs a full 4-block chain
warm-started away from truth, and saves the posterior samples plus the truth.
Deliberately computes no verdict: all rank statistics and all uniformity
testing live in the aggregation step, so the per-bin statistic can be changed
without re-running 10-20 chains.

Prerequisite: scripts/pilot_coverage_equilibration.py must have returned GO at
this (lmax, phi_n_lfs, n_samples) configuration. A rank test on chains that
have not equilibrated produces confidently wrong uniformity plots -- see that
script's header and achievements.md.

!! READ BEFORE INTERPRETING THE OUTPUT !!
The strict simulation-based-calibration recipe is: draw theta_true from the
prior, simulate data, then check that the rank of theta_true in the posterior
is uniform. That recipe is *not* fully available for the spectrum blocks here,
and the reason is structural rather than an implementation shortcut:

  Block 1's exact conditional is C_l|alm ~ InvGamma(l-0.5, S_l/2). Matching
  that against the C_l likelihood, which goes as C_l^{-(2l+1)/2} exp(-S_l/2C_l)
  = C_l^{-l-0.5} exp(-S_l/2C_l), shows the InvGamma(l-0.5, S_l/2) density
  C_l^{-l-0.5} exp(-S_l/2C_l) *is* the likelihood -- i.e. the implied prior on
  C_l is flat and improper. Block 4 has the same structure for C_L^phiphi.
  An improper prior cannot be drawn from, so there is no way to generate
  C_l_true ~ p(C_l) and the spectra cannot carry a strict SBC rank.

What this script therefore generates is the *conditional* ensemble: spectra
held at the fiducial values, with alm_true ~ N(0, C_l^fid) and
phi_true ~ N(0, C_L^phiphi,fid) drawn from their exactly-known Gaussian priors.
The field-level ranks (alm, phi) are the statistic this design supports most
directly. The spectrum-level ranks are still computed by the aggregator, but
they are an interval-coverage check against the realized power, not a strict
SBC rank, and must be labelled as such in any figure or claim. See the
aggregator's header for exactly what each reported number does and does not
establish.

Usage (single task):
  PYTHONPATH=diffcmb .venv/bin/python scripts/coverage_ensemble_chain.py \
      --realization 0 --lmax 128 --nside 128 \
      --n_burnin 100 --n_samples 600 --phi_n_lfs 80
"""
import argparse
import os
import time

import healpy as hp
import numpy as np
import tensorflow as tf

from diffcmb import CosmologyAdvancedSampling, run_gibbs_chain
from diffcmb.lensing import _alm_hp_to_packed, lens_map_tf
from diffcmb.power import call_CAMB_map

LCDM_PARAMS = [67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]

# Seed-stream offsets. Each realization gets a disjoint block so that the truth,
# the (independent) chain start, and the noise never share a stream -- a shared
# stream would correlate the start with the truth and quietly bias the ranks
# toward the centre.
_STREAM_TRUTH = 1_000_000
_STREAM_START = 2_000_000
_STREAM_NOISE = 3_000_000


def get_cl_phiphi(lmax):
    import camb

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=LCDM_PARAMS[0], ombh2=LCDM_PARAMS[1], omch2=LCDM_PARAMS[2],
        mnu=LCDM_PARAMS[3], omk=LCDM_PARAMS[4], tau=LCDM_PARAMS[5],
    )
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    dl_pp = results.get_lens_potential_cls(lmax=lmax - 1)[:, 0]
    cl_pp = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        if ell < len(dl_pp):
            norm = (ell * (ell + 1)) ** 2 / (2.0 * np.pi)
            cl_pp[ell] = dl_pp[ell] / norm
    return cl_pp


def _synalm_pair(cl_tt, cl_pp, lmax, seed):
    """Draw (alm, phi) from their Gaussian priors under a named seed.

    hp.synalm consumes numpy's global RNG state, so the stream is selected by
    seeding it explicitly rather than by passing a Generator.
    """
    np.random.seed(seed)
    alm_hp = hp.synalm(cl_tt, lmax=lmax - 1, new=True).astype(np.complex128)
    phi_hp = hp.synalm(cl_pp, lmax=lmax - 1, new=True).astype(np.complex128)
    return alm_hp, phi_hp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--realization", type=int, required=True,
                   help="realization index; sets all three seed streams")
    p.add_argument("--lmax", type=int, default=128)
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--noisesig", type=float, default=1.0)
    p.add_argument("--n_burnin", type=int, default=100)
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--hmc_step_size", type=float, default=0.01)
    p.add_argument("--n_lfs", type=int, default=10)
    p.add_argument("--phi_hmc_step_size", type=float, default=0.001)
    p.add_argument("--phi_n_lfs", type=int, default=80)
    p.add_argument("--outdir", type=str,
                   default="results/analysis/coverage_ensemble")
    p.add_argument("--checkpoint_every", type=int, default=50)
    args = p.parse_args()

    lmax, nside, r = args.lmax, args.nside, args.realization
    os.makedirs(args.outdir, exist_ok=True)
    ckpt = os.path.join(args.outdir, f"chain_r{r:03d}_ckpt.npz")
    out = os.path.join(args.outdir, f"chain_r{r:03d}.npz")

    print(f"=== Coverage ensemble, realization {r} (lmax={lmax}, nside={nside}) ===")
    print("Spectra held at fiducial; alm_true/phi_true drawn from their Gaussian "
          "priors.\nSpectrum-level ranks are interval coverage, NOT strict SBC "
          "(improper flat\nimplied prior -- see this script's header).\n")

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=args.noisesig,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    assert len(model.unmasked_idx) == model.NPIX, "ensemble assumes full-sky data"

    cl_true = call_CAMB_map(LCDM_PARAMS, lmax)
    cl_phiphi_true = get_cl_phiphi(lmax)

    alm_true_hp, phi_true_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_TRUTH + r
    )
    alm_true_packed = _alm_hp_to_packed(alm_true_hp, lmax)
    phi_true_packed = _alm_hp_to_packed(phi_true_hp, lmax)

    T_lensed_true = lens_map_tf(
        model, tf.constant(alm_true_packed, tf.float64), phi_true_hp
    ).numpy()
    if not np.all(np.isfinite(T_lensed_true)):
        raise RuntimeError(f"realization {r}: lens_map_tf produced NaN/Inf")

    rng_noise = np.random.default_rng(_STREAM_NOISE + r)
    noisy_map = T_lensed_true + rng_noise.normal(0.0, args.noisesig, size=model.NPIX)
    model.prior_map = noisy_map
    model.prior_map_masked = tf.convert_to_tensor(
        noisy_map[model.unmasked_idx], dtype=tf.float64
    )

    # Warm start from an independent prior draw, not a perturbation of truth.
    alm_start_hp, phi_start_hp = _synalm_pair(
        cl_true, cl_phiphi_true, lmax, _STREAM_START + r
    )
    alm_start_packed = _alm_hp_to_packed(alm_start_hp, lmax)
    phi_start_packed = _alm_hp_to_packed(phi_start_hp, lmax)
    start_corr = float(
        np.dot(phi_start_packed, phi_true_packed)
        / (np.linalg.norm(phi_start_packed) * np.linalg.norm(phi_true_packed))
    )
    print(f"  start-vs-truth phi cosine similarity={start_corr:+.4f} (~0 expected)")

    x0 = np.concatenate([np.log(cl_true[2:lmax]), alm_start_packed])

    t0 = time.time()
    samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = run_gibbs_chain(
        model,
        n_samples=args.n_samples,
        n_burnin=args.n_burnin,
        hmc_step_size=args.hmc_step_size,
        n_lfs=args.n_lfs,
        initial_params=x0,
        cl_phiphi_full=cl_phiphi_true,
        phi_initial=phi_start_packed,
        phi_hmc_step_size=args.phi_hmc_step_size,
        phi_n_lfs=args.phi_n_lfs,
        phi_mass_matrix='prior',
        sample_cl_phiphi=True,
        seed=r,
        checkpoint_path=ckpt,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed / 3600:.2f}h; alm accept={accepts.mean():.3f}")

    if not (np.all(np.isfinite(samples)) and np.all(np.isfinite(phi_samples))
            and np.all(np.isfinite(cl_phiphi_samples))):
        raise RuntimeError(f"realization {r}: NaN/Inf in samples")

    np.savez(
        out,
        realization=r, lmax=lmax, nside=nside,
        alm_samples=samples, phi_samples=phi_samples,
        cl_phiphi_samples=cl_phiphi_samples,
        logp=logp, accepts=accepts,
        cl_true=cl_true, cl_phiphi_true=cl_phiphi_true,
        alm_true_packed=alm_true_packed, phi_true_packed=phi_true_packed,
        start_cosine_similarity=start_corr,
        n_burnin=args.n_burnin, phi_n_lfs=args.phi_n_lfs,
        seconds_total=elapsed, seconds_per_sweep=elapsed / max(1, len(samples)),
    )
    print(f"Saved realization {r} to {out}")


if __name__ == "__main__":
    main()
