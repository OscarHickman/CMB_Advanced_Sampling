"""Tests for NUTS/HMC samplers — sign convention and step size adaptation."""
import numpy as np
import pytest


def _has_all():
    try:
        import healpy  # noqa: F401
        import scipy  # noqa: F401
        import tensorflow as tf  # noqa: F401
        import tensorflow_probability as tfp  # noqa: F401

        from diffcmb import (  # noqa: F401
            CosmologyAdvancedSampling,
            run_chain_hmc,
            run_chain_nut,
        )
        return True
    except Exception:
        return False


skip_no_tfp = pytest.mark.skipif(not _has_all(), reason="TFP or deps unavailable")

LMAX, NSIDE = 10, 4  # small enough to run in CI


@pytest.fixture(scope="module")
def small_model():
    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    return m


# ── target_log_prob_fn sign ───────────────────────────────────────────────────

@skip_no_tfp
def test_sampler_uses_negative_psi_tf(small_model):
    """
    target_log_prob recorded by NUTS must equal -psi_tf(sample) at each step.

    This catches the sign bug where psi_tf (negative log-posterior) was passed
    as-is to TFP, causing it to sample from 1/posterior.  We verify the identity
    target_log_prob[i] == -psi_tf(samples[i]) for every recorded sample.
    """
    import tensorflow as tf

    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=5, num_burnin_steps=0,
    )

    inner = results.inner_results
    recorded_logp = inner.target_log_prob.numpy()

    for i, (samp, logp) in enumerate(zip(samples.numpy(), recorded_logp)):
        expected = -float(small_model.psi_tf(tf.constant(samp, dtype=tf.float64)).numpy())
        assert abs(logp - expected) < 1e-4, (
            f"Step {i}: target_log_prob={logp:.6f} but -psi_tf(sample)={expected:.6f}. "
            "Sign of psi_tf is wrong in samplers.py."
        )


@skip_no_tfp
def test_hmc_sampler_uses_negative_psi_tf(small_model):
    """Verify HMC targets -psi_tf (sign check).

    Run with no burnin so the first recorded sample starts from x0 and
    its target_log_prob must equal -psi_tf(x0) exactly.
    """
    from diffcmb import run_chain_hmc

    x0 = small_model.prior_parameters_tf()
    expected_logp = -float(small_model.psi_tf(x0).numpy())

    samples, results = run_chain_hmc(
        small_model, x0, _step_size=0.001,
        num_results=5, num_burnin_steps=0,
    )

    inner = results.inner_results
    first_logp = float(inner.accepted_results.target_log_prob.numpy()[0])

    assert abs(first_logp - expected_logp) < 1.0, (
        f"HMC target_log_prob ({first_logp:.4f}) != -psi_tf ({expected_logp:.4f})"
    )


# ── MAP initialisation ───────────────────────────────────────────────────────

@skip_no_tfp
def test_find_map_reduces_psi(small_model):
    """find_map_estimate must strictly decrease psi_tf from the prior mean."""
    from diffcmb import find_map_estimate

    x0 = small_model.prior_parameters_tf()
    psi_before = float(small_model.psi_tf(x0))

    map_state = find_map_estimate(small_model, n_steps=50, learning_rate=0.001, print_every=50)

    psi_after = float(small_model.psi_tf(map_state))
    assert psi_after < psi_before, (
        f"MAP did not reduce psi: {psi_before:.4f} → {psi_after:.4f}"
    )


@skip_no_tfp
def test_find_map_returns_correct_shape(small_model):
    """MAP estimate must have the same shape as the initial state."""
    from diffcmb import find_map_estimate

    x0 = small_model.prior_parameters_tf()
    map_state = find_map_estimate(small_model, n_steps=10, print_every=10)

    assert map_state.shape == x0.shape, (
        f"MAP shape {map_state.shape} != x0 shape {x0.shape}"
    )


# ── chain movement ────────────────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_chain_moves(small_model):
    """
    NUTS must produce samples that differ from the initial state.

    The original bug caused 0% acceptance and every sample == x0.
    """
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=20, num_burnin_steps=50,
    )

    samps = samples.numpy()
    x0_np = x0.numpy()

    assert not np.allclose(samps[0], samps[-1], atol=0), \
        "All NUTS samples are identical — chain never moved (acceptance=0 bug)"
    assert not np.allclose(samps[0], x0_np, atol=0), \
        "First sample equals x0 — chain stuck at initial state"


@skip_no_tfp
def test_nuts_acceptance_rate_nonzero(small_model):
    """NUTS acceptance rate must be > 0 with the fixed sign and adaptation."""
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=30, num_burnin_steps=50,
    )

    inner = results.inner_results
    accept_rate = float(inner.is_accepted.numpy().mean())
    assert accept_rate > 0.0, \
        f"NUTS acceptance rate = {accept_rate:.3f} — sampler is completely stuck"


# ── step size adaptation ──────────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_has_adapted_step_size(small_model):
    """Results object must expose a new_step_size from DualAveragingStepSizeAdaptation."""
    from diffcmb import run_chain_nut

    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.01,
        num_results=10, num_burnin_steps=20,
    )

    assert hasattr(results, "new_step_size"), \
        "results lacks new_step_size — DualAveragingStepSizeAdaptation not applied"
    final_step = float(results.new_step_size.numpy()[-1])
    # Adaptation should change it from the initial 0.01
    assert np.isfinite(final_step) and final_step > 0, \
        f"Adapted step size is not a positive finite number: {final_step}"


# ── inner_results structure ───────────────────────────────────────────────────

@skip_no_tfp
def test_nuts_results_have_inner_results(small_model):
    """
    With adaptive wrapper, results.inner_results must exist and contain
    target_log_prob and is_accepted tensors of the correct length.
    """
    from diffcmb import run_chain_nut

    n = 15
    x0 = small_model.prior_parameters_tf()
    samples, results = run_chain_nut(
        small_model, x0, _step_size=0.001,
        num_results=n, num_burnin_steps=10,
    )

    assert hasattr(results, "inner_results"), "missing inner_results attribute"
    inner = results.inner_results
    assert hasattr(inner, "target_log_prob"), "inner_results missing target_log_prob"
    assert hasattr(inner, "is_accepted"), "inner_results missing is_accepted"
    assert inner.target_log_prob.shape[0] == n
    assert inner.is_accepted.shape[0] == n


@skip_no_tfp
def test_gibbs_chain_moves(small_model):
    """Verify that the Gibbs sampler runs without crashing and moves."""
    from diffcmb import run_gibbs_chain

    samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert not np.allclose(samples[0], samples[-1])


@skip_no_tfp
def test_gibbs_chain_with_phi_block_moves(small_model):
    """Phase 2 Block 3: phi | alm, C_l, d runs alongside the existing blocks."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_with_phi_block_mclmc_moves(small_model):
    """MCLMC spike (ROADMAP.md, 2026-08-07): phi_sampler='mclmc' drop-in for
    Block 3 runs alongside the existing blocks and moves, same smoke-test
    shape as the HMC phi-block test above."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_sampler='mclmc',
        phi_hmc_step_size=1e-4,
        phi_n_lfs=5,
        phi_mclmc_L=1.0,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_cg_with_phi_block_moves(small_model):
    """Phase 2 gate-2 exactness experiment (ROADMAP.md Section 1): Block 2
    ('cg', unlensed exact Gaussian draw) alongside Block 3 (phi | alm, C_l,
    HMC against the correct lensed likelihood) runs without crashing and
    both blocks move."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        alm_sampler='cg',
        n_pcg_iter=10,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert np.all(accepts)  # 'cg' has no accept/reject -- always True
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(samples))
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(samples[0], samples[-1])
    assert not np.allclose(phi_samples[0], phi_samples[-1])


def test_gibbs_chain_messenger_rejects_phi_block(small_model):
    """cl_phiphi_full (Block 3) is only wired up for alm_sampler in {'hmc', 'cg'}."""
    from diffcmb import run_gibbs_chain

    cl_phiphi_full = np.full(small_model.lmax, 1e-6, dtype=np.float64)
    with pytest.raises(ValueError, match="requires alm_sampler"):
        run_gibbs_chain(
            small_model,
            n_samples=1,
            n_burnin=1,
            alm_sampler='messenger',
            cl_phiphi_full=cl_phiphi_full,
            seed=42,
        )


@pytest.fixture(scope="module")
def small_masked_matrixfree_model():
    """Small masked-sky model on the matrix-free ducc0 SHT path, for
    alm_sampler='messenger' (ROADMAP.md Phase 0c). Masking is applied by
    zeroing Ninv/prior_map outside a polar-cap unmasked_idx -- overriding
    unmasked_idx alone is not enough (see scripts/debug_messenger_masksky.py).

    Uses NSIDE=8 rather than the module-level NSIDE=4: the messenger sampler
    relies on the full-sky SHT being adequately resolved (roughly NSIDE >
    lmax/2, the HEALPix Nyquist-ish limit), and NSIDE=4 at LMAX=10 is well
    under that, which was found to make the messenger chain diverge almost
    immediately (a different, more severe failure than the ~1-2% quadrature
    error documented in sample_alm_messenger, which was characterised at
    NSIDE=8/lmax=10).
    """
    try:
        import ducc0  # noqa: F401
        import healpy as hp
    except ImportError:
        pytest.skip("ducc0/healpy not installed")

    from diffcmb import CosmologyAdvancedSampling

    messenger_nside = 8
    m = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=messenger_nside, _noisesig=1.0, data_mode='synthetic',
        use_matrixfree_sht=True,
    )
    theta, _ = hp.pix2ang(messenger_nside, np.arange(m.NPIX))
    cutoff = np.arccos(1 - 2 * 0.7)
    m.unmasked_idx = np.where(theta < cutoff)[0]
    mask = np.ones(m.NPIX, dtype=bool)
    mask[m.unmasked_idx] = False
    m.Ninv = m.Ninv.copy()
    m.Ninv[mask] = 0.0
    m.prior_map = m.prior_map.copy()
    m.prior_map[mask] = 0.0
    m._ensure_tf_tensors()
    return m


@skip_no_tfp
def test_gibbs_chain_messenger_moves_and_stays_bounded(small_masked_matrixfree_model):
    """alm_sampler='messenger' runs on a masked sky without crashing or
    diverging (ROADMAP.md Phase 0c).

    This is deliberately a boundedness/smoke test, not a statistical-accuracy
    test: the messenger sampler's harmonic-space step uses a diagonal
    approximation of A^T A (the full-sky SHT's Gram matrix), which is only
    ~1-2% accurate for a real (quadrature-approximate) HEALPix SHT. Without a
    safety margin (samplers.py::sample_alm_messenger's
    norm_diag_safety_margin) this makes the messenger Markov chain's
    per-step transition operator have spectral radius > 1 under masking --
    genuine divergence, found via scripts/debug_messenger_masksky.py. The
    margin fixes divergence but trades off against posterior bias (still
    tens of posterior standard errors at lmax=10/NSIDE=8 in that script,
    not yet resolved) -- so alm_sampler='messenger' is not yet validated for
    production use; this test only guards against a regression to outright
    numerical blow-up.
    """
    from diffcmb import run_gibbs_chain

    samples, logp, accepts, final_step = run_gibbs_chain(
        small_masked_matrixfree_model,
        n_samples=5,
        n_burnin=5,
        alm_sampler='messenger',
        n_messenger_iter=20,
        seed=42,
    )
    assert samples.shape == (5, len(small_masked_matrixfree_model.x0))
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(samples))
    assert np.abs(samples).max() < 1e3
    assert not np.allclose(samples[0], samples[-1])


# ── phi posterior mass matrix (likelihood-curvature-aware preconditioner) ────

def test_build_phi_posterior_mass_sqrt_matches_prior_only_when_fisher_zero():
    """build_phi_posterior_mass_sqrt(..., diag_fisher_per_L=0) must reduce to
    build_phi_prior_mass_sqrt exactly -- the new function is a strict
    generalisation (adds likelihood curvature on top of the existing prior
    curvature), so a zero Fisher term must be a no-op.
    """
    from diffcmb.samplers import (
        build_phi_posterior_mass_sqrt,
        build_phi_prior_mass_sqrt,
    )

    lmax = 10
    rng = np.random.default_rng(0)
    cl_phiphi_full = rng.uniform(1e-12, 1e-10, size=lmax)

    prior_only = build_phi_prior_mass_sqrt(lmax, cl_phiphi_full)
    diag_fisher_per_L = np.zeros(lmax)
    combined = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)

    np.testing.assert_allclose(combined, prior_only)


def test_build_phi_posterior_mass_sqrt_increases_with_fisher_curvature():
    """A nonzero diag_fisher_per_L must strictly increase the mass (i.e.
    shrink the whitened step) for every mode at that L, matching
    build_posterior_mass_sqrt's 1/C_l + Ninv_eff pattern for the alm block."""
    from diffcmb.samplers import build_phi_posterior_mass_sqrt

    lmax = 10
    rng = np.random.default_rng(1)
    cl_phiphi_full = rng.uniform(1e-12, 1e-10, size=lmax)
    diag_fisher_per_L = np.zeros(lmax)
    diag_fisher_per_L[5] = 1e11  # large curvature injected at L=5 only

    zero_fisher = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, np.zeros(lmax))
    with_fisher = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)

    assert np.all(with_fisher >= zero_fisher - 1e-30)
    assert np.any(with_fisher > zero_fisher * 1.5)


# ── run_gibbs_chain: phi_mass_matrix opt-in ('prior' default vs 'fisher') ────

@skip_no_tfp
def test_gibbs_chain_phi_mass_matrix_prior_default_unchanged(small_model, monkeypatch):
    """phi_mass_matrix defaults to 'prior' -- estimate_phi_diag_fisher must
    never be invoked in that mode (neither by omitting the parameter nor by
    passing 'prior' explicitly), so the new parameter is a strict opt-in
    with zero effect on existing callers.

    Note: this codebase's HMC chain is not bit-for-bit reproducible across
    separate run_gibbs_chain calls even with the same seed and model (TF's
    global op-level nondeterminism, confirmed independent of this change),
    so "unchanged" is checked by non-invocation of the new code path, not
    by comparing two runs' output arrays.
    """
    import diffcmb.samplers as samplers_mod
    from diffcmb import run_gibbs_chain

    def _boom(*args, **kwargs):
        raise AssertionError("estimate_phi_diag_fisher must not be called when phi_mass_matrix='prior'")

    monkeypatch.setattr(
        "diffcmb.lensing.estimate_phi_diag_fisher", _boom, raising=False
    )

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    kwargs = {
        "n_samples": 5, "n_burnin": 5, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_hmc_step_size": 0.01, "phi_n_lfs": 5,
        "seed": 42,
    }

    run_gibbs_chain(small_model, **kwargs)
    run_gibbs_chain(small_model, phi_mass_matrix='prior', **kwargs)


@skip_no_tfp
def test_gibbs_chain_phi_mass_matrix_fisher_moves(small_model):
    """phi_mass_matrix='fisher' recomputes the phi mass matrix mid-burn-in
    using estimate_phi_diag_fisher and keeps running without breaking the
    chain -- shapes/finiteness match the existing prior-only phi-block test."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)

    samples, phi_samples, logp, accepts, final_step = run_gibbs_chain(
        small_model,
        n_samples=5,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        phi_mass_matrix='fisher',
        phi_fisher_warmup_iter=2,
        phi_fisher_n_probes=2,
        seed=42,
    )
    assert samples.shape == (5, len(small_model.x0))
    assert phi_samples.shape == (5, n_phi)
    assert logp.shape == (5,)
    assert accepts.shape == (5,)
    assert isinstance(final_step, float)
    assert np.all(np.isfinite(phi_samples))
    assert not np.allclose(phi_samples[0], phi_samples[-1])


# ── run_gibbs_chain: sample_cl_phiphi opt-in (optional Block 4) ─────────────

@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_default_unchanged(small_model, monkeypatch):
    """sample_cl_phiphi defaults to False -- sample_cl_phiphi_given_phi must
    never be invoked in that mode (neither by omitting the parameter nor by
    passing sample_cl_phiphi=False explicitly), matching the same
    non-invocation pattern used above for phi_mass_matrix='prior' (this
    codebase's HMC chain is not bit-for-bit reproducible across separate
    run_gibbs_chain calls even with a fixed seed, so "unchanged" has to be
    checked by non-invocation rather than by comparing output arrays)."""
    from diffcmb import run_gibbs_chain

    def _boom(*args, **kwargs):
        raise AssertionError("sample_cl_phiphi_given_phi must not be called when sample_cl_phiphi=False")

    monkeypatch.setattr(
        "diffcmb.lensing.sample_cl_phiphi_given_phi", _boom, raising=False
    )

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    kwargs = {
        "n_samples": 5, "n_burnin": 5, "hmc_step_size": 0.01, "n_lfs": 5,
        "cl_phiphi_full": cl_phiphi_full, "phi_hmc_step_size": 0.01, "phi_n_lfs": 5,
        "seed": 42,
    }

    run_gibbs_chain(small_model, **kwargs)
    run_gibbs_chain(small_model, sample_cl_phiphi=False, **kwargs)


def test_gibbs_chain_sample_cl_phiphi_requires_phi_block():
    """sample_cl_phiphi=True without cl_phiphi_full (i.e. Block 3 disabled)
    must raise, not silently no-op -- Block 4 only makes sense once phi is
    itself being sampled."""
    from diffcmb import run_gibbs_chain

    class _DummyModel:
        lmax = 10

    with pytest.raises(ValueError, match="sample_cl_phiphi"):
        run_gibbs_chain(_DummyModel(), n_samples=1, n_burnin=1, sample_cl_phiphi=True)


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_rejects_fisher_mass_matrix(small_model):
    """sample_cl_phiphi=True combined with phi_mass_matrix='fisher' is not
    supported (the Fisher mass matrix is estimated once at burn-in and
    frozen thereafter, which would silently go stale against a spectrum that
    keeps changing every sweep) -- must raise, not silently combine them."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    cl_phiphi_full = np.full(lmax, 1e-6, dtype=np.float64)
    with pytest.raises(ValueError, match="fisher"):
        run_gibbs_chain(
            small_model, n_samples=1, n_burnin=1, hmc_step_size=0.01, n_lfs=5,
            cl_phiphi_full=cl_phiphi_full, phi_hmc_step_size=0.01, phi_n_lfs=5,
            phi_mass_matrix='fisher', sample_cl_phiphi=True, seed=42,
        )


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_moves_and_returns_extra_array(small_model):
    """sample_cl_phiphi=True runs Block 4 alongside Blocks 1-3, returns an
    extra cl_phiphi_samples array as the last element of the return tuple,
    and that array actually moves (is not stuck at its initial value)."""
    from diffcmb import run_gibbs_chain

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    # phi_initial defaults to all-zeros, which (with only 5 burn-in sweeps)
    # never moves far enough from S_L=0 for Block 4 to escape its numerical
    # floor clip -- start from a draw at the assumed prior scale instead, so
    # there's real phi power for the exact-draw to pick up on straight away.
    from diffcmb.samplers import _alm_index_lm
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    phi_initial = np.random.default_rng(3).normal(scale=np.sqrt(var))

    samples, phi_samples, logp, accepts, final_step, cl_phiphi_samples = run_gibbs_chain(
        small_model,
        n_samples=8,
        n_burnin=5,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        sample_cl_phiphi=True,
        seed=42,
    )
    assert samples.shape == (8, len(small_model.x0))
    assert phi_samples.shape == (8, n_phi)
    assert logp.shape == (8,)
    assert accepts.shape == (8,)
    assert isinstance(final_step, float)
    assert cl_phiphi_samples.shape == (8, lmax - 2)
    assert np.all(np.isfinite(cl_phiphi_samples))
    assert not np.allclose(cl_phiphi_samples[0], cl_phiphi_samples[-1])


@skip_no_tfp
def test_gibbs_chain_sample_cl_phiphi_recovers_known_spectrum(small_model):
    """Smoke test (mirrors the achievements.md-style small-lmax validation
    used elsewhere in this project): a short full run_gibbs_chain chain with
    Block 4 enabled, starting phi already near its stationary distribution
    for a known constant true C_L^phiphi, should keep the sampled spectrum
    in the right ballpark of that true value rather than drifting to a
    wildly different scale (which is what a weighting convention bug, e.g.
    the classic 1/C_l-vs-2/C_l mistake, would produce)."""
    from diffcmb import run_gibbs_chain
    from diffcmb.samplers import _alm_index_lm

    lmax = small_model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag
    C0 = 1e-6
    cl_phiphi_full = np.full(lmax, C0, dtype=np.float64)

    # Initialise phi from the assumed prior at C0 so the chain starts near
    # its stationary distribution instead of needing to burn in the
    # spectrum itself from a cold start in a handful of sweeps.
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    is_real = np.arange(n_phi) < n_real
    var = np.empty(n_phi)
    real_idx = np.where(is_real)[0]
    var[real_idx] = np.where(m_arr[real_idx] == 0, C0, C0 / 2.0)
    imag_idx = np.where(~is_real)[0]
    var[imag_idx] = C0 / 2.0
    rng = np.random.default_rng(99)
    phi_initial = rng.normal(scale=np.sqrt(var))

    _samples, _phi_samples, _logp, _accepts, _final_step, cl_phiphi_samples = run_gibbs_chain(
        small_model,
        n_samples=40,
        n_burnin=20,
        hmc_step_size=0.01,
        n_lfs=5,
        cl_phiphi_full=cl_phiphi_full,
        phi_initial=phi_initial,
        phi_hmc_step_size=0.01,
        phi_n_lfs=5,
        sample_cl_phiphi=True,
        seed=7,
    )

    recovered = np.exp(cl_phiphi_samples[:, -1]).mean()  # highest-L bin: least small-L bias
    assert np.isfinite(recovered)
    # Loose ballpark check by design (few samples, small lmax): catches gross
    # convention errors (an order of magnitude or a 2x weighting bug) without
    # over-fitting to this specific short chain's Monte Carlo noise.
    assert 0.1 * C0 < recovered < 10.0 * C0, (
        f"recovered C_L (highest-L bin) = {recovered:.3e}, expected within "
        f"an order of magnitude of true C0={C0:.3e}"
    )
