"""Phase 1 lensing operator — gradient validation tests.

All tests require healpy (and most require TensorFlow).  They are automatically
skipped on environments where those libraries are absent (e.g. login nodes).

Validation strategy
-------------------
1. deflection_field: zero-phi → zero deflection; amplitude sanity check
2. precompute_lensing: weight normalisation
3. apply_lensing_tf: identity at zero phi; dL/dT_map autodiff vs FD
4. lens_map_phi_diff_tf: dL/dphi_alm autodiff vs FD  ← key Phase 1 check
5. psi_lensed: value matches unlensed posterior when phi=0 and noise→∞
6. alm end-to-end: dL/dalm through Y-matrix → apply_lensing pipeline vs FD
"""

import numpy as np
import pytest

try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Small problem size so tests run in seconds on a compute node
LMAX = 20
NSIDE = 16   # pixel size ≈ 220 arcmin; deflection ≪ pixel for typical phi amplitude


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_phi_packed(lmax, rng, amplitude=5e-4):
    """Random lensing potential in packed format; amplitude << pixel size."""
    from diffcmb.lensing import _alm_hp_to_packed
    size = hp.Alm.getsize(lmax)
    phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    ells = np.array([hp.Alm.getlm(lmax, i)[0] for i in range(size)], dtype=float)
    ells = np.maximum(ells, 1.0)
    phi_hp *= amplitude / ells**1.5
    phi_hp[0] = 0.0   # monopole = 0
    if lmax >= 2:
        phi_hp[1] = 0.0   # l=1, m=0 = 0
    return _alm_hp_to_packed(phi_hp.astype(np.complex128), lmax)


def _rand_alm_packed(lmax, rng, scale=10.0):
    """Random CMB alm in packed (real+imag) format."""
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    return rng.standard_normal(n_real + n_imag).astype(np.float64) * scale


def _make_model(lmax=LMAX, nside=NSIDE):
    """Build a minimal synthetic model (no Planck data)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128
    )
    model._ensure_tf_tensors()
    return model


# ---------------------------------------------------------------------------
# 1 — deflection_field basics
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_zero_phi():
    """Zero phi_alm → zero deflection."""
    from diffcmb.lensing import deflection_field
    phi_alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    d_theta, d_phi = deflection_field(phi_alm, NSIDE, LMAX)
    assert np.allclose(d_theta, 0.0, atol=1e-14)
    assert np.allclose(d_phi, 0.0, atol=1e-14)


@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_deflection_amplitude_small():
    """Realistic phi gives deflection ≪ pixel size."""
    from diffcmb.lensing import _alm_packed_to_hp, deflection_field
    rng = np.random.default_rng(1)
    phi = _rand_phi_packed(LMAX, rng, amplitude=5e-4)
    phi_hp = _alm_packed_to_hp(phi, LMAX)
    d_theta, d_phi = deflection_field(phi_hp, NSIDE, LMAX)
    pixel_size_rad = np.pi / (4 * NSIDE)  # approximate
    assert np.max(np.abs(d_theta)) < pixel_size_rad
    assert np.max(np.abs(d_phi)) < pixel_size_rad


# ---------------------------------------------------------------------------
# 2 — format round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_alm_format_round_trip():
    """packed → hp → packed is lossless."""
    from diffcmb.lensing import _alm_hp_to_packed, _alm_packed_to_hp
    rng = np.random.default_rng(7)
    packed = _rand_phi_packed(LMAX, rng)
    hp_alm = _alm_packed_to_hp(packed, LMAX)
    packed2 = _alm_hp_to_packed(hp_alm, LMAX)
    np.testing.assert_allclose(packed, packed2, atol=1e-14)


# ---------------------------------------------------------------------------
# 3 — precompute_lensing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
def test_precompute_weights_sum_to_one():
    """Bilinear weights always sum to 1 per pixel."""
    from diffcmb.lensing import _alm_packed_to_hp, precompute_lensing
    rng = np.random.default_rng(2)
    phi_hp = _alm_packed_to_hp(_rand_phi_packed(LMAX, rng), LMAX)
    pix = np.arange(hp.nside2npix(NSIDE))
    _, weights, _, _ = precompute_lensing(phi_hp, NSIDE, LMAX, pix)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4 — apply_lensing_tf: identity + dL/dT_map gradient
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_identity_at_zero_phi():
    """Zero deflection: T_lensed == T_unlensed at all pixels."""
    from diffcmb.lensing import apply_lensing_tf, precompute_lensing
    npix = hp.nside2npix(NSIDE)
    phi_alm = np.zeros(hp.Alm.getsize(LMAX), dtype=complex)
    pix = np.arange(npix)
    neighbors, weights, _, _ = precompute_lensing(phi_alm, NSIDE, LMAX, pix)
    rng = np.random.default_rng(3)
    T = tf.constant(rng.standard_normal(npix), dtype=tf.float64)
    T_lensed = apply_lensing_tf(
        T,
        tf.constant(neighbors, tf.int32),
        tf.constant(weights, tf.float64),
    )
    np.testing.assert_allclose(T_lensed.numpy(), T.numpy(), atol=1e-10)


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_apply_lensing_dT_grad_vs_fd():
    """dL/dT_map from TF autodiff agrees with finite differences."""
    from diffcmb.lensing import _alm_packed_to_hp, apply_lensing_tf, precompute_lensing
    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(17)
    phi_hp = _alm_packed_to_hp(_rand_phi_packed(LMAX, rng), LMAX)
    pix = np.arange(npix)
    neighbors, weights, _, _ = precompute_lensing(phi_hp, NSIDE, LMAX, pix)
    nbrs_tf = tf.constant(neighbors, tf.int32)
    wts_tf = tf.constant(weights, tf.float64)

    T_np = rng.standard_normal(npix)
    T_var = tf.Variable(T_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(apply_lensing_tf(T_var, nbrs_tf, wts_tf))
    g_auto = tape.gradient(loss, T_var).numpy()

    eps = 1e-5
    sampled = np.arange(0, npix, max(1, npix // 30))
    g_fd = np.zeros(npix)
    for i in sampled:
        T_p = T_np.copy()
        T_p[i] += eps
        T_m = T_np.copy()
        T_m[i] -= eps
        lp = tf.reduce_sum(apply_lensing_tf(tf.constant(T_p, tf.float64), nbrs_tf, wts_tf))
        lm = tf.reduce_sum(apply_lensing_tf(tf.constant(T_m, tf.float64), nbrs_tf, wts_tf))
        g_fd[i] = (lp.numpy() - lm.numpy()) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[sampled], g_fd[sampled], rtol=1e-4, atol=1e-8,
        err_msg="dL/dT_map autodiff vs FD mismatch"
    )


# ---------------------------------------------------------------------------
# 5 — lens_map_phi_diff_tf: dL/dphi_alm gradient validation  ← KEY TEST
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_phi_grad_deflection_adjoint_vs_fd():
    """dL/dphi_alm from custom_gradient agrees with finite differences.

    Uses a simple loss = sum(T_lensed) and checks the packed phi gradient.
    This validates the full chain:
        phi_packed → deflection → bilinear weights → T_lensed
    and its reverse.

    eps note (bug history, Phase 2 ROADMAP.md): a single phi_alm component
    perturbs the deflection field at every pixel simultaneously, and
    hp.get_interp_weights' bilinear scheme is only C0 (continuous) not C1
    (its derivative has genuine kinks at interpolation-cell boundaries).
    With eps=1e-6 (the original value here), a handful of the ~600 unmasked
    pixels' lensed positions happened to cross such a boundary within the
    perturbation, making *this FD reference* — not the analytic gradient —
    unstable: verified directly that this same FD estimate changes sign
    and swings by >100% between eps=1e-6 and eps=1e-7 for some components,
    while agreeing with the analytic gradient to ~1e-5 relative once eps is
    small enough (<=3e-9) not to cross a boundary. eps=1e-9 is safely in
    that regime (deflection_field is exactly linear in phi_alm, so no
    truncation/roundoff tradeoff pushes eps this small — checked stable
    across 1e-9..3e-10).
    """
    from diffcmb.lensing import lens_map_phi_diff_tf

    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(42)
    phi0 = _rand_phi_packed(LMAX, rng, amplitude=1e-4)
    T_np = rng.standard_normal(npix) * 50.0
    pix = np.arange(npix)

    phi_var = tf.Variable(phi0, dtype=tf.float64)
    T_tf = tf.constant(T_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        T_lensed = lens_map_phi_diff_tf(T_tf, phi_var, NSIDE, LMAX, pix)
        loss = tf.reduce_sum(T_lensed)
    g_auto = tape.gradient(loss, phi_var).numpy()

    # Finite differences over a subset of phi_packed components
    eps = 1e-9
    n_phi = len(phi0)
    sampled = np.arange(0, n_phi, max(1, n_phi // 20))
    g_fd = np.zeros(n_phi)
    for i in sampled:
        ph_p = phi0.copy()
        ph_p[i] += eps
        ph_m = phi0.copy()
        ph_m[i] -= eps
        lp = tf.reduce_sum(lens_map_phi_diff_tf(T_tf, tf.constant(ph_p, tf.float64), NSIDE, LMAX, pix))
        lm = tf.reduce_sum(lens_map_phi_diff_tf(T_tf, tf.constant(ph_m, tf.float64), NSIDE, LMAX, pix))
        g_fd[i] = (lp.numpy() - lm.numpy()) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[sampled], g_fd[sampled], rtol=0.02, atol=1e-6,
        err_msg="dL/dphi_alm autodiff vs FD mismatch"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_lens_map_phi_diff_tf_traceable_in_tf_function():
    """lens_map_phi_diff_tf must survive tf.function tracing (Phase 1.5,
    ROADMAP.md): both the bilinear-geometry precompute and the FD backward
    pass now go through tf.py_function rather than a bare .numpy() call
    (mirroring sht_ducc.py's masked_synthesis_tf), so this op can sit inside
    samplers.py's @tf.function-decorated grad/matvec wrappers, not just run
    in pure eager mode. Also checks the forward value and both gradients
    (w.r.t. T_map and phi) match the eager-mode result exactly."""
    from diffcmb.lensing import lens_map_phi_diff_tf

    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(7)
    phi0 = _rand_phi_packed(LMAX, rng, amplitude=1e-4)
    T_np = rng.standard_normal(npix) * 50.0
    pix = np.arange(npix)

    T_tf = tf.constant(T_np, dtype=tf.float64)
    phi_tf = tf.constant(phi0, dtype=tf.float64)

    def _run(T, phi):
        with tf.GradientTape() as tape:
            tape.watch([T, phi])
            out = lens_map_phi_diff_tf(T, phi, NSIDE, LMAX, pix)
            loss = tf.reduce_sum(out ** 2)
        g_T, g_phi = tape.gradient(loss, [T, phi])
        return out, g_T, g_phi

    out_eager, gT_eager, gphi_eager = _run(T_tf, phi_tf)

    traced = tf.function(_run)
    out_traced, gT_traced, gphi_traced = traced(T_tf, phi_tf)

    np.testing.assert_allclose(out_traced.numpy(), out_eager.numpy())
    np.testing.assert_allclose(gT_traced.numpy(), gT_eager.numpy())
    np.testing.assert_allclose(gphi_traced.numpy(), gphi_eager.numpy())
    assert np.all(np.isfinite(gphi_traced.numpy()))


# ---------------------------------------------------------------------------
# 6 — psi_lensed: value sanity + alm/phi gradient validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_zero_phi_matches_unlensed():
    """psi_lensed with phi=0 must equal model._psi_tf_raw (unlensed posterior)."""
    model = _make_model()
    lmax = model.lmax

    params_np = np.zeros(lmax - 2 + (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2)
    params_np[: lmax - 2] = 5.0

    params_tf = tf.constant(params_np, dtype=tf.float64)
    n_phi = (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2
    phi_tf = tf.zeros(n_phi, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed
    psi_lens_val = psi_lensed(model, params_tf, phi_tf).numpy()
    psi_unlens_val = model._psi_tf_raw(params_tf).numpy()

    # With phi=0 the lensing is the identity so psi_lensed == _psi_tf_raw
    np.testing.assert_allclose(
        psi_lens_val, psi_unlens_val, rtol=1e-6,
        err_msg="psi_lensed(phi=0) ≠ _psi_tf_raw"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_zero_phi_matches_unlensed_with_beam():
    """psi_lensed with phi=0 must equal model._psi_tf_raw when a beam is set,
    exactly as test_psi_lensed_zero_phi_matches_unlensed does unbeamed --
    confirms lens_map_tf/psi_lensed's beam_pixwin_per_l application
    (lensing.py) is consistent with _psi_tf_raw's (model.py, validated
    directly against an independent healpy ground truth in
    tests/test_model.py::test_psi_tf_beam_pixwin_matches_ground_truth_synthesis)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=LMAX, _NSIDE=NSIDE, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, beam_fwhm_arcmin=30.0,
    )
    model._ensure_tf_tensors()
    lmax = model.lmax

    params_np = np.zeros(lmax - 2 + (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2)
    params_np[: lmax - 2] = 5.0

    params_tf = tf.constant(params_np, dtype=tf.float64)
    n_phi = (lmax * (lmax + 1) // 2 - 3) + (lmax - 2) * (lmax - 1) // 2
    phi_tf = tf.zeros(n_phi, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed
    psi_lens_val = psi_lensed(model, params_tf, phi_tf).numpy()
    psi_unlens_val = model._psi_tf_raw(params_tf).numpy()

    np.testing.assert_allclose(
        psi_lens_val, psi_unlens_val, rtol=1e-6,
        err_msg="psi_lensed(phi=0, beamed) != _psi_tf_raw(beamed)"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_alm_grad_vs_fd():
    """dL/dalm from TF autodiff on psi_lensed agrees with finite differences."""
    from diffcmb.lensing import psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(11)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0   # log C_l
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_var = tf.Variable(params_np, dtype=tf.float64)
    phi_tf = tf.constant(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_var, phi_tf)
    g_auto = tape.gradient(val, params_var).numpy()

    # FD on alm components only (skip lncl for speed)
    eps = 1e-5
    alm_slice = slice(n_lncl, n_lncl + 5)   # check first 5 alm coefficients
    g_fd = np.zeros(len(params_np))
    for i in range(n_lncl, n_lncl + 5):
        p_p = params_np.copy()
        p_p[i] += eps
        p_m = params_np.copy()
        p_m[i] -= eps
        lp = psi_lensed(model, tf.constant(p_p, tf.float64), phi_tf).numpy()
        lm = psi_lensed(model, tf.constant(p_m, tf.float64), phi_tf).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[alm_slice], g_fd[alm_slice], rtol=1e-4, atol=1e-6,
        err_msg="dL/dalm autodiff vs FD mismatch in psi_lensed"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_psi_lensed_phi_grad_vs_fd():
    """dL/dphi_alm from TF autodiff on psi_lensed agrees with finite differences.

    eps note: see test_phi_grad_deflection_adjoint_vs_fd — a coarse FD eps
    here perturbs every pixel's lensed position simultaneously and can cross
    a genuine (C0-but-not-C1) HEALPix bilinear-interpolation-cell boundary,
    making the FD reference itself unstable rather than the analytic
    gradient being wrong. eps=1e-9 avoids that (checked stable down to 3e-10;
    deflection_field is exactly linear in phi_alm so there's no
    truncation/roundoff tradeoff forcing eps larger).
    """
    from diffcmb.lensing import psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag

    rng = np.random.default_rng(99)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_var = tf.Variable(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_tf, phi_var)
    g_auto = tape.gradient(val, phi_var).numpy()

    # FD on first 8 phi components
    eps = 1e-9
    g_fd = np.zeros(n_phi)
    for i in range(min(8, n_phi)):
        ph_p = phi_np.copy()
        ph_p[i] += eps
        ph_m = phi_np.copy()
        ph_m[i] -= eps
        lp = psi_lensed(model, params_tf, tf.constant(ph_p, tf.float64)).numpy()
        lm = psi_lensed(model, params_tf, tf.constant(ph_m, tf.float64)).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[:8], g_fd[:8], rtol=0.02, atol=1e-5,
        err_msg="dL/dphi_alm autodiff vs FD mismatch in psi_lensed"
    )


# ---------------------------------------------------------------------------
# 7 — log_prob_phi_block (Phase 2, Block 3 target) — value + gradient checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_log_prob_phi_block_zero_prior_matches_neg_psi_lensed():
    """With an infinite phi prior variance (cl_phiphi -> inf), the phi prior
    term vanishes and log_prob_phi_block reduces to -psi_lensed."""
    from diffcmb.lensing import log_prob_phi_block, psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(7)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_tf = tf.constant(_rand_phi_packed(lmax, rng, amplitude=1e-4), dtype=tf.float64)

    cl_phiphi_huge = np.full(lmax, 1e30)
    log_prob = log_prob_phi_block(model, params_tf, phi_tf, cl_phiphi_huge).numpy()
    neg_psi = -psi_lensed(model, params_tf, phi_tf).numpy()

    np.testing.assert_allclose(
        log_prob, neg_psi, rtol=1e-6,
        err_msg="log_prob_phi_block with cl_phiphi->inf should match -psi_lensed"
    )


@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_log_prob_phi_block_grad_vs_fd():
    """dlog_prob/dphi_alm from TF autodiff agrees with finite differences.

    This is the gradient Block 3's HMC step will need every leapfrog
    iteration, so it must be correct before Phase 2 wiring begins.
    """
    from diffcmb.lensing import log_prob_phi_block
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_phi = n_real + n_imag

    rng = np.random.default_rng(13)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)

    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)
    phi_var = tf.Variable(phi_np, dtype=tf.float64)
    cl_phiphi = np.full(lmax, 1e-8)   # tight but finite prior

    with tf.GradientTape() as tape:
        val = log_prob_phi_block(model, params_tf, phi_var, cl_phiphi)
    g_auto = tape.gradient(val, phi_var).numpy()

    eps = 1e-6
    g_fd = np.zeros(n_phi)
    for i in range(min(8, n_phi)):
        ph_p = phi_np.copy()
        ph_p[i] += eps
        ph_m = phi_np.copy()
        ph_m[i] -= eps
        lp = log_prob_phi_block(model, params_tf, tf.constant(ph_p, tf.float64), cl_phiphi).numpy()
        lm = log_prob_phi_block(model, params_tf, tf.constant(ph_m, tf.float64), cl_phiphi).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(
        g_auto[:8], g_fd[:8], rtol=0.02, atol=1e-5,
        err_msg="dlog_prob/dphi_alm autodiff vs FD mismatch in log_prob_phi_block"
    )


# ---------------------------------------------------------------------------
# 7 — matrix-free ducc0 SHT path matches the dense-SHT reference (Phase 2
#     gate: Block 3 was dense-SHT-only until this port; see ROADMAP.md
#     Section 1 and the "dense-reference discipline" in Standing discipline)
# ---------------------------------------------------------------------------

def _make_matrixfree_model(lmax=LMAX, nside=NSIDE):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    return model


def _sync_prior_map(dense_model, mf_model, rng):
    """Give both models the same random full-sky prior_map + masked/parts views."""
    npix = dense_model.NPIX
    shared_map = rng.standard_normal(npix)

    dense_model.prior_map = shared_map
    dense_model.prior_map_parts = [
        tf.convert_to_tensor(shared_map[dense_model.unmasked_idx], dtype=tf.float64)
    ]
    mf_model.prior_map = shared_map
    mf_model.prior_map_masked = tf.convert_to_tensor(
        shared_map[mf_model.unmasked_idx], dtype=tf.float64
    )


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_lens_map_tf_matrixfree_matches_dense():
    """lens_map_tf: matrix-free ducc0 full-sky synthesis vs the dense Y-matrix
    scatter-from-unmasked path, on a full-sky (no mask) model where the two
    should agree to numerical precision."""
    dense_model = _make_model()
    mf_model = _make_matrixfree_model()
    assert len(dense_model.unmasked_idx) == dense_model.NPIX, (
        "test assumes a full-sky (unmasked) model so dense scatter-from-unmasked "
        "and matrix-free full-sky synthesis cover identical pixels"
    )

    rng = np.random.default_rng(7)
    alm_packed = _rand_alm_packed(LMAX, rng)
    phi_hp = np.zeros(hp.Alm.getsize(LMAX - 1), dtype=np.complex128)  # zero phi

    from diffcmb.lensing import lens_map_tf
    T_dense = lens_map_tf(dense_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()
    T_mf = lens_map_tf(mf_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()

    np.testing.assert_allclose(T_mf, T_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_matches_dense():
    """psi_lensed: matrix-free vs dense-SHT give the same log-posterior value
    and the same gradient w.r.t. alm/C_l, given identical alm/phi/data."""
    dense_model = _make_model()
    mf_model = _make_matrixfree_model()
    assert len(dense_model.unmasked_idx) == dense_model.NPIX

    rng = np.random.default_rng(11)
    _sync_prior_map(dense_model, mf_model, rng)

    lmax = LMAX
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.Variable(params_np, dtype=tf.float64)

    phi_packed = _rand_phi_packed(lmax, rng, amplitude=1e-4)
    phi_tf = tf.constant(phi_packed, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed

    with tf.GradientTape() as tape:
        val_dense = psi_lensed(dense_model, params_tf, phi_tf)
    grad_dense = tape.gradient(val_dense, params_tf).numpy()

    with tf.GradientTape() as tape:
        val_mf = psi_lensed(mf_model, params_tf, phi_tf)
    grad_mf = tape.gradient(val_mf, params_tf).numpy()

    np.testing.assert_allclose(val_mf.numpy(), val_dense.numpy(), rtol=1e-6)
    np.testing.assert_allclose(grad_mf, grad_dense, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 7 — matrix-free vs dense, masked sky (ROADMAP.md Section 1: "Not yet
#     re-validated on a masked sky" -- the dense-reference discipline applied
#     to the masked-sky matrix-free lensing path for the first time, before
#     any masked-sky Phase 2 chain such as the planned f_sky~0.016
#     patch-scale sanity check.)
# ---------------------------------------------------------------------------

def _polar_cap_mask_idx(nside, f_sky):
    """Contiguous polar-cap mask index set covering the requested f_sky
    fraction of the sphere (same construction as test_samplers.py's
    small_masked_matrixfree_model fixture)."""
    npix = 12 * nside * nside
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    cutoff = np.arccos(1 - 2 * f_sky)
    return np.where(theta < cutoff)[0]


def _make_masked_dense_model(lmax=LMAX, nside=NSIDE, f_sky=0.3):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128,
    )
    model.unmasked_idx = _polar_cap_mask_idx(nside, f_sky)
    model._ensure_tf_tensors()
    return model


def _make_masked_matrixfree_model(lmax=LMAX, nside=NSIDE, f_sky=0.3):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from diffcmb import CosmologyAdvancedSampling
    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=100.0,
        data_mode="synthetic", dtype=tf.complex128, use_matrixfree_sht=True,
    )
    model.unmasked_idx = _polar_cap_mask_idx(nside, f_sky)
    model._ensure_tf_tensors()
    return model


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_lens_map_tf_matrixfree_matches_dense_masked_sky():
    """lens_map_tf: matrix-free ducc0 synthesis vs the dense Y-matrix path
    agree to numerical precision on a masked sky (f_sky~0.3, a polar cap),
    the first correctness check of the matrix-free path outside full-sky."""
    dense_model = _make_masked_dense_model()
    mf_model = _make_masked_matrixfree_model()
    np.testing.assert_array_equal(dense_model.unmasked_idx, mf_model.unmasked_idx)
    assert len(dense_model.unmasked_idx) < dense_model.NPIX, "mask should be nontrivial"

    rng = np.random.default_rng(23)
    alm_packed = _rand_alm_packed(LMAX, rng)
    phi_hp = np.zeros(hp.Alm.getsize(LMAX - 1), dtype=np.complex128)  # zero phi

    from diffcmb.lensing import lens_map_tf
    T_dense = lens_map_tf(dense_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()
    T_mf = lens_map_tf(mf_model, tf.constant(alm_packed, tf.float64), phi_hp).numpy()

    np.testing.assert_allclose(T_mf, T_dense, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_matches_dense_masked_sky_zero_phi():
    """psi_lensed: matrix-free vs dense give the same log-posterior value and
    gradient on a masked sky when phi=0 (identity lensing) -- a valid
    comparison because the dense path's zero-padding-outside-the-mask
    artifact (see the nonzero-phi note below) can't matter when there's no
    deflection to sample it. Nonzero-phi masked-sky matrix-free gradients
    are checked directly against finite differences instead, in
    test_psi_lensed_matrixfree_alm_grad_vs_fd_masked_sky below."""
    dense_model = _make_masked_dense_model()
    mf_model = _make_masked_matrixfree_model()
    np.testing.assert_array_equal(dense_model.unmasked_idx, mf_model.unmasked_idx)

    rng = np.random.default_rng(29)
    _sync_prior_map(dense_model, mf_model, rng)

    lmax = LMAX
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.Variable(params_np, dtype=tf.float64)

    phi_tf = tf.zeros(n_real + n_imag, dtype=tf.float64)

    from diffcmb.lensing import psi_lensed

    with tf.GradientTape() as tape:
        val_dense = psi_lensed(dense_model, params_tf, phi_tf)
    grad_dense = tape.gradient(val_dense, params_tf).numpy()

    with tf.GradientTape() as tape:
        val_mf = psi_lensed(mf_model, params_tf, phi_tf)
    grad_mf = tape.gradient(val_mf, params_tf).numpy()

    np.testing.assert_allclose(val_mf.numpy(), val_dense.numpy(), rtol=1e-6)
    np.testing.assert_allclose(grad_mf, grad_dense, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not HAS_HEALPY or not HAS_TF, reason="healpy/tf not installed")
def test_psi_lensed_matrixfree_alm_grad_vs_fd_masked_sky():
    """dL/dalm from TF autodiff on the matrix-free psi_lensed agrees with
    finite differences on a masked sky (f_sky~0.3), nonzero phi.

    NOTE (real finding, 2026-07-18): comparing matrix-free vs *dense*
    gradients at nonzero phi on a masked sky (mirroring the full-sky
    dense-reference test above) fails at up to 8% relative error on ~58% of
    components. Root cause is not a matrix-free bug: the dense path's
    lens_map_tf/psi_lensed only ever computes T_unlensed at the model's
    unmasked pixels and *zero-pads* everything else before bilinear
    deflection-interpolation (lensing.py's `unsorted_segment_sum` scatter) --
    an unphysical discontinuity at the mask edge that nonzero-phi deflection
    samples can land on. This is exactly the scenario sht_ducc.py's
    full_synthesis_tf docstring and CLAUDE.md were written to avoid ("lensing
    needs this because deflected positions can fall outside the eventual
    mask") -- the dense path was never a valid masked+lensed reference, only
    a valid masked+identity (phi=0) one (see the test above). So the correct
    validation of matrix-free masked-sky gradients is a direct FD check, not
    a dense comparison; this test is that check, and it passes on its own
    terms."""
    from diffcmb.lensing import psi_lensed
    model = _make_masked_matrixfree_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(31)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    params_var = tf.Variable(params_np, dtype=tf.float64)
    phi_tf = tf.constant(phi_np, dtype=tf.float64)

    with tf.GradientTape() as tape:
        val = psi_lensed(model, params_var, phi_tf)
    g_auto = tape.gradient(val, params_var).numpy()

    eps = 1e-5
    g_fd = np.zeros(len(params_np))
    for i in range(n_lncl, n_lncl + 5):
        p_p = params_np.copy()
        p_p[i] += eps
        p_m = params_np.copy()
        p_m[i] -= eps
        lp = psi_lensed(model, tf.constant(p_p, tf.float64), phi_tf).numpy()
        lm = psi_lensed(model, tf.constant(p_m, tf.float64), phi_tf).numpy()
        g_fd[i] = (lp - lm) / (2 * eps)

    alm_slice = slice(n_lncl, n_lncl + 5)
    np.testing.assert_allclose(
        g_auto[alm_slice], g_fd[alm_slice], rtol=1e-4, atol=1e-6,
        err_msg="dL/dalm autodiff vs FD mismatch in masked-sky matrix-free psi_lensed"
    )


# ---------------------------------------------------------------------------
# 8 — estimate_phi_diag_fisher: coordinate-sampled diagonal likelihood-
#     curvature estimate for the phi HMC preconditioner (ROADMAP.md Section
#     1, "Simulation validation" follow-up -- the prior-only phi mass matrix
#     was found to leave the phi block barely mixing at lmax=300).
#
#     Design note: an earlier joint-random-direction (true Hutchinson)
#     probe was tried and abandoned -- perturbing all ~n_phi modes at once
#     crosses the same C0-but-not-C1 bilinear-interpolation-cell boundary
#     documented in test_psi_lensed_phi_grad_vs_fd's eps note, and even
#     after rescaling eps to avoid that, the off-diagonal coupling in this
#     Hessian is large enough that Hutchinson's estimator needs far more
#     than a handful of probes to converge (empirically: 40 joint probes
#     gave per-probe estimates in the thousands against a true diagonal of
#     ~110). Perturbing one packed-phi coordinate at a time instead (the
#     same FD-of-analytic-gradient computation the brute-force reference
#     below uses, just on a sampled subset rather than every mode) is
#     stable and low-variance by construction.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TF or not HAS_HEALPY, reason="TF or healpy not installed")
def test_estimate_phi_diag_fisher_vs_dense_hessian_small_lmax():
    """estimate_phi_diag_fisher's per-L curvature estimate should closely
    match a brute-force FD-of-analytic-gradient diagonal Hessian of
    psi_lensed at one chosen L. With n_probes >= the mode count at that L,
    the estimator samples every mode there deterministically, so this
    should match the dense reference tightly (not just in expectation).
    """
    from diffcmb.lensing import estimate_phi_diag_fisher, psi_lensed
    model = _make_model()
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2

    rng = np.random.default_rng(21)
    params_np = np.zeros(n_lncl + n_real + n_imag)
    params_np[:n_lncl] = 5.0
    params_np[n_lncl:] = rng.standard_normal(n_real + n_imag) * 0.1
    params_tf = tf.constant(params_np, dtype=tf.float64)
    phi_np = _rand_phi_packed(lmax, rng, amplitude=1e-4)

    def grad_at(phi_val):
        phi_var = tf.Variable(phi_val, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = psi_lensed(model, params_tf, phi_var)
        return tape.gradient(val, phi_var).numpy()

    # Brute-force exact diagonal Hessian (central FD of the analytic
    # gradient, one index at a time) for every packed-phi mode at L=6,
    # matching estimate_phi_diag_fisher's per-L averaging convention.
    from diffcmb.samplers import _alm_index_lm
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)
    probe_L = 6
    idx_at_L = np.where(L_arr == probe_L)[0]
    assert len(idx_at_L) > 0

    eps = 1e-9
    dense_diag = np.zeros(len(idx_at_L))
    for k, i in enumerate(idx_at_L):
        phi_p = phi_np.copy()
        phi_p[i] += eps
        phi_m = phi_np.copy()
        phi_m[i] -= eps
        g_p = grad_at(phi_p)[i]
        g_m = grad_at(phi_m)[i]
        dense_diag[k] = (g_p - g_m) / (2 * eps)
    dense_ref_at_L = np.mean(np.maximum(dense_diag, 0.0))

    # n_probes=40 exceeds the mode count at every L for this small model
    # (max multiplicity is 2*lmax-1=39), so every L is sampled exhaustively
    # and deterministically -- the comparison below is not stochastic.
    diag_fisher_per_L = estimate_phi_diag_fisher(
        model, params_tf, tf.constant(phi_np, dtype=tf.float64), lmax,
        n_probes=40, rng=np.random.default_rng(5),
    )

    assert diag_fisher_per_L.shape == (lmax,)
    assert np.all(diag_fisher_per_L >= 0.0)
    np.testing.assert_allclose(
        diag_fisher_per_L[probe_L], dense_ref_at_L, rtol=0.05, atol=1e-6,
        err_msg="estimate_phi_diag_fisher diverges from the brute-force "
                "dense diagonal Hessian reference at L=6 (exhaustive sampling "
                "there should match closely, not just in expectation)"
    )
