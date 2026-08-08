"""MCLMC (microcanonical Langevin Monte Carlo) integrator — spike for Block 3 (phi|alm,C_l).

Hand-implemented in TF, reusing whatever potential/gradient the caller supplies
unchanged (in this repo: ``lensing.log_prob_phi_block`` and its ``tf.GradientTape``
autodiff gradient) — no JAX/blackjax dependency. Algorithm equations ported from
Robnik, De Luca, Silverstein & Seljak (arXiv:2212.08549); the isokinetic-leapfrog
update and Maruyama partial-refreshment formulas mirror blackjax's reference
implementation (``blackjax.mcmc.integrators``), read as a spec, not copied as code.

State is ``(x, u)``: position ``x`` (same whitened coordinates the existing HMC
Block 3 already samples in) and a unit-norm velocity ``u``. Unlike HMC, MCLMC has
no Metropolis-Hastings accept/reject — it is an unadjusted sampler; see
``mclmc_trajectory``'s divergence guard for the only rejection mechanism.

Public API
----------
* ``isokinetic_momentum_update`` — one ESH-dynamics velocity kick given the
  log-density gradient.
* ``position_update``            — one drift step (unit mass, whitened coords).
* ``mclachlan_step``              — one full deterministic integrator step,
  composing the two above with 2-stage palindromic coefficients.
* ``partially_refresh_momentum`` — Maruyama noise step that decoheres ``u``.
* ``mclmc_trajectory``            — one full Gibbs-sweep move: half-refresh,
  ``n_steps`` deterministic steps, half-refresh, divergence guard.
"""

from typing import Callable, NamedTuple

try:
    import tensorflow as tf
except ImportError:
    tf = None

# 2-stage palindromic (Minchenko/McLachlan) composition coefficients.
_B1 = 0.1931833275037836
_A1 = 0.5
_B2 = 1.0 - 2.0 * _B1


class MclmcDiagnostics(NamedTuple):
    """Per-trajectory diagnostics for the Block 3 MCLMC step.

    ``energy_error`` : kinetic-energy change accumulated by the integrator minus
        the actual change in the target log-density over the trajectory — should
        be O(step_size^2) for a correct, non-divergent trajectory.
    ``diverged``      : whether the divergence guard rejected this trajectory
        (non-finite or excessive |energy_error|), in which case the returned
        state is the unchanged input position with a freshly resampled velocity.
    """

    energy_error: "tf.Tensor"
    diverged: "tf.Tensor"


def isokinetic_momentum_update(u, grad_log_prob, step_size, coef):
    """One ESH-dynamics velocity kick.

    Rotates the unit-norm velocity ``u`` towards the (normalized) log-density
    gradient by an amount set by the gradient's magnitude, ``step_size`` and the
    sub-step coefficient ``coef``; renormalizes to keep ``u`` unit-norm exactly.

    Returns ``(u_new, kinetic_energy_change)`` — the latter accumulated across a
    trajectory feeds the divergence-guard energy check in ``mclmc_trajectory``.
    """
    d = tf.cast(tf.size(u), u.dtype)
    grad_norm = tf.linalg.norm(grad_log_prob)
    g_hat = tf.where(
        grad_norm > tf.constant(1e-13, dtype=u.dtype),
        grad_log_prob / grad_norm,
        tf.zeros_like(grad_log_prob),
    )
    proj = tf.tensordot(u, g_hat, axes=1)
    delta = step_size * coef * grad_norm / (d - 1.0)
    zeta = tf.exp(-delta)
    u_raw = g_hat * (1.0 - zeta) * (1.0 + zeta + proj * (1.0 - zeta)) + 2.0 * zeta * u
    u_new = u_raw / tf.linalg.norm(u_raw)
    kinetic_energy_change = (
        delta - tf.math.log(tf.constant(2.0, dtype=u.dtype))
        + tf.math.log(1.0 + proj + (1.0 - proj) * zeta**2)
    ) * (d - 1.0)
    return u_new, kinetic_energy_change


def position_update(x, u, step_size, coef):
    """One drift step: ``x_new = x + step_size * coef * u`` (unit mass, whitened coords)."""
    return x + step_size * coef * u


def mclachlan_step(x, u, grad_fn: Callable, step_size):
    """One full deterministic isokinetic-leapfrog step (2-stage palindromic composition).

    ``grad_fn(x) -> (log_prob, grad)`` is called three times (kick, drift-kick,
    drift-kick), matching the existing HMC target's ``tf.GradientTape``-based
    gradient — no change to the potential/gradient itself.

    Returns ``(x_new, u_new, kinetic_energy_change)`` for this one step.
    """
    dtype = x.dtype
    b1 = tf.constant(_B1, dtype=dtype)
    a1 = tf.constant(_A1, dtype=dtype)
    b2 = tf.constant(_B2, dtype=dtype)

    _, g = grad_fn(x)
    u, ke1 = isokinetic_momentum_update(u, g, step_size, b1)
    x = position_update(x, u, step_size, a1)

    _, g = grad_fn(x)
    u, ke2 = isokinetic_momentum_update(u, g, step_size, b2)
    x = position_update(x, u, step_size, a1)

    _, g = grad_fn(x)
    u, ke3 = isokinetic_momentum_update(u, g, step_size, b1)

    return x, u, ke1 + ke2 + ke3


def partially_refresh_momentum(u, L, step_size):
    """Maruyama noise step: decoheres ``u`` towards an isotropic random direction.

    ``L`` is the momentum-decoherence scale (larger L = slower decoherence,
    closer to deterministic-only dynamics). Uses the ambient (non-stateless) TF
    RNG, consistent with how the existing HMC Block 3 path relies on TFP's
    internal RNG rather than explicit seeds.
    """
    d = tf.cast(tf.size(u), u.dtype)
    nu = tf.sqrt((tf.exp(2.0 * step_size / L) - 1.0) / d)
    z = tf.random.normal(tf.shape(u), dtype=u.dtype)
    u_raw = u + nu * z
    return u_raw / tf.linalg.norm(u_raw)


def mclmc_trajectory(
    x,
    u,
    grad_fn: Callable,
    step_size,
    L,
    n_steps: int,
    energy_error_threshold: float = 100.0,
):
    """One full Gibbs-sweep MCLMC move: half-refresh, n_steps deterministic
    steps, half-refresh, then a divergence guard (no M-H accept/reject — MCLMC
    is unadjusted).

    ``grad_fn(x) -> (log_prob, grad)`` must be the same callable used inside
    ``mclachlan_step`` (so the log-density value at trajectory endpoints is
    obtained "for free" alongside the gradients already being computed each
    sub-step).

    ``n_steps`` must be a Python int (unrolled at trace time, mirroring
    ``num_leapfrog_steps`` for HMC) so this function can be wrapped whole in a
    single ``@tf.function`` per Gibbs sweep, matching the existing eager-mode
    ``tf.py_function`` memory-leak avoidance pattern used for the HMC path.

    On divergence (non-finite or |energy_error| > ``energy_error_threshold *
    sqrt(d)``): reject to the unchanged input position with a freshly
    resampled unit-norm velocity, mirroring blackjax's high-energy-error guard.

    Returns ``(x_new, u_new, MclmcDiagnostics)``.
    """
    dtype = x.dtype
    d = tf.cast(tf.size(x), dtype)
    step_size = tf.cast(step_size, dtype)
    L = tf.cast(L, dtype)
    half_step = step_size * 0.5

    logp0, _ = grad_fn(x)
    u = partially_refresh_momentum(u, L, half_step)

    x_new, u_new = x, u
    total_ke = tf.zeros([], dtype=dtype)
    for _ in range(n_steps):
        x_new, u_new, ke = mclachlan_step(x_new, u_new, grad_fn, step_size)
        total_ke += ke

    u_new = partially_refresh_momentum(u_new, L, half_step)
    logp1, _ = grad_fn(x_new)

    energy_error = total_ke - (logp1 - logp0)
    threshold = tf.constant(energy_error_threshold, dtype=dtype) * tf.sqrt(d)
    diverged = tf.logical_or(
        tf.logical_not(tf.math.is_finite(energy_error)),
        tf.abs(energy_error) > threshold,
    )

    x_out = tf.where(diverged, x, x_new)
    u_fresh = tf.random.normal(tf.shape(u), dtype=dtype)
    u_fresh = u_fresh / tf.linalg.norm(u_fresh)
    u_out = tf.where(diverged, u_fresh, u_new)

    return x_out, u_out, MclmcDiagnostics(energy_error=energy_error, diverged=diverged)
