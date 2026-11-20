"""Experimental, biased local energies from input-log-wavefunction gradients.

Unlike GWG sampling, this estimator has no Metropolis correction. The resulting
energy and VMC update are not an exact Rayleigh quotient or its exact gradient.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .network import (
    activation_fn,
    batched_wavefunction_matrix,
    contract_periodic_mps_balanced,
    wavefunction_matrix,
)


@partial(jax.jit, static_argnames=("cfg",))
def _cached_exact_neighbor_wavefunctions(params, states, cfg):
    """Compile all fixed-size sampled-neighbor chunks as one GPU program."""
    return batched_wavefunction_matrix(params, states, cfg)


def relaxed_backflow_wavefunction(params, states, cfg):
    """Continuous FP32 extension of plain site backflow at binary vertices.

    Deliberately bypass integer gathers, BF16 quantization and experimental
    kernels: their input derivatives need not be continuous or meaningful.
    The contraction and amplitude clipping match the standard FP32 network.
    """
    if (cfg.architecture != "mps" or cfg.mps_tensor_generation != "backflow"
            or cfg.k != 1 or cfg.mps_attention_blocks or cfg.mps_residual_blocks):
        raise ValueError("Input Taylor requires plain site backflow MPS with k=1")
    x = jnp.asarray(states, dtype=jnp.float32)
    batch, sites = x.shape
    site = jnp.broadcast_to(jnp.eye(sites, dtype=x.dtype), (batch, sites, sites))
    local = jnp.stack((1.0 - x, x), axis=-1)
    full = jnp.broadcast_to(x[:, None, :], (batch, sites, sites))
    h = jnp.concatenate((site, local, full), axis=-1).reshape((batch * sites, -1))
    layers = params["mps"]["tensor_mlp"]
    for layer in layers[:-1]:
        h = h @ layer["w"].astype(jnp.float32) + layer["b"].astype(jnp.float32)
        if "ln_scale" in layer:
            h = (jax.nn.standardize(h, axis=-1)
                 * layer["ln_scale"].astype(jnp.float32)
                 + layer["ln_bias"].astype(jnp.float32))
        h = activation_fn(h, cfg.activation)
    output = h @ layers[-1]["w"].astype(jnp.float32) + layers[-1]["b"].astype(jnp.float32)
    d = cfg.mps_bond_dim
    matrices = (output[:, :d*d] + 1j * output[:, d*d:]).reshape((batch, sites, 1, d, d))
    return contract_periodic_mps_balanced(matrices, cfg.logphi_real_clip)[:, 0]


def input_log_scores(psi_fn, states):
    """Return complex d(log psi)/dx, including phase, without a log branch cut."""
    def score(x):
        def components(y):
            psi = psi_fn(y[None, :])[0]
            return jnp.stack((psi.real, psi.imag))
        value, pullback = jax.vjp(components, x)
        real_grad = pullback(jnp.array([1., 0.], dtype=value.dtype))[0]
        imag_grad = pullback(jnp.array([0., 1.], dtype=value.dtype))[0]
        psi = value[0] + 1j * value[1]
        # A node has no log derivative. Leave it invalid, never silently clip it.
        return (real_grad + 1j * imag_grad) / psi
    return jax.vmap(score)(states)


def input_log_scores_and_hessians(psi_fn, states):
    """Return d(log psi)/dx and d2(log psi)/dx2 for real occupations."""
    def derivatives(x):
        def components(y):
            psi = psi_fn(y[None, :])[0]
            return jnp.stack((psi.real, psi.imag))
        value = components(x)
        jacobian = jax.jacfwd(components)(x)
        hessian = jax.jacfwd(jax.jacrev(components))(x)
        psi = value[0] + 1j * value[1]
        grad_psi = jacobian[0] + 1j * jacobian[1]
        hess_psi = hessian[0] + 1j * hessian[1]
        score = grad_psi / psi
        log_hessian = hess_psi / psi - jnp.outer(score, score)
        return score, log_hessian
    return jax.vmap(derivatives)(states)


@jax.jit
def taylor_neighbor_ratios(states, scores, neighbor_bits, active):
    """Return row-centered exp(g.delta) ratios for packed neighbor states."""
    active = jnp.asarray(active, dtype=jnp.bool_)
    def orbital(j, delta_log):
        occupation = ((neighbor_bits >> jnp.asarray(j, jnp.uint64)) & jnp.uint64(1))
        delta = occupation.astype(states.dtype) - states[:, j, None]
        return delta_log + scores[:, j, None] * delta
    delta_log = jax.lax.fori_loop(
        0, states.shape[1], orbital,
        jnp.zeros(neighbor_bits.shape, dtype=scores.dtype),
    )
    return jnp.exp(jnp.where(active, delta_log, 0))


@jax.jit
def second_order_taylor_neighbor_ratios(
    states, scores, hessians, neighbor_bits, active
):
    orbitals = jnp.arange(states.shape[1], dtype=jnp.uint64)
    occupations = (
        (neighbor_bits[..., None] >> orbitals[None, None, :]) & jnp.uint64(1)
    ).astype(states.dtype)
    delta = occupations - states[:, None, :]
    linear = jnp.einsum("bi,bwi->bw", scores, delta)
    quadratic = jnp.einsum("bwi,bij,bwj->bw", delta, hessians, delta)
    exponent = linear + 0.5 * quadratic
    return jnp.exp(jnp.where(active, exponent, 0))


@jax.jit
def taylor_local_energy_from_scores(states, scores, diag, neighbor_bits, coeff):
    """Row-centered exp(g.delta) ratios; no neighbor neural-network forward.

    Loop over orbitals rather than materializing [chains, row_width, Nphi].
    Padding is neutralized before exponentiation to prevent 0 * inf -> NaN.
    """
    ratios = taylor_neighbor_ratios(
        states, scores, neighbor_bits, coeff != 0
    )
    return diag + jnp.sum(coeff * ratios, axis=1)


@partial(jax.jit, static_argnames=("cfg",))
def backflow_input_log_scores(params, states, cfg):
    return input_log_scores(lambda x: relaxed_backflow_wavefunction(params, x, cfg), states)


@partial(jax.jit, static_argnames=("cfg",))
def backflow_input_log_scores_and_hessians(params, states, cfg):
    return input_log_scores_and_hessians(
        lambda x: relaxed_backflow_wavefunction(params, x, cfg), states
    )


def approximate_local_energy(params, states, diag, neighbor_bits, coeff, cfg,
                             gradient_batch_size=32):
    if not jax.config.x64_enabled:
        raise ValueError("Input Taylor packed-bit evaluation requires JAX_ENABLE_X64=true")
    params = jax.lax.stop_gradient(params)
    parts = []
    # Reverse-mode contraction activations are bounded independently of the
    # much larger sampler/estimator batch. Pad the final batch to one shape.
    for start in range(0, states.shape[0], gradient_batch_size):
        x = states[start:start + gradient_batch_size]
        count = x.shape[0]
        padded = jnp.pad(x, ((0, gradient_batch_size-count), (0, 0)), mode="edge")
        parts.append(backflow_input_log_scores(params, padded, cfg)[:count])
    scores = jnp.concatenate(parts)
    eloc = taylor_local_energy_from_scores(
        states, scores, jnp.asarray(diag).reshape(-1),
        jnp.asarray(neighbor_bits, dtype=jnp.uint64).reshape((states.shape[0], -1)),
        jnp.asarray(coeff).reshape((states.shape[0], -1)),
    )
    return jax.lax.stop_gradient(eloc)


def second_order_approximate_local_energy(
    params, states, diag, neighbor_bits, coeff, cfg, hessian_batch_size=4
):
    if not jax.config.x64_enabled:
        raise ValueError("Second-order Taylor requires JAX_ENABLE_X64=true")
    params = jax.lax.stop_gradient(params)
    score_parts = []
    hessian_parts = []
    for start in range(0, states.shape[0], hessian_batch_size):
        x = states[start:start + hessian_batch_size]
        count = x.shape[0]
        padded = jnp.pad(x, ((0, hessian_batch_size-count), (0, 0)), mode="edge")
        score, hessian = backflow_input_log_scores_and_hessians(
            params, padded, cfg
        )
        score_parts.append(score[:count])
        hessian_parts.append(hessian[:count])
    scores = jnp.concatenate(score_parts)
    hessians = jnp.concatenate(hessian_parts)
    bits = jnp.asarray(neighbor_bits, dtype=jnp.uint64).reshape(
        (states.shape[0], -1)
    )
    coefficients = jnp.asarray(coeff).reshape((states.shape[0], -1))
    ratios = second_order_taylor_neighbor_ratios(
        states, scores, hessians, bits, coefficients != 0
    )
    eloc = jnp.asarray(diag).reshape(-1) + jnp.sum(coefficients * ratios, axis=1)
    return jax.lax.stop_gradient(eloc)


def _sample_control_variate_indices(
    coeff, *, samples_per_row, top_per_row, uniform_mix, seed
):
    """Choose deterministic top terms and random tail terms with replacement."""
    coeff = np.asarray(coeff)
    active = coeff != 0
    rows, width = coeff.shape
    top_mask = np.zeros((rows, width), dtype=bool)
    magnitudes = np.abs(coeff)
    for row in range(rows):
        valid = np.flatnonzero(active[row])
        count = min(int(top_per_row), valid.size)
        if count:
            order = np.argpartition(magnitudes[row, valid], -count)[-count:]
            top_mask[row, valid[order]] = True

    tail = active & ~top_mask
    weights = np.where(tail, magnitudes, 0.0)
    weight_sum = weights.sum(axis=1, keepdims=True)
    tail_count = tail.sum(axis=1, keepdims=True)
    weighted = np.divide(
        weights, weight_sum,
        out=np.zeros_like(weights, dtype=np.float64),
        where=weight_sum != 0,
    )
    uniform = np.divide(
        tail.astype(np.float64), tail_count,
        out=np.zeros_like(weights, dtype=np.float64),
        where=tail_count != 0,
    )
    q = (1.0 - float(uniform_mix)) * weighted + float(uniform_mix) * uniform

    rng = np.random.default_rng(int(seed))
    sampled = np.zeros((rows, int(samples_per_row)), dtype=np.int64)
    sampled_q = np.ones_like(sampled, dtype=np.float64)
    for row in range(rows):
        valid = np.flatnonzero(tail[row])
        if valid.size:
            probabilities = q[row, valid]
            probabilities /= probabilities.sum()
            chosen = rng.choice(
                valid, size=int(samples_per_row), replace=True, p=probabilities
            )
            sampled[row] = chosen
            sampled_q[row] = q[row, chosen]
    return top_mask, sampled, sampled_q, tail.any(axis=1)


def control_variate_local_energy(
    params,
    states,
    diag,
    neighbor_bits,
    coeff,
    cfg,
    *,
    gradient_batch_size=32,
    exact_neighbors_per_row=64,
    exact_top_neighbors=8,
    uniform_mix=0.1,
    sampling_seed=0,
):
    """Unbiased Taylor control variate with exact sampled-neighbor corrections.

    Top-|Hxy| terms are evaluated exactly. The remaining correction is an
    importance-sampled, with-replacement mean, so every active tail term has a
    nonzero inclusion probability and the estimator remains unbiased.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Taylor control variate requires JAX_ENABLE_X64=true")
    params = jax.lax.stop_gradient(params)
    states = jnp.asarray(states, dtype=jnp.float32)
    bits_np = np.asarray(neighbor_bits, dtype=np.uint64).reshape((states.shape[0], -1))
    coeff_np = np.asarray(coeff).reshape((states.shape[0], -1))
    top_mask_np, sampled_idx_np, sampled_q_np, has_tail_np = (
        _sample_control_variate_indices(
            coeff_np,
            samples_per_row=exact_neighbors_per_row,
            top_per_row=exact_top_neighbors,
            uniform_mix=uniform_mix,
            seed=sampling_seed,
        )
    )

    score_parts = []
    for start in range(0, states.shape[0], gradient_batch_size):
        x = states[start:start + gradient_batch_size]
        count = x.shape[0]
        padded = jnp.pad(x, ((0, gradient_batch_size-count), (0, 0)), mode="edge")
        score_parts.append(backflow_input_log_scores(params, padded, cfg)[:count])
    scores = jnp.concatenate(score_parts)
    bits = jnp.asarray(bits_np, dtype=jnp.uint64)
    coefficients = jnp.asarray(coeff_np, dtype=jnp.complex128)
    active = coefficients != 0
    taylor_ratio = taylor_neighbor_ratios(states, scores, bits, active)
    approximate_terms = coefficients * taylor_ratio
    estimate = jnp.asarray(diag).reshape(-1) + jnp.sum(approximate_terms, axis=1)

    row_indices, top_indices = np.nonzero(top_mask_np)
    sampled_rows = np.repeat(
        np.arange(states.shape[0], dtype=np.int64), exact_neighbors_per_row
    )
    sampled_indices = sampled_idx_np.reshape(-1)
    exact_rows = np.concatenate((row_indices.astype(np.int64), sampled_rows))
    exact_indices = np.concatenate((top_indices.astype(np.int64), sampled_indices))
    exact_bits_np = bits_np[exact_rows, exact_indices]
    shifts = jnp.arange(cfg.num_sites, dtype=jnp.uint64)
    exact_states = (
        (jnp.asarray(exact_bits_np, dtype=jnp.uint64)[:, None] >> shifts[None, :])
        & jnp.uint64(1)
    ).astype(jnp.float32)
    exact_phi = _cached_exact_neighbor_wavefunctions(
        params, exact_states, cfg
    ).astype(jnp.complex128)[:, 0]
    base_phi = wavefunction_matrix(params, states, cfg).astype(jnp.complex128)[:, 0]
    exact_ratio = exact_phi / base_phi[jnp.asarray(exact_rows)]
    exact_term = coefficients[
        jnp.asarray(exact_rows), jnp.asarray(exact_indices)
    ] * exact_ratio
    approximate_selected = approximate_terms[
        jnp.asarray(exact_rows), jnp.asarray(exact_indices)
    ]
    correction = exact_term - approximate_selected

    top_count = row_indices.size
    if top_count:
        estimate = estimate.at[jnp.asarray(row_indices)].add(correction[:top_count])
    random_correction = correction[top_count:].reshape(
        states.shape[0], exact_neighbors_per_row
    )
    random_correction = random_correction / jnp.asarray(sampled_q_np)
    random_correction = jnp.mean(random_correction, axis=1)
    estimate = estimate + jnp.where(
        jnp.asarray(has_tail_np), random_correction, 0
    )
    return jax.lax.stop_gradient(estimate)


def exact_neighbor_importance_local_energy(
    params,
    states,
    diag,
    neighbor_bits,
    coeff,
    cfg,
    *,
    exact_neighbors_per_row=64,
    exact_top_neighbors=8,
    uniform_mix=0.1,
    sampling_seed=0,
):
    """Unbiased local energy from exact top terms plus sampled exact tail terms."""
    params = jax.lax.stop_gradient(params)
    states = jnp.asarray(states, dtype=jnp.float32)
    bits_np = np.asarray(neighbor_bits, dtype=np.uint64).reshape((states.shape[0], -1))
    coeff_np = np.asarray(coeff).reshape((states.shape[0], -1))
    top_mask_np, sampled_idx_np, sampled_q_np, has_tail_np = (
        _sample_control_variate_indices(
            coeff_np,
            samples_per_row=exact_neighbors_per_row,
            top_per_row=exact_top_neighbors,
            uniform_mix=uniform_mix,
            seed=sampling_seed,
        )
    )
    row_indices, top_indices = np.nonzero(top_mask_np)
    sampled_rows = np.repeat(
        np.arange(states.shape[0], dtype=np.int64), exact_neighbors_per_row
    )
    sampled_indices = sampled_idx_np.reshape(-1)
    exact_rows = np.concatenate((row_indices.astype(np.int64), sampled_rows))
    exact_indices = np.concatenate((top_indices.astype(np.int64), sampled_indices))
    exact_bits_np = bits_np[exact_rows, exact_indices]
    shifts = jnp.arange(cfg.num_sites, dtype=jnp.uint64)
    exact_states = (
        (jnp.asarray(exact_bits_np, dtype=jnp.uint64)[:, None] >> shifts[None, :])
        & jnp.uint64(1)
    ).astype(jnp.float32)
    exact_phi = _cached_exact_neighbor_wavefunctions(
        params, exact_states, cfg
    ).astype(jnp.complex128)[:, 0]
    base_phi = wavefunction_matrix(
        params, states, cfg
    ).astype(jnp.complex128)[:, 0]
    coefficients = jnp.asarray(coeff_np, dtype=jnp.complex128)
    exact_term = coefficients[
        jnp.asarray(exact_rows), jnp.asarray(exact_indices)
    ] * exact_phi / base_phi[jnp.asarray(exact_rows)]

    estimate = jnp.asarray(diag, dtype=jnp.complex128).reshape(-1)
    top_count = row_indices.size
    if top_count:
        estimate = estimate.at[jnp.asarray(row_indices)].add(
            exact_term[:top_count]
        )
    sampled_term = exact_term[top_count:].reshape(
        states.shape[0], exact_neighbors_per_row
    )
    sampled_term = sampled_term / jnp.asarray(sampled_q_np)
    estimate = estimate + jnp.where(
        jnp.asarray(has_tail_np), jnp.mean(sampled_term, axis=1), 0
    )
    return jax.lax.stop_gradient(estimate)
