"""Projected/VMC losses and parameter update implementations."""

from __future__ import annotations

import time
import os as _os
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse_linalg
import numpy as np
from jax.flatten_util import ravel_pytree

from .config import NetworkConfig, OptimizerConfig, TrainingConfig
from .continuous import continuous_logpsi
from .network import (
    activation_fn,
    batched_backflow_delta_neighbor_wavefunction,
    batched_local_backflow_neighbor_wavefunction,
    batched_wavefunction_matrix,
    batched_wavefunction_matrix_from_positions,
    block_until_ready_tree,
    build_metric_state_batches,
    determinant_logpsi_from_positions_direct,
    determinant_sr_scalars,
    local_conv_site_wavefunction_from_bank,
    mps_block_tensor_inputs,
    mps_selected_tensor_inputs,
    normalized_overlap_condition_jax,
    occupation_states_from_positions,
    projected_energy_from_psi,
    projected_ritz_jax,
    wavefunction,
    wavefunction_matrix,
)
from .physics import SparseHamiltonianRows


@partial(jax.jit, static_argnames=("capacity", "unique_capacity"))
def _gpu_sort_deduplicate_bitstrings(
    bits, *, capacity: int, unique_capacity: int
):
    """Return fixed-shape unique bitstrings and an exact inverse map."""

    bits = jnp.asarray(bits, dtype=jnp.uint64)
    indices = jnp.arange(capacity, dtype=jnp.int32)
    sentinel = jnp.uint64(0xFFFFFFFFFFFFFFFF)
    keys = jnp.where(bits == 0, sentinel, bits)
    sorted_keys, sorted_indices = jax.lax.sort(
        (keys, indices), dimension=0, num_keys=1
    )
    starts = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            sorted_keys[1:] != sorted_keys[:-1],
        )
    )
    group_ids = jnp.cumsum(starts.astype(jnp.int32)) - 1
    unique_positions = jnp.nonzero(
        starts, size=unique_capacity, fill_value=capacity - 1
    )[0]
    unique_bits = sorted_keys[unique_positions]
    unique_bits = jnp.where(unique_bits == sentinel, 0, unique_bits)
    inverse = jnp.zeros((capacity,), dtype=jnp.int32)
    inverse = inverse.at[sorted_indices].set(group_ids)
    return unique_bits, inverse, group_ids[-1] + 1


def _check_compact_dedup_capacity(group_count, capacity: int):
    count = int(np.asarray(group_count))
    if count > capacity:
        raise ValueError(
            "GPU unique neighbor count exceeds fixed capacity: "
            f"{count} > {capacity}"
        )


@jax.jit
def _kfac_gpu_solve_diagonal(a_factor, g_factor, aug_grad, damping):
    """Apply a diagonal-output KFAC inverse without leaving the GPU."""
    a = jnp.asarray(a_factor, dtype=jnp.float32)
    g = jnp.asarray(g_factor, dtype=jnp.float32)
    grad = jnp.asarray(aug_grad, dtype=jnp.float32)
    damp = jnp.asarray(damping, dtype=jnp.float32)
    a_damped = a + damp * jnp.eye(a.shape[0], dtype=a.dtype)
    left = jnp.linalg.solve(a_damped, grad)
    return -left / (g + damp)[None, :]


@jax.jit
def _kfac_gpu_solve_full(a_factor, g_factor, aug_grad, damping):
    """Apply a full Kronecker KFAC inverse without explicit matrix inverses."""
    a = jnp.asarray(a_factor, dtype=jnp.float32)
    g = jnp.asarray(g_factor, dtype=jnp.float32)
    grad = jnp.asarray(aug_grad, dtype=jnp.float32)
    damp = jnp.asarray(damping, dtype=jnp.float32)
    a_damped = a + damp * jnp.eye(a.shape[0], dtype=a.dtype)
    g_damped = g + damp * jnp.eye(g.shape[0], dtype=g.dtype)
    left = jnp.linalg.solve(a_damped, grad)
    return -jnp.linalg.solve(g_damped.T, left.T).T


@partial(jax.jit, static_argnames=("cfg",))
def _cached_neighbor_wavefunction_matrix(params, states, cfg: NetworkConfig):
    if _os.environ.get("NQS_NEIGHBOR_FIRST_LAYER_REUSE") == "1":
        cfg = replace(cfg, mps_first_layer_backend="reuse")
    return batched_wavefunction_matrix(params, states, cfg)


@partial(jax.jit, static_argnames=("cfg", "row_count"))
def _packed_neighbor_row_sum(params, bits, coeff, cfg: NetworkConfig, row_count: int):
    """Decode only the active forward chunk; retain all fixed-width slots."""
    if _os.environ.get("NQS_NEIGHBOR_FIRST_LAYER_REUSE") == "1":
        cfg = replace(cfg, mps_first_layer_backend="reuse")
    count = bits.shape[0]
    if row_count <= 0 or count % row_count:
        raise ValueError("packed scan requires complete fixed-width rows")
    batch = min(int(cfg.forward_batch_size or count), count)
    shifts = jnp.arange(cfg.num_sites, dtype=jnp.uint64)

    def evaluate(chunk):
        states = ((chunk[:, None] >> shifts[None, :]) & jnp.uint64(1)).astype(jnp.float32)
        return wavefunction_matrix(params, states, cfg).astype(jnp.complex64)

    phi = jnp.zeros((count, cfg.k), dtype=jnp.complex64)

    def body(index, carry):
        start = index * batch
        chunk = jax.lax.dynamic_slice(bits, (start,), (batch,))
        return jax.lax.dynamic_update_slice(carry, evaluate(chunk), (start, 0))

    phi = jax.lax.fori_loop(0, count // batch, body, phi)
    tail = count % batch
    if tail:
        phi = phi.at[-tail:].set(evaluate(bits[-tail:]))
    weighted = coeff[:, None] * phi.astype(jnp.complex128)
    return weighted.reshape(row_count, count // row_count, cfg.k).sum(axis=1, dtype=jnp.complex128)


@partial(jax.jit, static_argnames=("cfg",))
def _cached_local_backflow_neighbor_wavefunction(
    params,
    center_states,
    neighbor_bitstrings,
    parent_ids,
    cfg: NetworkConfig,
    site_tensor_bank=None,
):
    """Compile the complete cached-neighbor scan as one executable."""

    return batched_local_backflow_neighbor_wavefunction(
        params,
        center_states,
        neighbor_bitstrings,
        parent_ids,
        cfg,
        site_tensor_bank=site_tensor_bank,
    )

try:
    import optax
except ImportError:  # pragma: no cover - optional dependency.
    optax = None


@dataclass
class OptimizerState:
    optimizer: object | None = None
    state: object | None = None


def _gelu_derivative(x):
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    return 0.5 * (1.0 + jax.lax.erf(x * inv_sqrt2)) + (
        x * inv_sqrt_2pi * jnp.exp(-0.5 * x * x)
    )


def _activation_derivative(x, name: str):
    if name == "tanh":
        y = jnp.tanh(x)
        return 1.0 - y * y
    if name == "gelu":
        return _gelu_derivative(x)
    if name == "leaky_relu":
        return jnp.where(x >= 0.0, 1.0, 0.01)
    if name == "silu":
        sigmoid = jax.nn.sigmoid(x)
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    if name == "poly":
        return 1.0 - x * x
    return (x >= 0.0).astype(jnp.float32)


def _layer_norm_forward_cache(x, scale, bias):
    scale = scale.astype(jnp.float32)
    bias = bias.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = jax.lax.rsqrt(var + 1e-5)
    normalized = centered * inv_std
    y = normalized * scale + bias
    return y, (normalized, inv_std, scale)


def _layer_norm_backward(dy, cache):
    normalized, inv_std, scale = cache
    dg = dy * scale
    width = dy.shape[-1]
    sum_dg = jnp.sum(dg, axis=-1, keepdims=True)
    sum_dg_norm = jnp.sum(
        dg * normalized, axis=-1, keepdims=True
    )
    return (inv_std / float(width)) * (
        float(width) * dg - sum_dg - normalized * sum_dg_norm
    )


def _mean_channel_outer(channels):
    batch_size = max(1, int(channels[0].shape[0]))
    outer = channels[0].T @ channels[0]
    for channel in channels[1:]:
        outer = outer + channel.T @ channel
    return outer / float(batch_size)


def _attention_forward_cache(h, block, num_heads: int):
    """Pre-norm MHA over the site axis, returning output plus backward cache."""
    width = int(h.shape[-1])
    if width % num_heads:
        raise ValueError("attention width must be divisible by num_heads")
    normalized, ln_cache = _layer_norm_forward_cache(
        h, block["ln_scale"], block["ln_bias"]
    )
    qkv_w = block["qkv"]["w"].astype(jnp.float32)
    qkv_b = block["qkv"]["b"].astype(jnp.float32)
    qkv = normalized @ qkv_w + qkv_b
    batch_size, num_sites, _ = qkv.shape
    head_dim = width // num_heads
    qkv4 = jnp.transpose(
        qkv.reshape((batch_size, num_sites, 3, num_heads, head_dim)),
        (2, 0, 3, 1, 4),
    )
    query, key, value = qkv4[0], qkv4[1], qkv4[2]
    logits = jnp.einsum("bhid,bhjd->bhij", query, key) * (head_dim ** -0.5)
    weights = jax.nn.softmax(logits, axis=-1)
    context = jnp.einsum("bhij,bhjd->bhid", weights, value)
    context = jnp.transpose(context, (0, 2, 1, 3)).reshape(
        (batch_size, num_sites, width)
    )
    out_w = block["out"]["w"].astype(jnp.float32)
    out_b = block["out"]["b"].astype(jnp.float32)
    output = context @ out_w + out_b
    out = h + output
    cache = {
        "normalized": normalized, "ln_cache": ln_cache,
        "qkv4": qkv4, "weights": weights, "context": context,
        "qkv_w": qkv_w, "out_w": out_w,
        "head_dim": head_dim, "num_heads": num_heads,
        "width": width, "num_sites": num_sites,
    }
    return out, cache


def _attention_backward(grad_out, cache):
    """Return (grad_h, grad_qkv, grad_out) activation gradients for one block."""
    qkv_w = cache["qkv_w"]; out_w = cache["out_w"]
    context = cache["context"]; weights = cache["weights"]
    normalized = cache["normalized"]; ln_cache = cache["ln_cache"]
    head_dim = cache["head_dim"]; num_heads = cache["num_heads"]
    width = cache["width"]; num_sites = cache["num_sites"]
    batch_size = int(context.shape[0])
    qkv4 = cache["qkv4"]
    query, key, value = qkv4[0], qkv4[1], qkv4[2]
    grad_output = grad_out
    grad_context = grad_output @ jnp.swapaxes(out_w, -1, -2)
    grad_context4 = jnp.transpose(
        grad_context.reshape((batch_size, num_sites, num_heads, head_dim)),
        (0, 2, 1, 3),
    )
    grad_value = jnp.einsum("bhij,bhid->bhjd", weights, grad_context4)
    grad_weights = jnp.einsum("bhid,bhjd->bhij", grad_context4, value)
    grad_logits = weights * (
        grad_weights - jnp.sum(grad_weights * weights, axis=-1, keepdims=True)
    )
    scale = head_dim ** -0.5
    grad_q = jnp.einsum("bhij,bhjd->bhid", grad_logits, key) * scale
    grad_k = jnp.einsum("bhij,bhid->bhjd", grad_logits, query) * scale
    grad_qkv4 = jnp.stack((grad_q, grad_k, grad_value), axis=0)
    grad_qkv = jnp.transpose(grad_qkv4, (1, 3, 0, 2, 4)).reshape(
        (batch_size * num_sites, 3 * width)
    )
    grad_normalized = (grad_qkv @ jnp.swapaxes(qkv_w, -1, -2)).reshape(
        (batch_size, num_sites, width)
    )
    grad_h_norm = _layer_norm_backward(grad_normalized, ln_cache)
    return grad_out + grad_h_norm, grad_qkv, grad_out


def _activation_second_moment(activations):
    batch_size = max(1, int(activations.shape[0]))
    ones = jnp.ones((activations.shape[0], 1), dtype=jnp.float32)
    augmented = jnp.concatenate((activations, ones), axis=1)
    return (augmented.T @ augmented) / float(batch_size)


def _kfac_factors_from_output_channels(
    layer_params,
    inputs,
    output_delta_channels,
    activation: str,
    diagonal_output: bool = False,
):
    """Build dense-layer KFAC factors and return input score channels."""
    h = jnp.asarray(inputs, dtype=jnp.float32)
    activations_in = []
    hidden_caches = []
    for layer in layer_params[:-1]:
        w = layer["w"].astype(jnp.float32)
        b = layer["b"].astype(jnp.float32)
        activations_in.append(h)
        pre_linear = h @ w + b
        activation_input = pre_linear
        ln_cache = None
        if "ln_scale" in layer:
            activation_input, ln_cache = _layer_norm_forward_cache(
                pre_linear, layer["ln_scale"], layer["ln_bias"]
            )
        hidden_caches.append(
            {
                "w": w,
                "activation_input": activation_input,
                "ln_cache": ln_cache,
            }
        )
        h = activation_fn(activation_input, activation).astype(jnp.float32)

    final_w = layer_params[-1]["w"].astype(jnp.float32)
    activations_in.append(h)
    delta_channels = [
        jnp.asarray(delta, dtype=jnp.float32)
        for delta in output_delta_channels
    ]
    factor_count = len(layer_params)
    a_factors = [None] * factor_count
    g_factors = [None] * factor_count
    a_factors[-1] = _activation_second_moment(activations_in[-1])
    if diagonal_output:
        g_factors[-1] = sum(
            jnp.mean(delta * delta, axis=0) for delta in delta_channels
        )
    else:
        g_factors[-1] = _mean_channel_outer(delta_channels)

    back_channels = [delta @ final_w.T for delta in delta_channels]
    for layer_index in range(factor_count - 2, -1, -1):
        cache = hidden_caches[layer_index]
        derivative = _activation_derivative(
            cache["activation_input"], activation
        )
        layer_delta_channels = []
        for back in back_channels:
            delta = back * derivative
            if cache["ln_cache"] is not None:
                delta = _layer_norm_backward(delta, cache["ln_cache"])
            layer_delta_channels.append(delta)
        a_factors[layer_index] = _activation_second_moment(
            activations_in[layer_index]
        )
        g_factors[layer_index] = _mean_channel_outer(layer_delta_channels)
        back_channels = [
            delta @ cache["w"].T for delta in layer_delta_channels
        ]
    return a_factors, g_factors, back_channels


def _kfac_factors_mlp(pref_params, metric_states, net_cfg: NetworkConfig):
    state_batches = jnp.asarray(metric_states, dtype=jnp.float32)
    if state_batches.ndim == 2:
        state_batches = state_batches[:, None, :]
    if state_batches.ndim != 3 or state_batches.shape[1] != net_cfg.k:
        raise ValueError(
            "KFAC metric states must have shape [batch, k, num_sites]."
        )
    chain_count, k, num_sites = state_batches.shape
    states = state_batches.reshape((chain_count * k, num_sites))
    hidden_count = max(0, len(pref_params) - 1)
    activations_in = []
    hidden_caches = []
    h = states
    residual_flags = []
    residual_budget = int(getattr(net_cfg, "mps_residual_blocks", 0) or 0)
    residual_used = 0
    for layer in pref_params[:-1]:
        w = layer["w"].astype(jnp.float32)
        b = layer["b"].astype(jnp.float32)
        skip = h
        activations_in.append(h)
        pre_linear = h @ w + b
        activation_input = pre_linear
        ln_cache = None
        if "ln_scale" in layer:
            activation_input, ln_cache = _layer_norm_forward_cache(
                pre_linear, layer["ln_scale"], layer["ln_bias"]
            )
        hidden_caches.append(
            {
                "w": w,
                "activation_input": activation_input,
                "ln_cache": ln_cache,
            }
        )
        h = activation_fn(activation_input, net_cfg.activation).astype(
            jnp.float32
        )
        use_res = (
            residual_used < residual_budget
            and skip.shape[-1] == h.shape[-1]
        )
        if use_res:
            h = h + skip
            residual_used += 1
        residual_flags.append(use_res)

    final_layer = pref_params[-1]
    final_w = final_layer["w"].astype(jnp.float32)
    final_b = final_layer["b"].astype(jnp.float32)
    activations_in.append(h)
    out = h @ final_w + final_b
    log_amp_raw = out[:, :k]
    phase = out[:, k:]
    clip_mask = (
        jnp.abs(log_amp_raw) < float(net_cfg.logphi_real_clip)
    ).astype(jnp.float32)
    log_amp = jnp.clip(
        log_amp_raw,
        -float(net_cfg.logphi_real_clip),
        float(net_cfg.logphi_real_clip),
    )
    phi = jnp.exp(log_amp + 1j * phase).reshape(
        (chain_count, k, k)
    )
    inverse_transpose = jnp.swapaxes(jnp.linalg.inv(phi), -1, -2)
    determinant_score = (phi * inverse_transpose).reshape((-1, k))
    delta_channels = [
        jnp.concatenate(
            (
                jnp.real(determinant_score) * clip_mask,
                -jnp.imag(determinant_score),
            ),
            axis=1,
        ).astype(jnp.float32),
        jnp.concatenate(
            (
                jnp.imag(determinant_score) * clip_mask,
                jnp.real(determinant_score),
            ),
            axis=1,
        ).astype(jnp.float32),
    ]

    a_factors = [None] * (hidden_count + 1)
    g_factors = [None] * (hidden_count + 1)
    a_factors[-1] = _activation_second_moment(activations_in[-1])
    g_factors[-1] = _mean_channel_outer(delta_channels)

    back_channels = [delta @ final_w.T for delta in delta_channels]
    for layer_index in range(hidden_count - 1, -1, -1):
        cache = hidden_caches[layer_index]
        derivative = _activation_derivative(
            cache["activation_input"], net_cfg.activation
        )
        layer_delta_channels = []
        for back in back_channels:
            delta = back * derivative
            if cache["ln_cache"] is not None:
                delta = _layer_norm_backward(delta, cache["ln_cache"])
            layer_delta_channels.append(delta)
        a_factors[layer_index] = _activation_second_moment(
            activations_in[layer_index]
        )
        g_factors[layer_index] = _mean_channel_outer(
            layer_delta_channels
        )
        weight_back = [
            delta @ cache["w"].T for delta in layer_delta_channels
        ]
        if residual_flags[layer_index]:
            weight_back = [
                wb + cur for wb, cur in zip(weight_back, back_channels)
            ]
        back_channels = weight_back
    return a_factors, g_factors


def _mps_logpsi_output_score(matrices, logphi_real_clip: float):
    """Return output scores for ``log Tr[A_0 A_1 ... A_(N-1)]``."""
    batch_size, num_sites, bond_dim, _ = matrices.shape
    identity = jnp.broadcast_to(
        jnp.eye(bond_dim, dtype=matrices.dtype)[None, :, :],
        (batch_size, bond_dim, bond_dim),
    )

    left_environments = []
    left = identity
    log_scale = jnp.zeros((batch_size,), dtype=jnp.float32)
    for site in range(num_sites):
        left_environments.append(left)
        updated = jnp.einsum("bij,bjk->bik", left, matrices[:, site])
        norm = jnp.sqrt(jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)))
        safe_norm = jnp.maximum(norm, 1e-20)
        left = updated / safe_norm[:, None, None]
        log_scale = log_scale + jnp.log(safe_norm)
    final_amplitude = jnp.trace(left, axis1=-2, axis2=-1)
    raw_logabs = log_scale + jnp.log(
        jnp.maximum(jnp.abs(final_amplitude), 1e-20)
    )
    amplitude_clip_mask = (
        jnp.abs(raw_logabs) < float(logphi_real_clip)
    ).astype(jnp.float32)

    right_environments = [None] * num_sites
    right = identity
    for site in range(num_sites - 1, -1, -1):
        right_environments[site] = right
        updated = jnp.einsum(
            "bij,bjk->bik", matrices[:, site], right
        )
        norm = jnp.sqrt(jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)))
        right = updated / jnp.maximum(norm, 1e-20)[:, None, None]

    scores = []
    for site in range(num_sites):
        left = left_environments[site]
        right = right_environments[site]
        denominator = jnp.einsum(
            "bii->b",
            jnp.einsum(
                "bij,bjk,bkl->bil", left, matrices[:, site], right
            ),
        )
        safe_denominator = jnp.where(
            jnp.abs(denominator) > 1e-20,
            denominator,
            jnp.asarray(1e-20 + 0.0j, dtype=denominator.dtype),
        )
        scores.append(
            jnp.swapaxes(
                jnp.einsum("bij,bjk->bik", right, left), -1, -2
            ) / safe_denominator[:, None, None]
        )

    score = jnp.stack(scores, axis=1).reshape(
        (batch_size * num_sites, bond_dim * bond_dim)
    )
    clip_mask = jnp.repeat(amplitude_clip_mask, num_sites)[:, None]
    real_channel = jnp.concatenate(
        (jnp.real(score) * clip_mask, -jnp.imag(score)), axis=1
    ).astype(jnp.float32)
    imag_channel = jnp.concatenate(
        (jnp.imag(score) * clip_mask, jnp.real(score)), axis=1
    ).astype(jnp.float32)
    return real_channel, imag_channel


# kfac_mps_res_done
def _kfac_factors_mps(
    tensor_mlp_params,
    metric_states,
    net_cfg: NetworkConfig,
    attention_params=None,
    local_params=None,
    site_bank=None,
    site_bank_cache=None,
):
    if net_cfg.k != 1:
        raise ValueError("MPS KFAC currently supports only k=1.")
    state_batches = jnp.asarray(metric_states, dtype=jnp.float32)
    if state_batches.ndim == 3:
        if state_batches.shape[1] != 1:
            raise ValueError("MPS KFAC metric states must have k=1.")
        states = state_batches[:, 0, :]
    elif state_batches.ndim == 2:
        states = state_batches
    else:
        raise ValueError(
            "MPS KFAC metric states must have shape [batch, 1, num_sites]."
        )

    batch_size, num_sites = states.shape
    global_generation = (
        net_cfg.mps_tensor_generation == "global_backflow"
    )
    block_generation = net_cfg.mps_tensor_generation == "block_backflow"
    local_generation = net_cfg.mps_tensor_generation in {
        "local_backflow", "local_conv_backflow"
    }
    if global_generation:
        h = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.float32)
    elif block_generation:
        block_size = int(net_cfg.mps_block_size)
        n_blocks = num_sites // block_size
        h = mps_block_tensor_inputs(states, block_size)
        h = h.reshape((batch_size * n_blocks, h.shape[-1]))
    elif site_bank_cache is not None:
        if (net_cfg.mps_tensor_generation != "local_conv_backflow"
                or attention_params is not None
                or int(getattr(net_cfg, "mps_residual_blocks", 0) or 0)):
            raise ValueError("site-bank KFAC supports plain kernel-one local MLP")
        occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
        sites = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
        h = site_bank_cache["activations"][0][sites, occupancy].reshape(
            (batch_size * num_sites, -1)
        )
    elif local_generation:
        if local_params is None:
            raise ValueError("local-backflow KFAC requires local parameters")
        occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
        sites = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
        embedded = local_params["embedding"][sites, occupancy]
        if net_cfg.mps_tensor_generation == "local_conv_backflow":
            kernel_size = int(local_params["conv"]["kernel"].shape[0])
            radius = kernel_size // 2
            windows = tuple(
                jnp.roll(embedded, shift, axis=1)
                for shift in range(radius, -radius - 1, -1)
            )
            stacked = jnp.stack(windows, axis=2)
            conv = local_params["conv"]
            h = embedded + activation_fn(
                jnp.einsum(
                    "bnwe,weh->bnh",
                    stacked.astype(conv["kernel"].dtype),
                    conv["kernel"],
                ) + conv["b"],
                net_cfg.activation,
            )
        else:
            windows = (
                jnp.roll(embedded, 1, axis=1),
                embedded,
                jnp.roll(embedded, -1, axis=1),
            )
            h = jnp.concatenate(windows, axis=-1)
        h = h.reshape((batch_size * num_sites, h.shape[-1]))
    else:
        h = mps_selected_tensor_inputs(
            states, net_cfg.mps_tensor_generation
        )
        h = h.reshape((batch_size * num_sites, h.shape[-1]))

    hidden_count = max(0, len(tensor_mlp_params) - 1)
    activations_in = []
    hidden_caches = []
    residual_flags = []
    residual_budget = int(getattr(net_cfg, "mps_residual_blocks", 0) or 0)
    residual_used = 0
    attention_budget = 0 if attention_params is None else len(attention_params)
    attention_index = 0
    attention_used = []
    attention_caches = []
    for layer_index, layer in enumerate(tensor_mlp_params[:-1]):
        w = layer["w"].astype(jnp.float32)
        b = layer["b"].astype(jnp.float32)
        if site_bank_cache is not None:
            h = site_bank_cache["activations"][layer_index][sites, occupancy].reshape(
                (batch_size * num_sites, -1)
            )
            activation_input = site_bank_cache["activation_inputs"][layer_index][sites, occupancy].reshape(
                (batch_size * num_sites, -1)
            )
            cached_ln = site_bank_cache["ln_caches"][layer_index]
            ln_cache = None if cached_ln is None else (
                cached_ln[0][sites, occupancy].reshape((batch_size * num_sites, -1)),
                cached_ln[1][sites, occupancy].reshape((batch_size * num_sites, -1)),
                cached_ln[2],
            )
        else:
            pre_linear = h @ w + b
            activation_input = pre_linear
            ln_cache = None
            if "ln_scale" in layer:
                activation_input, ln_cache = _layer_norm_forward_cache(
                    pre_linear, layer["ln_scale"], layer["ln_bias"]
                )
        skip = h
        activations_in.append(h)
        hidden_caches.append(
            {
                "w": w,
                "activation_input": activation_input,
                "ln_cache": ln_cache,
            }
        )
        h = (site_bank_cache["activations"][layer_index + 1][sites, occupancy].reshape(
            (batch_size * num_sites, -1)
        ) if site_bank_cache is not None else activation_fn(
            activation_input, net_cfg.activation
        ).astype(jnp.float32))
        use_res = (
            residual_used < residual_budget
            and skip.shape[-1] == h.shape[-1]
        )
        if use_res:
            h = h + skip
            residual_used += 1
        residual_flags.append(use_res)
        if attention_index < attention_budget:
            seq = h.reshape((batch_size, num_sites, -1))
            seq_out, attn_cache = _attention_forward_cache(
                seq, attention_params[attention_index],
                net_cfg.mps_attention_heads,
            )
            h = seq_out.reshape((batch_size * num_sites, -1))
            attention_caches.append(attn_cache)
            attention_used.append(True)
            attention_index += 1
        else:
            attention_used.append(False)

    final_layer = tensor_mlp_params[-1]
    final_w = final_layer["w"].astype(jnp.float32)
    final_b = final_layer["b"].astype(jnp.float32)
    activations_in.append(h)
    output = (site_bank[sites, occupancy].reshape((batch_size * num_sites, -1))
              if site_bank_cache is not None else h @ final_w + final_b)
    matrix_size = net_cfg.mps_bond_dim * net_cfg.mps_bond_dim
    if global_generation:
        site_output = output.reshape(
            (batch_size, num_sites, 2 * matrix_size)
        )
        matrices = (
            site_output[..., :matrix_size]
            + 1j * site_output[..., matrix_size:]
        ).reshape(
            (
                batch_size,
                num_sites,
                net_cfg.mps_bond_dim,
                net_cfg.mps_bond_dim,
            )
        )
        local_delta_channels = _mps_logpsi_output_score(
            matrices, net_cfg.logphi_real_clip
        )
        delta_channels = [
            delta.reshape((batch_size, num_sites, 2 * matrix_size)).reshape(
                (batch_size, num_sites * 2 * matrix_size)
            )
            for delta in local_delta_channels
        ]
    elif block_generation:
        block_size = int(net_cfg.mps_block_size)
        n_blocks = num_sites // block_size
        block_output = output.reshape(
            (batch_size, n_blocks, block_size, 2 * matrix_size)
        )
        matrices = (
            block_output[..., :matrix_size]
            + 1j * block_output[..., matrix_size:]
        ).reshape(
            (
                batch_size,
                num_sites,
                net_cfg.mps_bond_dim,
                net_cfg.mps_bond_dim,
            )
        )
        local_delta = _mps_logpsi_output_score(
            matrices, net_cfg.logphi_real_clip
        )
        delta_channels = [
            delta.reshape(
                (batch_size, n_blocks, block_size, 2 * matrix_size)
            ).reshape((batch_size * n_blocks, block_size * 2 * matrix_size))
            for delta in local_delta
        ]
    else:
        matrices = (
            output[:, :matrix_size] + 1j * output[:, matrix_size:]
        ).reshape(
            (
                batch_size,
                num_sites,
                net_cfg.mps_bond_dim,
                net_cfg.mps_bond_dim,
            )
        )
        delta_channels = list(
            _mps_logpsi_output_score(matrices, net_cfg.logphi_real_clip)
        )

    a_factors = [None] * (hidden_count + 1)
    g_factors = [None] * (hidden_count + 1)
    a_factors[-1] = _activation_second_moment(activations_in[-1])
    # A diagonal output factor keeps D=32 KFAC practical while retaining the
    # Kronecker factor for the 400-dimensional incoming activation.
    g_factors[-1] = sum(
        jnp.mean(delta * delta, axis=0) for delta in delta_channels
    )

    attention_a_factors = []
    attention_g_factors = []
    back_channels = [delta @ final_w.T for delta in delta_channels]
    for layer_index in range(hidden_count - 1, -1, -1):
        if attention_used[layer_index]:
            attn_cache = attention_caches[layer_index]
            width = attn_cache["width"]
            new_back = []
            for back in back_channels:
                grad_h, grad_qkv, grad_out = _attention_backward(
                    back.reshape((batch_size, num_sites, -1)), attn_cache
                )
                new_back.append(grad_h.reshape(back.shape))
                attention_a_factors.append(
                    _activation_second_moment(
                        attn_cache["normalized"].reshape((-1, width))
                    )
                )
                attention_g_factors.append(
                    jnp.mean(grad_qkv.reshape((-1, 3 * width)) ** 2, axis=0)
                )
                attention_a_factors.append(
                    _activation_second_moment(
                        attn_cache["context"].reshape((-1, width))
                    )
                )
                attention_g_factors.append(
                    jnp.mean(grad_out.reshape((-1, width)) ** 2, axis=0)
                )
            back_channels = new_back
        cache = hidden_caches[layer_index]
        derivative = _activation_derivative(
            cache["activation_input"], net_cfg.activation
        )
        layer_delta_channels = []
        for back in back_channels:
            delta = back * derivative
            if cache["ln_cache"] is not None:
                delta = _layer_norm_backward(delta, cache["ln_cache"])
            layer_delta_channels.append(delta)
        a_factors[layer_index] = _activation_second_moment(
            activations_in[layer_index]
        )
        g_factors[layer_index] = _mean_channel_outer(layer_delta_channels)
        weight_back = [
            delta @ cache["w"].T for delta in layer_delta_channels
        ]
        if residual_flags[layer_index]:
            weight_back = [
                wb + cur for wb, cur in zip(weight_back, back_channels)
            ]
        back_channels = weight_back
    return a_factors + attention_a_factors, g_factors + attention_g_factors


@partial(jax.jit, static_argnames=("net_cfg",))
def _kfac_factors_mps_jit(
    tensor_mlp_params,
    metric_states,
    net_cfg: NetworkConfig,
    attention_params=None,
    local_params=None,
    site_bank=None,
    site_bank_cache=None,
):
    """Compile the complete MPS KFAC factor pass as one accelerator program."""

    return _kfac_factors_mps(
        tensor_mlp_params,
        metric_states,
        net_cfg,
        attention_params=attention_params,
        local_params=local_params,
        site_bank=site_bank,
        site_bank_cache=site_bank_cache,
    )


def _kfac_factors_mps_site_head(params, metric_states, net_cfg: NetworkConfig):
    """KFAC blocks for the global trunk and lightweight FiLM site head."""
    state_batches = jnp.asarray(metric_states, dtype=jnp.float32)
    if state_batches.ndim == 3:
        if state_batches.shape[1] != 1:
            raise ValueError("Site-head KFAC metric states must have k=1.")
        states = state_batches[:, 0, :]
    elif state_batches.ndim == 2:
        states = state_batches
    else:
        raise ValueError(
            "Site-head KFAC metric states must have shape [batch,1,num_sites]."
        )

    batch_size, num_sites = states.shape
    trunk = params["trunk"]
    head = params["head"]

    h = states
    for layer in trunk[:-1]:
        pre_linear = h @ layer["w"].astype(jnp.float32) + layer["b"].astype(
            jnp.float32
        )
        if "ln_scale" in layer:
            pre_linear, _ = _layer_norm_forward_cache(
                pre_linear, layer["ln_scale"], layer["ln_bias"]
            )
        h = activation_fn(pre_linear, net_cfg.activation).astype(jnp.float32)
    global_latent = (
        h @ trunk[-1]["w"].astype(jnp.float32)
        + trunk[-1]["b"].astype(jnp.float32)
    )

    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    site_indices = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    gamma = params["gamma"][site_indices, occupancy].astype(jnp.float32)
    beta = params["beta"][site_indices, occupancy].astype(jnp.float32)
    film_pre = gamma * global_latent[:, None, :] + beta
    site_latent = activation_fn(film_pre, net_cfg.activation).astype(jnp.float32)
    latent_dim = int(site_latent.shape[-1])

    head_input = site_latent.reshape((batch_size * num_sites, latent_dim))
    head_hidden = head_input
    for layer in head[:-1]:
        pre_linear = (
            head_hidden @ layer["w"].astype(jnp.float32)
            + layer["b"].astype(jnp.float32)
        )
        if "ln_scale" in layer:
            pre_linear, _ = _layer_norm_forward_cache(
                pre_linear, layer["ln_scale"], layer["ln_bias"]
            )
        head_hidden = activation_fn(
            pre_linear, net_cfg.activation
        ).astype(jnp.float32)
    output = (
        head_hidden @ head[-1]["w"].astype(jnp.float32)
        + head[-1]["b"].astype(jnp.float32)
    )
    matrix_size = net_cfg.mps_bond_dim * net_cfg.mps_bond_dim
    matrices = (
        output[:, :matrix_size] + 1j * output[:, matrix_size:]
    ).reshape(
        (
            batch_size,
            num_sites,
            net_cfg.mps_bond_dim,
            net_cfg.mps_bond_dim,
        )
    )
    output_delta_channels = list(
        _mps_logpsi_output_score(matrices, net_cfg.logphi_real_clip)
    )
    head_a, head_g, head_input_channels = _kfac_factors_from_output_channels(
        head,
        head_input,
        output_delta_channels,
        net_cfg.activation,
        diagonal_output=True,
    )

    film_derivative = _activation_derivative(film_pre, net_cfg.activation)
    film_delta_channels = [
        channel.reshape((batch_size, num_sites, latent_dim))
        * film_derivative
        for channel in head_input_channels
    ]
    trunk_output_channels = [
        jnp.sum(delta * gamma, axis=1) for delta in film_delta_channels
    ]
    trunk_a, trunk_g, _ = _kfac_factors_from_output_channels(
        trunk,
        states,
        trunk_output_channels,
        net_cfg.activation,
    )

    occupation_mask = jax.nn.one_hot(
        occupancy, 2, dtype=jnp.float32
    )[..., None]
    beta_fisher = sum(
        jnp.mean(
            occupation_mask * delta[:, :, None, :] ** 2,
            axis=0,
        )
        for delta in film_delta_channels
    )
    gamma_fisher = sum(
        jnp.mean(
            occupation_mask
            * (
                delta[:, :, None, :]
                * global_latent[:, None, None, :]
            )
            ** 2,
            axis=0,
        )
        for delta in film_delta_channels
    )
    return trunk_a, trunk_g, head_a, head_g, gamma_fisher, beta_fisher


def _solve_kfac_blocks(
    layer_params,
    layer_grads,
    a_factors,
    g_factors,
    damping: float,
    report_condition: bool,
):
    factor_damping = np.sqrt(damping)
    deltas = []
    condition_max = np.nan
    for layer, layer_grad, a_factor, g_factor in zip(
        layer_params, layer_grads, a_factors, g_factors
    ):
        a_np = np.asarray(a_factor, dtype=np.float64)
        g_np = np.asarray(g_factor, dtype=np.float64)
        a_damped = a_np + factor_damping * np.eye(a_np.shape[0])
        if g_np.ndim == 1:
            g_damped = g_np + factor_damping
        else:
            g_damped = g_np + factor_damping * np.eye(g_np.shape[0])
        if report_condition:
            g_condition = (
                float(np.max(g_damped) / np.min(g_damped))
                if g_np.ndim == 1
                else float(np.linalg.cond(g_damped))
            )
            block_condition = np.linalg.cond(a_damped) * g_condition
            if np.isfinite(block_condition):
                condition_max = (
                    float(block_condition)
                    if np.isnan(condition_max)
                    else max(float(condition_max), float(block_condition))
                )
        a_inv = jnp.asarray(np.linalg.inv(a_damped), dtype=jnp.float32)
        grad_w = jnp.asarray(layer_grad["w"], dtype=jnp.float32)
        grad_b = jnp.asarray(layer_grad["b"], dtype=jnp.float32)
        aug_grad = jnp.concatenate((grad_w, grad_b[None, :]), axis=0)
        if g_np.ndim == 1:
            g_inv = jnp.asarray(1.0 / g_damped, dtype=jnp.float32)
            aug_delta = -(a_inv @ aug_grad) * g_inv[None, :]
        else:
            g_inv = jnp.asarray(np.linalg.inv(g_damped), dtype=jnp.float32)
            aug_delta = -(a_inv @ aug_grad @ g_inv)
        layer_delta = {
            "w": aug_delta[:-1, :],
            "b": aug_delta[-1, :],
        }
        if "ln_scale" in layer_grad:
            layer_delta["ln_scale"] = -jnp.asarray(
                layer_grad["ln_scale"], dtype=jnp.float32
            )
            layer_delta["ln_bias"] = -jnp.asarray(
                layer_grad["ln_bias"], dtype=jnp.float32
            )
        deltas.append(layer_delta)
    return deltas, condition_max


def site_head_backflow_kfac_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    metric_positions=None,
):
    """Blockwise KFAC for trunk/head with diagonal FiLM Fisher blocks."""
    total_start = time.perf_counter()
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(metric_positions, dtype=jnp.int64)
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )

    factor_start = time.perf_counter()
    mps_params = params["mps"]
    factors = _kfac_factors_mps_site_head(
        mps_params, metric_states, net_cfg
    )
    jastrow_factors = None
    if net_cfg.mps_jastrow:
        jastrow_factors = _kfac_factors_mlp(
            mps_params["jastrow_mlp"], metric_states, net_cfg
        )
    factors, jastrow_factors = block_until_ready_tree(
        (factors, jastrow_factors)
    )
    time_factors = time.perf_counter() - factor_start

    solve_start = time.perf_counter()
    damping = max(float(opt_cfg.sr_diag_shift), 1e-8)
    trunk_a, trunk_g, head_a, head_g, gamma_fisher, beta_fisher = factors
    mps_grads = grad_tree["mps"]
    trunk_delta, trunk_condition = _solve_kfac_blocks(
        mps_params["trunk"],
        mps_grads["trunk"],
        trunk_a,
        trunk_g,
        damping,
        opt_cfg.report_fisher_condition,
    )
    head_delta, head_condition = _solve_kfac_blocks(
        mps_params["head"],
        mps_grads["head"],
        head_a,
        head_g,
        damping,
        opt_cfg.report_fisher_condition,
    )
    jastrow_delta = None
    jastrow_condition = np.nan
    if jastrow_factors is not None:
        jastrow_delta, jastrow_condition = _solve_kfac_blocks(
            mps_params["jastrow_mlp"],
            mps_grads["jastrow_mlp"],
            jastrow_factors[0],
            jastrow_factors[1],
            damping,
            opt_cfg.report_fisher_condition,
        )
    delta_tree = {
        "mps": {
            "trunk": trunk_delta,
            "gamma": -jnp.asarray(
                mps_grads["gamma"], dtype=jnp.float32
            )
            / (jnp.asarray(gamma_fisher, dtype=jnp.float32) + damping),
            "beta": -jnp.asarray(
                mps_grads["beta"], dtype=jnp.float32
            )
            / (jnp.asarray(beta_fisher, dtype=jnp.float32) + damping),
            "head": head_delta,
        }
    }
    if jastrow_delta is not None:
        delta_tree["mps"]["jastrow_mlp"] = jastrow_delta
    delta_tree = block_until_ready_tree(delta_tree)
    time_solve = time.perf_counter() - solve_start

    grad_flat, _ = ravel_pytree(grad_tree)
    delta_flat, _ = ravel_pytree(delta_tree)
    grad_norm = float(jnp.linalg.norm(grad_flat.astype(jnp.float64)))
    delta_norm = float(jnp.linalg.norm(delta_flat.astype(jnp.float64)))
    trust_scale = 1.0
    if opt_cfg.max_sr_step_norm > 0 and delta_norm > opt_cfg.max_sr_step_norm:
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta_tree = jax.tree_util.tree_map(
            lambda value: value * trust_scale, delta_tree
        )
        delta_flat, _ = ravel_pytree(delta_tree)
    lr = learning_rate_at_step(opt_cfg, step, total_steps)

    update_start = time.perf_counter()
    new_params = jax.tree_util.tree_map(
        lambda param, delta: (
            param.astype(jnp.float32) + lr * delta.astype(jnp.float32)
        ).astype(param.dtype),
        params,
        delta_tree,
    )
    new_params = block_until_ready_tree(new_params)
    time_update = time.perf_counter() - update_start
    finite_conditions = [
        value
        for value in (trunk_condition, head_condition, jastrow_condition)
        if np.isfinite(value)
    ]
    return new_params, {
        "optimizer_impl": "kfac",
        "kfac_variant": "mps_site_head_blockwise_diag_film",
        "lr": lr,
        "grad_norm": grad_norm,
        "delta_norm": float(jnp.linalg.norm(delta_flat)),
        "applied_update_norm": float(jnp.linalg.norm(lr * delta_flat)),
        "trust_region_scale": float(trust_scale),
        "kfac_cond_max": (
            np.nan if not finite_conditions else max(finite_conditions)
        ),
        "time_factors": time_factors,
        "time_solve": time_solve,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def kfac_natural_gradient_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    metric_positions=None,
    metric_batch_size: int | None = None,
    site_bank=None,
    site_bank_cache=None,
):
    if net_cfg.architecture not in {"mlp", "mps"}:
        raise ValueError(
            "KFAC natural_gradient currently supports only mlp and mps "
            "architectures."
        )
    if net_cfg.architecture == "mps" and net_cfg.k != 1:
        raise ValueError("MPS KFAC currently supports only k=1.")

    total_start = time.perf_counter()
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(
            metric_positions, dtype=jnp.int64
        )
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )
    factor_start = time.perf_counter()
    if net_cfg.architecture == "mlp":
        layer_params = params["pref"]
        layer_grads = grad_tree["pref"]
        factor_fn = _kfac_factors_mlp
        kfac_variant = "dense_mlp"
    else:
        local_generation = net_cfg.mps_tensor_generation in {
            "local_backflow", "local_conv_backflow"
        }
        mlp_key = "local_mlp" if local_generation else "tensor_mlp"
        layer_params = list(params["mps"][mlp_key])
        layer_grads = list(grad_tree["mps"][mlp_key])
        attention_block_params = None
        if net_cfg.mps_attention_blocks and "attention_blocks" in params["mps"]:
            attention_block_params = params["mps"]["attention_blocks"]
            attention_grads = grad_tree["mps"]["attention_blocks"]
            for block_p, block_g in zip(
                attention_block_params, attention_grads
            ):
                layer_params.append(block_p["qkv"])
                layer_grads.append(block_g["qkv"])
                layer_params.append(block_p["out"])
                layer_grads.append(block_g["out"])
        factor_fn = (
            _kfac_factors_mps_jit
            if _os.environ.get("NQS_KFAC_JIT_FACTORS") == "1"
            else _kfac_factors_mps
        )
        kfac_variant = (
            "mps_local_backflow_hybrid"
            if local_generation
            else (
            "mps_global_backflow_mlp_diag_output"
            if net_cfg.mps_tensor_generation == "global_backflow"
            else (
                "mps_backflow_tensor_mlp_diag_output"
                if net_cfg.mps_tensor_generation == "backflow"
                else "mps_shared_tensor_mlp_diag_output"
            )
            )
        )

    jastrow_factors = None

    total_metric = int(metric_states.shape[0])
    factor_batch = (
        total_metric
        if metric_batch_size is None
        else min(int(metric_batch_size), total_metric)
    )
    factor_kwargs = (
        {}
        if net_cfg.architecture == "mlp"
        else {
            "attention_params": attention_block_params,
            "local_params": params["mps"] if local_generation else None,
            "site_bank": site_bank,
            "site_bank_cache": site_bank_cache,
        }
    )
    if factor_batch >= total_metric:
        a_factors, g_factors = factor_fn(
            layer_params, metric_states, net_cfg, **factor_kwargs,
        )
        a_factors, g_factors = block_until_ready_tree(
            (a_factors, g_factors)
        )
    else:
        a_sums = None
        g_sums = None
        for start in range(0, total_metric, factor_batch):
            stop = min(start + factor_batch, total_metric)
            weight = stop - start
            a_chunk, g_chunk = factor_fn(
                layer_params, metric_states[start:stop],
                net_cfg, **factor_kwargs,
            )
            a_chunk, g_chunk = block_until_ready_tree((a_chunk, g_chunk))
            a_np = [np.asarray(value, dtype=np.float64) for value in a_chunk]
            g_np = [np.asarray(value, dtype=np.float64) for value in g_chunk]
            if a_sums is None:
                a_sums = [value * weight for value in a_np]
                g_sums = [value * weight for value in g_np]
            else:
                for index in range(len(a_sums)):
                    a_sums[index] += a_np[index] * weight
                    g_sums[index] += g_np[index] * weight
        a_factors = [
            jnp.asarray(value / total_metric, dtype=jnp.float32)
            for value in a_sums
        ]
        g_factors = [
            jnp.asarray(value / total_metric, dtype=jnp.float32)
            for value in g_sums
        ]
    if net_cfg.architecture == "mps" and net_cfg.mps_jastrow:
        jastrow_factors = _kfac_factors_mlp(
            params["mps"]["jastrow_mlp"], metric_states, net_cfg
        )
        jastrow_factors = block_until_ready_tree(jastrow_factors)
    time_factors = time.perf_counter() - factor_start

    solve_start = time.perf_counter()
    damping = max(float(opt_cfg.sr_diag_shift), 1e-8)
    factor_damping = np.sqrt(damping)
    gpu_solve = _os.environ.get("NQS_KFAC_GPU_SOLVE") == "1"
    pref_delta = []
    kfac_cond_max = np.nan
    for layer, layer_grad, a_factor, g_factor in zip(
        layer_params, layer_grads, a_factors, g_factors
    ):
        g_is_diagonal = int(g_factor.ndim) == 1
        if opt_cfg.report_fisher_condition:
            a_np = np.asarray(a_factor, dtype=np.float64)
            g_np = np.asarray(g_factor, dtype=np.float64)
            a_damped = a_np + factor_damping * np.eye(a_np.shape[0])
            g_damped = (
                g_np + factor_damping
                if g_is_diagonal
                else g_np + factor_damping * np.eye(g_np.shape[0])
            )
            g_condition = (
                float(np.max(g_damped) / np.min(g_damped))
                if g_is_diagonal
                else float(np.linalg.cond(g_damped))
            )
            block_cond = np.linalg.cond(a_damped) * g_condition
            if np.isfinite(block_cond):
                if np.isnan(kfac_cond_max):
                    kfac_cond_max = float(block_cond)
                else:
                    kfac_cond_max = max(
                        float(kfac_cond_max), float(block_cond)
                    )
        grad_w = jnp.asarray(layer_grad["w"], dtype=jnp.float32)
        grad_b = jnp.asarray(layer_grad["b"], dtype=jnp.float32)
        aug_grad = jnp.concatenate((grad_w, grad_b[None, :]), axis=0)
        if gpu_solve:
            solve = (
                _kfac_gpu_solve_diagonal
                if g_is_diagonal
                else _kfac_gpu_solve_full
            )
            aug_delta = solve(
                a_factor, g_factor, aug_grad, factor_damping
            )
        else:
            a_np = np.asarray(a_factor, dtype=np.float64)
            g_np = np.asarray(g_factor, dtype=np.float64)
            a_damped = a_np + factor_damping * np.eye(a_np.shape[0])
            a_inv = jnp.asarray(np.linalg.inv(a_damped), dtype=jnp.float32)
            if g_is_diagonal:
                g_damped = g_np + factor_damping
                g_inv = jnp.asarray(1.0 / g_damped, dtype=jnp.float32)
                aug_delta = -(a_inv @ aug_grad) * g_inv[None, :]
            else:
                g_damped = g_np + factor_damping * np.eye(g_np.shape[0])
                g_inv = jnp.asarray(np.linalg.inv(g_damped), dtype=jnp.float32)
                aug_delta = -(a_inv @ aug_grad @ g_inv)
        layer_delta = {
            "w": aug_delta[:-1, :],
            "b": aug_delta[-1, :],
        }
        if "ln_scale" in layer_grad:
            layer_delta["ln_scale"] = -jnp.asarray(
                layer_grad["ln_scale"], dtype=jnp.float32
            )
            layer_delta["ln_bias"] = -jnp.asarray(
                layer_grad["ln_bias"], dtype=jnp.float32
            )
        pref_delta.append(layer_delta)
    jastrow_delta = None
    if jastrow_factors is not None:
        jastrow_delta, jastrow_condition = _solve_kfac_blocks(
            params["mps"]["jastrow_mlp"],
            grad_tree["mps"]["jastrow_mlp"],
            jastrow_factors[0],
            jastrow_factors[1],
            damping,
            opt_cfg.report_fisher_condition,
        )
        if np.isfinite(jastrow_condition):
            kfac_cond_max = (
                float(jastrow_condition)
                if np.isnan(kfac_cond_max)
                else max(float(kfac_cond_max), float(jastrow_condition))
            )
    if net_cfg.architecture == "mlp":
        delta_tree = {"pref": pref_delta}
    elif net_cfg.mps_tensor_generation in {
        "local_backflow", "local_conv_backflow"
    }:
        local_delta = {
            "embedding": -jnp.asarray(
                grad_tree["mps"]["embedding"], dtype=jnp.float32
            ),
            "local_mlp": pref_delta,
        }
        if "conv" in params["mps"]:
            local_delta["conv"] = jax.tree_util.tree_map(
                lambda value: -jnp.asarray(value, dtype=jnp.float32),
                grad_tree["mps"]["conv"],
            )
        delta_tree = {"mps": local_delta}
    elif attention_block_params is not None:
        n_mlp = len(params["mps"]["tensor_mlp"])
        attention_delta = []
        for block_index, block_g in enumerate(
            grad_tree["mps"]["attention_blocks"]
        ):
            attention_delta.append(
                {
                    "ln_scale": -jnp.asarray(
                        block_g["ln_scale"], dtype=jnp.float32
                    ),
                    "ln_bias": -jnp.asarray(
                        block_g["ln_bias"], dtype=jnp.float32
                    ),
                    "qkv": pref_delta[n_mlp + 2 * block_index],
                    "out": pref_delta[n_mlp + 2 * block_index + 1],
                }
            )
        delta_tree = {
            "mps": {
                "tensor_mlp": pref_delta[:n_mlp],
                "attention_blocks": attention_delta,
            }
        }
    else:
        delta_tree = {"mps": {"tensor_mlp": pref_delta}}
    if jastrow_delta is not None:
        delta_tree["mps"]["jastrow_mlp"] = jastrow_delta
    delta_tree = block_until_ready_tree(delta_tree)
    time_solve = time.perf_counter() - solve_start

    grad_flat, _ = ravel_pytree(grad_tree)
    grad_norm = float(jnp.linalg.norm(grad_flat.astype(jnp.float64)))
    delta_flat, _ = ravel_pytree(delta_tree)
    delta_norm = float(jnp.linalg.norm(delta_flat))
    trust_scale = 1.0
    if (
        opt_cfg.max_sr_step_norm > 0
        and delta_norm > opt_cfg.max_sr_step_norm
    ):
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta_tree = jax.tree_util.tree_map(
            lambda x: x * trust_scale, delta_tree
        )
        delta_flat, _ = ravel_pytree(delta_tree)
    lr = learning_rate_at_step(opt_cfg, step, total_steps)

    update_start = time.perf_counter()

    def apply_delta(param, delta):
        updated = param.astype(jnp.float32) + lr * delta.astype(
            jnp.float32
        )
        return updated.astype(param.dtype)

    new_params = jax.tree_util.tree_map(apply_delta, params, delta_tree)
    new_params = block_until_ready_tree(new_params)
    time_update = time.perf_counter() - update_start
    return new_params, {
        "optimizer_impl": "kfac",
        "kfac_variant": kfac_variant,
        "lr": lr,
        "grad_norm": grad_norm,
        "delta_norm": float(jnp.linalg.norm(delta_flat)),
        "applied_update_norm": float(jnp.linalg.norm(lr * delta_flat)),
        "trust_region_scale": float(trust_scale),
        "kfac_cond_max": kfac_cond_max,
        "time_factors": time_factors,
        "time_solve": time_solve,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def kfac_condition_diagnostics(
    params,
    states,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    metric_positions=None,
):
    if (
        not opt_cfg.report_fisher_condition
        or net_cfg.architecture != "mlp"
    ):
        return {}
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(
            metric_positions, dtype=jnp.int64
        )
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )
    start = time.perf_counter()
    a_factors, g_factors = _kfac_factors_mlp(
        params["pref"], metric_states, net_cfg
    )
    a_factors, g_factors = block_until_ready_tree(
        (a_factors, g_factors)
    )
    damping = max(float(opt_cfg.sr_diag_shift), 1e-8)
    factor_damping = np.sqrt(damping)
    kfac_cond_max = np.nan
    for a_factor, g_factor in zip(a_factors, g_factors):
        a_np = np.asarray(a_factor, dtype=np.float64)
        g_np = np.asarray(g_factor, dtype=np.float64)
        block_cond = np.linalg.cond(
            a_np + factor_damping * np.eye(a_np.shape[0])
        ) * np.linalg.cond(
            g_np + factor_damping * np.eye(g_np.shape[0])
        )
        if np.isfinite(block_cond):
            if np.isnan(kfac_cond_max):
                kfac_cond_max = float(block_cond)
            else:
                kfac_cond_max = max(
                    float(kfac_cond_max), float(block_cond)
                )
    return {
        "kfac_cond_max": kfac_cond_max,
        "time_kfac_condition": time.perf_counter() - start,
    }


def direct_mps_natural_gradient_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    metric_positions=None,
):
    """Exact matrix-free SR for directly variational periodic MPS tensors."""
    if net_cfg.k != 1:
        raise ValueError("Direct MPS natural gradient currently supports k=1.")

    total_start = time.perf_counter()
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(metric_positions, dtype=jnp.int64)
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )
    if metric_states.ndim != 3 or metric_states.shape[1] != 1:
        raise ValueError(
            "Direct MPS SR metric states must have shape [batch,1,N]."
        )

    dmat_start = time.perf_counter()
    sampled_states = metric_states[:, 0, :]
    occupancy = jnp.clip(jnp.rint(sampled_states), 0, 1).astype(jnp.int32)
    raw = params["mps"]["tensors"].astype(jnp.float32)
    tensor_bank = raw[..., 0] + 1j * raw[..., 1]
    num_sites = int(sampled_states.shape[1])
    bond_dim = int(net_cfg.mps_bond_dim)
    site_indices = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    matrices = tensor_bank[site_indices, occupancy, 0]
    score_real, score_imag = _mps_logpsi_output_score(
        matrices, net_cfg.logphi_real_clip
    )
    matrix_size = bond_dim * bond_dim
    complex_scores = (
        score_real.astype(jnp.complex64)
        + 1j * score_imag.astype(jnp.complex64)
    ).reshape((-1, num_sites, 2 * matrix_size))
    selected_scores = jnp.stack(
        (
            complex_scores[..., :matrix_size],
            complex_scores[..., matrix_size:],
        ),
        axis=-1,
    ).reshape((-1, num_sites, bond_dim, bond_dim, 2))
    occupation_mask = jax.nn.one_hot(
        occupancy, 2, dtype=jnp.float32
    )
    dmat = (
        occupation_mask[..., None, None, None]
        * selected_scores[:, :, None, :, :, :]
    ).reshape((sampled_states.shape[0], -1))
    dmat.block_until_ready()
    time_dmat = time.perf_counter() - dmat_start

    fisher_start = time.perf_counter()
    weights = jnp.ones((dmat.shape[0],), dtype=jnp.float32) / max(
        1, dmat.shape[0]
    )
    centered = dmat - jnp.sum(
        weights[:, None].astype(jnp.complex64) * dmat,
        axis=0,
        keepdims=True,
    )
    grad_flat = jnp.real(
        grad_tree["mps"]["tensors"].reshape((-1,))
    ).astype(jnp.float32)
    centered.block_until_ready()
    grad_flat.block_until_ready()
    time_fisher = time.perf_counter() - fisher_start

    shift = float(opt_cfg.sr_diag_shift)

    def matvec(vector):
        projected = centered @ vector.astype(jnp.complex64)
        fisher_vector = jnp.real(
            centered.conj().T
            @ (weights.astype(jnp.complex64) * projected)
        ).astype(jnp.float32)
        return fisher_vector + shift * vector

    cg_start = time.perf_counter()
    delta, cg_info = jsp_sparse_linalg.cg(
        matvec,
        -grad_flat,
        tol=opt_cfg.cg_tol,
        atol=0.0,
        maxiter=opt_cfg.cg_maxiter,
    )
    delta.block_until_ready()
    time_cg = time.perf_counter() - cg_start

    update_start = time.perf_counter()
    delta_norm = float(jnp.linalg.norm(delta))
    trust_scale = 1.0
    if (
        opt_cfg.max_sr_step_norm > 0
        and delta_norm > opt_cfg.max_sr_step_norm
    ):
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta = delta * trust_scale
    lr = learning_rate_at_step(opt_cfg, step, total_steps)
    applied_delta = lr * delta
    tensor_params = params["mps"]["tensors"]
    updated_tensors = (
        tensor_params.astype(jnp.float32)
        + applied_delta.reshape(tensor_params.shape)
    ).astype(tensor_params.dtype)
    new_params = {"mps": {"tensors": updated_tensors}}
    new_params = block_until_ready_tree(new_params)
    time_update = time.perf_counter() - update_start
    return new_params, {
        "optimizer_impl": "matrix_free_sr",
        "lr": lr,
        "grad_norm": float(jnp.linalg.norm(grad_flat)),
        "delta_norm": delta_norm,
        "applied_update_norm": float(jnp.linalg.norm(applied_delta)),
        "cg_iters": -1 if cg_info is None else int(cg_info),
        "trust_region_scale": float(trust_scale),
        "fisher_cond": np.nan,
        "time_dmat": time_dmat,
        "time_fisher": time_fisher,
        "time_cg": time_cg,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def learning_rate_from_count(
    opt_cfg: OptimizerConfig, count: int, total_steps: int
) -> float:
    count = max(int(count), 0)
    if opt_cfg.constant_lr:
        return float(opt_cfg.lr)
    if opt_cfg.multi_step_lr:
        total_steps = max(int(total_steps), 1)
        stage = min((4 * count) // total_steps, 3)
        return float(opt_cfg.lr) * (0.5**stage)
    if opt_cfg.cyclic_lr:
        period = max(int(opt_cfg.cyclic_lr_period), 1)
        min_lr = (
            float(opt_cfg.cyclic_lr_min)
            if opt_cfg.cyclic_lr_min is not None
            else float(opt_cfg.lr) / 8.0
        )
        max_lr = float(opt_cfg.lr)
        phase = (count % period) / float(period)
        triangle = abs(2.0 * phase - 1.0)
        amplitude = 1.0
        if opt_cfg.cyclic_lr_mode == "triangular2":
            amplitude = 0.5 ** (count // period)
        return min_lr + (max_lr - min_lr) * triangle * amplitude
    return float(opt_cfg.lr) / (
        1.0 + float(count) / float(opt_cfg.decay_rate)
    )


def learning_rate_at_step(
    opt_cfg: OptimizerConfig, step: int, total_steps: int
) -> float:
    if opt_cfg.constant_lr:
        return float(opt_cfg.lr)
    if opt_cfg.multi_step_lr or opt_cfg.cyclic_lr:
        return learning_rate_from_count(opt_cfg, int(step) - 1, total_steps)
    return float(opt_cfg.lr) / (
        1.0 + float(step) / float(opt_cfg.decay_rate)
    )


def make_learning_rate_schedule(
    opt_cfg: OptimizerConfig, total_steps: int
):
    total_steps = max(int(total_steps), 1)
    default_cyclic_lr_min = (
        float(opt_cfg.cyclic_lr_min)
        if opt_cfg.cyclic_lr_min is not None
        else float(opt_cfg.lr) / 8.0
    )
    cyclic_period = max(int(opt_cfg.cyclic_lr_period), 1)

    def lr_schedule(count):
        count_arr = jnp.asarray(count)
        if opt_cfg.constant_lr:
            return jnp.asarray(opt_cfg.lr, dtype=jnp.float32)
        if opt_cfg.multi_step_lr:
            stage = jnp.minimum((count_arr * 4) // total_steps, 3)
            return jnp.asarray(opt_cfg.lr, dtype=jnp.float32) * jnp.power(
                jnp.asarray(0.5, dtype=jnp.float32),
                stage.astype(jnp.float32),
            )
        if opt_cfg.cyclic_lr:
            phase = jnp.mod(count_arr, cyclic_period) / jnp.asarray(
                cyclic_period, dtype=jnp.float32
            )
            triangle = jnp.abs(
                2.0 * phase - jnp.asarray(1.0, dtype=jnp.float32)
            )
            amplitude = jnp.asarray(1.0, dtype=jnp.float32)
            if opt_cfg.cyclic_lr_mode == "triangular2":
                amplitude = jnp.power(
                    jnp.asarray(0.5, dtype=jnp.float32),
                    (count_arr // cyclic_period).astype(jnp.float32),
                )
            min_lr = jnp.asarray(default_cyclic_lr_min, dtype=jnp.float32)
            max_lr = jnp.asarray(opt_cfg.lr, dtype=jnp.float32)
            return min_lr + (max_lr - min_lr) * triangle * amplitude
        return opt_cfg.lr / (1.0 + count_arr / opt_cfg.decay_rate)

    return lr_schedule


def projected_loss(
    params,
    states,
    h_sector,
    step: int,
    net_cfg: NetworkConfig,
    train_cfg: TrainingConfig,
):
    del step
    if net_cfg.k == 1:
        psi = wavefunction(params, states, net_cfg)
        energy = projected_energy_from_psi(psi, h_sector)
        wf_stats = _wavefunction_magnitude_stats(psi)
        return energy, {"energy": energy, **wf_stats}

    phi = wavefunction_matrix(params, states, net_cfg)
    evals, s_proj, _ = projected_ritz_jax(
        phi, h_sector, train_cfg.overlap_shift
    )
    n_loss = (
        train_cfg.loss_levels
        if train_cfg.loss_levels is not None
        else min(net_cfg.k, 2)
    )
    n_loss = max(1, min(int(n_loss), net_cfg.k))
    loss = jnp.real(jnp.sum(evals[:n_loss]))
    _, s_eigs, s_cond = normalized_overlap_condition_jax(s_proj)
    return loss, {
        "energy": evals[0],
        "evals": evals,
        "overlap_min": jnp.min(s_eigs),
        "overlap_max": jnp.max(s_eigs),
        "overlap_cond": s_cond,
    }


def _sparse_hphi(
    phi,
    sparse_rows: SparseHamiltonianRows,
):
    diag, cols, vals, mask = sparse_rows.as_tuple()
    return (
        diag[:, None] * phi
        + jnp.sum(
            vals[:, :, None]
            * mask[:, :, None]
            * phi[cols],
            axis=1,
            dtype=jnp.complex128,
        )
    )


def projected_loss_sparse(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    step: int,
    net_cfg: NetworkConfig,
    train_cfg: TrainingConfig,
):
    del step
    if net_cfg.k == 1:
        psi = wavefunction(params, states, net_cfg).astype(jnp.complex128)
        hpsi = _sparse_hphi(psi[:, None], sparse_rows)[:, 0]
        norm = jnp.real(jnp.vdot(psi, psi))
        energy = jnp.vdot(psi, hpsi) / jnp.maximum(norm, 1e-300)
        energy = jnp.real(energy)
        wf_stats = _wavefunction_magnitude_stats(psi)
        return energy, {"energy": energy, **wf_stats}

    phi = wavefunction_matrix(params, states, net_cfg).astype(jnp.complex128)
    hphi = _sparse_hphi(phi, sparse_rows)
    norm = max(1, phi.shape[0])
    s_proj = phi.conj().T @ phi / norm
    h_proj = phi.conj().T @ hphi / norm
    s_proj = 0.5 * (s_proj + s_proj.conj().T)
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    eye = jnp.eye(s_proj.shape[0], dtype=s_proj.dtype)
    chol = jnp.linalg.cholesky(s_proj + train_cfg.overlap_shift * eye)
    chol_inv = jax.scipy.linalg.solve_triangular(
        chol, eye, lower=True
    )
    h_orth = chol_inv @ h_proj @ chol_inv.conj().T
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    evals = jnp.linalg.eigvalsh(h_orth)
    n_loss = (
        train_cfg.loss_levels
        if train_cfg.loss_levels is not None
        else min(net_cfg.k, 2)
    )
    n_loss = max(1, min(int(n_loss), net_cfg.k))
    loss = jnp.real(jnp.sum(evals[:n_loss]))
    _, s_eigs, s_cond = normalized_overlap_condition_jax(s_proj)
    return loss, {
        "energy": evals[0],
        "evals": evals,
        "overlap_min": jnp.min(s_eigs),
        "overlap_max": jnp.max(s_eigs),
        "overlap_cond": s_cond,
    }


def _accumulate_offdiagonal(
    params,
    states,
    flat_neighbor_cols,
    flat_coeff,
    flat_row_ids,
    row_count: int,
    net_cfg: NetworkConfig,
    deduplicate_neighbors: bool = False,
    unique_padding_size: int | None = None,
    unique_padding_bucket_size: int | None = None,
):
    offdiag_flat = jnp.zeros(
        (row_count, net_cfg.k), dtype=jnp.complex128
    )
    total_neighbors = int(flat_neighbor_cols.shape[0])
    if total_neighbors == 0:
        return offdiag_flat

    if deduplicate_neighbors:
        unique_neighbor_cols_np, inverse_np = np.unique(
            np.asarray(flat_neighbor_cols, dtype=np.int64),
            return_inverse=True,
        )
        if unique_padding_size is not None:
            unique_padding_size = int(unique_padding_size)
            if unique_neighbor_cols_np.size > unique_padding_size:
                raise ValueError(
                    "unique neighbor count exceeds fixed padding size: "
                    f"{unique_neighbor_cols_np.size} > {unique_padding_size}."
                )
            padded_size = unique_padding_size
            if unique_padding_bucket_size is not None:
                bucket_size = int(unique_padding_bucket_size)
                padded_size = min(
                    unique_padding_size,
                    (
                        (unique_neighbor_cols_np.size + bucket_size - 1)
                        // bucket_size
                    )
                    * bucket_size,
                )
            padded_cols = np.zeros(padded_size, dtype=np.int64)
            padded_cols[: unique_neighbor_cols_np.size] = (
                unique_neighbor_cols_np
            )
            unique_neighbor_cols_np = padded_cols
        unique_neighbor_cols = jnp.asarray(
            unique_neighbor_cols_np, dtype=jnp.int64
        )
        inverse = jnp.asarray(inverse_np, dtype=jnp.int64)
        unique_neighbor_phi = batched_wavefunction_matrix_from_positions(
            params, states, unique_neighbor_cols, net_cfg
        ).astype(jnp.complex128)
        neighbor_phi = unique_neighbor_phi[inverse]
    else:
        batch_size = int(net_cfg.forward_batch_size)
        if total_neighbors > batch_size:
            for start in range(0, total_neighbors, batch_size):
                stop = min(start + batch_size, total_neighbors)
                neighbor_states = occupation_states_from_positions(
                    states, flat_neighbor_cols[start:stop], net_cfg
                )
                neighbor_phi = wavefunction_matrix(
                    params, neighbor_states, net_cfg
                ).astype(jnp.complex128)
                offdiag_flat = offdiag_flat.at[
                    flat_row_ids[start:stop]
                ].add(flat_coeff[start:stop, None] * neighbor_phi)
            return offdiag_flat
        neighbor_states = occupation_states_from_positions(
            states, flat_neighbor_cols, net_cfg
        )
        neighbor_phi = wavefunction_matrix(
            params, neighbor_states, net_cfg
        ).astype(jnp.complex128)
    offdiag_flat = offdiag_flat.at[flat_row_ids].add(
        flat_coeff[:, None] * neighbor_phi
    )
    return offdiag_flat


def _accumulate_offdiagonal_from_full_phi(
    full_phi,
    flat_neighbor_cols,
    flat_coeff,
    flat_row_ids,
    row_count: int,
):
    offdiag_flat = jnp.zeros(
        (row_count, full_phi.shape[1]), dtype=jnp.complex128
    )
    neighbor_phi = full_phi[flat_neighbor_cols].astype(jnp.complex128)
    return offdiag_flat.at[flat_row_ids].add(
        flat_coeff[:, None] * neighbor_phi
    )


def _accumulate_offdiagonal_from_bitstrings(
    params,
    neighbor_bitstrings,
    flat_coeff,
    flat_row_ids,
    row_count: int,
    net_cfg: NetworkConfig,
    unique_padding_size: int | None = None,
    unique_padding_bucket_size: int | None = None,
    center_states=None,
    site_tensor_bank=None,
    compact_offsets=None,
):
    """Evaluate and accumulate neighbors without sector-position indices."""
    import numpy as np

    flat_bits = np.asarray(neighbor_bitstrings, dtype=np.uint64).reshape(-1)
    coeff = jnp.asarray(
        flat_coeff,
        dtype=(
            jnp.complex64
            if _os.environ.get("NQS_COMPACT_COEFF_FP32") == "1"
            else jnp.complex128
        ),
    )
    row_ids = (
        None
        if flat_row_ids is None
        else jnp.asarray(flat_row_ids, dtype=jnp.int64)
    )
    offdiag = jnp.zeros((row_count, net_cfg.k), dtype=jnp.complex128)
    if flat_bits.size == 0:
        return offdiag

    if (
        _os.environ.get("NQS_BACKFLOW_DELTA_FIRST_LAYER") == "1"
        and net_cfg.architecture == "mps"
        and net_cfg.mps_tensor_generation == "backflow"
    ):
        if center_states is None:
            raise ValueError("backflow delta updates require center states")
        phi = batched_backflow_delta_neighbor_wavefunction(
            params,
            center_states,
            flat_bits,
            row_ids,
            net_cfg,
        ).astype(jnp.complex128)
        if _os.environ.get("NQS_CONTIGUOUS_ROW_REDUCE"):
            if flat_bits.size % row_count != 0:
                raise ValueError(
                    "contiguous row reduction requires a fixed row width"
                )
            row_width = flat_bits.size // row_count
            weighted = (coeff[:, None] * phi).reshape(
                row_count, row_width, net_cfg.k
            )
            return jnp.sum(weighted, axis=1, dtype=jnp.complex128)
        return offdiag.at[row_ids].add(coeff[:, None] * phi)

    if (
        (_os.environ.get("NQS_LOCAL_BACKFLOW_UPDATE") == "1"
         or site_tensor_bank is not None)
        and net_cfg.architecture == "mps"
        and net_cfg.mps_tensor_generation in {
            "local_backflow", "local_conv_backflow"
        }
    ):
        if center_states is None:
            raise ValueError(
                "local-backflow neighbor updates require center states"
            )
        local_neighbor_wavefunction = (
            _cached_local_backflow_neighbor_wavefunction
            if _os.environ.get("NQS_LOCAL_BACKFLOW_JIT_SCAN") == "1"
            else batched_local_backflow_neighbor_wavefunction
        )
        phi = local_neighbor_wavefunction(
            params,
            center_states,
            flat_bits,
            row_ids,
            net_cfg,
            site_tensor_bank=site_tensor_bank,
        ).astype(jnp.complex128)
        if _os.environ.get("NQS_CONTIGUOUS_ROW_REDUCE"):
            if flat_bits.size % row_count != 0:
                raise ValueError(
                    "contiguous row reduction requires a fixed row width"
                )
            row_width = flat_bits.size // row_count
            weighted = (coeff[:, None] * phi).reshape(
                row_count, row_width, net_cfg.k
            )
            return jnp.sum(weighted, axis=1, dtype=jnp.complex128)
        return offdiag.at[row_ids].add(coeff[:, None] * phi)

    neighbor_wavefunction_matrix = (
        _cached_neighbor_wavefunction_matrix
        if _os.environ.get("NQS_CACHE_BATCHED_WAVEFUNCTION_JIT") == "1"
        else batched_wavefunction_matrix
    )
    if compact_offsets is not None:
        if _os.environ.get("NQS_COMPACT_PALLAS_REDUCE") != "1":
            raise ValueError("staged compact rows require Pallas reduction")
        from .triton_kernels import compact_row_reduce_pallas

        entry_inverse = None
        if _os.environ.get("NQS_COMPACT_GPU_DEDUP") == "1":
            dedup_capacity = int(_os.environ["NQS_COMPACT_DEDUP_SIZE"])
            bits_jax, entry_inverse, group_count = _gpu_sort_deduplicate_bitstrings(
                jnp.asarray(flat_bits, dtype=jnp.uint64),
                capacity=flat_bits.size,
                unique_capacity=dedup_capacity,
            )
            if _os.environ.get("NQS_COMPACT_DEDUP_CHECK") == "1":
                jax.debug.callback(
                    partial(
                        _check_compact_dedup_capacity,
                        capacity=dedup_capacity,
                    ),
                    group_count,
                    ordered=True,
                )
        elif _os.environ.get("NQS_COMPACT_GLOBAL_DEDUP") == "1":
            valid_count = int(np.count_nonzero(flat_bits))
            if _unique_inverse_bits_hash is None:
                unique_bits, inverse = np.unique(
                    flat_bits[:valid_count], return_inverse=True
                )
            else:
                table, slots, cap_mask = _make_hash_workspace(valid_count)
                unique_bits, inverse = _unique_inverse_bits_hash(
                    flat_bits[:valid_count], table, slots, cap_mask
                )
            dedup_capacity = int(_os.environ["NQS_COMPACT_DEDUP_SIZE"])
            if unique_bits.size > dedup_capacity:
                raise ValueError(
                    "unique compact neighbor count exceeds fixed capacity: "
                    f"{unique_bits.size} > {dedup_capacity}"
                )
            padded_unique = np.zeros(dedup_capacity, dtype=np.uint64)
            padded_unique[: unique_bits.size] = unique_bits
            padded_inverse = np.zeros(flat_bits.size, dtype=np.int32)
            padded_inverse[:valid_count] = inverse.astype(
                np.int32, copy=False
            )
            bits_jax = jnp.asarray(padded_unique, dtype=jnp.uint64)
            entry_inverse = jnp.asarray(padded_inverse, dtype=jnp.int32)
        else:
            bits_jax = jnp.asarray(flat_bits, dtype=jnp.uint64)
        if _os.environ.get("NQS_COMPACT_BITSTRING_FORWARD") == "1":
            phi = batched_wavefunction_matrix_from_positions(
                params,
                bits_jax,
                jnp.arange(bits_jax.shape[0], dtype=jnp.int64),
                net_cfg,
            ).astype(jnp.complex128)
        else:
            shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
            states = (
                ((bits_jax[:, None] >> shifts[None, :]) & jnp.uint64(1))
                .astype(jnp.float32)
            )
            phi = neighbor_wavefunction_matrix(
                params, states, net_cfg
            ).astype(jnp.complex128)
        if entry_inverse is not None:
            phi = phi[entry_inverse]
        reduced = compact_row_reduce_pallas(
            coeff,
            phi[:, 0],
            jnp.asarray(compact_offsets, dtype=jnp.int32),
            block_width=1024,
        )
        return reduced[:, None]
    if _os.environ.get("NQS_COMPACT_VALID"):
        # Keep every real neighbor, including duplicates, while removing only
        # the zero-bitstring padding slots.  Bucket padding preserves a small
        # set of static JAX shapes without restoring the original row width.
        nonzero = flat_bits != 0
        valid_indices_np = np.flatnonzero(nonzero).astype(np.int64)
        if valid_indices_np.size == 0:
            return offdiag
        valid_bits = flat_bits[valid_indices_np]
        bucket = int(unique_padding_bucket_size or net_cfg.forward_batch_size)
        fixed_compact_size = _os.environ.get("NQS_COMPACT_VALID_SIZE")
        if fixed_compact_size:
            padded_count = int(fixed_compact_size)
            if valid_bits.size > padded_count:
                raise ValueError(
                    "valid neighbor count exceeds compact padding size: "
                    f"{valid_bits.size} > {padded_count}"
                )
        else:
            padded_count = (
                (valid_bits.size + bucket - 1) // bucket
            ) * bucket
        if _os.environ.get("NQS_DIAG_PAD"):
            print(
                "[diag_compact] flat=%d valid=%d padded=%d" % (
                    flat_bits.size, valid_bits.size, padded_count
                ),
                flush=True,
            )
        padded_bits = np.zeros((padded_count,), dtype=np.uint64)
        padded_bits[: valid_bits.size] = valid_bits
        valid_indices = jnp.asarray(valid_indices_np, dtype=jnp.int64)
        compact_coeff = jnp.zeros((padded_count,), dtype=jnp.complex128)
        compact_coeff = compact_coeff.at[: valid_bits.size].set(
            coeff[valid_indices]
        )
        use_pallas_reduce = (
            _os.environ.get("NQS_COMPACT_PALLAS_REDUCE") == "1"
        )
        if not use_pallas_reduce:
            compact_rows = jnp.zeros((padded_count,), dtype=jnp.int64)
            compact_rows = compact_rows.at[: valid_bits.size].set(
                row_ids[valid_indices]
            )
        shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
        bits_jax = jnp.asarray(padded_bits, dtype=jnp.uint64)
        states = (
            ((bits_jax[:, None] >> shifts[None, :]) & jnp.uint64(1))
            .astype(jnp.float32)
        )
        phi = neighbor_wavefunction_matrix(
            params, states, net_cfg
        ).astype(jnp.complex128)
        if use_pallas_reduce:
            from .triton_kernels import compact_row_reduce_pallas

            if flat_bits.size % row_count:
                raise ValueError(
                    "compact Pallas reduction requires complete source rows"
                )
            counts = np.count_nonzero(
                nonzero.reshape(row_count, flat_bits.size // row_count),
                axis=1,
            )
            offsets = np.empty((row_count + 1,), dtype=np.int32)
            offsets[0] = 0
            np.cumsum(counts, dtype=np.int64, out=offsets[1:])
            reduced = compact_row_reduce_pallas(
                compact_coeff,
                phi[:, 0],
                jnp.asarray(offsets),
                block_width=1024,
            )
            return reduced[:, None]
        return jax.ops.segment_sum(
            compact_coeff[:, None] * phi,
            compact_rows,
            num_segments=row_count,
            indices_are_sorted=True,
        )

    if _os.environ.get("NQS_NO_DEDUP"):
        if _os.environ.get("NQS_PACKED_NEIGHBOR_SCAN") == "1":
            if not _os.environ.get("NQS_CONTIGUOUS_ROW_REDUCE"):
                raise ValueError("packed scan requires contiguous row reduction")
            return _packed_neighbor_row_sum(
                params, jnp.asarray(flat_bits, dtype=jnp.uint64), coeff, net_cfg, row_count
            )
        shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
        bits_jax = jnp.asarray(flat_bits, dtype=jnp.uint64)
        states = (
            ((bits_jax[:, None] >> shifts[None, :]) & jnp.uint64(1))
            .astype(jnp.float32)
        )
        phi = neighbor_wavefunction_matrix(
            params, states, net_cfg
        ).astype(jnp.complex128)
        if _os.environ.get("NQS_CONTIGUOUS_ROW_REDUCE"):
            if flat_bits.size % row_count != 0:
                raise ValueError(
                    "contiguous row reduction requires a fixed row width"
                )
            row_width = flat_bits.size // row_count
            weighted = (coeff[:, None] * phi).reshape(
                row_count, row_width, net_cfg.k
            )
            return jnp.sum(weighted, axis=1, dtype=jnp.complex128)
        return offdiag.at[row_ids].add(coeff[:, None] * phi)

    # padding slots hold bitstring 0 (mask=0); exclude them from the unique set
    nonzero = flat_bits != 0
    if nonzero.sum() == 0:
        return offdiag
    if _unique_inverse_bits_hash is not None:
        vbits = flat_bits[nonzero]
        table, slots, cap_mask = _make_hash_workspace(vbits.size)
        unique_bits, inverse_sub = _unique_inverse_bits_hash(vbits, table, slots, cap_mask)
        inverse_np = np.zeros(flat_bits.size, dtype=np.int64)
        inverse_np[nonzero] = inverse_sub
    else:
        unique_bits, inverse_sub = np.unique(flat_bits[nonzero], return_inverse=True)
        inverse_np = np.zeros(flat_bits.size, dtype=np.int64)
        inverse_np[nonzero] = inverse_sub
    unique_count = unique_bits.size
    if _os.environ.get("NQS_DIAG_PAD"):
        print("[diag_pad] row_count=%d flat=%d nonzero=%d unique=%d limit=%s" % (
            row_count, flat_bits.size, int(nonzero.sum()), unique_count, unique_padding_size), flush=True)
    padded_count = unique_count
    if unique_padding_size is not None:
        limit = int(unique_padding_size)
        if unique_count > limit:
            raise ValueError(
                "unique neighbor count exceeds fixed padding size: "
                f"{unique_count} > {limit}."
            )
        padded_count = limit
    elif unique_padding_bucket_size is not None:
        bucket = int(unique_padding_bucket_size)
        padded_count = ((unique_count + bucket - 1) // bucket) * bucket
    if padded_count > unique_count:
        padded_bits = np.zeros((padded_count,), dtype=np.uint64)
        padded_bits[:unique_count] = unique_bits
        unique_bits = padded_bits

    shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
    bits_jax = jnp.asarray(unique_bits, dtype=jnp.uint64)
    unique_states = (
        ((bits_jax[:, None] >> shifts[None, :]) & jnp.uint64(1))
        .astype(jnp.float32)
    )
    unique_phi = neighbor_wavefunction_matrix(
        params, unique_states, net_cfg
    ).astype(jnp.complex128)
    inverse = jnp.asarray(inverse_np, dtype=jnp.int64)
    neighbor_phi = unique_phi[inverse]
    if _os.environ.get("NQS_CONTIGUOUS_ROW_REDUCE"):
        if flat_bits.size % row_count != 0:
            raise ValueError(
                "contiguous row reduction requires a fixed row width"
            )
        row_width = flat_bits.size // row_count
        weighted = (coeff[:, None] * neighbor_phi).reshape(
            row_count, row_width, net_cfg.k
        )
        return jnp.sum(weighted, axis=1, dtype=jnp.complex128)
    return offdiag.at[row_ids].add(coeff[:, None] * neighbor_phi)


try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

if njit is not None:

    @njit(cache=True)
    def _unique_inverse_bits_hash(values, table, slots, cap_mask):
        """Open-addressing hash dedup for uint64 bitstrings (0 = empty slot)."""
        unique = np.empty(values.size, dtype=np.uint64)
        inverse = np.empty(values.size, dtype=np.int64)
        count = 0
        for i in range(values.size):
            v = values[i]
            v_i = int(v)
            h = (v_i ^ (v_i >> 32)) & cap_mask
            while True:
                slot = table[h]
                if slot == 0:
                    table[h] = v
                    slots[h] = count
                    unique[count] = v
                    count += 1
                    inverse[i] = count - 1
                    break
                elif slot == v:
                    inverse[i] = slots[h]
                    break
                h = (h + 1) & cap_mask
        return unique[:count], inverse

    def _make_hash_workspace(size: int):
        cap = 1
        while cap < max(16, 2 * size):
            cap <<= 1
        return np.zeros(cap, dtype=np.uint64), np.empty(cap, dtype=np.int64), cap - 1

else:
    _unique_inverse_bits_hash = None


def _wavefunction_magnitude_stats(full_phi):
    abs_phi = jnp.abs(full_phi).astype(jnp.float64)
    log_abs_phi = jnp.log(jnp.maximum(abs_phi, 1e-300))
    return {
        "wf_l2_norm": jnp.sqrt(
            jnp.sum(abs_phi * abs_phi, dtype=jnp.float64)
        ),
        "wf_abs_mean": jnp.mean(abs_phi, dtype=jnp.float64),
        "wf_abs_max": jnp.max(abs_phi),
        "wf_abs_min": jnp.min(abs_phi),
        "wf_logabs_mean": jnp.mean(log_abs_phi, dtype=jnp.float64),
    }


def local_energies_sliced(
    params,
    states,
    diag_pos,
    cols_pos,
    vals_pos,
    mask_pos,
    positions,
    net_cfg: NetworkConfig,
    row_batch_size: int | None = None,
    deduplicate_neighbors: bool = False,
    return_wavefunction_stats: bool = False,
    neighbor_padding_size: int | None = None,
    neighbor_padding_bucket_size: int | None = None,
    neighbor_bitstrings_pos=None,
    neighbor_approximation: str = "exact",
    input_gradient_batch_size: int = 32,
    input_hessian_batch_size: int = 4,
    exact_neighbors_per_row: int = 64,
    exact_top_neighbors: int = 8,
    neighbor_sampling_uniform_mix: float = 0.1,
    neighbor_sampling_seed: int = 0,
    site_tensor_bank=None,
    neighbor_compact_offsets=None,
):
    del row_batch_size
    if site_tensor_bank is not None and (
        neighbor_approximation != "exact"
        or not deduplicate_neighbors
        or neighbor_bitstrings_pos is None
    ):
        raise ValueError("site tensor bank requires exact bitstring neighbor rows")
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]

    m, k = positions.shape

    if neighbor_approximation != "exact":
        if (neighbor_approximation not in {
                "input_log_taylor", "input_log_taylor2",
                "input_log_taylor_cv", "exact_neighbor_is"
            } or k != 1
                or net_cfg.k != 1 or neighbor_bitstrings_pos is None):
            raise ValueError("input_log_taylor requires k=1 and bitstring neighbor rows")
        from .neighbor_taylor import (
            approximate_local_energy,
            control_variate_local_energy,
            exact_neighbor_importance_local_energy,
            second_order_approximate_local_energy,
        )
        sampled_states = occupation_states_from_positions(states, positions[:, 0], net_cfg)
        approximation_coeff = jnp.where(
            mask_pos != 0, vals_pos * mask_pos, 0
        )
        if neighbor_approximation == "exact_neighbor_is":
            eloc = exact_neighbor_importance_local_energy(
                params, sampled_states, diag_pos, neighbor_bitstrings_pos,
                approximation_coeff, net_cfg,
                exact_neighbors_per_row=exact_neighbors_per_row,
                exact_top_neighbors=exact_top_neighbors,
                uniform_mix=neighbor_sampling_uniform_mix,
                sampling_seed=neighbor_sampling_seed,
            )
        elif neighbor_approximation == "input_log_taylor2":
            eloc = second_order_approximate_local_energy(
                params, sampled_states, diag_pos, neighbor_bitstrings_pos,
                approximation_coeff, net_cfg,
                hessian_batch_size=input_hessian_batch_size,
            )
        elif neighbor_approximation == "input_log_taylor_cv":
            eloc = control_variate_local_energy(
                params, sampled_states, diag_pos, neighbor_bitstrings_pos,
                approximation_coeff, net_cfg,
                gradient_batch_size=input_gradient_batch_size,
                exact_neighbors_per_row=exact_neighbors_per_row,
                exact_top_neighbors=exact_top_neighbors,
                uniform_mix=neighbor_sampling_uniform_mix,
                sampling_seed=neighbor_sampling_seed,
            )
        else:
            eloc = approximate_local_energy(
                params, sampled_states, diag_pos, neighbor_bitstrings_pos,
                approximation_coeff, net_cfg,
                gradient_batch_size=input_gradient_batch_size,
            )
        if return_wavefunction_stats:
            psi = wavefunction_matrix(jax.lax.stop_gradient(params), sampled_states, net_cfg)
            return eloc, _wavefunction_magnitude_stats(psi)
        return eloc

    eval_params = jax.lax.stop_gradient(params)
    if deduplicate_neighbors:
        sampled_states = occupation_states_from_positions(
            states, positions.reshape((-1,)), net_cfg
        )
        psi_flat = (
            local_conv_site_wavefunction_from_bank(
                site_tensor_bank, sampled_states, net_cfg
            )
            if site_tensor_bank is not None
            else wavefunction_matrix(eval_params, sampled_states, net_cfg)
        ).astype(jnp.complex128)
        full_phi = None
    elif states.ndim == 1:
        full_phi = batched_wavefunction_matrix_from_positions(
            eval_params,
            states,
            jnp.arange(states.shape[0], dtype=jnp.int64),
            net_cfg,
        ).astype(jnp.complex128)
        psi_flat = full_phi[positions.reshape((-1,))]
    elif net_cfg.architecture == "mps":
        full_phi = batched_wavefunction_matrix(
            eval_params,
            jnp.asarray(states, dtype=jnp.float32),
            net_cfg,
        ).astype(jnp.complex128)
        psi_flat = full_phi[positions.reshape((-1,))]
    else:
        full_phi = wavefunction_matrix(
            eval_params, jnp.asarray(states, dtype=jnp.float32), net_cfg
        ).astype(jnp.complex128)
        psi_flat = full_phi[positions.reshape((-1,))]
    psi_mats = psi_flat.reshape((m, k, net_cfg.k))

    if neighbor_compact_offsets is not None:
        nnz = None
        flat_neighbor_cols = None
    elif neighbor_bitstrings_pos is None:
        nnz = int(
            cols_pos.shape[2] if cols_pos.ndim == 3 else cols_pos.shape[1]
        )
        flat_neighbor_cols = cols_pos.reshape((-1,))
    else:
        neighbor_shape = np.shape(neighbor_bitstrings_pos)
        nnz = int(neighbor_shape[2] if len(neighbor_shape) == 3 else neighbor_shape[1])
        flat_neighbor_cols = None
    flat_coeff = (vals_pos * mask_pos).reshape((-1,))
    flat_row_ids = (
        None
        if nnz is None
        else jnp.repeat(jnp.arange(m * k, dtype=jnp.int64), nnz)
    )
    if neighbor_bitstrings_pos is not None:
        offdiag_flat = _accumulate_offdiagonal_from_bitstrings(
            eval_params,
            neighbor_bitstrings_pos,
            flat_coeff,
            flat_row_ids,
            m * k,
            net_cfg,
            unique_padding_size=neighbor_padding_size,
            unique_padding_bucket_size=neighbor_padding_bucket_size,
            center_states=(
                sampled_states
                if "sampled_states" in locals()
                else occupation_states_from_positions(
                    states, positions.reshape((-1,)), net_cfg
                )
            ),
            site_tensor_bank=site_tensor_bank,
            compact_offsets=neighbor_compact_offsets,
        )
    elif deduplicate_neighbors:
        offdiag_flat = _accumulate_offdiagonal(
            eval_params,
            states,
            flat_neighbor_cols,
            flat_coeff,
            flat_row_ids,
            m * k,
            net_cfg,
            deduplicate_neighbors=True,
            unique_padding_size=neighbor_padding_size,
            unique_padding_bucket_size=neighbor_padding_bucket_size,
        )
    else:
        offdiag_flat = _accumulate_offdiagonal_from_full_phi(
            full_phi,
            flat_neighbor_cols,
            flat_coeff,
            flat_row_ids,
            m * k,
        )
    offdiag_hpsi = offdiag_flat.reshape((m, k, net_cfg.k))
    hpsi_mats = diag_pos[:, :, None] * psi_mats + offdiag_hpsi
    local_matrix = jnp.linalg.inv(psi_mats) @ hpsi_mats
    eloc = jnp.trace(local_matrix, axis1=1, axis2=2)
    if return_wavefunction_stats:
        stats_phi = psi_flat if deduplicate_neighbors else full_phi
        return eloc, _wavefunction_magnitude_stats(stats_phi)
    return eloc


def local_energies(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    positions,
    net_cfg: NetworkConfig,
    row_batch_size: int | None = None,
    deduplicate_neighbors: bool = False,
    return_wavefunction_stats: bool = False,
):
    del row_batch_size
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]
    m, k = positions.shape
    diag, cols, vals, mask = sparse_rows.as_tuple()

    if states.ndim == 1:
        eval_params = (
            jax.lax.stop_gradient(params)
            if net_cfg.architecture == "mps"
            else params
        )
        full_phi = batched_wavefunction_matrix_from_positions(
            eval_params,
            states,
            jnp.arange(states.shape[0], dtype=jnp.int64),
            net_cfg,
        ).astype(jnp.complex128)
    elif net_cfg.architecture == "mps":
        full_phi = batched_wavefunction_matrix(
            jax.lax.stop_gradient(params),
            jnp.asarray(states, dtype=jnp.float32),
            net_cfg,
        ).astype(jnp.complex128)
    else:
        full_phi = wavefunction_matrix(
            params, jnp.asarray(states, dtype=jnp.float32), net_cfg
        ).astype(jnp.complex128)
    psi_flat = full_phi[positions.reshape((-1,))]
    psi_mats = psi_flat.reshape((m, k, net_cfg.k))

    neighbor_cols = cols[positions]
    flat_neighbor_cols = neighbor_cols.reshape((-1,))
    flat_coeff = (vals[positions] * mask[positions]).reshape((-1,))
    nnz = int(cols.shape[1])
    flat_row_ids = jnp.repeat(
        jnp.arange(m * k, dtype=jnp.int64), nnz
    )
    offdiag_flat = _accumulate_offdiagonal_from_full_phi(
        full_phi,
        flat_neighbor_cols,
        flat_coeff,
        flat_row_ids,
        m * k,
    )
    offdiag_hpsi = offdiag_flat.reshape((m, k, net_cfg.k))
    hpsi_mats = diag[positions][:, :, None] * psi_mats + offdiag_hpsi
    local_matrix = jnp.linalg.inv(psi_mats) @ hpsi_mats
    eloc = jnp.trace(local_matrix, axis1=1, axis2=2)
    if return_wavefunction_stats:
        return eloc, _wavefunction_magnitude_stats(full_phi)
    return eloc


def vmc_surrogate_loss_sliced(
    params,
    states,
    sparse_sliced,
    positions,
    net_cfg: NetworkConfig,
    row_batch_size: int | None = None,
    deduplicate_neighbors: bool = False,
    neighbor_padding_size: int | None = None,
    neighbor_padding_bucket_size: int | None = None,
):
    if len(sparse_sliced) == 5:
        diag_pos, cols_pos, vals_pos, mask_pos, neighbor_bitstrings_pos = sparse_sliced
    else:
        diag_pos, cols_pos, vals_pos, mask_pos = sparse_sliced
        neighbor_bitstrings_pos = None
    logpsi = determinant_logpsi_from_positions_direct(
        params, states, positions, net_cfg
    )
    eloc, wf_stats = local_energies_sliced(
        params,
        states,
        diag_pos,
        cols_pos,
        vals_pos,
        mask_pos,
        positions,
        net_cfg,
        row_batch_size=row_batch_size,
        deduplicate_neighbors=deduplicate_neighbors,
        return_wavefunction_stats=True,
        neighbor_padding_size=neighbor_padding_size,
        neighbor_padding_bucket_size=neighbor_padding_bucket_size,
        neighbor_bitstrings_pos=neighbor_bitstrings_pos,
    )
    eloc = jax.lax.stop_gradient(eloc)
    wf_stats = jax.tree_util.tree_map(jax.lax.stop_gradient, wf_stats)
    energy = jnp.real(jnp.mean(eloc))
    centered = jax.lax.stop_gradient(eloc - energy)
    loss = 2.0 * jnp.real(jnp.mean(jnp.conj(centered) * logpsi))
    variance = jnp.real(jnp.mean(jnp.abs(eloc - energy) ** 2))
    return loss, {
        "energy": energy,
        "variance": variance,
        "eloc_mean_imag": jnp.imag(jnp.mean(eloc)),
        **wf_stats,
    }


def vmc_surrogate_gradient_from_eloc(
    params,
    states,
    positions,
    eloc,
    global_energy,
    total_sample_count: int,
    net_cfg: NetworkConfig,
):
    """Return one chain microbatch's contribution to the global VMC loss."""
    logpsi = determinant_logpsi_from_positions_direct(
        params, states, positions, net_cfg
    )
    centered = jax.lax.stop_gradient(
        jnp.asarray(eloc, dtype=jnp.complex128) - global_energy
    )
    return (
        2.0
        * jnp.real(jnp.sum(jnp.conj(centered) * logpsi))
        / float(total_sample_count)
    )


def vmc_surrogate_loss(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    positions,
    net_cfg: NetworkConfig,
    row_batch_size: int | None = None,
    deduplicate_neighbors: bool = False,
):
    logpsi = determinant_logpsi_from_positions_direct(
        params, states, positions, net_cfg
    )
    eloc, wf_stats = local_energies(
        params,
        states,
        sparse_rows,
        positions,
        net_cfg,
        row_batch_size=row_batch_size,
        deduplicate_neighbors=deduplicate_neighbors,
        return_wavefunction_stats=True,
    )
    eloc = jax.lax.stop_gradient(eloc)
    wf_stats = jax.tree_util.tree_map(jax.lax.stop_gradient, wf_stats)
    energy = jnp.real(jnp.mean(eloc))
    centered = jax.lax.stop_gradient(eloc - energy)
    loss = 2.0 * jnp.real(jnp.mean(jnp.conj(centered) * logpsi))
    variance = jnp.real(jnp.mean(jnp.abs(eloc - energy) ** 2))
    return loss, {
        "energy": energy,
        "variance": variance,
        "eloc_mean_imag": jnp.imag(jnp.mean(eloc)),
        **wf_stats,
    }


def profile_vmc_loss_sliced(
    params,
    states,
    diag_pos,
    cols_pos,
    vals_pos,
    mask_pos,
    positions,
    net_cfg: NetworkConfig,
    deduplicate_neighbors: bool = False,
):
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]

    timings: dict[str, float] = {}
    counts: dict[str, int] = {}

    start = time.perf_counter()
    logpsi_states = occupation_states_from_positions(
        states, positions.reshape((-1,)), net_cfg
    )
    logpsi_states.block_until_ready()
    timings["logpsi_state_gather"] = time.perf_counter() - start

    start = time.perf_counter()
    logpsi_phi = wavefunction_matrix(
        params, logpsi_states, net_cfg
    )
    logpsi_phi.block_until_ready()
    timings["logpsi_network_forward"] = time.perf_counter() - start

    start = time.perf_counter()
    logpsi_mats = logpsi_phi.reshape(
        (positions.shape[0], positions.shape[1], net_cfg.k)
    )
    sign, logabsdet = jnp.linalg.slogdet(logpsi_mats)
    logpsi = logabsdet + 1j * jnp.angle(sign)
    logpsi.block_until_ready()
    timings["logpsi_slogdet"] = time.perf_counter() - start
    timings["logpsi"] = (
        timings["logpsi_state_gather"]
        + timings["logpsi_network_forward"]
        + timings["logpsi_slogdet"]
    )

    m, k = positions.shape

    start = time.perf_counter()
    sampled_states = occupation_states_from_positions(
        states, positions.reshape((-1,)), net_cfg
    )
    sampled_states.block_until_ready()
    timings["sampled_state_gather"] = time.perf_counter() - start

    start = time.perf_counter()
    psi_flat = wavefunction_matrix(
        params, sampled_states, net_cfg
    )
    psi_flat.block_until_ready()
    timings["sampled_network_forward"] = time.perf_counter() - start
    timings["sampled_forward"] = (
        timings["sampled_state_gather"]
        + timings["sampled_network_forward"]
    )
    psi_mats = psi_flat.reshape((m, k, net_cfg.k)).astype(jnp.complex128)

    flat_neighbor_cols = cols_pos.reshape((-1,))
    counts["neighbor_total"] = int(flat_neighbor_cols.shape[0])
    counts["neighbor_deduplicated"] = int(deduplicate_neighbors)
    if deduplicate_neighbors:
        start = time.perf_counter()
        flat_neighbor_cols_np = np.asarray(
            flat_neighbor_cols, dtype=np.int64
        )
        timings["neighbor_cols_d2h"] = time.perf_counter() - start

        start = time.perf_counter()
        try:
            from .stamp_dedup import (
                _WORKSPACE as _stamp_workspace,
                _unique_inverse_stamp,
            )
        except ImportError:  # pragma: no cover - optional profiling helper.
            _unique_inverse_stamp = None
        if _unique_inverse_stamp is None:
            evaluated_cols_np, inverse_np = np.unique(
                flat_neighbor_cols_np,
                return_inverse=True,
            )
        else:
            stamps, slots, generation = _stamp_workspace.next(
                int(states.shape[0])
            )
            evaluated_cols_np, inverse_np = _unique_inverse_stamp(
                flat_neighbor_cols_np,
                stamps,
                slots,
                generation,
            )
        timings["neighbor_unique_cpu"] = time.perf_counter() - start
        timings["neighbor_dedup_cpu"] = (
            timings["neighbor_cols_d2h"]
            + timings["neighbor_unique_cpu"]
        )
        counts["neighbor_unique"] = int(evaluated_cols_np.size)
        counts["neighbor_evaluated"] = int(evaluated_cols_np.size)

        start = time.perf_counter()
        evaluated_cols = jnp.asarray(
            evaluated_cols_np, dtype=jnp.int64
        )
        inverse = jnp.asarray(inverse_np, dtype=jnp.int64)
        block_until_ready_tree((evaluated_cols, inverse))
        timings["neighbor_indices_h2d"] = (
            time.perf_counter() - start
        )
    else:
        evaluated_cols = flat_neighbor_cols
        inverse = None
        timings["neighbor_cols_d2h"] = 0.0
        timings["neighbor_unique_cpu"] = 0.0
        timings["neighbor_dedup_cpu"] = 0.0
        timings["neighbor_indices_h2d"] = 0.0
        counts["neighbor_unique"] = counts["neighbor_total"]
        counts["neighbor_evaluated"] = counts["neighbor_total"]

    start = time.perf_counter()
    flat_coeff = (vals_pos * mask_pos).reshape((-1,))
    flat_coeff.block_until_ready()
    timings["coefficient_prepare"] = time.perf_counter() - start

    nnz = int(
        cols_pos.shape[2] if cols_pos.ndim == 3 else cols_pos.shape[1]
    )
    start = time.perf_counter()
    flat_row_ids = jnp.repeat(
        jnp.arange(m * k, dtype=jnp.int64), nnz
    )
    flat_row_ids.block_until_ready()
    timings["row_ids_prepare"] = time.perf_counter() - start

    # The chunked position evaluator gathers and casts only one forward batch
    # at a time. Its gather cost is included in the forward measurement.
    timings["neighbor_state_gather"] = 0.0
    start = time.perf_counter()
    evaluated_neighbor_phi = batched_wavefunction_matrix_from_positions(
        params, states, evaluated_cols, net_cfg
    ).astype(jnp.complex128)
    evaluated_neighbor_phi.block_until_ready()
    timings["neighbor_forward_first"] = time.perf_counter() - start

    start = time.perf_counter()
    evaluated_neighbor_phi = batched_wavefunction_matrix_from_positions(
        params, states, evaluated_cols, net_cfg
    ).astype(jnp.complex128)
    evaluated_neighbor_phi.block_until_ready()
    timings["neighbor_forward_warm"] = time.perf_counter() - start
    timings["neighbor_forward_compile_gap"] = max(
        0.0,
        timings["neighbor_forward_first"]
        - timings["neighbor_forward_warm"],
    )
    timings["neighbor_forward"] = timings["neighbor_forward_first"]

    start = time.perf_counter()
    if inverse is None:
        neighbor_phi = evaluated_neighbor_phi
    else:
        neighbor_phi = evaluated_neighbor_phi[inverse]
        neighbor_phi.block_until_ready()
    timings["neighbor_inverse_gather"] = time.perf_counter() - start

    start = time.perf_counter()
    offdiag_flat = jnp.zeros(
        (m * k, net_cfg.k), dtype=jnp.complex128
    )
    offdiag_flat = offdiag_flat.at[flat_row_ids].add(
        flat_coeff[:, None] * neighbor_phi
    )
    offdiag_flat.block_until_ready()
    timings["offdiag_segment_scatter"] = time.perf_counter() - start
    timings["offdiag_scatter"] = (
        timings["neighbor_inverse_gather"]
        + timings["offdiag_segment_scatter"]
    )

    start = time.perf_counter()
    offdiag_hpsi = offdiag_flat.reshape((m, k, net_cfg.k))
    hpsi_mats = diag_pos[:, :, None] * psi_mats + offdiag_hpsi
    local_matrix = jnp.linalg.inv(psi_mats) @ hpsi_mats
    eloc = jnp.trace(local_matrix, axis1=1, axis2=2)
    eloc.block_until_ready()
    timings["solve_trace"] = time.perf_counter() - start

    start = time.perf_counter()
    energy = jnp.real(jnp.mean(eloc))
    centered = eloc - energy
    loss = 2.0 * jnp.real(jnp.mean(jnp.conj(centered) * logpsi))
    variance = jnp.real(jnp.mean(jnp.abs(centered) ** 2))
    block_until_ready_tree((loss, energy, variance))
    timings["loss_reduction"] = time.perf_counter() - start

    common_local_energy = (
        timings["sampled_forward"]
        + timings["neighbor_dedup_cpu"]
        + timings["coefficient_prepare"]
        + timings["row_ids_prepare"]
        + timings["neighbor_indices_h2d"]
        + timings["neighbor_state_gather"]
        + timings["offdiag_scatter"]
        + timings["solve_trace"]
    )
    timings["local_energy_total"] = (
        common_local_energy + timings["neighbor_forward_first"]
    )
    timings["local_energy_warm_total"] = (
        common_local_energy + timings["neighbor_forward_warm"]
    )
    timings["profiled_total"] = (
        timings["logpsi"]
        + timings["local_energy_total"]
        + timings["loss_reduction"]
    )
    timings["profiled_warm_total"] = (
        timings["logpsi"]
        + timings["local_energy_warm_total"]
        + timings["loss_reduction"]
    )
    return {**timings, **counts}


def profile_vmc_loss(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    positions,
    net_cfg: NetworkConfig,
    deduplicate_neighbors: bool = False,
):
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]
    diag, cols, vals, mask = sparse_rows.as_tuple()
    positions_np = np.asarray(positions)
    return profile_vmc_loss_sliced(
        params,
        states,
        diag[positions_np],
        cols[positions_np],
        vals[positions_np],
        mask[positions_np],
        positions,
        net_cfg,
        deduplicate_neighbors=deduplicate_neighbors,
    )


def profile_vmc_autodiff_sliced(
    params,
    states,
    diag_pos,
    cols_pos,
    vals_pos,
    mask_pos,
    positions,
    net_cfg: NetworkConfig,
    deduplicate_neighbors: bool = False,
):
    timings: dict[str, float] = {}
    sparse_sliced = (diag_pos, cols_pos, vals_pos, mask_pos)

    def loss_fn(current_params):
        return vmc_surrogate_loss_sliced(
            current_params,
            states,
            sparse_sliced,
            positions,
            net_cfg,
            deduplicate_neighbors=deduplicate_neighbors,
        )

    # Warm both traces so the reported numbers reflect execution rather
    # than first-call tracing/compilation artifacts.
    block_until_ready_tree(loss_fn(params))
    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    block_until_ready_tree(value_and_grad(params))

    start = time.perf_counter()
    forward_result = loss_fn(params)
    block_until_ready_tree(forward_result)
    timings["loss_forward_eval"] = time.perf_counter() - start

    start = time.perf_counter()
    grad_result = value_and_grad(params)
    block_until_ready_tree(grad_result)
    timings["loss_value_and_grad_profiled"] = (
        time.perf_counter() - start
    )
    timings["loss_backward_extra"] = max(
        0.0,
        timings["loss_value_and_grad_profiled"]
        - timings["loss_forward_eval"],
    )
    return timings


def profile_vmc_autodiff(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    positions,
    net_cfg: NetworkConfig,
    deduplicate_neighbors: bool = False,
):
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]
    diag, cols, vals, mask = sparse_rows.as_tuple()
    positions_np = np.asarray(positions)
    return profile_vmc_autodiff_sliced(
        params,
        states,
        diag[positions_np],
        cols[positions_np],
        vals[positions_np],
        mask[positions_np],
        positions,
        net_cfg,
        deduplicate_neighbors=deduplicate_neighbors,
    )


def natural_gradient_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    metric_positions=None,
):
    total_start = time.perf_counter()
    theta, unravel = ravel_pytree(params)
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(
            metric_positions, dtype=jnp.int64
        )
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )

    def det_scalar_fn(flat_params):
        return determinant_sr_scalars(
            unravel(flat_params), metric_states, net_cfg
        )

    dmat_start = time.perf_counter()
    jac_re = jax.jacrev(lambda th: jnp.real(det_scalar_fn(th)))(theta)
    jac_im = jax.jacrev(lambda th: jnp.imag(det_scalar_fn(th)))(theta)
    dmat = (jac_re + 1j * jac_im).astype(jnp.complex64)
    dmat.block_until_ready()
    time_dmat = time.perf_counter() - dmat_start

    fisher_start = time.perf_counter()
    weights = jnp.ones(
        (dmat.shape[0],), dtype=jnp.float32
    ) / max(1, dmat.shape[0])
    d_mean = jnp.sum(
        weights[:, None].astype(jnp.complex64) * dmat,
        axis=0,
        keepdims=True,
    )
    dc = dmat - d_mean
    grad_flat, _ = ravel_pytree(grad_tree)
    grad_flat = jnp.real(grad_flat).astype(jnp.float32)
    fisher = jnp.real(
        dc.conj().T @ (weights[:, None].astype(jnp.complex64) * dc)
    ).astype(jnp.float32)
    fisher.block_until_ready()
    grad_flat.block_until_ready()
    time_fisher = time.perf_counter() - fisher_start
    fisher_cond = np.nan
    fisher_shifted_cond = np.nan
    fisher_min_eig = np.nan
    fisher_max_eig = np.nan
    if opt_cfg.report_fisher_condition:
        fisher_np = np.asarray(fisher, dtype=np.float64)
        fisher_eigs = np.linalg.eigvalsh(fisher_np)
        fisher_min_eig = float(fisher_eigs[0])
        fisher_max_eig = float(fisher_eigs[-1])
        if fisher_min_eig > 0.0:
            fisher_cond = fisher_max_eig / fisher_min_eig
        shifted_min = fisher_min_eig + float(opt_cfg.sr_diag_shift)
        shifted_max = fisher_max_eig + float(opt_cfg.sr_diag_shift)
        if shifted_min > 0.0:
            fisher_shifted_cond = shifted_max / shifted_min

    def matvec(v):
        return fisher @ v + opt_cfg.sr_diag_shift * v

    cg_start = time.perf_counter()
    delta, cg_info = jsp_sparse_linalg.cg(
        matvec,
        -grad_flat,
        tol=opt_cfg.cg_tol,
        atol=0.0,
        maxiter=opt_cfg.cg_maxiter,
    )
    delta.block_until_ready()
    time_cg = time.perf_counter() - cg_start

    update_start = time.perf_counter()
    delta_norm = float(jnp.linalg.norm(delta))
    trust_scale = 1.0
    if (
        opt_cfg.max_sr_step_norm > 0
        and delta_norm > opt_cfg.max_sr_step_norm
    ):
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta = delta * trust_scale
    lr = learning_rate_at_step(opt_cfg, step, total_steps)
    applied_delta = lr * delta
    new_theta = theta + applied_delta
    new_theta.block_until_ready()
    time_update = time.perf_counter() - update_start
    return unravel(new_theta), {
        "lr": lr,
        "grad_norm": float(jnp.linalg.norm(grad_flat)),
        "delta_norm": delta_norm,
        "applied_update_norm": float(jnp.linalg.norm(applied_delta)),
        "cg_iters": -1 if cg_info is None else int(cg_info),
        "trust_region_scale": float(trust_scale),
        "fisher_cond": fisher_cond,
        "fisher_shifted_cond": fisher_shifted_cond,
        "fisher_min_eig": fisher_min_eig,
        "fisher_max_eig": fisher_max_eig,
        "time_dmat": time_dmat,
        "time_fisher": time_fisher,
        "time_cg": time_cg,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def _solve_minsr_design(design, gradient, diagonal_shift):
    """Solve ``(A.T A + shift I) delta = -gradient`` in sample space."""

    design = jnp.asarray(design, dtype=jnp.float32)
    gradient = jnp.asarray(gradient, dtype=jnp.float32)
    shift = jnp.maximum(jnp.asarray(diagonal_shift, dtype=jnp.float32), 1e-8)
    precision = jax.lax.Precision.HIGHEST
    kernel = jnp.matmul(design, design.T, precision=precision)
    kernel = kernel + shift * jnp.eye(kernel.shape[0], dtype=kernel.dtype)
    rhs = jnp.matmul(design, gradient, precision=precision)
    chol = jnp.linalg.cholesky(kernel)
    intermediate = jsp_linalg.solve_triangular(chol, rhs, lower=True)
    dual = jsp_linalg.solve_triangular(chol.T, intermediate, lower=False)
    projected = jnp.matmul(design.T, dual, precision=precision)
    return -(gradient - projected) / shift


def minsr_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    total_steps: int,
    metric_positions=None,
    metric_batch_size: int | None = None,
):
    """Exact shifted SR using a real sample-space (MinSR) solve.

    For real parameters and complex log-derivatives ``O``, the Fisher matrix is
    ``Re(O.H @ O) / Ns``. Writing its centered real and imaginary parts as a
    ``(2 Ns) x Np`` real design matrix avoids materializing the ``Np x Np``
    Fisher matrix used by the legacy dense natural-gradient implementation.
    """

    total_start = time.perf_counter()
    theta, unravel = ravel_pytree(params)
    if metric_positions is None:
        metric_states = build_metric_state_batches(states, net_cfg)
    else:
        metric_positions = jnp.asarray(metric_positions, dtype=jnp.int64)
        if metric_positions.ndim == 1:
            metric_positions = metric_positions[:, None]
        metric_states = occupation_states_from_positions(
            states, metric_positions, net_cfg
        )

    available_samples = int(metric_states.shape[0])
    sample_count = available_samples
    if opt_cfg.sr_samples is not None:
        sample_count = min(available_samples, int(opt_cfg.sr_samples))
        metric_states = metric_states[:sample_count]
    if sample_count < 1:
        raise ValueError("SR requires at least one metric sample.")
    batch_size = min(sample_count, int(opt_cfg.sr_jacobian_batch_size))
    if metric_batch_size is not None:
        batch_size = min(batch_size, int(metric_batch_size))

    dmat_start = time.perf_counter()
    jacobian_chunks = []
    for start in range(0, sample_count, batch_size):
        state_chunk = metric_states[start : start + batch_size]

        def score_components(flat_params):
            scores = determinant_sr_scalars(
                unravel(flat_params), state_chunk, net_cfg
            )
            return jnp.stack((jnp.real(scores), jnp.imag(scores)), axis=1)

        jacobian_chunks.append(jax.jacrev(score_components)(theta))
    score_jacobian = (
        jacobian_chunks[0]
        if len(jacobian_chunks) == 1
        else jnp.concatenate(jacobian_chunks, axis=0)
    ).astype(jnp.float32)
    score_jacobian.block_until_ready()
    time_dmat = time.perf_counter() - dmat_start

    design_start = time.perf_counter()
    centered = score_jacobian - jnp.mean(
        score_jacobian, axis=0, keepdims=True
    )
    # Interleaving real/imaginary rows avoids a large transpose-copy. Row
    # ordering does not change A.T @ A or the sample-space solve.
    design = centered.reshape((2 * sample_count, theta.size)) / jnp.sqrt(
        jnp.asarray(sample_count, dtype=jnp.float32)
    )
    grad_flat, _ = ravel_pytree(grad_tree)
    grad_flat = jnp.real(grad_flat).astype(jnp.float32)
    design.block_until_ready()
    grad_flat.block_until_ready()
    del centered, score_jacobian, jacobian_chunks
    time_design = time.perf_counter() - design_start

    solve_start = time.perf_counter()
    delta = _solve_minsr_design(design, grad_flat, opt_cfg.sr_diag_shift)
    delta.block_until_ready()
    time_solve = time.perf_counter() - solve_start

    update_start = time.perf_counter()
    delta_norm = float(jnp.linalg.norm(delta))
    trust_scale = 1.0
    if (
        opt_cfg.max_sr_step_norm > 0
        and delta_norm > opt_cfg.max_sr_step_norm
    ):
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta = delta * trust_scale
    lr = learning_rate_at_step(opt_cfg, step, total_steps)
    applied_delta = lr * delta
    new_theta = theta + applied_delta
    new_theta.block_until_ready()
    time_update = time.perf_counter() - update_start
    return unravel(new_theta), {
        "optimizer_impl": "minsr",
        "lr": lr,
        "grad_norm": float(jnp.linalg.norm(grad_flat)),
        "delta_norm": delta_norm,
        "applied_update_norm": float(jnp.linalg.norm(applied_delta)),
        "trust_region_scale": float(trust_scale),
        "sr_samples": sample_count,
        "sr_available_samples": available_samples,
        "sr_jacobian_batch_size": batch_size,
        "sr_kernel_dim": 2 * sample_count,
        "time_dmat": time_dmat,
        "time_design": time_design,
        "time_solve": time_solve,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def continuous_natural_gradient_update(
    params,
    states,
    coordinates,
    active_rows,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    continuous_cfg,
    opt_cfg: OptimizerConfig,
    total_steps: int,
):
    total_start = time.perf_counter()
    theta, unravel = ravel_pytree(params)

    def logpsi_fn(flat_params):
        return continuous_logpsi(
            unravel(flat_params),
            states,
            coordinates,
            active_rows,
            net_cfg,
            continuous_cfg,
        )

    dmat_start = time.perf_counter()
    jac_re = jax.jacrev(lambda th: jnp.real(logpsi_fn(th)))(theta)
    jac_im = jax.jacrev(lambda th: jnp.imag(logpsi_fn(th)))(theta)
    dmat = (jac_re + 1j * jac_im).astype(jnp.complex64)
    dmat.block_until_ready()
    time_dmat = time.perf_counter() - dmat_start

    fisher_start = time.perf_counter()
    weights = jnp.ones(
        (dmat.shape[0],), dtype=jnp.float32
    ) / max(1, dmat.shape[0])
    d_mean = jnp.sum(
        weights[:, None].astype(jnp.complex64) * dmat,
        axis=0,
        keepdims=True,
    )
    centered = dmat - d_mean
    grad_flat, _ = ravel_pytree(grad_tree)
    grad_flat = jnp.real(grad_flat).astype(jnp.float32)
    fisher = jnp.real(
        centered.conj().T
        @ (weights[:, None].astype(jnp.complex64) * centered)
    ).astype(jnp.float32)
    fisher.block_until_ready()
    grad_flat.block_until_ready()
    time_fisher = time.perf_counter() - fisher_start
    fisher_cond = np.nan
    fisher_shifted_cond = np.nan
    fisher_min_eig = np.nan
    fisher_max_eig = np.nan
    if opt_cfg.report_fisher_condition:
        fisher_np = np.asarray(fisher, dtype=np.float64)
        fisher_eigs = np.linalg.eigvalsh(fisher_np)
        fisher_min_eig = float(fisher_eigs[0])
        fisher_max_eig = float(fisher_eigs[-1])
        if fisher_min_eig > 0.0:
            fisher_cond = fisher_max_eig / fisher_min_eig
        shifted_min = fisher_min_eig + float(opt_cfg.sr_diag_shift)
        shifted_max = fisher_max_eig + float(opt_cfg.sr_diag_shift)
        if shifted_min > 0.0:
            fisher_shifted_cond = shifted_max / shifted_min

    def matvec(vector):
        return fisher @ vector + opt_cfg.sr_diag_shift * vector

    cg_start = time.perf_counter()
    grad_norm_value = float(jnp.linalg.norm(grad_flat))
    if grad_norm_value < 1e-14:
        delta = jnp.zeros_like(grad_flat)
        cg_info = 0
    else:
        delta, cg_info = jsp_sparse_linalg.cg(
            matvec,
            -grad_flat,
            tol=opt_cfg.cg_tol,
            atol=0.0,
            maxiter=opt_cfg.cg_maxiter,
        )
    delta.block_until_ready()
    time_cg = time.perf_counter() - cg_start

    update_start = time.perf_counter()
    delta_norm = float(jnp.linalg.norm(delta))
    trust_scale = 1.0
    if (
        opt_cfg.max_sr_step_norm > 0
        and delta_norm > opt_cfg.max_sr_step_norm
    ):
        trust_scale = opt_cfg.max_sr_step_norm / delta_norm
        delta = delta * trust_scale
    lr = learning_rate_at_step(opt_cfg, step, total_steps)
    applied_delta = lr * delta
    new_theta = theta + applied_delta
    new_theta.block_until_ready()
    time_update = time.perf_counter() - update_start
    return unravel(new_theta), {
        "lr": lr,
        "grad_norm": grad_norm_value,
        "delta_norm": delta_norm,
        "applied_update_norm": float(jnp.linalg.norm(applied_delta)),
        "cg_iters": -1 if cg_info is None else int(cg_info),
        "trust_region_scale": float(trust_scale),
        "fisher_cond": fisher_cond,
        "fisher_shifted_cond": fisher_shifted_cond,
        "fisher_min_eig": fisher_min_eig,
        "fisher_max_eig": fisher_max_eig,
        "time_dmat": time_dmat,
        "time_fisher": time_fisher,
        "time_cg": time_cg,
        "time_update": time_update,
        "time_sr_total": time.perf_counter() - total_start,
    }


def signed_gradient_update(
    params,
    grad_tree,
    step: int,
    opt_cfg: OptimizerConfig,
    total_steps: int,
):
    start = time.perf_counter()
    theta, unravel = ravel_pytree(params)
    grad_flat, _ = ravel_pytree(grad_tree)
    grad_sign = jnp.sign(grad_flat)
    lr = learning_rate_at_step(opt_cfg, step, total_steps)
    new_theta = theta - lr * grad_sign
    new_theta.block_until_ready()
    return unravel(new_theta), {
        "lr": lr,
        "grad_norm": float(jnp.linalg.norm(grad_flat)),
        "grad_sign_norm": float(jnp.linalg.norm(grad_sign)),
        "time_update": time.perf_counter() - start,
    }


def make_optax_optimizer(opt_cfg: OptimizerConfig, total_steps: int):
    if optax is None:
        raise RuntimeError(
            f"{opt_cfg.name} optimizer requires optax. Install optax==0.2.5 "
            "in the active JAX environment."
        )
    lr_schedule = make_learning_rate_schedule(opt_cfg, total_steps)

    if opt_cfg.name == "adam":
        return optax.adam(
            learning_rate=lr_schedule,
            b1=opt_cfg.adam_b1,
            b2=opt_cfg.adam_b2,
            eps=opt_cfg.adam_eps,
        )
    if opt_cfg.name == "adamw":
        return optax.adamw(
            learning_rate=lr_schedule,
            b1=opt_cfg.adam_b1,
            b2=opt_cfg.adam_b2,
            eps=opt_cfg.adam_eps,
            weight_decay=opt_cfg.adam_weight_decay,
        )
    if opt_cfg.name == "muon":
        return optax.contrib.muon(
            learning_rate=lr_schedule,
            beta=opt_cfg.muon_beta,
            ns_steps=opt_cfg.muon_ns_steps,
            weight_decay=opt_cfg.muon_weight_decay,
            adaptive=opt_cfg.muon_adaptive,
        )
    raise ValueError(f"Unsupported Optax optimizer: {opt_cfg.name}")


def perturb_params(params, scale: float):
    def add_noise(x):
        if hasattr(x, "dtype") and jnp.issubdtype(
            x.dtype, jnp.floating
        ):
            return x + scale * jax.random.normal(
                jax.random.PRNGKey(0), x.shape, x.dtype
            )
        return x

    return jax.tree_util.tree_map(add_noise, params)


def init_optimizer_state(
    params, opt_cfg: OptimizerConfig, total_steps: int
) -> OptimizerState:
    if opt_cfg.name not in {"muon", "adam", "adamw"}:
        return OptimizerState()
    optimizer = make_optax_optimizer(opt_cfg, total_steps)
    return OptimizerState(optimizer, optimizer.init(params))


def optax_update(
    params,
    grad_tree,
    optimizer_state: OptimizerState,
    step: int,
    opt_cfg: OptimizerConfig,
    total_steps: int,
):
    start = time.perf_counter()
    updates, new_state = optimizer_state.optimizer.update(
        grad_tree, optimizer_state.state, params
    )
    new_params = optax.apply_updates(params, updates)
    new_params, updates = block_until_ready_tree((new_params, updates))
    grad_flat, _ = ravel_pytree(grad_tree)
    update_flat, _ = ravel_pytree(updates)
    return (
        new_params,
        OptimizerState(optimizer_state.optimizer, new_state),
        {
            "lr": learning_rate_from_count(
                opt_cfg, int(step) - 1, total_steps
            ),
            "grad_norm": float(
                jnp.linalg.norm(grad_flat.astype(jnp.float64))
            ),
            "applied_update_norm": float(
                jnp.linalg.norm(update_flat.astype(jnp.float64))
            ),
            "time_update": time.perf_counter() - start,
        },
    )


def apply_optimizer_update(
    params,
    states,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    opt_cfg: OptimizerConfig,
    optimizer_state: OptimizerState,
    total_steps: int,
    metric_positions=None,
    metric_batch_size: int | None = None,
    site_bank=None,
    site_bank_cache=None,
):
    diag_stats = {}
    if opt_cfg.name in {"muon", "adam", "adamw"}:
        new_params, optimizer_state, stats = optax_update(
            params,
            grad_tree,
            optimizer_state,
            step,
            opt_cfg,
            total_steps,
        )
        diag_stats = kfac_condition_diagnostics(
            params,
            states,
            net_cfg,
            opt_cfg,
            metric_positions=metric_positions,
        )
        stats.update(diag_stats)
        return new_params, optimizer_state, stats
    if opt_cfg.name == "signed_gradient":
        new_params, stats = signed_gradient_update(
            params, grad_tree, step, opt_cfg, total_steps
        )
    elif opt_cfg.name == "sr":
        new_params, stats = minsr_update(
            params,
            states,
            grad_tree,
            step,
            net_cfg,
            opt_cfg,
            total_steps,
            metric_positions=metric_positions,
            metric_batch_size=metric_batch_size,
        )
    else:
        if (
            net_cfg.architecture == "mps"
            and net_cfg.mps_tensor_generation == "direct"
        ):
            new_params, stats = direct_mps_natural_gradient_update(
                params,
                states,
                grad_tree,
                step,
                net_cfg,
                opt_cfg,
                total_steps,
                metric_positions=metric_positions,
            )
        elif (
            net_cfg.architecture == "mps"
            and net_cfg.mps_tensor_generation == "site_head_backflow"
        ):
            new_params, stats = site_head_backflow_kfac_update(
                params,
                states,
                grad_tree,
                step,
                net_cfg,
                opt_cfg,
                total_steps,
                metric_positions=metric_positions,
            )
        elif net_cfg.architecture in {"mlp", "mps"}:
            new_params, stats = kfac_natural_gradient_update(
                params,
                states,
                grad_tree,
                step,
                net_cfg,
                opt_cfg,
                total_steps,
                metric_positions=metric_positions,
                metric_batch_size=metric_batch_size,
                site_bank=site_bank,
                site_bank_cache=site_bank_cache,
            )
        else:
            new_params, stats = natural_gradient_update(
                params,
                states,
                grad_tree,
                step,
                net_cfg,
                opt_cfg,
                total_steps,
                metric_positions=metric_positions,
            )
    if stats.get("optimizer_impl") != "kfac":
        diag_stats = kfac_condition_diagnostics(
            params,
            states,
            net_cfg,
            opt_cfg,
            metric_positions=metric_positions,
        )
        stats.update(diag_stats)
    return new_params, optimizer_state, stats


def apply_continuous_optimizer_update(
    params,
    states,
    coordinates,
    active_rows,
    grad_tree,
    step: int,
    net_cfg: NetworkConfig,
    continuous_cfg,
    opt_cfg: OptimizerConfig,
    optimizer_state: OptimizerState,
    total_steps: int,
):
    if opt_cfg.name in {"muon", "adam", "adamw"}:
        return optax_update(
            params,
            grad_tree,
            optimizer_state,
            step,
            opt_cfg,
            total_steps,
        )
    if opt_cfg.name == "signed_gradient":
        new_params, stats = signed_gradient_update(
            params, grad_tree, step, opt_cfg, total_steps
        )
    else:
        new_params, stats = continuous_natural_gradient_update(
            params,
            states,
            coordinates,
            active_rows,
            grad_tree,
            step,
            net_cfg,
            continuous_cfg,
            opt_cfg,
            total_steps,
        )
    return new_params, optimizer_state, stats


def optimizer_log_suffix(
    stats: dict, time_update_total: float, opt_cfg: OptimizerConfig
) -> str:
    lr_text = f" lr={stats['lr']:.3e}" if "lr" in stats else ""
    cond_text = (
        ""
        if "kfac_cond_max" not in stats
        or not np.isfinite(stats["kfac_cond_max"])
        else f" KFACcond={stats['kfac_cond_max']:.3e}"
    )
    if stats.get("optimizer_impl") == "kfac":
        return (
            f"opt=kfac grad={stats['grad_norm']:.3e} "
            f"|delta|={stats['delta_norm']:.3e} "
            f"|dtheta|={stats['applied_update_norm']:.3e}{lr_text}"
            f"{cond_text} tr={stats['trust_region_scale']:.3e} "
            f"time(opt={time_update_total:.3f}s, "
            f"factors={stats['time_factors']:.3f}s, "
            f"solve={stats['time_solve']:.3f}s, "
            f"update={stats['time_update']:.3f}s)"
        )
    if stats.get("optimizer_impl") == "minsr":
        return (
            f"opt=sr solver=minsr grad={stats['grad_norm']:.3e} "
            f"|delta|={stats['delta_norm']:.3e} "
            f"|dtheta|={stats['applied_update_norm']:.3e}{lr_text} "
            f"samples={stats['sr_samples']}/{stats['sr_available_samples']} "
            f"jac_batch={stats['sr_jacobian_batch_size']} "
            f"kernel={stats['sr_kernel_dim']} "
            f"tr={stats['trust_region_scale']:.3e} "
            f"time(opt={time_update_total:.3f}s, "
            f"dmat={stats['time_dmat']:.3f}s, "
            f"design={stats['time_design']:.3f}s, "
            f"solve={stats['time_solve']:.3f}s, "
            f"update={stats['time_update']:.3f}s)"
        )
    if opt_cfg.name == "signed_gradient":
        return (
            f"opt=signed_gradient grad={stats['grad_norm']:.3e} "
            f"|sign|={stats['grad_sign_norm']:.3e}{lr_text} "
            f"time(update={stats['time_update']:.3f}s, "
            f"opt={time_update_total:.3f}s)"
        )
    if opt_cfg.name in {"muon", "adam", "adamw"}:
        return (
            f"opt={opt_cfg.name} grad={stats['grad_norm']:.3e} "
            f"|dtheta|={stats['applied_update_norm']:.3e}{lr_text}"
            f"{cond_text} "
            f"time(update={stats['time_update']:.3f}s, "
            f"opt={time_update_total:.3f}s)"
        )
    fisher_cond_text = (
        ""
        if "fisher_cond" not in stats
        or not np.isfinite(stats["fisher_cond"])
        else f" Fcond={stats['fisher_cond']:.3e}"
    )
    return (
        f"opt=natural_gradient grad={stats['grad_norm']:.3e} "
        f"|delta|={stats['delta_norm']:.3e} "
        f"|dtheta|={stats['applied_update_norm']:.3e}{lr_text}"
        f"{fisher_cond_text} "
        f"cg={stats['cg_iters']} tr={stats['trust_region_scale']:.3e} "
        f"time(opt={time_update_total:.3f}s, "
        f"dmat={stats['time_dmat']:.3f}s, "
        f"fisher={stats['time_fisher']:.3f}s, "
        f"cg={stats['time_cg']:.3f}s, "
        f"update={stats['time_update']:.3f}s)"
    )
