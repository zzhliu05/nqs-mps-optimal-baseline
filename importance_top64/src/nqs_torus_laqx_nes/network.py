"""JAX neural wavefunctions and projected diagonalization utilities."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import os as _os

from .config import NetworkConfig


def _apply_mps_backflow_triton_first_layer(params, states, activation: str):
    """Apply the Site Backflow MLP with an opt-in Triton first layer."""

    from .triton_kernels import structured_backflow_preactivation_triton

    layers = params["tensor_mlp"]
    work_dtype = layers[0]["w"].dtype
    first = layers[0]
    h = structured_backflow_preactivation_triton(
        states, first["w"], first["b"]
    )
    if "ln_scale" in first:
        h = (
            jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                work_dtype
            )
            * first["ln_scale"]
            + first["ln_bias"]
        )
    h = activation_fn(h, activation)
    for layer in layers[1:-1]:
        h = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            h = (
                jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                    work_dtype
                )
                * layer["ln_scale"]
                + layer["ln_bias"]
            )
        h = activation_fn(h, activation)
    return (h @ layers[-1]["w"] + layers[-1]["b"]).astype(jnp.float32)


def _apply_mps_backflow_structured_first_layer(
    params, states, activation: str
):
    """Apply the exact backflow input structure without materializing BxNx2N."""
    layers = params["tensor_mlp"]
    first = layers[0]
    work_dtype = first["w"].dtype
    num_sites = int(states.shape[1])
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    occupancy_work = occupancy.astype(work_dtype)
    weights = first["w"]
    global_weights = weights[num_sites + 2 :]
    global_projection = occupancy_work @ global_weights
    return _apply_mps_backflow_from_projection(
        params, occupancy, global_projection, activation
    )


def _backflow_first_preactivation_reuse_impl(first, states):
    """Reuse the global dot, rounding only after the full first-layer sum."""
    if not hasattr(jax.lax, "optimization_barrier"):
        raise RuntimeError("reuse first layer requires JAX with lax.optimization_barrier")
    n = states.shape[1]
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    weights = first["w"]
    projected = jax.lax.dot_general(
        occupancy.astype(weights.dtype), weights[n + 2:],
        (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
    )
    pre = (projected[:, None, :] + weights[:n].astype(jnp.float32)[None, :, :]
           + weights[n:n + 2].astype(jnp.float32)[occupancy])
    # Dense BF16 dot returns BF16 before the separate bias addition.
    rounded = jax.lax.optimization_barrier(pre.astype(weights.dtype))
    return rounded + first["b"]


@jax.custom_vjp
def _backflow_first_preactivation_reuse(first, states):
    return _backflow_first_preactivation_reuse_impl(first, states)


def _backflow_first_preactivation_reuse_fwd(first, states):
    return _backflow_first_preactivation_reuse_impl(first, states), (first, states)


def _backflow_first_preactivation_reuse_bwd(residual, cotangent):
    first, states = residual

    def dense_reference(layer, samples):
        inputs = mps_selected_tensor_inputs(samples, "backflow").astype(layer["w"].dtype)
        value = inputs.reshape(-1, inputs.shape[-1]) @ layer["w"] + layer["b"]
        return value.reshape(samples.shape[0], samples.shape[1], -1)

    # Neighbor evaluation has no backward. Preserve the original dense VJP
    # on training samples instead of changing BF16 gradient accumulation order.
    _, pullback = jax.vjp(dense_reference, first, states)
    return pullback(cotangent)


_backflow_first_preactivation_reuse.defvjp(
    _backflow_first_preactivation_reuse_fwd, _backflow_first_preactivation_reuse_bwd
)


def _apply_mps_backflow_reuse_first_layer(params, states, activation):
    layers = params["tensor_mlp"]
    first = layers[0]
    if _os.environ.get("NQS_NEIGHBOR_FIRST_LAYER_FUSED") == "1":
        if activation != "silu" or "ln_scale" not in first:
            raise ValueError("fused reuse requires LayerNorm and SiLU")
        from .triton_kernels import reuse_first_layernorm_silu
        h = reuse_first_layernorm_silu(first, states)
        fused_second = (
            _os.environ.get("NQS_NEIGHBOR_SECOND_GEMM_FUSED") == "1"
        )
        fused_second_third = (
            _os.environ.get("NQS_NEIGHBOR_GEMM23_FUSED") == "1"
        )
        if fused_second or fused_second_third:
            if len(layers) != 4 or activation != "silu":
                raise ValueError(
                    "second GEMM fusion requires the maintained backflow SiLU MLP"
                )
            second = layers[1]
            if "ln_scale" not in second:
                raise ValueError("second GEMM fusion requires LayerNorm")
            from .triton_kernels import gemm_layernorm_silu_pallas

            h = gemm_layernorm_silu_pallas(
                h,
                second["w"],
                second["b"],
                second["ln_scale"],
                second["ln_bias"],
                block_rows=16,
                block_k=16,
                num_warps=8,
                num_stages=2,
            )
            if fused_second_third:
                third = layers[2]
                h = gemm_layernorm_silu_pallas(
                    h,
                    third["w"],
                    third["b"],
                    third["ln_scale"],
                    third["ln_bias"],
                    block_rows=64,
                    block_k=64,
                    num_warps=8,
                    num_stages=2,
                )
                return _apply_mlp(layers[3:], h, activation)
            return _apply_mlp(layers[2:], h, activation)
        outputs = _apply_mlp(layers[1:], h, activation)
        if _os.environ.get("NQS_NEIGHBOR_OUTPUT_BF16") == "1":
            outputs = outputs.astype(jnp.bfloat16)
        return outputs
    h = _backflow_first_preactivation_reuse(first, states)
    h = h.reshape((-1, h.shape[-1]))
    if "ln_scale" in first and _os.environ.get("NQS_PALLAS_LAYERNORM_SILU"):
        if activation != "silu":
            raise ValueError("fused first layer requires silu")
        from .triton_kernels import layernorm_silu_pallas
        h = layernorm_silu_pallas(h, first["ln_scale"], first["ln_bias"])
    else:
        if "ln_scale" in first:
            h = (jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(h.dtype)
                 * first["ln_scale"] + first["ln_bias"])
        h = activation_fn(h, activation)
    outputs = _apply_mlp(layers[1:], h, activation)
    if _os.environ.get("NQS_NEIGHBOR_OUTPUT_BF16") == "1":
        outputs = outputs.astype(jnp.bfloat16)
    return outputs


def _apply_mps_backflow_from_projection(
    params, occupancy, global_projection, activation: str
):
    """Finish the Site Backflow MLP from its exact global projection."""

    layers = params["tensor_mlp"]
    first = layers[0]
    work_dtype = first["w"].dtype
    num_sites = int(occupancy.shape[1])
    weights = first["w"]
    site_weights = weights[:num_sites]
    local_weights = weights[num_sites : num_sites + 2]
    h = (
        site_weights[None, :, :]
        + local_weights[occupancy]
        + global_projection[:, None, :]
        + first["b"][None, None, :]
    )
    if "ln_scale" in first:
        h = (
            jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                work_dtype
            )
            * first["ln_scale"]
            + first["ln_bias"]
        )
    h = activation_fn(h, activation)
    for layer in layers[1:-1]:
        h = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            h = (
                jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                    work_dtype
                )
                * layer["ln_scale"]
                + layer["ln_bias"]
            )
        h = activation_fn(h, activation)
    return (h @ layers[-1]["w"] + layers[-1]["b"]).astype(jnp.float32)


def block_until_ready_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def occupation_states_from_positions(
    states, positions, cfg: NetworkConfig, dtype=jnp.float32
):
    """Gather occupations, unpacking compact uint64 sector states on demand."""
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if states.ndim != 1:
        return states[positions].astype(dtype)
    if cfg.num_sites is None:
        raise ValueError(
            "NetworkConfig.num_sites is required for packed sector states."
        )
    selected = states[positions].astype(jnp.uint64)
    shifts = jnp.arange(cfg.num_sites, dtype=jnp.uint64)
    return ((selected[..., None] >> shifts) & jnp.uint64(1)).astype(dtype)


def activation_fn(x, name: str):
    if name == "tanh":
        return jnp.tanh(x)
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "leaky_relu":
        return jax.nn.leaky_relu(x)
    if name == "silu":
        return jax.nn.silu(x)
    if name == "poly":
        return x - (x**3) / 3.0
    return jax.nn.relu(x)



def _init_std_for(cfg: NetworkConfig, fan_in: int) -> float:
    """Weight-init std: Kaiming/He sqrt(2/fan_in) or the fixed cfg.init_std."""
    if getattr(cfg, "init_scheme", "std") == "kaiming":
        return float(np.sqrt(2.0 / max(1, int(fan_in))))
    return float(cfg.init_std)


def init_mlp(
    key,
    in_dim: int,
    hidden_sizes: tuple[int, ...],
    out_dim: int,
    cfg: NetworkConfig,
    out_bias=None,
):
    params = []
    keys = jax.random.split(key, len(hidden_sizes) + 1)
    dim = in_dim
    bf16 = jnp.bfloat16
    for layer_key, width in zip(keys[:-1], hidden_sizes):
        w_key, _ = jax.random.split(layer_key)
        layer = {
            "w": _init_std_for(cfg, dim)
            * jax.random.normal(w_key, (dim, width), dtype=bf16),
            "b": jnp.zeros((width,), dtype=bf16),
        }
        if cfg.layer_norm:
            layer["ln_scale"] = jnp.ones((width,), dtype=bf16)
            layer["ln_bias"] = jnp.zeros((width,), dtype=bf16)
        params.append(layer)
        dim = width
    w_key, _ = jax.random.split(keys[-1])
    bias = (
        jnp.zeros((out_dim,), dtype=bf16)
        if out_bias is None
        else jnp.asarray(out_bias, dtype=bf16)
    )
    params.append(
        {
            "w": _init_std_for(cfg, dim)
            * jax.random.normal(w_key, (dim, out_dim), dtype=bf16),
            "b": bias,
        }
    )
    return params


def _init_linear(key, in_dim: int, out_dim: int, cfg: NetworkConfig):
    bf16 = jnp.bfloat16
    return {
        "w": _init_std_for(cfg, in_dim)
        * jax.random.normal(key, (in_dim, out_dim), dtype=bf16),
        "b": jnp.zeros((out_dim,), dtype=bf16),
    }


def _init_conv1d(key, width: int, cutoff: int, cfg: NetworkConfig):
    bf16 = jnp.bfloat16
    return {
        "kernel": cfg.init_std
        * jax.random.normal(key, (cutoff, width, width), dtype=bf16),
        "b": jnp.zeros((width,), dtype=bf16),
    }


def _init_attention_block(key, width: int, cfg: NetworkConfig):
    """Initialize one site-token self-attention block."""
    qkv_key, out_key = jax.random.split(key)
    return {
        "ln_scale": jnp.ones((width,), dtype=jnp.bfloat16),
        "ln_bias": jnp.zeros((width,), dtype=jnp.bfloat16),
        "qkv": _init_linear(qkv_key, width, 3 * width, cfg),
        "out": _init_linear(out_key, width, width, cfg),
    }


def init_vit(key, cfg: NetworkConfig):
    embed_dim = cfg.vit_embed_dim
    ffn_dim = embed_dim * cfg.vit_mlp_ratio
    keys = iter(jax.random.split(key, 4 + 4 * cfg.vit_num_layers))
    params = {
        "patch_embed": _init_linear(
            next(keys), cfg.vit_patch_size, embed_dim, cfg
        ),
        "position_proj": _init_linear(
            next(keys), 2 * cfg.vit_fourier_modes, embed_dim, cfg
        ),
        "blocks": [],
    }
    for _ in range(cfg.vit_num_layers):
        params["blocks"].append(
            {
                "ln1_scale": jnp.ones((embed_dim,), dtype=jnp.bfloat16),
                "ln1_bias": jnp.zeros((embed_dim,), dtype=jnp.bfloat16),
                "qkv": _init_linear(
                    next(keys), embed_dim, 3 * embed_dim, cfg
                ),
                "attention_out": _init_linear(
                    next(keys), embed_dim, embed_dim, cfg
                ),
                "ln2_scale": jnp.ones((embed_dim,), dtype=jnp.bfloat16),
                "ln2_bias": jnp.zeros((embed_dim,), dtype=jnp.bfloat16),
                "ffn_in": _init_linear(
                    next(keys), embed_dim, ffn_dim, cfg
                ),
                "ffn_out": _init_linear(
                    next(keys), ffn_dim, embed_dim, cfg
                ),
            }
        )
    params["final_norm"] = {
        "scale": jnp.ones((embed_dim,), dtype=jnp.bfloat16),
        "bias": jnp.zeros((embed_dim,), dtype=jnp.bfloat16),
    }
    params["head"] = _init_linear(next(keys), embed_dim, 2 * cfg.k, cfg)
    return params


def init_ace_nes_1d(key, num_sites: int, cfg: NetworkConfig):
    keys = iter(jax.random.split(key, 4 + cfg.nes_layers))
    hidden = cfg.nes_hidden
    params = {
        "embedding": cfg.init_std
        * jax.random.normal(
            next(keys), (2, hidden), dtype=jnp.bfloat16
        ),
        "blocks": [],
    }
    for _ in range(cfg.nes_layers):
        params["blocks"].append(
            {
                "ln_scale": jnp.ones((hidden,), dtype=jnp.bfloat16),
                "ln_bias": jnp.zeros((hidden,), dtype=jnp.bfloat16),
                "conv": _init_conv1d(
                    next(keys), hidden, cfg.nes_cutoff, cfg
                ),
            }
        )
    mlp_sizes = (cfg.nes_mlp_hidden,) * cfg.nes_mlp_layers
    params["site_mlp"] = init_mlp(
        next(keys), hidden, mlp_sizes, cfg.nes_mlp_hidden, cfg
    )
    orbital_dim = cfg.k * num_sites
    params["output_r"] = _init_linear(
        next(keys), cfg.nes_mlp_hidden, orbital_dim, cfg
    )
    params["output_i"] = _init_linear(
        next(keys), cfg.nes_mlp_hidden, orbital_dim, cfg
    )
    return params


def init_mps(key, num_sites: int, cfg: NetworkConfig):
    bond_dim = cfg.mps_bond_dim
    matrix_size = cfg.k * bond_dim * bond_dim
    if cfg.mps_tensor_generation == "direct":
        shape = (num_sites, 2, cfg.k, bond_dim, bond_dim, 2)
        tensors = cfg.init_std * jax.random.normal(
            key, shape, dtype=jnp.float32
        )
        diagonal = jnp.eye(bond_dim, dtype=jnp.float32) / jnp.sqrt(
            float(bond_dim)
        )
        tensors = tensors.at[..., 0].add(
            diagonal[None, None, None, :, :]
        )
        return {"tensors": tensors.astype(jnp.bfloat16)}
    if cfg.mps_tensor_generation in {
        "local_backflow", "local_conv_backflow"
    }:
        embedding_key, conv_key, mlp_key = jax.random.split(key, 3)
        embed_dim = cfg.mps_backflow_embed_dim
        out_bias = np.zeros((2 * matrix_size,), dtype=np.float32)
        diagonal_scale = 1.0 / np.sqrt(float(bond_dim))
        for state_index in range(cfg.k):
            offset = state_index * bond_dim * bond_dim
            for index in range(bond_dim):
                out_bias[offset + index * bond_dim + index] = diagonal_scale
        result = {
            "embedding": (
                cfg.init_std
                * jax.random.normal(
                    embedding_key,
                    (num_sites, 2, embed_dim),
                    dtype=jnp.bfloat16,
                )
            ),
        }
        if cfg.mps_tensor_generation == "local_conv_backflow":
            kernel_size = int(_os.environ.get("NQS_LOCAL_CONV_KERNEL_SIZE", "3"))
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("local convolution kernel size must be positive and odd")
            result["conv"] = _init_conv1d(
                conv_key, embed_dim, kernel_size, cfg
            )
            result["local_mlp"] = init_mlp(
                mlp_key,
                embed_dim,
                (embed_dim,),
                2 * matrix_size,
                cfg,
                out_bias=out_bias,
            )
        else:
            result["local_mlp"] = init_mlp(
                mlp_key,
                3 * embed_dim,
                cfg.hidden_sizes,
                2 * matrix_size,
                cfg,
                out_bias=out_bias,
            )
        return result
    if cfg.mps_tensor_generation == "site_head_backflow":
        trunk_key, beta_key, head_key = jax.random.split(key, 3)
        latent_dim = cfg.mps_backflow_embed_dim
        out_bias = np.zeros((2 * matrix_size,), dtype=np.float32)
        diagonal_scale = 1.0 / np.sqrt(float(bond_dim))
        for state_index in range(cfg.k):
            offset = state_index * bond_dim * bond_dim
            for index in range(bond_dim):
                out_bias[offset + index * bond_dim + index] = diagonal_scale
        return {
            "trunk": init_mlp(
                trunk_key,
                num_sites,
                cfg.hidden_sizes,
                latent_dim,
                cfg,
            ),
            "gamma": jnp.ones(
                (num_sites, 2, latent_dim), dtype=jnp.bfloat16
            ),
            "beta": (
                cfg.init_std
                * jax.random.normal(
                    beta_key,
                    (num_sites, 2, latent_dim),
                    dtype=jnp.bfloat16,
                )
            ),
            "head": init_mlp(
                head_key,
                latent_dim,
                (latent_dim,),
                2 * matrix_size,
                cfg,
                out_bias=out_bias,
            ),
        }
    global_generation = cfg.mps_tensor_generation == "global_backflow"
    site_count = num_sites if global_generation else 1
    out_bias = np.zeros(
        (site_count, 2 * matrix_size), dtype=np.float32
    )
    diagonal_scale = 1.0 / np.sqrt(float(bond_dim))
    for site_index in range(site_count):
        for state_index in range(cfg.k):
            offset = state_index * bond_dim * bond_dim
            for index in range(bond_dim):
                out_bias[
                    site_index, offset + index * bond_dim + index
                ] = diagonal_scale
    hidden_sizes = cfg.hidden_sizes
    if cfg.mps_tensor_generation == "block_backflow":
        block_size = int(cfg.mps_block_size)
        if num_sites % block_size:
            raise ValueError("num_sites must be divisible by mps_block_size.")
        input_dim = 2 * num_sites + block_size
        hidden_sizes = (cfg.mps_backflow_embed_dim,) + hidden_sizes
        out_dim = 2 * matrix_size
        block_bias = np.zeros((out_dim,), dtype=np.float32)
        # compensate the 1/M shorter chain: one matrix must carry M sites' worth
        diagonal_scale = np.sqrt(float(block_size)) / np.sqrt(float(bond_dim))
        for state_index in range(cfg.k):
            off = state_index * bond_dim * bond_dim
            for index in range(bond_dim):
                block_bias[off + index * bond_dim + index] = diagonal_scale
        tensor_mlp = init_mlp(
            key, input_dim, hidden_sizes, out_dim, cfg, out_bias=block_bias
        )
        return {"tensor_mlp": tensor_mlp}
    if global_generation:
        input_dim = num_sites
    elif cfg.mps_tensor_generation == "backflow":
        input_dim = 2 * num_sites + 2
        hidden_sizes = (cfg.mps_backflow_embed_dim,) + hidden_sizes
    else:
        input_dim = num_sites + 2
    tensor_mlp = init_mlp(
        key,
        input_dim,
        hidden_sizes,
        site_count * 2 * matrix_size,
        cfg,
        out_bias=out_bias.reshape((-1,)),
    )
    if cfg.mps_tensor_generation == "backflow":
        embedding = dict(tensor_mlp[0])
        embedding["w"] = embedding["w"].at[num_sites + 2 :, :].set(0)
        tensor_mlp[0] = embedding
    result = {"tensor_mlp": tensor_mlp}
    if cfg.mps_attention_blocks:
        attention_width = int(tensor_mlp[0]["w"].shape[1])
        if attention_width % cfg.mps_attention_heads:
            raise ValueError(
                "the first tensor-MPS hidden width must be divisible by "
                "mps_attention_heads"
            )
        attention_keys = jax.random.split(
            jax.random.fold_in(key, 7919), cfg.mps_attention_blocks
        )
        result["attention_blocks"] = [
            _init_attention_block(attention_key, attention_width, cfg)
            for attention_key in attention_keys
        ]
    return result


def mps_backflow_embedding_stats(params, num_sites: int) -> dict[str, float]:
    """Measure full-occupation weights in a per-site backflow embedding."""
    weights = np.asarray(
        jax.device_get(params["mps"]["tensor_mlp"][0]["w"]),
        dtype=np.float64,
    )
    split = num_sites + 2
    expected_rows = 2 * num_sites + 2
    if weights.ndim != 2 or weights.shape[0] != expected_rows:
        raise ValueError(
            "backflow embedding weight shape must be "
            f"[{expected_rows}, embed_dim], got {weights.shape}."
        )

    local = weights[:split]
    backflow = weights[split:]
    local_power = float(np.vdot(local, local).real)
    backflow_power = float(np.vdot(backflow, backflow).real)
    total_power = local_power + backflow_power
    local_norm = float(np.sqrt(local_power))
    backflow_norm = float(np.sqrt(backflow_power))
    local_rms = float(np.sqrt(local_power / local.size))
    backflow_rms = float(np.sqrt(backflow_power / backflow.size))

    return {
        "mps_embedding_local_norm": local_norm,
        "mps_embedding_backflow_norm": backflow_norm,
        "mps_embedding_backflow_power_fraction": (
            0.0 if total_power == 0.0 else backflow_power / total_power
        ),
        "mps_embedding_backflow_to_local_norm": (
            np.inf if local_norm == 0.0 else backflow_norm / local_norm
        ),
        "mps_embedding_backflow_to_local_rms": (
            np.inf if local_rms == 0.0 else backflow_rms / local_rms
        ),
    }


def init_params(seed: int, num_sites: int, cfg: NetworkConfig):
    key = jax.random.PRNGKey(seed)
    if cfg.architecture == "vit":
        return {"vit": init_vit(key, cfg)}
    if cfg.architecture == "ace_nes_1d":
        return {"nes": init_ace_nes_1d(key, num_sites, cfg)}
    if cfg.architecture == "mps":
        mps = init_mps(key, num_sites, cfg)
        if cfg.mps_jastrow:
            jastrow = init_mlp(
                jax.random.fold_in(key, 104729),
                num_sites,
                cfg.mps_jastrow_hidden_sizes,
                2,
                cfg,
            )
            jastrow[-1] = {
                **jastrow[-1],
                "w": jnp.zeros_like(jastrow[-1]["w"]),
                "b": jnp.zeros_like(jastrow[-1]["b"]),
            }
            mps["jastrow_mlp"] = jastrow
        return {"mps": mps}
    return {
        "pref": init_mlp(
            key, num_sites, cfg.hidden_sizes, 2 * cfg.k, cfg
        )
    }


@partial(jax.jit, static_argnames=("activation",))
def _apply_mlp(params, x, activation: str):
    work_dtype = params[0]["w"].dtype
    h = x.astype(work_dtype)
    fused_layernorm_silu = bool(
        _os.environ.get("NQS_PALLAS_LAYERNORM_SILU")
    )
    if fused_layernorm_silu and activation != "silu":
        raise ValueError(
            "NQS_PALLAS_LAYERNORM_SILU requires activation='silu'."
        )
    for layer in params[:-1]:
        h = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            if fused_layernorm_silu:
                from .triton_kernels import layernorm_silu_pallas

                h = layernorm_silu_pallas(
                    h, layer["ln_scale"], layer["ln_bias"]
                )
                continue
            h = (
                jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                    work_dtype
                )
                * layer["ln_scale"]
                + layer["ln_bias"]
            )
        h = activation_fn(h, activation)
    h = h @ params[-1]["w"] + params[-1]["b"]
    return h.astype(jnp.float32)


@partial(jax.jit, static_argnames=("activation", "residual_blocks"))
def _apply_mlp_residual(params, x, activation: str, residual_blocks: int):
    """MLP with optional identity skips between equal-width hidden layers."""
    work_dtype = params[0]["w"].dtype
    h = x.astype(work_dtype)
    residual_used = 0
    for layer in params[:-1]:
        skip = h
        h = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            h = (
                jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                    work_dtype
                )
                * layer["ln_scale"]
                + layer["ln_bias"]
            )
        h = activation_fn(h, activation)
        if (
            residual_used < residual_blocks
            and skip.shape[-1] == h.shape[-1]
        ):
            h = h + skip
            residual_used += 1
    h = h @ params[-1]["w"] + params[-1]["b"]
    return h.astype(jnp.float32)


def _apply_self_attention(h, block, num_heads: int):
    """Apply pre-norm multi-head attention over the site-token axis."""
    width = int(h.shape[-1])
    if width % num_heads:
        raise ValueError("attention width must be divisible by num_heads")
    normalized = _layer_norm(
        h, block["ln_scale"], block["ln_bias"]
    ).astype(block["qkv"]["w"].dtype)
    qkv = normalized @ block["qkv"]["w"] + block["qkv"]["b"]
    batch_size, num_sites, _ = qkv.shape
    head_dim = width // num_heads
    qkv = qkv.reshape((batch_size, num_sites, 3, num_heads, head_dim))
    qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
    query, key, value = qkv[0], qkv[1], qkv[2]
    logits = jnp.einsum(
        "bhid,bhjd->bhij",
        query.astype(jnp.float32),
        key.astype(jnp.float32),
    ) * (head_dim ** -0.5)
    weights = jax.nn.softmax(logits, axis=-1)
    context = jnp.einsum(
        "bhij,bhjd->bhid", weights, value.astype(jnp.float32)
    )
    context = jnp.transpose(context, (0, 2, 1, 3)).reshape(
        (batch_size, num_sites, width)
    )
    output = context.astype(block["out"]["w"].dtype)
    output = output @ block["out"]["w"] + block["out"]["b"]
    return h + output


@partial(
    jax.jit,
    static_argnames=(
        "activation",
        "attention_blocks",
        "attention_heads",
        "residual_blocks",
    ),
)
def _apply_sequence_mlp(
    params,
    x,
    activation: str,
    attention_blocks: int,
    attention_heads: int,
    residual_blocks: int,
):
    """Apply a per-site MLP with optional attention and residual blocks."""
    work_dtype = params["tensor_mlp"][0]["w"].dtype
    h = x.astype(work_dtype)
    residual_used = 0
    attention_index = 0
    for layer in params["tensor_mlp"][:-1]:
        skip = h
        h = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            h = (
                jax.nn.standardize(h.astype(jnp.float32), axis=-1).astype(
                    work_dtype
                )
                * layer["ln_scale"]
                + layer["ln_bias"]
            )
        h = activation_fn(h, activation).astype(work_dtype)
        if (
            residual_used < residual_blocks
            and skip.shape[-1] == h.shape[-1]
        ):
            h = h + skip
            residual_used += 1
        if attention_index < attention_blocks:
            h = _apply_self_attention(
                h,
                params["attention_blocks"][attention_index],
                attention_heads,
            ).astype(work_dtype)
            attention_index += 1
    h = h @ params["tensor_mlp"][-1]["w"] + params["tensor_mlp"][-1]["b"]
    return h.astype(jnp.float32)


def apply_mlp(params, x, cfg: NetworkConfig):
    return _apply_mlp(params, x, cfg.activation)


def patch_occupation_states(states, patch_size: int):
    num_sites = int(states.shape[-1])
    num_patches = (num_sites + patch_size - 1) // patch_size
    padded_sites = num_patches * patch_size
    padded = jnp.pad(
        states,
        ((0, 0), (0, padded_sites - num_sites)),
        mode="constant",
    )
    return padded.reshape((states.shape[0], num_patches, patch_size))


def periodic_fourier_features(
    positions,
    period: int,
    num_modes: int,
):
    positions = jnp.asarray(positions, dtype=jnp.float32)
    modes = jnp.arange(1, num_modes + 1, dtype=jnp.float32)
    angles = (
        2.0
        * jnp.pi
        * positions[:, None]
        * modes[None, :]
        / float(period)
    )
    return jnp.stack(
        (jnp.sin(angles), jnp.cos(angles)), axis=-1
    ).reshape((positions.shape[0], 2 * num_modes))


def periodic_fourier_patch_features(
    num_sites: int,
    patch_size: int,
    num_modes: int,
):
    num_patches = (num_sites + patch_size - 1) // patch_size
    starts = jnp.arange(num_patches, dtype=jnp.float32) * patch_size
    valid_counts = jnp.minimum(
        float(patch_size), float(num_sites) - starts
    )
    centers = starts + 0.5 * (valid_counts - 1.0)
    return periodic_fourier_features(centers, num_sites, num_modes)


def _linear(layer, x):
    return x @ layer["w"] + layer["b"]


def _layer_norm(x, scale, bias):
    normalized = jax.nn.standardize(
        x.astype(jnp.float32), axis=-1
    )
    return (
        normalized * scale.astype(jnp.float32)
        + bias.astype(jnp.float32)
    )


def _periodic_conv1d(hidden, kernel, bias):
    radius = kernel.shape[0] // 2
    padded = jnp.pad(hidden, ((0, 0), (radius, radius), (0, 0)), mode="wrap")
    out = jax.lax.conv_general_dilated(
        padded,
        kernel,
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )
    return out + bias


def _apply_dense_stack(layers, x, activation: str):
    work_dtype = layers[0]["w"].dtype
    h = x.astype(work_dtype)
    for layer in layers[:-1]:
        h = _linear(layer, h)
        if "ln_scale" in layer:
            h = _layer_norm(h, layer["ln_scale"], layer["ln_bias"]).astype(
                work_dtype
            )
        h = activation_fn(h, activation).astype(work_dtype)
    return _linear(layers[-1], h).astype(jnp.float32)


def _apply_vit(
    params,
    states,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    fourier_modes: int,
):
    work_dtype = params["patch_embed"]["w"].dtype
    patches = patch_occupation_states(states, patch_size).astype(work_dtype)
    h = _linear(params["patch_embed"], patches)
    position_features = periodic_fourier_patch_features(
        int(states.shape[1]), patch_size, fourier_modes
    ).astype(work_dtype)
    position_embedding = _linear(
        params["position_proj"], position_features
    )
    h = h + position_embedding[None, :, :]

    head_dim = embed_dim // num_heads
    attention_scale = jnp.asarray(
        head_dim**-0.5, dtype=jnp.float32
    )
    for block_index in range(num_layers):
        block = params["blocks"][block_index]
        normalized = _layer_norm(
            h, block["ln1_scale"], block["ln1_bias"]
        ).astype(work_dtype)
        qkv = _linear(block["qkv"], normalized)
        batch_size, num_patches, _ = qkv.shape
        qkv = qkv.reshape(
            (batch_size, num_patches, 3, num_heads, head_dim)
        )
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, key, value = qkv[0], qkv[1], qkv[2]
        logits = jnp.einsum(
            "bhpd,bhqd->bhpq",
            q.astype(jnp.float32),
            key.astype(jnp.float32),
        )
        attention = jax.nn.softmax(
            logits * attention_scale, axis=-1
        )
        context = jnp.einsum(
            "bhpq,bhqd->bhpd",
            attention,
            value.astype(jnp.float32),
        )
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(
            (batch_size, num_patches, embed_dim)
        )
        h = h + _linear(
            block["attention_out"], context.astype(work_dtype)
        )

        normalized = _layer_norm(
            h, block["ln2_scale"], block["ln2_bias"]
        ).astype(work_dtype)
        hidden = _linear(block["ffn_in"], normalized)
        hidden = jax.nn.gelu(hidden.astype(jnp.float32)).astype(
            work_dtype
        )
        h = h + _linear(block["ffn_out"], hidden)

    pooled = jnp.mean(h.astype(jnp.float32), axis=1)
    pooled = _layer_norm(
        pooled,
        params["final_norm"]["scale"],
        params["final_norm"]["bias"],
    )
    return _linear(params["head"], pooled.astype(work_dtype)).astype(
        jnp.float32
    )


def _apply_ace_nes_1d(
    params,
    states,
    activation: str,
    k: int,
    logphi_real_clip: float,
):
    work_dtype = params["embedding"].dtype
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    hidden = params["embedding"][occupancy]
    for block in params["blocks"]:
        residual = _periodic_conv1d(
            hidden.astype(work_dtype),
            block["conv"]["kernel"],
            block["conv"]["b"],
        )
        hidden = hidden + activation_fn(residual.astype(jnp.float32), activation).astype(
            work_dtype
        )
        hidden = _layer_norm(
            hidden,
            block["ln_scale"],
            block["ln_bias"],
        ).astype(work_dtype)

    hidden = _apply_dense_stack(
        params["site_mlp"], hidden, activation
    ).astype(work_dtype)
    out_r = _linear(params["output_r"], hidden).astype(jnp.float32)
    out_i = _linear(params["output_i"], hidden).astype(jnp.float32)
    orbitals = (out_r + 1j * out_i).reshape(
        (states.shape[0], states.shape[1], k, states.shape[1])
    )
    orbitals = jnp.transpose(orbitals, (0, 2, 1, 3))
    occupied = states.astype(orbitals.real.dtype)
    masked = (
        orbitals
        * occupied[:, None, :, None]
        * occupied[:, None, None, :]
    )
    eye = jnp.eye(states.shape[1], dtype=orbitals.dtype)
    mats = masked + eye[None, None, :, :] * (
        1.0 - occupied[:, None, None, :]
    )
    sign, logabsdet = jnp.linalg.slogdet(mats)
    logabsdet = jnp.clip(
        logabsdet, -logphi_real_clip, logphi_real_clip
    )
    return sign * jnp.exp(logabsdet)


def mps_selected_tensor_inputs(states, tensor_generation: str):
    """Build one tensor-generator input for every sample and MPS site."""
    batch_size, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    site_features = jnp.broadcast_to(
        jnp.eye(num_sites, dtype=jnp.float32)[None, :, :],
        (batch_size, num_sites, num_sites),
    )
    if tensor_generation == "backflow":
        local_occupation = jax.nn.one_hot(
            occupancy, 2, dtype=jnp.float32
        )
        full_occupation = jnp.broadcast_to(
            occupancy[:, None, :].astype(jnp.float32),
            (batch_size, num_sites, num_sites),
        )
        return jnp.concatenate(
            (site_features, local_occupation, full_occupation), axis=-1
        )
    occupation_features = jax.nn.one_hot(
        occupancy, 2, dtype=jnp.float32
    )
    return jnp.concatenate((site_features, occupation_features), axis=-1)


def _local_backflow_outputs(params, states, activation: str):
    """Generate all site tensors from a radius-one periodic local window."""
    states = jnp.asarray(states, dtype=jnp.float32)
    batch_size, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    sites = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    embedded = params["embedding"][sites, occupancy]
    if "conv" in params:
        kernel_size = int(params["conv"]["kernel"].shape[0])
        radius = kernel_size // 2
        window_values = tuple(
            jnp.roll(embedded, shift, axis=1)
            for shift in range(radius, -radius - 1, -1)
        )
        stacked = jnp.stack(window_values, axis=2)
        hidden = embedded + activation_fn(
            jnp.einsum(
                "bnwe,weh->bnh",
                stacked.astype(params["conv"]["kernel"].dtype),
                params["conv"]["kernel"],
            ) + params["conv"]["b"],
            activation,
        )
        local_inputs = hidden
    else:
        window_values = (
            jnp.roll(embedded, 1, axis=1),
            embedded,
            jnp.roll(embedded, -1, axis=1),
        )
        local_inputs = jnp.concatenate(window_values, axis=-1)
    outputs = _apply_mlp(
        params["local_mlp"],
        local_inputs.reshape(
            (batch_size * num_sites, local_inputs.shape[-1])
        ),
        activation,
    )
    return outputs.reshape((batch_size, num_sites, outputs.shape[-1]))


def local_conv_site_tensor_bank(params, cfg: NetworkConfig):
    """Evaluate the kernel-one site generator on its 2*Nphi distinct inputs."""
    bank, _ = local_conv_site_tensor_bank_with_cache(params, cfg)
    return bank


@partial(jax.jit, static_argnames=("cfg",))
def local_conv_site_tensor_bank_with_cache(params, cfg: NetworkConfig):
    """Return the site bank and KFAC activations from one MLP forward."""
    if (
        cfg.architecture != "mps"
        or cfg.mps_tensor_generation != "local_conv_backflow"
        or int(params["mps"]["conv"]["kernel"].shape[0]) != 1
        or cfg.num_sites != int(params["mps"]["embedding"].shape[0])
    ):
        raise ValueError("site tensor bank requires kernel-one local_conv_backflow")
    mps = params["mps"]
    embedded = mps["embedding"]
    hidden = embedded + activation_fn(
        jnp.einsum(
            "nce,eh->nch",
            embedded.astype(mps["conv"]["kernel"].dtype),
            mps["conv"]["kernel"][0],
        ) + mps["conv"]["b"],
        cfg.activation,
    )
    layers = mps["local_mlp"]
    work_dtype = layers[0]["w"].dtype
    h = hidden.reshape((2 * cfg.num_sites, hidden.shape[-1])).astype(work_dtype)
    activations = []
    activation_inputs = []
    ln_caches = []
    fused_ln = bool(_os.environ.get("NQS_PALLAS_LAYERNORM_SILU"))
    for layer in layers[:-1]:
        activations.append(h.astype(jnp.float32))
        pre_linear = h @ layer["w"] + layer["b"]
        if "ln_scale" in layer:
            x32 = pre_linear.astype(jnp.float32)
            mean = jnp.mean(x32, axis=-1, keepdims=True)
            centered = x32 - mean
            inv_std = jax.lax.rsqrt(jnp.mean(centered * centered, axis=-1, keepdims=True) + 1e-5)
            normalized = jax.nn.standardize(x32, axis=-1)
            activation_input = (
                normalized.astype(work_dtype) * layer["ln_scale"]
                + layer["ln_bias"]
            )
            ln_caches.append((normalized, inv_std, layer["ln_scale"].astype(jnp.float32)))
            if fused_ln:
                from .triton_kernels import layernorm_silu_pallas
                if cfg.activation != "silu":
                    raise ValueError("fused layer norm requires silu")
                h = layernorm_silu_pallas(pre_linear, layer["ln_scale"], layer["ln_bias"])
            else:
                h = activation_fn(activation_input, cfg.activation)
        else:
            activation_input = pre_linear
            ln_caches.append(None)
            h = activation_fn(pre_linear, cfg.activation)
        activation_inputs.append(activation_input.astype(jnp.float32))
    activations.append(h.astype(jnp.float32))
    output = (h @ layers[-1]["w"] + layers[-1]["b"]).astype(jnp.float32)
    bank = output.reshape((cfg.num_sites, 2, output.shape[-1]))

    def by_site(value):
        return value.reshape((cfg.num_sites, 2, -1))

    cache = {
        "activations": tuple(by_site(x) for x in activations),
        "activation_inputs": tuple(by_site(x) for x in activation_inputs),
        "ln_caches": tuple(
            None if x is None else (by_site(x[0]), by_site(x[1]), x[2])
            for x in ln_caches
        ),
    }
    return bank, cache


def local_conv_site_wavefunction_from_bank(bank, states, cfg: NetworkConfig):
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    sites = jnp.arange(cfg.num_sites, dtype=jnp.int32)[None, :]
    outputs = bank[sites, occupancy]
    return _contract_local_backflow_outputs(
        outputs, cfg.k, cfg.mps_bond_dim, cfg.logphi_real_clip,
        cfg.mps_contraction,
    )


def _contract_local_backflow_outputs(
    outputs, k: int, bond_dim: int, logphi_real_clip: float,
    contraction: str,
):
    batch_size, num_sites = outputs.shape[:2]
    matrix_size = k * bond_dim * bond_dim
    if _os.environ.get("NQS_PALLAS_REALBLOCK_TRIPLET"):
        if k != 1 or bond_dim > 32 or num_sites % 3:
            raise ValueError("local-backflow Pallas triplet requires k=1, D<=32, and a site count divisible by three")
        from .triton_kernels import persistent_realblock_triplet_pallas
        return persistent_realblock_triplet_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"):
        if k != 1 or bond_dim > 32 or num_sites % 4:
            raise ValueError("local-backflow Pallas quad requires k=1, D<=32, and a site count divisible by four")
        from .triton_kernels import persistent_realblock_quad_pallas
        return persistent_realblock_quad_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_PAIR"):
        if k != 1 or bond_dim > 32:
            raise ValueError("local-backflow Pallas pair requires k=1 and D<=32")
        from .triton_kernels import persistent_realblock_pair_pallas
        return persistent_realblock_pair_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def _apply_mps_local_backflow(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    contraction: str,
):
    outputs = _local_backflow_outputs(params, states, activation)
    return _contract_local_backflow_outputs(
        outputs, k, bond_dim, logphi_real_clip, contraction
    )


@partial(
    jax.jit,
    static_argnames=(
        "activation",
        "k",
        "bond_dim",
        "logphi_real_clip",
        "contraction",
    ),
)
def _local_backflow_neighbor_chunk(
    params,
    center_outputs,
    center_states,
    parent_ids,
    neighbor_states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    contraction: str,
    site_tensor_bank=None,
):
    """Evaluate neighbors by replacing only radius-one affected tensors."""
    batch_size, num_sites = neighbor_states.shape
    matrix_size = k * bond_dim * bond_dim
    occupancy = jnp.clip(jnp.rint(neighbor_states), 0, 1).astype(jnp.int32)
    parent_occupancy = jnp.clip(
        jnp.rint(center_states[parent_ids]), 0, 1
    ).astype(jnp.int32)
    changed = occupancy != parent_occupancy

    # A two-body scattering row changes at most four occupation bits.  Only
    # convolution windows overlapping those bits need to be regenerated.
    kernel_size = (
        int(params["conv"]["kernel"].shape[0]) if "conv" in params else 3
    )
    radius = kernel_size // 2
    offsets = jnp.arange(-radius, radius + 1, dtype=jnp.int32)
    affected = changed
    for shift in range(1, radius + 1):
        affected = (
            affected
            | jnp.roll(changed, shift, axis=1)
            | jnp.roll(changed, -shift, axis=1)
        )
    affected_count = min(4 * kernel_size, num_sites)
    affected_valid, affected_sites = jax.lax.top_k(
        affected.astype(jnp.int32), affected_count
    )

    rows = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    if site_tensor_bank is not None:
        if kernel_size != 1:
            raise ValueError("site tensor bank requires convolution kernel size one")
        replacements = site_tensor_bank[
            affected_sites, occupancy[rows, affected_sites]
        ]
    else:
        window_sites = (
            affected_sites[:, :, None] + offsets[None, None, :]
        ) % num_sites
        window_rows = jnp.arange(batch_size, dtype=jnp.int32)[:, None, None]
        window_occupancy = occupancy[window_rows, window_sites]
        embedded = params["embedding"][window_sites, window_occupancy]
        if "conv" in params:
            local_inputs = embedded[:, :, radius] + activation_fn(
                jnp.einsum(
                    "btwe,weh->bth",
                    embedded.astype(params["conv"]["kernel"].dtype),
                    params["conv"]["kernel"],
                ) + params["conv"]["b"],
                activation,
            )
            local_inputs = local_inputs.reshape(
                (batch_size * affected_count, embedded.shape[-1])
            )
        else:
            local_inputs = embedded.reshape(
                (batch_size * affected_count, 3 * embedded.shape[-1])
            )
        replacements = _apply_mlp(
            params["local_mlp"], local_inputs, activation
        ).reshape((batch_size, affected_count, 2 * matrix_size))

    if _os.environ.get("NQS_LOCAL_BACKFLOW_FUSED_CACHE_MAP"):
        use_triplet = bool(
            _os.environ.get("NQS_PALLAS_REALBLOCK_TRIPLET")
        )
        use_quad = bool(_os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"))
        if not (use_triplet or use_quad):
            raise ValueError(
                "mapped fused local-backflow cache requires the Pallas "
                "triplet or quad kernel"
            )
        divisor = 3 if use_triplet else 4
        if k != 1 or bond_dim > 32 or num_sites % divisor:
            raise ValueError(
                "mapped fused local-backflow cache requires k=1, D<=32, "
                f"and a site count divisible by {divisor}"
            )
        replacement_slots = -jnp.ones(
            (batch_size, num_sites), dtype=jnp.int32
        )
        slot_values = jnp.broadcast_to(
            jnp.arange(affected_count, dtype=jnp.int32)[None, :],
            affected_sites.shape,
        )
        slot_values = jnp.where(affected_valid != 0, slot_values, -1)
        replacement_slots = replacement_slots.at[
            rows, affected_sites
        ].set(slot_values)
        if use_triplet:
            from .triton_kernels import (
                persistent_cached_realblock_triplet_map_pallas,
            )

            return persistent_cached_realblock_triplet_map_pallas(
                center_outputs.astype(jnp.bfloat16),
                parent_ids,
                replacement_slots,
                replacements.astype(jnp.bfloat16),
                logphi_real_clip,
            )
        from .triton_kernels import persistent_cached_realblock_quad_map_pallas

        return persistent_cached_realblock_quad_map_pallas(
            center_outputs.astype(jnp.bfloat16),
            parent_ids,
            replacement_slots,
            replacements.astype(jnp.bfloat16),
            logphi_real_clip,
        )

    if _os.environ.get("NQS_LOCAL_BACKFLOW_FUSED_CACHE"):
        if not _os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"):
            raise ValueError(
                "fused local-backflow cache requires the Pallas quad kernel"
            )
        if k != 1 or bond_dim > 32 or num_sites % 4:
            raise ValueError(
                "fused local-backflow cache requires k=1, D<=32, and a "
                "site count divisible by four"
            )
        from .triton_kernels import persistent_cached_realblock_quad_pallas

        return persistent_cached_realblock_quad_pallas(
            center_outputs.astype(jnp.bfloat16),
            parent_ids,
            affected_sites,
            affected_valid,
            replacements.astype(jnp.bfloat16),
            logphi_real_clip,
        )

    outputs = center_outputs[parent_ids]
    previous = outputs[rows, affected_sites]
    replacements = jnp.where(
        affected_valid[..., None] != 0, replacements, previous
    )
    outputs = outputs.at[rows, affected_sites].set(replacements)
    if _os.environ.get("NQS_PALLAS_REALBLOCK_TRIPLET"):
        if k != 1 or bond_dim > 32 or num_sites % 3:
            raise ValueError(
                "local-backflow Pallas triplet requires k=1, D<=32, and "
                "a site count divisible by three"
            )
        from .triton_kernels import persistent_realblock_triplet_pallas

        return persistent_realblock_triplet_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"):
        if k != 1 or bond_dim > 32 or num_sites % 4:
            raise ValueError(
                "local-backflow Pallas quad requires k=1, D<=32, and "
                "a site count divisible by four"
            )
        from .triton_kernels import persistent_realblock_quad_pallas

        return persistent_realblock_quad_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_PAIR"):
        if k != 1 or bond_dim > 32:
            raise ValueError("local-backflow Pallas pair requires k=1 and D<=32")
        from .triton_kernels import persistent_realblock_pair_pallas

        return persistent_realblock_pair_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def batched_local_backflow_neighbor_wavefunction(
    params,
    center_states,
    neighbor_bitstrings,
    parent_ids,
    cfg: NetworkConfig,
    batch_size: int | None = None,
    site_tensor_bank=None,
):
    """Exact SCALE-style local update for radius-one local-backflow MPS."""
    if (
        cfg.architecture != "mps"
        or cfg.mps_tensor_generation not in {
            "local_backflow", "local_conv_backflow"
        }
    ):
        raise ValueError(
            "local neighbor updates require a local-backflow generator"
        )
    if cfg.k != 1:
        raise ValueError("local neighbor updates currently require k=1")
    effective_batch = int(batch_size or cfg.forward_batch_size)
    if effective_batch <= 0:
        raise ValueError("forward batch size must be positive")

    center_states = jnp.asarray(center_states, dtype=jnp.float32)
    bits = jnp.asarray(neighbor_bitstrings, dtype=jnp.uint64).reshape((-1,))
    parent_ids = jnp.asarray(parent_ids, dtype=jnp.int32).reshape((-1,))
    total = int(bits.shape[0])
    num_sites = int(center_states.shape[1])
    shifts = jnp.arange(num_sites, dtype=jnp.uint64)
    if site_tensor_bank is None:
        center_outputs = _local_backflow_outputs(
            params["mps"], center_states, cfg.activation
        )
    else:
        occupancy = jnp.clip(jnp.rint(center_states), 0, 1).astype(jnp.int32)
        sites = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
        center_outputs = site_tensor_bank[sites, occupancy]

    def evaluate(chunk_bits, chunk_parents):
        neighbor_states = (
            (chunk_bits[:, None] >> shifts[None, :]) & jnp.uint64(1)
        ).astype(jnp.float32)
        return _local_backflow_neighbor_chunk(
            params["mps"],
            center_outputs,
            center_states,
            chunk_parents,
            neighbor_states,
            cfg.activation,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.logphi_real_clip,
            cfg.mps_contraction,
            site_tensor_bank=site_tensor_bank,
        )

    if total <= effective_batch:
        return evaluate(bits, parent_ids)

    full_count = total // effective_batch
    out = jnp.zeros((total, cfg.k), dtype=jnp.complex64)

    def scan_body(carry, index):
        start = index * effective_batch
        chunk_bits = jax.lax.dynamic_slice(
            bits, (start,), (effective_batch,)
        )
        chunk_parents = jax.lax.dynamic_slice(
            parent_ids, (start,), (effective_batch,)
        )
        values = evaluate(chunk_bits, chunk_parents)
        return jax.lax.dynamic_update_slice(carry, values, (start, 0)), None

    out, _ = jax.lax.scan(scan_body, out, jnp.arange(full_count))
    remainder = total - full_count * effective_batch
    if remainder:
        start = full_count * effective_batch
        tail = evaluate(bits[start:], parent_ids[start:])
        out = jax.lax.dynamic_update_slice(out, tail, (start, 0))
    return out


@partial(
    jax.jit,
    static_argnames=("activation", "k", "bond_dim", "logphi_real_clip"),
)
def _backflow_delta_neighbor_chunk(
    params,
    center_states,
    center_global_projection,
    parent_ids,
    neighbor_states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
):
    """Evaluate Site Backflow neighbors with a sparse first-layer delta."""

    occupancy = jnp.clip(jnp.rint(neighbor_states), 0, 1).astype(jnp.int32)
    parent_occupancy = jnp.clip(
        jnp.rint(center_states[parent_ids]), 0, 1
    ).astype(jnp.int32)
    changed_valid, changed_sites = jax.lax.top_k(
        (occupancy != parent_occupancy).astype(jnp.int32), 4
    )
    rows = jnp.arange(occupancy.shape[0], dtype=jnp.int32)[:, None]
    deltas = (
        occupancy[rows, changed_sites] - parent_occupancy[rows, changed_sites]
    ).astype(params["tensor_mlp"][0]["w"].dtype)
    deltas = deltas * changed_valid.astype(deltas.dtype)
    num_sites = int(neighbor_states.shape[1])
    global_weights = params["tensor_mlp"][0]["w"][num_sites + 2 :]
    delta_projection = jnp.sum(
        deltas[..., None] * global_weights[changed_sites], axis=1
    )
    global_projection = (
        center_global_projection[parent_ids] + delta_projection
    )
    outputs = _apply_mps_backflow_from_projection(
        params, occupancy, global_projection, activation
    )
    matrix_size = k * bond_dim * bond_dim
    outputs = outputs.reshape(
        (occupancy.shape[0], num_sites, 2 * matrix_size)
    )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"):
        if k != 1 or bond_dim > 32 or num_sites % 4:
            raise ValueError(
                "backflow delta Pallas quad requires k=1, D<=32, and "
                "a site count divisible by four"
            )
        from .triton_kernels import persistent_realblock_quad_pallas

        return persistent_realblock_quad_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((occupancy.shape[0], num_sites, k, bond_dim, bond_dim))
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, "balanced"
    )


def batched_backflow_delta_neighbor_wavefunction(
    params,
    center_states,
    neighbor_bitstrings,
    parent_ids,
    cfg: NetworkConfig,
    batch_size: int | None = None,
):
    """Evaluate exact Site Backflow neighbors via sparse input deltas."""

    if cfg.architecture != "mps" or cfg.mps_tensor_generation != "backflow":
        raise ValueError("backflow delta updates require plain Site Backflow MPS")
    effective_batch = int(batch_size or cfg.forward_batch_size)
    center_states = jnp.asarray(center_states, dtype=jnp.float32)
    bits = jnp.asarray(neighbor_bitstrings, dtype=jnp.uint64).reshape((-1,))
    parent_ids = jnp.asarray(parent_ids, dtype=jnp.int32).reshape((-1,))
    total = int(bits.shape[0])
    num_sites = int(center_states.shape[1])
    shifts = jnp.arange(num_sites, dtype=jnp.uint64)
    first = params["mps"]["tensor_mlp"][0]
    work_dtype = first["w"].dtype
    center_occupancy = jnp.clip(jnp.rint(center_states), 0, 1).astype(work_dtype)
    global_weights = first["w"][num_sites + 2 :]
    center_global_projection = center_occupancy @ global_weights

    def evaluate(chunk_bits, chunk_parents):
        neighbor_states = (
            (chunk_bits[:, None] >> shifts[None, :]) & jnp.uint64(1)
        ).astype(jnp.float32)
        return _backflow_delta_neighbor_chunk(
            params["mps"],
            center_states,
            center_global_projection,
            chunk_parents,
            neighbor_states,
            cfg.activation,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.logphi_real_clip,
        )

    if total <= effective_batch:
        return evaluate(bits, parent_ids)
    full_count = total // effective_batch
    out = jnp.zeros((total, cfg.k), dtype=jnp.complex64)

    def scan_body(carry, index):
        start = index * effective_batch
        chunk_bits = jax.lax.dynamic_slice(bits, (start,), (effective_batch,))
        chunk_parents = jax.lax.dynamic_slice(
            parent_ids, (start,), (effective_batch,)
        )
        values = evaluate(chunk_bits, chunk_parents)
        return jax.lax.dynamic_update_slice(carry, values, (start, 0)), None

    out, _ = jax.lax.scan(scan_body, out, jnp.arange(full_count))
    remainder = total - full_count * effective_batch
    if remainder:
        start = full_count * effective_batch
        tail = evaluate(bits[start:], parent_ids[start:])
        out = jax.lax.dynamic_update_slice(out, tail, (start, 0))
    return out


def mps_block_tensor_inputs(states, block_size: int):
    """Block-level tensor-generator input.

    Each block of ``block_size`` consecutive MPS sites is described by the
    full occupation (N bits), the block-start one-hot (N) and the periodic
    binary window occ(j .. j+M-1) (M bits).  Returns ``(batch, n_blocks, 2N+M)``.
    """
    states = jnp.asarray(states, dtype=jnp.float32)
    batch_size, num_sites = states.shape
    if num_sites % block_size:
        raise ValueError("num_sites must be divisible by mps_block_size.")
    n_blocks = num_sites // block_size
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.float32)
    block_starts = jnp.arange(n_blocks, dtype=jnp.int32) * block_size
    start_onehot = jnp.eye(num_sites, dtype=jnp.float32)[block_starts]
    offsets = jnp.arange(block_size, dtype=jnp.int32)
    window_idx = (block_starts[:, None] + offsets[None, :]) % num_sites
    window = occupancy[:, window_idx]
    full_occ = jnp.broadcast_to(
        occupancy[:, None, :], (batch_size, n_blocks, num_sites)
    )
    start = jnp.broadcast_to(
        start_onehot[None, :, :], (batch_size, n_blocks, num_sites)
    )
    return jnp.concatenate((full_occ, start, window), axis=-1)


def _apply_mps_block_backflow(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    block_size: int,
    contraction: str,
):
    """backflow MPS whose generator emits M matrices per block at once."""
    states = jnp.asarray(states, dtype=jnp.float32)
    batch_size, num_sites = states.shape
    if num_sites % block_size:
        raise ValueError("num_sites must be divisible by mps_block_size.")
    n_blocks = num_sites // block_size
    matrix_size = k * bond_dim * bond_dim
    inputs = mps_block_tensor_inputs(states, block_size)
    input_dim = int(inputs.shape[-1])
    outputs = _apply_mlp(
        params["tensor_mlp"],
        inputs.reshape((batch_size * n_blocks, input_dim)),
        activation,
    )
    outputs = outputs.reshape((batch_size, n_blocks, 2 * matrix_size))
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, n_blocks, k, bond_dim, bond_dim))
    power = int(_os.environ.get("NQS_BLOCK_POWER", "1"))
    if power > 1:
        powered = matrices
        for _ in range(power - 1):
            powered = jnp.matmul(powered, matrices)
        matrices = powered
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def mps_generated_matrices(params, states, cfg: NetworkConfig):
    """Return generated MPS matrices before contraction."""
    if cfg.architecture != "mps":
        raise ValueError("mps_generated_matrices requires architecture='mps'.")
    if cfg.mps_tensor_generation not in {"sitewise", "backflow"}:
        raise ValueError(
            "matrix diagnostics currently support sitewise or backflow MPS."
        )
    states = jnp.asarray(states, dtype=jnp.float32)
    batch_size, num_sites = states.shape
    inputs = mps_selected_tensor_inputs(states, cfg.mps_tensor_generation)
    input_dim = int(inputs.shape[-1])
    matrix_size = cfg.k * cfg.mps_bond_dim * cfg.mps_bond_dim
    mps_params = params["mps"]
    if cfg.mps_attention_blocks or cfg.mps_residual_blocks:
        outputs = _apply_sequence_mlp(
            mps_params,
            inputs,
            cfg.activation,
            cfg.mps_attention_blocks,
            cfg.mps_attention_heads,
            cfg.mps_residual_blocks,
        )
    else:
        outputs = _apply_mlp(
            mps_params["tensor_mlp"],
            inputs.reshape((batch_size * num_sites, input_dim)),
            cfg.activation,
        )
    outputs = outputs.reshape((batch_size, num_sites, 2 * matrix_size))
    return (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape(
        (
            batch_size,
            num_sites,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.mps_bond_dim,
        )
    )


def mps_backflow_matrix_variation(
    params, states, cfg: NetworkConfig
) -> dict[str, np.ndarray]:
    """Measure matrix variation at fixed site and local occupation."""
    if cfg.mps_tensor_generation != "backflow":
        raise ValueError("matrix variation requires backflow tensor generation.")
    occupations = np.asarray(states, dtype=np.int8)
    matrices = np.asarray(
        jax.device_get(mps_generated_matrices(params, states, cfg)),
        dtype=np.complex128,
    )
    num_sites = occupations.shape[1]
    shape = (num_sites, 2, cfg.k)
    counts = np.zeros((num_sites, 2), dtype=np.int64)
    mean_norm = np.zeros(shape, dtype=np.float64)
    rms_norm = np.zeros(shape, dtype=np.float64)
    rms_deviation = np.zeros(shape, dtype=np.float64)
    relative_rms = np.zeros(shape, dtype=np.float64)
    coefficient_of_variation = np.zeros(shape, dtype=np.float64)

    for site in range(num_sites):
        for occupation_value in range(2):
            mask = occupations[:, site] == occupation_value
            counts[site, occupation_value] = int(np.count_nonzero(mask))
            for state_index in range(cfg.k):
                group = matrices[mask, site, state_index]
                mean_matrix = np.mean(group, axis=0)
                deviations = group - mean_matrix
                second_moment = float(
                    np.mean(np.sum(np.abs(group) ** 2, axis=(-2, -1)))
                )
                variance = float(
                    np.mean(
                        np.sum(np.abs(deviations) ** 2, axis=(-2, -1))
                    )
                )
                current_mean_norm = float(np.linalg.norm(mean_matrix))
                current_rms_norm = float(np.sqrt(max(second_moment, 0.0)))
                current_rms_deviation = float(np.sqrt(max(variance, 0.0)))
                index = (site, occupation_value, state_index)
                mean_norm[index] = current_mean_norm
                rms_norm[index] = current_rms_norm
                rms_deviation[index] = current_rms_deviation
                relative_rms[index] = (
                    0.0
                    if current_rms_norm == 0.0
                    else current_rms_deviation / current_rms_norm
                )
                coefficient_of_variation[index] = (
                    np.inf
                    if current_mean_norm == 0.0
                    else current_rms_deviation / current_mean_norm
                )

    return {
        "counts": counts,
        "mean_norm": mean_norm,
        "rms_norm": rms_norm,
        "rms_deviation": rms_deviation,
        "relative_rms": relative_rms,
        "coefficient_of_variation": coefficient_of_variation,
    }


def _apply_mps(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    tensor_generation: str = "sitewise",
    attention_blocks: int = 0,
    attention_heads: int = 4,
    residual_blocks: int = 0,
    contraction: str = "sequential",
    first_layer_backend: str = "jax",
):
    batch_size, num_sites = states.shape
    matrix_size = k * bond_dim * bond_dim
    if (
        _os.environ.get("NQS_STREAM_FINAL_CONTRACTION")
        and tensor_generation == "backflow"
        and not attention_blocks
        and not residual_blocks
        and first_layer_backend == "jax"
    ):
        inputs = mps_selected_tensor_inputs(states, tensor_generation)
        layers = params["tensor_mlp"]
        work_dtype = layers[0]["w"].dtype
        hidden = inputs.reshape(
            (batch_size * num_sites, inputs.shape[-1])
        ).astype(work_dtype)
        for layer in layers[:-1]:
            hidden = hidden @ layer["w"] + layer["b"]
            if "ln_scale" in layer:
                hidden = (
                    jax.nn.standardize(
                        hidden.astype(jnp.float32), axis=-1
                    ).astype(work_dtype)
                    * layer["ln_scale"]
                    + layer["ln_bias"]
                )
            hidden = activation_fn(hidden, activation)
        hidden = hidden.reshape((batch_size, num_sites, hidden.shape[-1]))
        final = layers[-1]
        product = jnp.broadcast_to(
            jnp.eye(bond_dim, dtype=jnp.complex64)[None, None],
            (batch_size, k, bond_dim, bond_dim),
        )
        log_scale = jnp.zeros((batch_size, k), dtype=jnp.float32)
        for site in range(num_sites):
            output = (
                hidden[:, site] @ final["w"] + final["b"]
            ).astype(jnp.float32)
            matrix = (
                output[:, :matrix_size]
                + 1j * output[:, matrix_size:]
            ).reshape((batch_size, k, bond_dim, bond_dim))
            product = jnp.matmul(product, matrix)
            norm = jnp.maximum(
                jnp.sqrt(
                    jnp.sum(jnp.abs(product) ** 2, axis=(-2, -1))
                ),
                1e-30,
            )
            product = product / norm[..., None, None]
            log_scale = log_scale + jnp.log(norm)
        amplitude = jnp.trace(product, axis1=-2, axis2=-1)
        log_amp = jnp.clip(
            log_scale
            + jnp.log(jnp.maximum(jnp.abs(amplitude), 1e-30)),
            -logphi_real_clip,
            logphi_real_clip,
        )
        return jnp.exp(log_amp + 1j * jnp.angle(amplitude))
    reuse_first_layer = first_layer_backend == "reuse"
    structured_first_layer = bool(
        _os.environ.get("NQS_STRUCTURED_BACKFLOW_FIRST_LAYER")
    )
    if reuse_first_layer:
        if tensor_generation != "backflow" or attention_blocks or residual_blocks:
            raise ValueError("Reused first layer requires plain backflow with jax backend")
        outputs = _apply_mps_backflow_reuse_first_layer(params, states, activation)
    elif structured_first_layer:
        if tensor_generation != "backflow" or attention_blocks or residual_blocks:
            raise ValueError(
                "Structured first layer requires plain Site Backflow MPS."
            )
        outputs = _apply_mps_backflow_structured_first_layer(
            params, states, activation
        )
    elif first_layer_backend == "triton":
        if tensor_generation != "backflow" or attention_blocks or residual_blocks:
            raise ValueError(
                "Triton first layer requires plain Site Backflow MPS."
            )
        outputs = _apply_mps_backflow_triton_first_layer(
            params, states, activation
        )
    else:
        inputs = mps_selected_tensor_inputs(states, tensor_generation)
        input_dim = int(inputs.shape[-1])
    if (
        not reuse_first_layer
        and not structured_first_layer
        and first_layer_backend != "triton"
        and (attention_blocks or residual_blocks)
    ):
        outputs = _apply_sequence_mlp(
            params,
            inputs,
            activation,
            attention_blocks,
            attention_heads,
            residual_blocks,
        )
    elif not reuse_first_layer and not structured_first_layer and first_layer_backend != "triton":
        outputs = _apply_mlp(
            params["tensor_mlp"],
            inputs.reshape((batch_size * num_sites, input_dim)),
            activation,
        )
    outputs = outputs.reshape((batch_size, num_sites, 2 * matrix_size))
    if _os.environ.get("NQS_PALLAS_REALBLOCK_TRIPLET"):
        if k != 1 or bond_dim > 32 or num_sites % 3:
            raise ValueError(
                "NQS_PALLAS_REALBLOCK_TRIPLET requires k=1, D<=32, and a "
                "site count divisible by three."
            )
        if reuse_first_layer and _os.environ.get("NQS_NEIGHBOR_TRIPLET_TAIL"):
            from .triton_kernels import persistent_realblock_triplet_tail_pallas

            return persistent_realblock_triplet_tail_pallas(
                outputs.astype(jnp.bfloat16), logphi_real_clip
            )
        from .triton_kernels import persistent_realblock_triplet_pallas

        return persistent_realblock_triplet_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_QUAD"):
        if k != 1 or bond_dim > 32 or num_sites % 4:
            raise ValueError(
                "NQS_PALLAS_REALBLOCK_QUAD requires k=1, D<=32, and a "
                "site count divisible by four."
            )
        if reuse_first_layer and _os.environ.get("NQS_NEIGHBOR_QUAD_TAIL"):
            from .triton_kernels import persistent_realblock_quad_tail_pallas

            return persistent_realblock_quad_tail_pallas(
                outputs.astype(jnp.bfloat16), logphi_real_clip
            )
        from .triton_kernels import persistent_realblock_quad_pallas

        return persistent_realblock_quad_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_PALLAS_REALBLOCK_PAIR"):
        if k != 1 or bond_dim > 32:
            raise ValueError(
                "NQS_PALLAS_REALBLOCK_PAIR requires k=1, D<=32, "
                "with any positive number of sites."
            )
        from .triton_kernels import persistent_realblock_pair_pallas

        return persistent_realblock_pair_pallas(
            outputs.astype(jnp.bfloat16), logphi_real_clip
        )
    if _os.environ.get("NQS_BF16_MATRICES"):
        real_part = outputs[..., :matrix_size].astype(jnp.bfloat16)
        imag_part = outputs[..., matrix_size:].astype(jnp.bfloat16)
        matrices = (
            real_part.astype(jnp.float32) + 1j * imag_part.astype(jnp.float32)
        ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))
    else:
        matrices = (
            outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
        ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))

    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def _generate_plain_backflow_matrices(
    mps_params, states, activation: str, k: int, bond_dim: int
):
    """Generate plain Site Backflow matrices without contracting them."""
    batch_size, num_sites = states.shape
    matrix_size = k * bond_dim * bond_dim
    inputs = mps_selected_tensor_inputs(states, "backflow")
    outputs = _apply_mlp(
        mps_params["tensor_mlp"],
        inputs.reshape((batch_size * num_sites, inputs.shape[-1])),
        activation,
    ).reshape((batch_size, num_sites, 2 * matrix_size))
    return (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))


@partial(
    jax.jit,
    static_argnames=("num_sites", "activation", "k", "bond_dim", "residual_blocks"),
)
def _build_mps_tensor_bank(
    params,
    num_sites: int,
    activation: str,
    k: int,
    bond_dim: int,
    residual_blocks: int = 0,
):
    """Build the parameter-only local tensor bank once for a forward batch."""
    site_features = jnp.repeat(jnp.eye(num_sites, dtype=jnp.float32), 2, axis=0)
    occupation_features = jnp.tile(jnp.eye(2, dtype=jnp.float32), (num_sites, 1))
    inputs = jnp.concatenate((site_features, occupation_features), axis=-1)
    matrix_size = k * bond_dim * bond_dim
    if residual_blocks:
        outputs = _apply_mlp_residual(
            params["tensor_mlp"], inputs, activation, residual_blocks
        )
    else:
        outputs = _apply_mlp(params["tensor_mlp"], inputs, activation)
    outputs = outputs.reshape((num_sites, 2, 2 * matrix_size))
    return (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((num_sites, 2, k, bond_dim, bond_dim))


@partial(
    jax.jit,
    static_argnames=("logphi_real_clip", "contraction", "lazy"),
)
def _apply_mps_tensor_bank_from_bank(
    tensor_bank,
    states,
    logphi_real_clip: float,
    contraction: str = "sequential",
    lazy: bool = False,
):
    """Gather occupation-selected site matrices and contract the MPS."""
    _, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    if lazy:
        return contract_periodic_mps_from_bank(
            tensor_bank, occupancy, logphi_real_clip, contraction
        )
    site_indices = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    matrices = tensor_bank[site_indices, occupancy]
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def _apply_mps_tensor_bank(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    residual_blocks: int = 0,
    contraction: str = "sequential",
    lazy: bool = False,
):
    """Evaluate all 2*N local tensors once, then gather by occupation."""
    tensor_bank = _build_mps_tensor_bank(
        params,
        int(states.shape[1]),
        activation,
        k,
        bond_dim,
        residual_blocks,
    )
    return _apply_mps_tensor_bank_from_bank(
        tensor_bank, states, logphi_real_clip, contraction, lazy
    )


def _apply_mps_direct(
    params,
    states,
    logphi_real_clip: float,
    contraction: str = "sequential",
):
    """Gather directly variational site matrices for each occupation."""
    _, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    raw = params["tensors"].astype(jnp.float32)
    tensor_bank = raw[..., 0] + 1j * raw[..., 1]
    site_indices = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    matrices = tensor_bank[site_indices, occupancy]
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def _apply_mps_global_backflow(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    residual_blocks: int = 0,
    contraction: str = "sequential",
):
    """Generate every selected site tensor from one full-state MLP call."""
    batch_size, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.float32)
    matrix_size = k * bond_dim * bond_dim
    if residual_blocks:
        outputs = _apply_mlp_residual(
            params["tensor_mlp"], occupancy, activation, residual_blocks
        )
    else:
        outputs = _apply_mlp(params["tensor_mlp"], occupancy, activation)
    outputs = outputs.reshape((batch_size, num_sites, 2 * matrix_size))
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def _apply_mps_site_head_backflow(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    residual_blocks: int = 0,
    contraction: str = "sequential",
):
    """Generate site tensors from one global trunk and small FiLM heads."""
    batch_size, num_sites = states.shape
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    if residual_blocks:
        global_latent = _apply_mlp_residual(
            params["trunk"], states, activation, residual_blocks
        ).astype(jnp.bfloat16)
    else:
        global_latent = _apply_mlp(
            params["trunk"], states, activation
        ).astype(jnp.bfloat16)
    site_indices = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
    gamma = params["gamma"][site_indices, occupancy]
    beta = params["beta"][site_indices, occupancy]
    site_latent = activation_fn(
        gamma * global_latent[:, None, :] + beta,
        activation,
    )
    latent_dim = int(site_latent.shape[-1])
    matrix_size = k * bond_dim * bond_dim
    head_inputs = site_latent.reshape((batch_size * num_sites, latent_dim))
    if residual_blocks:
        outputs = _apply_mlp_residual(
            params["head"], head_inputs, activation, residual_blocks
        )
    else:
        outputs = _apply_mlp(params["head"], head_inputs, activation)
    outputs = outputs.reshape((batch_size, num_sites, 2 * matrix_size))
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((batch_size, num_sites, k, bond_dim, bond_dim))
    return contract_periodic_mps_selected(
        matrices, logphi_real_clip, contraction
    )


def contract_periodic_mps(matrices, logphi_real_clip: float):
    """Return ``Tr[A_0 A_1 ... A_(N-1)]`` with stable rescaling."""
    batch_size, num_sites, channels, bond_dim, _ = matrices.shape
    product = jnp.broadcast_to(
        jnp.eye(bond_dim, dtype=matrices.dtype)[None, None, :, :],
        (batch_size, channels, bond_dim, bond_dim),
    )
    log_scale = jnp.zeros((batch_size, channels), dtype=jnp.float32)

    def contract_site(carry, site_matrix):
        current, current_log_scale = carry
        updated = jnp.einsum("bkij,bkjl->bkil", current, site_matrix)
        norm = jnp.sqrt(jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)))
        safe_norm = jnp.maximum(norm, 1e-30)
        updated = updated / safe_norm[..., None, None]
        return (updated, current_log_scale + jnp.log(safe_norm)), None

    (product, log_scale), _ = jax.lax.scan(
        contract_site,
        (product, log_scale),
        jnp.transpose(matrices, (1, 0, 2, 3, 4)),
        length=num_sites,
    )
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs),
        -logphi_real_clip,
        logphi_real_clip,
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def contract_periodic_mps_balanced(matrices, logphi_real_clip: float):
    """Contract a periodic MPS with a rescaled balanced product tree."""
    batch_size, num_sites, channels, bond_dim, _ = matrices.shape
    if num_sites <= 0:
        raise ValueError("MPS contraction requires at least one site.")

    segments = matrices
    log_scales = jnp.zeros(
        (batch_size, num_sites, channels), dtype=jnp.float32
    )
    while segments.shape[1] > 1:
        segment_count = int(segments.shape[1])
        pair_count = segment_count // 2
        paired = segments[:, : 2 * pair_count].reshape(
            (
                batch_size,
                pair_count,
                2,
                channels,
                bond_dim,
                bond_dim,
            )
        )
        products = jnp.matmul(paired[:, :, 0], paired[:, :, 1])
        norms = jnp.sqrt(jnp.sum(jnp.abs(products) ** 2, axis=(-2, -1)))
        safe_norms = jnp.maximum(norms, 1e-30)
        products = products / safe_norms[..., None, None]

        paired_scales = log_scales[:, : 2 * pair_count].reshape(
            (batch_size, pair_count, 2, channels)
        )
        product_scales = (
            paired_scales[:, :, 0]
            + paired_scales[:, :, 1]
            + jnp.log(safe_norms)
        )
        if segment_count % 2:
            segments = jnp.concatenate(
                (products, segments[:, -1:]), axis=1
            )
            log_scales = jnp.concatenate(
                (product_scales, log_scales[:, -1:]), axis=1
            )
        else:
            segments = products
            log_scales = product_scales

    product = segments[:, 0]
    log_scale = log_scales[:, 0]
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs),
        -logphi_real_clip,
        logphi_real_clip,
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def contract_periodic_mps_balanced_rsqrt(matrices, logphi_real_clip: float):
    """Balanced contraction using an explicit fused reciprocal norm."""
    batch_size, num_sites, channels, bond_dim, _ = matrices.shape
    if num_sites <= 0:
        raise ValueError("MPS contraction requires at least one site.")

    segments = matrices
    log_scales = jnp.zeros(
        (batch_size, num_sites, channels), dtype=jnp.float32
    )
    while segments.shape[1] > 1:
        segment_count = int(segments.shape[1])
        pair_count = segment_count // 2
        paired = segments[:, : 2 * pair_count].reshape(
            (
                batch_size,
                pair_count,
                2,
                channels,
                bond_dim,
                bond_dim,
            )
        )
        products = jnp.matmul(paired[:, :, 0], paired[:, :, 1])
        norm_sq = jnp.sum(
            jnp.real(products) * jnp.real(products)
            + jnp.imag(products) * jnp.imag(products),
            axis=(-2, -1),
        )
        inv_norms = jax.lax.rsqrt(jnp.maximum(norm_sq, 1e-60))
        products = products * inv_norms[..., None, None]

        paired_scales = log_scales[:, : 2 * pair_count].reshape(
            (batch_size, pair_count, 2, channels)
        )
        product_scales = (
            paired_scales[:, :, 0]
            + paired_scales[:, :, 1]
            - jnp.log(inv_norms)
        )
        if segment_count % 2:
            segments = jnp.concatenate(
                (products, segments[:, -1:]), axis=1
            )
            log_scales = jnp.concatenate(
                (product_scales, log_scales[:, -1:]), axis=1
            )
        else:
            segments = products
            log_scales = product_scales

    product = segments[:, 0]
    log_scale = log_scales[:, 0]
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs),
        -logphi_real_clip,
        logphi_real_clip,
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def contract_periodic_mps_fused4(matrices, logphi_real_clip: float):
    """Contract four-site blocks before entering the balanced product tree."""
    batch_size, num_sites, channels, bond_dim, _ = matrices.shape
    if num_sites <= 0:
        raise ValueError("MPS contraction requires at least one site.")

    block_count = num_sites // 4
    block_sites = 4 * block_count
    if block_count:
        blocks = matrices[:, :block_sites].reshape(
            (batch_size, block_count, 4, channels, bond_dim, bond_dim)
        )
        left_pairs = jnp.matmul(blocks[:, :, 0], blocks[:, :, 1])
        right_pairs = jnp.matmul(blocks[:, :, 2], blocks[:, :, 3])
        products = jnp.matmul(left_pairs, right_pairs)
        norms = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(products) ** 2, axis=(-2, -1))),
            1e-30,
        )
        segments = products / norms[..., None, None]
        log_scales = jnp.log(norms)
    else:
        segments = matrices[:, :0]
        log_scales = jnp.zeros(
            (batch_size, 0, channels), dtype=jnp.float32
        )

    if block_sites < num_sites:
        tail = matrices[:, block_sites:]
        segments = jnp.concatenate((segments, tail), axis=1)
        log_scales = jnp.concatenate(
            (
                log_scales,
                jnp.zeros(
                    (batch_size, num_sites - block_sites, channels),
                    dtype=jnp.float32,
                ),
            ),
            axis=1,
        )

    while segments.shape[1] > 1:
        segment_count = int(segments.shape[1])
        pair_count = segment_count // 2
        paired = segments[:, : 2 * pair_count].reshape(
            (
                batch_size,
                pair_count,
                2,
                channels,
                bond_dim,
                bond_dim,
            )
        )
        products = jnp.matmul(paired[:, :, 0], paired[:, :, 1])
        norms = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(products) ** 2, axis=(-2, -1))),
            1e-30,
        )
        products = products / norms[..., None, None]
        paired_scales = log_scales[:, : 2 * pair_count].reshape(
            (batch_size, pair_count, 2, channels)
        )
        product_scales = (
            paired_scales[:, :, 0]
            + paired_scales[:, :, 1]
            + jnp.log(norms)
        )
        if segment_count % 2:
            segments = jnp.concatenate(
                (products, segments[:, -1:]), axis=1
            )
            log_scales = jnp.concatenate(
                (product_scales, log_scales[:, -1:]), axis=1
            )
        else:
            segments = products
            log_scales = product_scales

    product = segments[:, 0]
    log_scale = log_scales[:, 0]
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs),
        -logphi_real_clip,
        logphi_real_clip,
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def _apply_mps_from_bank_sequential(bank, occupancy, logphi_real_clip):
    """Sequential contraction reading site matrices straight from the bank."""
    num_sites = int(bank.shape[0])
    batch_size = int(occupancy.shape[0])
    channels = int(bank.shape[2])
    bond_dim = int(bank.shape[3])
    product = jnp.broadcast_to(
        jnp.eye(bond_dim, dtype=bank.dtype)[None, None, :, :],
        (batch_size, channels, bond_dim, bond_dim),
    )
    log_scale = jnp.zeros((batch_size, channels), dtype=jnp.float32)

    def contract_site(carry, site):
        current, current_log_scale = carry
        occ = jax.lax.dynamic_slice(occupancy, (0, site), (batch_size, 1))[:, 0]
        site_bank = jax.lax.dynamic_index_in_dim(bank, site, axis=0, keepdims=False)
        site_matrix = jnp.take(site_bank, occ, axis=0)
        updated = jnp.einsum("bkij,bkjl->bkil", current, site_matrix)
        norm = jnp.sqrt(jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)))
        safe_norm = jnp.maximum(norm, 1e-30)
        updated = updated / safe_norm[..., None, None]
        return (updated, current_log_scale + jnp.log(safe_norm)), None

    (product, log_scale), _ = jax.lax.scan(
        contract_site, (product, log_scale), jnp.arange(num_sites)
    )
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs), -logphi_real_clip, logphi_real_clip
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def _apply_mps_from_bank_balanced(bank, occupancy, logphi_real_clip):
    """Balanced contraction whose first level reads straight from the bank."""
    num_sites = int(bank.shape[0])
    batch_size = int(occupancy.shape[0])
    channels = int(bank.shape[2])
    bond_dim = int(bank.shape[3])
    n_pairs = num_sites // 2
    even_idx = jnp.arange(n_pairs) * 2
    odd_idx = even_idx + 1
    even_occ = occupancy[:, even_idx]
    odd_occ = occupancy[:, odd_idx]
    a_even = bank[even_idx[None, :], even_occ]
    a_odd = bank[odd_idx[None, :], odd_occ]
    products = jnp.matmul(a_even, a_odd)
    norms = jnp.maximum(
        jnp.sqrt(jnp.sum(jnp.abs(products) ** 2, axis=(-2, -1))), 1e-30
    )
    segments = products / norms[..., None, None]
    log_scales = jnp.log(norms)
    if num_sites % 2:
        last_occ = occupancy[:, -1]
        last = jnp.take(bank[num_sites - 1], last_occ, axis=0)
        last_norm = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(last) ** 2, axis=(-2, -1))), 1e-30
        )
        segments = jnp.concatenate((segments, (last / last_norm[..., None, None])[:, None]), axis=1)
        log_scales = jnp.concatenate((log_scales, jnp.log(last_norm)[:, None]), axis=1)
    while segments.shape[1] > 1:
        seg_count = int(segments.shape[1])
        pair_count = seg_count // 2
        paired = segments[:, : 2 * pair_count].reshape(
            (batch_size, pair_count, 2, channels, bond_dim, bond_dim)
        )
        prod = jnp.matmul(paired[:, :, 0], paired[:, :, 1])
        pn = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(prod) ** 2, axis=(-2, -1))), 1e-30
        )
        prod = prod / pn[..., None, None]
        paired_scales = log_scales[:, : 2 * pair_count].reshape(
            (batch_size, pair_count, 2, channels)
        )
        prod_scales = paired_scales[:, :, 0] + paired_scales[:, :, 1] + jnp.log(pn)
        if seg_count % 2:
            segments = jnp.concatenate((prod, segments[:, -1:]), axis=1)
            log_scales = jnp.concatenate((prod_scales, log_scales[:, -1:]), axis=1)
        else:
            segments = prod
            log_scales = prod_scales
    product = segments[:, 0]
    log_scale = log_scales[:, 0]
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amplitude_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(
        log_scale + jnp.log(amplitude_abs), -logphi_real_clip, logphi_real_clip
    )
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def contract_periodic_mps_from_bank(bank, occupancy, logphi_real_clip, mode):
    if mode == "balanced_fp64":
        if not jax.config.jax_enable_x64:
            raise ValueError("balanced_fp64 requires JAX_ENABLE_X64=true.")
        return _apply_mps_from_bank_balanced(
            bank.astype(jnp.complex128), occupancy, logphi_real_clip
        )
    if mode == "balanced":
        return _apply_mps_from_bank_balanced(bank, occupancy, logphi_real_clip)
    if mode == "sequential":
        return _apply_mps_from_bank_sequential(bank, occupancy, logphi_real_clip)
    raise ValueError(f"Unsupported from-bank contraction: {mode}")


def contract_periodic_mps_sitemajor(matrices, logphi_real_clip: float):
    """Sequential site-major contraction: matrices are (N, batch, ch, D, D)."""
    num_sites, batch_size, channels, bond_dim, _ = matrices.shape
    product = jnp.broadcast_to(
        jnp.eye(bond_dim, dtype=matrices.dtype)[None, None],
        (batch_size, channels, bond_dim, bond_dim),
    )
    log_scale = jnp.zeros((batch_size, channels), dtype=jnp.float32)
    for site_index in range(num_sites):
        site_matrix = matrices[site_index]
        updated = jnp.einsum("bcij,bcjk->bcik", product, site_matrix)
        norm = jnp.sqrt(jnp.sum(jnp.abs(updated) ** 2, axis=(-2, -1)))
        safe = jnp.maximum(norm, 1e-30)
        product = updated / safe[..., None, None]
        log_scale = log_scale + jnp.log(safe)
    amplitude = jnp.trace(product, axis1=-2, axis2=-1)
    amp_abs = jnp.maximum(jnp.abs(amplitude), 1e-30)
    log_amp = jnp.clip(log_scale + jnp.log(amp_abs), -logphi_real_clip, logphi_real_clip)
    return jnp.exp(log_amp + 1j * jnp.angle(amplitude))


def _apply_mps_sitemajor(params, states, activation, k, bond_dim, logphi_real_clip,
                         tensor_generation, contraction):
    """Backflow MPS with (N, batch, ch, D, D) matrices and a sequential chain."""
    states = jnp.asarray(states, dtype=jnp.float32)
    batch_size, num_sites = states.shape
    matrix_size = k * bond_dim * bond_dim
    inputs = mps_selected_tensor_inputs(states, tensor_generation)
    input_dim = int(inputs.shape[-1])
    inputs_site = jnp.transpose(inputs, (1, 0, 2))
    outputs = _apply_mlp(
        params["tensor_mlp"],
        inputs_site.reshape((num_sites * batch_size, input_dim)),
        activation,
    )
    outputs = outputs.reshape((num_sites, batch_size, 2 * matrix_size))
    matrices = (
        outputs[..., :matrix_size] + 1j * outputs[..., matrix_size:]
    ).reshape((num_sites, batch_size, k, bond_dim, bond_dim))
    return contract_periodic_mps_sitemajor(matrices, logphi_real_clip)


def contract_periodic_mps_selected(
    matrices, logphi_real_clip: float, contraction: str
):
    if contraction == "balanced_fp64":
        if not jax.config.jax_enable_x64:
            raise ValueError("balanced_fp64 requires JAX_ENABLE_X64=true.")
        return contract_periodic_mps_balanced(
            matrices.astype(jnp.complex128), logphi_real_clip
        )
    if contraction == "balanced_rsqrt":
        return contract_periodic_mps_balanced_rsqrt(
            matrices, logphi_real_clip
        )
    if contraction == "fused4":
        return contract_periodic_mps_fused4(matrices, logphi_real_clip)
    if contraction == "balanced":
        return contract_periodic_mps_balanced(matrices, logphi_real_clip)
    if contraction == "sequential":
        return contract_periodic_mps(matrices, logphi_real_clip)
    raise ValueError(f"Unsupported MPS contraction: {contraction}")


def contract_open_mps(matrices, logphi_real_clip: float):
    """Compatibility alias for the periodic trace MPS contraction."""
    return contract_periodic_mps(matrices, logphi_real_clip)


@partial(
    jax.jit,
    static_argnames=("activation", "k", "logphi_real_clip"),
)
def _wavefunction_matrix_mlp(
    params,
    states,
    activation: str,
    k: int,
    logphi_real_clip: float,
    residual_blocks: int = 0,
):
    # static dispatch: only branch on plain python ints; tracers fall back to plain MLP
    use_residual = (
        isinstance(residual_blocks, int)
        and not isinstance(residual_blocks, bool)
        and int(residual_blocks) > 0
    )
    if use_residual:
        out = _apply_mlp_residual(
            params["pref"], states, activation, int(residual_blocks)
        )
    else:
        out = _apply_mlp(params["pref"], states, activation)
    log_amp = out[:, :k]
    phase = out[:, k:]
    log_amp = jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
    return jnp.exp(log_amp + 1j * phase)


@partial(
    jax.jit,
    static_argnames=( "activation", "k", "logphi_real_clip"),
)
def _wavefunction_matrix_ace_nes_1d(
    params,
    states,
    activation: str,
    k: int,
    logphi_real_clip: float,
):
    return _apply_ace_nes_1d(
        params["nes"], states, activation, k, logphi_real_clip
    )


@partial(
    jax.jit,
    static_argnames=(
        "activation",
        "k",
        "bond_dim",
        "logphi_real_clip",
        "tensor_generation",
        "attention_blocks",
        "attention_heads",
        "residual_blocks",
        "contraction",
        "first_layer_backend",
        "block_size",
        "tensor_bank_lazy",
        "sitemajor",
    ),
)
def _wavefunction_matrix_mps(
    params,
    states,
    activation: str,
    k: int,
    bond_dim: int,
    logphi_real_clip: float,
    tensor_generation: str,
    attention_blocks: int,
    attention_heads: int,
    residual_blocks: int,
    contraction: str,
    first_layer_backend: str,
    block_size: int = 3,
    tensor_bank_lazy: bool = False,
    sitemajor: bool = False,
):
    if sitemajor:
        return _apply_mps_sitemajor(
            params["mps"], states, activation, k, bond_dim,
            logphi_real_clip, tensor_generation, contraction,
        )
    if tensor_generation == "block_backflow":
        return _apply_mps_block_backflow(
            params["mps"],
            states,
            activation,
            k,
            bond_dim,
            logphi_real_clip,
            block_size,
            contraction,
        )
    if tensor_generation in {"local_backflow", "local_conv_backflow"}:
        return _apply_mps_local_backflow(
            params["mps"],
            states,
            activation,
            k,
            bond_dim,
            logphi_real_clip,
            contraction,
        )
    if tensor_generation == "direct":
        return _apply_mps_direct(
            params["mps"], states, logphi_real_clip, contraction
        )
    if tensor_generation == "tensor_bank":
        return _apply_mps_tensor_bank(
            params["mps"],
            states,
            activation,
            k,
            bond_dim,
            logphi_real_clip,
            residual_blocks,
            contraction,
            tensor_bank_lazy,
        )
    if tensor_generation == "global_backflow":
        return _apply_mps_global_backflow(
            params["mps"],
            states,
            activation,
            k,
            bond_dim,
            logphi_real_clip,
            residual_blocks,
            contraction,
        )
    if tensor_generation == "site_head_backflow":
        return _apply_mps_site_head_backflow(
            params["mps"],
            states,
            activation,
            k,
            bond_dim,
            logphi_real_clip,
            residual_blocks,
            contraction,
        )
    return _apply_mps(
        params["mps"],
        states,
        activation,
        k,
        bond_dim,
        logphi_real_clip,
        tensor_generation,
        attention_blocks,
        attention_heads,
        residual_blocks,
        contraction,
        first_layer_backend,
    )


@partial(
    jax.jit,
    static_argnames=(
        "patch_size",
        "embed_dim",
        "num_heads",
        "num_layers",
        "fourier_modes",
        "k",
        "logphi_real_clip",
    ),
)
def _wavefunction_matrix_vit(
    params,
    states,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    fourier_modes: int,
    k: int,
    logphi_real_clip: float,
):
    out = _apply_vit(
        params["vit"],
        states,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        fourier_modes,
    )
    log_amp = jnp.clip(
        out[:, :k], -logphi_real_clip, logphi_real_clip
    )
    phase = out[:, k:]
    return jnp.exp(log_amp + 1j * phase)


def apply_neural_jastrow(
    params, states, values, activation: str, log_clip: float
):
    output = _apply_mlp(
        params["mps"]["jastrow_mlp"], states, activation
    )
    log_amplitude = jnp.clip(output[:, 0], -log_clip, log_clip)
    phase = output[:, 1]
    return values * jnp.exp(log_amplitude + 1j * phase)[:, None]


def wavefunction_matrix(params, states, cfg: NetworkConfig):
    if cfg.architecture == "vit":
        return _wavefunction_matrix_vit(
            params,
            states,
            cfg.vit_patch_size,
            cfg.vit_embed_dim,
            cfg.vit_num_heads,
            cfg.vit_num_layers,
            cfg.vit_fourier_modes,
            cfg.k,
            cfg.logphi_real_clip,
        )
    if cfg.architecture == "ace_nes_1d":
        return _wavefunction_matrix_ace_nes_1d(
            params,
            states,
            cfg.activation,
            cfg.k,
            cfg.logphi_real_clip,
        )
    if cfg.architecture == "mps":
        values = _wavefunction_matrix_mps(
            params,
            states,
            cfg.activation,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.logphi_real_clip,
            cfg.mps_tensor_generation,
            cfg.mps_attention_blocks,
            cfg.mps_attention_heads,
            cfg.mps_residual_blocks,
            cfg.mps_contraction,
            cfg.mps_first_layer_backend,
            cfg.mps_block_size,
            cfg.mps_tensor_bank_lazy,
            cfg.mps_sitemajor,
        )
        if cfg.mps_jastrow:
            values = apply_neural_jastrow(
                params,
                states,
                values,
                cfg.activation,
                cfg.mps_jastrow_log_clip,
            )
        return values
    return _wavefunction_matrix_mlp(
        params,
        states,
        cfg.activation,
        cfg.k,
        cfg.logphi_real_clip,
        cfg.mps_residual_blocks,
    )


def wavefunction_compile_cache_size() -> int:
    total = 0
    for function in (
        _wavefunction_matrix_mlp,
        _wavefunction_matrix_mps,
        _wavefunction_matrix_ace_nes_1d,
        _wavefunction_matrix_vit,
    ):
        cache_size = getattr(function, "_cache_size", None)
        if cache_size is None:
            return -1
        total += int(cache_size())
    return total


def batched_wavefunction_matrix(
    params, states, cfg: NetworkConfig, batch_size: int | None = None
):
    effective_batch = (
        cfg.forward_batch_size if batch_size is None else batch_size
    )
    n_states = int(states.shape[0])
    n_sites = int(states.shape[1])
    if (
        effective_batch is None
        or effective_batch <= 0
        or n_states <= effective_batch
    ):
        return wavefunction_matrix(params, states, cfg)
    out = jnp.zeros((n_states, cfg.k), dtype=jnp.complex64)
    n_full = n_states // effective_batch

    contraction_batch_text = _os.environ.get("NQS_CONTRACTION_BATCH")
    if (
        contraction_batch_text
        and cfg.architecture == "mps"
        and cfg.mps_tensor_generation == "backflow"
        and not cfg.mps_attention_blocks
        and not cfg.mps_residual_blocks
    ):
        contraction_batch = int(contraction_batch_text)
        if contraction_batch % effective_batch:
            raise ValueError(
                "NQS_CONTRACTION_BATCH must be divisible by forward batch"
            )
        if n_states % contraction_batch:
            raise ValueError(
                "split contraction prototype requires an exact macro batch"
            )
        micro_count = contraction_batch // effective_batch
        macro_count = n_states // contraction_batch

        def macro_body(carry, macro_index):
            macro_start = macro_index * contraction_batch
            matrices = []
            for micro_index in range(micro_count):
                start = macro_start + micro_index * effective_batch
                chunk_states = jax.lax.dynamic_slice(
                    states, (start, 0), (effective_batch, n_sites)
                )
                matrices.append(
                    _generate_plain_backflow_matrices(
                        params["mps"],
                        chunk_states,
                        cfg.activation,
                        cfg.k,
                        cfg.mps_bond_dim,
                    )
                )
            macro_matrices = jnp.concatenate(matrices, axis=0)
            chunk = contract_periodic_mps_selected(
                macro_matrices,
                cfg.logphi_real_clip,
                cfg.mps_contraction,
            )
            return jax.lax.dynamic_update_slice(
                carry, chunk, (macro_start, 0)
            ), None

        out, _ = jax.lax.scan(
            macro_body, out, jnp.arange(macro_count)
        )
        return out

    # tensor_bank stays built once across chunks (depends only on params).
    if (
        cfg.architecture == "mps"
        and cfg.mps_tensor_generation == "tensor_bank"
    ):
        tensor_bank = _build_mps_tensor_bank(
            params["mps"],
            n_sites,
            cfg.activation,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.mps_residual_blocks,
        )
        lazy = bool(cfg.mps_tensor_bank_lazy)

        def bank_chunk_fn(i):
            start = i * effective_batch
            chunk_states = jax.lax.dynamic_slice(
                states, (start, 0), (effective_batch, n_sites)
            )
            return _apply_mps_tensor_bank_from_bank(
                tensor_bank,
                chunk_states,
                cfg.logphi_real_clip,
                cfg.mps_contraction,
                lazy,
            )

        parallel = int(getattr(cfg, "forward_parallel_chunks", 1) or 1)
        if parallel > 1 and n_full >= parallel:
            chunks = jax.lax.map(
                bank_chunk_fn, jnp.arange(n_full), batch_size=parallel
            )
            out = chunks.reshape((n_full * effective_batch, cfg.k))
            remainder = n_states - n_full * effective_batch
            if remainder > 0:
                tail = _apply_mps_tensor_bank_from_bank(
                    tensor_bank,
                    states[-remainder:],
                    cfg.logphi_real_clip,
                    cfg.mps_contraction,
                    lazy,
                )
                out = jnp.concatenate((out, tail), axis=0)
            return out

        def bank_scan_body(carry, i):
            chunk = bank_chunk_fn(i)
            return jax.lax.dynamic_update_slice(
                carry, chunk, (i * effective_batch, 0)
            ), None

        out, _ = jax.lax.scan(
            bank_scan_body, out, jnp.arange(n_full)
        )
        remainder = n_states - n_full * effective_batch
        if remainder > 0:
            tail = _apply_mps_tensor_bank_from_bank(
                tensor_bank,
                states[-remainder:],
                cfg.logphi_real_clip,
                cfg.mps_contraction,
                lazy,
            )
            out = jax.lax.dynamic_update_slice(
                out, tail, (n_full * effective_batch, 0)
            )
        return out

    def scan_body(carry, i):
        start = i * effective_batch
        chunk_states = jax.lax.dynamic_slice(
            states, (start, 0), (effective_batch, n_sites)
        )
        chunk = wavefunction_matrix(params, chunk_states, cfg)
        return jax.lax.dynamic_update_slice(carry, chunk, (start, 0)), None

    parallel = int(getattr(cfg, "forward_parallel_chunks", 1) or 1)
    if parallel > 1 and n_full >= parallel:
        def chunk_fn(i):
            start = i * effective_batch
            chunk_states = jax.lax.dynamic_slice(
                states, (start, 0), (effective_batch, n_sites)
            )
            return wavefunction_matrix(params, chunk_states, cfg)
        chunks = jax.lax.map(
            chunk_fn, jnp.arange(n_full), batch_size=parallel
        )
        out = chunks.reshape((n_full * effective_batch, cfg.k))
        remainder = n_states - n_full * effective_batch
        if remainder > 0:
            tail = wavefunction_matrix(params, states[-remainder:], cfg)
            out = jnp.concatenate((out, tail), axis=0)
        return out
    out, _ = jax.lax.scan(
        scan_body,
        out,
        jnp.arange(n_full),
        unroll=int(_os.environ.get("NQS_FORWARD_SCAN_UNROLL", "1")),
    )
    remainder = n_states - n_full * effective_batch
    if remainder > 0:
        tail = wavefunction_matrix(params, states[-remainder:], cfg)
        out = jax.lax.dynamic_update_slice(
            out, tail, (n_full * effective_batch, 0)
        )
    return out


def batched_wavefunction_matrix_from_positions(
    params,
    states,
    positions,
    cfg: NetworkConfig,
    batch_size: int | None = None,
):
    """Evaluate sector positions without materializing all states as floats."""
    effective_batch = (
        cfg.forward_batch_size if batch_size is None else batch_size
    )
    positions = jnp.asarray(positions, dtype=jnp.int64).reshape((-1,))
    n_positions = int(positions.shape[0])
    if n_positions == 0:
        return jnp.empty((0, cfg.k), dtype=jnp.complex64)
    if (
        effective_batch is None
        or effective_batch <= 0
        or n_positions <= effective_batch
    ):
        selected_states = occupation_states_from_positions(
            states, positions, cfg
        )
        return wavefunction_matrix(params, selected_states, cfg)

    tensor_bank = None
    if (
        cfg.architecture == "mps"
        and cfg.mps_tensor_generation == "tensor_bank"
    ):
        # Resolve packed width through the same metadata guard as each chunk.
        n_sites = int(occupation_states_from_positions(
            states, positions[:0], cfg
        ).shape[-1])
        tensor_bank = _build_mps_tensor_bank(
            params["mps"],
            n_sites,
            cfg.activation,
            cfg.k,
            cfg.mps_bond_dim,
            cfg.mps_residual_blocks,
        )

    out = jnp.zeros((n_positions, cfg.k), dtype=jnp.complex64)
    n_full = n_positions // effective_batch

    def scan_body(carry, i):
        start = i * effective_batch
        chunk_positions = jax.lax.dynamic_slice(
            positions, (start,), (effective_batch,)
        )
        chunk_states = occupation_states_from_positions(
            states, chunk_positions, cfg
        )
        if tensor_bank is None:
            chunk = wavefunction_matrix(params, chunk_states, cfg)
        else:
            chunk = _apply_mps_tensor_bank_from_bank(
                tensor_bank,
                chunk_states,
                cfg.logphi_real_clip,
                cfg.mps_contraction,
            )
        return jax.lax.dynamic_update_slice(carry, chunk, (start, 0)), None

    out, _ = jax.lax.scan(scan_body, out, jnp.arange(n_full))
    remainder = n_positions - n_full * effective_batch
    if remainder > 0:
        tail_positions = positions[-remainder:]
        tail_states = occupation_states_from_positions(
            states, tail_positions, cfg
        )
        if tensor_bank is None:
            tail = wavefunction_matrix(params, tail_states, cfg)
        else:
            tail = _apply_mps_tensor_bank_from_bank(
                tensor_bank,
                tail_states,
                cfg.logphi_real_clip,
                cfg.mps_contraction,
            )
        out = jax.lax.dynamic_update_slice(
            out, tail, (n_full * effective_batch, 0)
        )
    return out


def wavefunction(params, states, cfg: NetworkConfig):
    return wavefunction_matrix(params, states, cfg)[:, 0].reshape((-1,))


def build_metric_state_batches(states, cfg: NetworkConfig):
    if cfg.k == 1:
        return states[:, None, :]
    n_states = int(states.shape[0])
    if n_states < cfg.k:
        raise ValueError(f"sector_size={n_states} is smaller than k={cfg.k}.")
    rows = (
        jnp.arange(n_states)[:, None] + jnp.arange(cfg.k)[None, :]
    ) % n_states
    return states[rows]


def determinant_log_wavefunction(
    params, state_batches, cfg: NetworkConfig
):
    m, k, n_sites = state_batches.shape
    flat_states = state_batches.reshape((m * k, n_sites))
    phi_flat = batched_wavefunction_matrix(params, flat_states, cfg)
    psi_mats = phi_flat.reshape((m, k, cfg.k))
    sign, logabsdet = jnp.linalg.slogdet(psi_mats)
    return logabsdet + 1j * jnp.angle(sign)


def psi_matrices_from_positions(params, states, positions, cfg: NetworkConfig):
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]
    m, k = positions.shape
    flat_states = states[positions.reshape((-1,))]
    phi_flat = batched_wavefunction_matrix(params, flat_states, cfg)
    return phi_flat.reshape((m, k, cfg.k))


def deduplicated_wavefunction_matrix_from_positions(
    params,
    states,
    positions,
    cfg: NetworkConfig,
):
    """Evaluate repeated sector positions with one fixed-shape forward.

    If the requested batch is larger than the sector, evaluating the complete
    sector is the smallest guaranteed static shape. Otherwise ``jnp.unique``
    pads back to the request size, preserving one compilation shape while
    allowing future tighter bounds without changing the gather contract.
    """
    positions = jnp.asarray(positions, dtype=jnp.int64)
    flat_positions = positions.reshape((-1,))
    sector_size = int(states.shape[0])
    request_size = int(flat_positions.shape[0])
    if request_size == 0:
        return jnp.empty((0, cfg.k), dtype=jnp.complex128)

    if sector_size <= request_size:
        evaluated_positions = jnp.arange(sector_size, dtype=jnp.int64)
        inverse = flat_positions
    else:
        evaluated_positions, inverse = jnp.unique(
            flat_positions,
            return_inverse=True,
            size=request_size,
            fill_value=0,
        )
    evaluated_states = occupation_states_from_positions(
        states, evaluated_positions, cfg
    )
    evaluated_phi = wavefunction_matrix(params, evaluated_states, cfg)
    return evaluated_phi[inverse]


def determinant_logpsi_from_positions(
    params, states, positions, cfg: NetworkConfig
):
    psi_mats = psi_matrices_from_positions(params, states, positions, cfg)
    sign, logabsdet = jnp.linalg.slogdet(psi_mats)
    return logabsdet + 1j * jnp.angle(sign)


def psi_matrices_from_positions_direct(
    params, states, positions, cfg: NetworkConfig
):
    positions = jnp.asarray(positions, dtype=jnp.int64)
    if positions.ndim == 1:
        positions = positions[:, None]
    m, k = positions.shape
    flat_positions = positions.reshape((-1,))
    use_deduplicated_positions = (
        cfg.architecture == "mps"
        and cfg.mps_tensor_generation
        in {"sitewise", "tensor_bank", "backflow", "site_head_backflow"}
    )
    use_full_sector = (
        cfg.architecture == "mps"
        and cfg.mps_tensor_generation == "tensor_bank"
        and int(states.shape[0]) < int(flat_positions.shape[0])
    )
    if use_deduplicated_positions:
        phi_flat = deduplicated_wavefunction_matrix_from_positions(
            params, states, flat_positions, cfg
        )
    elif use_full_sector:
        full_phi = wavefunction_matrix(params, states, cfg)
        phi_flat = full_phi[flat_positions]
    else:
        flat_states = occupation_states_from_positions(
            states, flat_positions, cfg
        )
        phi_flat = wavefunction_matrix(params, flat_states, cfg)
    return phi_flat.reshape((m, k, cfg.k))


def determinant_logpsi_from_positions_direct(
    params, states, positions, cfg: NetworkConfig
):
    psi_mats = psi_matrices_from_positions_direct(
        params, states, positions, cfg
    )
    sign, logabsdet = jnp.linalg.slogdet(psi_mats)
    return logabsdet + 1j * jnp.angle(sign)


def determinant_sr_scalars(params, state_batches, cfg: NetworkConfig):
    m, k, n_sites = state_batches.shape
    flat_states = state_batches.reshape((m * k, n_sites))
    phi_flat = batched_wavefunction_matrix(params, flat_states, cfg)
    psi_mats = phi_flat.reshape((m, k, cfg.k))
    psi_inv = jax.lax.stop_gradient(jnp.linalg.inv(psi_mats))
    return jnp.einsum("mij,mji->m", psi_inv, psi_mats)


def projected_energy_from_psi(psi, h_sector):
    hpsi = h_sector @ psi
    norm = jnp.real(jnp.vdot(psi, psi))
    energy = jnp.vdot(psi, hpsi) / jnp.maximum(norm, 1e-300)
    return jnp.real(energy)


@jax.jit
def projected_ritz_jax(phi, h_sector, overlap_shift):
    norm = max(1, phi.shape[0])
    s_proj = phi.conj().T @ phi / norm
    hphi = h_sector @ phi
    h_proj = phi.conj().T @ hphi / norm
    s_proj = 0.5 * (s_proj + s_proj.conj().T)
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    eye = jnp.eye(s_proj.shape[0], dtype=s_proj.dtype)
    s_reg = s_proj + overlap_shift * eye
    chol = jnp.linalg.cholesky(s_reg)
    chol_inv = jsp_linalg.solve_triangular(chol, eye, lower=True)
    h_orth = chol_inv @ h_proj @ chol_inv.conj().T
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    return jnp.linalg.eigvalsh(h_orth), s_proj, h_proj


@jax.jit
def normalized_overlap_condition_jax(s_proj):
    diag = jnp.real(jnp.diag(s_proj))
    inv_sqrt = jax.lax.rsqrt(jnp.maximum(diag, 1e-14)).astype(s_proj.dtype)
    s_norm = inv_sqrt[:, None] * s_proj * inv_sqrt[None, :]
    s_norm = 0.5 * (s_norm + s_norm.conj().T)
    evals = jnp.linalg.eigvalsh(s_norm)
    cond = jnp.max(evals) / jnp.maximum(jnp.min(evals), 1e-12)
    return s_norm, evals, cond


def projected_diagonalization(
    params,
    states,
    h_exact,
    net_cfg: NetworkConfig,
    overlap_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    states_jax = jnp.asarray(states, dtype=jnp.float64)
    h_sector = jnp.asarray(h_exact, dtype=jnp.complex128)
    phi = wavefunction_matrix(params, states_jax, net_cfg)
    evals, s_proj, _ = projected_ritz_jax(
        phi, h_sector, overlap_shift
    )
    _, s_eigs, _ = normalized_overlap_condition_jax(s_proj)
    return np.asarray(evals), np.asarray(s_eigs)


def projected_diagonalization_sparse(
    params,
    states,
    sparse_rows,
    net_cfg: NetworkConfig,
    overlap_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    states_jax = jnp.asarray(states, dtype=jnp.float64)
    phi = batched_wavefunction_matrix(
        params, states_jax, net_cfg
    )
    diag, cols, vals, mask = sparse_rows.as_tuple()
    hphi = (
        diag[:, None] * phi
        + jnp.sum(
            vals[:, :, None]
            * mask[:, :, None]
            * phi[cols],
            axis=1,
            dtype=jnp.complex128,
        )
    )
    norm = max(1, phi.shape[0])
    s_proj = phi.conj().T @ phi / norm
    h_proj = phi.conj().T @ hphi / norm
    s_proj = 0.5 * (s_proj + s_proj.conj().T)
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    eye = jnp.eye(s_proj.shape[0], dtype=s_proj.dtype)
    chol = jnp.linalg.cholesky(s_proj + overlap_shift * eye)
    chol_inv = jsp_linalg.solve_triangular(chol, eye, lower=True)
    h_orth = chol_inv @ h_proj @ chol_inv.conj().T
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    evals = jnp.linalg.eigvalsh(h_orth)
    _, s_eigs, _ = normalized_overlap_condition_jax(s_proj)
    evals, s_eigs = block_until_ready_tree((evals, s_eigs))
    return np.asarray(evals), np.asarray(s_eigs)


def projected_ritz_states_sparse(
    params,
    states,
    sparse_rows,
    net_cfg: NetworkConfig,
    overlap_shift: float,
    projection_rcond: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Ritz energies and normalized occupation-basis state vectors.

    The columns of the returned state matrix are physical vectors in the
    sector occupation basis, rather than coefficients in the nonorthogonal
    neural-output basis.
    """
    states_jax = jnp.asarray(states, dtype=jnp.float64)
    phi_network = batched_wavefunction_matrix(params, states_jax, net_cfg)
    diag, cols, vals, mask = sparse_rows.as_tuple()
    hphi_network = (
        diag[:, None] * phi_network
        + jnp.sum(
            vals[:, :, None]
            * mask[:, :, None]
            * phi_network[cols],
            axis=1,
            dtype=jnp.complex128,
        )
    )
    norm = max(1, phi_network.shape[0])
    s_training = phi_network.conj().T @ phi_network / norm
    h_training = phi_network.conj().T @ hphi_network / norm
    s_training = 0.5 * (s_training + s_training.conj().T)
    h_training = 0.5 * (h_training + h_training.conj().T)
    eye = jnp.eye(s_training.shape[0], dtype=s_training.dtype)
    chol = jnp.linalg.cholesky(s_training + overlap_shift * eye)
    chol_inv = jsp_linalg.solve_triangular(chol, eye, lower=True)
    h_orth = chol_inv @ h_training @ chol_inv.conj().T
    h_orth = 0.5 * (h_orth + h_orth.conj().T)
    regularized_evals = jnp.linalg.eigvalsh(h_orth)
    _, s_eigs, _ = normalized_overlap_condition_jax(s_training)

    # Spectral matrix elements require genuinely orthonormal states.  Remove
    # linearly dependent neural outputs and diagonalize H in that exact span;
    # the overlap shift remains confined to the training/reporting path.
    phi = phi_network.astype(jnp.complex128)
    hphi = (
        diag[:, None] * phi
        + jnp.sum(
            vals[:, :, None]
            * mask[:, :, None]
            * phi[cols],
            axis=1,
            dtype=jnp.complex128,
        )
    )
    s_proj = phi.conj().T @ phi / norm
    s_proj = 0.5 * (s_proj + s_proj.conj().T)
    overlap_values, overlap_vectors = jnp.linalg.eigh(s_proj)
    overlap_values_np = np.asarray(overlap_values)
    threshold = projection_rcond * max(float(overlap_values_np[-1]), 1e-300)
    keep = overlap_values_np > threshold
    if not np.any(keep):
        raise ValueError(
            "No linearly independent NQS output survives projection_rcond."
        )
    kept_vectors = overlap_vectors[:, keep]
    kept_values = overlap_values[keep]
    orthogonalizer = kept_vectors / jnp.sqrt(
        norm * kept_values[None, :]
    )
    orthonormal_basis = phi @ orthogonalizer
    h_orthonormal = orthonormal_basis.conj().T @ (hphi @ orthogonalizer)
    h_orthonormal = 0.5 * (
        h_orthonormal + h_orthonormal.conj().T
    )
    spectral_evals, spectral_coeffs = jnp.linalg.eigh(h_orthonormal)
    wavefunctions = orthonormal_basis @ spectral_coeffs
    regularized_evals, spectral_evals, wavefunctions, s_eigs = (
        block_until_ready_tree(
            (regularized_evals, spectral_evals, wavefunctions, s_eigs)
        )
    )
    return (
        np.asarray(regularized_evals),
        np.asarray(spectral_evals),
        np.asarray(wavefunctions, dtype=np.complex128),
        np.asarray(s_eigs),
    )
