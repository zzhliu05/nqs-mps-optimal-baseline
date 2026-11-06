"""Optional Pallas/Triton kernels for forward-only MPS evaluation.

These kernels are experimental and are not used by the default training path.
Keeping them in a separate module makes rollback equivalent to leaving the
standard JAX backend selected.
"""

from __future__ import annotations

from functools import partial
import math
import os

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as plgpu
except ImportError:  # CPU/Windows wheels do not ship the Triton extension.
    plgpu = None


def pallas_triton_available() -> bool:
    """Return whether this JAX build exposes the Pallas Triton backend."""

    return plgpu is not None and hasattr(plgpu, "CompilerParams")


def _pallas_load(ref, index, *, mask, other):
    """Load through the JAX 0.5 or JAX 0.12 Pallas API."""

    if hasattr(pl, "load"):
        return pl.load(ref, index, mask=mask, other=other)
    return plgpu.load(ref.at[index], mask=mask, other=other)


def _pallas_store(ref, index, value, *, mask):
    """Store through the JAX 0.5 or JAX 0.12 Pallas API."""

    if hasattr(pl, "store"):
        return pl.store(ref, index, value, mask=mask)
    return plgpu.store(ref.at[index], value, mask=mask)


def _layernorm_silu_kernel(
    x_ref,
    scale_ref,
    bias_ref,
    output_ref,
    *,
    width: int,
    block_rows: int,
    block_width: int,
):
    """Fuse FP32 LayerNorm, BF16 affine, and SiLU for MLP rows."""

    row_start = pl.program_id(0) * block_rows
    rows = row_start + jnp.arange(block_rows)
    columns = jnp.arange(block_width)
    row_mask = rows < x_ref.shape[0]
    column_mask = columns < width
    mask = row_mask[:, None] & column_mask[None, :]

    values = _pallas_load(
        x_ref,
        (rows[:, None], columns[None, :]),
        mask=mask,
        other=0.0,
    ).astype(jnp.float32)
    denominator = jnp.asarray(width, dtype=jnp.float32)
    mean = jnp.sum(values, axis=1) / denominator
    centered = jnp.where(mask, values - mean[:, None], 0.0)
    variance = jnp.sum(centered * centered, axis=1) / denominator
    normalized = centered * jax.lax.rsqrt(variance[:, None] + 1e-5)

    scale = _pallas_load(
        scale_ref, (columns,), mask=column_mask, other=0.0
    )
    bias = _pallas_load(
        bias_ref, (columns,), mask=column_mask, other=0.0
    )
    affine = normalized.astype(jnp.bfloat16) * scale[None, :] + bias[None, :]
    # The JAX 0.12 Pallas/Triton lowering cannot divide BF16 tensors, which
    # is required by sigmoid.  Promote only the activation arithmetic, then
    # cast back to the BF16 layer output.
    affine_f32 = affine.astype(jnp.float32)
    activated = (
        affine_f32 * jax.nn.sigmoid(affine_f32)
    ).astype(jnp.bfloat16)
    _pallas_store(
        output_ref,
        (rows[:, None], columns[None, :]),
        activated,
        mask=mask,
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_rows",
        "num_warps",
        "num_stages",
        "interpret",
    ),
)
def _layernorm_silu_pallas_impl(
    x,
    scale,
    bias,
    *,
    block_rows: int = 4,
    num_warps: int = 4,
    num_stages: int = 1,
    interpret: bool = False,
):
    """Pallas implementation for the fixed-width hidden layers."""

    x = jnp.asarray(x, dtype=jnp.bfloat16)
    scale = jnp.asarray(scale, dtype=jnp.bfloat16)
    bias = jnp.asarray(bias, dtype=jnp.bfloat16)
    if x.ndim != 2:
        raise ValueError("fused LayerNorm expects a rank-2 input")
    width = int(x.shape[1])
    if scale.shape != (width,) or bias.shape != (width,):
        raise ValueError("scale and bias must match the hidden width")
    block_width = 1 << (width - 1).bit_length()
    output_shape = jax.ShapeDtypeStruct(x.shape, jnp.bfloat16)
    kernel = partial(
        _layernorm_silu_kernel,
        width=width,
        block_rows=block_rows,
        block_width=block_width,
    )
    if not interpret and not pallas_triton_available():
        raise RuntimeError("The Pallas Triton backend is unavailable.")
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(pl.cdiv(x.shape[0], block_rows),),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_layernorm_silu",
    )(x, scale, bias)


def _bf16_round_f32(value):
    """Explicit nearest-even BF16 rounding, resistant to cast cancellation."""
    bits = lax.bitcast_convert_type(value, jnp.uint32)
    rounded = (bits + jnp.uint32(0x7fff) + ((bits >> 16) & 1)) & jnp.uint32(0xffff0000)
    return jnp.where(jnp.isfinite(value), lax.bitcast_convert_type(rounded, jnp.float32), value)


def _reuse_first_ln_kernel(projection_ref, occupancy_ref, weights_ref, bias_ref,
                           scale_ref, ln_bias_ref, output_ref, *, sites, width,
                           block_rows, block_width):
    rows = pl.program_id(0) * block_rows + jnp.arange(block_rows)
    cols = jnp.arange(block_width)
    state = rows // sites
    site = rows % sites
    valid_rows = rows < output_ref.shape[0]
    valid_cols = cols < width
    mask = valid_rows[:, None] & valid_cols[None, :]
    occ = _pallas_load(occupancy_ref, (state, site), mask=valid_rows, other=0)
    global_value = _pallas_load(projection_ref, (state[:, None], cols[None, :]), mask=mask, other=0)
    site_value = _pallas_load(weights_ref, (site[:, None], cols[None, :]), mask=mask, other=0).astype(jnp.float32)
    local_value = _pallas_load(weights_ref, ((sites + occ)[:, None], cols[None, :]), mask=mask, other=0).astype(jnp.float32)
    bias = _pallas_load(bias_ref, (cols,), mask=valid_cols, other=0).astype(jnp.float32)
    # Preserve both BF16 roundings in dense dot -> bias -> LayerNorm.
    rounded = _bf16_round_f32((global_value + site_value) + local_value)
    values = _bf16_round_f32(rounded + bias[None, :])
    values = jnp.where(mask, values, 0.0)
    mean = jnp.sum(values, axis=1) / jnp.float32(width)
    centered = jnp.where(mask, values - mean[:, None], 0.0)
    variance = jnp.sum(centered * centered, axis=1) / jnp.float32(width)
    normalized = centered * lax.rsqrt(variance[:, None] + 1e-5)
    scale = _pallas_load(scale_ref, (cols,), mask=valid_cols, other=0)
    ln_bias = _pallas_load(ln_bias_ref, (cols,), mask=valid_cols, other=0)
    affine = normalized.astype(jnp.bfloat16) * scale[None, :] + ln_bias[None, :]
    affine_f32 = affine.astype(jnp.float32)
    activated = (affine_f32 * jax.nn.sigmoid(affine_f32)).astype(jnp.bfloat16)
    _pallas_store(output_ref, (rows[:, None], cols[None, :]), activated, mask=mask)


def reuse_first_layernorm_silu(first, states, *, interpret=False, block_rows=4):
    """Neighbor-only fused first layer; never used by sampler or gradient."""
    sites = states.shape[1]
    weights = first["w"]
    width = weights.shape[1]
    if weights.shape[0] != 2 * sites + 2 or weights.dtype != jnp.bfloat16:
        raise ValueError("fused reuse requires plain backflow BF16 weights")
    occupancy = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    projection = lax.dot_general(
        occupancy.astype(weights.dtype),
        weights[sites + 2:],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    kernel = partial(_reuse_first_ln_kernel, sites=sites, width=width,
                     block_rows=block_rows, block_width=1 << (width - 1).bit_length())
    if not interpret and not pallas_triton_available():
        raise RuntimeError("The Pallas Triton backend is unavailable.")
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(
            (occupancy.shape[0] * sites, width), jnp.bfloat16
        ),
        grid=(pl.cdiv(occupancy.shape[0] * sites, block_rows),),
        compiler_params=None if interpret else plgpu.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret, name="mps_reuse_first_ln_silu",
    )(projection, occupancy, weights, first["b"], first["ln_scale"], first["ln_bias"])


def _layernorm_silu_reference(x, scale, bias):
    normalized = jax.nn.standardize(x.astype(jnp.float32), axis=-1)
    affine = normalized.astype(x.dtype) * scale + bias
    return jax.nn.silu(affine)


@jax.custom_vjp
def layernorm_silu_pallas(x, scale, bias):
    """Trainable fused forward with the exact standard-JAX VJP."""

    return _layernorm_silu_pallas_impl(
        x,
        scale,
        bias,
        block_rows=int(os.environ.get("NQS_PALLAS_LN_BLOCK_ROWS", "4")),
        interpret=os.environ.get("NQS_PALLAS_LN_INTERPRET") == "1",
    )


def _layernorm_silu_pallas_fwd(x, scale, bias):
    value = _layernorm_silu_pallas_impl(
        x,
        scale,
        bias,
        block_rows=int(os.environ.get("NQS_PALLAS_LN_BLOCK_ROWS", "4")),
        interpret=os.environ.get("NQS_PALLAS_LN_INTERPRET") == "1",
    )
    return value, (x, scale, bias)


def _layernorm_silu_pallas_bwd(residual, cotangent):
    x, scale, bias = residual
    _, pullback = jax.vjp(_layernorm_silu_reference, x, scale, bias)
    return pullback(cotangent)


layernorm_silu_pallas.defvjp(
    _layernorm_silu_pallas_fwd,
    _layernorm_silu_pallas_bwd,
)


def _gemm_layernorm_silu_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    scale_ref,
    ln_bias_ref,
    output_ref,
    *,
    rows: int,
    in_width: int,
    out_width: int,
    block_rows: int,
    block_k: int,
    block_out: int,
):
    """Fuse a BF16 GEMM, FP32 LayerNorm, affine, and SiLU."""

    row_start = pl.program_id(0) * block_rows
    row = row_start + jnp.arange(block_rows)
    column = jnp.arange(block_out)
    row_mask = row < rows
    column_mask = column < out_width
    accumulator = jnp.zeros((block_rows, block_out), dtype=jnp.float32)

    padded_in = ((in_width + block_k - 1) // block_k) * block_k
    for k_start in range(0, padded_in, block_k):
        reduction = k_start + jnp.arange(block_k)
        x_tile = _pallas_load(
            x_ref,
            (row[:, None], reduction[None, :]),
            mask=row_mask[:, None] & (reduction[None, :] < in_width),
            other=0.0,
        ).astype(jnp.bfloat16)
        weight_tile = _pallas_load(
            weight_ref,
            (reduction[:, None], column[None, :]),
            mask=(reduction[:, None] < in_width) & column_mask[None, :],
            other=0.0,
        ).astype(jnp.bfloat16)
        accumulator = accumulator + jnp.matmul(
            x_tile,
            weight_tile,
            preferred_element_type=jnp.float32,
        )

    bias = _pallas_load(
        bias_ref, (column,), mask=column_mask, other=0.0
    ).astype(jnp.float32)
    # Production emits a BF16 dot result, then performs the separate BF16
    # bias addition. Preserve both roundings before FP32 normalization.
    rounded = _bf16_round_f32(accumulator)
    values = _bf16_round_f32(rounded + bias[None, :])
    denominator = jnp.asarray(out_width, dtype=jnp.float32)
    mean = jnp.sum(jnp.where(column_mask[None, :], values, 0.0), axis=1)
    mean = mean / denominator
    centered = jnp.where(
        column_mask[None, :], values - mean[:, None], 0.0
    )
    variance = jnp.sum(centered * centered, axis=1) / denominator
    normalized = centered * lax.rsqrt(variance[:, None] + 1e-5)
    scale = _pallas_load(
        scale_ref, (column,), mask=column_mask, other=0.0
    ).astype(jnp.bfloat16)
    ln_bias = _pallas_load(
        ln_bias_ref, (column,), mask=column_mask, other=0.0
    ).astype(jnp.bfloat16)
    affine = normalized.astype(jnp.bfloat16) * scale[None, :] + ln_bias[None, :]
    affine_fp32 = affine.astype(jnp.float32)
    activated = affine_fp32 * jax.nn.sigmoid(affine_fp32)
    _pallas_store(
        output_ref,
        (row[:, None], column[None, :]),
        activated.astype(jnp.bfloat16),
        mask=row_mask[:, None] & column_mask[None, :],
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_rows",
        "block_k",
        "num_warps",
        "num_stages",
        "interpret",
    ),
)
def gemm_layernorm_silu_pallas(
    x,
    weight,
    bias,
    scale,
    ln_bias,
    *,
    block_rows: int = 16,
    block_k: int = 32,
    num_warps: int = 8,
    num_stages: int = 2,
    interpret: bool = False,
):
    """Experimental forward-only GEMM+LayerNorm+SiLU fusion."""

    x = jnp.asarray(x, dtype=jnp.bfloat16)
    weight = jnp.asarray(weight, dtype=jnp.bfloat16)
    bias = jnp.asarray(bias, dtype=jnp.bfloat16)
    scale = jnp.asarray(scale, dtype=jnp.bfloat16)
    ln_bias = jnp.asarray(ln_bias, dtype=jnp.bfloat16)
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("fused GEMM+LayerNorm expects rank-2 inputs")
    rows, in_width = map(int, x.shape)
    if int(weight.shape[0]) != in_width:
        raise ValueError("GEMM reduction dimensions do not match")
    out_width = int(weight.shape[1])
    if any(value.shape != (out_width,) for value in (bias, scale, ln_bias)):
        raise ValueError("bias and LayerNorm parameters must match output width")
    block_out = 1 << (out_width - 1).bit_length()
    kernel = partial(
        _gemm_layernorm_silu_kernel,
        rows=rows,
        in_width=in_width,
        out_width=out_width,
        block_rows=block_rows,
        block_k=block_k,
        block_out=block_out,
    )
    if not interpret and not pallas_triton_available():
        raise RuntimeError("The Pallas Triton backend is unavailable.")
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages,
        )
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, out_width), jnp.bfloat16),
        grid=(pl.cdiv(rows, block_rows),),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_gemm_layernorm_silu",
    )(x, weight, bias, scale, ln_bias)


def _realblock_pair_kernel(
    outputs_ref,
    result_ref,
    *,
    bond: int,
    block: int,
):
    """Pack two complex D x D matrices and multiply them in one kernel."""

    batch = pl.program_id(0)
    pair = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond

    def load_block(site, left_index, right_index):
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        value = _pallas_load(
            outputs_ref,
            (batch, site, source),
            mask=(left_index < valid) & (right_index < valid),
            other=0.0,
        ).astype(jnp.bfloat16)
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    left = load_block(2 * pair, row, column)
    right = load_block(2 * pair + 1, row, column)
    product = left @ right
    _pallas_store(
        result_ref,
        (batch, pair, row, column),
        product.astype(jnp.bfloat16),
        mask=(row < valid) & (column < valid),
    )


def _infer_realblock_bond(outputs) -> int:
    """Infer D from the packed [real(DxD), imag(DxD)] output width."""

    if outputs.ndim != 3:
        raise ValueError("real-block kernels expect [batch, sites, 2 * D * D]")
    packed_width = int(outputs.shape[-1])
    if packed_width <= 0 or packed_width % 2:
        raise ValueError("real-block output width must equal 2 * D * D")
    matrix_size = packed_width // 2
    bond = math.isqrt(matrix_size)
    if bond * bond != matrix_size:
        raise ValueError("real-block output width must equal 2 * D * D")
    if bond > 32:
        raise ValueError("Pallas real-block kernels currently support D <= 32")
    return bond


def _realblock_tile_size(bond: int) -> int:
    return 32 if 2 * bond <= 32 else 64


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def _realblock_pair_pallas_impl(
    outputs,
    *,
    num_warps: int = 2,
    interpret: bool = False,
):
    """Site Backflow first-level contraction for statically shaped D <= 32."""

    outputs = jnp.asarray(outputs, dtype=jnp.bfloat16)
    bond = _infer_realblock_bond(outputs)
    block = _realblock_tile_size(bond)
    pairs = int(outputs.shape[1]) // 2
    kernel = partial(_realblock_pair_kernel, bond=bond, block=block)
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    paired = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (outputs.shape[0], pairs, 2 * bond, 2 * bond), jnp.bfloat16
        ),
        grid=(outputs.shape[0], pairs),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_pair",
    )(outputs)
    if outputs.shape[1] % 2:
        matrix_size = bond * bond
        real = outputs[:, -1:, :matrix_size].reshape(
            (outputs.shape[0], 1, bond, bond)
        )
        imag = outputs[:, -1:, matrix_size:].reshape(
            (outputs.shape[0], 1, bond, bond)
        )
        leftover = jnp.concatenate(
            (
                jnp.concatenate((real, -imag), axis=-1),
                jnp.concatenate((imag, real), axis=-1),
            ),
            axis=-2,
        ).astype(jnp.bfloat16)
        paired = jnp.concatenate((paired, leftover), axis=1)
    return paired


def _realblock_pair_reference(outputs):
    batch, sites, _ = outputs.shape
    bond = _infer_realblock_bond(outputs)
    matrix_size = bond * bond
    real = outputs[..., :matrix_size].reshape((batch, sites, bond, bond))
    imag = outputs[..., matrix_size:].reshape((batch, sites, bond, bond))
    blocks = jnp.concatenate(
        (
            jnp.concatenate((real, -imag), axis=-1),
            jnp.concatenate((imag, real), axis=-1),
        ),
        axis=-2,
    ).astype(jnp.bfloat16)
    pairs = sites // 2
    paired = jnp.matmul(
        blocks[:, : 2 * pairs : 2],
        blocks[:, 1 : 2 * pairs : 2],
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    if sites % 2:
        paired = jnp.concatenate((paired, blocks[:, -1:]), axis=1)
    return paired


@jax.custom_vjp
def realblock_pair_pallas(outputs):
    """Pallas forward with the exact standard-JAX pair-contraction VJP."""

    return _realblock_pair_pallas_impl(outputs)


def _realblock_pair_pallas_fwd(outputs):
    value = _realblock_pair_pallas_impl(outputs)
    return value, (outputs,)


def _realblock_pair_pallas_bwd(residual, cotangent):
    (outputs,) = residual
    _, pullback = jax.vjp(_realblock_pair_reference, outputs)
    return pullback(cotangent)


realblock_pair_pallas.defvjp(
    _realblock_pair_pallas_fwd,
    _realblock_pair_pallas_bwd,
)


def persistent_realblock_pair_pallas(outputs, logphi_real_clip: float):
    """Finish the persistent BF16 real-block tree after a fused first level."""

    bond = _infer_realblock_bond(outputs)
    segments = realblock_pair_pallas(outputs)
    scales = jnp.zeros(segments.shape[:2], dtype=jnp.float32)
    level = 1
    while segments.shape[1] > 1:
        level += 1
        count = int(segments.shape[1])
        pairs = count // 2
        products = jnp.matmul(
            segments[:, : 2 * pairs : 2],
            segments[:, 1 : 2 * pairs : 2],
            preferred_element_type=jnp.float32,
        )
        product_scales = (
            scales[:, : 2 * pairs : 2] + scales[:, 1 : 2 * pairs : 2]
        )
        if level % 2 == 0 or count <= 4:
            norm_sq = jnp.sum(
                products[..., :bond, :bond] ** 2
                + products[..., bond:, :bond] ** 2,
                axis=(-2, -1),
            )
            norms = jnp.maximum(jnp.sqrt(norm_sq), 1e-30)
            products = products / norms[..., None, None]
            product_scales = product_scales + jnp.log(norms)
        if pairs + count % 2 > 1:
            products = products.astype(jnp.bfloat16)
        if count % 2:
            products = jnp.concatenate((products, segments[:, -1:]), axis=1)
            product_scales = jnp.concatenate(
                (product_scales, scales[:, -1:]), axis=1
            )
        segments, scales = products, product_scales

    block = segments[:, 0].astype(jnp.float32)
    amplitude = (
        jnp.trace(block[..., :bond, :bond], axis1=-2, axis2=-1)
        + 1j * jnp.trace(block[..., bond:, :bond], axis1=-2, axis2=-1)
    )
    log_amp = scales[:, 0] + jnp.log(
        jnp.maximum(jnp.abs(amplitude), 1e-30)
    )
    value = jnp.exp(
        jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
        + 1j * jnp.angle(amplitude)
    )
    return value[:, None]


def _realblock_quad_kernel(
    outputs_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
):
    """Pack and contract four complex matrices, then normalize once."""

    batch = pl.program_id(0)
    quad = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond

    def load_block(site, left_index, right_index):
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        value = _pallas_load(
            outputs_ref,
            (batch, site, source),
            mask=(left_index < valid) & (right_index < valid),
            other=0.0,
        ).astype(jnp.bfloat16)
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    site = 4 * quad
    pair_left = (load_block(site, row, column) @
                 load_block(site + 1, row, column)).astype(jnp.bfloat16)
    pair_right = (load_block(site + 2, row, column) @
                  load_block(site + 3, row, column)).astype(jnp.bfloat16)
    product_fp32 = jnp.matmul(
        pair_left,
        pair_right,
        preferred_element_type=jnp.float32,
    )
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    norm_sq = jnp.sum(
        jnp.where(complex_half, product_fp32 * product_fp32, 0.0)
    )
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    normalized = product_fp32 / norm
    _pallas_store(
        result_ref,
        (batch, quad, row, column),
        normalized.astype(jnp.bfloat16),
        mask=physical,
    )
    _pallas_store(
        norm_ref,
        (batch, quad),
        norm,
        mask=True,
    )


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def _realblock_quad_pallas_impl(
    outputs,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Four-site contraction without level-one materialization for D <= 32."""

    outputs = jnp.asarray(outputs, dtype=jnp.bfloat16)
    bond = _infer_realblock_bond(outputs)
    block = _realblock_tile_size(bond)
    if outputs.shape[1] % 4:
        raise ValueError("real-block quad kernel requires sites divisible by four")
    quads = int(outputs.shape[1]) // 4
    kernel = partial(_realblock_quad_kernel, bond=bond, block=block)
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (outputs.shape[0], quads, 2 * bond, 2 * bond), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct(
                (outputs.shape[0], quads), jnp.float32
            ),
        ),
        grid=(outputs.shape[0], quads),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_quad",
    )(outputs)


def _cached_realblock_quad_kernel(
    center_outputs_ref,
    parent_ids_ref,
    affected_sites_ref,
    affected_valid_ref,
    replacements_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
    replacement_count: int,
):
    """Gather cached site outputs inside the first contraction kernel."""

    batch = pl.program_id(0)
    quad = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond
    parent = _pallas_load(parent_ids_ref, (batch,), mask=True, other=0)

    def replacement_slot(site):
        slot = jnp.asarray(0, dtype=jnp.int32)
        found = jnp.asarray(False)
        for index in range(replacement_count):
            candidate = _pallas_load(
                affected_sites_ref, (batch, index), mask=True, other=0
            )
            candidate_valid = _pallas_load(
                affected_valid_ref, (batch, index), mask=True, other=0
            )
            match = (candidate_valid != 0) & (candidate == site)
            slot = jnp.where(match, index, slot)
            found = found | match
        return slot, found

    def load_block(site, slot, use_replacement, left_index, right_index):
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        physical = (left_index < valid) & (right_index < valid)
        cached = _pallas_load(
            center_outputs_ref,
            (parent, site, source),
            mask=physical,
            other=0.0,
        )
        replacement = _pallas_load(
            replacements_ref,
            (batch, slot, source),
            mask=physical & use_replacement,
            other=0.0,
        )
        value = jnp.where(use_replacement, replacement, cached).astype(
            jnp.bfloat16
        )
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    site = 4 * quad
    slot0, replace0 = replacement_slot(site)
    slot1, replace1 = replacement_slot(site + 1)
    slot2, replace2 = replacement_slot(site + 2)
    slot3, replace3 = replacement_slot(site + 3)
    pair_left = (
        load_block(site, slot0, replace0, row, column)
        @ load_block(site + 1, slot1, replace1, row, column)
    ).astype(jnp.bfloat16)
    pair_right = (
        load_block(site + 2, slot2, replace2, row, column)
        @ load_block(site + 3, slot3, replace3, row, column)
    ).astype(jnp.bfloat16)
    product_fp32 = jnp.matmul(
        pair_left, pair_right, preferred_element_type=jnp.float32
    )
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    norm_sq = jnp.sum(
        jnp.where(complex_half, product_fp32 * product_fp32, 0.0)
    )
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    _pallas_store(
        result_ref,
        (batch, quad, row, column),
        (product_fp32 / norm).astype(jnp.bfloat16),
        mask=physical,
    )
    _pallas_store(norm_ref, (batch, quad), norm, mask=True)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def cached_realblock_quad_pallas(
    center_outputs,
    parent_ids,
    affected_sites,
    affected_valid,
    replacements,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """First four-site level without materializing per-neighbor outputs."""

    center_outputs = jnp.asarray(center_outputs, dtype=jnp.bfloat16)
    replacements = jnp.asarray(replacements, dtype=jnp.bfloat16)
    parent_ids = jnp.asarray(parent_ids, dtype=jnp.int32)
    affected_sites = jnp.asarray(affected_sites, dtype=jnp.int32)
    affected_valid = jnp.asarray(affected_valid, dtype=jnp.int32)
    bond = _infer_realblock_bond(center_outputs)
    block = _realblock_tile_size(bond)
    if center_outputs.shape[1] % 4:
        raise ValueError("cached quad kernel requires sites divisible by four")
    quads = int(center_outputs.shape[1]) // 4
    replacement_count = int(replacements.shape[1])
    kernel = partial(
        _cached_realblock_quad_kernel,
        bond=bond,
        block=block,
        replacement_count=replacement_count,
    )
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], quads, 2 * bond, 2 * bond),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], quads), jnp.float32
            ),
        ),
        grid=(parent_ids.shape[0], quads),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_cached_realblock_quad",
    )(
        center_outputs,
        parent_ids,
        affected_sites,
        affected_valid,
        replacements,
    )


def _cached_realblock_quad_map_kernel(
    center_outputs_ref,
    parent_ids_ref,
    replacement_slots_ref,
    replacements_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
):
    """Gather cached tensors using one O(1) replacement-slot lookup."""

    batch = pl.program_id(0)
    quad = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond
    parent = _pallas_load(parent_ids_ref, (batch,), mask=True, other=0)

    def load_block(site, left_index, right_index):
        slot = _pallas_load(
            replacement_slots_ref, (batch, site), mask=True, other=-1
        )
        use_replacement = slot >= 0
        safe_slot = jnp.maximum(slot, 0)
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        physical = (left_index < valid) & (right_index < valid)
        cached = _pallas_load(
            center_outputs_ref,
            (parent, site, source),
            mask=physical,
            other=0.0,
        )
        replacement = _pallas_load(
            replacements_ref,
            (batch, safe_slot, source),
            mask=physical & use_replacement,
            other=0.0,
        )
        value = jnp.where(use_replacement, replacement, cached).astype(
            jnp.bfloat16
        )
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    site = 4 * quad
    pair_left = (
        load_block(site, row, column)
        @ load_block(site + 1, row, column)
    ).astype(jnp.bfloat16)
    pair_right = (
        load_block(site + 2, row, column)
        @ load_block(site + 3, row, column)
    ).astype(jnp.bfloat16)
    product_fp32 = jnp.matmul(
        pair_left, pair_right, preferred_element_type=jnp.float32
    )
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    norm_sq = jnp.sum(
        jnp.where(complex_half, product_fp32 * product_fp32, 0.0)
    )
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    _pallas_store(
        result_ref,
        (batch, quad, row, column),
        (product_fp32 / norm).astype(jnp.bfloat16),
        mask=physical,
    )
    _pallas_store(norm_ref, (batch, quad), norm, mask=True)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def cached_realblock_quad_map_pallas(
    center_outputs,
    parent_ids,
    replacement_slots,
    replacements,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """First four-site level with a dense per-site replacement-slot map."""

    center_outputs = jnp.asarray(center_outputs, dtype=jnp.bfloat16)
    replacements = jnp.asarray(replacements, dtype=jnp.bfloat16)
    parent_ids = jnp.asarray(parent_ids, dtype=jnp.int32)
    replacement_slots = jnp.asarray(replacement_slots, dtype=jnp.int32)
    bond = _infer_realblock_bond(center_outputs)
    block = _realblock_tile_size(bond)
    if center_outputs.shape[1] % 4:
        raise ValueError("cached quad-map kernel requires sites divisible by four")
    quads = int(center_outputs.shape[1]) // 4
    kernel = partial(
        _cached_realblock_quad_map_kernel,
        bond=bond,
        block=block,
    )
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], quads, 2 * bond, 2 * bond),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], quads), jnp.float32
            ),
        ),
        grid=(parent_ids.shape[0], quads),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_cached_realblock_quad_map",
    )(
        center_outputs,
        parent_ids,
        replacement_slots,
        replacements,
    )


def _realblock_triplet_kernel(
    outputs_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
):
    """Pack and contract three complex matrices, then normalize once."""

    batch = pl.program_id(0)
    triplet = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond

    def load_block(site, left_index, right_index):
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        value = _pallas_load(
            outputs_ref,
            (batch, site, source),
            mask=(left_index < valid) & (right_index < valid),
            other=0.0,
        ).astype(jnp.bfloat16)
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    site = 3 * triplet
    pair = (
        load_block(site, row, column)
        @ load_block(site + 1, row, column)
    ).astype(jnp.bfloat16)
    product_fp32 = jnp.matmul(
        pair,
        load_block(site + 2, row, column),
        preferred_element_type=jnp.float32,
    )
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    norm_sq = jnp.sum(
        jnp.where(complex_half, product_fp32 * product_fp32, 0.0)
    )
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    _pallas_store(
        result_ref,
        (batch, triplet, row, column),
        (product_fp32 / norm).astype(jnp.bfloat16),
        mask=physical,
    )
    _pallas_store(norm_ref, (batch, triplet), norm, mask=True)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def _realblock_triplet_pallas_impl(
    outputs,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Three-site contraction without level-one materialization for D <= 32."""

    outputs = jnp.asarray(outputs, dtype=jnp.bfloat16)
    bond = _infer_realblock_bond(outputs)
    block = _realblock_tile_size(bond)
    if outputs.shape[1] % 3:
        raise ValueError("real-block triplet kernel requires sites divisible by three")
    triplets = int(outputs.shape[1]) // 3
    kernel = partial(_realblock_triplet_kernel, bond=bond, block=block)
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (outputs.shape[0], triplets, 2 * bond, 2 * bond),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (outputs.shape[0], triplets), jnp.float32
            ),
        ),
        grid=(outputs.shape[0], triplets),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_triplet",
    )(outputs)


def _cached_realblock_triplet_map_kernel(
    center_outputs_ref,
    parent_ids_ref,
    replacement_slots_ref,
    replacements_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
):
    """Gather cached tensors and contract one three-site segment."""

    batch = pl.program_id(0)
    triplet = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond
    parent = _pallas_load(parent_ids_ref, (batch,), mask=True, other=0)

    def load_block(site, left_index, right_index):
        slot = _pallas_load(
            replacement_slots_ref, (batch, site), mask=True, other=-1
        )
        use_replacement = slot >= 0
        safe_slot = jnp.maximum(slot, 0)
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        physical = (left_index < valid) & (right_index < valid)
        cached = _pallas_load(
            center_outputs_ref,
            (parent, site, source),
            mask=physical,
            other=0.0,
        )
        replacement = _pallas_load(
            replacements_ref,
            (batch, safe_slot, source),
            mask=physical & use_replacement,
            other=0.0,
        )
        value = jnp.where(use_replacement, replacement, cached).astype(
            jnp.bfloat16
        )
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    site = 3 * triplet
    pair = (
        load_block(site, row, column)
        @ load_block(site + 1, row, column)
    ).astype(jnp.bfloat16)
    product_fp32 = jnp.matmul(
        pair,
        load_block(site + 2, row, column),
        preferred_element_type=jnp.float32,
    )
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    norm_sq = jnp.sum(
        jnp.where(complex_half, product_fp32 * product_fp32, 0.0)
    )
    norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    _pallas_store(
        result_ref,
        (batch, triplet, row, column),
        (product_fp32 / norm).astype(jnp.bfloat16),
        mask=physical,
    )
    _pallas_store(norm_ref, (batch, triplet), norm, mask=True)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def cached_realblock_triplet_map_pallas(
    center_outputs,
    parent_ids,
    replacement_slots,
    replacements,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Fused cache gather and three-site contraction for Nphi % 3 == 0."""

    center_outputs = jnp.asarray(center_outputs, dtype=jnp.bfloat16)
    replacements = jnp.asarray(replacements, dtype=jnp.bfloat16)
    parent_ids = jnp.asarray(parent_ids, dtype=jnp.int32)
    replacement_slots = jnp.asarray(replacement_slots, dtype=jnp.int32)
    bond = _infer_realblock_bond(center_outputs)
    block = _realblock_tile_size(bond)
    if center_outputs.shape[1] % 3:
        raise ValueError(
            "cached triplet-map kernel requires sites divisible by three"
        )
    triplets = int(center_outputs.shape[1]) // 3
    kernel = partial(
        _cached_realblock_triplet_map_kernel,
        bond=bond,
        block=block,
    )
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], triplets, 2 * bond, 2 * bond),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(
                (parent_ids.shape[0], triplets), jnp.float32
            ),
        ),
        grid=(parent_ids.shape[0], triplets),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_cached_realblock_triplet_map",
    )(
        center_outputs,
        parent_ids,
        replacement_slots,
        replacements,
    )


def _realblock_triplet_reference(outputs):
    batch, sites, _ = outputs.shape
    bond = _infer_realblock_bond(outputs)
    matrix_size = bond * bond
    real = outputs[..., :matrix_size].reshape((batch, sites, bond, bond))
    imag = outputs[..., matrix_size:].reshape((batch, sites, bond, bond))
    blocks = jnp.concatenate(
        (
            jnp.concatenate((real, -imag), axis=-1),
            jnp.concatenate((imag, real), axis=-1),
        ),
        axis=-2,
    ).astype(jnp.bfloat16)
    pairs = jnp.matmul(
        blocks[:, 0::3],
        blocks[:, 1::3],
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    products = jnp.matmul(
        pairs,
        blocks[:, 2::3],
        preferred_element_type=jnp.float32,
    )
    norm_sq = jnp.sum(
        products[..., :bond, :bond] ** 2
        + products[..., bond:, :bond] ** 2,
        axis=(-2, -1),
    )
    norms = jnp.maximum(jnp.sqrt(norm_sq), 1e-30)
    return (products / norms[..., None, None]).astype(jnp.bfloat16), norms


@jax.custom_vjp
def realblock_triplet_pallas(outputs):
    """Pallas three-site forward with the standard-JAX reference VJP."""

    return _realblock_triplet_pallas_impl(outputs)


def _realblock_triplet_pallas_fwd(outputs):
    value = _realblock_triplet_pallas_impl(outputs)
    return value, (outputs,)


def _realblock_triplet_pallas_bwd(residual, cotangent):
    (outputs,) = residual
    _, pullback = jax.vjp(_realblock_triplet_reference, outputs)
    return pullback(cotangent)


realblock_triplet_pallas.defvjp(
    _realblock_triplet_pallas_fwd,
    _realblock_triplet_pallas_bwd,
)


def _realblock_quad_reference(outputs):
    bond = _infer_realblock_bond(outputs)
    pairs = _realblock_pair_reference(outputs)
    products = jnp.matmul(
        pairs[:, 0::2], pairs[:, 1::2], preferred_element_type=jnp.float32
    )
    norm_sq = jnp.sum(
        products[..., :bond, :bond] ** 2
        + products[..., bond:, :bond] ** 2,
        axis=(-2, -1),
    )
    norms = jnp.maximum(jnp.sqrt(norm_sq), 1e-30)
    return (products / norms[..., None, None]).astype(jnp.bfloat16), norms


@jax.custom_vjp
def realblock_quad_pallas(outputs):
    """Pallas four-site forward with the exact standard-JAX VJP."""

    return _realblock_quad_pallas_impl(outputs)


def _realblock_quad_pallas_fwd(outputs):
    value = _realblock_quad_pallas_impl(outputs)
    return value, (outputs,)


def _realblock_quad_pallas_bwd(residual, cotangent):
    (outputs,) = residual
    _, pullback = jax.vjp(_realblock_quad_reference, outputs)
    return pullback(cotangent)


realblock_quad_pallas.defvjp(
    _realblock_quad_pallas_fwd,
    _realblock_quad_pallas_bwd,
)


def _finish_realblock_tree(
    segments,
    norms,
    bond: int,
    logphi_real_clip: float,
    initial_level: int = 2,
):
    """Finish a normalized real-block contraction tree."""

    scales = jnp.log(norms)
    level = initial_level
    while segments.shape[1] > 1:
        level += 1
        count = int(segments.shape[1])
        pairs = count // 2
        products = jnp.matmul(
            segments[:, : 2 * pairs : 2],
            segments[:, 1 : 2 * pairs : 2],
            preferred_element_type=jnp.float32,
        )
        product_scales = (
            scales[:, : 2 * pairs : 2] + scales[:, 1 : 2 * pairs : 2]
        )
        if level % 2 == 0 or count <= 4:
            norm_sq = jnp.sum(
                products[..., :bond, :bond] ** 2
                + products[..., bond:, :bond] ** 2,
                axis=(-2, -1),
            )
            step_norms = jnp.maximum(jnp.sqrt(norm_sq), 1e-30)
            products = products / step_norms[..., None, None]
            product_scales = product_scales + jnp.log(step_norms)
        if pairs + count % 2 > 1:
            products = products.astype(jnp.bfloat16)
        if count % 2:
            products = jnp.concatenate((products, segments[:, -1:]), axis=1)
            product_scales = jnp.concatenate(
                (product_scales, scales[:, -1:]), axis=1
            )
        segments, scales = products, product_scales

    block = segments[:, 0].astype(jnp.float32)
    amplitude = (
        jnp.trace(block[..., :bond, :bond], axis1=-2, axis2=-1)
        + 1j * jnp.trace(block[..., bond:, :bond], axis1=-2, axis2=-1)
    )
    log_amp = scales[:, 0] + jnp.log(
        jnp.maximum(jnp.abs(amplitude), 1e-30)
    )
    value = jnp.exp(
        jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
        + 1j * jnp.angle(amplitude)
    )
    return value[:, None]


def _realblock_quad_tail_kernel(
    segments_ref,
    norms_ref,
    trace_real_ref,
    trace_imag_ref,
    scale_ref,
    *,
    bond: int,
    block: int,
):
    """Fuse the fixed 9->5->3->2->1 tail used by Nphi=36 quad MPS."""

    batch = pl.program_id(0)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)

    def load(index):
        index = jnp.asarray(index, dtype=jnp.int32)
        return _pallas_load(
            segments_ref,
            (batch, index, row, column),
            mask=physical,
            other=0.0,
        ).astype(jnp.bfloat16)

    def product(left, right):
        return jnp.matmul(
            left, right, preferred_element_type=jnp.float32
        )

    def normalize(value, cast):
        norm_sq = jnp.sum(
            jnp.where(complex_half, value * value, 0.0)
        )
        norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
        normalized = value / norm
        if cast:
            normalized = normalized.astype(jnp.bfloat16)
        return normalized, norm

    # Match _finish_realblock_tree exactly: the first 9->5 level is not
    # normalized, then every remaining level is normalized.
    pair0 = product(load(0), load(1)).astype(jnp.bfloat16)
    pair1 = product(load(2), load(3)).astype(jnp.bfloat16)
    left, norm_left = normalize(product(pair0, pair1), True)

    pair2 = product(load(4), load(5)).astype(jnp.bfloat16)
    pair3 = product(load(6), load(7)).astype(jnp.bfloat16)
    right, norm_right = normalize(product(pair2, pair3), True)

    merged, norm_merged = normalize(product(left, right), True)
    final, norm_final = normalize(product(merged, load(8)), False)
    initial_scale = jnp.asarray(0.0, dtype=jnp.float32)
    for index in range(9):
        index = jnp.asarray(index, dtype=jnp.int32)
        initial_scale = initial_scale + jnp.log(
            _pallas_load(norms_ref, (batch, index), mask=True, other=1.0)
        )
    scale = (
        initial_scale
        + jnp.log(norm_left)
        + jnp.log(norm_right)
        + jnp.log(norm_merged)
        + jnp.log(norm_final)
    )
    trace_real = jnp.sum(
        jnp.where((row == column) & (row < bond), final, 0.0)
    )
    trace_imag = jnp.sum(
        jnp.where(
            (row == column + bond) & (column < bond), final, 0.0
        )
    )
    _pallas_store(trace_real_ref, (batch,), trace_real, mask=True)
    _pallas_store(trace_imag_ref, (batch,), trace_imag, mask=True)
    _pallas_store(scale_ref, (batch,), scale, mask=True)


def _realblock_triplet_tail_prepare_kernel(
    segments_ref,
    norms_ref,
    result_ref,
    result_norm_ref,
    *,
    bond: int,
    block: int,
):
    """Fuse the normalized 17->9 level preceding the shared nine-node tail."""

    batch = pl.program_id(0)
    output = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    physical = (row < valid) & (column < valid)
    complex_half = physical & (column < bond)
    paired = output < 8
    left_index = 2 * output
    right_index = left_index + 1

    left = _pallas_load(
        segments_ref,
        (batch, left_index, row, column),
        mask=physical,
        other=0.0,
    ).astype(jnp.bfloat16)
    right = _pallas_load(
        segments_ref,
        (batch, right_index, row, column),
        mask=physical & paired,
        other=0.0,
    ).astype(jnp.bfloat16)
    product = jnp.matmul(left, right, preferred_element_type=jnp.float32)
    norm_sq = jnp.sum(jnp.where(complex_half, product * product, 0.0))
    step_norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
    normalized = (product / step_norm).astype(jnp.bfloat16)
    value = jnp.where(paired, normalized, left)

    left_norm = _pallas_load(
        norms_ref, (batch, left_index), mask=True, other=1.0
    )
    right_norm = _pallas_load(
        norms_ref, (batch, right_index), mask=paired, other=1.0
    )
    combined_norm = jnp.where(
        paired, left_norm * right_norm * step_norm, left_norm
    )
    _pallas_store(
        result_ref,
        (batch, output, row, column),
        value,
        mask=physical,
    )
    _pallas_store(
        result_norm_ref, (batch, output), combined_norm, mask=True
    )


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def realblock_triplet_tail_prepare_pallas(
    segments,
    norms,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Reduce the 17 Nphi=51 triplet segments to the shared nine-node tail."""

    segments = jnp.asarray(segments, dtype=jnp.bfloat16)
    norms = jnp.asarray(norms, dtype=jnp.float32)
    bond = int(segments.shape[-1]) // 2
    block = _realblock_tile_size(bond)
    if int(segments.shape[1]) != 17:
        raise ValueError("triplet tail preparation requires exactly 17 segments")
    kernel = partial(
        _realblock_triplet_tail_prepare_kernel,
        bond=bond,
        block=block,
    )
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (segments.shape[0], 9, 2 * bond, 2 * bond), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct((segments.shape[0], 9), jnp.float32),
        ),
        grid=(segments.shape[0], 9),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_triplet_tail_prepare",
    )(segments, norms)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def realblock_quad_tail_pallas(
    segments,
    norms,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Contract the nine Nphi=36 quad segments in one forward-only kernel."""

    segments = jnp.asarray(segments, dtype=jnp.bfloat16)
    norms = jnp.asarray(norms, dtype=jnp.float32)
    bond = int(segments.shape[-1]) // 2
    block = _realblock_tile_size(bond)
    if int(segments.shape[1]) != 9:
        raise ValueError("quad tail kernel requires exactly nine segments")
    kernel = partial(
        _realblock_quad_tail_kernel,
        bond=bond,
        block=block,
    )
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((segments.shape[0],), jnp.float32),
            jax.ShapeDtypeStruct((segments.shape[0],), jnp.float32),
            jax.ShapeDtypeStruct((segments.shape[0],), jnp.float32),
        ),
        grid=(segments.shape[0],),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_quad_tail",
    )(segments, norms)


def persistent_realblock_quad_tail_pallas(
    outputs,
    logphi_real_clip: float,
    *,
    tail_num_warps: int = 4,
):
    """Four-site kernel plus a fused Nphi=36 contraction tail."""

    segments, norms = realblock_quad_pallas(outputs)
    trace_real, trace_imag, scale = realblock_quad_tail_pallas(
        segments, norms, num_warps=tail_num_warps
    )
    amplitude = trace_real + 1j * trace_imag
    log_amp = scale + jnp.log(jnp.maximum(jnp.abs(amplitude), 1e-30))
    value = jnp.exp(
        jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
        + 1j * jnp.angle(amplitude)
    )
    return value[:, None]


def persistent_realblock_triplet_tail_pallas(
    outputs,
    logphi_real_clip: float,
    *,
    tail_num_warps: int = 4,
):
    """Three-site kernel plus the fused Nphi=51 contraction tail."""

    segments, norms = realblock_triplet_pallas(outputs)
    segments, norms = realblock_triplet_tail_prepare_pallas(
        segments, norms, num_warps=tail_num_warps
    )
    trace_real, trace_imag, scale = realblock_quad_tail_pallas(
        segments, norms, num_warps=tail_num_warps
    )
    amplitude = trace_real + 1j * trace_imag
    log_amp = scale + jnp.log(jnp.maximum(jnp.abs(amplitude), 1e-30))
    value = jnp.exp(
        jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
        + 1j * jnp.angle(amplitude)
    )
    return value[:, None]


def persistent_realblock_quad_pallas(outputs, logphi_real_clip: float):
    """Finish the persistent BF16 tree after a fused four-site level."""

    bond = _infer_realblock_bond(outputs)
    segments, norms = realblock_quad_pallas(outputs)
    return _finish_realblock_tree(segments, norms, bond, logphi_real_clip)


def persistent_realblock_triplet_pallas(outputs, logphi_real_clip: float):
    """Finish the persistent BF16 tree after a fused three-site level."""

    bond = _infer_realblock_bond(outputs)
    segments, norms = realblock_triplet_pallas(outputs)
    return _finish_realblock_tree(
        segments,
        norms,
        bond,
        logphi_real_clip,
        initial_level=1,
    )


def persistent_cached_realblock_quad_pallas(
    center_outputs,
    parent_ids,
    affected_sites,
    affected_valid,
    replacements,
    logphi_real_clip: float,
):
    """Contract cached local-backflow outputs without a dense neighbor copy."""

    bond = _infer_realblock_bond(center_outputs)
    segments, norms = cached_realblock_quad_pallas(
        center_outputs,
        parent_ids,
        affected_sites,
        affected_valid,
        replacements,
    )
    return _finish_realblock_tree(segments, norms, bond, logphi_real_clip)


def persistent_cached_realblock_quad_map_pallas(
    center_outputs,
    parent_ids,
    replacement_slots,
    replacements,
    logphi_real_clip: float,
):
    """Contract cached outputs with O(1) replacement lookup per site."""

    bond = _infer_realblock_bond(center_outputs)
    segments, norms = cached_realblock_quad_map_pallas(
        center_outputs,
        parent_ids,
        replacement_slots,
        replacements,
    )
    return _finish_realblock_tree(segments, norms, bond, logphi_real_clip)


def persistent_cached_realblock_triplet_map_pallas(
    center_outputs,
    parent_ids,
    replacement_slots,
    replacements,
    logphi_real_clip: float,
):
    """Contract cached outputs through the fused three-site map kernel."""

    bond = _infer_realblock_bond(center_outputs)
    segments, norms = cached_realblock_triplet_map_pallas(
        center_outputs,
        parent_ids,
        replacement_slots,
        replacements,
    )
    return _finish_realblock_tree(
        segments,
        norms,
        bond,
        logphi_real_clip,
        initial_level=1,
    )


def _realblock_oct_kernel(
    outputs_ref,
    result_ref,
    norm_ref,
    *,
    bond: int,
    block: int,
):
    """Contract eight sites through three tree levels in one program."""

    batch = pl.program_id(0)
    octet = pl.program_id(1)
    row = jnp.arange(block)[:, None]
    column = jnp.arange(block)[None, :]
    valid = 2 * bond
    matrix_size = bond * bond

    def load_block(site, left_index, right_index):
        base_row = left_index % bond
        base_column = right_index % bond
        imaginary = (left_index < bond) != (right_index < bond)
        source = (
            imaginary.astype(jnp.int32) * matrix_size
            + base_row * bond
            + base_column
        )
        value = _pallas_load(
            outputs_ref,
            (batch, site, source),
            mask=(left_index < valid) & (right_index < valid),
            other=0.0,
        ).astype(jnp.bfloat16)
        negative = (left_index < bond) & (right_index >= bond)
        return jnp.where(negative, -value, value)

    def normalized_quad(site):
        pair_left = (load_block(site, row, column) @
                     load_block(site + 1, row, column)).astype(jnp.bfloat16)
        pair_right = (load_block(site + 2, row, column) @
                      load_block(site + 3, row, column)).astype(jnp.bfloat16)
        product = jnp.matmul(
            pair_left,
            pair_right,
            preferred_element_type=jnp.float32,
        )
        physical = (row < valid) & (column < valid)
        complex_half = physical & (column < bond)
        norm_sq = jnp.sum(
            jnp.where(complex_half, product * product, 0.0)
        )
        norm = jnp.sqrt(jnp.maximum(norm_sq, 1e-30))
        return (product / norm).astype(jnp.bfloat16), norm

    site = 8 * octet
    left, left_norm = normalized_quad(site)
    right, right_norm = normalized_quad(site + 4)
    product = jnp.matmul(left, right, preferred_element_type=jnp.float32)
    physical = (row < valid) & (column < valid)
    _pallas_store(
        result_ref,
        (batch, octet, row, column),
        product.astype(jnp.bfloat16),
        mask=physical,
    )
    # Pallas/Triton cannot lower jnp.stack when both operands are scalar
    # registers. Store the two normalization factors independently instead.
    _pallas_store(norm_ref, (batch, octet, 0), left_norm, mask=True)
    _pallas_store(norm_ref, (batch, octet, 1), right_norm, mask=True)


@partial(jax.jit, static_argnames=("num_warps", "interpret"))
def _realblock_oct_pallas_impl(
    outputs,
    *,
    num_warps: int = 4,
    interpret: bool = False,
):
    """Fixed D=10 eight-site contraction for site counts divisible by eight."""

    outputs = jnp.asarray(outputs, dtype=jnp.bfloat16)
    if outputs.ndim != 3 or outputs.shape[-1] != 200:
        raise ValueError("real-block oct kernel expects [batch, sites, 200]")
    if outputs.shape[1] % 8:
        raise ValueError("real-block oct kernel requires sites divisible by eight")
    octets = int(outputs.shape[1]) // 8
    kernel = partial(_realblock_oct_kernel, bond=10, block=32)
    compiler_params = (
        None
        if interpret
        else plgpu.CompilerParams(num_warps=num_warps, num_stages=1)
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (outputs.shape[0], octets, 20, 20), jnp.bfloat16
            ),
            jax.ShapeDtypeStruct(
                (outputs.shape[0], octets, 2), jnp.float32
            ),
        ),
        grid=(outputs.shape[0], octets),
        compiler_params=compiler_params,
        interpret=interpret,
        name="mps_realblock_oct",
    )(outputs)


def _realblock_oct_reference(outputs):
    quads, quad_norms = _realblock_quad_reference(outputs)
    products = jnp.matmul(
        quads[:, 0::2], quads[:, 1::2], preferred_element_type=jnp.float32
    )
    products = products.astype(jnp.bfloat16)
    paired_norms = quad_norms.reshape(
        (quad_norms.shape[0], quad_norms.shape[1] // 2, 2)
    )
    return products, paired_norms


def realblock_oct_pallas(outputs):
    """Experimental forward-only eight-site Pallas contraction."""

    return _realblock_oct_pallas_impl(outputs)


def persistent_realblock_oct_pallas(outputs, logphi_real_clip: float):
    """Finish the BF16 tree after octets plus a possible four-site tail."""

    site_count = int(outputs.shape[1])
    octet_sites = (site_count // 8) * 8
    segments, pair_norms = realblock_oct_pallas(outputs[:, :octet_sites])
    scales = jnp.sum(jnp.log(pair_norms), axis=-1)
    if octet_sites < site_count:
        tail_segments, tail_norms = realblock_quad_pallas(outputs[:, octet_sites:])
        segments = jnp.concatenate((segments, tail_segments), axis=1)
        scales = jnp.concatenate((scales, jnp.log(tail_norms)), axis=1)

    level = 3
    while segments.shape[1] > 1:
        level += 1
        count = int(segments.shape[1])
        pairs = count // 2
        products = jnp.matmul(
            segments[:, : 2 * pairs : 2],
            segments[:, 1 : 2 * pairs : 2],
            preferred_element_type=jnp.float32,
        )
        product_scales = (
            scales[:, : 2 * pairs : 2] + scales[:, 1 : 2 * pairs : 2]
        )
        if level % 2 == 0 or count <= 4:
            norm_sq = jnp.sum(
                products[..., :10, :10] ** 2
                + products[..., 10:, :10] ** 2,
                axis=(-2, -1),
            )
            step_norms = jnp.maximum(jnp.sqrt(norm_sq), 1e-30)
            products = products / step_norms[..., None, None]
            product_scales = product_scales + jnp.log(step_norms)
        if pairs + count % 2 > 1:
            products = products.astype(jnp.bfloat16)
        if count % 2:
            products = jnp.concatenate((products, segments[:, -1:]), axis=1)
            product_scales = jnp.concatenate(
                (product_scales, scales[:, -1:]), axis=1
            )
        segments, scales = products, product_scales

    block = segments[:, 0].astype(jnp.float32)
    amplitude = (
        jnp.trace(block[..., :10, :10], axis1=-2, axis2=-1)
        + 1j * jnp.trace(block[..., 10:, :10], axis1=-2, axis2=-1)
    )
    log_amp = scales[:, 0] + jnp.log(
        jnp.maximum(jnp.abs(amplitude), 1e-30)
    )
    value = jnp.exp(
        jnp.clip(log_amp, -logphi_real_clip, logphi_real_clip)
        + 1j * jnp.angle(amplitude)
    )
    return value[:, None]


def _structured_backflow_first_layer_kernel(
    states_ref,
    weights_ref,
    bias_ref,
    output_ref,
    *,
    num_sites: int,
    block_batch: int,
    block_hidden: int,
    block_k: int,
    apply_silu: bool,
):
    """Compute SiLU(W [site, local occupation, full state] + b).

    The full-state projection is evaluated once per physical state and reused
    for every site token.  The site and local-occupation one-hot projections
    are loaded directly from the first-layer weight matrix, so the expanded
    ``(batch, num_sites, 2 * num_sites + 2)`` input is never materialized.
    """

    batch_start = pl.program_id(0) * block_batch
    hidden_start = pl.program_id(1) * block_hidden
    batch_idx = batch_start + jnp.arange(block_batch)
    hidden_idx = hidden_start + jnp.arange(block_hidden)
    batch_mask = batch_idx < states_ref.shape[0]
    hidden_mask = hidden_idx < weights_ref.shape[1]

    accumulator = jnp.zeros(
        (block_batch, block_hidden), dtype=jnp.float32
    )
    for k_start in range(0, num_sites, block_k):
        site_idx = k_start + jnp.arange(block_k)
        site_mask = site_idx < num_sites
        state_tile = plgpu.load(
            states_ref.at[batch_idx[:, None], site_idx[None, :]],
            mask=batch_mask[:, None] & site_mask[None, :],
            other=0.0,
        ).astype(jnp.bfloat16)
        weight_tile = plgpu.load(
            weights_ref.at[
                (num_sites + 2 + site_idx)[:, None],
                hidden_idx[None, :],
            ],
            mask=site_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).astype(jnp.bfloat16)
        accumulator += state_tile @ weight_tile

    bias = plgpu.load(
        bias_ref.at[hidden_idx],
        mask=hidden_mask,
        other=0.0,
    )
    local_zero = plgpu.load(
        weights_ref.at[
            num_sites + jnp.zeros_like(hidden_idx), hidden_idx
        ],
        mask=hidden_mask,
        other=0.0,
    )
    local_one = plgpu.load(
        weights_ref.at[
            num_sites + 1 + jnp.zeros_like(hidden_idx), hidden_idx
        ],
        mask=hidden_mask,
        other=0.0,
    )

    def write_site(site, unused):
        del unused
        site_weight = plgpu.load(
            weights_ref.at[
                site + jnp.zeros_like(hidden_idx), hidden_idx
            ],
            mask=hidden_mask,
            other=0.0,
        )
        occupation = plgpu.load(
            states_ref.at[
                batch_idx, site + jnp.zeros_like(batch_idx)
            ],
            mask=batch_mask,
            other=0.0,
        )
        local_weight = jnp.where(
            occupation[:, None] > 0.5,
            local_one[None, :],
            local_zero[None, :],
        )
        value = (
            accumulator.astype(jnp.bfloat16)
            + bias[None, :]
            + site_weight[None, :]
            + local_weight
        )
        if apply_silu:
            value_f32 = value.astype(jnp.float32)
            value = (
                value_f32 * jax.nn.sigmoid(value_f32)
            ).astype(jnp.bfloat16)
        plgpu.store(
            output_ref.at[
                batch_idx[:, None],
                site + jnp.zeros((block_batch, block_hidden), jnp.int32),
                hidden_idx[None, :],
            ],
            value,
            mask=batch_mask[:, None] & hidden_mask[None, :],
        )
        return None

    lax.fori_loop(0, num_sites, write_site, None)


@partial(
    jax.jit,
    static_argnames=(
        "num_sites",
        "block_batch",
        "block_hidden",
        "block_k",
        "num_warps",
        "num_stages",
        "apply_silu",
        "interpret",
    ),
)
def structured_backflow_first_layer_triton(
    states,
    weights,
    bias,
    *,
    num_sites: int,
    block_batch: int = 16,
    block_hidden: int = 64,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    apply_silu: bool = True,
    interpret: bool = False,
):
    """Fused first layer for Site Backflow MPS neighbor inference."""

    states = jnp.asarray(states, dtype=jnp.bfloat16)
    weights = jnp.asarray(weights, dtype=jnp.bfloat16)
    bias = jnp.asarray(bias, dtype=jnp.bfloat16)
    if states.ndim != 2 or states.shape[1] != num_sites:
        raise ValueError("states must have shape (batch, num_sites)")
    if weights.shape[0] != 2 * num_sites + 2:
        raise ValueError("weights have the wrong Site Backflow input size")
    if bias.shape != (weights.shape[1],):
        raise ValueError("bias shape must match the hidden dimension")

    output_shape = jax.ShapeDtypeStruct(
        (states.shape[0], num_sites, weights.shape[1]), jnp.bfloat16
    )
    kernel = partial(
        _structured_backflow_first_layer_kernel,
        num_sites=num_sites,
        block_batch=block_batch,
        block_hidden=block_hidden,
        block_k=block_k,
        apply_silu=apply_silu,
    )
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(
            pl.cdiv(states.shape[0], block_batch),
            pl.cdiv(weights.shape[1], block_hidden),
        ),
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        interpret=interpret,
        name="structured_backflow_first_layer",
    )(states, weights, bias)


def structured_backflow_preactivation_jax(states, weights, bias):
    """Structured preactivation matching the expanded first-layer GEMM."""

    num_sites = states.shape[1]
    occupation = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    global_projection = (
        states.astype(weights.dtype) @ weights[num_sites + 2 :, :]
    )
    site_projection = weights[:num_sites, :]
    local_projection = weights[num_sites : num_sites + 2, :][occupation]
    return (
        global_projection[:, None, :]
        + site_projection[None, :, :]
        + local_projection
        + bias[None, None, :]
    )


def _structured_backflow_preactivation_triton_impl(states, weights, bias):
    return structured_backflow_first_layer_triton(
        states,
        weights,
        bias,
        num_sites=int(states.shape[1]),
        block_batch=8,
        block_hidden=32,
        block_k=16,
        num_warps=4,
        num_stages=1,
        apply_silu=False,
    )


@jax.custom_vjp
def structured_backflow_preactivation_triton(states, weights, bias):
    """Trainable Triton forward with an equivalent structured JAX VJP."""

    return _structured_backflow_preactivation_triton_impl(
        states, weights, bias
    )


def _structured_backflow_preactivation_triton_fwd(states, weights, bias):
    value = _structured_backflow_preactivation_triton_impl(
        states, weights, bias
    )
    return value, (states, weights, bias)


def _structured_backflow_preactivation_triton_bwd(residual, cotangent):
    states, weights, bias = residual
    _, pullback = jax.vjp(
        structured_backflow_preactivation_jax, states, weights, bias
    )
    return pullback(cotangent)


structured_backflow_preactivation_triton.defvjp(
    _structured_backflow_preactivation_triton_fwd,
    _structured_backflow_preactivation_triton_bwd,
)


@jax.jit
def structured_backflow_first_layer_jax(states, weights, bias):
    """Reference implementation used for correctness and timing checks."""

    return jax.nn.silu(
        structured_backflow_preactivation_jax(states, weights, bias)
    )


def _compact_row_reduce_kernel(
    coeff_real_ref,
    coeff_imag_ref,
    phi_real_ref,
    phi_imag_ref,
    offsets_ref,
    output_real_ref,
    output_imag_ref,
    *,
    block_width: int,
):
    """Reduce one contiguous sparse-Hamiltonian row per Triton program."""

    row = pl.program_id(0)
    lane = jnp.arange(block_width)
    start = _pallas_load(offsets_ref, (row,), mask=True, other=0)
    stop = _pallas_load(offsets_ref, (row + 1,), mask=True, other=0)
    index = start + lane
    valid = index < stop
    coeff_real = _pallas_load(
        coeff_real_ref, (index,), mask=valid, other=0.0
    ).astype(jnp.float32)
    coeff_imag = _pallas_load(
        coeff_imag_ref, (index,), mask=valid, other=0.0
    ).astype(jnp.float32)
    phi_real = _pallas_load(
        phi_real_ref, (index,), mask=valid, other=0.0
    ).astype(jnp.float32)
    phi_imag = _pallas_load(
        phi_imag_ref, (index,), mask=valid, other=0.0
    ).astype(jnp.float32)
    product_real = coeff_real * phi_real - coeff_imag * phi_imag
    product_imag = coeff_real * phi_imag + coeff_imag * phi_real
    _pallas_store(
        output_real_ref,
        (row,),
        jnp.sum(jnp.where(valid, product_real, 0.0)),
        mask=True,
    )
    _pallas_store(
        output_imag_ref,
        (row,),
        jnp.sum(jnp.where(valid, product_imag, 0.0)),
        mask=True,
    )


@partial(jax.jit, static_argnames=("block_width", "interpret"))
def compact_row_reduce_pallas(
    coeff,
    phi,
    offsets,
    *,
    block_width: int = 1024,
    interpret: bool = False,
):
    """Multiply and reduce sorted compact neighbors without scatter/segment."""

    coeff = jnp.asarray(coeff)
    phi = jnp.asarray(phi)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    if coeff.ndim != 1 or phi.ndim != 1 or coeff.shape != phi.shape:
        raise ValueError("compact row reduction expects matching flat vectors")
    if offsets.ndim != 1 or offsets.shape[0] < 2:
        raise ValueError("compact row offsets must contain row_count + 1 values")
    if block_width & (block_width - 1):
        raise ValueError("compact row reduction block width must be a power of two")
    row_count = offsets.shape[0] - 1
    output_shape = (
        jax.ShapeDtypeStruct((row_count,), jnp.float32),
        jax.ShapeDtypeStruct((row_count,), jnp.float32),
    )
    if not interpret and not pallas_triton_available():
        raise RuntimeError("The Pallas Triton backend is unavailable.")
    real, imag = pl.pallas_call(
        partial(_compact_row_reduce_kernel, block_width=block_width),
        out_shape=output_shape,
        grid=(row_count,),
        compiler_params=(
            None
            if interpret
            else plgpu.CompilerParams(num_warps=8, num_stages=1)
        ),
        interpret=interpret,
        name="compact_neighbor_row_reduce",
    )(
        coeff.real.astype(jnp.float32),
        coeff.imag.astype(jnp.float32),
        phi.real.astype(jnp.float32),
        phi.imag.astype(jnp.float32),
        offsets,
    )
    return (real + 1j * imag).astype(jnp.complex128)
