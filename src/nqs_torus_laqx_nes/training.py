"""Projected and variational Monte Carlo training loops."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import RunConfig
from .continuous import (
    ContinuousEvaluationPlan,
    ContinuousMetropolisSamplerJAX,
    continuous_vmc_surrogate_loss,
)
from .diagnostics import probability_metrics
from .diagnostics import plot_probability_distribution
from .diagnostics import plot_live_vmc_energy
from .network import (
    block_until_ready_tree,
    local_conv_site_tensor_bank,
    local_conv_site_tensor_bank_with_cache,
    local_conv_site_wavefunction_from_bank,
    occupation_states_from_positions,
    mps_backflow_embedding_stats,
    wavefunction_compile_cache_size,
)
from .optimization import (
    apply_continuous_optimizer_update,
    apply_optimizer_update,
    init_optimizer_state,
    local_energies_sliced,
    optimizer_log_suffix,
    profile_vmc_autodiff,
    profile_vmc_autodiff_sliced,
    profile_vmc_loss,
    profile_vmc_loss_sliced,
    perturb_params,
    projected_loss,
    projected_loss_sparse,
    vmc_surrogate_loss,
    vmc_surrogate_loss_sliced,
    vmc_surrogate_gradient_from_eloc,
)
from .physics import SparseHamiltonianRows, pack_sparse_bitstring_rows
from .physics import BasisData, TorusCoulombHamiltonian
from .sampling import MetropolisSamplerJAX


@dataclass
class TrainingResult:
    params: object
    history: list[dict]


def _should_profile_vmc_timing(config: RunConfig, step: int) -> bool:
    return (
        config.training.vmc_timing_analysis
        and (
            step == 1
            or step % config.training.vmc_timing_interval == 0
            or step == config.training.n_steps
        )
    )


def _has_converged(
    history: list[dict], config: RunConfig, step: int
) -> bool:
    train_cfg = config.training
    window = int(train_cfg.converge_window)
    if (
        train_cfg.converge_tol <= 0
        or step < max(train_cfg.min_steps, 2 * window)
    ):
        return False
    recent = np.array(
        [item["energy"] for item in history[-window:]], dtype=float
    )
    previous = np.array(
        [item["energy"] for item in history[-2 * window : -window]],
        dtype=float,
    )
    delta = abs(float(np.mean(recent) - np.mean(previous)))
    if delta < train_cfg.converge_tol:
        print(f"converged at step {step}: moving-window delta={delta:.3e}")
        return True
    return False


def _select_vmc_estimator_positions(
    positions_np,
    *,
    pool_chains: int,
    estimator_chains: int | None,
    n_samples: int,
    step: int,
):
    """Select a rotating chain subset while preserving every sample block."""
    positions_np = np.asarray(positions_np)
    if positions_np.dtype != np.uint64:
        positions_np = positions_np.astype(np.int64, copy=False)
    if estimator_chains is None or estimator_chains == pool_chains:
        return positions_np
    expected = int(pool_chains) * int(n_samples)
    if positions_np.shape[0] != expected:
        raise ValueError(
            "sampled position count does not match n_chains * n_samples."
        )
    if not 0 < estimator_chains <= pool_chains:
        raise ValueError(
            "estimator_chains must be in [1, pool_chains]."
        )
    sample_shape = positions_np.shape[1:]
    sampled = positions_np.reshape(
        (int(n_samples), int(pool_chains), *sample_shape)
    )
    start = ((int(step) - 1) * int(estimator_chains)) % int(pool_chains)
    chain_ids = (
        start + np.arange(int(estimator_chains), dtype=np.int64)
    ) % int(pool_chains)
    return sampled[:, chain_ids].reshape(
        (int(n_samples) * int(estimator_chains), *sample_shape)
    )


def _stage_cpu_sector_rows(
    sector_bitstrings_np,
    positions_np,
    sparse_cpu,
):
    """Gather one sampled-row working set from a CPU-resident sector table."""
    positions_np = np.asarray(positions_np, dtype=np.int64)
    cols = np.asarray(sparse_cpu.cols, dtype=np.int64)
    mask = np.asarray(sparse_cpu.mask)
    row_positions = positions_np
    if row_positions.ndim == 1:
        row_positions = row_positions[:, None]
    fallback = np.broadcast_to(
        row_positions[..., None], cols.shape
    )
    safe_cols = np.where(mask > 0, cols, fallback)
    combined = np.concatenate(
        (row_positions.reshape(-1), safe_cols.reshape(-1))
    )
    unique_global, inverse = np.unique(combined, return_inverse=True)
    position_count = row_positions.size
    local_positions = inverse[:position_count].reshape(row_positions.shape)
    local_cols = inverse[position_count:].reshape(cols.shape)
    local_states = jnp.asarray(
        np.asarray(sector_bitstrings_np, dtype=np.uint64)[unique_global],
        dtype=jnp.uint64,
    )
    sparse_local = (
        jnp.asarray(sparse_cpu.diag, dtype=jnp.complex128),
        jnp.asarray(local_cols, dtype=jnp.int64),
        jnp.asarray(sparse_cpu.vals, dtype=jnp.complex128),
        jnp.asarray(sparse_cpu.mask, dtype=jnp.float64),
    )
    return (
        local_states,
        jnp.asarray(local_positions, dtype=jnp.int64),
        sparse_local,
    )


def _stage_cpu_sector_samples(sector_bitstrings_np, positions_np):
    """Gather sampled states only, preserving a static per-chain shape."""
    positions_np = np.asarray(positions_np)
    if positions_np.dtype == np.uint64:
        if positions_np.ndim == 1:
            positions_np = positions_np[:, None]
        packed = np.ascontiguousarray(
            positions_np.reshape(-1), dtype=np.uint64
        )
    else:
        positions_np = np.asarray(positions_np, dtype=np.int64)
        if positions_np.ndim == 1:
            positions_np = positions_np[:, None]
        packed = np.asarray(sector_bitstrings_np, dtype=np.uint64)[
            positions_np.reshape(-1)
        ]
    local_states = jnp.asarray(packed, dtype=jnp.uint64)
    local_positions = jnp.arange(packed.size, dtype=jnp.int64).reshape(
        positions_np.shape
    )
    return local_states, local_positions


@partial(jax.jit, static_argnames=("net_cfg", "total_samples"))
def _site_bank_chunk_loss_and_grad(
    bank, states, positions, eloc, energy, total_samples, net_cfg
):
    def loss_fn(current_bank):
        occupied = occupation_states_from_positions(
            states, positions.reshape((-1,)), net_cfg
        )
        psi = local_conv_site_wavefunction_from_bank(
            current_bank, occupied, net_cfg
        )
        centered = jax.lax.stop_gradient(eloc - energy)
        return (
            2.0 * jnp.real(jnp.sum(jnp.conj(centered) * jnp.log(psi.reshape((-1,)))))
            / float(total_samples)
        )

    return jax.value_and_grad(loss_fn)(bank)


def _vmc_chain_batched_loss_and_grad(
    params,
    states,
    positions_np,
    sparse_rows,
    chain_batch_size: int,
    neighbor_padding_size: int | None,
    neighbor_padding_bucket_size: int | None,
    fixed_sparse_row_nnz: int | None,
    config: RunConfig,
    tk: int,
    hamiltonian: TorusCoulombHamiltonian | None,
    basis: BasisData | None,
    stream_sampled_rows: bool,
    sector_bitstrings_np=None,
    sparse_rows_from_bitstrings: bool = False,
    neighbor_sampling_seed: int = 0,
    shared_site_bank=None,
    site_bank_pullback=None,
):
    """Accumulate VMC updates over chain batches."""
    total_samples = int(positions_np.shape[0])
    if total_samples % chain_batch_size != 0:
        raise ValueError(
            "vmc_chain_batch_size must divide n_chains * n_samples so all "
            "microbatches have one static JAX shape."
        )

    eloc_parts = []
    wf_l2_sq = 0.0
    wf_abs_sum = 0.0
    wf_logabs_sum = 0.0
    wf_abs_max = 0.0
    wf_abs_min = np.inf
    wf_count = 0
    sparse_row_build_time = 0.0
    sparse_row_transfer_time = 0.0
    sparse_row_slice_time = 0.0
    local_energy_eval_time = 0.0
    sampled_state_stage_time = 0.0
    gradient_eval_time = 0.0
    metric_stage_time = 0.0
    strict_site_bank = os.environ.get("NQS_SITE_MPS_BANK_WHOLE_STEP") == "1"
    use_site_bank = (strict_site_bank or
                     os.environ.get("NQS_SITE_MPS_BANK_LOCAL_ENERGY") == "1")
    if use_site_bank:
        if (
            config.network.architecture != "mps"
            or config.network.mps_tensor_generation != "local_conv_backflow"
            or config.network.k != 1
            or config.training.vmc_neighbor_approximation != "exact"
            or not sparse_rows_from_bitstrings
            or not stream_sampled_rows
            or int(params["mps"]["conv"]["kernel"].shape[0]) != 1
        ):
            raise ValueError(
                "NQS_SITE_MPS_BANK_LOCAL_ENERGY requires kernel-one, exact, "
                "bitstring-row local_conv_backflow VMC"
            )
        if strict_site_bank:
            if shared_site_bank is None or site_bank_pullback is None:
                raise ValueError("whole-step bank and VJP must be supplied")
            site_bank = shared_site_bank
        else:
            timer = time.perf_counter()
            site_bank = local_conv_site_tensor_bank(params, config.network)
            block_until_ready_tree(site_bank)
            local_energy_eval_time += time.perf_counter() - timer
    else:
        site_bank = None

    for start in range(0, total_samples, chain_batch_size):
        stop = start + chain_batch_size
        positions_chunk_np = positions_np[start:stop]
        positions_chunk = jnp.asarray(positions_chunk_np, dtype=jnp.int64)
        compact_neighbors_cpu = None
        compact_offsets_cpu = None
        if stream_sampled_rows:
            if hamiltonian is None or (
                basis is None and not sparse_rows_from_bitstrings
            ):
                raise ValueError(
                    "chain-batched row streaming requires hamiltonian and basis."
                )
            timer = time.perf_counter()
            if sparse_rows_from_bitstrings:
                if sector_bitstrings_np is None and positions_chunk_np.dtype != np.uint64:
                    raise ValueError(
                        "bitstring sparse rows require a CPU sector bitstring table."
                    )
                chunk = np.asarray(positions_chunk_np)
                sampled_bitstrings = (
                    np.ascontiguousarray(chunk, dtype=np.uint64)
                    if chunk.dtype == np.uint64
                    else np.asarray(
                        sector_bitstrings_np, dtype=np.uint64
                    )[chunk]
                )
                sparse_cpu = (
                    hamiltonian.build_sparse_hamiltonian_bitstring_rows(
                        sampled_bitstrings,
                        pad_to_nnz=fixed_sparse_row_nnz,
                    )
                )
            else:
                sparse_cpu = hamiltonian.build_sparse_hamiltonian_rows_subset(
                    tk,
                    basis,
                    positions_chunk_np,
                    pad_to_nnz=fixed_sparse_row_nnz,
                )
            sparse_row_build_time += time.perf_counter() - timer
            if (
                sparse_rows_from_bitstrings
                and os.environ.get("NQS_COMPACT_STAGE_CPU") == "1"
            ):
                timer = time.perf_counter()
                compact_capacity = int(
                    os.environ["NQS_COMPACT_VALID_SIZE"]
                )
                (
                    compact_neighbors_cpu,
                    compact_vals_cpu,
                    compact_offsets_cpu,
                ) = pack_sparse_bitstring_rows(
                    sparse_cpu,
                    compact_capacity,
                    value_dtype=(
                        np.complex64
                        if os.environ.get("NQS_COMPACT_COEFF_FP32") == "1"
                        else np.complex128
                    ),
                )
                sparse_row_build_time += time.perf_counter() - timer
            timer = time.perf_counter()
            if sparse_rows_from_bitstrings:
                states_chunk = jnp.asarray(
                    sampled_bitstrings.reshape(-1), dtype=jnp.uint64
                )
                positions_chunk = jnp.arange(
                    sampled_bitstrings.size, dtype=jnp.int64
                ).reshape(sampled_bitstrings.shape)
                if compact_neighbors_cpu is None:
                    sparse_chunk = (
                        jnp.asarray(sparse_cpu.diag, dtype=jnp.complex128),
                        None,
                        jnp.asarray(sparse_cpu.vals, dtype=jnp.complex128),
                        jnp.asarray(sparse_cpu.mask, dtype=jnp.float64),
                    )
                else:
                    sparse_chunk = (
                        jnp.asarray(sparse_cpu.diag, dtype=jnp.complex128),
                        None,
                        jnp.asarray(
                            compact_vals_cpu,
                            dtype=(
                                jnp.complex64
                                if os.environ.get("NQS_COMPACT_COEFF_FP32") == "1"
                                else jnp.complex128
                            ),
                        ),
                        jnp.asarray(1.0, dtype=jnp.float64),
                    )
                    compact_offsets_cpu = jnp.asarray(
                        compact_offsets_cpu, dtype=jnp.int32
                    )
            elif states is None:
                states_chunk, positions_chunk, sparse_chunk = (
                    _stage_cpu_sector_rows(
                        sector_bitstrings_np,
                        positions_chunk_np,
                        sparse_cpu,
                    )
                )
            else:
                states_chunk = states
                sparse_chunk = (
                    jnp.asarray(sparse_cpu.diag, dtype=jnp.complex128),
                    jnp.asarray(sparse_cpu.cols, dtype=jnp.int64),
                    jnp.asarray(sparse_cpu.vals, dtype=jnp.complex128),
                    jnp.asarray(sparse_cpu.mask, dtype=jnp.float64),
                )
            block_until_ready_tree((sparse_chunk, compact_offsets_cpu))
            sparse_row_transfer_time += time.perf_counter() - timer
        else:
            states_chunk = states
            timer = time.perf_counter()
            sparse_chunk = (
                sparse_rows.diag[positions_chunk_np],
                sparse_rows.cols[positions_chunk_np],
                sparse_rows.vals[positions_chunk_np],
                sparse_rows.mask[positions_chunk_np],
            )
            block_until_ready_tree(sparse_chunk)
            sparse_row_slice_time += time.perf_counter() - timer

        timer = time.perf_counter()
        eloc_chunk, wf_stats = local_energies_sliced(
            params,
            states_chunk,
            sparse_chunk[0],
            sparse_chunk[1],
            sparse_chunk[2],
            sparse_chunk[3],
            positions_chunk,
            config.network,
            deduplicate_neighbors=True,
            return_wavefunction_stats=True,
            neighbor_approximation=config.training.vmc_neighbor_approximation,
            input_gradient_batch_size=config.training.vmc_input_gradient_batch_size,
            input_hessian_batch_size=(
                config.training.vmc_input_hessian_batch_size
            ),
            exact_neighbors_per_row=(
                config.training.vmc_exact_neighbors_per_row
            ),
            exact_top_neighbors=config.training.vmc_exact_top_neighbors,
            neighbor_sampling_uniform_mix=(
                config.training.vmc_neighbor_sampling_uniform_mix
            ),
            neighbor_sampling_seed=neighbor_sampling_seed + start,
            neighbor_padding_size=neighbor_padding_size,
            neighbor_padding_bucket_size=neighbor_padding_bucket_size,
            neighbor_bitstrings_pos=(
                (compact_neighbors_cpu if compact_neighbors_cpu is not None
                 else sparse_cpu.neighbor_bitstrings)
                if sparse_rows_from_bitstrings
                else None
            ),
            neighbor_compact_offsets=compact_offsets_cpu,
            site_tensor_bank=(
                jax.lax.stop_gradient(site_bank) if use_site_bank else None
            ),
        )
        eloc_chunk, wf_stats = block_until_ready_tree((eloc_chunk, wf_stats))
        if (config.training.vmc_neighbor_approximation != "exact"
                and not np.isfinite(np.asarray(eloc_chunk)).all()):
            raise FloatingPointError(
                "Non-finite approximate/sampled local energy. Use "
                "--vmc_neighbor_approximation exact; no automatic clipping applied."
            )
        local_energy_eval_time += time.perf_counter() - timer
        eloc_parts.append(np.asarray(eloc_chunk, dtype=np.complex128))
        chunk_count = chain_batch_size * config.network.k
        wf_l2_sq += float(wf_stats["wf_l2_norm"]) ** 2
        wf_abs_sum += float(wf_stats["wf_abs_mean"]) * chunk_count
        wf_logabs_sum += float(wf_stats["wf_logabs_mean"]) * chunk_count
        wf_abs_max = max(wf_abs_max, float(wf_stats["wf_abs_max"]))
        wf_abs_min = min(wf_abs_min, float(wf_stats["wf_abs_min"]))
        wf_count += chunk_count

    eloc_np = np.concatenate(eloc_parts)
    energy = float(np.real(np.mean(eloc_np)))
    variance = float(np.real(np.mean(np.abs(eloc_np - energy) ** 2)))
    eloc_mean_imag = float(np.imag(np.mean(eloc_np)))

    def chunk_loss_fn(
        current_params, current_states, current_positions, current_eloc
    ):
        return vmc_surrogate_gradient_from_eloc(
            current_params,
            current_states,
            current_positions,
            current_eloc,
            energy,
            total_samples,
            config.network,
        )

    grad_fn = jax.value_and_grad(chunk_loss_fn) if not strict_site_bank else None
    bank_grad_sum = None
    grad_sums = None
    loss = 0.0
    for start in range(0, total_samples, chain_batch_size):
        stop = start + chain_batch_size
        positions_chunk_np = positions_np[start:stop]
        timer = time.perf_counter()
        if states is None:
            states_chunk, positions_chunk = _stage_cpu_sector_samples(
                sector_bitstrings_np, positions_chunk_np
            )
        else:
            states_chunk = states
            positions_chunk = jnp.asarray(
                positions_chunk_np, dtype=jnp.int64
            )
        eloc_chunk = jnp.asarray(eloc_np[start:stop], dtype=jnp.complex128)
        block_until_ready_tree((states_chunk, positions_chunk, eloc_chunk))
        sampled_state_stage_time += time.perf_counter() - timer
        timer = time.perf_counter()
        if strict_site_bank:
            chunk_loss, chunk_grad = _site_bank_chunk_loss_and_grad(
                site_bank, states_chunk, positions_chunk, eloc_chunk,
                energy, total_samples, config.network,
            )
        else:
            chunk_loss, chunk_grad = grad_fn(
                params, states_chunk, positions_chunk, eloc_chunk
            )
        chunk_loss, chunk_grad = block_until_ready_tree(
            (chunk_loss, chunk_grad)
        )
        loss += float(chunk_loss)
        if strict_site_bank:
            bank_grad_sum = (chunk_grad if bank_grad_sum is None
                             else bank_grad_sum + chunk_grad)
        else:
            chunk_grad_np = jax.tree_util.tree_map(np.asarray, chunk_grad)
            if grad_sums is None:
                grad_sums = jax.tree_util.tree_map(np.array, chunk_grad_np)
            else:
                grad_sums = jax.tree_util.tree_map(
                    lambda total, value: total + value,
                    grad_sums,
                    chunk_grad_np,
                )
        gradient_eval_time += time.perf_counter() - timer
    grad_tree = (site_bank_pullback(bank_grad_sum)[0] if strict_site_bank
                 else jax.tree_util.tree_map(jnp.asarray, grad_sums))
    aux = {
        "energy": jnp.asarray(energy, dtype=jnp.float64),
        "variance": jnp.asarray(variance, dtype=jnp.float64),
        "eloc_mean_imag": jnp.asarray(eloc_mean_imag, dtype=jnp.float64),
        "wf_l2_norm": jnp.asarray(np.sqrt(wf_l2_sq), dtype=jnp.float64),
        "wf_abs_mean": jnp.asarray(wf_abs_sum / wf_count, dtype=jnp.float64),
        "wf_abs_max": jnp.asarray(wf_abs_max, dtype=jnp.float64),
        "wf_abs_min": jnp.asarray(wf_abs_min, dtype=jnp.float64),
        "wf_logabs_mean": jnp.asarray(
            wf_logabs_sum / wf_count, dtype=jnp.float64
        ),
    }
    timer = time.perf_counter()
    if states is None:
        metric_states, metric_positions = _stage_cpu_sector_samples(
            sector_bitstrings_np, positions_np
        )
    else:
        metric_states = states
        metric_positions = jnp.asarray(positions_np, dtype=jnp.int64)
    block_until_ready_tree((metric_states, metric_positions))
    metric_stage_time += time.perf_counter() - timer
    return (
        jnp.asarray(loss, dtype=jnp.float64),
        aux,
        grad_tree,
        sparse_row_build_time,
        sparse_row_transfer_time,
        sparse_row_slice_time,
        local_energy_eval_time,
        sampled_state_stage_time,
        gradient_eval_time,
        metric_stage_time,
        metric_states,
        metric_positions,
    )


def train_projected(
    params,
    states_np,
    sparse_rows: SparseHamiltonianRows,
    config: RunConfig,
) -> TrainingResult:
    states = jnp.asarray(states_np, dtype=jnp.float32)
    history = []

    def loss_fn(
        current_params,
        current_states,
        current_sparse_rows,
        current_step,
    ):
        return projected_loss_sparse(
            current_params,
            current_states,
            current_sparse_rows,
            current_step,
            config.network,
            config.training,
        )

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    optimizer_state = init_optimizer_state(
        params, config.optimizer, config.training.n_steps
    )
    for step in range(1, config.training.n_steps + 1):
        step_start = time.perf_counter()
        loss_start = time.perf_counter()
        (loss, aux), grad_tree = value_and_grad(
            params, states, sparse_rows, step
        )
        loss, aux, grad_tree = block_until_ready_tree(
            (loss, aux, grad_tree)
        )
        time_loss_grad = time.perf_counter() - loss_start

        optimizer_start = time.perf_counter()
        params, optimizer_state, opt_stats = apply_optimizer_update(
            params,
            states,
            grad_tree,
            step,
            config.network,
            config.optimizer,
            optimizer_state,
            config.training.n_steps,
        )
        if (
            config.training.perturb_steps > 0
            and step <= config.training.perturb_steps
        ):
            scale = config.training.perturb_scale * (
                1.0
                - step / (config.training.perturb_steps + 1)
            )
            params = perturb_params(params, scale)
        time_optimizer = time.perf_counter() - optimizer_start
        time_step = time.perf_counter() - step_start

        embedding_stats = (
            mps_backflow_embedding_stats(params, int(states.shape[-1]))
            if config.network.architecture == "mps"
            and config.network.mps_tensor_generation == "backflow"
            else {}
        )

        history.append(
            {
                "step": step,
                "energy": float(aux["energy"]),
                "loss": float(loss),
                **(
                    {}
                    if "wf_l2_norm" not in aux
                    else {
                        "wf_l2_norm": float(aux["wf_l2_norm"]),
                        "wf_abs_mean": float(aux["wf_abs_mean"]),
                        "wf_abs_max": float(aux["wf_abs_max"]),
                        "wf_abs_min": float(aux["wf_abs_min"]),
                        "wf_logabs_mean": float(aux["wf_logabs_mean"]),
                    }
                ),
                "accept": np.nan,
                "time_loss_grad": time_loss_grad,
                "time_optimizer": time_optimizer,
                "time_step": time_step,
                **embedding_stats,
                **opt_stats,
            }
        )
        if (
            step == 1
            or step % 10 == 0
            or step == config.training.n_steps
        ):
            if "evals" in aux:
                shown = " ".join(
                    f"{float(value):.10f}"
                    for value in np.asarray(aux["evals"])[
                        : min(4, config.network.k)
                    ]
                )
                diag = (
                    f"Ritz=[{shown}] "
                    f"S_eigs=[{float(aux['overlap_min']):.3e},"
                    f"{float(aux['overlap_max']):.3e}] "
                    f"Scond={float(aux['overlap_cond']):.3e}"
                )
            else:
                diag = f"Ritz=[{float(aux['energy']):.10f}]"
            wf_text = (
                ""
                if "wf_l2_norm" not in aux
                else f" ||psi||_2={float(aux['wf_l2_norm']):.3e}"
            )
            print(
                f"step {step:5d}: {diag} "
                f"{wf_text} "
                f"{optimizer_log_suffix(opt_stats, time_optimizer, config.optimizer)} "
                f"time(loss_grad={time_loss_grad:.3f}s, total={time_step:.3f}s)"
            )
        if _has_converged(history, config, step):
            break
    return TrainingResult(params, history)


def train_vmc(
    params,
    states_np,
    sparse_rows: SparseHamiltonianRows | None,
    tk: int,
    config: RunConfig,
    h_exact=None,
    hamiltonian: TorusCoulombHamiltonian | None = None,
    basis: BasisData | None = None,
    sector_bitstrings_np=None,
) -> TrainingResult:
    # Store one uint64 bitstring per sector state when available. Selected
    # samples and neighbors are unpacked only after indexing.
    cpu_resident_sector = states_np is None and sector_bitstrings_np is not None
    if cpu_resident_sector:
        states = None
    elif sector_bitstrings_np is None:
        states = jnp.asarray(states_np, dtype=jnp.int8)
    else:
        states = jnp.asarray(sector_bitstrings_np, dtype=jnp.uint64)
    history = []
    live_exact_reference = None
    if config.output.live_plot_interval > 0 and h_exact is not None:
        from scipy.linalg import eigh

        live_exact_reference = float(
            np.sum(
                eigh(
                h_exact,
                    subset_by_index=[0, config.network.k - 1],
                eigvals_only=True,
                check_finite=False,
                )
            )
        )
    stream_sampled_rows = (
        config.training.vmc_row_batch_size is not None
        and sparse_rows is None
    )
    chain_batch_size = config.training.vmc_chain_batch_size
    if config.training.vmc_neighbor_approximation == "input_log_taylor":
        print("WARNING: input-log Taylor neighbor ratios enabled. Energy/variance "
              "and VMC updates are biased; no variational upper-bound guarantee. "
              "Sampler unchanged; input gradients use continuous FP32 backflow.", flush=True)
    elif config.training.vmc_neighbor_approximation == "input_log_taylor2":
        print(
            "WARNING: second-order input-log Taylor neighbor ratios enabled. "
            "Energy/variance and VMC updates remain biased; no variational "
            "upper-bound guarantee.",
            flush=True,
        )
    elif config.training.vmc_neighbor_approximation == "input_log_taylor_cv":
        print(
            "Taylor control variate enabled: local energy and VMC force are "
            "unbiased over neighbor sampling; reported variance includes "
            "neighbor-estimator noise. "
            f"exact_random={config.training.vmc_exact_neighbors_per_row}, "
            f"exact_top={config.training.vmc_exact_top_neighbors}, "
            f"uniform_mix={config.training.vmc_neighbor_sampling_uniform_mix}.",
            flush=True,
        )
    elif config.training.vmc_neighbor_approximation == "exact_neighbor_is":
        print(
            "Exact-neighbor importance sampling enabled: local energy and VMC "
            "force are unbiased over neighbor sampling; reported variance "
            "includes neighbor-estimator noise. "
            f"exact_random={config.training.vmc_exact_neighbors_per_row}, "
            f"exact_top={config.training.vmc_exact_top_neighbors}, "
            f"uniform_mix={config.training.vmc_neighbor_sampling_uniform_mix}.",
            flush=True,
        )
    estimator_chains = (
        config.training.vmc_estimator_chains
        or config.sampler.n_chains
    )
    backflow_neighbor_forward = (
        config.network.architecture == "mps"
        and config.network.mps_tensor_generation
        in {"backflow", "site_head_backflow"}
    )
    use_neighbor_forward = (
        config.training.vmc_deduplicate_neighbors
        or backflow_neighbor_forward
        or config.training.vmc_sparse_rows_from_bitstrings
    )
    if chain_batch_size is not None and not use_neighbor_forward:
        raise ValueError(
            "vmc_chain_batch_size currently requires "
            "--vmc_deduplicate_neighbors."
        )
    fixed_sparse_row_nnz = config.training.vmc_fixed_neighbor_row_nnz
    neighbor_padding_bucket_size = (
        config.training.vmc_neighbor_padding_bucket_size
    )
    fixed_neighbor_padding_size = None
    use_sliced = (
        chain_batch_size is not None
        or stream_sampled_rows
        or use_neighbor_forward
        or int(sparse_rows.diag.shape[0]) > 10000
    )
    if use_sliced:

        def loss_fn(
            current_params, current_states, sparse_sliced, current_positions
        ):
            return vmc_surrogate_loss_sliced(
                current_params,
                current_states,
                sparse_sliced,
                current_positions,
                config.network,
                deduplicate_neighbors=use_neighbor_forward,
                neighbor_padding_size=fixed_neighbor_padding_size,
                neighbor_padding_bucket_size=neighbor_padding_bucket_size,
            )

        if stream_sampled_rows:
            sliced_reason = "streamed sampled rows"
        elif backflow_neighbor_forward:
            sliced_reason = "backflow neighbor forward"
        elif chain_batch_size is not None:
            sliced_reason = "chain microbatching"
        else:
            sliced_reason = f"sector_size={sparse_rows.diag.shape[0]} > 10000"
        print(f"  using sliced loss ({sliced_reason})")
    else:

        def loss_fn(
            current_params,
            current_states,
            current_sparse_rows,
            current_positions,
        ):
            return vmc_surrogate_loss(
                current_params,
                current_states,
                current_sparse_rows,
                current_positions,
                config.network,
                deduplicate_neighbors=use_neighbor_forward,
            )

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    if config.training.vmc_neighbor_approximation == "input_log_taylor":
        print("  VMC neighbor evaluation: row-centered input-log Taylor; "
              "no neighbor forward or neighbor deduplication")
    elif config.training.vmc_neighbor_approximation == "input_log_taylor2":
        print("  VMC neighbor evaluation: second-order row-centered "
              "input-log Taylor; no neighbor forward")
    elif config.training.vmc_neighbor_approximation == "input_log_taylor_cv":
        print("  VMC neighbor evaluation: input-log Taylor control variate "
              "with sampled exact-neighbor correction")
    elif config.training.vmc_neighbor_approximation == "exact_neighbor_is":
        print("  VMC neighbor evaluation: exact top terms plus unbiased "
              "importance-sampled exact tail")
    elif use_neighbor_forward and os.environ.get("NQS_NO_DEDUP"):
        print("  VMC neighbor evaluation: exact row-major neighbors, no deduplication")
    elif use_neighbor_forward:
        mode = "backflow MPS" if backflow_neighbor_forward else "configured"
        padding_mode = (
            "dynamic unique shapes"
            if config.training.vmc_disable_neighbor_padding
            else "fixed padding"
        )
        print(
            "  VMC neighbor evaluation: global deduplication with "
            f"{padding_mode} ({mode})"
        )
    if config.training.vmc_sparse_rows_from_bitstrings:
        print(
            "  sparse-row targets: sampled bitstrings; neighbor sector-index "
            "lookup disabled"
        )
    else:
        print(
            "  VMC local energy: full-sector wavefunction forward, "
            "then Hamiltonian-neighbor gather"
        )
    if config.sampler.proposal_mode != "sector_uniform":
        print(
            f"  sampler proposal mode: "
            f"{config.sampler.proposal_mode}"
        )
    if stream_sampled_rows:
        print(
            "  local-energy row streaming: full sampled batch"
        )
        print(
            "  sampled sparse rows will be built on demand during "
            "local-energy evaluation"
        )
    elif use_sliced:
        print("  local-energy evaluation: full sampled batch")
        if config.training.vmc_prebuild_sparse_rows:
            print("  full sparse rows are prebuilt and sliced per sample batch")
    if chain_batch_size is not None:
        print(
            "  VMC chain microbatching: "
            f"batch_size={chain_batch_size}, global gradient/KFAC accumulation"
        )
    if estimator_chains != config.sampler.n_chains:
        print(
            "  VMC sampler pool: "
            f"pool_chains={config.sampler.n_chains}, "
            f"estimator_chains={estimator_chains}, "
            "selection=rotating contiguous window"
        )
    probability_snapshot_steps = set(config.output.prob_snapshot_steps)
    if probability_snapshot_steps and config.output.prob_snapshot_dir is None:
        print("  skip probability snapshots: prob_snapshot_dir is not set")
    sampler = MetropolisSamplerJAX(
        params,
        sector_bitstrings_np if cpu_resident_sector else states,
        config.sampler.n_chains,
        config.network,
        seed=config.training.seed + 10000 + tk,
        proposal_mode=config.sampler.proposal_mode,
        sparse_rows=sparse_rows,
        tempering_betas=config.sampler.tempering_betas,
        tempering_swap_interval=config.sampler.tempering_swap_interval,
        sector_states_host=(
            sector_bitstrings_np if cpu_resident_sector else states_np
        ),
        sector_nphi=config.system.num_sites,
        sector_n_particles=config.system.ne,
        sector_on_host=cpu_resident_sector,
        streaming_hamiltonian=hamiltonian,
        streaming_basis=basis,
        streaming_tk=tk,
        streaming_row_nnz=fixed_sparse_row_nnz,
        hpsi_candidates=config.sampler.hpsi_candidates,
        sector_tk=(
            tk if config.sampler.proposal_mode == "conserved_pair" else None
        ),
    )
    if cpu_resident_sector:
        if chain_batch_size is None or not stream_sampled_rows:
            raise ValueError(
                "CPU-resident sectors require streamed sampled rows and "
                "vmc_chain_batch_size."
            )
        if config.sampler.proposal_mode == "conserved_pair":
            print(
                "  sector storage: none; chains are packed uint64 bitstrings "
                "with symmetric conserved-pair proposals"
            )
        else:
            print(
                "  sector storage: CPU uint64 table; GPU receives only sampled "
                "and evaluated neighbor states"
            )
    if sampler.n_replicas > 1:
        print(
            "  parallel tempering: "
            f"target_chains={sampler.n_target_chains}, "
            f"replicas={sampler.n_replicas}, "
            f"total_chains={sampler.n_chains}, "
            f"betas={np.array2string(sampler.tempering_betas, precision=4)}, "
            f"swap_interval={sampler.tempering_swap_interval}"
        )
    if sampler.allowed_outgoing_cpu is not None:
        allowed_table = sampler.allowed_outgoing_cpu
        fixed_sparse_row_nnz = int(allowed_table.max_row_nnz)
        print(
            "  CPU allowed-outgoing table: "
            f"entries={allowed_table.outgoing_masks.size}, "
            f"memory={allowed_table.nbytes / (1024**2):.2f} MiB, "
            f"proposal_backend={sampler.cpu_proposal_backend}"
        )
    if sampler.laqx_proposal_cpu is not None:
        proposal_table = sampler.laqx_proposal_cpu
        print(
            "  LaQX Hamiltonian proposal table: "
            f"rows={proposal_table.cols.shape[0]}, "
            f"max_neighbors={proposal_table.cols.shape[1]}, "
            f"memory={proposal_table.nbytes / (1024**2):.2f} MiB, "
            "weight=max(|H_ij|, 0.05)"
        )
    if (use_neighbor_forward and not config.training.vmc_disable_neighbor_padding
            and config.training.vmc_neighbor_approximation == "exact"):
        if fixed_sparse_row_nnz is None:
            if sparse_rows is None:
                raise ValueError(
                    "fixed neighbor padding requires sparse rows or a "
                    "proposal table with max_row_nnz."
                )
            fixed_sparse_row_nnz = int(sparse_rows.cols.shape[-1])
        padding_upper_bound = int(
            (chain_batch_size or (
                sampler.n_target_chains * config.sampler.n_samples
            ))
            * config.network.k
            * fixed_sparse_row_nnz
        )
        sector_size_for_padding = (
            int(basis.number_tk[tk])
            if config.sampler.proposal_mode == "conserved_pair"
            and basis is not None
            else sampler.sector_size
        )
        fixed_neighbor_padding_size = min(
            sector_size_for_padding, padding_upper_bound
        )
    if config.training.vmc_neighbor_approximation != "exact":
        print(
            "  sampled/approximated row width="
            f"{fixed_sparse_row_nnz or 'dynamic'}; full-neighbor bucket unused"
        )
    elif use_neighbor_forward and os.environ.get("NQS_NO_DEDUP"):
        print(
            f"  neighbor padding: fixed row width={fixed_sparse_row_nnz or 0}; "
            "unique-neighbor bucket is inactive in no-dedup mode"
        )
    elif use_neighbor_forward:
        print(
            "  neighbor padding: "
            f"enabled={fixed_neighbor_padding_size is not None}, "
            f"max_row_nnz={fixed_sparse_row_nnz or 0}, "
            f"slots={fixed_neighbor_padding_size or 0}, "
            f"bucket={neighbor_padding_bucket_size or 0}"
        )
    optimizer_state = init_optimizer_state(
        params, config.optimizer, config.training.n_steps
    )

    for step in range(1, config.training.n_steps + 1):
        profile_this_step = (
            chain_batch_size is None
            and _should_profile_vmc_timing(config, step)
        )
        step_start = time.perf_counter()
        sample_start = time.perf_counter()
        strict_site_bank = os.environ.get("NQS_SITE_MPS_BANK_WHOLE_STEP") == "1"
        site_bank = site_bank_pullback = site_bank_cache = None
        if strict_site_bank:
            if (
                chain_batch_size is None
                or config.network.mps_tensor_generation != "local_conv_backflow"
                or config.network.k != 1
                or config.network.mps_attention_blocks
                or config.network.mps_residual_blocks
                or config.network.mps_jastrow
                or config.training.vmc_neighbor_approximation != "exact"
                or not cpu_resident_sector
                or not config.training.vmc_sparse_rows_from_bitstrings
                or not sampler.sector_on_host
                or sampler.proposal_mode != "sector_uniform"
            ):
                raise ValueError(
                    "whole-step site bank requires exact, bitstring-row, "
                    "CPU-sector, chain-batched sector-uniform VMC"
                )
            site_bank, site_bank_pullback, site_bank_cache = jax.vjp(
                lambda p: local_conv_site_tensor_bank_with_cache(
                    p, config.network
                ), params, has_aux=True
            )
            block_until_ready_tree((site_bank, site_bank_cache))
        sampler.set_params(params, site_tensor_bank=site_bank)
        n_burn = (
            config.sampler.n_burn_warmup
            if (
                config.sampler.n_burn_warmup is not None
                and step == 1
            )
            else config.sampler.n_burn
        )
        if step == 1 and config.sampler.n_burn_warmup is not None:
            print(f"  warmup burn: n_burn={n_burn}")
        positions_np, accept = sampler.sample(
            n_burn,
            config.sampler.n_between,
            config.sampler.n_samples,
        )
        time_sample = time.perf_counter() - sample_start
        positions_np = _select_vmc_estimator_positions(
            positions_np,
            pool_chains=config.sampler.n_chains,
            estimator_chains=estimator_chains,
            n_samples=config.sampler.n_samples,
            step=step,
        )
        positions = jnp.asarray(
            positions_np,
            dtype=(jnp.uint64 if positions_np.dtype == np.uint64 else jnp.int64),
        )
        metric_states = states
        metric_positions = positions

        loss_start = time.perf_counter()
        wavefunction_cache_before = (
            wavefunction_compile_cache_size()
            if profile_this_step
            else -1
        )
        sparse_row_build_time = 0.0
        sparse_row_transfer_time = 0.0
        sparse_row_slice_time = 0.0
        chain_local_energy_time = 0.0
        chain_sample_stage_time = 0.0
        chain_gradient_time = 0.0
        chain_metric_stage_time = 0.0
        if chain_batch_size is not None:
            (
                loss,
                aux,
                grad_tree,
                sparse_row_build_time,
                sparse_row_transfer_time,
                sparse_row_slice_time,
                chain_local_energy_time,
                chain_sample_stage_time,
                chain_gradient_time,
                chain_metric_stage_time,
                metric_states,
                metric_positions,
            ) = _vmc_chain_batched_loss_and_grad(
                params,
                states,
                positions_np,
                sparse_rows,
                chain_batch_size,
                fixed_neighbor_padding_size,
                neighbor_padding_bucket_size,
                fixed_sparse_row_nnz,
                config,
                tk,
                hamiltonian,
                basis,
                stream_sampled_rows,
                sector_bitstrings_np=(
                    sector_bitstrings_np if cpu_resident_sector else None
                ),
                sparse_rows_from_bitstrings=(
                    config.training.vmc_sparse_rows_from_bitstrings
                ),
                neighbor_sampling_seed=(
                    config.training.seed + step * 1000003
                ),
                shared_site_bank=site_bank,
                site_bank_pullback=site_bank_pullback,
            )
        elif use_sliced:
            if stream_sampled_rows:
                if hamiltonian is None or basis is None:
                    raise ValueError(
                        "on-demand VMC row streaming requires hamiltonian and basis."
                    )
                sparse_build_start = time.perf_counter()
                sparse_sliced_cpu = (
                    hamiltonian.build_sparse_hamiltonian_rows_subset(
                        tk,
                        basis,
                        positions_np,
                        pad_to_nnz=fixed_sparse_row_nnz,
                    )
                )
                sparse_row_build_time = (
                    time.perf_counter() - sparse_build_start
                )
                sparse_transfer_start = time.perf_counter()
                sparse_sliced = (
                    jnp.asarray(
                        sparse_sliced_cpu.diag, dtype=jnp.complex128
                    ),
                    jnp.asarray(
                        sparse_sliced_cpu.cols, dtype=jnp.int64
                    ),
                    jnp.asarray(
                        sparse_sliced_cpu.vals, dtype=jnp.complex128
                    ),
                    jnp.asarray(
                        sparse_sliced_cpu.mask, dtype=jnp.float64
                    ),
                )
                block_until_ready_tree(sparse_sliced)
                sparse_row_transfer_time = (
                    time.perf_counter() - sparse_transfer_start
                )
            else:
                sparse_slice_start = time.perf_counter()
                sparse_sliced = (
                    sparse_rows.diag[positions_np],
                    sparse_rows.cols[positions_np],
                    sparse_rows.vals[positions_np],
                    sparse_rows.mask[positions_np],
                )
                block_until_ready_tree(sparse_sliced)
                sparse_row_slice_time = (
                    time.perf_counter() - sparse_slice_start
                )
            (loss, aux), grad_tree = value_and_grad(
                params, states, sparse_sliced, positions
            )
        else:
            (loss, aux), grad_tree = value_and_grad(
                params, states, sparse_rows, positions
            )
        loss, aux, grad_tree = block_until_ready_tree(
            (loss, aux, grad_tree)
        )
        time_loss_grad = time.perf_counter() - loss_start
        wavefunction_cache_after = (
            wavefunction_compile_cache_size()
            if profile_this_step
            else -1
        )
        timing_breakdown = None
        if profile_this_step:
            timing_analysis_start = time.perf_counter()
            if use_sliced:
                timing_breakdown = profile_vmc_loss_sliced(
                    params,
                    states,
                    sparse_sliced[0],
                    sparse_sliced[1],
                    sparse_sliced[2],
                    sparse_sliced[3],
                    positions,
                    config.network,
                    deduplicate_neighbors=use_neighbor_forward,
                )
            else:
                timing_breakdown = profile_vmc_loss(
                    params,
                    states,
                    sparse_rows,
                    positions,
                    config.network,
                    deduplicate_neighbors=use_neighbor_forward,
                )
            timing_breakdown["sparse_row_build_cpu"] = (
                sparse_row_build_time
            )
            timing_breakdown["sparse_row_transfer"] = (
                sparse_row_transfer_time
            )
            timing_breakdown["sparse_row_slice"] = (
                sparse_row_slice_time
            )
            if use_sliced:
                timing_breakdown.update(
                    profile_vmc_autodiff_sliced(
                        params,
                        states,
                        sparse_sliced[0],
                        sparse_sliced[1],
                        sparse_sliced[2],
                        sparse_sliced[3],
                        positions,
                        config.network,
                        deduplicate_neighbors=use_neighbor_forward,
                    )
                )
            else:
                timing_breakdown.update(
                    profile_vmc_autodiff(
                        params,
                        states,
                        sparse_rows,
                        positions,
                        config.network,
                        deduplicate_neighbors=use_neighbor_forward,
                    )
                )
            timing_breakdown["loss_grad_total"] = time_loss_grad
            timing_breakdown["loss_grad_compute"] = max(
                0.0,
                time_loss_grad
                - sparse_row_build_time
                - sparse_row_transfer_time
                - sparse_row_slice_time,
            )
            timing_breakdown["wavefunction_cache_before"] = (
                wavefunction_cache_before
            )
            timing_breakdown["wavefunction_cache_after"] = (
                wavefunction_cache_after
            )
            timing_breakdown["wavefunction_cache_misses"] = (
                max(
                    0,
                    wavefunction_cache_after - wavefunction_cache_before,
                )
                if wavefunction_cache_before >= 0
                and wavefunction_cache_after >= 0
                else -1
            )
            timing_breakdown["dynamic_shape_overhead_estimate"] = max(
                0.0,
                timing_breakdown["loss_grad_compute"]
                - timing_breakdown["loss_value_and_grad_profiled"],
            )
            timing_breakdown["timing_analysis_overhead"] = (
                time.perf_counter() - timing_analysis_start
            )

        optimizer_start = time.perf_counter()
        params, optimizer_state, opt_stats = apply_optimizer_update(
            params,
            metric_states,
            grad_tree,
            step,
            config.network,
            config.optimizer,
            optimizer_state,
            config.training.n_steps,
            metric_positions=metric_positions,
            metric_batch_size=chain_batch_size,
            site_bank=site_bank,
            site_bank_cache=site_bank_cache,
        )
        if (
            config.training.perturb_steps > 0
            and step <= config.training.perturb_steps
        ):
            scale = config.training.perturb_scale * (
                1.0
                - step / (config.training.perturb_steps + 1)
            )
            params = perturb_params(params, scale)
        time_optimizer = time.perf_counter() - optimizer_start
        time_step = time.perf_counter() - step_start

        embedding_stats = (
            mps_backflow_embedding_stats(
                params, int(config.system.num_sites)
            )
            if config.network.architecture == "mps"
            and config.network.mps_tensor_generation == "backflow"
            else {}
        )

        history.append(
            {
                "step": step,
                "energy": float(aux["energy"]),
                "loss": float(loss),
                "variance": float(aux["variance"]),
                "eloc_mean_imag": float(aux["eloc_mean_imag"]),
                "wf_l2_norm": float(aux["wf_l2_norm"]),
                "wf_abs_mean": float(aux["wf_abs_mean"]),
                "wf_abs_max": float(aux["wf_abs_max"]),
                "wf_abs_min": float(aux["wf_abs_min"]),
                "wf_logabs_mean": float(aux["wf_logabs_mean"]),
                "accept": float(accept),
                "sampler_pool_chains": int(config.sampler.n_chains),
                "estimator_chains": int(estimator_chains),
                "neighbor_approximation": config.training.vmc_neighbor_approximation,
                **(
                    {"swap_accept": sampler.swap_acceptance_rate()}
                    if sampler.n_replicas > 1
                    else {}
                ),
                "time_sample": time_sample,
                "time_loss_grad": time_loss_grad,
                "time_optimizer": time_optimizer,
                "time_step": time_step,
                **(
                    {
                        "timing_sparse_row_build_cpu": sparse_row_build_time,
                        "timing_sparse_row_transfer": sparse_row_transfer_time,
                        "timing_chain_local_energy": chain_local_energy_time,
                        "timing_chain_sample_stage": chain_sample_stage_time,
                        "timing_chain_gradient": chain_gradient_time,
                        "timing_chain_metric_stage": chain_metric_stage_time,
                    }
                    if chain_batch_size is not None
                    else {}
                ),
                **embedding_stats,
                **opt_stats,
                **(
                    {}
                    if timing_breakdown is None
                    else {
                        f"timing_{key}": value
                        for key, value in timing_breakdown.items()
                    }
                ),
            }
        )
        if (
            config.output.live_plot_interval > 0
            and (
                step % config.output.live_plot_interval == 0
                or step == config.training.n_steps
            )
        ):
            plot_live_vmc_energy(
                history,
                config.output.live_plot_output,
                live_exact_reference,
            )
        if (
            step == 1
            or step % 10 == 0
            or step == config.training.n_steps
        ):
            print(
                f"step {step:5d}: VMC E={float(aux['energy']):.10f} "
                f"Var(E_loc)={float(aux['variance']):.3e} "
                f"Im<E_loc>={float(aux['eloc_mean_imag']):.3e} "
                f"||psi||_2={float(aux['wf_l2_norm']):.3e} "
                f"acc={accept:.3f} "
                + (
                    f"swap={sampler.swap_acceptance_rate():.3f} "
                    if sampler.n_replicas > 1
                    else ""
                )
                + f"{optimizer_log_suffix(opt_stats, time_optimizer, config.optimizer)} "
                f"time(sample={time_sample:.3f}s, "
                f"loss_grad={time_loss_grad:.3f}s, total={time_step:.3f}s)"
            )
            if timing_breakdown is not None:
                print(
                    "  loss_grad timing: "
                    f"build_rows={timing_breakdown['sparse_row_build_cpu']:.3f}s "
                    f"h2d={timing_breakdown['sparse_row_transfer']:.3f}s "
                    f"slice_rows={timing_breakdown['sparse_row_slice']:.3f}s "
                    f"logpsi_gather={timing_breakdown['logpsi_state_gather']:.3f}s "
                    f"logpsi_net={timing_breakdown['logpsi_network_forward']:.3f}s "
                    f"logpsi_det={timing_breakdown['logpsi_slogdet']:.3f}s "
                    f"sample_gather={timing_breakdown['sampled_state_gather']:.3f}s "
                    f"sample_net={timing_breakdown['sampled_network_forward']:.3f}s "
                    f"cols_d2h={timing_breakdown['neighbor_cols_d2h']:.3f}s "
                    f"unique_cpu={timing_breakdown['neighbor_unique_cpu']:.3f}s "
                    f"idx_h2d={timing_breakdown['neighbor_indices_h2d']:.3f}s "
                    f"neighbor_gather={timing_breakdown['neighbor_state_gather']:.3f}s "
                    f"neighbor_net_first={timing_breakdown['neighbor_forward_first']:.3f}s "
                    f"neighbor_net_warm={timing_breakdown['neighbor_forward_warm']:.3f}s "
                    f"inverse_gather={timing_breakdown['neighbor_inverse_gather']:.3f}s "
                    f"scatter={timing_breakdown['offdiag_segment_scatter']:.3f}s "
                    f"solve={timing_breakdown['solve_trace']:.3f}s "
                    f"reduce={timing_breakdown['loss_reduction']:.3f}s "
                    f"loss_fwd={timing_breakdown['loss_forward_eval']:.3f}s "
                    f"vgrad_warm={timing_breakdown['loss_value_and_grad_profiled']:.3f}s "
                    f"backward_extra={timing_breakdown['loss_backward_extra']:.3f}s "
                    f"profiled_first={timing_breakdown['profiled_total']:.3f}s "
                    f"profiled_warm={timing_breakdown['profiled_warm_total']:.3f}s "
                    f"loss_compute={timing_breakdown['loss_grad_compute']:.3f}s "
                    f"shape_overhead={timing_breakdown['dynamic_shape_overhead_estimate']:.3f}s "
                    f"total={timing_breakdown['loss_grad_total']:.3f}s "
                    f"cache_miss={timing_breakdown['wavefunction_cache_misses']} "
                    f"profile_overhead={timing_breakdown['timing_analysis_overhead']:.3f}s "
                    f"neighbors={timing_breakdown['neighbor_total']} "
                    f"evaluated={timing_breakdown['neighbor_evaluated']} "
                    f"dedup={timing_breakdown['neighbor_deduplicated']}"
                )
        if (
            config.network.k == 1
            and step % 100 == 0
            and config.sampler.prob_n_samples > 0
        ):
            diagnostic_start = time.perf_counter()
            metrics = probability_metrics(
                params,
                states_np,
                h_exact,
                config.sampler.n_chains,
                config.sampler.prob_n_burn,
                config.sampler.prob_n_between,
                config.sampler.prob_n_samples,
                config.training.seed + 200000 + tk + step,
                config,
                sampler=sampler,
            )
            (
                nn_l1,
                nn_l2,
                mc_l1,
                mc_l2,
                nm_l1,
                nm_l2,
                mc_acc,
                mc_n,
            ) = metrics
            print(
                f"  prob diag step {step:5d}: "
                f"NN L1={nn_l1:.4e} L2={nn_l2:.4e} | "
                f"MC L1={mc_l1:.4e} L2={mc_l2:.4e} | "
                f"NNvsMC L1={nm_l1:.4e} L2={nm_l2:.4e} "
                f"acc={mc_acc:.3f} samples={mc_n} "
                f"time={time.perf_counter() - diagnostic_start:.2f}s"
            )
        if (
            config.network.k == 1
            and step in probability_snapshot_steps
            and config.output.prob_snapshot_dir is not None
        ):
            if h_exact is None:
                print(
                    f"  skip probability snapshot step {step}: "
                    "dense Hamiltonian is unavailable"
                )
            else:
                snapshot_dir = Path(config.output.prob_snapshot_dir)
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_output = (
                    snapshot_dir / f"probability_step_{step:05d}.png"
                )
                plot_probability_distribution(
                    params,
                    states_np,
                    h_exact,
                    tk,
                    str(snapshot_output),
                    config,
                    seed=config.training.seed + 400000 + tk + step,
                    title_suffix=f", step {step}",
                )
        if _has_converged(history, config, step):
            break
    return TrainingResult(params, history)


def _device_peak_memory_bytes() -> int | None:
    try:
        stats = jax.devices()[0].memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if not stats:
        return None
    for key in ("peak_bytes_in_use", "peak_bytes_reserved"):
        if key in stats:
            return int(stats[key])
    return None


def train_continuous_vmc(
    params,
    states_np,
    sparse_rows: SparseHamiltonianRows,
    plan: ContinuousEvaluationPlan,
    tk: int,
    config: RunConfig,
) -> TrainingResult:
    if config.network.k != 1:
        raise ValueError("continuous_vmc currently requires network.k == 1.")
    states = jnp.asarray(states_np, dtype=jnp.float64)
    history = []

    def loss_fn(current_params, coordinates):
        return continuous_vmc_surrogate_loss(
            current_params,
            states,
            sparse_rows,
            plan,
            coordinates,
            config.network,
            config.continuous,
        )

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    sampler = ContinuousMetropolisSamplerJAX(
        params,
        states_np,
        plan.active_rows,
        config.sampler.n_chains,
        config.network,
        config.continuous,
        seed=config.training.seed + 30000 + tk,
    )
    optimizer_state = init_optimizer_state(
        params, config.optimizer, config.training.n_steps
    )
    print(
        "  continuous occupation VMC: "
        f"basis={config.continuous.basis}, "
        f"active_rows={int(plan.active_rows.shape[0])}, "
        f"unique_states={plan.unique_count}, "
        f"neighbors={plan.neighbor_count}, "
        f"forward_batch_size={config.network.forward_batch_size}"
    )

    for step in range(1, config.training.n_steps + 1):
        step_start = time.perf_counter()
        sample_start = time.perf_counter()
        sampler.set_params(params)
        n_burn = (
            config.sampler.n_burn_warmup
            if (
                config.sampler.n_burn_warmup is not None
                and step == 1
            )
            else config.sampler.n_burn
        )
        adapt_width = step == 1
        if step == 1:
            print(
                f"  continuous warmup burn: n_burn={n_burn}, "
                f"initial_width={sampler.proposal_width:.6g}"
            )
        coordinates_np, accept = sampler.sample(
            n_burn,
            config.sampler.n_between,
            config.sampler.n_samples,
            adapt_width=adapt_width,
            freeze_after_burn=adapt_width,
        )
        time_sample = time.perf_counter() - sample_start
        coordinates = jnp.asarray(coordinates_np, dtype=jnp.float64)

        loss_start = time.perf_counter()
        try:
            (loss, aux), grad_tree = value_and_grad(params, coordinates)
            loss, aux, grad_tree = block_until_ready_tree(
                (loss, aux, grad_tree)
            )
        except Exception as exc:
            message = str(exc).lower()
            if (
                "out of memory" in message
                or "resource_exhausted" in message
                or "resource exhausted" in message
            ):
                peak = _device_peak_memory_bytes()
                peak_text = (
                    "unknown"
                    if peak is None
                    else f"{peak / (1024**3):.3f} GiB"
                )
                print(
                    "  continuous_vmc OOM diagnostics: "
                    f"active_rows={int(plan.active_rows.shape[0])}, "
                    f"unique_states={plan.unique_count}, "
                    f"neighbors={plan.neighbor_count}, "
                    f"peak_device_memory={peak_text}"
                )
            raise
        time_loss_grad = time.perf_counter() - loss_start

        invalid_nodes = int(aux["invalid_node_count"])
        if invalid_nodes:
            raise FloatingPointError(
                "continuous_vmc encountered "
                f"{invalid_nodes} samples with |Psi(x)| < "
                f"{config.continuous.node_epsilon:.3e}; "
                f"minimum |Psi|={float(aux['min_abs_psi']):.3e}."
            )

        optimizer_start = time.perf_counter()
        params, optimizer_state, opt_stats = (
            apply_continuous_optimizer_update(
                params,
                states,
                coordinates,
                plan.active_rows,
                grad_tree,
                step,
                config.network,
                config.continuous,
                config.optimizer,
                optimizer_state,
                config.training.n_steps,
            )
        )
        if (
            config.training.perturb_steps > 0
            and step <= config.training.perturb_steps
        ):
            scale = config.training.perturb_scale * (
                1.0
                - step / (config.training.perturb_steps + 1)
            )
            params = perturb_params(params, scale)
        time_optimizer = time.perf_counter() - optimizer_start
        time_step = time.perf_counter() - step_start

        history.append(
            {
                "step": step,
                "energy": float(aux["energy"]),
                "loss": float(loss),
                "variance": float(aux["variance"]),
                "eloc_mean_imag": float(aux["eloc_mean_imag"]),
                "accept": float(accept),
                "proposal_width": float(sampler.proposal_width),
                "invalid_node_count": invalid_nodes,
                "min_abs_psi": float(aux["min_abs_psi"]),
                "unique_state_count": plan.unique_count,
                "neighbor_count": plan.neighbor_count,
                "active_union_count": int(aux["active_union_count"]),
                "peak_device_memory_bytes": _device_peak_memory_bytes(),
                "time_sample": time_sample,
                "time_loss_grad": time_loss_grad,
                "time_optimizer": time_optimizer,
                "time_step": time_step,
                **opt_stats,
            }
        )
        if (
            step == 1
            or step % 10 == 0
            or step == config.training.n_steps
        ):
            print(
                f"step {step:5d}: continuous VMC "
                f"E={float(aux['energy']):.10f} "
                f"Var(E_loc)={float(aux['variance']):.3e} "
                f"Im<E_loc>={float(aux['eloc_mean_imag']):.3e} "
                f"acc={accept:.3f} width={sampler.proposal_width:.4g} "
                f"active={int(aux['active_union_count'])} "
                f"{optimizer_log_suffix(opt_stats, time_optimizer, config.optimizer)} "
                f"time(sample={time_sample:.3f}s, "
                f"loss_grad={time_loss_grad:.3f}s, total={time_step:.3f}s)"
            )
        if _has_converged(history, config, step):
            break
    return TrainingResult(params, history)
