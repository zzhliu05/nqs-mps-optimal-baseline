"""Benchmark a sample-sharded SR Jacobian without gathering it to one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from nqs_mps.runtime import configure
from nqs_mps.settings import core_config, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", type=int, default=4)
    args = parser.parse_args()

    settings = load_settings(args.config)
    configure(settings)

    import jax
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree

    from nqs_torus_laqx_nes.network import (
        determinant_sr_scalars,
        init_params,
    )

    config = core_config(settings, args.output.parent / "unused")
    devices = jax.local_devices()[: args.devices]
    if len(devices) != args.devices:
        raise RuntimeError(
            f"Requested {args.devices} devices, found {len(devices)}"
        )

    sample_count = int(config.optimizer.sr_samples or settings["sampling"]["chains"])
    if sample_count % len(devices):
        raise ValueError("SR samples must be divisible by the device count")
    local_samples = sample_count // len(devices)
    jacobian_batch = int(config.optimizer.sr_jacobian_batch_size)
    if local_samples % jacobian_batch:
        raise ValueError("Local SR samples must be divisible by the Jacobian batch")

    params = init_params(
        config.training.seed, config.system.nphi, config.network
    )
    theta, unravel = ravel_pytree(params)
    parameter_count = int(theta.size)

    rng = np.random.default_rng(config.training.seed)
    states = np.zeros(
        (sample_count, config.network.k, config.system.nphi),
        dtype=np.float32,
    )
    for sample in range(sample_count):
        occupied = rng.choice(
            config.system.nphi, size=config.system.ne, replace=False
        )
        states[sample, 0, occupied] = 1.0
    states = states.reshape(
        (len(devices), local_samples, config.network.k, config.system.nphi)
    )

    replicated_theta = jnp.broadcast_to(
        theta[None, :], (len(devices), parameter_count)
    )
    sharded_states = jnp.asarray(states)

    def local_jacobian(flat_params, state_batch):
        def score_components(candidate):
            scores = determinant_sr_scalars(
                unravel(candidate), state_batch, config.network
            )
            return jnp.stack((jnp.real(scores), jnp.imag(scores)), axis=1)

        return jax.jacrev(score_components)(flat_params).astype(jnp.float32)

    jacobian_map = jax.pmap(local_jacobian, in_axes=(0, 0))
    psum_mean = jax.pmap(
        lambda local_sum: jax.lax.psum(local_sum, "devices") / sample_count,
        axis_name="devices",
    )
    center_map = jax.pmap(
        lambda block, mean: (block - mean).reshape(
            (2 * jacobian_batch, parameter_count)
        )
        / jnp.sqrt(jnp.asarray(sample_count, dtype=jnp.float32)),
        in_axes=(0, 0),
    )

    start = time.perf_counter()
    raw_blocks = []
    local_sum = None
    for offset in range(0, local_samples, jacobian_batch):
        block = jacobian_map(
            replicated_theta,
            sharded_states[:, offset : offset + jacobian_batch],
        )
        block.block_until_ready()
        raw_blocks.append(block)
        block_sum = jnp.sum(block, axis=1)
        local_sum = block_sum if local_sum is None else local_sum + block_sum
    jacobian_seconds = time.perf_counter() - start

    center_start = time.perf_counter()
    global_mean = psum_mean(local_sum)
    global_mean.block_until_ready()
    design_blocks = []
    for block in raw_blocks:
        centered = center_map(block, global_mean)
        centered.block_until_ready()
        design_blocks.append(centered)
    del raw_blocks
    center_seconds = time.perf_counter() - center_start

    norm_start = time.perf_counter()
    local_norm = sum(jnp.sum(block * block, axis=(1, 2)) for block in design_blocks)
    global_norm = jax.pmap(
        lambda value: jax.lax.psum(value, "devices"),
        axis_name="devices",
    )(local_norm)
    global_norm.block_until_ready()
    norm_seconds = time.perf_counter() - norm_start

    result = {
        "devices": [str(device) for device in devices],
        "samples": sample_count,
        "samples_per_device": local_samples,
        "jacobian_batch_size": jacobian_batch,
        "parameters": parameter_count,
        "local_design_shape": [2 * local_samples, parameter_count],
        "local_design_gib": (
            2 * local_samples * parameter_count * 4 / 2**30
        ),
        "global_design_gib": sample_count * 2 * parameter_count * 4 / 2**30,
        "jacobian_seconds": jacobian_seconds,
        "centering_seconds": center_seconds,
        "norm_seconds": norm_seconds,
        "design_norm_squared": float(np.asarray(global_norm)[0]),
        "total_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
