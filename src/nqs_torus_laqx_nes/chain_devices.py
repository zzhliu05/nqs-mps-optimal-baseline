"""Opt-in local-energy device parallelism, with one global gradient/KFAC update."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import time


def batch_starts(count, batch, devices):
    if batch <= 0 or devices <= 0 or count <= 0 or count % (batch * devices):
        raise ValueError("Need equally many complete chain batches per device")
    return [list(range(i * batch, count, devices * batch)) for i in range(devices)]


def combine_records(records):
    records = sorted(records, key=lambda item: item[0])
    result = {}
    for key in records[0][1]:
        values = [record[key] for _, record in records]
        if key == "eloc_parts":
            result[key] = [part for parts in values for part in parts]
        elif key == "wf_abs_max":
            result[key] = max(values)
        elif key == "wf_abs_min":
            result[key] = min(values)
        elif key in ("strict_site_bank", "use_site_bank", "site_bank"):
            if any(value is not None and value is not False for value in values):
                raise ValueError("Multi-device site banks are not supported")
            result[key] = values[0]
        else:
            result[key] = sum(values)
    return result


def evaluate_chain_batches(evaluate, arguments):
    device_count = int(os.environ.get("NQS_LOCAL_ENERGY_DEVICES", "1"))
    if device_count == 1:
        return evaluate(**arguments)
    if device_count != 2:
        raise ValueError("Only one or two local-energy devices are supported")
    import jax
    import numba

    devices = jax.local_devices(backend="gpu")
    if len(devices) < device_count:
        raise ValueError(f"Requested {device_count} GPUs, found {len(devices)}")
    if (arguments["states"] is not None
            or not arguments["sparse_rows_from_bitstrings"]
            or not arguments["stream_sampled_rows"]
            or arguments["shared_site_bank"] is not None
            or arguments["config"].network.mps_tensor_generation != "backflow"
            or arguments["config"].training.vmc_neighbor_approximation != "exact"):
        raise ValueError("Multi-device path requires exact streamed bitstring backflow")
    batch = arguments["chain_batch_size"]
    starts = batch_starts(len(arguments["positions_np"]), batch, device_count)
    # Prepare the shared immutable cache before either worker can access it.
    if os.environ.get("NQS_CPU_ROW_BUILDER") == "masked_full":
        from .cpu_full_rows import fixed_coefficients
        fixed_coefficients(arguments["hamiltonian"])
    threads = max(1, numba.get_num_threads() // device_count)
    begin = time.perf_counter()

    def work(index):
        previous_threads = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            with jax.default_device(devices[index]):
                params = jax.device_put(arguments["params"], devices[index])
                records = []
                for start in starts[index]:
                    kwargs = dict(arguments, params=params,
                                  positions_np=arguments["positions_np"][start:start + batch],
                                  neighbor_sampling_seed=arguments["neighbor_sampling_seed"] + start)
                    records.append((start, evaluate(**kwargs)))
                return records
        finally:
            numba.set_num_threads(previous_threads)

    # Local scope ensures workers finish even when a batch fails.
    with ThreadPoolExecutor(max_workers=device_count) as pool:
        records = [record for group in pool.map(work, range(device_count)) for record in group]
    result = combine_records(records)
    print("CHAIN_DEVICE_TIMING " + json.dumps({
        "devices": device_count, "total_samples": len(arguments["positions_np"]),
        "chain_batch": batch, "cpu_threads_per_worker": threads,
        "local_energy_stage_wall_s": time.perf_counter() - begin,
        "local_energy_eval_work_s": result["local_energy_eval_time"],
        "cpu_rows_work_s": result["sparse_row_build_time"],
    }), flush=True)
    return result
