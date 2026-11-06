"""Configure only maintained execution paths before importing JAX kernels."""

import os


def environment(settings):
    e = settings["execution"]
    flags = {
        "JAX_ENABLE_X64": "true", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "NQS_MATERIALIZE_SECTOR": (
            "0" if settings["sampling"]["proposal"] == "conserved_pair" else "1"
        ),
        "NQS_CONTIGUOUS_ROW_REDUCE": "1", "NQS_DISABLE_ROW_CACHE": "1",
        "NQS_CACHE_BATCHED_WAVEFUNCTION_JIT": "1",
    }
    if e.get("neighbor_execution", "dense") != "dedup_hash":
        flags["NQS_NO_DEDUP"] = "1"
    else:
        flags["NQS_DIAG_PAD"] = "1"
    if e.get("neighbor_execution", "dense") in {
        "compact_valid", "compact_pallas", "compact_staged",
        "compact_staged_bits", "compact_dedup", "compact_gpu_dedup",
        "compact_gpu_dedup56", "compact_gpu_dedup55"
    }:
        total = e["chain_batch"] * e["row_width"]
        batch = e["forward_batch"]
        compact = min(total, ((4 * total + 5 * batch - 1) // (5 * batch)) * batch)
        flags["NQS_COMPACT_VALID"] = "1"
        flags["NQS_COMPACT_VALID_SIZE"] = str(compact)
        if e.get("neighbor_execution") in {
            "compact_pallas", "compact_staged", "compact_staged_bits",
            "compact_dedup", "compact_gpu_dedup", "compact_gpu_dedup56",
            "compact_gpu_dedup55"
        }:
            flags["NQS_COMPACT_PALLAS_REDUCE"] = "1"
        if e.get("neighbor_execution") in {
            "compact_staged", "compact_staged_bits", "compact_dedup",
            "compact_gpu_dedup", "compact_gpu_dedup56",
            "compact_gpu_dedup55"
        }:
            flags["NQS_COMPACT_STAGE_CPU"] = "1"
        if e.get("neighbor_execution") == "compact_staged_bits":
            flags["NQS_COMPACT_BITSTRING_FORWARD"] = "1"
        if e.get("neighbor_execution") == "compact_dedup":
            dedup = min(
                total,
                ((76 * total + 100 * batch - 1) // (100 * batch)) * batch,
            )
            flags["NQS_COMPACT_GLOBAL_DEDUP"] = "1"
            flags["NQS_COMPACT_DEDUP_SIZE"] = str(dedup)
        if e.get("neighbor_execution") in {
            "compact_gpu_dedup", "compact_gpu_dedup56",
            "compact_gpu_dedup55"
        }:
            chunks = (
                55
                if e.get("neighbor_execution") == "compact_gpu_dedup55"
                else (
                    56
                    if e.get("neighbor_execution") == "compact_gpu_dedup56"
                    else e.get("dedup_chunks", 58)
                )
            )
            dedup = min(total, chunks * batch)
            flags["NQS_COMPACT_GPU_DEDUP"] = "1"
            flags["NQS_COMPACT_DEDUP_SIZE"] = str(dedup)
            flags["NQS_COMPACT_DEDUP_CHECK"] = "1"
            flags["NQS_COMPACT_COEFF_FP32"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        flags[name] = str(e["cpu_threads"])
    if e["backend"] == "m4":
        contraction_flag = (
            "NQS_PALLAS_REALBLOCK_QUAD"
            if settings["system"]["nphi"] % 4 == 0
            else "NQS_PALLAS_REALBLOCK_TRIPLET"
        )
        flags.update(
            NQS_PALLAS_LAYERNORM_SILU="1",
            NQS_PALLAS_LN_BLOCK_ROWS=str(e.get("layernorm_block_rows", 4)),
            NQS_KFAC_JIT_FACTORS="1",
            NQS_KFAC_GPU_SOLVE="1",
        )
        flags[contraction_flag] = "1"
    if settings["network"]["generation"] == "local_conv_backflow":
        flags["NQS_LOCAL_CONV_KERNEL_SIZE"] = str(settings["network"]["conv_kernel_size"])
    if e.get("neighbor_execution", "dense") == "packed_scan":
        flags["NQS_PACKED_NEIGHBOR_SCAN"] = "1"
    if e.get("first_layer_execution", "dense") in {
        "reuse", "reuse_fused", "reuse_fused_tail",
        "reuse_fused_tail_bf16", "reuse_fused_tail_gemm2",
        "reuse_fused_tail_gemm23",
    }:
        flags["NQS_NEIGHBOR_FIRST_LAYER_REUSE"] = "1"
    if e.get("first_layer_execution", "dense") in {
        "reuse_fused", "reuse_fused_tail", "reuse_fused_tail_bf16",
        "reuse_fused_tail_gemm2",
        "reuse_fused_tail_gemm23",
    }:
        flags["NQS_NEIGHBOR_FIRST_LAYER_FUSED"] = "1"
    if e.get("first_layer_execution", "dense") in {
        "reuse_fused_tail", "reuse_fused_tail_bf16",
        "reuse_fused_tail_gemm2",
        "reuse_fused_tail_gemm23",
    }:
        tail_flag = (
            "NQS_NEIGHBOR_QUAD_TAIL"
            if settings["system"]["nphi"] % 4 == 0
            else "NQS_NEIGHBOR_TRIPLET_TAIL"
        )
        flags[tail_flag] = "1"
    if e.get("first_layer_execution", "dense") == "reuse_fused_tail_bf16":
        flags["NQS_NEIGHBOR_OUTPUT_BF16"] = "1"
    if e.get("first_layer_execution", "dense") == "reuse_fused_tail_gemm2":
        flags["NQS_NEIGHBOR_SECOND_GEMM_FUSED"] = "1"
    if e.get("first_layer_execution", "dense") == "reuse_fused_tail_gemm23":
        flags["NQS_NEIGHBOR_GEMM23_FUSED"] = "1"
    if e.get("row_workspace", "full") == "fixed":
        flags["NQS_FIXED_ROW_WORKSPACE"] = "1"
    if e.get("row_workspace", "full") == "grouped_hash":
        flags["NQS_GROUPED_HASH_ROWS"] = "1"
    if e.get("scan_unroll", 1) > 1:
        flags["NQS_FORWARD_SCAN_UNROLL"] = str(e["scan_unroll"])
    if e.get("sampler_scan", 0) == 1:
        flags["NQS_CONSERVED_PAIR_SCAN"] = "1"
    return flags


def configure(settings):
    # Historical switches are often tested by string truthiness, so "0" is not
    # a reliable disable. The canonical entrypoint owns its complete NQS scope.
    removed = {key: os.environ.pop(key) for key in list(os.environ) if key.startswith("NQS_")}
    flags = environment(settings)
    os.environ.update(flags)
    return sorted(removed), flags

