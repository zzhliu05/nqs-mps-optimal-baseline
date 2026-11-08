"""One explicit input schema, without contradictory CLI/environment aliases."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass
class System:
    nphi: int = 36
    p: int = 1
    q: int = 3
    tk: int = 5
    landau_level: int = 0
    ratio_rect: float = 1.0
    coulomb_cut: int | None = None


@dataclass
class Network:
    generation: str = "backflow"
    hidden_sizes: list[int] = field(default_factory=lambda: [397, 397])
    bond_dim: int = 10
    embed_dim: int = 128
    init_std: float = 0.01
    conv_kernel_size: int = 3


@dataclass
class Sampling:
    proposal: str = "conserved_pair"
    chains: int = 4096
    estimator_chains: int | None = None
    burn: int = 40
    between: int = 1


@dataclass
class Training:
    steps: int = 2000
    seed: int = 362039
    optimizer: str = "kfac"
    learning_rate: float = 0.005
    learning_rate_schedule: str = "constant"
    decay_rate: int = 1000
    damping: float = 0.001
    max_step_norm: float = 10.0
    max_step_norm_final: float | None = None
    max_step_norm_ramp_start: int = 0
    max_step_norm_ramp_end: int = 0
    wavefunction_trust_kl: float | None = None
    gradient_clip_norm: float | None = None
    sr_samples: int | None = None
    sr_jacobian_batch_size: int = 32
    kfac_diagnostics: bool = False
    kfac_diagnostic_samples: int = 512
    kfac_trace_probes: int = 4


@dataclass
class Execution:
    backend: str = "m4"
    neighbor_execution: str = "dense"
    dedup_chunks: int = 58
    first_layer_execution: str = "dense"
    row_workspace: str = "full"
    layernorm_block_rows: int = 4
    scan_unroll: int = 1
    sampler_scan: int = 0
    chain_batch: int = 4096
    forward_batch: int = 32768
    row_width: int = 608
    cpu_threads: int = 18
    max_basis_states: int = 50_000_000


SECTIONS = {"system": System, "network": Network, "sampling": Sampling,
            "training": Training, "execution": Execution}


def load_settings(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(data) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}")
    settings = {}
    for name, cls in SECTIONS.items():
        values = data.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be an object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} inputs: {sorted(unknown)}")
        settings[name] = asdict(cls(**values))
    validate(settings)
    return settings


def validate(settings):
    s, n, p, t, e = (settings[k] for k in SECTIONS)
    integer_keys = {"system": ("nphi", "p", "q", "tk", "landau_level"),
                    "network": ("bond_dim", "embed_dim", "conv_kernel_size"),
                    "sampling": ("chains", "burn", "between"),
                    "training": ("steps", "seed", "sr_jacobian_batch_size",
                                 "decay_rate",
                                 "max_step_norm_ramp_start",
                                 "max_step_norm_ramp_end",
                                 "kfac_diagnostic_samples", "kfac_trace_probes"),
                    "execution": tuple(k for k in e if k not in {
                        "backend", "neighbor_execution", "first_layer_execution",
                        "row_workspace", "sampler_scan",
                    })}
    for section, names in integer_keys.items():
        for key in names:
            if type(settings[section][key]) is not int:
                raise ValueError(f"{section}.{key} must be an integer")
    if not 1 <= s["nphi"] <= 63:
        raise ValueError("Full packed-state training currently supports Nphi<=63, not the dense forward proxy")
    if s["q"] <= 0 or not 0 < s["p"] <= s["q"] or s["nphi"] % s["q"]:
        raise ValueError("Require integral filling: Nphi divisible by q, 0<p<=q")
    if not 0 <= s["tk"] < s["nphi"] or s["landau_level"] < 0:
        raise ValueError("Invalid momentum sector or Landau level")
    if s["coulomb_cut"] is not None and (type(s["coulomb_cut"]) is not int or s["coulomb_cut"] <= 0):
        raise ValueError("coulomb_cut must be null or a positive integer")
    for section, keys in (("system", ["ratio_rect"]), ("network", ["init_std"]),
                          ("training", ["learning_rate", "damping", "max_step_norm"])):
        for key in keys:
            value = settings[section][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
    if t["gradient_clip_norm"] is not None:
        value = t["gradient_clip_norm"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("training.gradient_clip_norm must be null or finite and positive")
    if t["wavefunction_trust_kl"] is not None:
        value = t["wavefunction_trust_kl"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("training.wavefunction_trust_kl must be null or finite and positive")
    if t["max_step_norm_final"] is not None:
        value = t["max_step_norm_final"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("training.max_step_norm_final must be null or finite and positive")
        if not 0 <= t["max_step_norm_ramp_start"] < t["max_step_norm_ramp_end"] <= t["steps"]:
            raise ValueError("Require 0 <= step-norm ramp start < end <= training steps")
    if n["generation"] not in {"backflow", "sitewise", "tensor_bank", "direct", "local_conv_backflow", "global_backflow"}:
        raise ValueError("Unsupported MPS generation in maintained interface")
    if not isinstance(n["hidden_sizes"], list) or not n["hidden_sizes"] or any(type(w) is not int or w <= 0 for w in n["hidden_sizes"]):
        raise ValueError("hidden_sizes must be a nonempty list of positive integers")
    if n["bond_dim"] <= 0 or n["embed_dim"] <= 0 or n["conv_kernel_size"] <= 0 or n["conv_kernel_size"] % 2 != 1:
        raise ValueError("Invalid bond/embed size or convolution kernel (must be odd)")
    if t["optimizer"] not in {"kfac", "sr", "adam", "sign"} or t["steps"] < 1:
        raise ValueError("Require optimizer kfac/sr/adam/sign and positive steps")
    if t["learning_rate_schedule"] not in {"constant", "inverse_time"}:
        raise ValueError(
            "training.learning_rate_schedule must be constant or inverse_time"
        )
    if t["decay_rate"] <= 0:
        raise ValueError("training.decay_rate must be positive")
    if t["sr_samples"] is not None and (
        type(t["sr_samples"]) is not int or t["sr_samples"] <= 0
    ):
        raise ValueError("training.sr_samples must be null or a positive integer")
    if type(t["kfac_diagnostics"]) is not bool:
        raise ValueError("training.kfac_diagnostics must be boolean")
    if n["generation"] == "direct" and t["optimizer"] == "kfac":
        raise ValueError("Direct MPS has no MLP KFAC factors; select adam")
    if p["chains"] <= 0 or p["burn"] < 0 or p["between"] < 0:
        raise ValueError("Invalid sampler counts")
    if p["proposal"] not in {"conserved_pair", "sector_uniform"}:
        raise ValueError("sampling.proposal must be conserved_pair or sector_uniform")
    estimator = p["estimator_chains"] if p["estimator_chains"] is not None else p["chains"]
    if type(estimator) is not int or not 0 < estimator <= p["chains"]:
        raise ValueError("estimator_chains must be positive and no larger than chains")
    if any(e[key] <= 0 for key in integer_keys["execution"]):
        raise ValueError("Execution sizes and thread count must be positive")
    if estimator % e["chain_batch"]:
        raise ValueError("estimator_chains must be divisible by chain_batch")
    if e["backend"] not in {"m4", "reference"}:
        raise ValueError("backend must be m4 or reference")
    if e["neighbor_execution"] not in {
        "dense", "packed_scan", "dedup_hash", "compact_valid",
        "compact_pallas", "compact_staged", "compact_staged_bits",
        "compact_dedup", "compact_gpu_dedup", "compact_gpu_dedup56",
        "compact_gpu_dedup55"
    }:
        raise ValueError(
            "neighbor_execution must be dense, packed_scan, dedup_hash or "
            "compact_valid, compact_pallas, compact_staged, "
            "compact_staged_bits, compact_dedup, compact_gpu_dedup or "
            "compact_gpu_dedup56 or compact_gpu_dedup55"
        )
    if e["neighbor_execution"] == "packed_scan" and n["generation"] != "backflow":
        raise ValueError("packed_scan currently requires plain backflow")
    if e["first_layer_execution"] not in {
        "dense", "reuse", "reuse_fused", "reuse_fused_tail",
        "reuse_fused_tail_bf16", "reuse_fused_tail_gemm2",
        "reuse_fused_tail_gemm23",
    }:
        raise ValueError(
            "first_layer_execution must be dense, reuse, reuse_fused or "
            "reuse_fused_tail, reuse_fused_tail_bf16 or "
            "reuse_fused_tail_gemm2"
        )
    if e["first_layer_execution"] != "dense" and n["generation"] != "backflow":
        raise ValueError("reuse first layer requires plain backflow")
    if e["row_workspace"] not in {"full", "fixed", "grouped_hash"}:
        raise ValueError("row_workspace must be full, fixed or grouped_hash")
    if e["layernorm_block_rows"] not in {1, 2, 4, 8, 16, 32}:
        raise ValueError("layernorm_block_rows must be 1, 2, 4, 8, 16 or 32")
    if type(e["sampler_scan"]) is not int or e["sampler_scan"] not in {0, 1}:
        raise ValueError("sampler_scan must be 0 or 1")
    if e["backend"] == "m4" and (
        (s["nphi"] % 4 and s["nphi"] % 3)
        or n["bond_dim"] > 32
        or n["generation"] != "backflow"
    ):
        raise ValueError(
            "Maintained fused profile requires plain backflow, Nphi divisible "
            "by 4 or 3, and D<=32"
        )


def core_config(settings, output):
    from nqs_torus_laqx_nes.config import (RunConfig, SystemConfig, NetworkConfig,
                                         SamplerConfig, OptimizerConfig, TrainingConfig, OutputConfig)
    s, n, p, t, e = (settings[k] for k in SECTIONS)
    output = Path(output)
    optimizer_name = {
        "kfac": "natural_gradient",
        "sr": "sr",
        "adam": "adam",
        "sign": "signed_gradient",
    }[t["optimizer"]]
    result = RunConfig(
        system=SystemConfig(**s),
        network=NetworkConfig(k=1, architecture="mps", num_sites=s["nphi"],
            hidden_sizes=tuple(n["hidden_sizes"]), activation="silu", layer_norm=True,
            init_std=n["init_std"], forward_batch_size=e["forward_batch"],
            mps_tensor_generation=n["generation"], mps_contraction="balanced",
            mps_bond_dim=n["bond_dim"], mps_backflow_embed_dim=n["embed_dim"]),
        sampler=SamplerConfig(proposal_mode=p["proposal"], n_chains=p["chains"],
            n_burn=p["burn"], n_between=p["between"], n_samples=1, prob_n_samples=0),
        optimizer=OptimizerConfig(name=optimizer_name,
            lr=t["learning_rate"],
            constant_lr=t["learning_rate_schedule"] == "constant",
            decay_rate=t["decay_rate"], sr_diag_shift=t["damping"],
            sr_samples=t["sr_samples"], sr_jacobian_batch_size=t["sr_jacobian_batch_size"],
            kfac_diagnostics=t["kfac_diagnostics"],
            kfac_diagnostic_samples=t["kfac_diagnostic_samples"],
            kfac_trace_probes=t["kfac_trace_probes"],
            max_sr_step_norm=t["max_step_norm"],
            max_sr_step_norm_final=t["max_step_norm_final"],
            max_sr_step_norm_ramp_start=t["max_step_norm_ramp_start"],
            max_sr_step_norm_ramp_end=t["max_step_norm_ramp_end"],
            wavefunction_trust_kl=t["wavefunction_trust_kl"],
            gradient_clip_norm=t["gradient_clip_norm"]),
        training=TrainingConfig(optimize_method="vmc", n_steps=t["steps"], min_steps=t["steps"],
            seed=t["seed"], vmc_row_batch_size=e["chain_batch"], vmc_chain_batch_size=e["chain_batch"],
            vmc_estimator_chains=p["estimator_chains"], vmc_sparse_rows_from_bitstrings=True,
            vmc_deduplicate_neighbors=e["neighbor_execution"] == "dedup_hash",
            vmc_disable_neighbor_padding=e["neighbor_execution"] == "dedup_hash",
            vmc_fixed_neighbor_row_nnz=e["row_width"],
            vmc_neighbor_padding_bucket_size=(
                e["forward_batch"]
                if e["neighbor_execution"] == "dedup_hash"
                else None
            )),
        output=OutputConfig(skip_final_ritz=True, energy_curve_output=str(output / "energy.png"),
                            history_output=str(output / "history.csv"),
                            params_output=str(output / "params.pkl")),
    )
    result.validate()
    return result

