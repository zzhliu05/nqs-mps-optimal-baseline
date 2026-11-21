"""Run a validated JSON MPS experiment with source and timing provenance."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

from .settings import load_settings, validate, core_config
from .runtime import configure, environment


def source_hashes():
    src = Path(__file__).resolve().parents[1]
    return {file.relative_to(src).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
            for package in ("nqs_mps", "nqs_torus_laqx_nes")
            for file in sorted((src / package).rglob("*.py"))}


def load_checkpoint(path, sector, config):
    import pickle
    import jax
    import jax.numpy as jnp
    from nqs_torus_laqx_nes.network import init_params
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    saved = payload.get("config", {})
    for key in ("nphi", "p", "q", "landau_level", "ratio_rect", "coulomb_cut"):
        if saved.get("system", {}).get(key) != getattr(config.system, key):
            raise ValueError(f"Checkpoint system mismatch: {key}")
    if sector != config.system.tk:
        raise ValueError("Resume sector must match the requested sector")
    expected = init_params(config.training.seed, config.system.nphi, config.network)
    params = jax.tree_util.tree_map(jnp.asarray, payload["sectors"][sector])
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(expected):
        raise ValueError("Checkpoint parameter structure mismatch")
    if any(a.shape != b.shape for a, b in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(expected))):
        raise ValueError("Checkpoint parameter shapes mismatch")
    for key in ("mps_tensor_generation", "mps_bond_dim", "hidden_sizes", "activation", "layer_norm"):
        before, after = saved.get("network", {}).get(key), getattr(config.network, key)
        if key == "hidden_sizes":
            before, after = tuple(before or ()), tuple(after)
        if before != after:
            raise ValueError(f"Checkpoint network mismatch: {key}")
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print resolved config without training")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--tk", type=int)
    parser.add_argument("--backend", choices=("m4", "reference"))
    parser.add_argument("--resume", type=Path, help="Trusted local params.pkl only; pickle can execute code")
    parser.add_argument("--resume-sector", type=int)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    for section, key, value in (("training", "steps", args.steps), ("system", "tk", args.tk), ("execution", "backend", args.backend)):
        if value is not None:
            settings[section][key] = value
    validate(settings)
    if args.resume_sector is not None and args.resume is None:
        parser.error("--resume-sector requires --resume")
    cfg = core_config(settings, args.output.resolve())
    resolved = {"settings": settings, "core_config": asdict(cfg), "environment": environment(settings),
                "timing_kind": "full_vmc_training_step", "source_sha256": source_hashes(),
                "resume": str(args.resume.resolve()) if args.resume else None,
                "resume_semantics": "parameters only; sampler and optimizer state restart; steps count continuation updates"}
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; use a new directory to preserve existing runs")
    args.output.mkdir(parents=True, exist_ok=True)
    removed, flags = configure(settings)
    resolved["cleared_inherited_nqs_keys"] = removed
    (args.output / "resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    start = time.perf_counter()
    status = {"state": "failed"}
    try:
        import jax
        import numpy as np
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        if settings["execution"]["backend"] == "m4" and not any(d.platform == "gpu" for d in devices):
            raise RuntimeError("M4 requires a CUDA GPU/Pallas runtime; no silent reference fallback")
        from nqs_torus_laqx_nes.physics import _sector_completion_counts
        count = int(_sector_completion_counts(cfg.system.nphi, cfg.system.ne)[0, cfg.system.ne, cfg.system.tk])
        if count <= 0:
            raise ValueError(f"Sector has {count} states and is empty")
        materializes_sector = settings["sampling"]["proposal"] == "sector_uniform"
        if (
            materializes_sector
            and count > settings["execution"]["max_basis_states"]
        ):
            raise ValueError(
                f"Sector has {count} states; exceeds configured safe "
                "materialization limit"
            )
        from nqs_torus_laqx_nes import workflow
        original_init = workflow.init_params
        if args.resume:
            params = load_checkpoint(args.resume, args.resume_sector if args.resume_sector is not None else cfg.system.tk, cfg)
            workflow.init_params = lambda seed, num_sites, net_cfg: params
        metadata = {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
                    "devices": [str(d) for d in devices], "sector_size": count,
                    "sampling_semantics": (
                        "uniform proposal over materialized fixed-particle "
                        "fixed-momentum sector"
                        if materializes_sector
                        else "sector-table-free symmetric conserved-pair "
                        "proposal on packed uint64 bitstring chains"
                    ),
                    "sector_materialized": materializes_sector,
                    "package_path": str(Path(workflow.__file__).resolve())}
        (args.output / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            result = workflow.run(cfg)
            jax.block_until_ready([sector.params for sector in result.sector_results])
        finally:
            workflow.init_params = original_init
        status = {"state": "completed", "steps_per_sector": [len(s.training.history) for s in result.sector_results]}
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["process_wall_seconds"] = time.perf_counter() - start
        (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

