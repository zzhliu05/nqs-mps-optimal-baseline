"""Probability and plotting diagnostics for torus NQS runs."""

from __future__ import annotations

import csv
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from .config import RunConfig
from .network import wavefunction_matrix
from .sampling import MetropolisSamplerJAX


def probability_metrics(
    params,
    states_np,
    h_exact,
    n_chains: int,
    prob_n_burn: int,
    prob_n_between: int,
    prob_n_samples: int,
    seed: int,
    config: RunConfig,
    sampler: MetropolisSamplerJAX | None = None,
    sparse_rows=None,
):
    if states_np.shape[0] <= 10000:
        all_phi = []
        batch_size = config.network.forward_batch_size
        for start in range(0, states_np.shape[0], batch_size):
            stop = min(start + batch_size, states_np.shape[0])
            chunk = np.asarray(
                wavefunction_matrix(
                    params,
                    jnp.asarray(states_np[start:stop], dtype=jnp.float32),
                    config.network,
                )
            )
            all_phi.append(chunk)
        nn_amp = np.concatenate(all_phi, axis=0)[:, 0]
        nn_prob = np.abs(nn_amp) ** 2
        nn_prob = nn_prob / np.sum(nn_prob)
    else:
        nn_prob = None

    if sampler is None:
        sampler = MetropolisSamplerJAX(
            params,
            states_np,
            n_chains,
            config.network,
            seed=seed,
            proposal_mode=config.sampler.proposal_mode,
            sparse_rows=sparse_rows,
            tempering_betas=config.sampler.tempering_betas,
            tempering_swap_interval=config.sampler.tempering_swap_interval,
        )
    else:
        sampler.set_params(params)
    positions, accept = sampler.sample(
        prob_n_burn, prob_n_between, prob_n_samples
    )
    counts = np.bincount(
        positions.reshape(-1), minlength=states_np.shape[0]
    ).astype(np.float64)
    mc_prob = counts / max(1.0, float(np.sum(counts)))

    nm_l1 = (
        float(np.sum(np.abs(nn_prob - mc_prob)))
        if nn_prob is not None
        else float("nan")
    )
    nm_l2 = (
        float(np.sqrt(np.sum((nn_prob - mc_prob) ** 2)))
        if nn_prob is not None
        else float("nan")
    )

    if h_exact is not None and nn_prob is not None:
        _, evecs = eigh(h_exact, subset_by_index=[0, 0])
        exact_prob = np.abs(evecs[:, 0]) ** 2
        exact_prob = exact_prob / np.sum(exact_prob)
        nn_l1 = float(np.sum(np.abs(nn_prob - exact_prob)))
        nn_l2 = float(np.sqrt(np.sum((nn_prob - exact_prob) ** 2)))
        mc_l1 = float(np.sum(np.abs(mc_prob - exact_prob)))
        mc_l2 = float(np.sqrt(np.sum((mc_prob - exact_prob) ** 2)))
    else:
        nn_l1 = nn_l2 = mc_l1 = mc_l2 = float("nan")

    total_samples = n_chains * prob_n_samples
    return (
        nn_l1,
        nn_l2,
        mc_l1,
        mc_l2,
        nm_l1,
        nm_l2,
        float(accept),
        int(total_samples),
    )


def plot_probability_distribution(
    params,
    states,
    h_exact,
    tk: int,
    output: str,
    config: RunConfig,
    *,
    seed: int | None = None,
    title_suffix: str = "",
    sparse_rows=None,
) -> str | None:
    if config.network.k != 1:
        print("skip probability plot: diagnostic is implemented for k=1 only.")
        return None
    if states.shape[0] > 10000:
        print(
            "skip probability plot: sector too large for full wavefunction evaluation."
        )
        return None

    evals, evecs = np.linalg.eigh(h_exact)
    exact_prob = np.abs(evecs[:, 0]) ** 2
    exact_prob = exact_prob / np.sum(exact_prob)
    nn_amp = np.asarray(
        wavefunction_matrix(
            params,
            jnp.asarray(states, dtype=jnp.float32),
            config.network,
        )[:, 0]
    )
    nn_prob = np.abs(nn_amp) ** 2
    nn_prob = nn_prob / np.sum(nn_prob)
    sampler = MetropolisSamplerJAX(
        params,
        states,
        config.sampler.n_chains,
        config.network,
        seed=(
            config.training.seed + 100000 + tk
            if seed is None
            else seed
        ),
        proposal_mode=config.sampler.proposal_mode,
        sparse_rows=sparse_rows,
        tempering_betas=config.sampler.tempering_betas,
        tempering_swap_interval=config.sampler.tempering_swap_interval,
    )
    positions, accept = sampler.sample(
        config.sampler.prob_n_burn,
        config.sampler.prob_n_between,
        config.sampler.prob_n_samples,
    )
    counts = np.bincount(
        positions.reshape(-1), minlength=states.shape[0]
    ).astype(np.float64)
    mc_prob = counts / max(1.0, float(np.sum(counts)))

    x = np.arange(states.shape[0])
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_top.plot(
        x,
        exact_prob,
        color="black",
        linewidth=1.0,
        label=r"exact $|\psi_i|^2$",
    )
    ax_top.plot(
        x,
        nn_prob,
        color="tab:green",
        linewidth=1.0,
        alpha=0.9,
        label=r"JAX NN $|\psi_i|^2$",
    )
    ax_top.scatter(
        x, mc_prob, color="tab:red", s=8, alpha=0.65, label="MC frequency"
    )
    ax_top.set_ylabel("probability")
    ax_top.set_title(
        rf"JAX MC / NN / ED probability distributions, $t_k={tk}$, "
        rf"$k={config.network.k}$, {config.network.architecture}"
        f"{title_suffix}"
    )
    ax_top.legend(frameon=False)
    ax_top.grid(alpha=0.25)

    diff_mc = mc_prob - exact_prob
    diff_nn = nn_prob - exact_prob
    ax_bottom.axhline(0.0, color="black", linewidth=0.8)
    ax_bottom.scatter(
        x, diff_mc, color="tab:blue", s=8, alpha=0.65, label="MC-ED"
    )
    ax_bottom.plot(
        x,
        diff_nn,
        color="tab:green",
        linewidth=1.0,
        alpha=0.9,
        label="NN-ED",
    )
    ax_bottom.set_xlabel("sector basis index")
    ax_bottom.set_ylabel("difference")
    ax_bottom.legend(frameon=False)
    ax_bottom.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    mc_l1 = float(np.sum(np.abs(diff_mc)))
    mc_l2 = float(np.sqrt(np.sum(diff_mc * diff_mc)))
    nn_l1 = float(np.sum(np.abs(diff_nn)))
    nn_l2 = float(np.sqrt(np.sum(diff_nn * diff_nn)))
    print(f"probability plot saved to {output}")
    print(
        "MC distribution diagnostic: "
        f"samples={config.sampler.n_chains * config.sampler.prob_n_samples}, "
        f"accept={accept:.3f}, L1={mc_l1:.6e}, L2={mc_l2:.6e}"
    )
    print(
        f"NN distribution diagnostic: L1={nn_l1:.6e}, L2={nn_l2:.6e}"
    )
    print(f"exact ground energy for probability plot: {evals[0]:.12f}")
    return str(Path(output))


def write_training_history_csv(results, output: str) -> str | None:
    rows = []
    for result in results:
        if result is None:
            continue
        for item in result.training.history:
            row = {"tk": result.tk}
            row.update(item)
            rows.append(row)
    if not rows:
        return None

    preferred = [
        "tk",
        "step",
        "energy",
        "loss",
        "variance",
        "eloc_mean_imag",
        "wf_l2_norm",
        "wf_abs_mean",
        "wf_abs_max",
        "wf_abs_min",
        "wf_logabs_mean",
        "accept",
        "time_sample",
        "time_loss_grad",
        "time_optimizer",
        "time_step",
        "mps_embedding_local_norm",
        "mps_embedding_backflow_norm",
        "mps_embedding_backflow_power_fraction",
        "mps_embedding_backflow_to_local_norm",
        "mps_embedding_backflow_to_local_rms",
        "grad_norm",
        "kfac_cond_max",
        "step_norm",
        "lr",
    ]
    keys = set().union(*(row.keys() for row in rows))
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys.difference(fieldnames)))

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"training history saved to {path}")
    return str(path)


def plot_live_vmc_energy(
    history: list[dict],
    output: str,
    exact_reference_energy: float | None = None,
) -> str:
    """Atomically refresh a compact VMC convergence plot during training."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([item["step"] for item in history], dtype=int)
    energies = np.asarray([item["energy"] for item in history], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(steps, energies, color="tab:blue", linewidth=1.0, label="VMC")
    if exact_reference_energy is not None:
        ax.axhline(
            exact_reference_energy,
            color="black",
            linewidth=1.0,
            linestyle="--",
            label=f"ED reference = {exact_reference_energy:.10f}",
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("energy")
    ax.set_title(f"VMC convergence through step {int(steps[-1])}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(path)
    return str(path)


def plot_energy_curves(results, config: RunConfig) -> str | None:
    valid = [r for r in results if r is not None and r.training.history]
    if not valid:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for result in valid:
        steps = np.array(
            [item["step"] for item in result.training.history], dtype=int
        )
        energies = np.array(
            [item["energy"] for item in result.training.history], dtype=float
        )
        ax.plot(
            steps,
            energies,
            linewidth=1.2,
            alpha=0.85,
            label=f"tk={result.tk}",
        )
        if result.exact_eigenvalues is not None:
            level_count = min(
                config.network.k, len(result.exact_eigenvalues)
            )
            ax.axhline(
                float(np.sum(result.exact_eigenvalues[:level_count])),
                color="black",
                linewidth=0.7,
                alpha=0.15,
            )
    ax.set_xlabel("training step")
    ax.set_ylabel("lowest Ritz energy")
    ax.set_title(
        rf"JAX energy convergence, $N_\phi={config.system.nphi}$, "
        rf"$N_e={config.system.ne}$, $k={config.network.k}$, "
        rf"{config.network.architecture}"
    )
    if len(valid) <= 10:
        ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output = config.output.energy_curve_output or (
        f"NQS_torus_debug_jax_energy_Nphi{config.system.nphi}"
        f"_Ne{config.system.ne}_tk{config.system.tk}"
        f"_{config.network.architecture}.png"
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"energy convergence plot saved to {output}")
    return str(Path(output))


def plot_sector_spectrum(results, config: RunConfig) -> str | None:
    valid = [
        result
        for result in results
        if result is not None and len(result.nqs_eigenvalues) > 0
    ]
    if not valid:
        return None

    energies = np.concatenate(
        [np.asarray(result.nqs_eigenvalues, dtype=float) for result in valid]
    )
    exact_valid = [
        result for result in valid if result.exact_eigenvalues is not None
    ]
    if exact_valid:
        exact_energies = np.concatenate(
            [
                np.asarray(
                    result.exact_eigenvalues[
                        : len(result.nqs_eigenvalues)
                    ],
                    dtype=float,
                )
                for result in exact_valid
            ]
        )
        energies = np.concatenate([energies, exact_energies])
    emin = float(np.min(energies))

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for result in valid:
        y_nqs = np.asarray(result.nqs_eigenvalues, dtype=float) - emin
        ax.scatter(
            np.full_like(y_nqs, result.tk, dtype=float),
            y_nqs,
            color="tab:red",
            s=28,
            label="JAX NQS" if result.tk == valid[0].tk else None,
        )
        if result.exact_eigenvalues is not None:
            y_exact = (
                np.asarray(
                    result.exact_eigenvalues[: len(result.nqs_eigenvalues)],
                    dtype=float,
                )
                - emin
            )
            ax.scatter(
                np.full_like(y_exact, result.tk, dtype=float),
                y_exact,
                color="black",
                marker="x",
                s=32,
                label="exact" if result.tk == valid[0].tk else None,
            )

    ax.set_xlabel(r"$t_k$")
    ax.set_ylabel(r"$E-E_{\min}$")
    ax.set_title(
        rf"JAX NQS torus spectrum, $N_\phi={config.system.nphi}$, "
        rf"$N_e={config.system.ne}$, $k={config.network.k}$, "
        rf"{config.network.architecture}"
    )
    ax.set_xticks(
        np.arange(
            0,
            config.system.nphi,
            max(1, config.system.nphi // 6),
        )
    )
    ax.grid(alpha=0.25)
    if exact_valid:
        ax.legend(frameon=False)
    fig.tight_layout()
    output = config.output.plot_output or (
        f"NQS_torus_debug_jax_spec_Nphi{config.system.nphi}"
        f"_Ne{config.system.ne}_k{config.network.k}"
        f"_{config.network.architecture}"
        f"_ratio{config.system.ratio_rect}.png"
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"spectrum plot saved to {output}")
    return str(Path(output))
