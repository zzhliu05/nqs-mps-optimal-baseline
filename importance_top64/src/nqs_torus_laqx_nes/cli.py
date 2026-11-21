"""Command-line interface for the modular torus NQS package."""

from __future__ import annotations

import argparse

from .config import (
    ContinuousConfig,
    NetworkConfig,
    OptimizerConfig,
    OutputConfig,
    RunConfig,
    SamplerConfig,
    SystemConfig,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "JAX NQS projected diagonalization for "
            "ED_torus_debug Coulomb Hamiltonian"
        )
    )
    parser.add_argument("--Nphi", type=int, default=24)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=3)
    parser.add_argument("--tk", type=int, default=0)
    parser.add_argument("--all_sectors", action="store_true")
    parser.add_argument("--ratio_rect", type=float, default=1.0)
    parser.add_argument(
        "--landau_level",
        type=int,
        default=0,
        help=(
            "Orbital Landau-level index used in the projected Coulomb "
            "form factor. Use 0 for the LLL and 1 for the first excited "
            "Landau level."
        ),
    )
    parser.add_argument("--coulomb_cut", type=int, default=None)
    parser.add_argument("--disable_numba_coulomb", action="store_true")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument(
        "--network_architecture",
        choices=["ace_nes_1d", "mlp", "mps", "vit"],
        default="ace_nes_1d",
    )
    parser.add_argument(
        "--hidden_sizes", type=int, nargs="+", default=[64, 64]
    )
    parser.add_argument(
        "--activation",
        choices=["silu", "tanh", "poly", "relu", "gelu", "leaky_relu"],
        default="silu",
    )
    parser.add_argument("--no_layer_norm", action="store_true")
    parser.add_argument("--logphi_real_clip", type=float, default=30.0)
    parser.add_argument("--init_std", type=float, default=0.01)
    parser.add_argument("--nes_hidden", type=int, default=58)
    parser.add_argument("--nes_layers", type=int, default=4)
    parser.add_argument("--nes_mlp_hidden", type=int, default=58)
    parser.add_argument("--nes_mlp_layers", type=int, default=1)
    parser.add_argument("--nes_cutoff", type=int, default=3)
    parser.add_argument("--vit_patch_size", type=int, default=3)
    parser.add_argument("--vit_embed_dim", type=int, default=256)
    parser.add_argument("--vit_num_heads", type=int, default=8)
    parser.add_argument("--vit_num_layers", type=int, default=4)
    parser.add_argument("--vit_mlp_ratio", type=int, default=4)
    parser.add_argument("--vit_fourier_modes", type=int, default=4)
    parser.add_argument(
        "--mps_bond_dim",
        type=int,
        default=16,
        help="Bond dimension of the periodic trace neural MPS ansatz.",
    )
    parser.add_argument(
        "--mps_tensor_generation",
        choices=[
            "direct",
            "sitewise",
            "tensor_bank",
            "backflow",
            "local_backflow",
            "local_conv_backflow",
            "global_backflow",
            "site_head_backflow",
        ],
        default="sitewise",
        help=(
            "Optimize the 2*Nphi MPS matrices directly, generate local "
            "tensors per configuration, evaluate a shared 2*Nphi tensor "
            "bank, condition every site tensor on the full occupation "
            "configuration per site (backflow), or generate all site "
            "tensors in one network call (global_backflow), or use one "
            "finite-window shared MLP with exact local neighbor updates "
            "(local_backflow), or use one "
            "global trunk followed by FiLM-conditioned lightweight site "
            "heads (site_head_backflow)."
        ),
    )
    parser.add_argument(
        "--mps_contraction",
        choices=["sequential", "balanced", "balanced_fp64", "fused4"],
        default="sequential",
        help=(
            "Contract periodic MPS matrices sequentially, as a balanced "
            "tree, a complex128 balanced tree (requires JAX x64), or "
            "four-site fused blocks."
        ),
    )
    parser.add_argument(
        "--mps_first_layer_backend",
        choices=["jax", "triton"],
        default="jax",
        help=(
            "Backend for the Site Backflow first layer. Triton is "
            "experimental and falls back to a structured JAX VJP in backward."
        ),
    )
    parser.add_argument(
        "--mps_backflow_embed_dim",
        type=int,
        default=128,
        help=(
            "Embedding width for backflow MPS. In site_head_backflow this "
            "sets both the global latent and lightweight head width."
        ),
    )
    parser.add_argument(
        "--mps_residual_blocks",
        type=int,
        default=0,
        help="Number of parameter-free residual skips in the MPS generator.",
    )
    parser.add_argument(
        "--mps_attention_blocks",
        type=int,
        default=0,
        help=(
            "Number of learned self-attention blocks over site tokens in "
            "sitewise/backflow MPS generators."
        ),
    )
    parser.add_argument(
        "--mps_attention_heads",
        type=int,
        default=4,
        help="Number of heads in each MPS self-attention block.",
    )
    parser.add_argument(
        "--mps_jastrow",
        action="store_true",
        help=(
            "Multiply the MPS by a zero-initialized translationally invariant "
            "density Jastrow factor with one trainable coefficient per "
            "inequivalent periodic distance."
        ),
    )
    parser.add_argument(
        "--mps_jastrow_log_clip",
        type=float,
        default=20.0,
        help="Absolute clip applied to the logarithm of the Jastrow factor.",
    )
    parser.add_argument(
        "--mps_jastrow_hidden_sizes",
        nargs="+",
        type=int,
        default=[64, 64],
        help="Hidden widths of the unrestricted neural Jastrow MLP.",
    )
    parser.add_argument(
        "--sampler_mode",
        choices=[
            "sector_uniform",
            "momentum_scattering",
            "laqx_hamiltonian",
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        ],
        default="sector_uniform",
    )
    parser.add_argument("--n_chains", type=int, default=1024)
    parser.add_argument("--sampler_hpsi_candidates", type=int, default=4)
    parser.add_argument("--n_burn", type=int, default=20)
    parser.add_argument(
        "--n_burn_warmup",
        type=int,
        default=None,
        help="Large initial burn for the first VMC step, then fall back to n_burn",
    )
    parser.add_argument("--n_between", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument(
        "--parallel_tempering",
        action="store_true",
        help="Use replica exchange and retain only the beta=1 chains for VMC.",
    )
    parser.add_argument("--tempering_replicas", type=int, default=4)
    parser.add_argument("--tempering_beta_min", type=float, default=0.25)
    parser.add_argument("--tempering_swap_interval", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--min_steps", type=int, default=20)
    parser.add_argument("--converge_tol", type=float, default=0.0)
    parser.add_argument("--converge_window", type=int, default=10)
    parser.add_argument("--loss_levels", type=int, default=None)
    parser.add_argument(
        "--optimize_method",
        choices=["projected", "vmc", "continuous_vmc"],
        default="vmc",
    )
    parser.add_argument(
        "--continuous_basis",
        choices=["fourier", "compact_overlap"],
        default="fourier",
    )
    parser.add_argument("--continuous_harmonic", type=int, default=1)
    parser.add_argument(
        "--continuous_overlap_epsilon", type=float, default=0.1
    )
    parser.add_argument(
        "--continuous_proposal_width", type=float, default=0.05
    )
    parser.add_argument(
        "--continuous_target_acceptance", type=float, default=0.5
    )
    parser.add_argument(
        "--continuous_adapt_rate", type=float, default=0.1
    )
    parser.add_argument(
        "--continuous_min_width", type=float, default=1e-4
    )
    parser.add_argument(
        "--continuous_max_width", type=float, default=0.5
    )
    parser.add_argument(
        "--continuous_node_epsilon", type=float, default=1e-14
    )
    parser.add_argument("--overlap_shift", type=float, default=1e-8)
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="SR update step size."
    )
    parser.add_argument("--sr_diag_shift", type=float, default=1e-3)
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--cg_maxiter", type=int, default=2000)
    parser.add_argument("--max_sr_step_norm", type=float, default=10.0)
    parser.add_argument("--projection_rcond", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--vmc_row_batch_size", type=int, default=None)
    parser.add_argument(
        "--vmc_neighbor_approximation",
        choices=(
            "exact", "input_log_taylor", "input_log_taylor2",
            "input_log_taylor_cv", "exact_neighbor_is",
        ),
        default="exact",
        help=(
            "Experimental input-log-gradient neighbor estimator: direct Taylor "
            "is biased; Taylor-CV uses sampled exact corrections."
        ),
    )
    parser.add_argument("--vmc_input_gradient_batch_size", type=int, default=32)
    parser.add_argument("--vmc_input_hessian_batch_size", type=int, default=4)
    parser.add_argument("--vmc_exact_neighbors_per_row", type=int, default=64)
    parser.add_argument("--vmc_exact_top_neighbors", type=int, default=8)
    parser.add_argument(
        "--vmc_neighbor_sampling_uniform_mix", type=float, default=0.1
    )
    parser.add_argument(
        "--vmc_chain_batch_size",
        type=int,
        default=None,
        help=(
            "Accumulate VMC gradients and KFAC factors over fixed-size "
            "chain microbatches before one optimizer update."
        ),
    )
    parser.add_argument(
        "--vmc_estimator_chains",
        type=int,
        default=None,
        help=(
            "Maintain and burn the full n_chains sampler pool, but use only "
            "this many rotating chains per step for local energy, gradients, "
            "and the optimizer metric. Defaults to all chains."
        ),
    )
    parser.add_argument(
        "--vmc_fixed_neighbor_row_nnz",
        type=int,
        default=None,
        help=(
            "Use an explicit fixed Hamiltonian-neighbor row width for "
            "streamed VMC padding."
        ),
    )
    parser.add_argument(
        "--vmc_neighbor_padding_bucket_size",
        type=int,
        default=None,
        help=(
            "Round the deduplicated neighbor count up to this many slots, "
            "bounded by the fixed maximum padding size."
        ),
    )
    parser.add_argument("--vmc_prebuild_sparse_rows", action="store_true")
    parser.add_argument(
        "--vmc_sparse_rows_from_bitstrings",
        action="store_true",
        help=(
            "Build streamed VMC Hamiltonian rows directly from sampled "
            "bitstrings, avoiding neighbor-to-sector-index lookup."
        ),
    )
    parser.add_argument(
        "--vmc_deduplicate_neighbors",
        action="store_true",
        help="Deduplicate sampled Hamiltonian neighbors before NQS forward.",
    )
    parser.add_argument(
        "--vmc_disable_neighbor_dedup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vmc_disable_neighbor_padding",
        action="store_true",
        help=(
            "Keep neighbor deduplication but forward the dynamic unique-state "
            "set without fixed padding."
        ),
    )
    parser.add_argument("--vmc_timing_analysis", action="store_true")
    parser.add_argument("--vmc_timing_interval", type=int, default=10)
    parser.add_argument("--build_exact", action="store_true")
    parser.add_argument(
        "--skip_final_ritz",
        action="store_true",
        help="Skip the post-training full-sector Ritz evaluation.",
    )
    parser.add_argument("--plot_output", type=str, default=None)
    parser.add_argument("--energy_curve_output", type=str, default=None)
    parser.add_argument("--prob_plot_output", type=str, default=None)
    parser.add_argument("--history_output", type=str, default=None)
    parser.add_argument(
        "--state_archive_output",
        type=str,
        default=None,
        help=(
            "Save normalized NQS Ritz vectors from every completed sector "
            "for spectral-weight post-processing."
        ),
    )
    parser.add_argument(
        "--params_output",
        type=str,
        default=None,
        help=(
            "Save final network parameters and the complete run config as "
            "a host-side pickle checkpoint."
        ),
    )
    parser.add_argument("--live_plot_output", type=str, default=None)
    parser.add_argument("--live_plot_interval", type=int, default=0)
    parser.add_argument("--prob_snapshot_dir", type=str, default=None)
    parser.add_argument(
        "--prob_snapshot_steps",
        type=int,
        nargs="*",
        default=[],
        help="Training steps at which to save NN/MC/ED probability plots.",
    )
    parser.add_argument("--prob_n_burn", type=int, default=2048)
    parser.add_argument("--prob_n_between", type=int, default=1)
    parser.add_argument("--prob_n_samples", type=int, default=1)
    parser.add_argument(
        "--constant_lr",
        action="store_true",
        help="Use a constant learning rate with no decay schedule.",
    )
    parser.add_argument("--decay_rate", type=int, default=1000)
    parser.add_argument(
        "--multi_step_lr",
        action="store_true",
        help=(
            "Split n_steps into four stages with lr multipliers "
            "1, 1/2, 1/4, 1/8 instead of smooth decay."
        ),
    )
    parser.add_argument(
        "--cyclic_lr",
        action="store_true",
        help=(
            "Use a cyclic learning-rate schedule with lr as the peak "
            "value and cyclic_lr_min as the trough."
        ),
    )
    parser.add_argument(
        "--cyclic_lr_min",
        type=float,
        default=None,
        help=(
            "Minimum learning rate for cyclic_lr. Defaults to lr/8 when "
            "cyclic_lr is enabled and this flag is omitted."
        ),
    )
    parser.add_argument(
        "--cyclic_lr_period",
        type=int,
        default=1000,
        help="Full optimizer-step period of the cyclic learning-rate schedule.",
    )
    parser.add_argument(
        "--cyclic_lr_mode",
        choices=["triangular", "triangular2"],
        default="triangular",
        help="Cyclic learning-rate waveform.",
    )
    parser.add_argument(
        "--optimizer",
        choices=[
            "natural_gradient",
            "signed_gradient",
            "muon",
            "adam",
            "adamw",
        ],
        default="signed_gradient",
    )
    parser.add_argument("--adam_b1", type=float, default=0.9)
    parser.add_argument("--adam_b2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_beta", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--muon_adaptive", action="store_true")
    parser.add_argument("--report_fisher_condition", action="store_true")
    parser.add_argument("--forward_batch_size", type=int, default=8192)
    parser.add_argument("--forward_parallel_chunks", type=int, default=1)
    parser.add_argument(
        "--perturb_steps",
        type=int,
        default=0,
        help="Add random perturbation to params for first N steps of each training loop",
    )
    parser.add_argument("--perturb_scale", type=float, default=1e-4)
    return parser


def config_from_namespace(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        system=SystemConfig(
            nphi=args.Nphi,
            p=args.p,
            q=args.q,
            tk=args.tk,
            all_sectors=args.all_sectors,
            ratio_rect=args.ratio_rect,
            landau_level=args.landau_level,
            coulomb_cut=args.coulomb_cut,
            disable_numba_coulomb=args.disable_numba_coulomb,
        ),
        network=NetworkConfig(
            num_sites=args.Nphi,
            k=args.k,
            architecture=args.network_architecture,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
            layer_norm=not args.no_layer_norm,
            logphi_real_clip=args.logphi_real_clip,
            init_std=args.init_std,
            forward_batch_size=args.forward_batch_size,
            forward_parallel_chunks=args.forward_parallel_chunks,
            nes_hidden=args.nes_hidden,
            nes_layers=args.nes_layers,
            nes_mlp_hidden=args.nes_mlp_hidden,
            nes_mlp_layers=args.nes_mlp_layers,
            nes_cutoff=args.nes_cutoff,
            vit_patch_size=args.vit_patch_size,
            vit_embed_dim=args.vit_embed_dim,
            vit_num_heads=args.vit_num_heads,
            vit_num_layers=args.vit_num_layers,
            vit_mlp_ratio=args.vit_mlp_ratio,
            vit_fourier_modes=args.vit_fourier_modes,
            mps_bond_dim=args.mps_bond_dim,
            mps_tensor_generation=args.mps_tensor_generation,
            mps_contraction=args.mps_contraction,
            mps_first_layer_backend=args.mps_first_layer_backend,
            mps_backflow_embed_dim=args.mps_backflow_embed_dim,
            mps_residual_blocks=args.mps_residual_blocks,
            mps_attention_blocks=args.mps_attention_blocks,
            mps_attention_heads=args.mps_attention_heads,
            mps_jastrow=args.mps_jastrow,
            mps_jastrow_hidden_sizes=tuple(args.mps_jastrow_hidden_sizes),
            mps_jastrow_log_clip=args.mps_jastrow_log_clip,
        ),
        sampler=SamplerConfig(
            proposal_mode=args.sampler_mode,
            hpsi_candidates=args.sampler_hpsi_candidates,
            n_chains=args.n_chains,
            n_burn=args.n_burn,
            n_burn_warmup=args.n_burn_warmup,
            n_between=args.n_between,
            n_samples=args.n_samples,
            prob_n_burn=args.prob_n_burn,
            prob_n_between=args.prob_n_between,
            prob_n_samples=args.prob_n_samples,
            parallel_tempering=args.parallel_tempering,
            tempering_replicas=args.tempering_replicas,
            tempering_beta_min=args.tempering_beta_min,
            tempering_swap_interval=args.tempering_swap_interval,
        ),
        continuous=ContinuousConfig(
            basis=args.continuous_basis,
            harmonic=args.continuous_harmonic,
            overlap_epsilon=args.continuous_overlap_epsilon,
            proposal_width=args.continuous_proposal_width,
            target_acceptance=args.continuous_target_acceptance,
            adapt_rate=args.continuous_adapt_rate,
            min_width=args.continuous_min_width,
            max_width=args.continuous_max_width,
            node_epsilon=args.continuous_node_epsilon,
        ),
        optimizer=OptimizerConfig(
            name=args.optimizer,
            lr=args.lr,
            constant_lr=args.constant_lr,
            decay_rate=args.decay_rate,
            multi_step_lr=args.multi_step_lr,
            cyclic_lr=args.cyclic_lr,
            cyclic_lr_min=args.cyclic_lr_min,
            cyclic_lr_period=args.cyclic_lr_period,
            cyclic_lr_mode=args.cyclic_lr_mode,
            report_fisher_condition=args.report_fisher_condition,
            sr_diag_shift=args.sr_diag_shift,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
            max_sr_step_norm=args.max_sr_step_norm,
            adam_b1=args.adam_b1,
            adam_b2=args.adam_b2,
            adam_eps=args.adam_eps,
            adam_weight_decay=args.adam_weight_decay,
            muon_beta=args.muon_beta,
            muon_ns_steps=args.muon_ns_steps,
            muon_weight_decay=args.muon_weight_decay,
            muon_adaptive=args.muon_adaptive,
        ),
        training=TrainingConfig(
            optimize_method=args.optimize_method,
            n_steps=args.n_steps,
            min_steps=args.min_steps,
            converge_tol=args.converge_tol,
            converge_window=args.converge_window,
            loss_levels=args.loss_levels,
            overlap_shift=args.overlap_shift,
            projection_rcond=args.projection_rcond,
            seed=args.seed,
            skip_train=args.skip_train,
            perturb_steps=args.perturb_steps,
            perturb_scale=args.perturb_scale,
            vmc_row_batch_size=args.vmc_row_batch_size,
            vmc_chain_batch_size=args.vmc_chain_batch_size,
            vmc_estimator_chains=args.vmc_estimator_chains,
            vmc_neighbor_approximation=args.vmc_neighbor_approximation,
            vmc_input_gradient_batch_size=args.vmc_input_gradient_batch_size,
            vmc_input_hessian_batch_size=args.vmc_input_hessian_batch_size,
            vmc_exact_neighbors_per_row=args.vmc_exact_neighbors_per_row,
            vmc_exact_top_neighbors=args.vmc_exact_top_neighbors,
            vmc_neighbor_sampling_uniform_mix=(
                args.vmc_neighbor_sampling_uniform_mix
            ),
            vmc_fixed_neighbor_row_nnz=args.vmc_fixed_neighbor_row_nnz,
            vmc_neighbor_padding_bucket_size=(
                args.vmc_neighbor_padding_bucket_size
            ),
            vmc_prebuild_sparse_rows=args.vmc_prebuild_sparse_rows,
            vmc_sparse_rows_from_bitstrings=(
                args.vmc_sparse_rows_from_bitstrings
            ),
            vmc_deduplicate_neighbors=(
                args.vmc_deduplicate_neighbors
                and not args.vmc_disable_neighbor_dedup
            ),
            vmc_disable_neighbor_padding=args.vmc_disable_neighbor_padding,
            vmc_timing_analysis=args.vmc_timing_analysis,
            vmc_timing_interval=args.vmc_timing_interval,
        ),
        output=OutputConfig(
            build_exact=args.build_exact,
            skip_final_ritz=args.skip_final_ritz,
            plot_output=args.plot_output,
            energy_curve_output=args.energy_curve_output,
            prob_plot_output=args.prob_plot_output,
            history_output=args.history_output,
            prob_snapshot_dir=args.prob_snapshot_dir,
            prob_snapshot_steps=tuple(args.prob_snapshot_steps),
            live_plot_output=args.live_plot_output,
            live_plot_interval=args.live_plot_interval,
            state_archive_output=args.state_archive_output,
            params_output=args.params_output,
        ),
    )


def parse_config(argv=None) -> tuple[argparse.Namespace, RunConfig]:
    args = build_parser().parse_args(argv)
    config = config_from_namespace(args)
    config.validate()
    return args, config
