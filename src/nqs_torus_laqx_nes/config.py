"""Typed configuration for the torus NQS workflow."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SystemConfig:
    nphi: int = 24
    p: int = 1
    q: int = 3
    tk: int = 0
    all_sectors: bool = False
    ratio_rect: float = 1.0
    landau_level: int = 0
    coulomb_cut: int | None = None
    disable_numba_coulomb: bool = False

    @property
    def num_sites(self) -> int:
        return self.nphi

    @property
    def ne(self) -> int:
        return (self.nphi // self.q) * self.p

    def validate(self) -> None:
        if self.nphi <= 0:
            raise ValueError("Nphi must be positive.")
        if self.q <= 0:
            raise ValueError("q must be positive.")
        if self.p < 0:
            raise ValueError("p must be non-negative.")
        if not 0 <= self.ne <= self.nphi:
            raise ValueError(
                f"Invalid particle number Ne={self.ne}; expected 0 <= Ne <= Nphi."
            )
        if not 0 <= self.tk < self.nphi:
            raise ValueError(f"tk={self.tk} must satisfy 0 <= tk < Nphi={self.nphi}.")
        if self.ratio_rect <= 0:
            raise ValueError("ratio_rect must be positive.")
        if self.landau_level < 0:
            raise ValueError("landau_level must be non-negative.")
        if self.coulomb_cut is not None and self.coulomb_cut <= 0:
            raise ValueError("coulomb_cut must be positive when provided.")


@dataclass(frozen=True)
class NetworkConfig:
    k: int = 1
    architecture: str = "ace_nes_1d"
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "silu"
    layer_norm: bool = True
    logphi_real_clip: float = 30.0
    init_std: float = 0.01
    forward_batch_size: int = 8192
    forward_parallel_chunks: int = 1
    nes_hidden: int = 58
    nes_layers: int = 4
    nes_mlp_hidden: int = 58
    nes_mlp_layers: int = 1
    nes_cutoff: int = 3
    vit_patch_size: int = 3
    vit_embed_dim: int = 256
    vit_num_heads: int = 8
    vit_num_layers: int = 4
    vit_mlp_ratio: int = 4
    vit_fourier_modes: int = 4
    mps_bond_dim: int = 16
    mps_tensor_generation: str = "sitewise"
    mps_contraction: str = "sequential"
    mps_block_size: int = 3
    mps_sitemajor: bool = False
    mps_tensor_bank_lazy: bool = False
    mps_first_layer_backend: str = "jax"
    mps_backflow_embed_dim: int = 128
    # Optional sequence blocks for neural MPS tensor generators.  These are
    # disabled by default to preserve the existing parameterization.
    mps_residual_blocks: int = 0
    mps_attention_blocks: int = 0
    mps_attention_heads: int = 4
    mps_jastrow: bool = False
    mps_jastrow_hidden_sizes: tuple[int, ...] = (64, 64)
    mps_jastrow_log_clip: float = 20.0
    num_sites: int | None = None

    def validate(self) -> None:
        if self.num_sites is not None and self.num_sites <= 0:
            raise ValueError("num_sites must be positive when provided.")
        if self.k <= 0:
            raise ValueError("k must be positive.")
        if self.architecture not in {"mlp", "mps", "vit", "ace_nes_1d"}:
            raise ValueError(
                f"Unsupported network architecture: {self.architecture}"
            )
        if self.architecture in {"mlp", "mps"}:
            if not self.hidden_sizes or any(
                width <= 0 for width in self.hidden_sizes
            ):
                raise ValueError(
                    "hidden_sizes must contain positive widths."
                )
            if self.activation not in {
                "tanh",
                "relu",
                "gelu",
                "leaky_relu",
                "silu",
                "poly",
            }:
                raise ValueError(
                    f"Unsupported activation: {self.activation}"
                )
        if self.mps_bond_dim <= 0:
            raise ValueError("mps_bond_dim must be positive.")
        if self.mps_jastrow_log_clip <= 0:
            raise ValueError("mps_jastrow_log_clip must be positive.")
        if self.mps_jastrow and (
            not self.mps_jastrow_hidden_sizes
            or any(width <= 0 for width in self.mps_jastrow_hidden_sizes)
        ):
            raise ValueError("mps_jastrow_hidden_sizes must be positive.")
        if self.mps_jastrow and self.architecture != "mps":
            raise ValueError("mps_jastrow requires architecture='mps'.")
        if self.mps_contraction not in {
            "sequential", "balanced", "balanced_fp64", "fused4"
        }:
            raise ValueError(
                "mps_contraction must be 'sequential', 'balanced', "
                "'balanced_fp64', or 'fused4'."
            )
        if self.mps_contraction == "balanced_fp64" and self.mps_sitemajor:
            raise ValueError(
                "balanced_fp64 is not supported with mps_sitemajor."
            )
        if self.mps_block_size < 1:
            raise ValueError("mps_block_size must be at least 1.")
        if self.mps_first_layer_backend not in {"jax", "triton", "reuse"}:
            raise ValueError(
                "mps_first_layer_backend must be 'jax', 'triton' or 'reuse'."
            )
        if self.mps_first_layer_backend in {"triton", "reuse"} and (
            self.architecture != "mps"
            or self.mps_tensor_generation != "backflow"
            or self.mps_attention_blocks
            or self.mps_residual_blocks
        ):
            raise ValueError(
                "the experimental first layer currently requires a backflow MPS "
                "without attention or residual blocks."
            )
        if self.mps_backflow_embed_dim <= 0:
            raise ValueError("mps_backflow_embed_dim must be positive.")
        if self.mps_residual_blocks < 0:
            raise ValueError("mps_residual_blocks must be non-negative.")
        if self.mps_attention_blocks < 0:
            raise ValueError("mps_attention_blocks must be non-negative.")
        if self.mps_attention_heads <= 0:
            raise ValueError("mps_attention_heads must be positive.")
        if self.mps_attention_blocks and self.mps_tensor_generation not in {
            "sitewise",
            "backflow",
        }:
            raise ValueError(
                "mps_attention_blocks currently requires tensor generation "
                "'sitewise' or 'backflow'."
            )
        if self.mps_residual_blocks and self.mps_tensor_generation == "direct":
            raise ValueError(
                "mps_residual_blocks requires a neural MPS tensor generator; "
                "direct MPS has no hidden layers."
            )
        attention_width = (
            self.mps_backflow_embed_dim
            if self.mps_tensor_generation == "backflow"
            else self.hidden_sizes[0]
        )
        if self.mps_attention_blocks and attention_width % self.mps_attention_heads:
            raise ValueError(
                "the first tensor-MPS hidden width must be divisible by "
                "mps_attention_heads."
            )
        if self.mps_tensor_generation not in {
            "direct",
            "sitewise",
            "tensor_bank",
            "backflow",
            "local_backflow",
            "local_conv_backflow",
            "global_backflow",
            "site_head_backflow",
        }:
            raise ValueError(
                "mps_tensor_generation must be 'direct', 'sitewise', "
                "'tensor_bank', 'backflow', 'local_backflow', "
                "'local_conv_backflow', "
                "'global_backflow', or "
                "'site_head_backflow'."
            )
        if self.architecture == "ace_nes_1d":
            if self.activation not in {
                "silu",
                "tanh",
                "poly",
                "gelu",
                "relu",
                "leaky_relu",
            }:
                raise ValueError(
                    f"Unsupported NES activation: {self.activation}"
                )
            nes_integer_values = {
                "nes_hidden": self.nes_hidden,
                "nes_layers": self.nes_layers,
                "nes_mlp_hidden": self.nes_mlp_hidden,
                "nes_mlp_layers": self.nes_mlp_layers,
                "nes_cutoff": self.nes_cutoff,
            }
            for name, value in nes_integer_values.items():
                if value <= 0:
                    raise ValueError(f"{name} must be positive.")
            if self.nes_cutoff % 2 == 0:
                raise ValueError("nes_cutoff must be odd.")
        if self.logphi_real_clip <= 0:
            raise ValueError("logphi_real_clip must be positive.")
        if self.init_std < 0:
            raise ValueError("init_std must be non-negative.")
        if self.forward_batch_size <= 0:
            raise ValueError("forward_batch_size must be positive.")
        if self.forward_parallel_chunks <= 0:
            raise ValueError("forward_parallel_chunks must be positive.")
        vit_integer_values = {
            "vit_patch_size": self.vit_patch_size,
            "vit_embed_dim": self.vit_embed_dim,
            "vit_num_heads": self.vit_num_heads,
            "vit_num_layers": self.vit_num_layers,
            "vit_mlp_ratio": self.vit_mlp_ratio,
            "vit_fourier_modes": self.vit_fourier_modes,
        }
        for name, value in vit_integer_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.vit_embed_dim % self.vit_num_heads != 0:
            raise ValueError(
                "vit_embed_dim must be divisible by vit_num_heads."
            )


@dataclass(frozen=True)
class SamplerConfig:
    proposal_mode: str = "sector_uniform"
    hpsi_candidates: int = 4
    n_chains: int = 1024
    n_burn: int = 20
    n_burn_warmup: int | None = None
    n_between: int = 1
    n_samples: int = 1
    prob_n_burn: int = 2048
    prob_n_between: int = 1
    prob_n_samples: int = 1
    parallel_tempering: bool = False
    tempering_replicas: int = 4
    tempering_beta_min: float = 0.25
    tempering_swap_interval: int = 1

    @property
    def tempering_betas(self) -> tuple[float, ...] | None:
        if not self.parallel_tempering:
            return None
        if self.tempering_replicas == 1:
            return (1.0,)
        return tuple(
            self.tempering_beta_min
            ** (replica / (self.tempering_replicas - 1))
            for replica in range(self.tempering_replicas)
        )

    def validate(self) -> None:
        if self.proposal_mode not in {
            "sector_uniform",
            "conserved_pair",
            "momentum_scattering",
            "laqx_hamiltonian",
            "hamiltonian_weighted",
            "hamiltonian_pair",
            "hamiltonian_hpsi_mtm",
        }:
            raise ValueError(
                f"Unsupported sampler proposal mode: {self.proposal_mode}"
            )
        integer_values = {
            "hpsi_candidates": self.hpsi_candidates,
            "n_chains": self.n_chains,
            "n_burn": self.n_burn,
            "n_between": self.n_between,
            "n_samples": self.n_samples,
            "prob_n_burn": self.prob_n_burn,
            "prob_n_between": self.prob_n_between,
            "prob_n_samples": self.prob_n_samples,
        }
        for name, value in integer_values.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        if self.n_burn_warmup is not None and self.n_burn_warmup < 0:
            raise ValueError("n_burn_warmup must be non-negative when provided.")
        if self.tempering_replicas < 2:
            raise ValueError("tempering_replicas must be at least 2.")
        if not 0 < self.tempering_beta_min < 1:
            raise ValueError("tempering_beta_min must lie strictly between 0 and 1.")
        if self.tempering_swap_interval <= 0:
            raise ValueError("tempering_swap_interval must be positive.")


@dataclass(frozen=True)
class ContinuousConfig:
    basis: str = "fourier"
    harmonic: int = 1
    overlap_epsilon: float = 0.1
    proposal_width: float = 0.05
    target_acceptance: float = 0.5
    adapt_rate: float = 0.1
    min_width: float = 1e-4
    max_width: float = 0.5
    node_epsilon: float = 1e-14

    def validate(self) -> None:
        if self.basis not in {"fourier", "compact_overlap"}:
            raise ValueError(
                f"Unsupported continuous basis: {self.basis}"
            )
        if self.harmonic <= 0:
            raise ValueError("continuous harmonic must be positive.")
        if not 0 < self.overlap_epsilon < 1:
            raise ValueError(
                "continuous overlap_epsilon must lie strictly between 0 and 1."
            )
        if not self.min_width <= self.proposal_width <= self.max_width:
            raise ValueError(
                "continuous proposal_width must lie between min_width and max_width."
            )
        if not 0 < self.target_acceptance < 1:
            raise ValueError(
                "continuous target_acceptance must lie strictly between 0 and 1."
            )
        if self.adapt_rate <= 0:
            raise ValueError("continuous adapt_rate must be positive.")
        if self.min_width <= 0 or self.max_width <= 0:
            raise ValueError("continuous proposal width bounds must be positive.")
        if self.min_width > self.max_width:
            raise ValueError(
                "continuous min_width must not exceed max_width."
            )
        if self.node_epsilon <= 0:
            raise ValueError("continuous node_epsilon must be positive.")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "signed_gradient"
    lr: float = 1e-2
    constant_lr: bool = False
    decay_rate: int = 1000
    multi_step_lr: bool = False
    cyclic_lr: bool = False
    cyclic_lr_min: float | None = None
    cyclic_lr_period: int = 1000
    cyclic_lr_mode: str = "triangular"
    report_fisher_condition: bool = False
    sr_diag_shift: float = 1e-3
    sr_samples: int | None = None
    sr_jacobian_batch_size: int = 32
    kfac_diagnostics: bool = False
    kfac_diagnostic_samples: int = 512
    kfac_trace_probes: int = 4
    cg_tol: float = 1e-8
    cg_maxiter: int = 2000
    max_sr_step_norm: float = 10.0
    wavefunction_trust_kl: float | None = None
    gradient_clip_norm: float | None = None
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0
    muon_adaptive: bool = False

    def validate(self) -> None:
        if self.name not in {
            "natural_gradient",
            "sr",
            "signed_gradient",
            "muon",
            "adam",
            "adamw",
        }:
            raise ValueError(f"Unsupported optimizer: {self.name}")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if self.sr_samples is not None and self.sr_samples <= 0:
            raise ValueError("sr_samples must be positive when provided.")
        if self.sr_jacobian_batch_size <= 0:
            raise ValueError("sr_jacobian_batch_size must be positive.")
        if self.kfac_diagnostic_samples <= 0:
            raise ValueError("kfac_diagnostic_samples must be positive.")
        if self.kfac_trace_probes < 0:
            raise ValueError("kfac_trace_probes must be non-negative.")
        enabled_schedules = sum(
            (
                bool(self.constant_lr),
                bool(self.multi_step_lr),
                bool(self.cyclic_lr),
            )
        )
        if enabled_schedules > 1:
            raise ValueError(
                "constant_lr, multi_step_lr, and cyclic_lr are mutually exclusive."
            )
        if self.decay_rate <= 0:
            raise ValueError("decay_rate must be positive.")
        if self.cyclic_lr_period <= 0:
            raise ValueError("cyclic_lr_period must be positive.")
        if self.cyclic_lr_mode not in {"triangular", "triangular2"}:
            raise ValueError(
                "cyclic_lr_mode must be one of {'triangular', 'triangular2'}."
            )
        if self.cyclic_lr_min is not None:
            if self.cyclic_lr_min <= 0:
                raise ValueError("cyclic_lr_min must be positive when provided.")
            if self.cyclic_lr_min > self.lr:
                raise ValueError("cyclic_lr_min must not exceed lr.")
        if self.sr_diag_shift < 0 or self.cg_tol <= 0 or self.cg_maxiter <= 0:
            raise ValueError("Invalid natural-gradient solver settings.")
        if self.max_sr_step_norm < 0:
            raise ValueError("max_sr_step_norm must be non-negative.")
        if (
            self.wavefunction_trust_kl is not None
            and self.wavefunction_trust_kl <= 0
        ):
            raise ValueError(
                "wavefunction_trust_kl must be positive when provided."
            )
        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if not 0 <= self.adam_b1 < 1 or not 0 <= self.adam_b2 < 1:
            raise ValueError("Adam beta values must be in [0, 1).")
        if self.adam_eps <= 0 or self.muon_ns_steps <= 0:
            raise ValueError("Optimizer epsilon and iteration counts must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    optimize_method: str = "vmc"
    n_steps: int = 10000
    min_steps: int = 20
    converge_tol: float = 0.0
    converge_window: int = 10
    loss_levels: int | None = None
    overlap_shift: float = 1e-8
    projection_rcond: float = 1e-10
    seed: int = 1234
    skip_train: bool = False
    perturb_steps: int = 0
    perturb_scale: float = 1e-4
    vmc_row_batch_size: int | None = None
    vmc_chain_batch_size: int | None = None
    vmc_estimator_chains: int | None = None
    vmc_neighbor_approximation: str = "exact"
    vmc_input_gradient_batch_size: int = 32
    vmc_input_hessian_batch_size: int = 4
    vmc_exact_neighbors_per_row: int = 64
    vmc_exact_top_neighbors: int = 8
    vmc_neighbor_sampling_uniform_mix: float = 0.1
    vmc_fixed_neighbor_row_nnz: int | None = None
    vmc_neighbor_padding_bucket_size: int | None = None
    vmc_prebuild_sparse_rows: bool = False
    vmc_deduplicate_neighbors: bool = False
    vmc_sparse_rows_from_bitstrings: bool = False
    vmc_disable_neighbor_padding: bool = False
    vmc_timing_analysis: bool = False
    vmc_timing_interval: int = 10

    def validate(self) -> None:
        if self.optimize_method not in {
            "projected",
            "vmc",
            "continuous_vmc",
        }:
            raise ValueError(f"Unsupported optimize_method: {self.optimize_method}")
        if self.vmc_neighbor_approximation not in {
            "exact", "input_log_taylor", "input_log_taylor2",
            "input_log_taylor_cv", "exact_neighbor_is"
        }:
            raise ValueError("Unsupported vmc_neighbor_approximation")
        if self.vmc_input_gradient_batch_size <= 0:
            raise ValueError("vmc_input_gradient_batch_size must be positive")
        if self.vmc_input_hessian_batch_size <= 0:
            raise ValueError("vmc_input_hessian_batch_size must be positive")
        if self.vmc_exact_neighbors_per_row <= 0:
            raise ValueError("vmc_exact_neighbors_per_row must be positive")
        if self.vmc_exact_top_neighbors < 0:
            raise ValueError("vmc_exact_top_neighbors must be non-negative")
        if not 0.0 < self.vmc_neighbor_sampling_uniform_mix <= 1.0:
            raise ValueError(
                "vmc_neighbor_sampling_uniform_mix must be in (0, 1]"
            )
        if self.n_steps < 0 or self.min_steps < 0:
            raise ValueError("Training step counts must be non-negative.")
        if self.converge_tol < 0 or self.converge_window <= 0:
            raise ValueError("Invalid convergence settings.")
        if self.loss_levels is not None and self.loss_levels <= 0:
            raise ValueError("loss_levels must be positive when provided.")
        if self.overlap_shift < 0 or self.projection_rcond <= 0:
            raise ValueError("Invalid projected diagonalization settings.")
        if self.perturb_steps < 0 or self.perturb_scale < 0:
            raise ValueError("Invalid parameter perturbation settings.")
        if (
            self.vmc_row_batch_size is not None
            and self.vmc_row_batch_size <= 0
        ):
            raise ValueError(
                "vmc_row_batch_size must be positive when provided."
            )
        if (
            self.vmc_chain_batch_size is not None
            and self.vmc_chain_batch_size <= 0
        ):
            raise ValueError(
                "vmc_chain_batch_size must be positive when provided."
            )
        if (
            self.vmc_estimator_chains is not None
            and self.vmc_estimator_chains <= 0
        ):
            raise ValueError(
                "vmc_estimator_chains must be positive when provided."
            )
        if (
            self.vmc_fixed_neighbor_row_nnz is not None
            and self.vmc_fixed_neighbor_row_nnz <= 0
        ):
            raise ValueError(
                "vmc_fixed_neighbor_row_nnz must be positive when provided."
            )
        if (
            self.vmc_neighbor_padding_bucket_size is not None
            and self.vmc_neighbor_padding_bucket_size <= 0
        ):
            raise ValueError(
                "vmc_neighbor_padding_bucket_size must be positive when "
                "provided."
            )
        if self.vmc_timing_interval <= 0:
            raise ValueError("vmc_timing_interval must be positive.")


@dataclass(frozen=True)
class OutputConfig:
    build_exact: bool = False
    skip_final_ritz: bool = False
    plot_output: str | None = None
    energy_curve_output: str | None = None
    prob_plot_output: str | None = None
    history_output: str | None = None
    prob_snapshot_dir: str | None = None
    prob_snapshot_steps: tuple[int, ...] = ()
    live_plot_output: str | None = None
    live_plot_interval: int = 0
    state_archive_output: str | None = None
    params_output: str | None = None

    def validate(self) -> None:
        if self.skip_final_ritz and self.state_archive_output is not None:
            raise ValueError(
                "state_archive_output is unavailable when skip_final_ritz is enabled."
            )
        if any(step <= 0 for step in self.prob_snapshot_steps):
            raise ValueError("prob_snapshot_steps must contain positive steps.")
        if self.live_plot_interval < 0:
            raise ValueError("live_plot_interval must be non-negative.")
        if self.live_plot_interval > 0 and self.live_plot_output is None:
            raise ValueError(
                "live_plot_output is required when live_plot_interval is positive."
            )


@dataclass(frozen=True)
class RunConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    continuous: ContinuousConfig = field(default_factory=ContinuousConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        self.system.validate()
        self.network.validate()
        self.sampler.validate()
        self.continuous.validate()
        self.optimizer.validate()
        self.training.validate()
        self.output.validate()
        if self.training.vmc_neighbor_approximation != "exact":
            if (self.training.optimize_method != "vmc"
                    or not self.training.vmc_sparse_rows_from_bitstrings
                    or self.network.architecture != "mps"
                    or self.network.mps_tensor_generation != "backflow"
                    or self.network.k != 1
                    or self.network.mps_attention_blocks
                    or self.network.mps_residual_blocks):
                raise ValueError(
                    "sampled/approximated neighbor modes require VMC streamed bitstring rows, "
                    "plain site backflow MPS and k=1"
                )
        estimator_chains = self.training.vmc_estimator_chains
        if estimator_chains is not None:
            if self.training.optimize_method != "vmc":
                raise ValueError(
                    "vmc_estimator_chains requires optimize_method='vmc'."
                )
            if estimator_chains > self.sampler.n_chains:
                raise ValueError(
                    "vmc_estimator_chains cannot exceed n_chains."
                )
            chain_batch_size = self.training.vmc_chain_batch_size
            estimator_samples = estimator_chains * self.sampler.n_samples
            if (
                chain_batch_size is not None
                and estimator_samples % chain_batch_size != 0
            ):
                raise ValueError(
                    "vmc_chain_batch_size must divide "
                    "vmc_estimator_chains * n_samples."
                )
        if (
            self.training.optimize_method == "continuous_vmc"
            and self.network.k != 1
        ):
            raise ValueError("continuous_vmc currently requires network.k == 1.")
        if self.training.vmc_sparse_rows_from_bitstrings:
            if self.training.optimize_method != "vmc":
                raise ValueError(
                    "vmc_sparse_rows_from_bitstrings requires optimize_method='vmc'."
                )
            if (
                self.training.vmc_row_batch_size is None
                or self.training.vmc_chain_batch_size is None
                or self.training.vmc_prebuild_sparse_rows
            ):
                raise ValueError(
                    "vmc_sparse_rows_from_bitstrings requires streamed rows "
                    "and vmc_chain_batch_size, and is incompatible with "
                    "vmc_prebuild_sparse_rows."
                )
            if self.sampler.proposal_mode not in {
                "sector_uniform",
                "conserved_pair",
                "hamiltonian_weighted",
                "hamiltonian_pair",
                "hamiltonian_hpsi_mtm",
            }:
                raise ValueError(
                    "vmc_sparse_rows_from_bitstrings currently requires "
                    "sampler_mode='sector_uniform', 'conserved_pair', or "
                    "a streaming Hamiltonian proposal."
                )
            if self.training.vmc_timing_analysis:
                raise ValueError(
                    "vmc_timing_analysis is not yet supported with "
                    "vmc_sparse_rows_from_bitstrings."
                )
            if (
                not self.output.skip_final_ritz
                or self.sampler.prob_n_samples > 0
                or self.output.prob_plot_output is not None
                or self.output.prob_snapshot_steps
            ):
                raise ValueError(
                    "vmc_sparse_rows_from_bitstrings requires "
                    "--skip_final_ritz, --prob_n_samples 0, and no "
                    "probability plots/snapshots."
                )
        if (
            self.sampler.parallel_tempering
            and self.training.optimize_method != "vmc"
        ):
            raise ValueError(
                "parallel tempering is currently supported only for discrete vmc."
            )
