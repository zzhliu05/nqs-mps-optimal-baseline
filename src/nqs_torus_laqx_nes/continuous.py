"""Continuous occupation-space basis, evaluator, and Metropolis sampler."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .config import ContinuousConfig, NetworkConfig
from .network import batched_wavefunction_matrix
from .physics import SparseHamiltonianRows


@dataclass(frozen=True)
class ContinuousEvaluationPlan:
    """CPU-built inverse maps for one globally deduplicated evaluation."""

    active_rows: object
    unique_state_indices: object
    row_to_unique: object
    neighbor_to_unique: object
    neighbor_count: int

    def to_jax(self) -> "ContinuousEvaluationPlan":
        return ContinuousEvaluationPlan(
            active_rows=jnp.asarray(self.active_rows, dtype=jnp.int64),
            unique_state_indices=jnp.asarray(
                self.unique_state_indices, dtype=jnp.int64
            ),
            row_to_unique=jnp.asarray(self.row_to_unique, dtype=jnp.int64),
            neighbor_to_unique=jnp.asarray(
                self.neighbor_to_unique, dtype=jnp.int64
            ),
            neighbor_count=self.neighbor_count,
        )

    @property
    def unique_count(self) -> int:
        return int(np.shape(self.unique_state_indices)[0])


def build_continuous_evaluation_plan(
    sparse_rows: SparseHamiltonianRows,
) -> ContinuousEvaluationPlan:
    """Collect active rows and all valid neighbors, then deduplicate globally."""
    diag = np.asarray(sparse_rows.diag)
    cols = np.asarray(sparse_rows.cols, dtype=np.int64)
    mask = np.asarray(sparse_rows.mask) > 0
    if diag.ndim != 1 or cols.ndim != 2 or mask.shape != cols.shape:
        raise ValueError("Invalid sparse Hamiltonian row shapes.")

    active_rows = np.arange(diag.shape[0], dtype=np.int64)
    valid_neighbors = cols[mask]
    unique_state_indices = np.unique(
        np.concatenate((active_rows, valid_neighbors))
    ).astype(np.int64)
    lookup = np.full(diag.shape[0], -1, dtype=np.int64)
    lookup[unique_state_indices] = np.arange(
        unique_state_indices.shape[0], dtype=np.int64
    )
    row_to_unique = lookup[active_rows]
    neighbor_to_unique = lookup[cols]
    if np.any(row_to_unique < 0) or np.any(neighbor_to_unique[mask] < 0):
        raise RuntimeError("Failed to construct continuous inverse mapping.")
    neighbor_to_unique = np.where(mask, neighbor_to_unique, 0)
    return ContinuousEvaluationPlan(
        active_rows=active_rows,
        unique_state_indices=unique_state_indices,
        row_to_unique=row_to_unique,
        neighbor_to_unique=neighbor_to_unique,
        neighbor_count=int(np.count_nonzero(mask)),
    )


def fourier_support_mask(coordinates, occupations):
    """Return the full support of the Fourier product basis."""
    coordinates = jnp.atleast_2d(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    occupations = jnp.atleast_2d(
        jnp.asarray(occupations, dtype=jnp.float64)
    )
    return jnp.ones(
        (coordinates.shape[0], occupations.shape[0]), dtype=jnp.bool_
    )


def fourier_product_basis(coordinates, occupations, harmonic: int = 1):
    """Evaluate the orthonormal product basis chi_K(x) on the unit torus."""
    coordinates = jnp.atleast_2d(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    occupations = jnp.atleast_2d(
        jnp.asarray(occupations, dtype=jnp.float64)
    )
    if coordinates.shape[1] != occupations.shape[1]:
        raise ValueError(
            "coordinate and occupation arrays must have the same Nphi."
        )
    signs = 1.0 - 2.0 * occupations
    phases = (
        2.0
        * jnp.pi
        * harmonic
        * jnp.einsum("bi,ki->bk", coordinates, signs)
    )
    return jnp.exp(1j * phases).astype(jnp.complex128)


def compact_overlap_functions(coordinates, epsilon: float = 0.1):
    """Evaluate the normalized single-site functions f_epsilon and g_epsilon."""
    x = jnp.asarray(coordinates, dtype=jnp.float64)
    a = 0.5 - 0.5 * epsilon
    b = 0.5 + 0.5 * epsilon
    s = (x - a) / epsilon

    f_raw = jnp.where(
        x <= a,
        1.0,
        jnp.where(x < b, 1.0 - s, 0.0),
    )
    g_raw = jnp.where(
        x <= a,
        0.0,
        jnp.where(x < b, s * (2.0 * s - 1.0), 1.0),
    )
    f_norm_sq = 0.5 - epsilon / 6.0
    g_norm_sq = 0.5 - 11.0 * epsilon / 30.0
    return (
        f_raw / jnp.sqrt(f_norm_sq),
        g_raw / jnp.sqrt(g_norm_sq),
    )


def compact_overlap_product_basis(
    coordinates,
    occupations,
    epsilon: float = 0.1,
):
    """Evaluate products of compact-overlap single-site occupation functions."""
    coordinates = jnp.atleast_2d(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    occupations = jnp.atleast_2d(
        jnp.asarray(occupations, dtype=jnp.float64)
    )
    if coordinates.shape[1] != occupations.shape[1]:
        raise ValueError(
            "coordinate and occupation arrays must have the same Nphi."
        )
    f_values, g_values = compact_overlap_functions(
        coordinates, epsilon
    )
    selected = jnp.where(
        occupations[None, :, :] > 0.5,
        g_values[:, None, :],
        f_values[:, None, :],
    )
    return jnp.prod(selected, axis=-1).astype(jnp.complex128)


def compact_overlap_support_mask(
    coordinates,
    occupations,
    epsilon: float = 0.1,
):
    """Return exact nonzero support for compact-overlap product states."""
    coordinates = jnp.atleast_2d(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    occupations = jnp.atleast_2d(
        jnp.asarray(occupations, dtype=jnp.float64)
    )
    f_values, g_values = compact_overlap_functions(
        coordinates, epsilon
    )
    selected = jnp.where(
        occupations[None, :, :] > 0.5,
        g_values[:, None, :],
        f_values[:, None, :],
    )
    return jnp.all(selected != 0.0, axis=-1)


def continuous_product_basis(
    coordinates,
    occupations,
    continuous_cfg: ContinuousConfig,
):
    if continuous_cfg.basis == "compact_overlap":
        return compact_overlap_product_basis(
            coordinates,
            occupations,
            continuous_cfg.overlap_epsilon,
        )
    return fourier_product_basis(
        coordinates, occupations, continuous_cfg.harmonic
    )


def continuous_support_mask(
    coordinates,
    occupations,
    continuous_cfg: ContinuousConfig,
):
    if continuous_cfg.basis == "compact_overlap":
        return compact_overlap_support_mask(
            coordinates,
            occupations,
            continuous_cfg.overlap_epsilon,
        )
    return fourier_support_mask(coordinates, occupations)


def _unique_coefficients(
    params,
    states,
    unique_state_indices,
    net_cfg: NetworkConfig,
):
    unique_states = jnp.asarray(states, dtype=jnp.float32)[
        unique_state_indices
    ]
    return batched_wavefunction_matrix(
        params, unique_states, net_cfg
    )[:, 0].astype(jnp.complex128)


def continuous_wavefunction(
    params,
    states,
    coordinates,
    active_rows,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
):
    """Reconstruct Psi(x) from the active occupation basis rows."""
    active_rows = jnp.asarray(active_rows, dtype=jnp.int64)
    active_states = jnp.asarray(states, dtype=jnp.float64)[active_rows]
    coefficients = _unique_coefficients(
        params, states, active_rows, net_cfg
    )
    chi = continuous_product_basis(
        coordinates, active_states, continuous_cfg
    )
    support = continuous_support_mask(
        coordinates, active_states, continuous_cfg
    )
    terms = jnp.where(
        support, chi * coefficients[None, :], jnp.complex128(0.0)
    )
    return jnp.sum(terms, axis=1, dtype=jnp.complex128), support


def continuous_psi_hpsi(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    plan: ContinuousEvaluationPlan,
    coordinates,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
):
    """Evaluate Psi(x) and H Psi(x) using one global unique-state forward."""
    diag, _, vals, mask = sparse_rows.as_tuple()
    unique_coefficients = _unique_coefficients(
        params, states, plan.unique_state_indices, net_cfg
    )
    row_coefficients = unique_coefficients[plan.row_to_unique]
    neighbor_coefficients = unique_coefficients[plan.neighbor_to_unique]
    hpsi_rows = (
        diag[plan.active_rows] * row_coefficients
        + jnp.sum(
            vals[plan.active_rows]
            * mask[plan.active_rows]
            * neighbor_coefficients,
            axis=1,
            dtype=jnp.complex128,
        )
    )

    active_states = jnp.asarray(states, dtype=jnp.float64)[
        plan.active_rows
    ]
    chi = continuous_product_basis(
        coordinates, active_states, continuous_cfg
    )
    support = continuous_support_mask(
        coordinates, active_states, continuous_cfg
    )
    chi_supported = jnp.where(
        support, chi, jnp.complex128(0.0)
    )
    psi = jnp.sum(
        chi_supported * row_coefficients[None, :],
        axis=1,
        dtype=jnp.complex128,
    )
    hpsi = jnp.sum(
        chi_supported * hpsi_rows[None, :],
        axis=1,
        dtype=jnp.complex128,
    )
    return psi, hpsi, support


def continuous_logpsi(
    params,
    states,
    coordinates,
    active_rows,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
):
    psi, _ = continuous_wavefunction(
        params,
        states,
        coordinates,
        active_rows,
        net_cfg,
        continuous_cfg,
    )
    abs_psi = jnp.abs(psi)
    safe_abs = jnp.maximum(abs_psi, continuous_cfg.node_epsilon)
    return jnp.log(safe_abs) + 1j * jnp.angle(psi)


def continuous_vmc_surrogate_loss(
    params,
    states,
    sparse_rows: SparseHamiltonianRows,
    plan: ContinuousEvaluationPlan,
    coordinates,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
):
    psi, hpsi, support = continuous_psi_hpsi(
        params,
        states,
        sparse_rows,
        plan,
        coordinates,
        net_cfg,
        continuous_cfg,
    )
    abs_psi = jnp.abs(psi)
    valid = abs_psi >= continuous_cfg.node_epsilon
    safe_psi = jnp.where(valid, psi, jnp.complex128(1.0))
    eloc = jnp.where(valid, hpsi / safe_psi, jnp.complex128(0.0))
    eloc = jax.lax.stop_gradient(eloc)
    valid_weight = valid.astype(jnp.float64)
    valid_count = jnp.sum(valid_weight)
    normalization = jnp.maximum(valid_count, 1.0)
    energy_complex = jnp.sum(valid_weight * eloc) / normalization
    energy = jnp.real(energy_complex)
    centered = jax.lax.stop_gradient(eloc - energy)
    logpsi = jnp.log(
        jnp.maximum(abs_psi, continuous_cfg.node_epsilon)
    ) + 1j * jnp.angle(psi)
    loss = (
        2.0
        * jnp.real(
            jnp.sum(
                valid_weight * jnp.conj(centered) * logpsi
            )
            / normalization
        )
    )
    variance = (
        jnp.sum(valid_weight * jnp.abs(eloc - energy) ** 2)
        / normalization
    )
    return loss, {
        "energy": energy,
        "variance": jnp.real(variance),
        "eloc_mean_imag": jnp.imag(energy_complex),
        "invalid_node_count": valid.shape[0] - valid_count,
        "min_abs_psi": jnp.min(abs_psi),
        "active_union_count": jnp.sum(jnp.any(support, axis=0)),
    }


def continuous_logprob(
    params,
    states,
    coordinates,
    active_rows,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
):
    psi, _ = continuous_wavefunction(
        params,
        states,
        coordinates,
        active_rows,
        net_cfg,
        continuous_cfg,
    )
    abs_psi = jnp.abs(psi)
    return jnp.where(
        abs_psi >= continuous_cfg.node_epsilon,
        2.0 * jnp.log(abs_psi),
        -jnp.inf,
    )


@partial(
    jax.jit,
    static_argnames=("net_cfg", "continuous_cfg", "n_chains", "nphi"),
)
def _continuous_step_fn(
    params,
    states,
    active_rows,
    key,
    coordinates,
    logprob,
    proposal_width,
    net_cfg: NetworkConfig,
    continuous_cfg: ContinuousConfig,
    n_chains: int,
    nphi: int,
):
    delta_key, accept_key = jax.random.split(key)
    delta = jax.random.uniform(
        delta_key,
        shape=(n_chains, nphi),
        minval=-proposal_width,
        maxval=proposal_width,
        dtype=jnp.float64,
    )
    proposal = jnp.mod(coordinates + delta, 1.0)
    proposal_logprob = continuous_logprob(
        params,
        states,
        proposal,
        active_rows,
        net_cfg,
        continuous_cfg,
    )
    log_uniform = jnp.log(
        jax.random.uniform(
            accept_key, shape=(n_chains,), dtype=jnp.float64
        )
    )
    accept = log_uniform < (proposal_logprob - logprob)
    new_coordinates = jnp.where(
        accept[:, None], proposal, coordinates
    )
    new_logprob = jnp.where(accept, proposal_logprob, logprob)
    acceptance = jnp.mean(accept.astype(jnp.float64))
    return new_coordinates, new_logprob, acceptance


class ContinuousMetropolisSamplerJAX:
    """Persistent periodic Metropolis chains in [0, 1)^Nphi."""

    def __init__(
        self,
        params,
        states,
        active_rows,
        n_chains: int,
        net_cfg: NetworkConfig,
        continuous_cfg: ContinuousConfig,
        seed: int = 1234,
    ):
        if net_cfg.k != 1:
            raise ValueError("continuous sampler currently requires k == 1.")
        self.params = params
        self.states = jnp.asarray(states, dtype=jnp.float64)
        self.active_rows = jnp.asarray(active_rows, dtype=jnp.int64)
        self.n_chains = int(n_chains)
        self.nphi = int(self.states.shape[1])
        self.net_cfg = net_cfg
        self.continuous_cfg = continuous_cfg
        self.proposal_width = float(continuous_cfg.proposal_width)
        self.width_frozen = False
        self.adaptation_steps = 0
        self.key = jax.random.PRNGKey(seed)
        self.coordinates = None
        self.logprob = None
        self.n_accepted = 0
        self.n_proposed = 0
        self.reinitialize()

    def _split(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def reinitialize(self) -> None:
        if self.continuous_cfg.basis == "compact_overlap":
            state_key, coordinate_key = jax.random.split(self._split())
            state_indices = jax.random.randint(
                state_key,
                shape=(self.n_chains,),
                minval=0,
                maxval=self.states.shape[0],
                dtype=jnp.int64,
            )
            occupations = self.states[state_indices]
            uniform = jax.random.uniform(
                coordinate_key,
                shape=(self.n_chains, self.nphi),
                minval=1e-12,
                maxval=1.0 - 1e-12,
                dtype=jnp.float64,
            )
            epsilon = self.continuous_cfg.overlap_epsilon
            a = 0.5 - 0.5 * epsilon
            b = 0.5 + 0.5 * epsilon
            self.coordinates = jnp.where(
                occupations > 0.5,
                a + (1.0 - a) * uniform,
                b * uniform,
            )
        else:
            self.coordinates = jax.random.uniform(
                self._split(),
                shape=(self.n_chains, self.nphi),
                dtype=jnp.float64,
            )
        self.refresh_logprob()
        self.n_accepted = 0
        self.n_proposed = 0

    def refresh_logprob(self) -> None:
        self.logprob = continuous_logprob(
            self.params,
            self.states,
            self.coordinates,
            self.active_rows,
            self.net_cfg,
            self.continuous_cfg,
        )

    def set_params(self, params) -> None:
        self.params = params
        self.refresh_logprob()

    def _adapt_width(self, acceptance: float) -> None:
        if self.width_frozen:
            return
        self.adaptation_steps += 1
        learning_rate = (
            self.continuous_cfg.adapt_rate
            / np.sqrt(self.adaptation_steps)
        )
        log_width = np.log(self.proposal_width) + learning_rate * (
            acceptance - self.continuous_cfg.target_acceptance
        )
        self.proposal_width = float(
            np.clip(
                np.exp(log_width),
                self.continuous_cfg.min_width,
                self.continuous_cfg.max_width,
            )
        )

    def step(self, adapt_width: bool = False) -> float:
        self.coordinates, self.logprob, acceptance = _continuous_step_fn(
            self.params,
            self.states,
            self.active_rows,
            self._split(),
            self.coordinates,
            self.logprob,
            jnp.asarray(self.proposal_width, dtype=jnp.float64),
            self.net_cfg,
            self.continuous_cfg,
            self.n_chains,
            self.nphi,
        )
        rate = float(acceptance)
        if adapt_width:
            self._adapt_width(rate)
        self.n_accepted += int(round(rate * self.n_chains))
        self.n_proposed += self.n_chains
        return rate

    def sample(
        self,
        n_burn: int,
        n_between: int,
        n_samples: int,
        adapt_width: bool = False,
        freeze_after_burn: bool = False,
    ) -> tuple[np.ndarray, float]:
        for _ in range(n_burn):
            self.step(adapt_width=adapt_width)
        if freeze_after_burn:
            self.width_frozen = True

        coordinates = []
        acceptances = []
        for _ in range(n_samples):
            for _ in range(n_between):
                acceptances.append(self.step())
            coordinates.append(np.asarray(self.coordinates))
        coordinates_out = (
            np.concatenate(coordinates, axis=0)
            if coordinates
            else np.empty((0, self.nphi), dtype=np.float64)
        )
        acceptance = (
            float(np.mean(acceptances)) if acceptances else 0.0
        )
        return coordinates_out, acceptance

    def acceptance_rate(self) -> float:
        return self.n_accepted / max(1, self.n_proposed)
