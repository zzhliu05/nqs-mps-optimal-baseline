from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nqs_mps.settings import core_config, load_settings
from nqs_torus_laqx_nes.optimization import _solve_minsr_design


def test_minsr_sample_space_matches_parameter_space():
    rng = np.random.default_rng(20261107)
    design = rng.normal(size=(8, 13)).astype(np.float32)
    gradient = rng.normal(size=(13,)).astype(np.float32)
    shift = 0.07

    actual = np.asarray(
        _solve_minsr_design(jnp.asarray(design), jnp.asarray(gradient), shift)
    )
    expected = np.linalg.solve(
        design.T @ design + shift * np.eye(design.shape[1]), -gradient
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_public_sr_config_maps_to_core_optimizer(tmp_path):
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "nphi51_tk0_params_0p2m_sr_2000.json"
    )
    settings = load_settings(source)
    config = core_config(settings, tmp_path / "output")
    assert config.optimizer.name == "sr"
    assert config.optimizer.sr_diag_shift == 0.001
    assert config.optimizer.sr_samples == 512
    assert config.optimizer.sr_jacobian_batch_size == 32


def test_public_kfac_diagnostic_config_maps_to_core_optimizer(tmp_path):
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "nphi51_tk0_kfac_diagnostic.json"
    )
    settings = load_settings(source)
    config = core_config(settings, tmp_path / "output")
    assert config.optimizer.name == "natural_gradient"
    assert config.optimizer.kfac_diagnostics is True
    assert config.optimizer.kfac_diagnostic_samples == 512
    assert config.optimizer.kfac_trace_probes == 4
