# -*- coding: utf-8 -*-
import os
import numpy as np

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline


def test_pipeline_execution_cpu(tmp_path):
    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")

    exp_config = ExperimentConfig()
    exp_config.epochs = 2
    exp_config.generator_mode = "stochastic_pigan"
    exp_config.batch_size = 8
    exp_config.latent_dim = 8
    exp_config.save_frequency = 1
    exp_config.analysis_num_samples = 2
    exp_config.generate_plots = False
    exp_config.generate_extended_plots = False

    sys_config = SystemConfig(
        use_gpu=False,
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
    )

    pipeline = PIGANPipeline(exp_config, sys_config, runs_dir=tmp_path)
    generator, discriminator, analyzer, history = pipeline.run()

    assert generator is not None
    assert discriminator is not None
    assert analyzer is not None
    assert isinstance(history, list)
    assert generator.latent_dim == exp_config.latent_dim
    assert generator.in_channels == 3
    assert pipeline.device.type == "cpu"

    checkpoint_files = list((tmp_path / "checkpoints").glob("checkpoint_epoch_*.pt"))
    assert len(checkpoint_files) >= 1

    residual = np.load(tmp_path / "results" / "pde_residual.npy")
    assert np.allclose(residual[0, :], 0.0, atol=1e-12)
    assert np.allclose(residual[-1, :], 0.0, atol=1e-12)
    assert np.allclose(residual[:, 0], 0.0, atol=1e-12)
    assert np.allclose(residual[:, -1], 0.0, atol=1e-12)
