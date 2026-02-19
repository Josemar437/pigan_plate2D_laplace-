# -*- coding: utf-8 -*-
import os

import torch

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline


def _build_base_configs():
    exp = ExperimentConfig()
    exp.generator_mode = "stochastic_pigan"
    exp.batch_size = 8
    exp.latent_dim = 8
    exp.save_frequency = 1
    exp.analysis_num_samples = 2
    exp.generate_plots = False
    exp.generate_extended_plots = False

    sys = SystemConfig(
        use_gpu=False,
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
    )
    return exp, sys


def test_resume_from_checkpoint_cpu(tmp_path):
    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")

    exp1, sys1 = _build_base_configs()
    exp1.epochs = 2

    pipeline1 = PIGANPipeline(exp1, sys1, runs_dir=tmp_path)
    pipeline1.run()

    ckpt2 = tmp_path / "checkpoints" / "checkpoint_epoch_2.pt"
    assert ckpt2.exists()

    saved = torch.load(ckpt2, map_location="cpu")
    assert "config" in saved
    assert "rng_state" in saved
    assert "numpy" in saved["rng_state"]
    assert "python" in saved["rng_state"]
    assert "torch_cpu" in saved["rng_state"]

    exp2, sys2 = _build_base_configs()
    exp2.epochs = 3
    exp2.resume_checkpoint = str(ckpt2)
    exp2.strict_checkpoint_loading = True

    pipeline2 = PIGANPipeline(exp2, sys2, runs_dir=tmp_path)
    _, _, _, history2 = pipeline2.run()

    assert isinstance(history2, list)
    assert len(history2) == 1
    assert int(history2[0].get("epoch", 0)) == 3

    ckpt3 = tmp_path / "checkpoints" / "checkpoint_epoch_3.pt"
    assert ckpt3.exists()
