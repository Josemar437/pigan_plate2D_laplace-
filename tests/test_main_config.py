# -*- coding: utf-8 -*-
import argparse
import json

from main import _build_experiment_config, _build_system_config


def _args(config_path: str, **overrides):
    defaults = {
        "config": config_path,
        "generator_mode": None,
        "latent_dim": None,
        "save_frequency": None,
        "checkpoint_dir": None,
        "resume_checkpoint": None,
        "no_strict_checkpoint": False,
        "seed": None,
        "deterministic": None,
        "deterministic_warn_only": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_main_loads_experiment_config_json(tmp_path):
    tmp_path = tmp_path
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "source_best_trial": "metadata-only.json",
                "generator_mode": "deterministic_adversarial",
                "epochs": 7,
                "steps_per_epoch": 2,
                "grid_size_x": 18,
                "grid_size_y": 20,
                "batch_size": 5,
                "save_frequency": 3,
            }
        ),
        encoding="utf-8",
    )

    config = _build_experiment_config(_args(str(config_path)))

    assert config.generator_mode == "deterministic_adversarial"
    assert config.latent_dim == 0
    assert config.epochs == 7
    assert config.steps_per_epoch == 2
    assert config.grid_size_x == 18
    assert config.grid_size_y == 20
    assert config.batch_size == 5
    assert config.save_frequency == 3


def test_cli_overrides_config_json(tmp_path):
    tmp_path = tmp_path
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "generator_mode": "stochastic_pigan",
                "latent_dim": 8,
                "save_frequency": 100,
                "use_double": False,
            }
        ),
        encoding="utf-8",
    )

    args = _args(
        str(config_path),
        generator_mode="deterministic_adversarial",
        save_frequency=25,
        seed=123,
    )
    exp_config = _build_experiment_config(args)
    sys_config = _build_system_config(args)

    assert exp_config.generator_mode == "deterministic_adversarial"
    assert exp_config.latent_dim == 0
    assert exp_config.save_frequency == 25
    assert sys_config.seed == 123
    assert sys_config.use_double is False


def test_system_config_json_seed_is_preserved(tmp_path):
    tmp_path = tmp_path
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"seed": 77}), encoding="utf-8")

    sys_config = _build_system_config(_args(str(config_path)))

    assert sys_config.seed == 77
