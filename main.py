# -*- coding: utf-8 -*-
"""CLI do treino PI-GAN Laplace 2D definido em `src/pipeline.py`."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline


def _load_config_payload(config_path: str | None) -> dict:
    """Carrega um JSON de configuração, quando informado."""
    if not config_path:
        return {}
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _config_kwargs(payload: dict, config_type: type) -> dict:
    """Filtra chaves do JSON para os campos aceitos pelo dataclass informado."""
    valid_fields = {field.name for field in fields(config_type)}
    return {key: value for key, value in payload.items() if key in valid_fields}


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Mescla JSON, overrides da CLI e validacao de `ExperimentConfig`."""
    payload = _load_config_payload(getattr(args, "config", None))
    config_kwargs = _config_kwargs(payload, ExperimentConfig)
    if args.generator_mode is not None:
        config_kwargs["generator_mode"] = str(args.generator_mode)
    config = ExperimentConfig(**config_kwargs)
    if args.latent_dim is not None:
        config.latent_dim = int(args.latent_dim)
    if args.save_frequency is not None:
        config.save_frequency = max(0, int(args.save_frequency))
    if args.checkpoint_dir:
        config.checkpoint_dir = str(args.checkpoint_dir)
    if args.resume_checkpoint:
        config.resume_checkpoint = str(args.resume_checkpoint)
    if args.no_strict_checkpoint:
        config.strict_checkpoint_loading = False
    config.__post_init__()
    return config


def _build_system_config(args: argparse.Namespace) -> SystemConfig:
    """Constrói a configuração de sistema a partir de JSON e overrides da CLI."""
    payload = _load_config_payload(getattr(args, "config", None))
    system_kwargs = {
        "log_file": "runs/logs/training.log",
        "use_double": True,
    }
    system_kwargs.update(_config_kwargs(payload, SystemConfig))

    if getattr(args, "seed", None) is not None:
        system_kwargs["seed"] = int(args.seed)
    if getattr(args, "deterministic", None) is not None:
        system_kwargs["deterministic_run"] = bool(args.deterministic)
    if bool(getattr(args, "deterministic_warn_only", False)):
        system_kwargs["deterministic_warn_only"] = True

    return SystemConfig(**system_kwargs)


def main() -> None:
    """Executa treino, avaliacao e escrita de artefatos em `runs/`."""
    parser = argparse.ArgumentParser(description="Pipeline PI-GAN para Calor 2D")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Arquivo JSON com parâmetros de ExperimentConfig/SystemConfig.",
    )
    parser.add_argument(
        "--generator-mode",
        type=str,
        default=None,
        choices=["stochastic_pigan", "deterministic_adversarial"],
        help=(
            "Modo gerativo explícito: estocástico PI-GAN ou determinístico adversarial. "
            "Se omitido, usa o padrão do ExperimentConfig."
        ),
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Dimensão de z (obrigatória para stochastic_pigan).",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=None,
        help="Frequência (épocas) para salvar checkpoints no fluxo ativo.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Diretório de checkpoints (padrão: runs/checkpoints).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Caminho para checkpoint (.pt) para retomar o treino no fluxo ativo.",
    )
    parser.add_argument(
        "--no-strict-checkpoint",
        action="store_true",
        help="Permite retomar checkpoint com strict=False no carregamento de estados.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed global para torch/numpy/python.",
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Força execução determinística (reproduzível).",
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Desativa determinismo estrito para maximizar performance.",
    )
    parser.set_defaults(deterministic=None)
    parser.add_argument(
        "--deterministic-warn-only",
        action="store_true",
        help="Não interrompe se algum operador CUDA não tiver caminho determinístico.",
    )
    args = parser.parse_args()

    system_config = _build_system_config(args)
    experiment_config = _build_experiment_config(args)

    pipeline = PIGANPipeline(experiment_config, system_config)
    pipeline.run()


if __name__ == "__main__":
    main()

