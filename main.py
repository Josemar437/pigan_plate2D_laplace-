# -*- coding: utf-8 -*-
"""
Script principal para execução do pipeline PI-GAN.

Esta implementação corresponde a uma PI-GAN híbrida, definida como uma rede
adversarial generativa informada pela física, com penalização explícita do
resíduo da PDE e regularização adversarial.
"""

import argparse

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Constrói a configuração do experimento com base nos argumentos da CLI.

    Parâmetros:
        args: Argumentos parseados da linha de comando.

    Retorno:
        Um objeto ExperimentConfig configurado.
    """
    config_kwargs = {}
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


def main() -> None:
    """
    Ponto de entrada principal para o treinamento e avaliação da PI-GAN.
    """
    parser = argparse.ArgumentParser(description="Pipeline PI-GAN para Calor 2D")
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

    system_kwargs = {
        "log_file": "runs/logs/training.log",
        "seed": int(args.seed),
        "use_double": True,
    }
    if args.deterministic is not None:
        system_kwargs["deterministic_run"] = bool(args.deterministic)
    if bool(args.deterministic_warn_only):
        system_kwargs["deterministic_warn_only"] = True
    system_config = SystemConfig(**system_kwargs)
    experiment_config = _build_experiment_config(args)

    pipeline = PIGANPipeline(experiment_config, system_config)
    pipeline.run()


if __name__ == "__main__":
    main()

