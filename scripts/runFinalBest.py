#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Executa o treino final de 4000 épocas usando o melhor trial do Optuna.

Padrões:
- saída do estudo: runs_optuna_physics_first_60t/best_trial.json
- horizonte de treino: 4000 épocas
- modo adversarial determinístico com foco físico
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, SystemConfig  # noqa: E402
from src.pipeline import PIGANPipeline  # noqa: E402


def _read_best_trial(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de melhor trial nao encontrado: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Conteudo do best_trial.json invalido.")
    if "best_params" not in data or not isinstance(data["best_params"], dict):
        raise ValueError("best_trial.json nao contem best_params valido.")
    return data


def _read_study_summary_for_best_trial(best_trial_path: Path) -> Dict[str, Any]:
    summary_path = best_trial_path.with_name("study_summary.json")
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _scaled(value: int, *, src_epochs: int, dst_epochs: int) -> int:
    if src_epochs <= 0:
        return int(value)
    factor = float(dst_epochs) / float(src_epochs)
    return max(1, int(round(float(value) * factor)))


def _has_control_search_schema(best_params: Dict[str, Any]) -> bool:
    return "gen_lr" in best_params and "disc_lr_ratio" in best_params


def _build_experiment_config(
    *,
    best_params: Dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    grid_size: int,
    batch_size: int,
    target_residual: float,
    tuning_epochs: int,
    scale_schedules: bool,
) -> ExperimentConfig:
    if _has_control_search_schema(best_params):
        return _build_control_search_experiment_config(
            best_params=best_params,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            grid_size=grid_size,
            batch_size=batch_size,
            target_residual=target_residual,
            tuning_epochs=tuning_epochs,
            scale_schedules=scale_schedules,
        )

    return _build_physics_first_experiment_config(
        best_params=best_params,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        grid_size=grid_size,
        batch_size=batch_size,
        target_residual=target_residual,
        tuning_epochs=tuning_epochs,
        scale_schedules=scale_schedules,
    )


def _build_physics_first_experiment_config(
    *,
    best_params: Dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    grid_size: int,
    batch_size: int,
    target_residual: float,
    tuning_epochs: int,
    scale_schedules: bool,
) -> ExperimentConfig:
    lr_g = float(best_params["lr_g"])
    lr_d = lr_g * float(best_params["lr_d_ratio"])
    lr_d1 = lr_d * float(best_params["lr_d1_ratio"])
    lr_d2 = lr_d * float(best_params["lr_d2_ratio"])

    adv_warmup = int(best_params["adv_warmup_epochs"])
    plateau_patience = int(best_params["plateau_patience"])
    plateau_cooldown = int(best_params["plateau_cooldown"])
    if scale_schedules:
        adv_warmup = _scaled(adv_warmup, src_epochs=tuning_epochs, dst_epochs=epochs)
        plateau_patience = _scaled(plateau_patience, src_epochs=tuning_epochs, dst_epochs=epochs)
        plateau_cooldown = _scaled(plateau_cooldown, src_epochs=tuning_epochs, dst_epochs=epochs)

    exp = ExperimentConfig(generator_mode="deterministic_adversarial")
    exp.epochs = int(epochs)
    exp.steps_per_epoch = int(steps_per_epoch)
    exp.batch_size = int(batch_size)
    exp.grid_size_x = int(grid_size)
    exp.grid_size_y = int(grid_size)
    exp.generator_base_channels = 12
    exp.generator_depth = 3
    exp.generator_zero_init_final = True
    exp.generator_use_batch_norm = False

    exp.discriminator_base_channels = 12
    exp.discriminator_capacity_scale = float(best_params["discriminator_capacity_scale"])
    exp.discriminator_dropout = float(best_params["discriminator_dropout"])
    exp.discriminator_spectral_norm = True

    exp.lambda_adv1 = float(best_params["lambda_adv1"])
    exp.lambda_adv2 = float(best_params["lambda_adv2"])
    exp.lambda_pde = float(best_params["lambda_pde"])
    exp.lambda_pde_raw = 0.0
    exp.lambda_bc = 0.0

    exp.gen_lr = float(lr_g)
    exp.disc_lr = float(lr_d)
    exp.disc_lr_d1 = float(lr_d1)
    exp.disc_lr_d2 = float(lr_d2)
    exp.max_grad_norm = float(best_params["max_grad_norm"])

    exp.n_critic = 1
    exp.disc_update_every = int(best_params["disc_update_every"])
    exp.target_adv_over_pde = float(best_params["target_adv_over_pde"])
    exp.gradnorm_target_adv_to_pde = float(best_params["gradnorm_target_adv_to_pde"])
    exp.adv_warmup_epochs = int(adv_warmup)
    exp.adv_residual_gate_target = float(best_params["adv_residual_gate_target"])
    exp.adv_residual_gate_min = 0.01

    exp.adaptive_lambda_pde = True
    exp.residual_scale_reference = float(best_params["residual_scale_reference"])
    exp.lambda_pde_growth_exponent = float(best_params["lambda_pde_growth_exponent"])
    exp.lambda_pde_min = float(best_params["lambda_pde_min"])
    exp.lambda_pde_max = float(best_params["lambda_pde_max"])
    exp.lambda_pde_ema_beta = 0.9

    exp.divergence_window = 16
    exp.divergence_ratio_threshold = 1.2
    exp.divergence_patience = 2
    exp.lr_drop_factor = 0.5
    exp.max_lr_drops = 10

    exp.plateau_scheduler_enabled = True
    exp.plateau_metric_key = "g_residual_mean_abs"
    exp.plateau_mode = "min"
    exp.plateau_patience = int(plateau_patience)
    exp.plateau_factor = float(best_params["plateau_factor"])
    exp.plateau_min_delta = float(best_params["plateau_min_delta"])
    exp.plateau_cooldown = int(plateau_cooldown)
    exp.plateau_max_drops = 10
    exp.plateau_reduce_discriminators = False

    exp.residual_tolerance_target = float(target_residual)
    exp.fdm_tol = 1e-12
    exp.fdm_max_iter = 100000

    exp.analysis_num_samples = 1
    exp.generate_plots = True
    exp.generate_extended_plots = True
    exp.save_frequency = max(50, int(epochs // 40))
    exp.__post_init__()
    return exp


def _build_control_search_experiment_config(
    *,
    best_params: Dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
    grid_size: int,
    batch_size: int,
    target_residual: float,
    tuning_epochs: int,
    scale_schedules: bool,
) -> ExperimentConfig:
    lr_g = float(best_params["gen_lr"])
    lr_d = lr_g * float(best_params["disc_lr_ratio"])

    adv_warmup = int(best_params["adv_warmup_epochs"])
    stagnation_patience = int(best_params["adv_stagnation_patience"])
    stagnation_cooldown = int(best_params["adv_stagnation_cooldown"])
    if scale_schedules:
        adv_warmup = _scaled(adv_warmup, src_epochs=tuning_epochs, dst_epochs=epochs)
        stagnation_patience = _scaled(
            stagnation_patience, src_epochs=tuning_epochs, dst_epochs=epochs
        )
        stagnation_cooldown = _scaled(
            stagnation_cooldown, src_epochs=tuning_epochs, dst_epochs=epochs
        )

    exp = ExperimentConfig(generator_mode="stochastic_pigan")
    exp.epochs = int(epochs)
    exp.steps_per_epoch = int(steps_per_epoch)
    exp.batch_size = int(batch_size)
    exp.grid_size_x = int(grid_size)
    exp.grid_size_y = int(grid_size)
    exp.latent_dim = 8
    exp.generator_base_channels = 12
    exp.generator_depth = 3
    exp.generator_zero_init_final = True

    exp.discriminator_base_channels = 12
    exp.use_physical_coordinates = True
    exp.use_reference_discriminator = True

    exp.gen_lr = float(lr_g)
    exp.disc_lr = float(lr_d)
    exp.disc_lr_d1 = float(lr_d)
    exp.disc_lr_d2 = float(lr_d)
    exp.max_grad_norm = float(best_params["max_grad_norm"])

    exp.lambda_adv1 = float(best_params["lambda_adv1"])
    exp.lambda_adv2 = float(best_params["lambda_adv2"])
    exp.lambda_pde = float(best_params["lambda_pde"])
    exp.lambda_pde_raw = 0.0
    exp.lambda_bc = 0.0
    exp.lambda_gp = float(best_params["lambda_gp"])
    exp.lambda_gp_d1 = exp.lambda_gp
    exp.lambda_gp_d2 = exp.lambda_gp

    exp.n_critic = int(best_params["n_critic"])
    exp.disc_update_every = int(best_params["disc_update_every"])

    exp.dynamic_adv_balance = True
    exp.target_adv_over_pde = float(best_params["target_adv_over_pde"])
    exp.adv_scale_min = float(best_params["adv_scale_min"])
    exp.adv_scale_max = float(best_params["adv_scale_max"])
    exp.gradnorm_balance = True
    exp.gradnorm_target_adv_to_pde = float(best_params["gradnorm_target_adv_to_pde"])

    exp.adv_warmup_epochs = int(adv_warmup)
    exp.adv_residual_gate_target = float(best_params["adv_residual_gate_target"])
    exp.adv_residual_gate_min = float(best_params["adv_residual_gate_min"])
    exp.adv_residual_gate_hysteresis = bool(best_params["adv_residual_gate_hysteresis"])
    exp.adv_residual_gate_power = float(best_params["adv_residual_gate_power"])
    exp.adv_residual_gate_off_threshold = (
        float(best_params["adv_residual_gate_target"])
        * float(best_params["adv_residual_gate_off_ratio"])
    )

    exp.critic_pause_on_overgap = True
    exp.max_critic_gap = float(best_params["max_critic_gap"])
    exp.critic_pause_gap_factor = float(best_params["critic_pause_gap_factor"])
    exp.critic_resume_gap_factor = (
        float(best_params["critic_pause_gap_factor"])
        * float(best_params["critic_resume_gap_ratio"])
    )
    exp.critic_gap_penalty = float(best_params["critic_gap_penalty"])
    exp.critic_drift = float(best_params["critic_drift"])

    exp.adv_stagnation_boost = True
    exp.adv_stagnation_patience = int(stagnation_patience)
    exp.adv_stagnation_rel_tol = float(best_params["adv_stagnation_rel_tol"])
    exp.adv_stagnation_boost_factor = float(best_params["adv_stagnation_boost_factor"])
    exp.adv_stagnation_min_gate = float(best_params["adv_stagnation_min_gate"])
    exp.adv_stagnation_cooldown = int(stagnation_cooldown)

    exp.adaptive_lambda_pde = True
    exp.residual_scale_reference = float(best_params["residual_scale_reference"])
    exp.lambda_pde_growth_exponent = float(best_params["lambda_pde_growth_exponent"])
    exp.lambda_pde_min = float(best_params["lambda_pde_min"])
    exp.lambda_pde_max = float(best_params["lambda_pde_max"])
    exp.lambda_pde_ema_beta = 0.9

    exp.residual_tolerance_target = float(target_residual)
    exp.fdm_tol = 1e-12
    exp.fdm_max_iter = 100000

    exp.analysis_num_samples = 8
    exp.generate_plots = True
    exp.generate_extended_plots = True
    exp.save_frequency = max(50, int(epochs // 40))
    exp.__post_init__()
    return exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa treino final da PI-GAN a partir do melhor trial do Optuna."
    )
    parser.add_argument(
        "--best-trial",
        type=str,
        default="runs_optuna_physics_first_60t/best_trial.json",
        help="Caminho para best_trial.json gerado pelo Optuna.",
    )
    parser.add_argument("--runs-dir", type=str, default="runs_final_4000_best")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--steps-per-epoch", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-residual", type=float, default=1e-3)
    parser.add_argument("--tuning-epochs", type=int, default=160)
    parser.add_argument(
        "--search-steps-per-epoch",
        type=int,
        default=0,
        help=(
            "steps_per_epoch usado na busca Optuna. Use 0 quando desconhecido; "
            "o script registra a incerteza e bloqueia mudancas sem --allow-steps-mismatch."
        ),
    )
    parser.add_argument(
        "--allow-steps-mismatch",
        action="store_true",
        help="Permite treino final com steps_per_epoch diferente da busca documentada.",
    )
    parser.add_argument("--no-scale-schedules", action="store_true")
    parser.add_argument(
        "--physics-refine-steps",
        type=int,
        default=0,
        help="Passos de refinamento fisico final. Default 0 para preservar atribuicao ao treino GAN.",
    )
    parser.add_argument(
        "--physics-refine-lr",
        type=float,
        default=2.0e-5,
        help="Learning rate do refinamento fisico, usado apenas se --physics-refine-steps > 0.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--no-strict-checkpoint", action="store_true")
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Multiplica gen_lr/disc_lr ao montar a configuracao efetiva.",
    )
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="Carrega pesos do checkpoint, mas reinicia otimizadores com os LRs do best_trial.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")
    best_path = Path(args.best_trial)
    if not best_path.is_absolute():
        best_path = PROJECT_ROOT / best_path

    best_data = _read_best_trial(best_path)
    study_summary = _read_study_summary_for_best_trial(best_path)
    best_params = dict(best_data["best_params"])

    search_steps_per_epoch = int(args.search_steps_per_epoch)
    if search_steps_per_epoch < 0:
        raise ValueError("--search-steps-per-epoch deve ser >= 0.")
    if search_steps_per_epoch > 0 and int(args.steps_per_epoch) != search_steps_per_epoch:
        if not bool(args.allow_steps_mismatch):
            raise ValueError(
                "steps_per_epoch do treino final difere da busca Optuna "
                f"({int(args.steps_per_epoch)} != {search_steps_per_epoch}). "
                "Use --allow-steps-mismatch para registrar uma mudanca intencional."
            )

    exp = _build_experiment_config(
        best_params=best_params,
        epochs=int(args.epochs),
        steps_per_epoch=int(args.steps_per_epoch),
        grid_size=int(args.grid_size),
        batch_size=int(args.batch_size),
        target_residual=float(args.target_residual),
        tuning_epochs=int(args.tuning_epochs),
        scale_schedules=not bool(args.no_scale_schedules),
    )
    if bool(args.no_plots):
        exp.generate_plots = False
        exp.generate_extended_plots = False
    physics_refine_steps = max(0, int(args.physics_refine_steps))
    exp.physics_refine_enable = physics_refine_steps > 0
    exp.physics_refine_steps = physics_refine_steps
    exp.physics_refine_lr = float(args.physics_refine_lr)
    if exp.physics_refine_enable and exp.physics_refine_lr <= 0.0:
        raise ValueError("--physics-refine-lr deve ser > 0 quando o refinamento esta ativo.")
    if str(args.resume_checkpoint).strip():
        exp.resume_checkpoint = str(args.resume_checkpoint).strip()
    if bool(args.no_strict_checkpoint):
        exp.strict_checkpoint_loading = False
    if bool(args.reset_optimizer_on_resume):
        exp.load_checkpoint_optimizer_state = False
        exp.restore_checkpoint_rng_state = False

    lr_scale = float(args.lr_scale)
    if lr_scale <= 0.0:
        raise ValueError("--lr-scale deve ser > 0.")
    exp.gen_lr *= lr_scale
    exp.disc_lr *= lr_scale
    if exp.disc_lr_d1 is not None:
        exp.disc_lr_d1 *= lr_scale
    if exp.disc_lr_d2 is not None:
        exp.disc_lr_d2 *= lr_scale

    use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    sys_cfg = SystemConfig(
        use_gpu=use_gpu,
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
        log_file=None,
        seed=int(args.seed),
        use_double=True,
    )

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "results").mkdir(parents=True, exist_ok=True)
    (runs_dir / "results" / "final_4000_effective_config.json").write_text(
        json.dumps(
            {
                "source_best_trial": str(best_path),
                "study_name": str(best_data.get("study_name", "")),
                "best_value": best_data.get("best_value"),
                "study_summary_available": bool(study_summary),
                "best_params": best_params,
                "epochs": int(exp.epochs),
                "steps_per_epoch": int(exp.steps_per_epoch),
                "search_steps_per_epoch_reference": (
                    int(search_steps_per_epoch) if search_steps_per_epoch > 0 else None
                ),
                "steps_per_epoch_matches_search": (
                    bool(int(exp.steps_per_epoch) == search_steps_per_epoch)
                    if search_steps_per_epoch > 0
                    else None
                ),
                "steps_mismatch_allowed": bool(args.allow_steps_mismatch),
                "grid_size_x": int(exp.grid_size_x),
                "grid_size_y": int(exp.grid_size_y),
                "batch_size": int(exp.batch_size),
                "gen_lr": float(exp.gen_lr),
                "disc_lr": float(exp.disc_lr),
                "disc_lr_d1": float(exp.disc_lr_d1 or exp.disc_lr),
                "disc_lr_d2": float(exp.disc_lr_d2 or exp.disc_lr),
                "lr_scale": float(lr_scale),
                "lambda_adv1": float(exp.lambda_adv1),
                "lambda_adv2": float(exp.lambda_adv2),
                "lambda_pde": float(exp.lambda_pde),
                "lambda_bc": float(exp.lambda_bc),
                "n_critic": int(exp.n_critic),
                "disc_update_every": int(exp.disc_update_every),
                "target_adv_over_pde": float(exp.target_adv_over_pde),
                "gradnorm_target_adv_to_pde": float(exp.gradnorm_target_adv_to_pde),
                "adv_warmup_epochs": int(exp.adv_warmup_epochs),
                "adv_residual_gate_target": float(exp.adv_residual_gate_target),
                "residual_scale_reference": float(exp.residual_scale_reference),
                "lambda_pde_growth_exponent": float(exp.lambda_pde_growth_exponent),
                "lambda_pde_min": float(exp.lambda_pde_min),
                "lambda_pde_max": float(exp.lambda_pde_max),
                "plateau_patience": int(exp.plateau_patience),
                "plateau_factor": float(exp.plateau_factor),
                "plateau_min_delta": float(exp.plateau_min_delta),
                "plateau_cooldown": int(exp.plateau_cooldown),
                "target_residual": float(exp.residual_tolerance_target),
                "physics_refine_enable": bool(exp.physics_refine_enable),
                "physics_refine_steps": int(exp.physics_refine_steps),
                "physics_refine_lr": float(exp.physics_refine_lr),
                "physics_refine_step_ratio_vs_train_updates": (
                    float(exp.physics_refine_steps)
                    / float(max(1, int(exp.epochs) * int(exp.steps_per_epoch)))
                ),
                "scale_schedules": not bool(args.no_scale_schedules),
                "tuning_epochs_reference": int(args.tuning_epochs),
                "resume_checkpoint": str(getattr(exp, "resume_checkpoint", "")),
                "strict_checkpoint_loading": bool(getattr(exp, "strict_checkpoint_loading", True)),
                "load_checkpoint_optimizer_state": bool(
                    getattr(exp, "load_checkpoint_optimizer_state", True)
                ),
                "restore_checkpoint_rng_state": bool(
                    getattr(exp, "restore_checkpoint_rng_state", True)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[run] best_trial={best_path}")
    print(f"[run] runs_dir={runs_dir}")
    print(
        f"[run] epochs={exp.epochs} steps_per_epoch={exp.steps_per_epoch} "
        f"grid={exp.grid_size_x}x{exp.grid_size_y} batch={exp.batch_size}"
    )
    if search_steps_per_epoch > 0:
        print(f"[run] search_steps_per_epoch={search_steps_per_epoch}")
    else:
        print("[run] search_steps_per_epoch=unknown")
    print(
        f"[run] physics_refine_enable={int(exp.physics_refine_enable)} "
        f"physics_refine_steps={exp.physics_refine_steps}"
    )
    print(
        f"[run] lr_g={exp.gen_lr:.3e} lr_d1={float(exp.disc_lr_d1 or exp.disc_lr):.3e} "
        f"lr_d2={float(exp.disc_lr_d2 or exp.disc_lr):.3e}"
    )

    pipeline = PIGANPipeline(experiment_config=exp, system_config=sys_cfg, runs_dir=runs_dir)
    _, _, metrics, _ = pipeline.run()
    print("[feito] métricas finais")
    print(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
