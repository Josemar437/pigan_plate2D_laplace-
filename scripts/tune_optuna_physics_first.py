#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Busca com Optuna focada em convergência física estrita da PI-GAN.

O alvo principal de otimização é a magnitude do resíduo bruto da PDE,
com pruning orientado por métrica física por época (`g_residual_mean_abs`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna nao encontrado. Instale com: pip install optuna"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline
from scripts.optuna_common import (
    best_train_residual as _best_train_residual,
    round_hyperparams as _round_hyperparams,
    study_summary_payload as _study_summary_payload,
    tail_mean_abs as _tail_mean_abs,
    tail_std as _tail_std,
    write_json as _write_json,
)


def _score_trial(
    metrics: Dict[str, float],
    history: List[Dict[str, float]],
    *,
    target_residual: float,
) -> float:
    # A componente física domina; o adversarial atua como penalização leve de estabilidade.
    phys = float(metrics.get("pde_residual_mean", float("inf")))
    if not np.isfinite(phys):
        return float("inf")

    d_noise = _tail_std(history, "d1_total") + _tail_std(history, "d2_total")
    gap_mag = _tail_mean_abs(history, "d1_gap") + _tail_mean_abs(history, "d2_gap")
    adv_conflict = _tail_mean_abs(
        [
            {
                "adv_conflict": float(item.get("g_adv1", 0.0) + item.get("g_adv2", 0.0))
            }
            for item in history
        ],
        "adv_conflict",
    )
    if not np.isfinite(d_noise):
        d_noise = 0.0
    if not np.isfinite(gap_mag):
        gap_mag = 0.0
    if not np.isfinite(adv_conflict):
        adv_conflict = 0.0

    score = float(phys)
    score += 2e-2 * float(d_noise)
    score += 5e-3 * float(gap_mag)
    score += 1e-2 * float(adv_conflict)

    target = max(float(target_residual), 1e-12)
    if phys > target:
        score *= 1.0 + math.log10(max(phys / target, 1.0))
    return float(score)


def _build_experiment_config(trial: optuna.Trial, args: argparse.Namespace) -> ExperimentConfig:
    lr_g = trial.suggest_float("lr_g", 1e-5, 5e-4, log=True)
    # Por padrão, mantém LR dos discriminadores até 10x menor que o do gerador.
    lr_d_ratio = trial.suggest_float("lr_d_ratio", 0.01, 0.10, log=True)
    lr_d = lr_g * lr_d_ratio
    lr_d1 = lr_d * trial.suggest_float("lr_d1_ratio", 0.7, 1.0)
    lr_d2 = lr_d * trial.suggest_float("lr_d2_ratio", 0.7, 1.0)

    exp = ExperimentConfig(generator_mode="deterministic_adversarial")
    exp.epochs = int(args.epochs)
    exp.steps_per_epoch = int(args.steps_per_epoch)
    exp.batch_size = int(args.batch_size)
    exp.grid_size_x = int(args.grid_size)
    exp.grid_size_y = int(args.grid_size)
    exp.generator_base_channels = int(args.generator_base_channels)
    exp.generator_depth = int(args.generator_depth)
    exp.generator_zero_init_final = True

    exp.discriminator_base_channels = int(args.discriminator_base_channels)
    exp.discriminator_capacity_scale = trial.suggest_float(
        "discriminator_capacity_scale", 0.50, 0.90
    )
    exp.discriminator_dropout = trial.suggest_float("discriminator_dropout", 0.10, 0.35)
    exp.discriminator_spectral_norm = True

    exp.lambda_adv1 = trial.suggest_float("lambda_adv1", 1e-4, 2e-2, log=True)
    exp.lambda_adv2 = trial.suggest_float("lambda_adv2", 1e-4, 4e-2, log=True)
    exp.lambda_pde = trial.suggest_float("lambda_pde", 10.0, 200.0, log=True)
    exp.lambda_pde_raw = 0.0
    exp.lambda_bc = float(args.lambda_bc)

    exp.gen_lr = float(lr_g)
    exp.disc_lr = float(lr_d)
    exp.disc_lr_d1 = float(lr_d1)
    exp.disc_lr_d2 = float(lr_d2)
    exp.max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.5)

    exp.n_critic = 1
    exp.disc_update_every = trial.suggest_int("disc_update_every", 2, 6)
    exp.target_adv_over_pde = trial.suggest_float("target_adv_over_pde", 5e-4, 5e-2, log=True)
    exp.gradnorm_target_adv_to_pde = trial.suggest_float(
        "gradnorm_target_adv_to_pde", 2e-2, 3e-1, log=True
    )
    exp.adv_warmup_epochs = trial.suggest_int(
        "adv_warmup_epochs",
        max(50, int(args.epochs // 2)),
        max(200, int(2 * args.epochs)),
    )
    exp.adv_residual_gate_target = trial.suggest_float(
        "adv_residual_gate_target", 5e-3, 2.0, log=True
    )
    exp.adv_residual_gate_min = 0.01

    exp.adaptive_lambda_pde = True
    exp.residual_scale_reference = trial.suggest_float(
        "residual_scale_reference", 1e-3, 1e-1, log=True
    )
    exp.lambda_pde_growth_exponent = trial.suggest_float(
        "lambda_pde_growth_exponent", 0.3, 1.2
    )
    exp.lambda_pde_min = trial.suggest_float("lambda_pde_min", 5.0, 40.0)
    exp.lambda_pde_max = trial.suggest_float("lambda_pde_max", 100.0, 400.0, log=True)
    exp.lambda_pde_ema_beta = 0.9

    exp.divergence_window = 16
    exp.divergence_ratio_threshold = 1.2
    exp.divergence_patience = 2
    exp.lr_drop_factor = 0.5
    exp.max_lr_drops = 8

    exp.plateau_scheduler_enabled = True
    exp.plateau_metric_key = "g_residual_mean_abs"
    exp.plateau_mode = "min"
    exp.plateau_patience = trial.suggest_int("plateau_patience", 8, 30)
    exp.plateau_factor = trial.suggest_float("plateau_factor", 0.3, 0.7)
    exp.plateau_min_delta = trial.suggest_float("plateau_min_delta", 1e-8, 1e-4, log=True)
    exp.plateau_cooldown = trial.suggest_int("plateau_cooldown", 4, 12)
    exp.plateau_max_drops = 8
    exp.plateau_reduce_discriminators = False

    exp.residual_tolerance_target = float(args.target_residual)
    exp.fdm_tol = float(args.fdm_tol)
    exp.fdm_max_iter = int(args.fdm_max_iter)

    exp.analysis_num_samples = 1
    exp.generate_plots = False
    exp.generate_extended_plots = False
    exp.save_frequency = 0
    exp.__post_init__()
    return exp


def _build_system_config(*, seed: int, use_gpu: bool) -> SystemConfig:
    return SystemConfig(
        use_gpu=bool(use_gpu),
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
        log_file=None,
        seed=int(seed),
        use_double=True,
    )


def _trial_summary(
    *,
    trial: optuna.Trial,
    state: str,
    history: List[Dict[str, float]],
    metrics: Optional[Dict[str, float]],
    score: Optional[float],
    target_residual: float,
    pruned: bool,
) -> Dict[str, Any]:
    best_res, best_epoch = _best_train_residual(history)
    payload: Dict[str, Any] = {
        "trial_number": int(trial.number),
        "state": str(state),
        "params": _round_hyperparams(
            dict(trial.params),
            sigfigs=3,
            scientific=True,
        ),
        "pruned": bool(pruned),
        "score": None if score is None else float(score),
        "epochs_recorded": int(len(history)),
        "best_train_residual_mean_abs": float(best_res),
        "best_train_epoch": int(best_epoch),
        "target_residual": float(target_residual),
        "target_reached_in_train": bool(best_res <= float(target_residual)),
        "tail_std_d1_total": _tail_std(history, "d1_total"),
        "tail_std_d2_total": _tail_std(history, "d2_total"),
        "tail_mean_abs_d1_gap": _tail_mean_abs(history, "d1_gap"),
        "tail_mean_abs_d2_gap": _tail_mean_abs(history, "d2_gap"),
    }
    if history:
        payload["final_epoch_metrics"] = {k: float(v) for k, v in history[-1].items()}
    if metrics is not None:
        payload["eval_metrics"] = {k: float(v) for k, v in metrics.items()}
        payload["target_reached_in_eval"] = bool(
            float(metrics.get("pde_residual_mean", float("inf"))) <= float(target_residual)
        )
    return payload


def build_objective(
    *,
    args: argparse.Namespace,
    output_root: Path,
    use_gpu: bool,
) -> "callable[[optuna.Trial], float]":
    trials_dir = output_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        exp_cfg = _build_experiment_config(trial, args)
        sys_cfg = _build_system_config(seed=int(args.seed + trial.number), use_gpu=use_gpu)
        pipeline = PIGANPipeline(experiment_config=exp_cfg, system_config=sys_cfg, runs_dir=trial_dir)

        epoch_history: List[Dict[str, float]] = []
        metrics: Optional[Dict[str, float]] = None
        score: Optional[float] = None
        try:
            hx, hy = pipeline._prepare_physics_fields()
            pipeline.generator, pipeline.discriminator = pipeline._create_models()
            pipeline.trainer = pipeline._create_trainer(hx, hy)

            def _on_epoch(summary: Dict[str, float]) -> None:
                epoch_history.append(dict(summary))
                prune_key = str(args.prune_metric)
                value = float(summary.get(prune_key, summary.get("g_residual_mean_abs", float("inf"))))
                trial.report(value, int(summary.get("epoch", 0)))
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned at epoch={int(summary.get('epoch', 0))} {prune_key}={value:.4e}"
                    )

            history = pipeline.trainer.train(epoch_callback=_on_epoch)
            if history:
                epoch_history = history
            metrics = pipeline._evaluate()
            score = _score_trial(metrics, epoch_history, target_residual=float(args.target_residual))

            best_train_res, best_epoch = _best_train_residual(epoch_history)
            trial.set_user_attr("best_train_residual_mean_abs", float(best_train_res))
            trial.set_user_attr("best_train_epoch", int(best_epoch))
            trial.set_user_attr("eval_pde_residual_mean", float(metrics.get("pde_residual_mean", float("inf"))))
            trial.set_user_attr("eval_pde_residual_max", float(metrics.get("pde_residual_max", float("inf"))))
            trial.set_user_attr(
                "target_reached_in_eval",
                bool(float(metrics.get("pde_residual_mean", float("inf"))) <= float(args.target_residual)),
            )

            payload = _trial_summary(
                trial=trial,
                state="COMPLETE",
                history=epoch_history,
                metrics=metrics,
                score=score,
                target_residual=float(args.target_residual),
                pruned=False,
            )
            _write_json(trial_dir / "results" / "optuna_trial_summary.json", payload)
            return float(score)

        except optuna.TrialPruned:
            payload = _trial_summary(
                trial=trial,
                state="PRUNED",
                history=epoch_history,
                metrics=metrics,
                score=None,
                target_residual=float(args.target_residual),
                pruned=True,
            )
            _write_json(trial_dir / "results" / "optuna_trial_summary.json", payload)
            raise
        finally:
            if pipeline.device.type == "cuda":
                pipeline.memory_manager.clear_cache()

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajuste com Optuna com objetivo físico prioritário.")
    parser.add_argument("--output-root", type=str, default="runs_optuna_physics_first")
    parser.add_argument("--study-name", type=str, default="pigan_physics_first")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=0, help="Segundos. 0 desativa timeout.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--show-progress", action="store_true")

    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--steps-per-epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--generator-base-channels", type=int, default=12)
    parser.add_argument("--generator-depth", type=int, default=3)
    parser.add_argument("--discriminator-base-channels", type=int, default=10)
    parser.add_argument("--lambda-bc", type=float, default=20.0)
    parser.add_argument("--target-residual", type=float, default=1e-3)
    parser.add_argument("--fdm-tol", type=float, default=1e-12)
    parser.add_argument("--fdm-max-iter", type=int, default=100000)

    parser.add_argument("--prune-metric", type=str, default="g_residual_mean_abs")
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=20)
    parser.add_argument("--pruner-interval", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")
    use_gpu = bool(args.use_gpu and torch.cuda.is_available())

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(0, int(args.pruner_startup_trials)),
        n_warmup_steps=max(0, int(args.pruner_warmup_epochs)),
        interval_steps=max(1, int(args.pruner_interval)),
    )
    storage = str(args.storage).strip() or None
    study = optuna.create_study(
        study_name=str(args.study_name),
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=storage is not None,
    )

    print(f"[env] python={sys.executable}")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} use_gpu={use_gpu}")
    print(
        f"[study] name={study.study_name} trials={int(args.trials)} epochs={int(args.epochs)} "
        f"steps_per_epoch={int(args.steps_per_epoch)} grid={int(args.grid_size)}"
    )

    objective = build_objective(args=args, output_root=output_root, use_gpu=use_gpu)
    timeout = None if int(args.timeout) <= 0 else int(args.timeout)
    study.optimize(
        objective,
        n_trials=max(1, int(args.trials)),
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=bool(args.show_progress),
    )

    summary = _study_summary_payload(
        study,
        top_k=int(args.top_k),
        params_sigfigs=3,
        params_scientific=True,
    )
    _write_json(output_root / "study_summary.json", summary)

    if summary.get("best_params") is not None:
        _write_json(
            output_root / "best_trial.json",
            {
                "study_name": study.study_name,
                "best_value": summary["best_value"],
                "best_params": summary["best_params"],
                "best_user_attrs": summary.get("best_user_attrs", {}),
            },
        )
        print("[feito] melhor trial")
        print(json.dumps(summary["best_params"], indent=2))
        print(f"[feito] best_value={summary['best_value']:.6e}")
    else:
        print("[feito] nenhum trial concluído encontrado")


if __name__ == "__main__":
    main()
