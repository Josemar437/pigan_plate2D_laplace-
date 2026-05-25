#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ajuste com Optuna focado na dinÃ¢mica de controle da PI-GAN.

Esta busca explora os controles de estabilizaÃ§Ã£o do trainer:
- histerese do gate por resÃ­duo;
- histerese de pausa/retomada dos crÃ­ticos;
- reforÃ§o adversarial por estagnaÃ§Ã£o.

O objetivo prioriza qualidade fÃ­sica do resÃ­duo e penaliza inatividade
adversarial (crÃ­ticos pausados em excesso ou gate muito fechado).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Optuna nÃ£o encontrado. Instale com: pip install optuna") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, SystemConfig  # noqa: E402
from src.pipeline import PIGANPipeline  # noqa: E402
from scripts.optunaCommon import (  # noqa: E402
    best_train_residual as _best_train_residual,
    round_hyperparams as _round_hyperparams,
    study_summary_payload as _study_summary_payload,
    tail_mean_abs as _tail_mean_abs,
    write_json as _write_json,
)


def _safe_float(x: Any, default: float = float("inf")) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(val):
        return float(default)
    return val


def _parse_cli_scalar(raw: str) -> Any:
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(c in text for c in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        try:
            return float(text)
        except Exception:
            return text


def _load_json_object(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON invalido em {path}: esperado objeto.")
    return payload


def _load_fixed_params(args: argparse.Namespace) -> Dict[str, Any]:
    fixed: Dict[str, Any] = {}

    best_trial_path = str(getattr(args, "fixed_from_best_trial", "")).strip()
    if best_trial_path:
        payload = _load_json_object(Path(best_trial_path))
        best_params = payload.get("best_params", payload)
        if not isinstance(best_params, dict):
            raise ValueError("best_trial.json sem campo best_params valido.")
        fixed.update(best_params)

    fixed_params_file = str(getattr(args, "fixed_params_file", "")).strip()
    if fixed_params_file:
        payload = _load_json_object(Path(fixed_params_file))
        fixed.update(payload)

    fixed_param_items = list(getattr(args, "fixed_param", []) or [])
    for item in fixed_param_items:
        token = str(item).strip()
        if "=" not in token:
            raise ValueError(
                f"Parametro --fixed-param invalido: '{token}'. Use o formato chave=valor."
            )
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Parametro --fixed-param invalido: '{token}'.")
        fixed[key] = _parse_cli_scalar(raw_value)

    return fixed


def _fixed_or_none(
    trial: optuna.Trial,
    name: str,
    fixed_params: Dict[str, Any],
) -> Optional[Any]:
    if name not in fixed_params:
        return None
    return trial.suggest_categorical(name, [fixed_params[name]])


def _suggest_float(
    trial: optuna.Trial,
    name: str,
    low: float,
    high: float,
    *,
    fixed_params: Dict[str, Any],
    log: bool = False,
) -> float:
    fixed = _fixed_or_none(trial, name, fixed_params)
    if fixed is not None:
        return float(fixed)
    return float(trial.suggest_float(name, low, high, log=log))


def _suggest_int(
    trial: optuna.Trial,
    name: str,
    low: int,
    high: int,
    *,
    fixed_params: Dict[str, Any],
) -> int:
    fixed = _fixed_or_none(trial, name, fixed_params)
    if fixed is not None:
        return int(fixed)
    return int(trial.suggest_int(name, low, high))


def _suggest_categorical(
    trial: optuna.Trial,
    name: str,
    choices: Sequence[Any],
    *,
    fixed_params: Dict[str, Any],
) -> Any:
    fixed = _fixed_or_none(trial, name, fixed_params)
    if fixed is not None:
        return fixed
    return trial.suggest_categorical(name, list(choices))


def _score_trial(
    *,
    metrics: Dict[str, float],
    history: List[Dict[str, float]],
    health: Dict[str, float],
    max_paused_ratio: float,
    max_reduced_ratio: float,
    min_adv_gate_end: float,
    min_adv_gate_open_ratio: float,
) -> float:
    """
    Minimiza uma pontuaÃ§Ã£o fÃ­sica com penalizaÃ§Ã£o de inatividade adversarial.

    Quanto menor, melhor.
    """
    pde_mean = _safe_float(metrics.get("pde_residual_mean"))
    pde_max = _safe_float(metrics.get("pde_residual_max"))
    rel_l2 = _safe_float(metrics.get("relative_l2_error"))
    rmse = _safe_float(metrics.get("rmse"))
    if not np.isfinite(pde_mean):
        return float("inf")

    best_train_res, _ = _best_train_residual(history)
    final_train_res = _safe_float(
        history[-1].get("g_residual_mean_abs") if history else float("inf")
    )
    plateau_drift = max(0.0, final_train_res - best_train_res)

    paused_ratio = float(np.clip(_safe_float(health.get("critics_paused_ratio"), 1.0), 0.0, 1.0))
    reduced_ratio = float(np.clip(_safe_float(health.get("critics_reduced_ratio"), 1.0), 0.0, 1.0))
    adv_gate_end = float(np.clip(_safe_float(health.get("adv_gate_end"), 0.0), 0.0, 1.0))
    adv_gate_open_ratio = float(np.clip(_safe_float(health.get("adv_gate_open_ratio"), 0.0), 0.0, 1.0))
    d1_active = float(np.clip(_safe_float(health.get("d1_nonzero_ratio"), 0.0), 0.0, 1.0))
    d2_active = float(np.clip(_safe_float(health.get("d2_nonzero_ratio"), 0.0), 0.0, 1.0))

    # Base da pontuaÃ§Ã£o: qualidade fÃ­sica.
    score = 1.00 * pde_mean
    score += 0.05 * pde_max
    score += 2.0e3 * rel_l2
    score += 5.0e1 * rmse
    score += 0.30 * plateau_drift

    # Penaliza crÃ­ticos inativos e baixa participaÃ§Ã£o adversarial.
    if paused_ratio > max_paused_ratio:
        score *= 1.0 + 2.5 * (paused_ratio - max_paused_ratio)
    if reduced_ratio > max_reduced_ratio:
        score *= 1.0 + 1.5 * (reduced_ratio - max_reduced_ratio)
    if adv_gate_end < min_adv_gate_end:
        score *= 1.0 + 2.0 * (min_adv_gate_end - adv_gate_end)
    if adv_gate_open_ratio < min_adv_gate_open_ratio:
        score *= 1.0 + 1.5 * (min_adv_gate_open_ratio - adv_gate_open_ratio)
    if d1_active < 0.7:
        score *= 1.0 + 0.5 * (0.7 - d1_active)
    if d2_active < 0.7:
        score *= 1.0 + 0.5 * (0.7 - d2_active)

    return float(score)


def _build_experiment_config(trial: optuna.Trial, args: argparse.Namespace) -> ExperimentConfig:
    exp = ExperimentConfig(generator_mode=str(args.generator_mode))
    focus_final_refine = bool(getattr(args, "focus_final_refine", False))
    fixed_params = dict(getattr(args, "fixed_params", {}) or {})

    # Base execution setup.
    exp.epochs = int(args.epochs)
    exp.steps_per_epoch = int(args.steps_per_epoch)
    exp.batch_size = int(args.batch_size)
    exp.grid_size_x = int(args.grid_size)
    exp.grid_size_y = int(args.grid_size)
    exp.analysis_num_samples = int(args.analysis_num_samples)
    exp.generate_plots = False
    exp.generate_extended_plots = False
    exp.save_frequency = 0

    # Keep architecture stable while tuning control dynamics.
    exp.latent_dim = int(args.latent_dim)
    exp.generator_base_channels = int(args.generator_base_channels)
    exp.generator_depth = int(args.generator_depth)
    exp.discriminator_base_channels = int(args.discriminator_base_channels)
    exp.use_physical_coordinates = True
    exp.use_reference_discriminator = True

    # Optimizer region.
    exp.gen_lr = _suggest_float(
        trial, "gen_lr", 8e-5, 2.5e-4, fixed_params=fixed_params, log=True
    )
    disc_ratio = _suggest_float(
        trial, "disc_lr_ratio", 0.4, 1.1, fixed_params=fixed_params
    )
    exp.disc_lr = float(exp.gen_lr * disc_ratio)
    exp.disc_lr_d1 = exp.disc_lr
    exp.disc_lr_d2 = exp.disc_lr
    exp.max_grad_norm = _suggest_float(
        trial, "max_grad_norm", 0.5, 2.0, fixed_params=fixed_params
    )

    # Adversarial/physics weights.
    exp.lambda_adv1 = _suggest_float(
        trial, "lambda_adv1", 3e-2, 3e-1, fixed_params=fixed_params, log=True
    )
    exp.lambda_adv2 = _suggest_float(
        trial, "lambda_adv2", 8e-2, 8e-1, fixed_params=fixed_params, log=True
    )
    exp.lambda_pde = _suggest_float(
        trial, "lambda_pde", 8.0, 60.0, fixed_params=fixed_params, log=True
    )
    exp.lambda_gp = _suggest_float(
        trial, "lambda_gp", 6.0, 15.0, fixed_params=fixed_params
    )
    exp.lambda_gp_d1 = exp.lambda_gp
    exp.lambda_gp_d2 = exp.lambda_gp
    exp.n_critic = _suggest_int(trial, "n_critic", 1, 3, fixed_params=fixed_params)
    exp.disc_update_every = _suggest_int(
        trial, "disc_update_every", 1, 3, fixed_params=fixed_params
    )

    exp.dynamic_adv_balance = True
    exp.target_adv_over_pde = _suggest_float(
        trial, "target_adv_over_pde", 0.15, 0.90, fixed_params=fixed_params
    )
    exp.adv_scale_min = _suggest_float(
        trial, "adv_scale_min", 0.30, 2.00, fixed_params=fixed_params
    )
    exp.adv_scale_max = _suggest_float(
        trial, "adv_scale_max", 8.0, 80.0, fixed_params=fixed_params, log=True
    )
    exp.gradnorm_balance = True
    exp.gradnorm_target_adv_to_pde = _suggest_float(
        trial,
        "gradnorm_target_adv_to_pde",
        0.10,
        0.90,
        fixed_params=fixed_params,
    )

    # Gate controls.
    exp.adv_warmup_epochs = _suggest_int(
        trial,
        "adv_warmup_epochs",
        max(5, int(args.epochs * 0.05)),
        max(20, int(args.epochs * 0.75)),
        fixed_params=fixed_params,
    )
    gate_target_low = 5e-3 if focus_final_refine else 1e-3
    gate_target_high = 4e-1 if focus_final_refine else 2e-1
    gate_min_low = 0.20 if focus_final_refine else 0.02
    gate_min_high = 0.90 if focus_final_refine else 0.40
    exp.adv_residual_gate_target = _suggest_float(
        trial,
        "adv_residual_gate_target",
        gate_target_low,
        gate_target_high,
        fixed_params=fixed_params,
        log=True,
    )
    exp.adv_residual_gate_min = _suggest_float(
        trial,
        "adv_residual_gate_min",
        gate_min_low,
        gate_min_high,
        fixed_params=fixed_params,
    )
    exp.adv_residual_gate_hysteresis = bool(
        _suggest_categorical(
            trial,
            "adv_residual_gate_hysteresis",
            [True, False],
            fixed_params=fixed_params,
        )
    )
    exp.adv_residual_gate_power = _suggest_float(
        trial, "adv_residual_gate_power", 0.35, 1.40, fixed_params=fixed_params
    )
    off_ratio = _suggest_float(
        trial, "adv_residual_gate_off_ratio", 0.05, 1.0, fixed_params=fixed_params
    )
    exp.adv_residual_gate_off_threshold = float(exp.adv_residual_gate_target * off_ratio)

    # Critic pause hysteresis.
    exp.critic_pause_on_overgap = True
    exp.max_critic_gap = _suggest_float(
        trial, "max_critic_gap", 8.0, 18.0, fixed_params=fixed_params
    )
    exp.critic_pause_gap_factor = _suggest_float(
        trial, "critic_pause_gap_factor", 1.05, 2.60, fixed_params=fixed_params
    )
    resume_ratio = _suggest_float(
        trial, "critic_resume_gap_ratio", 0.10, 1.00, fixed_params=fixed_params
    )
    exp.critic_resume_gap_factor = float(exp.critic_pause_gap_factor * resume_ratio)
    exp.critic_gap_penalty = _suggest_float(
        trial, "critic_gap_penalty", 0.01, 0.20, fixed_params=fixed_params
    )
    exp.critic_drift = _suggest_float(
        trial, "critic_drift", 1e-3, 1.5e-2, fixed_params=fixed_params, log=True
    )

    # Stagnation boost.
    exp.adv_stagnation_boost = True
    exp.adv_stagnation_patience = _suggest_int(
        trial, "adv_stagnation_patience", 8, 60, fixed_params=fixed_params
    )
    exp.adv_stagnation_rel_tol = _suggest_float(
        trial,
        "adv_stagnation_rel_tol",
        5e-5,
        2e-2,
        fixed_params=fixed_params,
        log=True,
    )
    exp.adv_stagnation_boost_factor = _suggest_float(
        trial, "adv_stagnation_boost_factor", 1.05, 1.80, fixed_params=fixed_params
    )
    exp.adv_stagnation_min_gate = _suggest_float(
        trial, "adv_stagnation_min_gate", 0.15, 0.95, fixed_params=fixed_params
    )
    exp.adv_stagnation_cooldown = _suggest_int(
        trial, "adv_stagnation_cooldown", 0, 15, fixed_params=fixed_params
    )

    # Dynamic lambda_pde controls.
    exp.adaptive_lambda_pde = True
    exp.residual_scale_reference = _suggest_float(
        trial,
        "residual_scale_reference",
        5e-4,
        5e-2,
        fixed_params=fixed_params,
        log=True,
    )
    exp.lambda_pde_growth_exponent = _suggest_float(
        trial, "lambda_pde_growth_exponent", 0.30, 1.20, fixed_params=fixed_params
    )
    exp.lambda_pde_min = _suggest_float(
        trial, "lambda_pde_min", 4.0, 20.0, fixed_params=fixed_params
    )
    exp.lambda_pde_max = _suggest_float(
        trial, "lambda_pde_max", 40.0, 220.0, fixed_params=fixed_params
    )

    # Optional late-stage precision refine controls.
    exp.precision_refine_enable = bool(focus_final_refine)
    exp.precision_refine_use_adv_progressive_start = True
    if focus_final_refine:
        exp.precision_refine_start_epoch = _suggest_int(
            trial,
            "precision_refine_start_epoch",
            max(20, int(args.epochs * 0.45)),
            max(40, int(args.epochs * 0.90)),
            fixed_params=fixed_params,
        )
        exp.precision_refine_n_critic = _suggest_int(
            trial, "precision_refine_n_critic", 2, 6, fixed_params=fixed_params
        )
        exp.precision_refine_n_critic_ramp_epochs = _suggest_int(
            trial,
            "precision_refine_n_critic_ramp_epochs",
            max(20, int(args.epochs * 0.10)),
            max(40, int(args.epochs * 0.60)),
            fixed_params=fixed_params,
        )
        exp.precision_refine_lambda_pde_max_scale = _suggest_float(
            trial,
            "precision_refine_lambda_pde_max_scale",
            0.45,
            0.95,
            fixed_params=fixed_params,
        )
    else:
        exp.precision_refine_start_epoch = 0
        exp.precision_refine_n_critic = int(exp.n_critic)
        exp.precision_refine_n_critic_ramp_epochs = max(1, int(args.epochs * 0.25))
        exp.precision_refine_lambda_pde_max_scale = 1.0

    # Numerical setup.
    exp.fdm_tol = float(args.fdm_tol)
    exp.fdm_max_iter = int(args.fdm_max_iter)
    exp.boundary_sine_amplitude = float(args.boundary_sine_amplitude)
    exp.use_wgan_gp = True

    exp.__post_init__()
    return exp


def _build_system_config(*, seed: int, use_gpu: bool) -> SystemConfig:
    return SystemConfig(
        use_gpu=bool(use_gpu),
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
        seed=int(seed),
        log_file=None,
        use_double=True,
        deterministic_run=True,
        deterministic_warn_only=False,
    )


def _trial_payload(
    *,
    trial: optuna.Trial,
    state: str,
    score: Optional[float],
    metrics: Optional[Dict[str, float]],
    health: Optional[Dict[str, float]],
    history: List[Dict[str, float]],
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
        "score": None if score is None else float(score),
        "epochs_recorded": int(len(history)),
        "best_train_residual_mean_abs": float(best_res),
        "best_train_epoch": int(best_epoch),
        "tail_mean_abs_d1_gap": _tail_mean_abs(history, "d1_gap", window=20),
        "tail_mean_abs_d2_gap": _tail_mean_abs(history, "d2_gap", window=20),
    }
    if history:
        payload["final_epoch_metrics"] = {k: float(v) for k, v in history[-1].items()}
    if metrics is not None:
        payload["eval_metrics"] = {k: float(v) for k, v in metrics.items()}
    if health is not None:
        payload["adversarial_health"] = {k: float(v) for k, v in health.items()}
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
        pipeline = PIGANPipeline(exp_cfg, sys_cfg, runs_dir=trial_dir)

        history: List[Dict[str, float]] = []
        metrics: Optional[Dict[str, float]] = None
        health: Optional[Dict[str, float]] = None
        score: Optional[float] = None
        try:
            hx, hy = pipeline._prepare_physics_fields()
            pipeline.generator, pipeline.discriminator = pipeline._create_models()
            pipeline.trainer = pipeline._create_trainer(hx, hy)
            pipeline._baseline_metrics = None

            def _on_epoch(summary: Dict[str, float]) -> None:
                history.append(dict(summary))
                key = str(args.prune_metric)
                value = float(summary.get(key, summary.get("g_residual_mean_abs", float("inf"))))
                trial.report(value, int(summary.get("epoch", 0)))
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"pruned at epoch={int(summary.get('epoch', 0))} {key}={value:.4e}"
                    )

            trained = pipeline.trainer.train(epoch_callback=_on_epoch)
            if trained:
                history = trained

            metrics = pipeline._evaluate()
            health = pipeline._compute_adversarial_health(history)
            score = _score_trial(
                metrics=metrics,
                history=history,
                health=health,
                max_paused_ratio=float(args.max_paused_ratio),
                max_reduced_ratio=float(args.max_reduced_ratio),
                min_adv_gate_end=float(args.min_adv_gate_end),
                min_adv_gate_open_ratio=float(args.min_adv_gate_open_ratio),
            )
            pipeline._save_results(metrics, history)

            best_res, best_epoch = _best_train_residual(history)
            trial.set_user_attr("best_train_residual_mean_abs", float(best_res))
            trial.set_user_attr("best_train_epoch", int(best_epoch))
            trial.set_user_attr(
                "eval_pde_residual_mean",
                float(metrics.get("pde_residual_mean", float("inf"))),
            )
            trial.set_user_attr("eval_rmse", float(metrics.get("rmse", float("inf"))))
            trial.set_user_attr(
                "critics_paused_ratio",
                float(health.get("critics_paused_ratio", 0.0)),
            )
            trial.set_user_attr("adv_gate_end", float(health.get("adv_gate_end", 0.0)))

            _write_json(
                trial_dir / "results" / "optuna_trial_summary.json",
                _trial_payload(
                    trial=trial,
                    state="COMPLETE",
                    score=score,
                    metrics=metrics,
                    health=health,
                    history=history,
                ),
            )
            return float(score)

        except optuna.TrialPruned:
            _write_json(
                trial_dir / "results" / "optuna_trial_summary.json",
                _trial_payload(
                    trial=trial,
                    state="PRUNED",
                    score=None,
                    metrics=metrics,
                    health=health,
                    history=history,
                ),
            )
            raise
        finally:
            if pipeline.device.type == "cuda":
                pipeline.memory_manager.clear_cache()

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajuste com Optuna para hiperparÃ¢metros de controle da PI-GAN."
    )
    parser.add_argument("--output-root", type=str, default="runs_optuna_control")
    parser.add_argument("--study-name", type=str, default="pigan_control_search")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=0, help="seconds; 0 disables timeout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--fixed-from-best-trial",
        type=str,
        default="",
        help="Caminho para best_trial.json; fixa os parametros de best_params.",
    )
    parser.add_argument(
        "--fixed-params-file",
        type=str,
        default="",
        help="JSON com pares chave/valor para fixar parametros de busca.",
    )
    parser.add_argument(
        "--fixed-param",
        action="append",
        default=[],
        help="Fixa parametro individual no formato chave=valor (pode repetir).",
    )

    parser.add_argument(
        "--generator-mode",
        type=str,
        default="stochastic_pigan",
        choices=["stochastic_pigan", "deterministic_adversarial"],
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--steps-per-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--analysis-num-samples", type=int, default=8)
    parser.add_argument("--generator-base-channels", type=int, default=12)
    parser.add_argument("--generator-depth", type=int, default=3)
    parser.add_argument("--discriminator-base-channels", type=int, default=12)
    parser.add_argument("--boundary-sine-amplitude", type=float, default=0.0)
    parser.add_argument("--fdm-tol", type=float, default=1e-12)
    parser.add_argument("--fdm-max-iter", type=int, default=100000)

    parser.add_argument("--max-paused-ratio", type=float, default=0.35)
    parser.add_argument("--max-reduced-ratio", type=float, default=0.70)
    parser.add_argument("--min-adv-gate-end", type=float, default=0.60)
    parser.add_argument("--min-adv-gate-open-ratio", type=float, default=0.55)
    parser.add_argument(
        "--focus-final-refine",
        action="store_true",
        help="Ativa busca focada em fase final: gate mais aberto + precision_refine_*.",
    )

    parser.add_argument("--prune-metric", type=str, default="g_residual_mean_abs")
    parser.add_argument("--pruner-startup-trials", type=int, default=6)
    parser.add_argument("--pruner-warmup-epochs", type=int, default=25)
    parser.add_argument("--pruner-interval", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")
    use_gpu = bool(args.use_gpu and torch.cuda.is_available())

    try:
        args.fixed_params = _load_fixed_params(args)
    except Exception as exc:
        raise SystemExit(f"Erro ao carregar parametros fixos: {exc}") from exc

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.fixed_params:
        _write_json(
            output_root / "fixed_params.json",
            _round_hyperparams(
                dict(args.fixed_params),
                sigfigs=3,
                scientific=True,
            ),
        )

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
    if args.fixed_params:
        fixed_keys = sorted(str(k) for k in args.fixed_params.keys())
        preview = ", ".join(fixed_keys[:8])
        if len(fixed_keys) > 8:
            preview += ", ..."
        print(f"[study] fixed_params={len(fixed_keys)} ({preview})")

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
        print("[feito] melhor trial encontrado")
        print(json.dumps(summary["best_params"], indent=2))
        print(f"[feito] best_value={summary['best_value']:.6e}")
    else:
        print("[feito] nenhum trial concluÃ­do encontrado")


if __name__ == "__main__":
    main()

