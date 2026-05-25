#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Varredura ativa de hiperparâmetros da PI-GAN no pipeline atual.

Este script executa uma busca em duas fases diretamente sobre:
main.py -> PIGANPipeline -> FieldPIGANTrainer

Fase 1:
- ranqueamento inicial com treinos curtos;
- conjunto fixo de candidatos em regiões estáveis de WGAN-GP + PDE.

Fase 2:
- execuções de confirmação com treinos mais longos para baseline + top-k da fase 1.

Saídas:
- summary_phase1.json
- summary_phase2.json
- best_config.json
- tuning_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig, SystemConfig  # noqa: E402
from src.pipeline import PIGANPipeline  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    lambda_adv1: float
    lambda_adv2: float
    lambda_gp: float
    n_critic: int
    disc_lr: float
    critic_drift: float
    target_adv_over_pde: float
    max_critic_gap: float
    critic_gap_penalty: float
    d2_pair_noise_std: float


def build_candidates() -> List[Candidate]:
    # Baseline replica a configuração ativa atual.
    baseline = Candidate(
        name="baseline_active",
        lambda_adv1=0.15,
        lambda_adv2=0.35,
        lambda_gp=10.0,
        n_critic=2,
        disc_lr=1e-4,
        critic_drift=5e-3,
        target_adv_over_pde=0.25,
        max_critic_gap=8.0,
        critic_gap_penalty=0.05,
        d2_pair_noise_std=5e-3,
    )

    # Busca em vizinhança curada de equilíbrio estável entre adversarial e PDE.
    candidates = [
        baseline,
        Candidate(
            name="gp8_target020",
            lambda_adv1=0.15,
            lambda_adv2=0.35,
            lambda_gp=8.0,
            n_critic=2,
            disc_lr=1e-4,
            critic_drift=5e-3,
            target_adv_over_pde=0.20,
            max_critic_gap=8.0,
            critic_gap_penalty=0.05,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="gp12_target030",
            lambda_adv1=0.15,
            lambda_adv2=0.35,
            lambda_gp=12.0,
            n_critic=2,
            disc_lr=1e-4,
            critic_drift=5e-3,
            target_adv_over_pde=0.30,
            max_critic_gap=8.0,
            critic_gap_penalty=0.05,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="adv_low",
            lambda_adv1=0.10,
            lambda_adv2=0.25,
            lambda_gp=10.0,
            n_critic=2,
            disc_lr=1e-4,
            critic_drift=5e-3,
            target_adv_over_pde=0.22,
            max_critic_gap=8.0,
            critic_gap_penalty=0.05,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="adv_high",
            lambda_adv1=0.20,
            lambda_adv2=0.45,
            lambda_gp=10.0,
            n_critic=2,
            disc_lr=1e-4,
            critic_drift=5e-3,
            target_adv_over_pde=0.28,
            max_critic_gap=8.0,
            critic_gap_penalty=0.05,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="ncritic1_soft",
            lambda_adv1=0.15,
            lambda_adv2=0.35,
            lambda_gp=10.0,
            n_critic=1,
            disc_lr=1e-4,
            critic_drift=7e-3,
            target_adv_over_pde=0.25,
            max_critic_gap=8.0,
            critic_gap_penalty=0.06,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="disc_lr_low",
            lambda_adv1=0.15,
            lambda_adv2=0.35,
            lambda_gp=10.0,
            n_critic=2,
            disc_lr=7e-5,
            critic_drift=7e-3,
            target_adv_over_pde=0.25,
            max_critic_gap=8.0,
            critic_gap_penalty=0.06,
            d2_pair_noise_std=6e-3,
        ),
        Candidate(
            name="disc_lr_high_gap_guard",
            lambda_adv1=0.15,
            lambda_adv2=0.35,
            lambda_gp=10.0,
            n_critic=2,
            disc_lr=1.2e-4,
            critic_drift=8e-3,
            target_adv_over_pde=0.25,
            max_critic_gap=6.0,
            critic_gap_penalty=0.08,
            d2_pair_noise_std=4e-3,
        ),
        Candidate(
            name="gp15_strong_guard",
            lambda_adv1=0.12,
            lambda_adv2=0.30,
            lambda_gp=15.0,
            n_critic=2,
            disc_lr=9e-5,
            critic_drift=1e-2,
            target_adv_over_pde=0.24,
            max_critic_gap=6.0,
            critic_gap_penalty=0.10,
            d2_pair_noise_std=5e-3,
        ),
        Candidate(
            name="gp6_light_guard",
            lambda_adv1=0.16,
            lambda_adv2=0.40,
            lambda_gp=6.0,
            n_critic=2,
            disc_lr=9e-5,
            critic_drift=8e-3,
            target_adv_over_pde=0.26,
            max_critic_gap=7.0,
            critic_gap_penalty=0.07,
            d2_pair_noise_std=5e-3,
        ),
    ]
    return candidates


def _history_series(history: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(item.get(key, np.nan)) for item in history], dtype=float)


def _safe_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _safe_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def _safe_slope(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    x = np.arange(finite.size, dtype=float)
    coeffs = np.polyfit(x, finite, deg=1)
    return float(coeffs[0])


def summarize_history(history: List[Dict[str, float]], window: int = 20) -> Dict[str, float]:
    if not history:
        return {"epochs_recorded": 0.0}

    win = max(2, min(int(window), len(history)))
    d1_gap = _history_series(history, "d1_gap")
    d2_gap = _history_series(history, "d2_gap")
    d1_total = _history_series(history, "d1_total")
    d2_total = _history_series(history, "d2_total")
    d1_gp = _history_series(history, "d1_gp")
    d2_gp = _history_series(history, "d2_gp")
    g_total = _history_series(history, "g_total")
    g_pde = _history_series(history, "g_pde")
    g_adv_over_pde = _history_series(history, "g_adv_over_pde")
    g_res_l2 = _history_series(history, "g_residual_l2")

    tail = slice(-win, None)
    return {
        "epochs_recorded": float(len(history)),
        "g_total_last": float(g_total[-1]),
        "g_pde_last": float(g_pde[-1]),
        "g_total_tail_mean": _safe_mean(g_total[tail]),
        "g_pde_tail_mean": _safe_mean(g_pde[tail]),
        "g_adv_over_pde_tail_mean": _safe_mean(g_adv_over_pde[tail]),
        "g_adv_over_pde_tail_std": _safe_std(g_adv_over_pde[tail]),
        "g_residual_l2_tail_mean": _safe_mean(g_res_l2[tail]),
        "d1_total_tail_mean": _safe_mean(d1_total[tail]),
        "d2_total_tail_mean": _safe_mean(d2_total[tail]),
        "d1_gap_tail_mean": _safe_mean(d1_gap[tail]),
        "d1_gap_tail_std": _safe_std(d1_gap[tail]),
        "d2_gap_tail_mean": _safe_mean(d2_gap[tail]),
        "d2_gap_tail_std": _safe_std(d2_gap[tail]),
        "d1_gp_tail_mean": _safe_mean(d1_gp[tail]),
        "d2_gp_tail_mean": _safe_mean(d2_gp[tail]),
        "d1_total_tail_slope": _safe_slope(d1_total[tail]),
        "g_total_tail_slope": _safe_slope(g_total[tail]),
    }


def build_experiment_config(
    candidate: Candidate,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    generate_plots: bool,
    run_fast_profile: bool,
) -> ExperimentConfig:
    exp = ExperimentConfig()
    exp.epochs = int(epochs)
    exp.steps_per_epoch = 1
    exp.batch_size = int(batch_size)
    exp.analysis_num_samples = 16
    exp.save_frequency = 0
    exp.seed = int(seed) if hasattr(exp, "seed") else seed

    if run_fast_profile:
        exp.grid_size_x = 48
        exp.grid_size_y = 48
        exp.generator_base_channels = 12
        exp.discriminator_base_channels = 12
    else:
        exp.grid_size_x = 64
        exp.grid_size_y = 64
        exp.generator_base_channels = 16
        exp.discriminator_base_channels = 16
    exp.generator_depth = 3

    exp.lambda_adv1 = float(candidate.lambda_adv1)
    exp.lambda_adv2 = float(candidate.lambda_adv2)
    exp.lambda_gp = float(candidate.lambda_gp)
    exp.n_critic = int(candidate.n_critic)
    exp.disc_lr = float(candidate.disc_lr)
    exp.critic_drift = float(candidate.critic_drift)
    exp.target_adv_over_pde = float(candidate.target_adv_over_pde)
    exp.max_critic_gap = float(candidate.max_critic_gap)
    exp.critic_gap_penalty = float(candidate.critic_gap_penalty)
    exp.d2_pair_noise_std = float(candidate.d2_pair_noise_std)

    exp.generate_plots = bool(generate_plots)
    exp.generate_extended_plots = bool(generate_plots)
    exp.__post_init__()
    return exp


def build_system_config(*, seed: int, use_gpu: bool) -> SystemConfig:
    return SystemConfig(
        use_gpu=bool(use_gpu),
        mixed_precision=bool(use_gpu),
        compile_model=False,
        dynamic_batch_size=False,
        log_file=None,
        seed=int(seed),
    )


def run_candidate(
    candidate: Candidate,
    *,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    use_gpu: bool,
    run_fast_profile: bool,
) -> Dict[str, Any]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = build_experiment_config(
        candidate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        generate_plots=False,
        run_fast_profile=run_fast_profile,
    )
    sys_cfg = build_system_config(seed=seed, use_gpu=use_gpu)

    pipeline = PIGANPipeline(experiment_config=exp_cfg, system_config=sys_cfg, runs_dir=run_dir)
    _, _, metrics, history = pipeline.run()

    return {
        "candidate": asdict(candidate),
        "run_dir": str(run_dir),
        "epochs": int(len(history)),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "history_summary": summarize_history(history),
        "profile": {
            "batch_size": int(exp_cfg.batch_size),
            "grid_size_x": int(exp_cfg.grid_size_x),
            "grid_size_y": int(exp_cfg.grid_size_y),
            "generator_base_channels": int(exp_cfg.generator_base_channels),
            "discriminator_base_channels": int(exp_cfg.discriminator_base_channels),
            "generator_depth": int(exp_cfg.generator_depth),
            "use_gpu": bool(use_gpu),
        },
    }


def score_result(
    result: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    target_adv_ratio: float,
) -> float:
    eps = 1e-12
    m = result["metrics"]
    b = baseline["metrics"]
    h = result["history_summary"]
    hb = baseline["history_summary"]

    rmse_ratio = float(m["rmse"]) / max(float(b["rmse"]), eps)
    pde_ratio = float(m["pde_residual_mean"]) / max(float(b["pde_residual_mean"]), eps)

    adv_mean = max(float(h.get("g_adv_over_pde_tail_mean", 0.0)), eps)
    adv_balance = abs(math.log(adv_mean / max(float(target_adv_ratio), eps)))

    d1_gap_mean = abs(float(h.get("d1_gap_tail_mean", 0.0)))
    d1_gap_std = abs(float(h.get("d1_gap_tail_std", 0.0)))
    d2_gap_std = abs(float(h.get("d2_gap_tail_std", 0.0)))

    base_gap_std = abs(float(hb.get("d1_gap_tail_std", 0.0))) + abs(float(hb.get("d2_gap_tail_std", 0.0)))
    gap_std_ratio = (d1_gap_std + d2_gap_std) / max(base_gap_std, eps)

    d1_gp_dev = abs(float(h.get("d1_gp_tail_mean", 1.0)) - 1.0)
    d2_gp_dev = abs(float(h.get("d2_gp_tail_mean", 1.0)) - 1.0)

    collapse_penalty = 0.0
    if float(h.get("g_total_last", 0.0)) < 1e-4 and float(m.get("rmse", 0.0)) > float(b.get("rmse", 0.0)):
        collapse_penalty += 1.0
    if d1_gap_mean > 2.0:
        collapse_penalty += 0.5

    # Menor pontuação indica melhor resultado.
    return (
        0.33 * rmse_ratio
        + 0.33 * pde_ratio
        + 0.12 * adv_balance
        + 0.10 * gap_std_ratio
        + 0.06 * d1_gap_mean
        + 0.06 * (d1_gp_dev + d2_gp_dev)
        + collapse_penalty
    )


def score_all(results: List[Dict[str, Any]], baseline_name: str) -> List[Dict[str, Any]]:
    by_name = {item["candidate"]["name"]: item for item in results}
    if baseline_name not in by_name:
        raise ValueError(f"Baseline '{baseline_name}' não encontrada nos resultados.")
    baseline = by_name[baseline_name]
    target_adv_ratio = float(baseline["candidate"]["target_adv_over_pde"])

    scored: List[Dict[str, Any]] = []
    for item in results:
        enriched = dict(item)
        enriched["score"] = float(
            score_result(item, baseline, target_adv_ratio=target_adv_ratio)
        )
        scored.append(enriched)
    scored.sort(key=lambda x: x["score"])
    return scored


def select_top_non_baseline(
    scored: List[Dict[str, Any]],
    *,
    top_k: int,
    baseline_name: str,
) -> List[Candidate]:
    out: List[Candidate] = []
    for item in scored:
        if item["candidate"]["name"] == baseline_name:
            continue
        out.append(Candidate(**item["candidate"]))
        if len(out) >= int(top_k):
            break
    return out


def save_markdown_report(
    output_root: Path,
    *,
    phase1_scored: List[Dict[str, Any]],
    phase2_scored: List[Dict[str, Any]],
    best: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Relatório de Ajuste Ativo da PI-GAN")
    lines.append("")
    lines.append("## Melhor Candidato")
    lines.append("")
    lines.append(f"- name: {best['candidate']['name']}")
    lines.append(f"- score: {best['score']:.6f}")
    lines.append(f"- rmse: {best['metrics']['rmse']:.6e}")
    lines.append(f"- pde_residual_mean: {best['metrics']['pde_residual_mean']:.6e}")
    lines.append(f"- d1_gap_tail_mean: {best['history_summary'].get('d1_gap_tail_mean', float('nan')):.6e}")
    lines.append(f"- adv_over_pde_tail_mean: {best['history_summary'].get('g_adv_over_pde_tail_mean', float('nan')):.6e}")
    lines.append("")
    lines.append("## Hiperparâmetros do Candidato")
    lines.append("")
    lines.append("json")
    lines.append(json.dumps(best["candidate"], indent=2))
    lines.append("")
    lines.append("")

    def _top_table(scored: List[Dict[str, Any]], title: str, max_rows: int = 8) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| rank | name | score | rmse | pde_residual_mean | d1_gap_tail_mean |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for idx, item in enumerate(scored[:max_rows], start=1):
            lines.append(
                f"| {idx} | {item['candidate']['name']} | {item['score']:.4f} | "
                f"{item['metrics']['rmse']:.3e} | {item['metrics']['pde_residual_mean']:.3e} | "
                f"{item['history_summary'].get('d1_gap_tail_mean', float('nan')):.3e} |"
            )
        lines.append("")

    _top_table(phase1_scored, "Ranking da Fase 1")
    _top_table(phase2_scored, "Ranking da Fase 2")

    (output_root / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajuste ativo de hiperparâmetros da PI-GAN.")
    parser.add_argument("--output-root", type=str, default="runs_optimality_active")
    parser.add_argument("--search-epochs", type=int, default=40)
    parser.add_argument("--confirm-epochs", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--fast-profile", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PIGAN_ALLOW_CPU", "1")
    use_gpu = bool(args.use_gpu and torch.cuda.is_available())

    output_root = Path(args.output_root)
    phase1_root = output_root / "phase1"
    phase2_root = output_root / "phase2"
    output_root.mkdir(parents=True, exist_ok=True)
    phase1_root.mkdir(parents=True, exist_ok=True)
    phase2_root.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    if int(args.candidate_limit) > 0:
        candidates = candidates[: int(args.candidate_limit)]
    baseline_name = "baseline_active"

    print(f"[env] python={sys.executable}")
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} use_gpu={use_gpu}")
    if use_gpu:
        print(f"[env] device={torch.cuda.get_device_name(0)}")
    print(
        f"[phase1] candidates={len(candidates)} epochs={int(args.search_epochs)} "
        f"batch={int(args.batch_size)} fast_profile={bool(args.fast_profile)}"
    )

    phase1_results: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[phase1] {idx:02d}/{len(candidates):02d} -> {candidate.name}")
        run_dir = phase1_root / candidate.name
        result = run_candidate(
            candidate,
            run_dir=run_dir,
            epochs=int(args.search_epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            use_gpu=use_gpu,
            run_fast_profile=bool(args.fast_profile),
        )
        phase1_results.append(result)

    phase1_scored = score_all(phase1_results, baseline_name=baseline_name)
    (output_root / "summary_phase1.json").write_text(
        json.dumps(phase1_scored, indent=2), encoding="utf-8"
    )

    selected = select_top_non_baseline(
        phase1_scored, top_k=int(args.top_k), baseline_name=baseline_name
    )
    selected_names = [c.name for c in selected]
    print(f"[phase2] selected={selected_names}")

    baseline_candidate = next(c for c in candidates if c.name == baseline_name)
    confirm_candidates = [baseline_candidate, *selected]
    print(f"[phase2] candidates={len(confirm_candidates)} epochs={int(args.confirm_epochs)}")

    phase2_results: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(confirm_candidates, start=1):
        print(f"[phase2] {idx:02d}/{len(confirm_candidates):02d} -> {candidate.name}")
        run_dir = phase2_root / candidate.name
        result = run_candidate(
            candidate,
            run_dir=run_dir,
            epochs=int(args.confirm_epochs),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            use_gpu=use_gpu,
            run_fast_profile=bool(args.fast_profile),
        )
        phase2_results.append(result)

    phase2_scored = score_all(phase2_results, baseline_name=baseline_name)
    (output_root / "summary_phase2.json").write_text(
        json.dumps(phase2_scored, indent=2), encoding="utf-8"
    )

    best_non_baseline = [r for r in phase2_scored if r["candidate"]["name"] != baseline_name]
    best = best_non_baseline[0] if best_non_baseline else phase2_scored[0]
    (output_root / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    save_markdown_report(output_root, phase1_scored=phase1_scored, phase2_scored=phase2_scored, best=best)

    print("[feito] melhor candidato")
    print(json.dumps(best["candidate"], indent=2))
    print(f"[feito] melhor score={best['score']:.6f}")
    print("[feito] melhores métricas")
    print(json.dumps(best["metrics"], indent=2))


if __name__ == "__main__":
    main()
