# -*- coding: utf-8 -*-
"""Launcher reprodutivel para o treino final rebalanceado."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BEST_TRIAL = "runs_optuna_rebalance_32t/best_trial.json"
DEFAULT_RUNS_DIR = "runs_final_rebalance_4000"
DEFAULT_SEARCH_STEPS_PER_EPOCH = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa scripts/runFinalBest.py com defaults compativeis com "
            "runs_optuna_rebalance_32t."
        )
    )
    parser.add_argument("--best-trial", default=DEFAULT_BEST_TRIAL)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=DEFAULT_SEARCH_STEPS_PER_EPOCH,
        help=(
            "Default 1 para manter o mesmo regime do optunaSearch.py, "
            "ja que o best_trial.json antigo nao documenta esse valor."
        ),
    )
    parser.add_argument(
        "--search-steps-per-epoch",
        type=int,
        default=DEFAULT_SEARCH_STEPS_PER_EPOCH,
        help="Valor documentado/assumido para a busca que gerou o best_trial.",
    )
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tuning-epochs", type=int, default=180)
    parser.add_argument("--target-residual", type=float, default=1e-3)
    parser.add_argument(
        "--physics-refine-steps",
        type=int,
        default=0,
        help="Default 0. Use >0 somente como ablation/refinamento declarado.",
    )
    parser.add_argument("--physics-refine-lr", type=float, default=2.0e-5)
    parser.add_argument("--allow-steps-mismatch", action="store_true")
    parser.add_argument("--no-scale-schedules", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/runFinalBest.py",
        "--best-trial",
        str(args.best_trial),
        "--runs-dir",
        str(args.runs_dir),
        "--epochs",
        str(int(args.epochs)),
        "--steps-per-epoch",
        str(int(args.steps_per_epoch)),
        "--search-steps-per-epoch",
        str(int(args.search_steps_per_epoch)),
        "--grid-size",
        str(int(args.grid_size)),
        "--batch-size",
        str(int(args.batch_size)),
        "--tuning-epochs",
        str(int(args.tuning_epochs)),
        "--target-residual",
        str(float(args.target_residual)),
        "--physics-refine-steps",
        str(max(0, int(args.physics_refine_steps))),
        "--physics-refine-lr",
        str(float(args.physics_refine_lr)),
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.allow_steps_mismatch):
        command.append("--allow-steps-mismatch")
    if bool(args.no_scale_schedules):
        command.append("--no-scale-schedules")
    if bool(args.no_plots):
        command.append("--no-plots")
    if bool(args.use_gpu):
        command.append("--use-gpu")
    return command


def main() -> int:
    args = parse_args()
    command = build_command(args)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
