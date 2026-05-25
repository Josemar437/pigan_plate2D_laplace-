#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Funções utilitárias compartilhadas para scripts de ajuste com Optuna."""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def history_array(history: List[Dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([float(item.get(key, np.nan)) for item in history], dtype=float)


def tail_mean_abs(history: List[Dict[str, float]], key: str, window: int = 30) -> float:
    arr = history_array(history, key)
    if arr.size == 0:
        return float("nan")
    tail = arr[-min(window, arr.size):]
    tail = np.abs(tail[np.isfinite(tail)])
    if tail.size == 0:
        return float("nan")
    return float(np.mean(tail))


def tail_std(history: List[Dict[str, float]], key: str, window: int = 30) -> float:
    arr = history_array(history, key)
    if arr.size == 0:
        return float("nan")
    tail = arr[-min(window, arr.size):]
    tail = tail[np.isfinite(tail)]
    if tail.size <= 1:
        return 0.0
    return float(np.std(tail))


def best_train_residual(history: List[Dict[str, float]]) -> Tuple[float, int]:
    arr = history_array(history, "g_residual_mean_abs")
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return float("inf"), 0
    finite_vals = arr[finite_idx]
    local = int(np.argmin(finite_vals))
    epoch_idx = int(finite_idx[local]) + 1
    return float(finite_vals[local]), epoch_idx


def round_sigfigs(value: float, sigfigs: int = 3) -> float:
    if int(sigfigs) < 1:
        raise ValueError("sigfigs deve ser >= 1")
    val = float(value)
    if not np.isfinite(val) or val == 0.0:
        return val
    return float(f"{val:.{int(sigfigs) - 1}e}")


def format_scientific(value: float, sigfigs: int = 3) -> str:
    if int(sigfigs) < 1:
        raise ValueError("sigfigs deve ser >= 1")
    rounded = round_sigfigs(float(value), sigfigs=int(sigfigs))
    if np.isnan(rounded):
        return "nan"
    if np.isposinf(rounded):
        return "inf"
    if np.isneginf(rounded):
        return "-inf"
    return f"{rounded:.{int(sigfigs) - 1}e}"


def round_hyperparams(
    params: Dict[str, Any],
    *,
    sigfigs: int = 3,
    scientific: bool = True,
) -> Dict[str, Any]:
    rounded: Dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, bool):
            rounded[key] = value
            continue
        if isinstance(value, numbers.Integral):
            rounded[key] = int(value)
            continue
        if isinstance(value, numbers.Real):
            val = round_sigfigs(float(value), sigfigs=int(sigfigs))
            rounded[key] = (
                format_scientific(val, sigfigs=int(sigfigs))
                if scientific
                else float(val)
            )
            continue
        rounded[key] = value
    return rounded


def study_summary_payload(
    study: optuna.Study,
    top_k: int = 10,
    *,
    params_sigfigs: Optional[int] = None,
    params_scientific: bool = False,
) -> Dict[str, Any]:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: float(t.value if t.value is not None else float("inf")))
    top = []
    for t in ranked[: max(1, int(top_k))]:
        params = dict(t.params)
        if params_sigfigs is not None:
            params = round_hyperparams(
                params,
                sigfigs=int(params_sigfigs),
                scientific=bool(params_scientific),
            )
        top.append(
            {
                "trial_number": int(t.number),
                "value": None if t.value is None else float(t.value),
                "params": params,
                "user_attrs": dict(t.user_attrs),
            }
        )

    payload: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "n_trials": int(len(study.trials)),
        "n_complete": int(len(completed)),
        "n_pruned": int(
            sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        ),
        "best_value": None,
        "best_params": None,
        "best_user_attrs": None,
        "top_trials": top,
    }
    if ranked:
        best = ranked[0]
        payload["best_value"] = float(best.value)
        best_params = dict(best.params)
        if params_sigfigs is not None:
            best_params = round_hyperparams(
                best_params,
                sigfigs=int(params_sigfigs),
                scientific=bool(params_scientific),
            )
        payload["best_params"] = best_params
        payload["best_user_attrs"] = dict(best.user_attrs)
    return payload


__all__ = [
    "write_json",
    "history_array",
    "tail_mean_abs",
    "tail_std",
    "best_train_residual",
    "round_sigfigs",
    "format_scientific",
    "round_hyperparams",
    "study_summary_payload",
]
