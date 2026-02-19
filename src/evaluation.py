# -*- coding: utf-8 -*-
"""Métricas de avaliação para modelos de campo PI-GAN."""

from __future__ import annotations

from typing import Dict

import torch

from src.models import LaplacianLayer


@torch.no_grad()
def compute_field_metrics(
    pred_field: torch.Tensor,
    ref_field: torch.Tensor,
    laplacian: LaplacianLayer,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Calcula diversas métricas de erro e consistência física para o campo predito.

    Parâmetros:
        pred_field: Campo de temperatura predito pelo modelo [1, 1, H, W].
        ref_field: Campo de temperatura de referência (FDM ou exato) [1, 1, H, W].
        laplacian: Camada que aplica o operador Laplaciano para cálculo de resíduo.
        interior_mask: Máscara binária para os pontos internos do domínio.
        boundary_mask: Máscara binária para os pontos de fronteira.

    Retorno:
        Dicionário contendo métricas como MAE, RMSE, R2, erro L2 relativo,
        máximo erro absoluto e resíduos da PDE.
    """
    diff = pred_field - ref_field
    abs_diff = diff.abs()

    # Métricas de regressão no campo completo.
    mae = abs_diff.mean()
    rmse = torch.sqrt((diff ** 2).mean())

    flat_ref = ref_field.reshape(-1)
    flat_pred = pred_field.reshape(-1)
    ss_res = torch.sum((flat_ref - flat_pred) ** 2)
    ref_mean = torch.mean(flat_ref)
    ss_tot = torch.sum((flat_ref - ref_mean) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    # MAPE calculado apenas onde |ref| > eps para evitar divisão por zero.
    eps = 1e-8
    valid = flat_ref.abs() > eps
    if bool(valid.any()):
        mape = (
            torch.mean(
                torch.abs((flat_pred[valid] - flat_ref[valid]) / flat_ref[valid])
            )
            * 100.0
        )
    else:
        mape = torch.tensor(0.0, device=pred_field.device, dtype=pred_field.dtype)

    rel_l2 = torch.linalg.norm(diff.reshape(-1), ord=2) / (
        torch.linalg.norm(ref_field.reshape(-1), ord=2) + 1e-12
    )
    max_error = abs_diff.max()

    # Resíduos da PDE.
    residual = laplacian(pred_field)
    pde_residual_mean = (residual.abs() * interior_mask).sum() / (
        interior_mask.sum() + 1e-12
    )
    pde_residual_l2 = torch.sqrt(
        ((residual ** 2) * interior_mask).sum() / (interior_mask.sum() + 1e-12)
    )
    pde_residual_max = (residual.abs() * interior_mask).max()

    # Erro nas fronteiras.
    boundary_error = (abs_diff * boundary_mask).sum() / (boundary_mask.sum() + 1e-12)

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "mape": float(mape.item()),
        "r2": float(r2.item()),
        "relative_l2_error": float(rel_l2.item()),
        "max_error": float(max_error.item()),
        "pde_residual_mean": float(pde_residual_mean.item()),
        "pde_residual_l2": float(pde_residual_l2.item()),
        "pde_residual_max": float(pde_residual_max.item()),
        "boundary_error": float(boundary_error.item()),
    }

