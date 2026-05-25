# -*- coding: utf-8 -*-
"""Metricas finais salvas pelo pipeline para `temperature_pred` vs `T_ref`."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from src.models import LaplacianLayer


@torch.no_grad()
def compute_field_metrics(
    pred_field: torch.Tensor,
    ref_field: torch.Tensor,
    laplacian: LaplacianLayer,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    neumann_mask: Optional[torch.Tensor] = None,
    neumann_dy: float = 1.0,
) -> Dict[str, float]:
    """Compara campos `[1,1,H,W]` e mede erro, residuo PDE e fronteira."""
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
    neumann_max_error = torch.tensor(0.0, device=pred_field.device, dtype=pred_field.dtype)
    neumann_mse = torch.tensor(0.0, device=pred_field.device, dtype=pred_field.dtype)
    if neumann_mask is not None:
        dy = max(float(neumann_dy), 1e-12)
        neumann_residual = torch.zeros_like(pred_field)
        neumann_residual[:, :, 0, 1:-1] = (
            pred_field[:, :, 1, 1:-1] - pred_field[:, :, 0, 1:-1]
        ) / dy
        neumann_residual[:, :, -1, 1:-1] = (
            pred_field[:, :, -1, 1:-1] - pred_field[:, :, -2, 1:-1]
        ) / dy
        neumann_mse = ((neumann_residual ** 2) * neumann_mask).sum() / (
            neumann_mask.sum() + 1e-12
        )
        neumann_max_error = (neumann_residual.abs() * neumann_mask).max()

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
        "neumann_mse": float(neumann_mse.item()),
        "neumann_max_error": float(neumann_max_error.item()),
    }

