# -*- coding: utf-8 -*-
"""
Perdas físicas da PI-GAN: L_PDE e penalização de Neumann.

L_PDE = E[|R0|] no interior, com R0 = laplacian(T0).
Neumann é imposto indiretamente (não é restrição forte em T0).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _interior_mean(
    tensor: torch.Tensor,
    interior_mask: torch.Tensor,
    *,
    corner_weight: float = 0.0,
    corner_band_points: int = 0,
) -> torch.Tensor:
    """Média ponderada no interior; peso extra opcional nos cantos."""
    weights = interior_mask.to(dtype=tensor.dtype)
    if corner_weight > 0.0 and corner_band_points > 0:
        h = int(interior_mask.shape[-2])
        w = int(interior_mask.shape[-1])
        band = int(corner_band_points)
        corner_mask = torch.zeros_like(weights)
        corner_mask[..., :band, :band] = 1.0
        corner_mask[..., :band, -band:] = 1.0
        corner_mask[..., -band:, :band] = 1.0
        corner_mask[..., -band:, -band:] = 1.0
        corner_mask = corner_mask * interior_mask
        n_corners = corner_mask.sum().clamp_min(1.0)
        n_total = interior_mask.sum().clamp_min(1.0)
        corner_frac = n_corners / n_total
        if corner_frac > 0.0:
            boost = 1.0 + float(corner_weight) / float(corner_frac)
            weights = weights + corner_mask * (boost - 1.0)
            weights = weights * interior_mask

    denom = weights.sum().clamp_min(1.0)
    return (tensor.abs() * weights).sum() / denom


def loss_pde(
    R0: torch.Tensor,
    interior_mask: torch.Tensor,
    *,
    corner_weight: float = 0.0,
    corner_band_points: int = 0,
) -> torch.Tensor:
    """
    L_PDE = E[|laplacian(T0)|] restrito ao interior.

    Parâmetros:
        R0: Resíduo Laplaciano [B,1,H,W] (bordas zero).
        interior_mask: Máscara do interior [1,1,H,W] ou [B,1,H,W].
    """
    if interior_mask.shape[0] == 1 and R0.shape[0] > 1:
        mask = interior_mask.expand_as(R0)
    else:
        mask = interior_mask
    return _interior_mean(R0, mask, corner_weight=corner_weight, corner_band_points=corner_band_points)


def loss_pde_from_field(
    T0: torch.Tensor,
    laplacian_op: nn.Module,
    interior_mask: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    """Calcula R0 via operador e retorna L_PDE."""
    R0 = laplacian_op(T0)
    return loss_pde(R0, interior_mask, **kwargs)  # type: ignore[arg-type]


def loss_pde_weighted(
    R0: torch.Tensor,
    weight_map: torch.Tensor,
    weight_sum: torch.Tensor,
) -> torch.Tensor:
    """L_PDE com mapa de pesos do interior (ex.: ênfase em cantos)."""
    num = (R0.abs() * weight_map).sum()
    den = weight_sum * float(max(1, int(R0.shape[0]))) + 1e-12
    return num / den


def neumann_residual(T0: torch.Tensor, *, dy: float = 1.0) -> torch.Tensor:
    """Resíduo de derivada normal discreta nas bordas horizontais (Neumann homogêneo alvo zero)."""
    dy_val = float(dy)
    residual = torch.zeros_like(T0)
    residual[:, :, 0, 1:-1] = (T0[:, :, 1, 1:-1] - T0[:, :, 0, 1:-1]) / dy_val
    residual[:, :, -1, 1:-1] = (T0[:, :, -1, 1:-1] - T0[:, :, -2, 1:-1]) / dy_val
    return residual


def neumann_loss(
    T0: torch.Tensor,
    neumann_mask: torch.Tensor,
    *,
    dy: float = 1.0,
) -> torch.Tensor:
    """
    Penalização de Neumann: NOT a hard constraint on T0.

    Imposto indiretamente via derivada normal + referência FDM (D2).
    """
    residual = neumann_residual(T0, dy=dy)
    batch = max(1, int(T0.shape[0]))
    num = ((residual ** 2) * neumann_mask).sum()
    den = neumann_mask.sum() * float(batch) + 1e-12
    return num / den
