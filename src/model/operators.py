# -*- coding: utf-8 -*-
"""
Operadores físicos discretos da PI-GAN: Laplaciano, máscara phi e extensão g.

O Laplaciano não treinável produz R0 = nabla^2 T0 nos pontos internos para L_PDE.
As funções g/phi suportam T0 = g + phi * N0 (imposição forte de Dirichlet lateral).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import build_dirichlet_extension, build_hard_constraint_mask


def build_g_field(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    lx: float,
    ly: float,
    t_left: float,
    t_right: float,
    boundary_sine_amplitude: float = 0.0,
) -> torch.Tensor:
    """Extensão g(x,y) das condições de Dirichlet laterais ao domínio."""
    return build_dirichlet_extension(
        x_grid,
        y_grid,
        lx=lx,
        ly=ly,
        t_left=t_left,
        t_right=t_right,
        boundary_sine_amplitude=boundary_sine_amplitude,
    )


def build_phi_mask(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    lx: float,
    ly: float,
) -> torch.Tensor:
    """Máscara phi que se anula nas bordas de Dirichlet (x=0, x=1)."""
    return build_hard_constraint_mask(x_grid, y_grid, lx=lx, ly=ly)


class LaplacianOperator(nn.Module):
    """
    Laplaciano discreto 3x3 centrado, não treinável, via convolução com buffer fixo.

    Usado na formulação PI-GAN: L_PDE = E[|R0|] com R0 aplicado apenas no interior.
    """

    def __init__(self, hx: float, hy: Optional[float] = None) -> None:
        super().__init__()
        hy_val = float(hy if hy is not None else hx)
        hx_val = float(hx)
        if hx_val <= 0.0 or hy_val <= 0.0:
            raise ValueError("hx e hy devem ser positivos.")

        # Stencil 5-pontos centrado para nabla^2 T em malha uniforme.
        kernel = torch.tensor(
            [
                [0.0, 1.0 / (hy_val * hy_val), 0.0],
                [
                    1.0 / (hx_val * hx_val),
                    -2.0 * (1.0 / (hx_val * hx_val) + 1.0 / (hy_val * hy_val)),
                    1.0 / (hx_val * hx_val),
                ],
                [0.0, 1.0 / (hy_val * hy_val), 0.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    def forward(self, T0: torch.Tensor) -> torch.Tensor:
        """
        Calcula R0 = laplacian(T0) com zeros nas bordas (apenas interior preenchido).

        Parâmetros:
            T0: Campo candidato [B, 1, H, W].

        Retorno:
            Mapa de resíduo [B, 1, H, W] com bordas zeradas.
        """
        if T0.ndim != 4 or T0.shape[1] != 1:
            raise ValueError("T0 deve ter formato [B,1,H,W].")
        if T0.shape[2] < 3 or T0.shape[3] < 3:
            raise ValueError("As dimensões espaciais de T0 devem ser >= 3.")

        kernel = self.laplacian_kernel.to(device=T0.device, dtype=T0.dtype)
        # Convolução sem padding: resultado válido só no interior [1:-1, 1:-1].
        interior = F.conv2d(T0, kernel, padding=0)
        R0 = torch.zeros_like(T0)
        R0[:, :, 1:-1, 1:-1] = interior
        return R0


# Alias de compatibilidade com código e checkpoints legados.
LaplacianLayer = LaplacianOperator
