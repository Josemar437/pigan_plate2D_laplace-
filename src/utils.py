# -*- coding: utf-8 -*-
"""Geometria discreta usada pelo pipeline Laplace 2D.

Este modulo define a malha `[0,LX] x [0,LY]`, a extensao dos contornos
Dirichlet laterais `g` e a mascara `phi` usada por `UNetGenerator2D` em
`T = g + phi*N`.
"""

from __future__ import annotations

from typing import Tuple

import torch


def create_cartesian_grid(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retorna `x_grid,y_grid` com shape `[ny,nx]` e indexacao `ij`."""
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    x = torch.linspace(0.0, float(lx), int(nx), device=device, dtype=dtype)
    y = torch.linspace(0.0, float(ly), int(ny), device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    return x_grid, y_grid


def build_dirichlet_extension(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    lx: float,
    ly: float,
    t_left: float,
    t_right: float,
    boundary_sine_amplitude: float = 0.0,
) -> torch.Tensor:
    """Constroi `g(x,y)` compativel com Dirichlet lateral e Neumann horizontal."""
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devem ter a mesma forma.")

    x_hat = x_grid / (float(lx) + 1e-12)

    t_left_t = torch.tensor(float(t_left), device=x_grid.device, dtype=x_grid.dtype)
    t_right_t = torch.tensor(float(t_right), device=x_grid.device, dtype=x_grid.dtype)
    if abs(float(boundary_sine_amplitude)) > 1e-12:
        raise ValueError(
            "boundary_sine_amplitude deve ser 0.0 para o problema misto "
            "Dirichlet-Neumann: T(0,y)=T_LEFT, T(LX,y)=T_RIGHT, dT/dy=0."
        )

    return t_left_t + (t_right_t - t_left_t) * x_hat


def build_hard_constraint_mask(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    lx: float,
    ly: float,
) -> torch.Tensor:
    """Retorna `phi` normalizado para `T=g+phi*N`, zero nas bordas Dirichlet."""
    x_hat = x_grid / (float(lx) + 1e-12)
    phi = x_hat * (1.0 - x_hat)
    # Normaliza para max=1, mantendo escala útil para o termo corretivo phi*N.
    # Valores nas laterais permanecem zero, preservando Dirichlet forte.
    phi_max = torch.amax(phi).clamp_min(1e-12)
    return phi / phi_max


def build_domain_masks(
    ny: int,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retorna mascaras `[1,1,ny,nx]` para interior `[1:-1,1:-1]` e borda."""
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    interior = torch.zeros((1, 1, ny, nx), device=device, dtype=dtype)
    interior[:, :, 1:-1, 1:-1] = 1.0
    boundary = 1.0 - interior
    return interior, boundary


def build_mixed_boundary_masks(
    ny: int,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna mascaras para interior, Dirichlet lateral e Neumann horizontal."""
    interior, _ = build_domain_masks(ny, nx, device=device, dtype=dtype)
    dirichlet = torch.zeros((1, 1, ny, nx), device=device, dtype=dtype)
    dirichlet[:, :, :, 0] = 1.0
    dirichlet[:, :, :, -1] = 1.0

    neumann = torch.zeros((1, 1, ny, nx), device=device, dtype=dtype)
    neumann[:, :, 0, 1:-1] = 1.0
    neumann[:, :, -1, 1:-1] = 1.0
    return interior, dirichlet, neumann


