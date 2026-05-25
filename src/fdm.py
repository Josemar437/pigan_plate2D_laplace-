# -*- coding: utf-8 -*-
"""Referencia numerica FDM-SOR para a placa Laplace 2D do pipeline."""

from __future__ import annotations

from typing import Tuple

import torch


def _apply_dirichlet_boundaries(
    field: torch.Tensor, boundary_field: torch.Tensor
) -> None:
    """Mantem as quatro bordas iguais ao campo `g` construido em `src/utils.py`."""
    field[0, :] = boundary_field[0, :]
    field[-1, :] = boundary_field[-1, :]
    field[:, 0] = boundary_field[:, 0]
    field[:, -1] = boundary_field[:, -1]


def _apply_mixed_boundaries(
    field: torch.Tensor,
    *,
    t_left: float,
    t_right: float,
) -> None:
    """Aplica Dirichlet nas laterais e Neumann homogeneo em baixo/cima."""
    left = torch.tensor(float(t_left), device=field.device, dtype=field.dtype)
    right = torch.tensor(float(t_right), device=field.device, dtype=field.dtype)

    field[:, 0] = left
    field[:, -1] = right
    field[0, 1:-1] = field[1, 1:-1]
    field[-1, 1:-1] = field[-2, 1:-1]
    field[0, 0] = left
    field[-1, 0] = left
    field[0, -1] = right
    field[-1, -1] = right


def solve_laplace_mixed_dirichlet_neumann(
    initial_field: torch.Tensor,
    *,
    lx: float,
    ly: float,
    t_left: float,
    t_right: float,
    tol: float = 1e-6,
    max_iter: int = 20000,
    omega: float = 1.7,
) -> Tuple[torch.Tensor, int]:
    """Resolve Laplace com T(0,y), T(LX,y) fixos e dT/dy=0 em y=0,LY."""
    if initial_field.ndim != 2:
        raise ValueError("initial_field deve ser um tensor 2D [ny, nx].")

    ny, nx = initial_field.shape
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    hx = float(lx) / float(nx - 1)
    hy = float(ly) / float(ny - 1)
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (hx2 + hy2)
    omega = float(omega)
    if not (0.0 < omega < 2.0):
        raise ValueError("omega deve satisfazer 0 < omega < 2 para convergência SOR.")

    t = initial_field.clone()
    _apply_mixed_boundaries(t, t_left=t_left, t_right=t_right)

    ii = torch.arange(1, ny - 1, device=t.device).view(-1, 1)
    jj = torch.arange(1, nx - 1, device=t.device).view(1, -1)
    red_mask = (ii + jj) % 2 == 0
    black_mask = ~red_mask

    for it in range(1, int(max_iter) + 1):
        max_update = 0.0

        for mask in (red_mask, black_mask):
            interior_view = t[1:-1, 1:-1]
            old_vals = interior_view[mask].clone()

            gs_update = (
                (t[1:-1, 2:] + t[1:-1, :-2]) * hy2
                + (t[2:, 1:-1] + t[:-2, 1:-1]) * hx2
            ) / denom
            new_vals = (1.0 - omega) * old_vals + omega * gs_update[mask]
            interior_view[mask] = new_vals

            if old_vals.numel() > 0:
                step_update = torch.max(torch.abs(new_vals - old_vals)).item()
                max_update = max(max_update, float(step_update))

        _apply_mixed_boundaries(t, t_left=t_left, t_right=t_right)

        if max_update < float(tol):
            return t, it

    return t, int(max_iter)


def solve_laplace_dirichlet(
    boundary_field: torch.Tensor,
    *,
    lx: float,
    ly: float,
    tol: float = 1e-6,
    max_iter: int = 20000,
    omega: float = 1.7,
    initial_guess: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, int]:
    """Resolve a referencia `T_ref` usada por `D2` e pelas metricas finais."""
    if boundary_field.ndim != 2:
        raise ValueError("boundary_field deve ser um tensor 2D [ny, nx].")

    ny, nx = boundary_field.shape
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    hx = float(lx) / float(nx - 1)
    hy = float(ly) / float(ny - 1)
    hx2 = hx * hx
    hy2 = hy * hy
    denom = 2.0 * (hx2 + hy2)
    omega = float(omega)
    if not (0.0 < omega < 2.0):
        raise ValueError("omega deve satisfazer 0 < omega < 2 para convergência SOR.")

    if initial_guess is None:
        t = boundary_field.clone()
    else:
        if initial_guess.shape != boundary_field.shape:
            raise ValueError(
                "initial_guess deve ter a mesma forma que boundary_field."
            )
        t = initial_guess.clone()
        _apply_dirichlet_boundaries(t, boundary_field)

    # Garante que os valores de contorno permaneçam fixos.
    _apply_dirichlet_boundaries(t, boundary_field)

    ii = torch.arange(1, ny - 1, device=t.device).view(-1, 1)
    jj = torch.arange(1, nx - 1, device=t.device).view(1, -1)
    red_mask = (ii + jj) % 2 == 0
    black_mask = ~red_mask

    for it in range(1, int(max_iter) + 1):
        max_update = 0.0

        for mask in (red_mask, black_mask):
            interior_view = t[1:-1, 1:-1]
            old_vals = interior_view[mask].clone()

            gs_update = (
                (t[1:-1, 2:] + t[1:-1, :-2]) * hy2
                + (t[2:, 1:-1] + t[:-2, 1:-1]) * hx2
            ) / denom
            new_vals = (1.0 - omega) * old_vals + omega * gs_update[mask]
            interior_view[mask] = new_vals

            if old_vals.numel() > 0:
                step_update = torch.max(torch.abs(new_vals - old_vals)).item()
                max_update = max(max_update, float(step_update))

        # Re-aplica contorno para evitar deriva por efeitos numéricos.
        _apply_dirichlet_boundaries(t, boundary_field)

        if max_update < float(tol):
            return t, it

    return t, int(max_iter)


