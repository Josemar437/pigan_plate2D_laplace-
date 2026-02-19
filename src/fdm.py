# -*- coding: utf-8 -*-
"""Solucionador de diferenças finitas para a equação de Laplace 2D estacionária."""

from __future__ import annotations

from typing import Tuple

import torch


def _apply_dirichlet_boundaries(
    field: torch.Tensor, boundary_field: torch.Tensor
) -> None:
    """
    Copia os valores de contorno de boundary_field para field in-place.

    Parâmetros:
        field: Tensor do campo a ser atualizado.
        boundary_field: Tensor contendo os valores de contorno desejados.
    """
    field[0, :] = boundary_field[0, :]
    field[-1, :] = boundary_field[-1, :]
    field[:, 0] = boundary_field[:, 0]
    field[:, -1] = boundary_field[:, -1]


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
    """
    Resolve ∇²T = 0 em uma malha retangular com condições de contorno de Dirichlet.

    Utiliza o método SOR (Successive Over-Relaxation) com padrão Red-Black para
    otimização em GPU.

    Parâmetros:
        boundary_field: Tensor 2D [ny, nx] com valores de contorno nas bordas.
            Valores internos são ignorados a menos que initial_guess seja None.
        lx: Comprimento físico do domínio no eixo X.
        ly: Comprimento físico do domínio no eixo Y.
        tol: Tolerância de convergência baseada na atualização máxima interna.
        max_iter: Número máximo de iterações do SOR.
        omega: Fator de relaxação (1.0 -> Gauss-Seidel, (1, 2) -> SOR).
        initial_guess: Campo inicial opcional para acelerar a convergência.

    Retorno:
        Uma tupla (campo_final, iteracoes_executadas).

    Exceções:
        ValueError: Se as dimensões do campo ou parâmetros forem inválidos.
    """
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


