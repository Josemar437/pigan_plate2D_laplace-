# -*- coding: utf-8 -*-
"""Utilitários para construção de malhas 2D e manipulação de contornos."""

from __future__ import annotations

import math
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
    """
    Cria as malhas de coordenadas X e Y para um domínio retangular.

    Parâmetros:
        nx: Número de pontos no eixo X.
        ny: Número de pontos no eixo Y.
        lx: Comprimento físico total no eixo X.
        ly: Comprimento físico total no eixo Y.
        device: Dispositivo (CPU/GPU) onde os tensores serão alocados.
        dtype: Tipo de dado dos tensores (default: float32).

    Retorno:
        Uma tupla (x_grid, y_grid) com tensores de forma [ny, nx].

    Exceções:
        ValueError: Se nx ou ny forem menores que 3.
    """
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
    """
    Constrói uma extensão g(x,y) que satisfaz as condições de contorno de Dirichlet.

    As bordas laterais são constantes T_LEFT/T_RIGHT. As bordas superior e
    inferior são perfis lineares com uma possível perturbação senoidal.
    O interior é interpolado via Coons Patch.

    Parâmetros:
        x_grid: Malha de coordenadas X.
        y_grid: Malha de coordenadas Y.
        lx: Comprimento do domínio em X.
        ly: Comprimento do domínio em Y.
        t_left: Temperatura na borda esquerda (X=0).
        t_right: Temperatura na borda direita (X=LX).
        boundary_sine_amplitude: Amplitude da perturbação senoidal nas bordas.

    Retorno:
        Um tensor com a extensão suave das condições de contorno.
    """
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid e y_grid devem ter a mesma forma.")

    x_hat = x_grid / (float(lx) + 1e-12)
    y_hat = y_grid / (float(ly) + 1e-12)

    t_left_t = torch.tensor(float(t_left), device=x_grid.device, dtype=x_grid.dtype)
    t_right_t = torch.tensor(float(t_right), device=x_grid.device, dtype=x_grid.dtype)
    amplitude = torch.tensor(
        float(boundary_sine_amplitude), device=x_grid.device, dtype=x_grid.dtype
    )

    linear_x = t_left_t + (t_right_t - t_left_t) * x_hat
    sine = torch.sin(math.pi * x_hat)

    bottom = linear_x + amplitude * sine
    top = linear_x - amplitude * sine
    left = torch.full_like(y_grid, t_left_t)
    right = torch.full_like(y_grid, t_right_t)

    g00 = t_left_t
    g10 = t_right_t
    g01 = t_left_t
    g11 = t_right_t

    # Coons Patch bilineal para extensão das fronteiras.
    coons = (
        (1.0 - x_hat) * left
        + x_hat * right
        + (1.0 - y_hat) * bottom
        + y_hat * top
        - (
            (1.0 - x_hat) * (1.0 - y_hat) * g00
            + x_hat * (1.0 - y_hat) * g10
            + (1.0 - x_hat) * y_hat * g01
            + x_hat * y_hat * g11
        )
    )
    return coons


def build_hard_constraint_mask(
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    lx: float,
    ly: float,
) -> torch.Tensor:
    """
    Calcula o termo de decaimento phi(x,y) para imposição forte de contorno.

    Calcula phi(x,y) = x_norm * (1 - x_norm) * y_norm * (1 - y_norm), que é
    zero em todas as bordas e máximo no centro.

    Parâmetros:
        x_grid: Malha de coordenadas X.
        y_grid: Malha de coordenadas Y.
        lx: Comprimento do domínio em X.
        ly: Comprimento do domínio em Y.

    Retorno:
        Um tensor com a máscara de imposição de contorno (distância à fronteira).
    """
    x_hat = x_grid / (float(lx) + 1e-12)
    y_hat = y_grid / (float(ly) + 1e-12)
    phi = x_hat * (1.0 - x_hat) * y_hat * (1.0 - y_hat)
    # Normaliza para max=1, mantendo escala útil para o termo corretivo phi*N.
    # Valores de borda permanecem zero, preservando Dirichlet forte.
    phi_max = torch.amax(phi).clamp_min(1e-12)
    return phi / phi_max


def build_domain_masks(
    ny: int,
    nx: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrói máscaras binárias para separar o interior e as bordas.

    Parâmetros:
        ny: Número de pontos em Y.
        nx: Número de pontos em X.
        device: Dispositivo de alocação.
        dtype: Tipo de dado (default: float32).

    Retorno:
        Uma tupla (máscara_interior, máscara_fronteira) com forma [1, 1, ny, nx].
    """
    if nx < 3 or ny < 3:
        raise ValueError("A malha deve ter pelo menos nx >= 3 e ny >= 3.")

    interior = torch.zeros((1, 1, ny, nx), device=device, dtype=dtype)
    interior[:, :, 1:-1, 1:-1] = 1.0
    boundary = 1.0 - interior
    return interior, boundary


