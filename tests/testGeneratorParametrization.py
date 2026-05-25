# -*- coding: utf-8 -*-
"""Testes da parametrização T0 = g + phi * N0 (Dirichlet forte nas laterais)."""

import torch

from src.model.generator import UNetGenerator2D
from src.model.operators import build_g_field, build_phi_mask
from src.utils import create_cartesian_grid


def test_dirichlet_exact_on_lateral_boundaries() -> None:
    nx, ny = 11, 9
    device = torch.device("cpu")
    x_grid, y_grid = create_cartesian_grid(nx, ny, 1.0, 1.0, device=device)
    g = build_g_field(
        x_grid,
        y_grid,
        lx=1.0,
        ly=1.0,
        t_left=200.0,
        t_right=100.0,
    ).unsqueeze(0).unsqueeze(0)
    phi = build_phi_mask(x_grid, y_grid, lx=1.0, ly=1.0).unsqueeze(0).unsqueeze(0)

    generator = UNetGenerator2D(
        in_channels=1,
        latent_dim=0,
        base_channels=8,
        depth=3,
        use_batch_norm=False,
        hard_constraint=True,
    )
    generator.eval()
    with torch.no_grad():
        T0 = generator(g, phi)

    assert T0[:, :, :, 0].allclose(g[:, :, :, 0], atol=0.0, rtol=0.0)
    assert T0[:, :, :, -1].allclose(g[:, :, :, -1], atol=0.0, rtol=0.0)
