# -*- coding: utf-8 -*-
import torch

from src.models import LaplacianLayer


def _make_grid(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0):
    x = torch.linspace(0.0, lx, nx, dtype=torch.float32)
    y = torch.linspace(0.0, ly, ny, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def test_laplacian_quadratic_field_matches_constant_interior():
    nx, ny = 17, 17
    lx, ly = 1.0, 1.0
    hx = lx / float(nx - 1)
    hy = ly / float(ny - 1)

    xx, yy = _make_grid(nx, ny, lx=lx, ly=ly)
    field = (xx.pow(2) + yy.pow(2)).unsqueeze(0).unsqueeze(0)  # Laplaciano = 4

    lap = LaplacianLayer(hx=hx, hy=hy)
    residual = lap(field)

    interior = residual[:, :, 1:-1, 1:-1]
    expected = torch.full_like(interior, 4.0)
    assert torch.allclose(interior, expected, atol=5e-4, rtol=1e-4)

    # O resíduo na borda deve ser zero para evitar artefatos de padding.
    assert torch.allclose(residual[:, :, 0, :], torch.zeros_like(residual[:, :, 0, :]), atol=0.0)
    assert torch.allclose(residual[:, :, -1, :], torch.zeros_like(residual[:, :, -1, :]), atol=0.0)
    assert torch.allclose(residual[:, :, :, 0], torch.zeros_like(residual[:, :, :, 0]), atol=0.0)
    assert torch.allclose(residual[:, :, :, -1], torch.zeros_like(residual[:, :, :, -1]), atol=0.0)


def test_laplacian_linear_field_is_zero_interior():
    nx, ny = 19, 19
    lx, ly = 1.0, 1.0
    hx = lx / float(nx - 1)
    hy = ly / float(ny - 1)

    xx, yy = _make_grid(nx, ny, lx=lx, ly=ly)
    field = (2.0 * xx - 3.0 * yy + 5.0).unsqueeze(0).unsqueeze(0)  # Harmônico

    lap = LaplacianLayer(hx=hx, hy=hy)
    residual = lap(field)
    interior = residual[:, :, 1:-1, 1:-1]
    # Acúmulo em float32 no stencil gera truncamento ~1e-3 nessa malha.
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-3, rtol=0.0)
