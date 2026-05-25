# -*- coding: utf-8 -*-
"""Testes da fase final de refinamento fisico."""

import torch
from torch import nn

from src.models import LaplacianLayer
from src.trainer import FieldPIGANTrainer, FieldTrainerConfig


class _TrainableFieldGenerator(nn.Module):
    latent_dim = 0

    def __init__(self, initial_raw: torch.Tensor) -> None:
        super().__init__()
        self.raw = nn.Parameter(initial_raw.clone())

    def forward(self, base_field, phi_mask, z=None, coord_field=None):
        return base_field + phi_mask * self.raw


class _TinyDualDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.physics_discriminator = nn.Conv2d(1, 1, kernel_size=1)
        self.data_discriminator = nn.Conv2d(2, 1, kernel_size=1)

    def forward_physics(self, residual_map):
        return self.physics_discriminator(residual_map).mean(dim=(1, 2, 3), keepdim=True)

    def forward_data(self, pair):
        return self.data_discriminator(pair).mean(dim=(1, 2, 3), keepdim=True)


def test_physics_refinement_reduces_laplacian_residual_and_preserves_boundary():
    ny = nx = 12
    dtype = torch.float64
    device = torch.device("cpu")
    hx = hy = 1.0 / float(nx - 1)
    y = torch.linspace(0.0, 1.0, ny, dtype=dtype).view(1, 1, ny, 1)
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype).view(1, 1, 1, nx)

    base = torch.zeros(1, 1, ny, nx, dtype=dtype)
    phi = (x * (1.0 - x) * y * (1.0 - y)).expand_as(base)
    reference = torch.zeros_like(base)
    interior = torch.zeros_like(base)
    interior[:, :, 1:-1, 1:-1] = 1.0
    boundary = 1.0 - interior

    initial_raw = 0.8 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    generator = _TrainableFieldGenerator(initial_raw.expand_as(base))
    discriminator = _TinyDualDiscriminator().to(dtype=dtype)
    laplacian = LaplacianLayer(hx=hx, hy=hy).to(dtype=dtype)
    cfg = FieldTrainerConfig(
        epochs=1,
        batch_size=1,
        gen_lr=1e-2,
        physics_refine_enable=True,
        physics_refine_steps=200,
        physics_refine_lr=1e-2,
        physics_refine_batch_size=1,
        physics_refine_lambda_data=0.0,
        physics_refine_patience=0,
        residual_tolerance_target=1e-4,
    )
    trainer = FieldPIGANTrainer(
        generator=generator,
        discriminator=discriminator,
        laplacian=laplacian,
        base_field=base,
        phi_mask=phi,
        coord_field=None,
        reference_field=reference,
        interior_mask=interior,
        boundary_mask=boundary,
        config=cfg,
        device=device,
    )

    with torch.no_grad():
        before = trainer._residual_mean_abs(laplacian(trainer.predict())).item()

    history = trainer.refine_physics()

    with torch.no_grad():
        pred = trainer.predict()
        after = trainer._residual_mean_abs(laplacian(pred)).item()
        boundary_error = ((pred - reference).abs() * boundary).max().item()

    assert history
    assert history[-1]["physics_refine_residual_mean_abs"] < before
    assert after < before * 0.5
    assert boundary_error < 1e-12
