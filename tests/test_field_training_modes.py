# -*- coding: utf-8 -*-
"""Testes de validação da separação de modos de treino da PI-GAN."""

import math

import torch

from src.fdm import solve_laplace_dirichlet
from src.utils import (
    build_dirichlet_extension,
    build_domain_masks,
    build_hard_constraint_mask,
    create_cartesian_grid,
)
from src.models import LaplacianLayer, create_field_pigan_models
from src.trainer import FieldPIGANTrainer, FieldTrainerConfig


def _build_trainer(
    *,
    lambda_adv1: float,
    lambda_adv2: float,
    lambda_pde: float,
    lambda_bc: float,
    latent_dim: int,
) -> FieldPIGANTrainer:
    device = torch.device("cpu")
    nx = 16
    ny = 16
    lx = 1.0
    ly = 1.0

    x_grid, y_grid = create_cartesian_grid(nx, ny, lx, ly, device=device, dtype=torch.float32)
    g_field = build_dirichlet_extension(
        x_grid,
        y_grid,
        lx=lx,
        ly=ly,
        t_left=200.0,
        t_right=100.0,
        boundary_sine_amplitude=0.0,
    )
    phi = build_hard_constraint_mask(x_grid, y_grid, lx=lx, ly=ly)

    boundary_values = g_field.clone()
    boundary_values[1:-1, 1:-1] = 0.0
    ref, _ = solve_laplace_dirichlet(
        boundary_values,
        lx=lx,
        ly=ly,
        tol=1e-6,
        max_iter=2000,
        omega=1.0,
        initial_guess=g_field,
    )

    interior_mask, boundary_mask = build_domain_masks(ny, nx, device=device, dtype=torch.float32)

    generator, discriminator = create_field_pigan_models(
        generator_config={
            "in_channels": 1,
            "latent_dim": int(latent_dim),
            "base_channels": 8,
            "depth": 3,
            "use_batch_norm": False,
            "hard_constraint": True,
        },
        discriminator_config={"base_channels": 8},
        device=device,
        logger=None,
    )

    hx = lx / float(nx - 1)
    hy = ly / float(ny - 1)
    laplacian = LaplacianLayer(hx=hx, hy=hy).to(device)

    cfg = FieldTrainerConfig(
        epochs=1,
        steps_per_epoch=1,
        batch_size=4,
        n_critic=2,
        gen_lr=1e-4,
        disc_lr=1e-4,
        betas=(0.5, 0.9),
        weight_decay=0.0,
        lambda_adv1=float(lambda_adv1),
        lambda_adv2=float(lambda_adv2),
        lambda_pde=float(lambda_pde),
        lambda_bc=float(lambda_bc),
        lambda_gp=0.0,
        use_wgan_gp=False,
        d1_real_noise_std=0.0,
        d2_pair_noise_std=0.0,
        critic_drift=0.0,
        max_critic_gap=8.0,
        critic_gap_penalty=0.0,
        residual_tanh_scale=1.0,
        use_tanh_on_residual=False,
        dynamic_adv_balance=False,
        target_adv_over_pde=0.25,
        adv_scale_ema_beta=0.9,
        adv_scale_min=0.25,
        adv_scale_max=50.0,
        pde_norm_ema_beta=0.99,
        grad_clip=0.0,
        generator_mode="stochastic_pigan" if latent_dim > 0 else "deterministic_adversarial",
        adaptive_lambda_pde=False,
        lambda_pde_raw=0.0,
        residual_mean_abs_target=0.0,
        adv_warmup_epochs=0,
        adv_residual_gate_target=0.0,
        gradnorm_balance=False,
    )

    return FieldPIGANTrainer(
        generator=generator,
        discriminator=discriminator,
        laplacian=laplacian,
        base_field=g_field.unsqueeze(0).unsqueeze(0),
        phi_mask=phi.unsqueeze(0).unsqueeze(0),
        coord_field=None,
        reference_field=ref.unsqueeze(0).unsqueeze(0),
        interior_mask=interior_mask,
        boundary_mask=boundary_mask,
        config=cfg,
        device=device,
        logger=None,
    )


def test_zero_adversarial_behaves_like_pinn_mode() -> None:
    trainer = _build_trainer(
        lambda_adv1=0.0,
        lambda_adv2=0.0,
        lambda_pde=10.0,
        lambda_bc=20.0,
        latent_dim=4,
    )

    metrics = trainer.train_step()
    expected = metrics["g_lambda_pde_dyn"] * metrics["g_pde_raw_penalty"] + 20.0 * metrics["g_bc"]
    assert abs(metrics["g_total"] - expected) < 1e-4

    preds = trainer.predict(num_samples=2)
    assert torch.allclose(preds[0], preds[1], atol=1e-6)


def test_zero_pde_keeps_adversarial_training_active() -> None:
    trainer = _build_trainer(
        lambda_adv1=1.0,
        lambda_adv2=1.0,
        lambda_pde=0.0,
        lambda_bc=0.0,
        latent_dim=4,
    )

    metrics = trainer.train_step()
    expected = metrics["g_adv1"] + metrics["g_adv2"]
    assert abs(metrics["g_total"] - expected) < 1e-3

    for key in ["d1_total", "d2_total", "g_total", "d1_gap", "d2_gap"]:
        assert math.isfinite(metrics[key])

    assert metrics["n_critic"] == 2.0
