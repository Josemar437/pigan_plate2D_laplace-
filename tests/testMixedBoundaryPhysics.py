# -*- coding: utf-8 -*-
import torch

from src.fdm import solve_laplace_mixed_dirichlet_neumann
from src.models import LaplacianLayer, create_field_pigan_models
from src.trainer import FieldPIGANTrainer, FieldTrainerConfig
from src.utils import (
    build_dirichlet_extension,
    build_hard_constraint_mask,
    build_mixed_boundary_masks,
    create_cartesian_grid,
)


def _linear_solution(nx: int, ny: int, t_left: float, t_right: float) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    return (t_left + (t_right - t_left) * x).repeat(ny, 1)


def test_mixed_fdm_matches_linear_analytic_solution() -> None:
    nx, ny = 23, 17
    t_left, t_right = 200.0, 100.0
    initial = torch.zeros(ny, nx, dtype=torch.float64)

    ref, iterations = solve_laplace_mixed_dirichlet_neumann(
        initial,
        lx=1.0,
        ly=1.0,
        t_left=t_left,
        t_right=t_right,
        tol=1e-12,
        max_iter=5000,
        omega=1.0,
    )

    expected = _linear_solution(nx, ny, t_left, t_right)
    assert iterations < 5000
    assert torch.allclose(ref, expected, atol=1e-10, rtol=0.0)
    assert torch.allclose(ref[0, 1:-1], ref[1, 1:-1], atol=1e-12, rtol=0.0)
    assert torch.allclose(ref[-1, 1:-1], ref[-2, 1:-1], atol=1e-12, rtol=0.0)


def test_mixed_masks_select_only_lateral_dirichlet_and_horizontal_neumann() -> None:
    _, dirichlet, neumann = build_mixed_boundary_masks(
        5, 6, device=torch.device("cpu"), dtype=torch.float32
    )

    assert dirichlet.sum().item() == 10.0
    assert torch.all(dirichlet[:, :, :, 0] == 1.0)
    assert torch.all(dirichlet[:, :, :, -1] == 1.0)
    assert torch.all(dirichlet[:, :, 0, 1:-1] == 0.0)
    assert torch.all(dirichlet[:, :, -1, 1:-1] == 0.0)

    assert neumann.sum().item() == 8.0
    assert torch.all(neumann[:, :, 0, 1:-1] == 1.0)
    assert torch.all(neumann[:, :, -1, 1:-1] == 1.0)
    assert torch.all(neumann[:, :, :, 0] == 0.0)
    assert torch.all(neumann[:, :, :, -1] == 0.0)


def test_hard_constraint_preserves_lateral_dirichlet_without_forcing_top_bottom() -> None:
    nx, ny = 11, 9
    device = torch.device("cpu")
    x_grid, y_grid = create_cartesian_grid(nx, ny, 1.0, 1.0, device=device)
    base = build_dirichlet_extension(
        x_grid,
        y_grid,
        lx=1.0,
        ly=1.0,
        t_left=200.0,
        t_right=100.0,
    ).unsqueeze(0).unsqueeze(0)
    phi = build_hard_constraint_mask(x_grid, y_grid, lx=1.0, ly=1.0).unsqueeze(0).unsqueeze(0)

    generator, _ = create_field_pigan_models(
        generator_config={
            "in_channels": 1,
            "latent_dim": 0,
            "base_channels": 8,
            "depth": 3,
            "use_batch_norm": False,
            "hard_constraint": True,
            "zero_init_final": False,
        },
        discriminator_config={"base_channels": 8},
        device=device,
        logger=None,
    )
    with torch.no_grad():
        generator.final_conv.weight.zero_()
        generator.final_conv.bias.fill_(3.0)

    pred = generator(base, phi)

    assert torch.allclose(pred[:, :, :, 0], base[:, :, :, 0], atol=0.0, rtol=0.0)
    assert torch.allclose(pred[:, :, :, -1], base[:, :, :, -1], atol=0.0, rtol=0.0)
    assert not torch.allclose(pred[:, :, 0, 1:-1], base[:, :, 0, 1:-1], atol=1e-6, rtol=0.0)


def test_trainer_boundary_and_neumann_losses_are_batch_normalized() -> None:
    nx, ny = 8, 6
    device = torch.device("cpu")
    dtype = torch.float32
    x_grid, y_grid = create_cartesian_grid(nx, ny, 1.0, 1.0, device=device, dtype=dtype)
    base = build_dirichlet_extension(
        x_grid,
        y_grid,
        lx=1.0,
        ly=1.0,
        t_left=200.0,
        t_right=100.0,
    ).unsqueeze(0).unsqueeze(0)
    phi = build_hard_constraint_mask(x_grid, y_grid, lx=1.0, ly=1.0).unsqueeze(0).unsqueeze(0)
    interior, dirichlet, neumann = build_mixed_boundary_masks(ny, nx, device=device, dtype=dtype)

    generator, discriminator = create_field_pigan_models(
        generator_config={
            "in_channels": 1,
            "latent_dim": 0,
            "base_channels": 8,
            "depth": 3,
            "use_batch_norm": False,
            "hard_constraint": True,
        },
        discriminator_config={"base_channels": 8},
        device=device,
        logger=None,
    )
    trainer = FieldPIGANTrainer(
        generator=generator,
        discriminator=discriminator,
        laplacian=LaplacianLayer(hx=1.0 / (nx - 1), hy=1.0 / (ny - 1)),
        base_field=base,
        phi_mask=phi,
        coord_field=None,
        reference_field=base,
        interior_mask=interior,
        boundary_mask=dirichlet,
        neumann_mask=neumann,
        config=FieldTrainerConfig(
            epochs=1,
            batch_size=3,
            lambda_adv1=0.0,
            lambda_adv2=0.0,
            lambda_pde=0.0,
            lambda_bc=1.0,
            lambda_neumann=1.0,
            neumann_dy=1.0 / (ny - 1),
            adaptive_lambda_pde=False,
            gradnorm_balance=False,
        ),
        device=device,
        logger=None,
    )

    pred = base.expand(3, -1, -1, -1).clone()
    target = base.expand(3, -1, -1, -1).clone()
    pred[:, :, :, 0] += 2.0
    pred[:, :, 0, 1:-1] += 4.0

    assert torch.allclose(trainer._boundary_mse(pred, target), torch.tensor(2.0))
    expected_neumann = (4.0 / (1.0 / (ny - 1))) ** 2 / 2.0
    assert torch.allclose(trainer._neumann_mse(pred), torch.tensor(expected_neumann))
