# -*- coding: utf-8 -*-
"""Compatibilidade: reexporta perdas canônicas de src.losses."""

from src.losses.adversarial import (
    discriminator_loss_wgan_gp,
    gap_penalty,
    generator_adversarial_loss,
    gradient_penalty,
)
from src.losses.physical import loss_pde, loss_pde_weighted, neumann_loss

# Mantém PIGANLossComputation para testes legados.
from src.physics.pdeResidual import PDE_Residual_Computer
from typing import Dict, Optional, Tuple

import torch


class PIGANLossComputation:
    """Replica componentes de loss da PI-GAN para testes unitários."""

    def __init__(
        self,
        grid_size_x: int,
        grid_size_y: int,
        use_gpu: bool = True,
        pde_kernel_type: str = "centered",
    ) -> None:
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.pde_computer = PDE_Residual_Computer(
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            use_gpu=use_gpu,
            kernel_type=pde_kernel_type,
        )

    def compute_pde_loss(
        self,
        prediction: torch.Tensor,
        reference_residual_scale: float = 1.0,
        use_abs: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual, pde_stats = self.pde_computer.compute_pde_residual(
            prediction,
            use_abs=use_abs,
        )
        loss = residual.mean() / (reference_residual_scale + 1e-8)
        stats = {
            "pde_loss_raw": residual.mean(),
            "pde_loss_normalized": loss,
        }
        stats.update(pde_stats)
        return loss, stats

    def compute_adversarial_loss(
        self,
        discriminator_fake: torch.Tensor,
        discriminator_real: Optional[torch.Tensor] = None,
        loss_type: str = "wasserstein",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if loss_type == "wasserstein":
            loss = generator_adversarial_loss(discriminator_fake)
        elif loss_type == "hinge":
            loss = torch.nn.functional.relu(1.0 - discriminator_fake).mean()
        else:
            raise ValueError(f"loss_type '{loss_type}' não suportado")

        stats = {
            "adv_loss": loss,
            "d_fake_mean": discriminator_fake.mean(),
            "d_fake_std": discriminator_fake.std(),
        }
        if discriminator_real is not None:
            stats["d_real_mean"] = discriminator_real.mean()
            stats["d_real_std"] = discriminator_real.std()
            stats["d_gap"] = (discriminator_real.mean() - discriminator_fake.mean()).abs()
        return loss, stats

    def compute_boundary_loss(
        self,
        prediction: torch.Tensor,
        boundary_target: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        loss_type: str = "l1",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if boundary_mask is None:
            b, c, h, w = prediction.shape
            boundary_mask = torch.zeros_like(prediction, dtype=torch.bool)
            boundary_mask[:, :, 0, :] = True
            boundary_mask[:, :, -1, :] = True
            boundary_mask[:, :, :, 0] = True
            boundary_mask[:, :, :, -1] = True

        diff = (prediction - boundary_target).abs() if loss_type == "l1" else (prediction - boundary_target) ** 2
        masked = diff * boundary_mask.to(dtype=diff.dtype)
        denom = boundary_mask.to(dtype=diff.dtype).sum().clamp_min(1.0)
        loss = masked.sum() / denom
        bc_mae = masked.sum() / denom
        return loss, {"boundary_loss": loss, "bc_mae": bc_mae}

    def compose_generator_loss(
        self,
        loss_pde: torch.Tensor,
        loss_adv: torch.Tensor,
        *,
        loss_bc: torch.Tensor,
        lambda_pde: float,
        lambda_adv: float,
        lambda_bc: float,
        loss_diversity: Optional[torch.Tensor] = None,
        lambda_diversity: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return compose_generator_loss(
            loss_pde,
            loss_adv,
            loss_bc,
            lambda_pde=lambda_pde,
            lambda_adv=lambda_adv,
            lambda_bc=lambda_bc,
            loss_diversity=loss_diversity,
            lambda_diversity=lambda_diversity,
        )


def compose_generator_loss(
    loss_pde: torch.Tensor,
    loss_adv: torch.Tensor,
    loss_bc: torch.Tensor,
    *,
    lambda_pde: float,
    lambda_adv: float,
    lambda_bc: float,
    loss_diversity: Optional[torch.Tensor] = None,
    lambda_diversity: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compõe perda do gerador com pesos explícitos (testes)."""
    pde_term = lambda_pde * loss_pde
    adv_term = lambda_adv * loss_adv
    bc_term = lambda_bc * loss_bc
    total = pde_term + adv_term + bc_term
    breakdown: Dict[str, torch.Tensor] = {
        "loss_total": total,
        "loss_pde_term": pde_term,
        "loss_adv_term": adv_term,
        "loss_bc_term": bc_term,
        "loss_pde_raw": loss_pde,
    }
    if loss_diversity is not None and lambda_diversity > 0.0:
        div_term = lambda_diversity * loss_diversity
        total = total + div_term
        breakdown["loss_diversity_term"] = div_term
    breakdown["loss_total"] = total
    return total, breakdown
