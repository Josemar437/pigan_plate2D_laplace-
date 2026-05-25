# -*- coding: utf-8 -*-
"""
Perdas adversariais WGAN-GP da PI-GAN para D1 e D2.

L_Dk = E[D(fake)] - E[D(real)] + lambda_gp * GP_k + lambda_drift * E[D(real)^2] + gap_penalty_k
L_advk (gerador) = -E[D_k(fake)]
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn


def gradient_penalty(
    discriminator: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    """GP_k: penalidade de gradiente WGAN-GP (||grad D||_2 ≈ 1)."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device, dtype=real_samples.dtype)
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = discriminator(interpolates)
    grad_outputs = torch.ones_like(d_interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def gap_penalty(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    *,
    max_critic_gap: float,
    critic_gap_penalty: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    gap = |E[D(real)] - E[D(fake)]|; excess = relu(gap - max_critic_gap);
    penalty = critic_gap_penalty * excess^2
    """
    gap = real_scores.mean() - fake_scores.mean()
    excess = torch.relu(gap.abs() - float(max_critic_gap))
    penalty = float(critic_gap_penalty) * excess.pow(2)
    return penalty, gap


def drift_penalty(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    *,
    lambda_drift: float,
) -> torch.Tensor:
    """Regularização de drift: lambda_drift * E[D(real)^2] (com termo fake simétrico no trainer)."""
    return float(lambda_drift) * (real_scores.pow(2).mean() + fake_scores.pow(2).mean())


def discriminator_loss_wgan_gp(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gp: torch.Tensor,
    *,
    lambda_gp: float,
    lambda_drift: float,
    max_critic_gap: float,
    critic_gap_penalty: float,
    include_fake_drift: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L_D = E[D(fake)] - E[D(real)] + lambda_gp*GP + drift + gap_penalty.

    Retorno: (loss, gap, gap_penalty, drift_term)
    """
    gap_pen, gap = gap_penalty(
        real_scores,
        fake_scores,
        max_critic_gap=max_critic_gap,
        critic_gap_penalty=critic_gap_penalty,
    )
    if include_fake_drift:
        drift = drift_penalty(real_scores, fake_scores, lambda_drift=lambda_drift)
    else:
        drift = float(lambda_drift) * real_scores.pow(2).mean()

    loss = (
        (fake_scores.mean() - real_scores.mean())
        + float(lambda_gp) * gp
        + drift
        + gap_pen
    )
    return loss, gap, gap_pen, drift


def generator_adversarial_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """L_adv = -E[D(fake)] para minimização no gerador."""
    return -fake_scores.mean()
