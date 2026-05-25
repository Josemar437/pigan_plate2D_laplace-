# -*- coding: utf-8 -*-
"""Perdas canônicas da PI-GAN (física e adversarial)."""

from src.losses.adversarial import (
    discriminator_loss_wgan_gp,
    gap_penalty,
    generator_adversarial_loss,
    gradient_penalty,
)
from src.losses.physical import loss_pde, loss_pde_weighted, neumann_loss

__all__ = [
    "loss_pde",
    "loss_pde_weighted",
    "neumann_loss",
    "gradient_penalty",
    "gap_penalty",
    "discriminator_loss_wgan_gp",
    "generator_adversarial_loss",
]
