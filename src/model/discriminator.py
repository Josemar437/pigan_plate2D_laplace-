# -*- coding: utf-8 -*-
"""
Discriminadores duais da PI-GAN.

D1: espaço de resíduo físico (Laplace / R0).
D2: espaço de campo pareado com solução FDM de referência [T0, T_ref].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.model.generator import UNetGenerator2D


def _maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    if not bool(enabled):
        return layer
    return nn.utils.spectral_norm(layer)


def _build_discriminator_net(
    in_channels: int,
    base_channels: int,
    *,
    dropout: float = 0.0,
    capacity_scale: float = 1.0,
    use_spectral_norm: bool = False,
) -> nn.Sequential:
    c = max(8, int(round(float(base_channels) * float(capacity_scale))))
    drop = float(max(0.0, min(float(dropout), 0.8)))

    layers: list[nn.Module] = [
        _maybe_spectral_norm(nn.Conv2d(in_channels, c, kernel_size=3, padding=1), use_spectral_norm),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    if drop > 0.0:
        layers.append(nn.Dropout2d(drop))

    in_ch = c
    for out_ch in (c * 2, c * 3):
        layers.extend(
            [
                _maybe_spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    use_spectral_norm,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        if drop > 0.0:
            layers.append(nn.Dropout2d(drop))
        in_ch = out_ch

    layers.extend(
        [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            _maybe_spectral_norm(nn.Linear(c * 3, 1), use_spectral_norm),
        ]
    )
    return nn.Sequential(*layers)


class _BaseDiscriminator2D(nn.Module):
    default_in_channels: int = 1

    def __init__(
        self,
        in_channels: Optional[int] = None,
        base_channels: int = 32,
        *,
        dropout: float = 0.0,
        capacity_scale: float = 1.0,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        resolved_in_channels = (
            int(self.default_in_channels) if in_channels is None else int(in_channels)
        )
        self.net = _build_discriminator_net(
            in_channels=resolved_in_channels,
            base_channels=base_channels,
            dropout=dropout,
            capacity_scale=capacity_scale,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(self, input_map: torch.Tensor) -> torch.Tensor:
        return self.net(input_map)


class PhysicsDiscriminator2D(_BaseDiscriminator2D):
    """D1: critica mapas de resíduo físico R0 no espaço do Laplaciano."""

    default_in_channels: int = 1


class DataDiscriminator2D(_BaseDiscriminator2D):
    """D2: critica pares [T0, T_ref] contra referência FDM."""

    default_in_channels: int = 2


class FieldDualDiscriminator(nn.Module):
    """Contêiner D1 + D2 da formulação min-max PI-GAN."""

    def __init__(
        self,
        physics_discriminator: PhysicsDiscriminator2D,
        data_discriminator: DataDiscriminator2D,
    ) -> None:
        super().__init__()
        self.physics_discriminator = physics_discriminator
        self.data_discriminator = data_discriminator

    def forward_physics(self, residual_map: torch.Tensor) -> torch.Tensor:
        return self.physics_discriminator(residual_map)

    def forward_data(self, pair_map: torch.Tensor) -> torch.Tensor:
        return self.data_discriminator(pair_map)

    def count_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            if p.requires_grad:
                total += p.numel()
        return total


def _initialize_module_weights(module: nn.Module, scheme: str) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weight = getattr(module, "weight_orig", None)
        if weight is None:
            weight = getattr(module, "weight", None)
        if weight is not None:
            if str(scheme).lower() == "xavier":
                nn.init.xavier_uniform_(weight)
            else:
                nn.init.kaiming_normal_(weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.zeros_(bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def create_field_pigan_models(
    generator_config: Dict[str, Any],
    discriminator_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    logger: Optional[Any] = None,
) -> Tuple[UNetGenerator2D, FieldDualDiscriminator]:
    """Fábrica do par gerador + discriminadores duais."""
    gen_cfg = dict(generator_config)
    disc_cfg = dict(discriminator_config)

    generator = UNetGenerator2D(
        in_channels=int(gen_cfg.get("in_channels", 1)),
        latent_dim=int(gen_cfg.get("latent_dim", 0)),
        base_channels=int(gen_cfg.get("base_channels", 32)),
        depth=int(gen_cfg.get("depth", 4)),
        use_batch_norm=bool(gen_cfg.get("use_batch_norm", True)),
        hard_constraint=bool(gen_cfg.get("hard_constraint", True)),
        output_smoothing_steps=int(gen_cfg.get("output_smoothing_steps", 0)),
        output_smoothing_strength=float(gen_cfg.get("output_smoothing_strength", 0.0)),
        activation=str(gen_cfg.get("activation", "silu")),
        pooling=str(gen_cfg.get("pooling", "avg")),
    )

    base_channels = int(disc_cfg.get("base_channels", 32))
    disc_dropout = float(disc_cfg.get("dropout", 0.0))
    disc_capacity_scale = float(disc_cfg.get("capacity_scale", 1.0))
    disc_spectral_norm = bool(disc_cfg.get("use_spectral_norm", False))
    d1 = PhysicsDiscriminator2D(
        in_channels=1,
        base_channels=base_channels,
        dropout=disc_dropout,
        capacity_scale=disc_capacity_scale,
        use_spectral_norm=disc_spectral_norm,
    )
    d2 = DataDiscriminator2D(
        in_channels=2,
        base_channels=base_channels,
        dropout=disc_dropout,
        capacity_scale=disc_capacity_scale,
        use_spectral_norm=disc_spectral_norm,
    )
    dual = FieldDualDiscriminator(d1, d2)

    generator_init = str(gen_cfg.get("init", "kaiming")).strip().lower()
    discriminator_init = str(disc_cfg.get("init", "kaiming")).strip().lower()
    generator.apply(lambda m: _initialize_module_weights(m, generator_init))
    if bool(gen_cfg.get("zero_init_final", True)):
        nn.init.zeros_(generator.final_conv.weight)
        if generator.final_conv.bias is not None:
            nn.init.zeros_(generator.final_conv.bias)
    dual.apply(lambda m: _initialize_module_weights(m, discriminator_init))

    if device is not None:
        generator.to(device)
        dual.to(device)

    if logger:
        gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        disc_params = dual.count_parameters()
        logger.info(
            "Modelos de campo PI-GAN inicializados",
            generator_params=gen_params,
            discriminator_params=disc_params,
            latent_dim=generator.latent_dim,
            hard_constraint=generator.hard_constraint,
            discriminator_dropout=disc_dropout,
            discriminator_capacity_scale=disc_capacity_scale,
            discriminator_spectral_norm=disc_spectral_norm,
            generator_init=generator_init,
            discriminator_init=discriminator_init,
            generator_zero_init_final=bool(gen_cfg.get("zero_init_final", True)),
            device=str(device),
        )

    return generator, dual
