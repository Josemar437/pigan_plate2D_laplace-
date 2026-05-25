# -*- coding: utf-8 -*-
"""
Gerador PI-GAN: rede N0 com imposição forte T0 = g + phi * N0.

A parametrização garante Dirichlet exato nas laterais onde phi=0 (Lagaris et al.).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_activation(name: str, negative_slope: float = 0.2) -> nn.Module:
    key = str(name).strip().lower()
    if key in {"leaky_relu", "lrelu", "leaky"}:
        return nn.LeakyReLU(float(negative_slope), inplace=True)
    if key in {"relu"}:
        return nn.ReLU(inplace=True)
    if key in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if key in {"gelu"}:
        return nn.GELU()
    if key in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Ativacao invalida para gerador: {name}")


class ConvBlock(nn.Module):
    """Bloco convolucional duplo 3x3 com normalização opcional."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm,
            )
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_build_activation(activation))
        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm,
            )
        )
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_build_activation(activation))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling: conv + pool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
        pooling: str = "max",
    ) -> None:
        super().__init__()
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
        pool_key = str(pooling).strip().lower()
        if pool_key == "max":
            self.pool = nn.MaxPool2d(kernel_size=2)
        elif pool_key in {"avg", "average"}:
            self.pool = nn.AvgPool2d(kernel_size=2)
        else:
            raise ValueError(f"Pooling invalido para gerador: {pooling}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class UpBlock(nn.Module):
    """Upsampling com skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(
            in_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class HardConstraintLayer(nn.Module):
    """
    Camada final explícita: T0 = g + phi * N0 — imposição forte de Dirichlet.

    phi deve ser zero nas bordas onde Dirichlet é prescrito; então T0|_boundary = g.
    """

    def forward(
        self,
        g: torch.Tensor,
        phi: torch.Tensor,
        N0: torch.Tensor,
    ) -> torch.Tensor:
        # T0 = g + phi * N0 — strong Dirichlet imposition (Lagaris et al., Eq. 4)
        return g + phi * N0


class UNetGenerator2D(nn.Module):
    """Gerador U-Net 2D com saída N0 e composição opcional T0 = g + phi * N0."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 0,
        base_channels: int = 32,
        depth: int = 4,
        use_batch_norm: bool = True,
        hard_constraint: bool = True,
        output_smoothing_steps: int = 0,
        output_smoothing_strength: float = 0.0,
        activation: str = "silu",
        pooling: str = "avg",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.hard_constraint = bool(hard_constraint)
        self.output_smoothing_steps = max(0, int(output_smoothing_steps))
        self.output_smoothing_strength = float(
            max(0.0, min(float(output_smoothing_strength), 1.0))
        )
        self.activation = str(activation).strip().lower()
        self.pooling = str(pooling).strip().lower()
        self.hard_constraint_layer = HardConstraintLayer()
        smooth_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 16.0
        self.register_buffer("_smooth_kernel", smooth_kernel, persistent=False)

        total_in_channels = self.in_channels + max(0, self.latent_dim)
        depth = max(3, min(int(depth), 5))

        self.enc0 = ConvBlock(
            total_in_channels,
            base_channels,
            use_batch_norm=use_batch_norm,
            activation=self.activation,
        )
        if self.pooling == "max":
            self.initial_pool = nn.MaxPool2d(kernel_size=2)
        elif self.pooling in {"avg", "average"}:
            self.initial_pool = nn.AvgPool2d(kernel_size=2)
        else:
            raise ValueError(f"Pooling invalido para gerador: {self.pooling}")

        downs = []
        channels = base_channels
        skip_channels = [base_channels]
        for _ in range(1, depth):
            next_channels = channels * 2
            downs.append(
                DownBlock(
                    channels,
                    next_channels,
                    use_batch_norm=use_batch_norm,
                    activation=self.activation,
                    pooling=self.pooling,
                )
            )
            channels = next_channels
            skip_channels.append(channels)
        self.downs = nn.ModuleList(downs)

        self.bottleneck = ConvBlock(
            channels,
            channels * 2,
            use_batch_norm=use_batch_norm,
            activation=self.activation,
        )
        channels *= 2

        ups = []
        for skip_ch in reversed(skip_channels):
            up_in_channels = channels + skip_ch
            up_out_channels = skip_ch
            ups.append(
                UpBlock(
                    up_in_channels,
                    up_out_channels,
                    use_batch_norm=use_batch_norm,
                    activation=self.activation,
                )
            )
            channels = up_out_channels
        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Conv2d(channels, 1, kernel_size=1)

    def _smooth_raw_output(self, N0: torch.Tensor) -> torch.Tensor:
        steps = int(self.output_smoothing_steps)
        strength = float(self.output_smoothing_strength)
        if steps <= 0 or strength <= 0.0:
            return N0

        pad_mode = self._resolve_smoothing_pad_mode(N0)
        kernel = self._smooth_kernel.to(device=N0.device, dtype=N0.dtype)
        smoothed = N0
        for _ in range(steps):
            smoothed = F.conv2d(
                F.pad(smoothed, [1, 1, 1, 1], mode=pad_mode),
                kernel,
            )
        return (1.0 - strength) * N0 + strength * smoothed

    @staticmethod
    def _deterministic_algorithms_enabled() -> Optional[bool]:
        checkers = [
            getattr(torch, "are_deterministic_algorithms_enabled", None),
            getattr(getattr(torch, "_C", None), "_get_deterministic_algorithms", None),
        ]
        for check in checkers:
            if callable(check):
                try:
                    return bool(check())
                except Exception:
                    continue
        return None

    @classmethod
    def _resolve_smoothing_pad_mode(cls, raw: torch.Tensor) -> str:
        deterministic_enabled = cls._deterministic_algorithms_enabled()
        if raw.is_cuda:
            if deterministic_enabled is None:
                return "replicate"
            if deterministic_enabled:
                return "replicate"
        return "reflect"

    def _compose_input(
        self,
        g: torch.Tensor,
        coord_field: Optional[torch.Tensor],
    ) -> torch.Tensor:
        expected_coord_channels = max(0, int(self.in_channels) - 1)
        if expected_coord_channels <= 0:
            return g
        if coord_field is None:
            raise ValueError(
                "coord_field é obrigatório quando in_channels > 1 "
                "(coordenadas físicas esperadas)."
            )
        if coord_field.ndim != 4:
            raise ValueError("coord_field deve ter formato [B,C,H,W].")
        if coord_field.shape[1] != expected_coord_channels:
            raise ValueError(
                f"coord_field deve ter {expected_coord_channels} canais; recebido "
                f"{coord_field.shape[1]}."
            )
        if coord_field.shape[0] != g.shape[0]:
            raise ValueError("A dimensão de batch de coord_field deve coincidir com g.")
        if coord_field.shape[2:] != g.shape[2:]:
            raise ValueError("A dimensão espacial de coord_field deve coincidir com g.")
        return torch.cat([g, coord_field], dim=1)

    def _inject_latent(self, input_field: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        if self.latent_dim <= 0:
            return input_field
        if z is None:
            raise ValueError("O tensor latente z é obrigatório quando latent_dim > 0.")
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z deve ter formato [batch, {self.latent_dim}]; recebido {tuple(z.shape)}."
            )
        b, _, h, w = input_field.shape
        if z.shape[0] != b:
            raise ValueError("A dimensão de batch de z deve coincidir com a entrada.")

        z_map = z.view(b, self.latent_dim, 1, 1).expand(b, self.latent_dim, h, w)
        return torch.cat([input_field, z_map], dim=1)

    def decode_N0(
        self,
        g: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        coord_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Saída bruta da rede N0 antes da imposição de contorno."""
        input_field = self._compose_input(g, coord_field)
        x = self._inject_latent(input_field, z)

        skips = []
        x = self.enc0(x)
        skips.append(x)
        x = self.initial_pool(x)

        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)

        x = self.bottleneck(x)
        for idx, up in enumerate(self.ups):
            skip = skips[-(idx + 1)]
            x = up(x, skip)

        N0 = self.final_conv(x)
        return self._smooth_raw_output(N0)

    def forward(
        self,
        g: torch.Tensor,
        phi: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        coord_field: Optional[torch.Tensor] = None,
        *,
        base_field: Optional[torch.Tensor] = None,
        phi_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Retorna T0 = g + phi * N0 quando hard_constraint=True; caso contrário, N0.

        Parâmetros:
            g: Extensão das condições de contorno [B, 1, H, W].
            phi: Máscara que se anula nas bordas Dirichlet [B, 1, H, W].
            z: Vetor latente opcional.
            coord_field: Coordenadas físicas opcionais.
            base_field: Alias legado de g.
            phi_mask: Alias legado de phi.
        """
        if base_field is not None:
            g = base_field
        if phi_mask is not None:
            phi = phi_mask

        if g.ndim != 4 or g.shape[1] != 1:
            raise ValueError("g deve ter formato [B,1,H,W].")
        if phi.shape != g.shape:
            raise ValueError("phi deve ter o mesmo formato de g.")

        N0 = self.decode_N0(g, z=z, coord_field=coord_field)
        if self.hard_constraint:
            T0 = self.hard_constraint_layer(g, phi, N0)
            return T0
        return N0
