# -*- coding: utf-8 -*-
"""Modelos de campo CNN para PI-GAN fisicamente consistente em Laplace 2D."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_activation(name: str, negative_slope: float = 0.2) -> nn.Module:
    """Constrói ativação do gerador, priorizando opções suaves para PDE."""
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
    """
    Bloco de convolução dupla: duas convoluções 3x3 seguidas por BatchNorm e LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        """
        Inicializa o bloco de convolução.

        Parâmetros:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            use_batch_norm: Se True, aplica Normalização por Lote (Batch Normalization).
        """
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
    """
    Estágio de redução de resolução (Downsampling): Bloco de convolução + MaxPool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
        pooling: str = "max",
    ):
        """
        Inicializa o bloco de downsampling.

        Parâmetros:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            use_batch_norm: Se True, utiliza BatchNorm no bloco de convolução.
        """
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        return features, self.pool(features)


class UpBlock(nn.Module):
    """
    Estágio de aumento de resolução (Upsampling) com conexão de salto (skip connection).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        """
        Inicializa o bloco de upsampling.

        Parâmetros:
            in_channels: Número de canais de entrada (incluindo skip connection).
            out_channels: Número de canais de saída.
            use_batch_norm: Se True, utiliza BatchNorm no bloco de convolução.
        """
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


class UNetGenerator2D(nn.Module):
    """Gerador CNN 2D com restrição de Dirichlet e injeção opcional de latente."""

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
    ):
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
        channels *= 2  # bottleneck output channels

        ups = []
        for skip_ch in reversed(skip_channels):
            # Entrada da etapa de subida: feature upsampled + skip connection.
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

    def _smooth_raw_output(self, raw: torch.Tensor) -> torch.Tensor:
        steps = int(self.output_smoothing_steps)
        strength = float(self.output_smoothing_strength)
        if steps <= 0 or strength <= 0.0:
            return raw

        # O backward de reflection_pad2d em CUDA pode ser não determinístico.
        # O padding é escolhido para preservar determinismo quando necessário.
        pad_mode = self._resolve_smoothing_pad_mode(raw)

        kernel = self._smooth_kernel.to(device=raw.device, dtype=raw.dtype)
        smoothed = raw
        for _ in range(steps):
            smoothed = F.conv2d(
                F.pad(smoothed, [1, 1, 1, 1], mode=pad_mode),
                kernel,
            )
        return (1.0 - strength) * raw + strength * smoothed

    @staticmethod
    def _deterministic_algorithms_enabled() -> Optional[bool]:
        """Verifica, com compatibilidade entre versões, o modo determinístico."""
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
        """
        Define o modo de padding da suavização com segurança para CUDA.

        Em CUDA, quando o modo determinístico está ativo (ou não pode ser
        consultado de forma confiável), usa ``replicate`` para evitar
        não determinismo no backward de ``reflect``.
        """
        deterministic_enabled = cls._deterministic_algorithms_enabled()
        if raw.is_cuda:
            if deterministic_enabled is None:
                return "replicate"
            if deterministic_enabled:
                return "replicate"
        return "reflect"

    def _compose_input(
        self,
        base_field: torch.Tensor,
        coord_field: Optional[torch.Tensor],
    ) -> torch.Tensor:
        expected_coord_channels = max(0, int(self.in_channels) - 1)
        if expected_coord_channels <= 0:
            return base_field
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
        if coord_field.shape[0] != base_field.shape[0]:
            raise ValueError("A dimensão de batch de coord_field deve coincidir com base_field.")
        if coord_field.shape[2:] != base_field.shape[2:]:
            raise ValueError("A dimensão espacial de coord_field deve coincidir com base_field.")
        return torch.cat([base_field, coord_field], dim=1)

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

    def forward(
        self,
        base_field: torch.Tensor,
        phi_mask: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        coord_field: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Executa a passagem para frente (forward pass) do gerador.

        Parâmetros:
            base_field: Extensão das condições de contorno [B, 1, H, W].
            phi_mask: Máscara de distância à fronteira [B, 1, H, W].
            z: Tensor latente opcional [B, latent_dim].
            coord_field: Coordenadas fisicas normalizadas [B, 2, H, W], opcional.

        Retorno:
            Campo de temperatura resultante [B, 1, H, W].
        """
        # base_field e phi_mask devem seguir o formato [B, 1, H, W].
        if base_field.ndim != 4 or base_field.shape[1] != 1:
            raise ValueError("base_field deve ter formato [B,1,H,W].")
        if phi_mask.shape != base_field.shape:
            raise ValueError("phi_mask deve ter o mesmo formato de base_field.")

        input_field = self._compose_input(base_field, coord_field)
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

        raw = self.final_conv(x)
        raw = self._smooth_raw_output(raw)
        if self.hard_constraint:
            # T = g + phi*N mantém Dirichlet exata porque phi=0 na fronteira.
            return base_field + phi_mask * raw
        return raw


class LaplacianLayer(nn.Module):
    """
    Camada que aplica um stencil Laplaciano 3x3 fixo (não treinável) via Conv2D.
    """

    def __init__(self, hx: float, hy: Optional[float] = None):
        """
        Inicializa a camada Laplaciana.

        Parâmetros:
            hx: Tamanho do passo na direção X.
            hy: Tamanho do passo na direção Y (default: mesmo que hx).

        Exceções:
            ValueError: Se hx ou hy forem não positivos.
        """
        super().__init__()
        hy = float(hy if hy is not None else hx)
        hx = float(hx)
        if hx <= 0.0 or hy <= 0.0:
            raise ValueError("hx e hy devem ser positivos.")

        kernel = torch.tensor(
            [
                [0.0, 1.0 / (hy * hy), 0.0],
                [1.0 / (hx * hx), -2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy)), 1.0 / (hx * hx)],
                [0.0, 1.0 / (hy * hy), 0.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # O stencil é aplicado apenas no interior para evitar artefatos de padding.
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=0, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        for p in self.conv.parameters():
            p.requires_grad_(False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Calcula o Laplaciano do campo de entrada.

        Parâmetros:
            field: Campo escalar de entrada [B, 1, H, W].

        Retorno:
            O resíduo Laplaciano calculado apenas nos pontos internos (bordas zero).
        """
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("field deve ter formato [B,1,H,W].")
        if field.shape[2] < 3 or field.shape[3] < 3:
            raise ValueError("As dimensões espaciais de field devem ser >= 3.")

        if self.conv.weight.dtype != field.dtype or self.conv.weight.device != field.device:
            self.conv = self.conv.to(device=field.device, dtype=field.dtype)

        interior = self.conv(field)  # [B,1,H-2,W-2]
        residual = torch.zeros_like(field)
        residual[:, :, 1:-1, 1:-1] = interior
        return residual


def _maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    """
    Aplica Normalização Espectral a uma camada se habilitado.

    Parâmetros:
        layer: Módulo PyTorch a ser normalizado.
        enabled: Se True, aplica spectral_norm.

    Retorno:
        A camada (original ou normalizada).
    """
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
    """
    Constrói o backbone CNN compartilhado para os discriminadores.

    Parâmetros:
        in_channels: Canais de entrada.
        base_channels: Número base de filtros.
        dropout: Taxa de dropout (0.0 a 0.8).
        capacity_scale: Fator de escala para a capacidade (número de filtros).
        use_spectral_norm: Se True, aplica normalização espectral nas convoluções.

    Retorno:
        Sequência de camadas do discriminador.
    """
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
    """
    Classe base compartilhada para o backbone dos discriminadores.
    """
    default_in_channels: int = 1

    def __init__(
        self,
        in_channels: Optional[int] = None,
        base_channels: int = 32,
        *,
        dropout: float = 0.0,
        capacity_scale: float = 1.0,
        use_spectral_norm: bool = False,
    ):
        """
        Inicializa o discriminador base.

        Parâmetros:
            in_channels: Canais de entrada (opcional, usa default_in_channels se None).
            base_channels: Canais base.
            dropout: Taxa de dropout.
            capacity_scale: Escala de capacidade.
            use_spectral_norm: Uso de normalização espectral.
        """
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
    """D1: discrimina mapas de resíduo físico R=∇²T."""

    default_in_channels: int = 1


class DataDiscriminator2D(_BaseDiscriminator2D):
    """D2: compara o campo candidato com o campo numérico de referência."""

    default_in_channels: int = 2


class FieldDualDiscriminator(nn.Module):
    """
    Contêiner para o par de discriminadores: D1 (física) e D2 (dados).
    """

    def __init__(
        self,
        physics_discriminator: PhysicsDiscriminator2D,
        data_discriminator: DataDiscriminator2D,
    ):
        """
        Inicializa o contêiner de discriminadores duais.

        Parâmetros:
            physics_discriminator: Instância do discriminador de resíduos físicos.
            data_discriminator: Instância do discriminador de comparação de dados.
        """
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
    """
    Inicializa pesos de camadas Conv2D, Linear e BatchNorm conforme o esquema.

    Parâmetros:
        module: Módulo PyTorch a ser inicializado.
        scheme: Esquema de inicialização ("xavier" ou "kaiming").
    """
    # Compatível com camadas padrão e com normalização espectral.
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
    """
    Fábrica para criar os modelos gerador e discriminadores da PI-GAN de campo.

    Parâmetros:
        generator_config: Dicionário com parâmetros do gerador.
        discriminator_config: Dicionário com parâmetros dos discriminadores.
        device: Dispositivo onde os modelos serão alocados.
        logger: Logger opcional para diagnóstico dos modelos criados.

    Retorno:
        Uma tupla (gerador, discriminador_dual).
    """
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
            generator_output_smoothing_steps=int(gen_cfg.get("output_smoothing_steps", 0)),
            generator_output_smoothing_strength=float(
                gen_cfg.get("output_smoothing_strength", 0.0)
            ),
            generator_activation=str(gen_cfg.get("activation", "silu")),
            generator_pooling=str(gen_cfg.get("pooling", "avg")),
            device=str(device),
        )

    return generator, dual


__all__ = [
    "UNetGenerator2D",
    "LaplacianLayer",
    "PhysicsDiscriminator2D",
    "DataDiscriminator2D",
    "FieldDualDiscriminator",
    "create_field_pigan_models",
]

