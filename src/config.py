# -*- coding: utf-8 -*-
"""Configurações do projeto PI-GAN (SystemConfig e ExperimentConfig)."""
import gc
import importlib
import logging
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

psutil = None
try:
    psutil = importlib.import_module("psutil")
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn(
        "`psutil` não encontrado. O monitoramento de CPU e a otimização de "
        "'num_workers' estarão desativados. Instale com: pip install psutil"
    )

structlog = None
try:
    structlog = importlib.import_module("structlog")
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    warnings.warn(
        "`structlog` não encontrado. Usando logger padrão. Para logs "
        "estruturados, instale com: pip install structlog"
    )


class PIGANError(Exception):
    """Exceção base para erros específicos do projeto PI-GAN."""


class GPUMemoryError(PIGANError):
    """Lançada quando ocorrem erros relacionados à memória da GPU."""


class ModelConfigurationError(PIGANError):
    """Lançada para erros em parâmetros de configuração de modelos ou sistema."""


class EnhancedLogger:
    """
    Logger compatível com structlog quando este não está disponível.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Inicializa o logger.

        Parâmetros:
            logger: Instância do logger padrão do Python.
        """
        self.logger = logger

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """
        Formata uma mensagem de log com argumentos chave-valor.

        Parâmetros:
            message: Mensagem principal.
            **kwargs: Metadados adicionais.

        Retorno:
            Mensagem formatada como string.
        """
        if not kwargs:
            return message
        parts = [message]
        for key, value in kwargs.items():
            parts.append(f"  {key}: {value}")
        return "\n".join(parts)

    def info(self, message: str, **kwargs: Any) -> None:
        """Loga uma mensagem de nível INFO."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Loga uma mensagem de nível WARNING."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Loga uma mensagem de nível ERROR."""
        self.logger.error(self._format_message(message, **kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        """Loga uma mensagem de nível DEBUG."""
        self.logger.debug(self._format_message(message, **kwargs))


@dataclass
class SystemConfig:
    """
    Configurações globais do sistema, hardware e otimizações de baixo nível.
    """

    use_gpu: bool = True
    gpu_id: Optional[int] = None
    mixed_precision: bool = False
    compile_model: bool = False

    max_memory_usage: float = 0.9
    dynamic_batch_size: bool = True
    gradient_checkpointing: bool = False
    pin_memory: bool = True

    use_tf32: bool = False
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2

    log_gpu_memory: bool = True
    profile_performance: bool = False
    detect_anomaly: bool = False
    log_file: Optional[str] = None

    use_double: bool = False
    eps: float = 1e-16
    seed: int = 42
    deterministic_run: bool = True
    deterministic_warn_only: bool = False

    def __post_init__(self) -> None:
        """
        Executa validações pós-inicialização e configura o sistema para reproduzibilidade.

        Exceções:
            ModelConfigurationError: Se algum parâmetro de sistema for inválido.
        """
        if self.max_memory_usage <= 0 or self.max_memory_usage > 1.0:
            raise ModelConfigurationError("max_memory_usage deve estar entre 0 e 1")
        if self.num_workers < 0:
            raise ModelConfigurationError("num_workers deve ser >= 0")
        if self.seed < 0:
            raise ModelConfigurationError("seed deve ser >= 0")

        if bool(self.deterministic_run):
            # Mantém o comportamento de CUDA/cuDNN determinístico entre execuções.
            self.cudnn_benchmark = False
            self.cudnn_deterministic = True
            self.use_tf32 = False
            if self.gpu_id is None:
                # Evita mudanças causadas pela auto-seleção da GPU "mais livre".
                self.gpu_id = 0

        if HAS_PSUTIL and self.num_workers == 4:
            cpu_count = psutil.cpu_count(logical=False)
            if cpu_count:
                self.num_workers = min(cpu_count, 8)

    @property
    def allow_cpu(self) -> bool:
        """
        Retorna se a execução em CPU é permitida via variável de ambiente.
        """
        return os.environ.get("PIGAN_ALLOW_CPU") == "1"


@dataclass
class ExperimentConfig:
    """
    Configurações do experimento (modelo, treinamento, pesos, hiperparâmetros).

    Nota (Laplace 2D estacionária):
    - `generator_mode='stochastic_pigan'` exige `latent_dim > 0` (PI-GAN gerativa).
    - `generator_mode='deterministic_adversarial'` força `latent_dim=0` (modo não estocástico).
    - Saída do gerador permanece linear (sem ativação final).
    """

    # === Parâmetros físicos ===
    T_LEFT: float = 200.0
    T_RIGHT: float = 100.0
    LX: float = 1.0
    LY: float = 1.0
    boundary_sine_amplitude: float = 1.0

    # === Arquitetura do gerador ===
    # generator_mode:
    # - "stochastic_pigan": PI-GAN gerativa classica (z ~ N(0,1), latent_dim > 0)
    # - "deterministic_adversarial": problema deterministico com regularizacao adversarial
    generator_mode: str = "stochastic_pigan"
    latent_dim: int = 8
    generator_base_channels: int = 12
    generator_depth: int = 3
    generator_use_batch_norm: bool = False
    generator_zero_init_final: bool = True
    generator_output_smoothing_steps: int = 2
    generator_output_smoothing_strength: float = 0.35
    generator_activation: str = "silu"
    generator_pooling: str = "avg"
    use_physical_coordinates: bool = True
    hard_constraint_bc: bool = True

    # === Arquitetura do discriminador ===
    discriminator_base_channels: int = 12
    discriminator_capacity_scale: float = 0.72
    discriminator_dropout: float = 0.20
    discriminator_spectral_norm: bool = True

    # === Treinamento ===
    epochs: int = 4000
    batch_size: int = 16
    gen_lr: float = 1.15e-4
    disc_lr: float = 8.625e-5
    disc_lr_d1: Optional[float] = 8.625e-5
    disc_lr_d2: Optional[float] = 8.625e-5
    betas: Tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 1e-5
    n_critic: int = 1
    disc_update_every: int = 1
    # Para evitar explosões de gradiente nas fronteiras
    max_grad_norm: float = 1.85
    steps_per_epoch: int = 1

    # === Domínio de campo 2D ===
    grid_size_x: int = 32
    grid_size_y: int = 32
    fdm_tol: float = 1e-12
    fdm_max_iter: int = 100000
    fdm_omega: float = 1.0

    lambda_adv1: float = 5.0e-1
    lambda_adv2: float = 2.0e-1
    lambda_pde: float = 37.0
    lambda_bc: float = 20.0
    lambda_gp: float = 8.0
    lambda_gp_d1: Optional[float] = 8.0
    lambda_gp_d2: Optional[float] = 8.0

    use_wgan_gp: bool = True
    d1_real_noise_std: float = 0.0
    d2_pair_noise_std: float = 5e-3
    use_reference_discriminator: bool = True
    d1_real_residual_mode: str = "reference"
    critic_drift: float = 5e-3
    max_critic_gap: float = 11.0
    critic_gap_penalty: float = 0.09
    residual_tanh_scale: float = 0.0
    use_tanh_on_residual: bool = False
    dynamic_adv_balance: bool = True
    target_adv_over_pde: float = 0.20
    adv_scale_ema_beta: float = 0.9
    adv_scale_min: float = 0.50
    adv_scale_max: float = 20.0
    pde_norm_ema_beta: float = 0.99
    lambda_pde_raw: float = 0.0
    residual_mean_abs_target: float = 1.0
    adv_residual_gate_target: float = 0.01
    adv_residual_gate_min: float = 0.09
    adv_residual_gate_hysteresis: bool = True
    adv_residual_gate_off_threshold: float = 1e-4
    adv_residual_gate_power: float = 1.2
    adv_warmup_epochs: int = 120
    critic_pause_on_overgap: bool = True
    critic_pause_gap_factor: float = 1.20
    critic_resume_gap_factor: float = 0.36
    adv_stagnation_boost: bool = True
    adv_stagnation_patience: int = 50
    adv_stagnation_rel_tol: float = 5e-3
    adv_stagnation_boost_factor: float = 1.4
    adv_stagnation_min_gate: float = 0.50
    adv_stagnation_cooldown: int = 8
    adv_progressive_from_history: bool = True
    adv_progressive_history_path: Optional[str] = None
    adv_progressive_metric_key: str = "g_residual_mean_abs"
    adv_progressive_stagnation_patience: int = 150
    adv_progressive_stagnation_rel_tol: float = 5e-3
    adv_progressive_min_epoch: int = 2400
    adv_progressive_epoch_shift: int = 25
    adv_progressive_ramp_epochs: int = 600
    adv_progressive_max_multiplier: float = 2.8
    adv_progressive_power: float = 1.25
    pde_corner_sampling_ratio: float = 0.10
    pde_corner_band_points: int = 2
    precision_refine_enable: bool = True
    precision_refine_start_epoch: int = 0
    precision_refine_use_adv_progressive_start: bool = True
    precision_refine_n_critic: int = 3
    precision_refine_n_critic_ramp_epochs: int = 500
    precision_refine_lambda_pde_max_scale: float = 0.72
    adaptive_lambda_pde: bool = True
    residual_tolerance_target: float = 1e-3
    residual_scale_reference: float = 1.0e-2
    lambda_pde_growth_exponent: float = 0.60
    lambda_pde_min: float = 19.0
    lambda_pde_max: float = 98.0
    lambda_pde_ema_beta: float = 0.9
    gradnorm_balance: bool = True
    gradnorm_target_adv_to_pde: float = 0.35
    gradnorm_ema_beta: float = 0.9
    gradnorm_scale_min: float = 0.05
    gradnorm_scale_max: float = 1.0
    divergence_window: int = 16
    divergence_ratio_threshold: float = 1.2
    divergence_patience: int = 2
    lr_drop_factor: float = 0.5
    max_lr_drops: int = 10
    plateau_scheduler_enabled: bool = True
    plateau_metric_key: str = "g_residual_mean_abs"
    plateau_mode: str = "min"
    plateau_patience: int = 40
    plateau_factor: float = 0.5
    plateau_min_delta: float = 1e-5
    plateau_cooldown: int = 10
    plateau_max_drops: int = 10
    plateau_reduce_discriminators: bool = True
    activation_abs_limit: float = 1e6
    residual_hist_bins: int = 12
    early_stop_on_nonfinite: bool = True
    generator_init: str = "kaiming"
    discriminator_init: str = "kaiming"

    save_frequency: int = 250
    checkpoint_dir: Optional[str] = None

    # === Análise / Visualização ===
    analysis_num_samples: int = 50
    dpi: int = 600
    generate_plots: bool = True
    generate_extended_plots: bool = True
    resume_checkpoint: Optional[str] = None
    strict_checkpoint_loading: bool = True

    def __post_init__(self) -> None:
        valid_modes = {"stochastic_pigan", "deterministic_adversarial"}
        if self.generator_mode not in valid_modes:
            raise ModelConfigurationError(
                f"generator_mode inválido: {self.generator_mode}. "
                f"Use um de {sorted(valid_modes)}."
            )

        if float(self.lambda_adv1) < 0.0 or float(self.lambda_adv2) < 0.0:
            raise ModelConfigurationError("lambda_adv1/lambda_adv2 devem ser >= 0.")
        if float(self.boundary_sine_amplitude) < 0.0:
            raise ModelConfigurationError("boundary_sine_amplitude deve ser >= 0.")

        if float(self.gen_lr) <= 0.0 or float(self.disc_lr) <= 0.0:
            raise ModelConfigurationError("gen_lr e disc_lr devem ser > 0.")
        if self.disc_lr_d1 is not None and float(self.disc_lr_d1) <= 0.0:
            raise ModelConfigurationError("disc_lr_d1 deve ser > 0 quando definido.")
        if self.disc_lr_d2 is not None and float(self.disc_lr_d2) <= 0.0:
            raise ModelConfigurationError("disc_lr_d2 deve ser > 0 quando definido.")

        if float(self.lambda_bc) < 0.0:
            raise ModelConfigurationError("lambda_bc deve ser >= 0.")

        if int(self.n_critic) < 1:
            raise ModelConfigurationError("n_critic deve ser >= 1.")
        if int(self.disc_update_every) < 1:
            raise ModelConfigurationError("disc_update_every deve ser >= 1.")
        if float(self.lambda_gp) < 0.0:
            raise ModelConfigurationError("lambda_gp deve ser >= 0.")
        if self.lambda_gp_d1 is not None and float(self.lambda_gp_d1) < 0.0:
            raise ModelConfigurationError("lambda_gp_d1 deve ser >= 0 quando definido.")
        if self.lambda_gp_d2 is not None and float(self.lambda_gp_d2) < 0.0:
            raise ModelConfigurationError("lambda_gp_d2 deve ser >= 0 quando definido.")
        if float(self.lambda_pde) < 0.0:
            raise ModelConfigurationError("lambda_pde deve ser >= 0.")
        if float(self.lambda_pde_raw) < 0.0:
            raise ModelConfigurationError("lambda_pde_raw deve ser >= 0.")
        if float(self.residual_tolerance_target) <= 0.0:
            raise ModelConfigurationError("residual_tolerance_target deve ser > 0.")
        if float(self.residual_scale_reference) <= 0.0:
            raise ModelConfigurationError("residual_scale_reference deve ser > 0.")
        if float(self.lambda_pde_growth_exponent) <= 0.0:
            raise ModelConfigurationError("lambda_pde_growth_exponent deve ser > 0.")
        if float(self.lambda_pde_min) <= 0.0:
            raise ModelConfigurationError("lambda_pde_min deve ser > 0.")
        if float(self.lambda_pde_max) < float(self.lambda_pde_min):
            raise ModelConfigurationError("lambda_pde_max deve ser >= lambda_pde_min.")
        if not (0.0 <= float(self.lambda_pde_ema_beta) < 1.0):
            raise ModelConfigurationError("lambda_pde_ema_beta deve estar em [0, 1).")
        if float(self.residual_mean_abs_target) < 0.0:
            raise ModelConfigurationError("residual_mean_abs_target deve ser >= 0.")
        if float(self.adv_residual_gate_target) < 0.0:
            raise ModelConfigurationError("adv_residual_gate_target deve ser >= 0.")
        if not (0.0 <= float(self.adv_residual_gate_min) <= 1.0):
            raise ModelConfigurationError("adv_residual_gate_min deve estar em [0, 1].")
        if float(self.adv_residual_gate_off_threshold) < 0.0:
            raise ModelConfigurationError("adv_residual_gate_off_threshold deve ser >= 0.")
        if float(self.adv_residual_gate_power) <= 0.0:
            raise ModelConfigurationError("adv_residual_gate_power deve ser > 0.")
        if bool(self.adv_residual_gate_hysteresis) and float(self.adv_residual_gate_target) > 0.0:
            if float(self.adv_residual_gate_off_threshold) > float(self.adv_residual_gate_target):
                raise ModelConfigurationError(
                    "Com histerese ativa, adv_residual_gate_off_threshold deve ser <= "
                    "adv_residual_gate_target."
                )
        if int(self.adv_warmup_epochs) < 0:
            raise ModelConfigurationError("adv_warmup_epochs deve ser >= 0.")
        if float(self.critic_pause_gap_factor) < 1.0:
            raise ModelConfigurationError("critic_pause_gap_factor deve ser >= 1.")
        if float(self.critic_resume_gap_factor) < 0.0:
            raise ModelConfigurationError("critic_resume_gap_factor deve ser >= 0.")
        if float(self.critic_resume_gap_factor) > float(self.critic_pause_gap_factor):
            raise ModelConfigurationError(
                "critic_resume_gap_factor deve ser <= critic_pause_gap_factor."
            )
        if int(self.adv_stagnation_patience) < 1:
            raise ModelConfigurationError("adv_stagnation_patience deve ser >= 1.")
        if float(self.adv_stagnation_rel_tol) < 0.0:
            raise ModelConfigurationError("adv_stagnation_rel_tol deve ser >= 0.")
        if float(self.adv_stagnation_boost_factor) < 1.0:
            raise ModelConfigurationError("adv_stagnation_boost_factor deve ser >= 1.")
        if not (0.0 <= float(self.adv_stagnation_min_gate) <= 1.0):
            raise ModelConfigurationError("adv_stagnation_min_gate deve estar em [0, 1].")
        if int(self.adv_stagnation_cooldown) < 0:
            raise ModelConfigurationError("adv_stagnation_cooldown deve ser >= 0.")
        if int(self.adv_progressive_stagnation_patience) < 1:
            raise ModelConfigurationError("adv_progressive_stagnation_patience deve ser >= 1.")
        if float(self.adv_progressive_stagnation_rel_tol) < 0.0:
            raise ModelConfigurationError("adv_progressive_stagnation_rel_tol deve ser >= 0.")
        if int(self.adv_progressive_min_epoch) < 1:
            raise ModelConfigurationError("adv_progressive_min_epoch deve ser >= 1.")
        if int(self.adv_progressive_ramp_epochs) < 1:
            raise ModelConfigurationError("adv_progressive_ramp_epochs deve ser >= 1.")
        if float(self.adv_progressive_max_multiplier) < 1.0:
            raise ModelConfigurationError("adv_progressive_max_multiplier deve ser >= 1.")
        if float(self.adv_progressive_power) <= 0.0:
            raise ModelConfigurationError("adv_progressive_power deve ser > 0.")
        if not (0.0 <= float(self.pde_corner_sampling_ratio) < 1.0):
            raise ModelConfigurationError("pde_corner_sampling_ratio deve estar em [0,1).")
        if int(self.pde_corner_band_points) < 1:
            raise ModelConfigurationError("pde_corner_band_points deve ser >= 1.")
        if int(self.precision_refine_start_epoch) < 0:
            raise ModelConfigurationError("precision_refine_start_epoch deve ser >= 0.")
        if int(self.precision_refine_n_critic) < 1:
            raise ModelConfigurationError("precision_refine_n_critic deve ser >= 1.")
        if int(self.precision_refine_n_critic_ramp_epochs) < 1:
            raise ModelConfigurationError("precision_refine_n_critic_ramp_epochs deve ser >= 1.")
        if not (0.0 < float(self.precision_refine_lambda_pde_max_scale) <= 1.0):
            raise ModelConfigurationError(
                "precision_refine_lambda_pde_max_scale deve estar em (0, 1]."
            )
        if float(self.gradnorm_target_adv_to_pde) <= 0.0:
            raise ModelConfigurationError("gradnorm_target_adv_to_pde deve ser > 0.")
        if not (0.0 <= float(self.gradnorm_ema_beta) < 1.0):
            raise ModelConfigurationError("gradnorm_ema_beta deve estar em [0, 1).")
        if float(self.gradnorm_scale_min) <= 0.0:
            raise ModelConfigurationError("gradnorm_scale_min deve ser > 0.")
        if float(self.gradnorm_scale_max) < float(self.gradnorm_scale_min):
            raise ModelConfigurationError("gradnorm_scale_max deve ser >= gradnorm_scale_min.")
        if int(self.divergence_window) < 4:
            raise ModelConfigurationError("divergence_window deve ser >= 4.")
        if float(self.divergence_ratio_threshold) <= 1.0:
            raise ModelConfigurationError("divergence_ratio_threshold deve ser > 1.")
        if int(self.divergence_patience) < 1:
            raise ModelConfigurationError("divergence_patience deve ser >= 1.")
        if not (0.0 < float(self.lr_drop_factor) < 1.0):
            raise ModelConfigurationError("lr_drop_factor deve estar em (0,1).")
        if int(self.max_lr_drops) < 0:
            raise ModelConfigurationError("max_lr_drops deve ser >= 0.")
        if str(self.plateau_mode).strip().lower() not in {"min", "max"}:
            raise ModelConfigurationError("plateau_mode deve ser 'min' ou 'max'.")
        if not str(self.plateau_metric_key).strip():
            raise ModelConfigurationError("plateau_metric_key nao pode ser vazio.")
        if int(self.plateau_patience) < 1:
            raise ModelConfigurationError("plateau_patience deve ser >= 1.")
        if not (0.0 < float(self.plateau_factor) < 1.0):
            raise ModelConfigurationError("plateau_factor deve estar em (0,1).")
        if float(self.plateau_min_delta) < 0.0:
            raise ModelConfigurationError("plateau_min_delta deve ser >= 0.")
        if int(self.plateau_cooldown) < 0:
            raise ModelConfigurationError("plateau_cooldown deve ser >= 0.")
        if int(self.plateau_max_drops) < 0:
            raise ModelConfigurationError("plateau_max_drops deve ser >= 0.")
        if float(self.activation_abs_limit) <= 0.0:
            raise ModelConfigurationError("activation_abs_limit deve ser > 0.")
        if int(self.residual_hist_bins) < 4:
            raise ModelConfigurationError("residual_hist_bins deve ser >= 4.")
        if float(self.discriminator_capacity_scale) <= 0.0:
            raise ModelConfigurationError("discriminator_capacity_scale deve ser > 0.")
        if not (0.0 <= float(self.discriminator_dropout) < 1.0):
            raise ModelConfigurationError("discriminator_dropout deve estar em [0,1).")
        if int(self.generator_output_smoothing_steps) < 0:
            raise ModelConfigurationError("generator_output_smoothing_steps deve ser >= 0.")
        if not (0.0 <= float(self.generator_output_smoothing_strength) <= 1.0):
            raise ModelConfigurationError(
                "generator_output_smoothing_strength deve estar em [0,1]."
            )
        valid_generator_activations = {"leaky_relu", "relu", "silu", "gelu", "tanh"}
        if str(self.generator_activation).strip().lower() not in valid_generator_activations:
            raise ModelConfigurationError(
                "generator_activation invalida. Use uma de "
                f"{sorted(valid_generator_activations)}."
            )
        valid_generator_pooling = {"max", "avg"}
        if str(self.generator_pooling).strip().lower() not in valid_generator_pooling:
            raise ModelConfigurationError(
                "generator_pooling invalido. Use uma de "
                f"{sorted(valid_generator_pooling)}."
            )
        if str(self.generator_init).lower() not in {"kaiming", "xavier"}:
            raise ModelConfigurationError("generator_init deve ser 'kaiming' ou 'xavier'.")
        if str(self.discriminator_init).lower() not in {"kaiming", "xavier"}:
            raise ModelConfigurationError("discriminator_init deve ser 'kaiming' ou 'xavier'.")
        if float(self.max_critic_gap) <= 0.0:
            raise ModelConfigurationError("max_critic_gap deve ser > 0.")
        if float(self.critic_gap_penalty) < 0.0:
            raise ModelConfigurationError("critic_gap_penalty deve ser >= 0.")
        if float(self.target_adv_over_pde) <= 0.0:
            raise ModelConfigurationError("target_adv_over_pde deve ser > 0.")
        if not (0.0 <= float(self.adv_scale_ema_beta) < 1.0):
            raise ModelConfigurationError("adv_scale_ema_beta deve estar em [0, 1).")
        if float(self.adv_scale_min) <= 0.0:
            raise ModelConfigurationError("adv_scale_min deve ser > 0.")
        if float(self.adv_scale_max) < float(self.adv_scale_min):
            raise ModelConfigurationError("adv_scale_max deve ser >= adv_scale_min.")
        if not (0.0 <= float(self.pde_norm_ema_beta) < 1.0):
            raise ModelConfigurationError("pde_norm_ema_beta deve estar em [0, 1).")
        if str(self.d1_real_residual_mode) not in {"reference", "zero"}:
            raise ModelConfigurationError(
                "d1_real_residual_mode inválido. Use 'reference' ou 'zero'."
            )
        if not bool(getattr(self, "use_reference_discriminator", True)):
            if float(self.lambda_adv2) != 0.0:
                warnings.warn(
                    "use_reference_discriminator=False: forçando lambda_adv2=0 para "
                    "modo PI-GAN sem discriminador de referência."
                )
            self.lambda_adv2 = 0.0

        if self.generator_mode == "stochastic_pigan" and int(self.latent_dim) <= 0:
            raise ModelConfigurationError(
                "Modo stochastic_pigan exige latent_dim > 0 para ser gerativo. "
                "Para modo deterministico use generator_mode='deterministic_adversarial'."
            )
        if self.generator_mode == "deterministic_adversarial":
            # Modo explicitamente nao estocastico.
            self.latent_dim = 0


class GPUMemoryManager:
    """
    Gerencia e monitora o uso de memória da GPU.
    """

    def __init__(self, device: torch.device, max_usage: float = 0.85) -> None:
        """
        Inicializa o gerenciador de memória GPU.

        Parâmetros:
            device: Dispositivo PyTorch a ser monitorado.
            max_usage: Porcentagem máxima de uso de memória sugerida.
        """
        self.device = device
        self.max_usage = max_usage
        self.is_cuda = device.type == "cuda"

    def get_memory_info(self) -> Dict[str, float]:
        """
        Coleta informações detalhadas sobre a memória GPU.

        Retorno:
            Dicionário contendo estatísticas de uso em GB e porcentagem.
        """
        if not self.is_cuda:
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": float("inf"),
                "total_gb": float("inf"),
                "usage_percent": 0.0,
            }

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        free = total - allocated
        usage_percent = (allocated / total) * 100

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "total_gb": total,
            "usage_percent": usage_percent,
        }

    def optimize_batch_size(self, base_batch_size: int, model_memory_gb: float = 1.0) -> int:
        """
        Sugere um tamanho de lote otimizado com base na memória disponível.

        Parâmetros:
            base_batch_size: Tamanho de lote desejado/máximo.
            model_memory_gb: Estimativa de memória do modelo em GB.

        Retorno:
            Tamanho de lote (batch size) ajustado.
        """
        if not self.is_cuda:
            return base_batch_size

        memory_info = self.get_memory_info()
        available_memory = memory_info["free_gb"] * self.max_usage
        memory_per_batch = model_memory_gb * 20

        if memory_per_batch > 0:
            max_batch_size = max(1, int(available_memory / memory_per_batch))
            optimal_batch_size = min(base_batch_size, max_batch_size)
        else:
            optimal_batch_size = base_batch_size

        return optimal_batch_size

    def clear_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def check_memory_usage(self, threshold: float = 0.9) -> bool:
        if not self.is_cuda:
            return False
        memory_info = self.get_memory_info()
        return memory_info["usage_percent"] / 100 > threshold


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> Any:
    """
    Configura o sistema de logging do projeto.

    Parâmetros:
        level: Nível de severidade do log (ex: logging.INFO).
        log_file: Caminho opcional para arquivo de log.

    Retorno:
        Um logger configurado (structlog se disponível, senão EnhancedLogger).
    """
    if HAS_STRUCTLOG:
        console_renderer = structlog.processors.KeyValueRenderer(
            key_order=["timestamp", "level", "event"],
            drop_missing=True,
            sort_keys=False,
        )
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.format_exc_info,
            console_renderer,
        ]
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root_logger = logging.getLogger("pigan")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.propagate = False

    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger.addHandler(file_handler)

    if HAS_STRUCTLOG:
        return structlog.get_logger("pigan")
    return EnhancedLogger(root_logger)


def setup_torch_optimizations(config: SystemConfig, logger: Optional[Any] = None) -> None:
    """
    Configura otimizações globais do PyTorch e sementes de aleatoriedade.

    Parâmetros:
        config: Configuração de sistema.
        logger: Logger para registrar as otimizações aplicadas.
    """
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 64
    except (ImportError, AttributeError):
        pass

    if config.use_double:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    if bool(getattr(config, "deterministic_run", False)):
        # Necessário para kernels GEMM determinísticos em diversas versões do PyTorch.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=bool(getattr(config, "deterministic_warn_only", False)),
            )
        except Exception:
            # Compatibilidade com versões antigas do PyTorch.
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    if not config.use_gpu:
        if HAS_PSUTIL:
            torch.set_num_threads(min(psutil.cpu_count(), 8))
        else:
            torch.set_num_threads(4)

    torch.autograd.set_detect_anomaly(config.detect_anomaly)

    if config.compile_model and hasattr(torch, "compile"):
        try:
            test_model = torch.nn.Linear(2, 1)
            torch.compile(test_model)
            if logger:
                logger.info("Compilação JIT habilitada")
        except Exception as exc:
            config.compile_model = False
            if logger:
                logger.warning("Compilação JIT desabilitada", error=str(exc))


def _configure_cuda_optimizations(config: SystemConfig) -> None:
    """
    Configura flags específicas de baixo nível para kernels CUDA (backend cuDNN).

    Parâmetros:
        config: Configuração de sistema.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(config.use_tf32)
    torch.backends.cudnn.allow_tf32 = bool(config.use_tf32)
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(config.cudnn_deterministic)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass


def get_device(config: SystemConfig, logger: Optional[Any] = None) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Seleciona e diagnostica o dispositivo de computação (CPU ou GPU).

    Parâmetros:
        config: Configuração de sistema.
        logger: Logger para registrar o diagnóstico.

    Retorno:
        Uma tupla contendo o dispositivo PyTorch e um dicionário com metadados.
    """
    device_info = {
        "type": "cpu",
        "name": "CPU",
        "memory_gb": 0.0,
        "compute_capability": None,
        "multi_gpu": False,
        "gpu_count": 0,
    }

    cuda_available = False
    cuda_device_count = 0
    cuda_device_name0 = None
    cuda_error = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device_count = int(torch.cuda.device_count())
            if cuda_device_count > 0:
                cuda_device_name0 = torch.cuda.get_device_name(0)
    except Exception as exc:
        cuda_error = str(exc)

    if logger:
        logger.info(
            "CUDA diagnóstico",
            cuda_available=cuda_available,
            device_count=cuda_device_count,
            device_name_0=cuda_device_name0 or "N/A",
            use_gpu=config.use_gpu,
            gpu_id=config.gpu_id,
        )

    # Seleção centralizada do dispositivo de execução.
    device = torch.device("cuda" if cuda_available else "cpu")

    if not config.use_gpu:
        device = torch.device("cpu")
        return device, device_info

    if device.type != "cuda":
        if logger:
            logger.warning(
                "CUDA indisponível; usando CPU",
                error=cuda_error or "N/A",
            )
        device = torch.device("cpu")
        return device, device_info

    if cuda_device_count <= 0:
        if logger:
            logger.warning("CUDA detectado mas sem GPUs visíveis; usando CPU")
        device = torch.device("cpu")
        return device, device_info

    try:
        gpu_count = cuda_device_count

        if config.gpu_id is not None:
            if config.gpu_id >= gpu_count:
                raise GPUMemoryError(
                    f"GPU {config.gpu_id} não existe. GPUs disponíveis: 0-{gpu_count-1}"
                )
            selected_gpu = int(config.gpu_id)
        else:
            selected_gpu = 0
            max_free_memory = 0
            for i in range(gpu_count):
                torch.cuda.set_device(i)
                free_memory = (
                    torch.cuda.get_device_properties(i).total_memory
                    - torch.cuda.memory_allocated(i)
                )
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    selected_gpu = i

        _configure_cuda_optimizations(config)
        device = torch.device(f"cuda:{selected_gpu}")

        # Verificação rápida para garantir alocação e sincronização na CUDA.
        torch.empty(1, device=device)
        torch.cuda.synchronize(device)

        props = torch.cuda.get_device_properties(selected_gpu)
        device_info.update(
            {
                "type": "cuda",
                "name": props.name,
                "memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_gpu": gpu_count > 1,
                "gpu_count": gpu_count,
                "selected_gpu": selected_gpu,
            }
        )

        if logger:
            logger.info(
                "GPU selecionada",
                gpu_id=selected_gpu,
                name=props.name,
                memory_gb=f"{props.total_memory / 1e9:.1f}",
                compute_capability=f"{props.major}.{props.minor}",
                total_gpus=gpu_count,
            )

        return device, device_info

    except GPUMemoryError:
        raise
    except Exception as exc:
        if logger:
            logger.warning(
                "CUDA detectado mas inutilizável; revertendo para CPU",
                error=str(exc),
            )
        device = torch.device("cpu")
        return device, device_info


def initialize_system(
    system_config: SystemConfig,
) -> Tuple[SystemConfig, torch.device, GPUMemoryManager, Any]:
    """
    Inicializa todos os subcomponentes de sistema (logging, random, device, memory).

    Parâmetros:
        system_config: Configuração base do sistema.

    Retorno:
        Uma tupla (system_config, device, memory_manager, logger).
    """
    logger = setup_logging(level=logging.INFO, log_file=Path(system_config.log_file) if system_config.log_file else None)
    setup_torch_optimizations(system_config, logger)
    device, device_info = get_device(system_config, logger)
    memory_manager = GPUMemoryManager(device, system_config.max_memory_usage)

    if logger:
        logger.info(
            "Sistema inicializado",
            device_type=device_info["type"],
            device_name=device_info["name"],
            memory_gb=device_info.get("memory_gb", 0),
            mixed_precision=system_config.mixed_precision,
            compile_model=system_config.compile_model,
            seed=system_config.seed,
            deterministic_run=bool(getattr(system_config, "deterministic_run", False)),
            deterministic_warn_only=bool(
                getattr(system_config, "deterministic_warn_only", False)
            ),
        )

    return system_config, device, memory_manager, logger


__all__ = [
    "SystemConfig",
    "ExperimentConfig",
    "PIGANError",
    "GPUMemoryError",
    "ModelConfigurationError",
    "EnhancedLogger",
    "GPUMemoryManager",
    "setup_logging",
    "setup_torch_optimizations",
    "get_device",
    "initialize_system",
]




