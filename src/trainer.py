# -*- coding: utf-8 -*-
"""Loop de treinamento para o PI-GAN de campo fisicamente consistente."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models import FieldDualDiscriminator, LaplacianLayer, UNetGenerator2D


@dataclass
class FieldTrainerConfig:
    """
    Configurações exaustivas para o treinamento do PI-GAN de campo.

    Atributos:
        epochs: Número total de épocas.
        steps_per_epoch: Passos de otimização por época.
        batch_size: Tamanho do lote.
        n_critic: Número de atualizações do discriminador por passo do gerador.
        gen_lr: Taxa de aprendizado do gerador.
        disc_lr: Taxa de aprendizado base dos discriminadores.
        betas: Betas para o otimizador Adam.
        weight_decay: Decaimento de peso (L2).
        lambda_adv1: Peso ponderado da perda adversarial D1 (física).
        lambda_adv2: Peso ponderado da perda adversarial D2 (dados).
        use_reference_discriminator: Se habilita o discriminador D2 guiado por FDM.
        lambda_pde: Peso base da perda de resíduo da PDE.
        lambda_bc: Peso da perda de condições de contorno (se não usar hard constraint).
        lambda_gp: Peso do Gradient Penalty (WGAN-GP).
        use_wgan_gp: Se ativa o Gradient Penalty.
        d1_real_noise_std: Desvio padrão do ruído nos alvos reais de D1.
        d2_pair_noise_std: Desvio padrão do ruído nos alvos de D2.
        critic_drift: Coeficiente de drift para estabilidade WGAN.
        max_critic_gap: Gap máximo permitido entre real/fake antes da penalidade.
        critic_gap_penalty: Intensidade da penalidade de gap excessivo.
        residual_tanh_scale: Escala para normalização tanh do resíduo (opcional).
        use_tanh_on_residual: Se aplica tanh no resíduo antes de D1.
        dynamic_adv_balance: Ativa o equilíbrio dinâmico entre adversarial e física.
        target_adv_over_pde: Alvo da razão adversarial/pde.
        adv_scale_ema_beta: Beta do EMA para escala adversarial.
        adv_scale_min: Escala mínima para o termo adversarial.
        adv_scale_max: Escala máxima para o termo adversarial.
        pde_norm_ema_beta: Beta do EMA para normalização do resíduo PDE.
        grad_clip: Valor de corte para clipping de gradiente.
        generator_mode: Modo de operação do gerador (ex: 'adversarial').
        save_frequency: Frequência (épocas) para salvar checkpoints.
        checkpoint_dir: Diretório para salvar modelos.
        disc_update_every: Frequência de atualização dos discriminadores.
        disc_lr_d1: Taxa de aprendizado específica para D1 (opcional).
        disc_lr_d2: Taxa de aprendizado específica para D2 (opcional).
        lambda_gp_d1: Peso GP específico para D1 (opcional).
        lambda_gp_d2: Peso GP específico para D2 (opcional).
        d1_real_residual_mode: Fonte do resíduo real para D1 ('reference' ou 'zero').
        lambda_pde_raw: Peso adicional para a perda bruta da PDE.
        residual_mean_abs_target: Alvo de erro médio absoluto para o resíduo.
        adv_residual_gate_target: Alvo do gate de resíduo para ativar adversarial.
        adv_residual_gate_min: Valor mínimo do gate.
        adv_residual_gate_hysteresis: Ativa gate Schmitt (liga/desliga com memória).
        adv_residual_gate_off_threshold: Limiar para desligar gate adversarial.
        adv_residual_gate_power: Expoente do gate suave (quando sem histerese).
        adv_warmup_epochs: Épocas de aquecimento para o termo adversarial.
        critic_pause_on_overgap: Pausa discriminadores se o gap for extremo.
        critic_pause_gap_factor: Multiplicador de max_critic_gap para entrar em pausa.
        critic_resume_gap_factor: Multiplicador para sair da pausa (histerese).
        adv_stagnation_boost: Aumenta adversarial quando o resíduo estagna.
        adv_stagnation_patience: Passos sem melhoria antes do boost.
        adv_stagnation_rel_tol: Tolerância relativa para considerar melhoria.
        adv_stagnation_boost_factor: Fator multiplicativo do boost adversarial.
        adv_stagnation_min_gate: Gate mínimo para permitir boost de estagnação.
        adv_stagnation_cooldown: Espera entre boosts consecutivos.
        pde_corner_sampling_ratio: Fração alvo da massa de amostragem PDE nos cantos.
        pde_corner_band_points: Largura (em pontos internos) da região de canto.
        adaptive_lambda_pde: Ativa ajuste adaptativo de lambda_pde.
        residual_tolerance_target: Alvo de tolerância para o resíduo.
        residual_scale_reference: Escala de referência para cálculo de lambda_pde.
        lambda_pde_growth_exponent: Expoente de crescimento de lambda_pde.
        lambda_pde_min: Valor mínimo adaptativo para lambda_pde.
        lambda_pde_max: Valor máximo adaptativo para lambda_pde.
        lambda_pde_ema_beta: Beta do EMA para lambda_pde adaptativo.
        gradnorm_balance: Ativa balanceamento via norma de gradientes.
        gradnorm_target_adv_to_pde: Razão alvo entre normas de grad. adv e pde.
        gradnorm_ema_beta: Beta do EMA para balanceamento de gradNorm.
        gradnorm_scale_min: Escala mínima via gradNorm.
        gradnorm_scale_max: Escala máxima via gradNorm.
        divergence_window: Janela para detecção de divergência.
        divergence_ratio_threshold: Limiar de razão de perda para considerar divergência.
        divergence_patience: Paciência para ativar redução de LR por divergência.
        lr_drop_factor: Fator de redução de taxa de aprendizado.
        max_lr_drops: Máximo de reduções permitidas por divergência.
        plateau_scheduler_enabled: Ativa redução de LR por platô de métrica.
        plateau_metric_key: Chave da métrica monitorada para platô.
        plateau_mode: Modo do monitor ('min' ou 'max').
        plateau_patience: Paciência para detecção de platô.
        plateau_factor: Fator de redução de LR em platô.
        plateau_min_delta: Delta mínimo para considerar melhoria.
        plateau_cooldown: Épocas de espera após redução por platô.
        plateau_max_drops: Máximo de reduções permitidas por platô.
        plateau_reduce_discriminators: Se reduz LR dos discriminadores também.
        activation_abs_limit: Limite absoluto para ativações (segurança).
        residual_hist_bins: Número de bins para histogramas de resíduo.
        early_stop_on_nonfinite: Para treino se NaN/Inf detectado.
    """
    epochs: int
    steps_per_epoch: int
    batch_size: int
    n_critic: int
    gen_lr: float
    disc_lr: float
    betas: tuple[float, float]
    weight_decay: float
    lambda_adv1: float
    lambda_adv2: float
    lambda_pde: float
    lambda_bc: float
    lambda_gp: float
    use_wgan_gp: bool
    d1_real_noise_std: float
    d2_pair_noise_std: float
    critic_drift: float
    max_critic_gap: float
    critic_gap_penalty: float
    residual_tanh_scale: float
    use_tanh_on_residual: bool
    dynamic_adv_balance: bool
    target_adv_over_pde: float
    adv_scale_ema_beta: float
    adv_scale_min: float
    adv_scale_max: float
    pde_norm_ema_beta: float
    grad_clip: float
    generator_mode: str
    use_reference_discriminator: bool = True
    save_frequency: int = 0
    checkpoint_dir: Optional[str] = None
    disc_update_every: int = 1
    disc_lr_d1: Optional[float] = None
    disc_lr_d2: Optional[float] = None
    lambda_gp_d1: Optional[float] = None
    lambda_gp_d2: Optional[float] = None
    d1_real_residual_mode: str = "reference"
    lambda_pde_raw: float = 0.0
    residual_mean_abs_target: float = 0.0
    adv_residual_gate_target: float = 0.0
    adv_residual_gate_min: float = 0.1
    adv_residual_gate_hysteresis: bool = False
    adv_residual_gate_off_threshold: float = 0.0
    adv_residual_gate_power: float = 1.0
    adv_warmup_epochs: int = 0
    critic_pause_on_overgap: bool = False
    critic_pause_gap_factor: float = 1.25
    critic_resume_gap_factor: float = 0.75
    adv_stagnation_boost: bool = False
    adv_stagnation_patience: int = 40
    adv_stagnation_rel_tol: float = 1e-3
    adv_stagnation_boost_factor: float = 1.15
    adv_stagnation_min_gate: float = 0.5
    adv_stagnation_cooldown: int = 5
    adv_progressive_enable: bool = False
    adv_progressive_start_epoch: int = 0
    adv_progressive_ramp_epochs: int = 1
    adv_progressive_max_multiplier: float = 1.0
    adv_progressive_power: float = 1.0
    pde_corner_sampling_ratio: float = 0.0
    pde_corner_band_points: int = 2
    precision_refine_enable: bool = False
    precision_refine_start_epoch: int = 0
    precision_refine_n_critic: int = 1
    precision_refine_n_critic_ramp_epochs: int = 1
    precision_refine_lambda_pde_max_scale: float = 1.0
    adaptive_lambda_pde: bool = True
    residual_tolerance_target: float = 1e-3
    residual_scale_reference: float = 1.0
    lambda_pde_growth_exponent: float = 0.5
    lambda_pde_min: float = 1.0
    lambda_pde_max: float = 500.0
    lambda_pde_ema_beta: float = 0.9
    gradnorm_balance: bool = True
    gradnorm_target_adv_to_pde: float = 0.25
    gradnorm_ema_beta: float = 0.9
    gradnorm_scale_min: float = 0.05
    gradnorm_scale_max: float = 1.0
    divergence_window: int = 20
    divergence_ratio_threshold: float = 1.3
    divergence_patience: int = 2
    lr_drop_factor: float = 0.1
    max_lr_drops: int = 3
    plateau_scheduler_enabled: bool = False
    plateau_metric_key: str = "g_residual_mean_abs"
    plateau_mode: str = "min"
    plateau_patience: int = 25
    plateau_factor: float = 0.5
    plateau_min_delta: float = 1e-4
    plateau_cooldown: int = 5
    plateau_max_drops: int = 4
    plateau_reduce_discriminators: bool = False
    activation_abs_limit: float = 1e6
    residual_hist_bins: int = 12
    early_stop_on_nonfinite: bool = True


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Ativa ou desativa o cálculo de gradientes para todos os parâmetros de um módulo.

    Parâmetros:
        module: O módulo PyTorch a ser modificado.
        requires_grad: Se verdadeiro, ativa os gradientes.
    """
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def _gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula a penalidade de gradiente (Gradient Penalty) para WGAN-GP.

    Parâmetros:
        discriminator: O modelo discriminador.
        real_samples: Lote de amostras reais.
        fake_samples: Lote de amostras geradas.

    Retorno:
        Valor escalar da penalidade de gradiente.
    """
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


def _global_grad_norm(parameters: List[torch.Tensor]) -> torch.Tensor:
    """
    Calcula a norma L2 total do gradiente de uma lista de parâmetros.

    Parâmetros:
        parameters: Lista de tensores PyTorch que possuem gradientes.

    Retorno:
        Valor escalar da norma total.
    """
    total = None
    for p in parameters:
        if p.grad is None:
            continue
        contrib = p.grad.detach().pow(2).sum()
        total = contrib if total is None else total + contrib
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


class FieldPIGANTrainer:
    """
    Treinador para PI-GAN de campo 2D com arquitetura de discriminadores duais.

    Gerencia D1 (especialista em resíduo PDE), D2 (especialista em dados)
    e um Gerador com imposição de condições de contorno.
    """

    def __init__(
        self,
        *,
        generator: UNetGenerator2D,
        discriminator: FieldDualDiscriminator,
        laplacian: LaplacianLayer,
        base_field: torch.Tensor,      # [1,1,H,W]
        phi_mask: torch.Tensor,        # [1,1,H,W]
        coord_field: Optional[torch.Tensor],  # [1,2,H,W] or None
        reference_field: torch.Tensor,  # [1,1,H,W]
        interior_mask: torch.Tensor,   # [1,1,H,W]
        boundary_mask: torch.Tensor,   # [1,1,H,W]
        config: FieldTrainerConfig,
        device: torch.device,
        logger: Optional[Any] = None,
    ):
        """
        Inicializa o treinador FieldPIGANTrainer.

        Parâmetros:
            generator: O gerador UNet2D.
            discriminator: O discriminador dual (física + dados).
            laplacian: Camada de cálculo do Laplaciano.
            base_field: Extensão suave das condições de contorno.
            phi_mask: Máscara de imposição de contorno (hard constraint).
            coord_field: Coordenadas físicas normalizadas (x,y), quando usadas.
            reference_field: Solução numérica de referência (FDM).
            interior_mask: Máscara binária do interior do domínio.
            boundary_mask: Máscara binária da fronteira do domínio.
            config: Objeto de configuração do treinamento.
            device: Dispositivo de execução (cuda/cpu).
            logger: Instância de log opcional.
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.laplacian = laplacian.to(device)

        self.base_field = base_field.to(device)
        self.phi_mask = phi_mask.to(device)
        self.coord_field = None if coord_field is None else coord_field.to(device)
        self.reference_field = reference_field.to(device)
        self.interior_mask = interior_mask.to(device)
        self.boundary_mask = boundary_mask.to(device)

        self.cfg = config
        self.device = device
        self.logger = logger

        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=float(config.gen_lr),
            betas=tuple(config.betas),
            weight_decay=float(config.weight_decay),
        )
        disc_lr_d1 = float(config.disc_lr if config.disc_lr_d1 is None else config.disc_lr_d1)
        disc_lr_d2 = float(config.disc_lr if config.disc_lr_d2 is None else config.disc_lr_d2)
        self.opt_d1 = optim.Adam(
            self.discriminator.physics_discriminator.parameters(),
            lr=disc_lr_d1,
            betas=tuple(config.betas),
            weight_decay=float(config.weight_decay),
        )
        self.opt_d2 = optim.Adam(
            self.discriminator.data_discriminator.parameters(),
            lr=disc_lr_d2,
            betas=tuple(config.betas),
            weight_decay=float(config.weight_decay),
        )

        self._interior_count = self.interior_mask.sum().clamp_min(1.0)
        (
            self._pde_train_weight_map,
            self._pde_train_weight_sum,
            self._pde_corner_weight_factor,
            self._pde_corner_mass_effective,
        ) = self._build_pde_train_weight_map()
        if self.logger:
            self.logger.info(
                "Foco de cantos PDE configurado",
                target_ratio=float(getattr(self.cfg, "pde_corner_sampling_ratio", 0.0)),
                effective_ratio=float(self._pde_corner_mass_effective),
                corner_weight_factor=float(self._pde_corner_weight_factor),
                corner_band_points=int(getattr(self.cfg, "pde_corner_band_points", 2)),
            )
        self.save_frequency = max(0, int(getattr(self.cfg, "save_frequency", 0)))
        checkpoint_dir = getattr(self.cfg, "checkpoint_dir", None)
        self.checkpoint_dir: Optional[Path] = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._adv_scale_ema = 1.0
        self._adv_grad_scale_ema = 1.0
        self._lambda_pde_dyn = max(float(config.lambda_pde), 1e-8)
        self._pde_norm_ema = 1.0
        self._last_pde_scale = 1.0
        self._critics_paused_state = False
        self._adv_gate_enabled = True
        self._best_residual_for_adv = float("inf")
        self._adv_stagnation_steps = 0
        self._adv_stagnation_cooldown_counter = 0
        self._last_adv_boost_multiplier = 1.0
        self._last_adv_progressive_multiplier = 1.0
        self._last_n_critic_effective = max(1, int(config.n_critic))
        self._last_lambda_pde_max_effective = float(getattr(config, "lambda_pde_max", config.lambda_pde))
        self._initialize_pde_normalization()
        self.start_epoch = 1
        self.last_loaded_epoch = 0
        self.last_loaded_metrics: Dict[str, float] = {}
        self._current_epoch = 1
        self._global_step = 0
        self._last_d1_gap = 0.0
        self._last_d2_gap = 0.0
        self._lr_drop_count = 0
        self._divergence_streak = 0
        self._g_total_window: deque[float] = deque(maxlen=max(8, int(self.cfg.divergence_window)))
        self._residual_window: deque[float] = deque(maxlen=max(8, int(self.cfg.divergence_window)))
        self._plateau_best: Optional[float] = None
        self._plateau_bad_epochs = 0
        self._plateau_cooldown_counter = 0
        self._plateau_drop_count = 0
        self._stop_requested = False
        self._stop_reason = ""
        with torch.no_grad():
            self.reference_residual = self.laplacian(self.reference_field).detach()

    def _initialize_pde_normalization(self) -> None:
        """Estimate initial PDE residual scale to keep physics loss numerically well-conditioned."""
        try:
            was_training = self.generator.training
            self.generator.train()
            probe_batch = max(1, min(int(self.cfg.batch_size), 32))
            with torch.no_grad():
                base = self._expand(self.base_field, probe_batch)
                phi = self._expand(self.phi_mask, probe_batch)
                coords = self._expand_coords(probe_batch)
                z = self._sample_latent(probe_batch)
                pred = self.generator(base, phi, z=z, coord_field=coords)
                residual = self.laplacian(pred)
                rms = self._residual_l2(residual).detach().item()
            if not was_training:
                self.generator.eval()
            if np.isfinite(rms) and rms > 1e-6:
                self._pde_norm_ema = float(rms)
            else:
                self._pde_norm_ema = 1.0
            self._last_pde_scale = float(self._pde_norm_ema)
        except Exception:
            self._pde_norm_ema = 1.0
            self._last_pde_scale = 1.0

    @staticmethod
    def _serialize_numpy_state(state: tuple[Any, ...]) -> Dict[str, Any]:
        """Converte estado RNG do NumPy para estrutura segura em weights_only."""
        bit_gen, keys, pos, has_gauss, cached_gaussian = state
        return {
            "bit_generator": str(bit_gen),
            "keys": np.asarray(keys, dtype=np.uint32).tolist(),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }

    @staticmethod
    def _deserialize_numpy_state(state: Any) -> Optional[tuple[Any, ...]]:
        """Aceita estado legado em tupla ou estado serializado em dicionário."""
        if isinstance(state, tuple):
            return state
        if not isinstance(state, dict):
            return None
        try:
            keys = np.asarray(state.get("keys", []), dtype=np.uint32)
            return (
                str(state.get("bit_generator", "MT19937")),
                keys,
                int(state.get("pos", 0)),
                int(state.get("has_gauss", 0)),
                float(state.get("cached_gaussian", 0.0)),
            )
        except Exception:
            return None

    @staticmethod
    def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": FieldPIGANTrainer._serialize_numpy_state(np.random.get_state()),
            "python": random.getstate(),
        }
        if device.type == "cuda" and torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
        cpu_state = state.get("torch_cpu")
        if cpu_state is not None:
            torch.set_rng_state(cpu_state)
        cuda_state = state.get("torch_cuda")
        if (
            cuda_state is not None
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(cuda_state)
        numpy_state = FieldPIGANTrainer._deserialize_numpy_state(state.get("numpy"))
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)

    @staticmethod
    def _shorten_items(items: List[str], max_items: int = 8) -> str:
        if not items:
            return "none"
        if len(items) <= max_items:
            return ", ".join(items)
        shown = ", ".join(items[:max_items])
        return f"{shown}, ... (+{len(items) - max_items} more)"

    @staticmethod
    def _analyze_state_dict_compatibility(
        model_state: Dict[str, torch.Tensor],
        checkpoint_state: Dict[str, Any],
    ) -> Tuple[List[str], List[str], List[str]]:
        model_keys = set(model_state.keys())
        ckpt_keys = set(checkpoint_state.keys())

        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)

        shape_mismatch: List[str] = []
        for key in sorted(model_keys & ckpt_keys):
            ckpt_tensor = checkpoint_state[key]
            if not hasattr(ckpt_tensor, "shape"):
                shape_mismatch.append(
                    f"{key}: tipo no checkpoint ({type(ckpt_tensor).__name__}) sem atributo shape"
                )
                continue
            model_shape = tuple(model_state[key].shape)
            ckpt_shape = tuple(ckpt_tensor.shape)
            if ckpt_shape != model_shape:
                shape_mismatch.append(f"{key}: ckpt{ckpt_shape} vs model{model_shape}")

        return missing, unexpected, shape_mismatch

    def _load_model_state(
        self,
        model: nn.Module,
        checkpoint_state: Dict[str, Any],
        *,
        strict: bool,
        model_name: str,
    ) -> bool:
        current_state = model.state_dict()
        missing, unexpected, shape_mismatch = self._analyze_state_dict_compatibility(
            current_state,
            checkpoint_state,
        )

        if strict and (missing or unexpected or shape_mismatch):
            raise ValueError(
                f"Incompatibilidade de checkpoint em {model_name}. "
                f"missing_keys=[{self._shorten_items(missing)}]; "
                f"unexpected_keys=[{self._shorten_items(unexpected)}]; "
                f"shape_mismatch=[{self._shorten_items(shape_mismatch)}]. "
                "Verifique se a arquitetura/config do checkpoint coincide com a atual."
            )

        if strict:
            model.load_state_dict(checkpoint_state, strict=True)
            return True

        compatible_state: Dict[str, Any] = {}
        for key, value in checkpoint_state.items():
            if key not in current_state:
                continue
            if not hasattr(value, "shape"):
                continue
            if tuple(value.shape) != tuple(current_state[key].shape):
                continue
            compatible_state[key] = value

        model.load_state_dict(compatible_state, strict=False)

        full_load = not (missing or unexpected or shape_mismatch)
        if (not full_load) and self.logger:
            self.logger.warning(
                "Checkpoint carregado parcialmente",
                model=model_name,
                missing_keys=len(missing),
                unexpected_keys=len(unexpected),
                shape_mismatch=len(shape_mismatch),
            )
        return full_load

    def _zero_scalar(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((), device=self.device, dtype=dtype)

    def _sample_latent(self, batch_size: int, deterministic: bool = False) -> Optional[torch.Tensor]:
        latent_dim = int(getattr(self.generator, "latent_dim", 0))
        if latent_dim <= 0:
            return None

        mode = str(self.cfg.generator_mode)
        if deterministic or mode == "deterministic_adversarial":
            return torch.zeros(batch_size, latent_dim, device=self.device)

        return torch.randn(batch_size, latent_dim, device=self.device)

    def _expand(self, field: torch.Tensor, batch_size: int) -> torch.Tensor:
        return field.expand(batch_size, -1, -1, -1)

    def _expand_coords(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.coord_field is None:
            return None
        return self.coord_field.expand(batch_size, -1, -1, -1)

    def _boundary_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_sq = (pred - target) ** 2
        num = (diff_sq * self.boundary_mask).sum()
        den = self.boundary_mask.sum() + 1e-12
        return num / den

    def _boundary_max_abs_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_abs = (pred - target).abs() * self.boundary_mask
        return diff_abs.max()

    def _interior_denominator(self, residual: torch.Tensor) -> torch.Tensor:
        batch = max(1, int(residual.shape[0]))
        return self._interior_count * float(batch) + 1e-12

    def _build_pde_train_weight_map(self) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Constrói mapa de pesos do termo PDE para dar maior foco aos cantos.

        A estratégia é equivalente a concentrar uma fração alvo dos pontos de
        amostragem nas regiões de canto, sem introduzir ruído estocástico no
        cálculo da loss (importante para estabilidade/determinismo).
        """
        interior = (self.interior_mask > 0.5).to(dtype=self.interior_mask.dtype)
        weights = interior.clone()

        target_ratio = float(np.clip(getattr(self.cfg, "pde_corner_sampling_ratio", 0.0), 0.0, 0.95))
        band = max(1, int(getattr(self.cfg, "pde_corner_band_points", 2)))
        h = int(interior.shape[-2])
        w = int(interior.shape[-1])
        ih = max(1, h - 2)
        iw = max(1, w - 2)
        bh = min(band, ih)
        bw = min(band, iw)

        corner_mask = torch.zeros_like(interior)
        corner_mask[:, :, 1 : 1 + bh, 1 : 1 + bw] = 1.0
        corner_mask[:, :, 1 : 1 + bh, w - 1 - bw : w - 1] = 1.0
        corner_mask[:, :, h - 1 - bh : h - 1, 1 : 1 + bw] = 1.0
        corner_mask[:, :, h - 1 - bh : h - 1, w - 1 - bw : w - 1] = 1.0
        corner_mask = corner_mask * interior

        n_total = float(interior.sum().item())
        n_corner = float(corner_mask.sum().item())
        natural_ratio = (n_corner / n_total) if n_total > 0.0 else 0.0
        corner_weight_factor = 1.0
        if target_ratio > natural_ratio + 1e-12 and n_corner > 0.0 and n_total > n_corner:
            numerator = target_ratio * (n_total - n_corner)
            denominator = max(n_corner * (1.0 - target_ratio), 1e-12)
            corner_weight_factor = max(1.0, float(numerator / denominator))
            weights = torch.where(
                corner_mask > 0.5,
                torch.full_like(weights, float(corner_weight_factor)),
                weights,
            )
            weights = weights * interior

        weight_sum = weights.sum().clamp_min(1e-12)
        corner_mass_effective = float(((weights * corner_mask).sum() / weight_sum).item())
        return (
            weights,
            weight_sum,
            float(corner_weight_factor),
            float(corner_mass_effective),
        )

    def _pde_training_mean_abs(self, residual: torch.Tensor) -> torch.Tensor:
        """Média absoluta do resíduo PDE com foco ponderado em cantos."""
        num = (residual.abs() * self._pde_train_weight_map).sum()
        den = self._pde_train_weight_sum * float(max(1, int(residual.shape[0]))) + 1e-12
        return num / den

    def _pde_mse(self, residual: torch.Tensor) -> torch.Tensor:
        residual_rms = self._residual_l2(residual).detach().item()
        beta = float(np.clip(self.cfg.pde_norm_ema_beta, 0.0, 0.9999))
        target = max(float(residual_rms), 1e-6)
        self._pde_norm_ema = beta * float(self._pde_norm_ema) + (1.0 - beta) * target
        self._last_pde_scale = max(float(self._pde_norm_ema), 1e-6)

        normalized_residual = residual / self._last_pde_scale
        res_sq = normalized_residual ** 2
        num = (res_sq * self.interior_mask).sum()
        den = self._interior_denominator(residual)
        return num / den

    def _residual_mean_abs(self, residual: torch.Tensor) -> torch.Tensor:
        num = (residual.abs() * self.interior_mask).sum()
        den = self._interior_denominator(residual)
        return num / den

    def _residual_max_abs(self, residual: torch.Tensor) -> torch.Tensor:
        masked = residual.abs() * self.interior_mask
        return masked.max()

    def _residual_histograms(self, residual: torch.Tensor) -> Dict[str, float]:
        bins = max(4, int(getattr(self.cfg, "residual_hist_bins", 12)))
        mask = self.interior_mask.expand_as(residual) > 0.5
        values = residual.abs()[mask]
        if values.numel() <= 1:
            metrics: Dict[str, float] = {}
            for i in range(bins):
                metrics[f"g_res_hist_lin_{i}"] = 0.0
                metrics[f"g_res_hist_log_{i}"] = 0.0
            metrics["g_res_hist_log_min"] = -12.0
            metrics["g_res_hist_log_max"] = 0.0
            return metrics

        # Histogramas são apenas diagnóstico; em CPU evitam não determinismo do CUDA histc.
        hist_values = values.detach().to(device="cpu", dtype=torch.float32)
        max_val = max(float(hist_values.max().item()), 1e-6)
        hist_lin = torch.histc(hist_values, bins=bins, min=0.0, max=max_val)
        hist_lin = hist_lin / (hist_lin.sum() + 1e-12)

        log_values = torch.log10(hist_values + 1e-12)
        log_min = -12.0
        log_max = float(torch.ceil(log_values.max()).item())
        if log_max <= log_min:
            log_max = log_min + 1.0
        hist_log = torch.histc(log_values, bins=bins, min=log_min, max=log_max)
        hist_log = hist_log / (hist_log.sum() + 1e-12)

        metrics = {f"g_res_hist_lin_{i}": float(hist_lin[i].item()) for i in range(bins)}
        metrics.update({f"g_res_hist_log_{i}": float(hist_log[i].item()) for i in range(bins)})
        metrics["g_res_hist_log_min"] = float(log_min)
        metrics["g_res_hist_log_max"] = float(log_max)
        return metrics

    @staticmethod
    def _is_finite_tensor(t: torch.Tensor) -> bool:
        return bool(torch.isfinite(t).all().item())

    @staticmethod
    def _is_finite_number(x: float) -> bool:
        return bool(np.isfinite(float(x)))

    @staticmethod
    def _module_has_non_finite_params(module: nn.Module) -> bool:
        for p in module.parameters():
            if not torch.isfinite(p.detach()).all():
                return True
        return False

    def _grad_norm_from_loss(self, loss: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        if not loss.requires_grad:
            return self._zero_scalar(dtype=loss.dtype)
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        total = self._zero_scalar(dtype=loss.dtype)
        for g in grads:
            if g is None:
                continue
            total = total + g.detach().pow(2).sum()
        return torch.sqrt(total + 1e-12)

    def _update_lambda_pde_dynamic(self, residual_mean_abs: float) -> float:
        base = max(float(self.cfg.lambda_pde) + float(getattr(self.cfg, "lambda_pde_raw", 0.0)), 1e-8)
        if not bool(getattr(self.cfg, "adaptive_lambda_pde", True)):
            self._lambda_pde_dyn = base
            return self._lambda_pde_dyn

        target = max(float(getattr(self.cfg, "residual_scale_reference", 1.0)), 1e-12)
        ratio = max(float(residual_mean_abs) / target, 1.0)
        gain = max(float(getattr(self.cfg, "lambda_pde_growth_exponent", 0.5)), 1e-6)
        desired = base * (1.0 + gain * np.log10(ratio))
        desired = float(
            np.clip(
                desired,
                float(getattr(self.cfg, "lambda_pde_min", base)),
                self._effective_lambda_pde_max(),
            )
        )
        beta = float(np.clip(getattr(self.cfg, "lambda_pde_ema_beta", 0.9), 0.0, 0.9999))
        self._lambda_pde_dyn = beta * float(self._lambda_pde_dyn) + (1.0 - beta) * desired
        return float(self._lambda_pde_dyn)

    def _gradnorm_adv_scale(
        self,
        *,
        pde_term: torch.Tensor,
        adv_term: torch.Tensor,
        params: List[torch.Tensor],
    ) -> tuple[float, float, float]:
        if not bool(getattr(self.cfg, "gradnorm_balance", True)):
            return 1.0, 0.0, 0.0

        pde_grad_norm = float(self._grad_norm_from_loss(pde_term, params).item())
        adv_grad_norm = float(self._grad_norm_from_loss(adv_term, params).item())
        target_ratio = max(float(getattr(self.cfg, "gradnorm_target_adv_to_pde", 0.25)), 1e-6)
        raw_scale = target_ratio * pde_grad_norm / max(adv_grad_norm, 1e-12)
        clipped = float(
            np.clip(
                raw_scale,
                float(getattr(self.cfg, "gradnorm_scale_min", 0.05)),
                float(getattr(self.cfg, "gradnorm_scale_max", 1.0)),
            )
        )
        beta = float(np.clip(getattr(self.cfg, "gradnorm_ema_beta", 0.9), 0.0, 0.9999))
        self._adv_grad_scale_ema = beta * float(self._adv_grad_scale_ema) + (1.0 - beta) * clipped
        return float(self._adv_grad_scale_ema), pde_grad_norm, adv_grad_norm

    def _check_loss_finiteness(
        self,
        *,
        g_loss: torch.Tensor,
        g_pde_raw: torch.Tensor,
        g_adv1: torch.Tensor,
        g_adv2: torch.Tensor,
    ) -> None:
        if not self._is_finite_tensor(g_loss):
            raise FloatingPointError("g_loss contem NaN/Inf.")
        if not self._is_finite_tensor(g_pde_raw):
            raise FloatingPointError("g_pde_raw contem NaN/Inf.")
        if not self._is_finite_tensor(g_adv1) or not self._is_finite_tensor(g_adv2):
            raise FloatingPointError("termos adversariais de G contem NaN/Inf.")

    def _reduce_learning_rates(self, factor: float, *, include_discriminators: bool = True) -> Dict[str, float]:
        factor = float(np.clip(factor, 1e-4, 0.99))
        opts = [self.opt_g]
        if bool(include_discriminators):
            opts.extend([self.opt_d1, self.opt_d2])
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = max(1e-8, float(group["lr"]) * factor)
        return {
            "lr_g": float(self.opt_g.param_groups[0]["lr"]),
            "lr_d1": float(self.opt_d1.param_groups[0]["lr"]),
            "lr_d2": float(self.opt_d2.param_groups[0]["lr"]),
        }

    def _maybe_reduce_lrs_on_drift(self, summary: Dict[str, float]) -> Optional[Dict[str, float]]:
        self._g_total_window.append(float(summary.get("g_total", 0.0)))
        self._residual_window.append(float(summary.get("g_residual_mean_abs", 0.0)))
        window = max(4, int(getattr(self.cfg, "divergence_window", 20)))
        if len(self._g_total_window) < 2 * window:
            return None

        g_arr = np.asarray(list(self._g_total_window), dtype=float)
        r_arr = np.asarray(list(self._residual_window), dtype=float)
        prev_g = float(np.mean(g_arr[-2 * window: -window]))
        last_g = float(np.mean(g_arr[-window:]))
        prev_r = float(np.mean(r_arr[-2 * window: -window]))
        last_r = float(np.mean(r_arr[-window:]))
        ratio_threshold = float(getattr(self.cfg, "divergence_ratio_threshold", 1.3))

        diverging = (
            self._is_finite_number(prev_g)
            and self._is_finite_number(last_g)
            and prev_g > 1e-12
            and last_g > ratio_threshold * prev_g
            and last_r >= 0.98 * max(prev_r, 1e-12)
        )
        if diverging:
            self._divergence_streak += 1
        else:
            self._divergence_streak = 0

        patience = max(1, int(getattr(self.cfg, "divergence_patience", 2)))
        max_drops = max(0, int(getattr(self.cfg, "max_lr_drops", 3)))
        if self._divergence_streak < patience or self._lr_drop_count >= max_drops:
            return None

        self._divergence_streak = 0
        self._lr_drop_count += 1
        return self._reduce_learning_rates(float(getattr(self.cfg, "lr_drop_factor", 0.1)))

    def _maybe_reduce_lrs_on_plateau(self, summary: Dict[str, float]) -> Optional[Dict[str, float]]:
        if not bool(getattr(self.cfg, "plateau_scheduler_enabled", False)):
            return None

        key = str(getattr(self.cfg, "plateau_metric_key", "g_residual_mean_abs")).strip()
        if not key:
            return None
        metric_val = summary.get(key)
        if metric_val is None or not self._is_finite_number(metric_val):
            return None

        mode = str(getattr(self.cfg, "plateau_mode", "min")).strip().lower()
        min_delta = max(float(getattr(self.cfg, "plateau_min_delta", 1e-4)), 0.0)
        current = float(metric_val)

        improved = False
        if self._plateau_best is None:
            improved = True
        elif mode == "max":
            improved = current > float(self._plateau_best) + min_delta
        else:
            improved = current < float(self._plateau_best) - min_delta

        if improved:
            self._plateau_best = current
            self._plateau_bad_epochs = 0
            return None

        if self._plateau_cooldown_counter > 0:
            self._plateau_cooldown_counter -= 1
            return None

        self._plateau_bad_epochs += 1
        patience = max(1, int(getattr(self.cfg, "plateau_patience", 25)))
        max_drops = max(0, int(getattr(self.cfg, "plateau_max_drops", 4)))
        if self._plateau_bad_epochs < patience or self._plateau_drop_count >= max_drops:
            return None

        self._plateau_bad_epochs = 0
        self._plateau_drop_count += 1
        self._plateau_cooldown_counter = max(0, int(getattr(self.cfg, "plateau_cooldown", 5)))
        factor = float(getattr(self.cfg, "plateau_factor", 0.5))
        include_discriminators = bool(getattr(self.cfg, "plateau_reduce_discriminators", False))
        return self._reduce_learning_rates(factor, include_discriminators=include_discriminators)

    def _clip(self, params: List[torch.Tensor]) -> None:
        if self.cfg.grad_clip <= 0.0:
            return
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.cfg.grad_clip))

    def _residual_l2(self, residual: torch.Tensor) -> torch.Tensor:
        den = self._interior_denominator(residual)
        return torch.sqrt(((residual ** 2) * self.interior_mask).sum() / den)

    def _format_residual_for_d1(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Normaliza e formata o campo de resíduos para entrada no discriminador D1.

        Aplica escala e, opcionalmente, a função tanh baseada na configuração.

        Parâmetros:
            residual: Tensores de resíduo PDE.

        Retorno:
            Resíduo formatado (normalizado/escalonado).
        """
        masked = residual * self.interior_mask
        scale_cfg = float(getattr(self.cfg, "residual_tanh_scale", 0.0))
        if scale_cfg > 0.0:
            scale = torch.tensor(scale_cfg, device=masked.device, dtype=masked.dtype)
        else:
            den = self._interior_denominator(masked)
            scale = torch.sqrt((masked.pow(2).sum() / den).detach())
        scaled = masked / (scale + 1e-12)

        if bool(getattr(self.cfg, "use_tanh_on_residual", False)):
            return torch.tanh(scaled)
        return scaled

    def _gp_weight(self, critic_name: str) -> float:
        if critic_name == "d1":
            specific = getattr(self.cfg, "lambda_gp_d1", None)
        elif critic_name == "d2":
            specific = getattr(self.cfg, "lambda_gp_d2", None)
        else:
            raise ValueError(f"Critico invalido para GP: {critic_name}")
        if specific is None:
            return float(self.cfg.lambda_gp)
        return float(specific)

    def _maybe_gp(
        self,
        forward_fn: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        gp_weight: float,
    ) -> torch.Tensor:
        if not bool(self.cfg.use_wgan_gp) or gp_weight <= 0.0:
            return self._zero_scalar(dtype=real.dtype)
        return _gradient_penalty(forward_fn, real, fake)

    def _adv_warmup_factor(self) -> float:
        warmup_epochs = max(0, int(getattr(self.cfg, "adv_warmup_epochs", 0)))
        if warmup_epochs <= 0:
            return 1.0
        progress = float(self._current_epoch) / float(warmup_epochs)
        return float(np.clip(progress, 0.0, 1.0))

    def _adv_residual_gate(self, residual_l2: float) -> float:
        target = float(getattr(self.cfg, "adv_residual_gate_target", 0.0))
        if target <= 0.0:
            self._adv_gate_enabled = True
            return 1.0

        min_gate = float(np.clip(getattr(self.cfg, "adv_residual_gate_min", 0.1), 0.0, 1.0))
        residual_l2 = max(float(residual_l2), 1e-12)

        if bool(getattr(self.cfg, "adv_residual_gate_hysteresis", False)):
            on_threshold = max(target, 1e-12)
            off_threshold = float(getattr(self.cfg, "adv_residual_gate_off_threshold", 0.0))
            off_threshold = max(0.0, min(off_threshold, on_threshold))
            if self._adv_gate_enabled:
                if residual_l2 <= off_threshold:
                    self._adv_gate_enabled = False
            else:
                if residual_l2 >= on_threshold:
                    self._adv_gate_enabled = True
            return 1.0 if self._adv_gate_enabled else min_gate

        power = max(float(getattr(self.cfg, "adv_residual_gate_power", 1.0)), 1e-6)
        ratio = residual_l2 / max(target, 1e-12)
        gate = ratio ** power
        self._adv_gate_enabled = gate >= 0.5
        return float(np.clip(gate, min_gate, 1.0))

    def _critics_should_pause(self) -> bool:
        if not bool(getattr(self.cfg, "critic_pause_on_overgap", False)):
            self._critics_paused_state = False
            return False

        pause_factor = max(1.0, float(getattr(self.cfg, "critic_pause_gap_factor", 1.25)))
        resume_factor = float(getattr(self.cfg, "critic_resume_gap_factor", 0.75))
        resume_factor = float(np.clip(resume_factor, 0.0, pause_factor))

        pause_gap = float(self.cfg.max_critic_gap) * pause_factor
        resume_gap = float(self.cfg.max_critic_gap) * resume_factor
        current_gap = max(abs(float(self._last_d1_gap)), abs(float(self._last_d2_gap)))

        if self._critics_paused_state:
            if current_gap <= resume_gap:
                self._critics_paused_state = False
        else:
            if current_gap >= pause_gap:
                self._critics_paused_state = True
        return self._critics_paused_state

    def _apply_adv_stagnation_boost(
        self,
        *,
        adv_scale: float,
        residual_mean_abs: float,
        adv_gate: float,
    ) -> float:
        """Increase adversarial pressure when PDE residual plateaus with active gate."""
        self._last_adv_boost_multiplier = 1.0
        if not bool(getattr(self.cfg, "adv_stagnation_boost", False)):
            return float(adv_scale)

        residual = max(float(residual_mean_abs), 1e-16)
        rel_tol = max(float(getattr(self.cfg, "adv_stagnation_rel_tol", 1e-3)), 0.0)
        if not np.isfinite(self._best_residual_for_adv):
            self._best_residual_for_adv = residual
            self._adv_stagnation_steps = 0
        else:
            improvement_threshold = self._best_residual_for_adv * (1.0 - rel_tol)
            if residual < improvement_threshold:
                self._best_residual_for_adv = residual
                self._adv_stagnation_steps = 0
            else:
                self._adv_stagnation_steps += 1

        if self._adv_stagnation_cooldown_counter > 0:
            self._adv_stagnation_cooldown_counter -= 1
            return float(adv_scale)

        min_gate = float(np.clip(getattr(self.cfg, "adv_stagnation_min_gate", 0.5), 0.0, 1.0))
        patience = max(1, int(getattr(self.cfg, "adv_stagnation_patience", 40)))
        if adv_gate < min_gate or self._adv_stagnation_steps < patience:
            return float(adv_scale)

        boost_factor = max(float(getattr(self.cfg, "adv_stagnation_boost_factor", 1.15)), 1.0)
        boosted = min(float(self.cfg.adv_scale_max), float(adv_scale) * boost_factor)
        self._last_adv_boost_multiplier = boosted / max(float(adv_scale), 1e-12)
        self._adv_stagnation_steps = 0
        self._adv_stagnation_cooldown_counter = max(
            0, int(getattr(self.cfg, "adv_stagnation_cooldown", 5))
        )
        return float(boosted)

    def _adv_progressive_multiplier(self) -> float:
        """Progressive lambda_adv ramp scheduled from a stagnation epoch."""
        self._last_adv_progressive_multiplier = 1.0
        if not bool(getattr(self.cfg, "adv_progressive_enable", False)):
            return 1.0

        start_epoch = int(getattr(self.cfg, "adv_progressive_start_epoch", 0))
        if start_epoch <= 0 or int(self._current_epoch) < start_epoch:
            return 1.0

        max_multiplier = max(
            1.0,
            float(getattr(self.cfg, "adv_progressive_max_multiplier", 1.0)),
        )
        if max_multiplier <= 1.0:
            return 1.0

        ramp_epochs = max(1, int(getattr(self.cfg, "adv_progressive_ramp_epochs", 1)))
        power = max(float(getattr(self.cfg, "adv_progressive_power", 1.0)), 1e-6)
        progress = (float(self._current_epoch) - float(start_epoch) + 1.0) / float(ramp_epochs)
        progress = float(np.clip(progress, 0.0, 1.0))
        eased = progress ** power
        multiplier = 1.0 + (max_multiplier - 1.0) * eased
        self._last_adv_progressive_multiplier = float(multiplier)
        return float(multiplier)

    def _precision_refine_progress(self) -> float:
        if not bool(getattr(self.cfg, "precision_refine_enable", False)):
            return 0.0
        start_epoch = int(getattr(self.cfg, "precision_refine_start_epoch", 0))
        if start_epoch <= 0 or int(self._current_epoch) < start_epoch:
            return 0.0
        ramp_epochs = max(1, int(getattr(self.cfg, "precision_refine_n_critic_ramp_epochs", 1)))
        progress = (float(self._current_epoch) - float(start_epoch) + 1.0) / float(ramp_epochs)
        return float(np.clip(progress, 0.0, 1.0))

    def _effective_n_critic(self) -> int:
        base = max(1, int(self.cfg.n_critic))
        if not bool(getattr(self.cfg, "precision_refine_enable", False)):
            self._last_n_critic_effective = int(base)
            return int(base)
        target = max(base, int(getattr(self.cfg, "precision_refine_n_critic", base)))
        alpha = self._precision_refine_progress()
        eff = base + int(np.floor((target - base) * alpha + 1e-12))
        eff = int(np.clip(eff, base, target))
        self._last_n_critic_effective = int(eff)
        return int(eff)

    def _effective_lambda_pde_max(self) -> float:
        base_max = max(float(getattr(self.cfg, "lambda_pde_max", float(self.cfg.lambda_pde))), 1e-8)
        min_cap = max(float(getattr(self.cfg, "lambda_pde_min", 1e-8)), 1e-8)
        if not bool(getattr(self.cfg, "precision_refine_enable", False)):
            eff = max(min_cap, base_max)
            self._last_lambda_pde_max_effective = float(eff)
            return float(eff)

        scale = float(
            np.clip(
                getattr(self.cfg, "precision_refine_lambda_pde_max_scale", 1.0),
                1e-6,
                1.0,
            )
        )
        alpha = self._precision_refine_progress()
        eff_scale = 1.0 - alpha * (1.0 - scale)
        eff = max(min_cap, base_max * eff_scale)
        self._last_lambda_pde_max_effective = float(eff)
        return float(eff)

    def _zero_d1_metrics(self, *, gap: Optional[float] = None) -> Dict[str, float]:
        resolved_gap = float(self._last_d1_gap if gap is None else gap)
        return {
            "d1_total": 0.0,
            "d1_real": 0.0,
            "d1_fake": 0.0,
            "d1_gap": resolved_gap,
            "d1_gp": 0.0,
            "d1_gp_lambda": float(self._gp_weight("d1")),
            "d1_drift": 0.0,
            "d1_gap_penalty": 0.0,
            "d1_grad_norm": 0.0,
            "d1_residual_l2_fake": 0.0,
        }

    def _zero_d2_metrics(self, *, gap: Optional[float] = None) -> Dict[str, float]:
        resolved_gap = float(self._last_d2_gap if gap is None else gap)
        return {
            "d2_total": 0.0,
            "d2_real": 0.0,
            "d2_fake": 0.0,
            "d2_gap": resolved_gap,
            "d2_gp": 0.0,
            "d2_gp_lambda": float(self._gp_weight("d2")),
            "d2_drift": 0.0,
            "d2_gap_penalty": 0.0,
            "d2_grad_norm": 0.0,
        }

    def _update_d1(self, base: torch.Tensor, phi: torch.Tensor) -> Dict[str, float]:
        """
        Executa um passo de otimização para o discriminador de física (D1).

        D1 tenta distinguir resíduos reais (da referência) de resíduos falsos
        (das predições do gerador).

        Parâmetros:
            base: Extensão das condições de contorno.
            phi: Máscara do domínio.

        Retorno:
            Dicionário com perdas e métricas do passo D1.
        """
        if float(self.cfg.lambda_adv1) <= 0.0:
            return self._zero_d1_metrics()

        _set_requires_grad(self.discriminator.physics_discriminator, True)
        _set_requires_grad(self.discriminator.data_discriminator, False)

        self.opt_d1.zero_grad(set_to_none=True)
        z = self._sample_latent(base.size(0))
        coords = self._expand_coords(base.size(0))

        with torch.no_grad():
            pred_detached = self.generator(base, phi, z=z, coord_field=coords)
            residual_fake_raw = self.laplacian(pred_detached)

        # Fonte do resíduo real para D1:
        # - "reference": mapa de resíduo da referência numérica.
        # - "zero": resíduo idealizado nulo (compatibilidade com fluxo legado).
        mode = str(getattr(self.cfg, "d1_real_residual_mode", "reference")).strip().lower()
        if mode == "reference":
            residual_real_raw = self._expand(self.reference_residual, base.size(0))
        elif mode == "zero":
            residual_real_raw = torch.zeros_like(residual_fake_raw)
        else:
            raise ValueError(
                f"d1_real_residual_mode invalido: {mode}. Use 'reference' ou 'zero'."
            )
        if float(self.cfg.d1_real_noise_std) > 0.0:
            residual_real_raw = residual_real_raw + torch.randn_like(residual_real_raw) * float(
                self.cfg.d1_real_noise_std
            )

        residual_real = self._format_residual_for_d1(residual_real_raw)
        residual_fake = self._format_residual_for_d1(residual_fake_raw.detach())

        d1_real_score = self.discriminator.forward_physics(residual_real)
        d1_fake_score = self.discriminator.forward_physics(residual_fake)

        gp_weight = self._gp_weight("d1")
        gp = self._maybe_gp(
            self.discriminator.forward_physics,
            residual_real,
            residual_fake.detach(),
            gp_weight,
        )
        d1_drift = float(self.cfg.critic_drift) * (
            d1_real_score.pow(2).mean() + d1_fake_score.pow(2).mean()
        )
        d1_gap = d1_real_score.mean() - d1_fake_score.mean()
        d1_gap_excess = torch.relu(d1_gap.abs() - float(self.cfg.max_critic_gap))
        d1_gap_penalty = float(self.cfg.critic_gap_penalty) * d1_gap_excess.pow(2)

        # Objetivo do crítico WGAN implementado como minimização:
        # L_Df = E[D_f(fake)] - E[D_f(real)] + lambda_gp*GP_f + drift
        d1_loss = (
            (d1_fake_score.mean() - d1_real_score.mean())
            + gp_weight * gp
            + d1_drift
            + d1_gap_penalty
        )
        d1_loss.backward()

        params = list(self.discriminator.physics_discriminator.parameters())
        d1_grad = _global_grad_norm(params)
        self._clip(params)
        self.opt_d1.step()

        return {
            "d1_total": float(d1_loss.item()),
            "d1_real": float(d1_real_score.mean().item()),
            "d1_fake": float(d1_fake_score.mean().item()),
            "d1_gap": float(d1_gap.item()),
            "d1_gp": float(gp.item()),
            "d1_gp_lambda": float(gp_weight),
            "d1_drift": float(d1_drift.item()),
            "d1_gap_penalty": float(d1_gap_penalty.item()),
            "d1_grad_norm": float(d1_grad.item()),
            "d1_residual_l2_fake": float(self._residual_l2(residual_fake_raw).item()),
        }

    def _update_d2(
        self, base: torch.Tensor, phi: torch.Tensor, ref: torch.Tensor
    ) -> Dict[str, float]:
        """
        Executa um passo de otimização para o discriminador de dados (D2).

        D2 tenta distinguir pares reais [ref, ref] de pares falsos [pred, ref].

        Parâmetros:
            base: Extensão das condições de contorno.
            phi: Máscara do domínio.
            ref: Campo de referência FDM.

        Retorno:
            Dicionário com perdas e métricas do passo D2.
        """
        if (not bool(getattr(self.cfg, "use_reference_discriminator", True))) or (
            float(self.cfg.lambda_adv2) <= 0.0
        ):
            return self._zero_d2_metrics()

        _set_requires_grad(self.discriminator.physics_discriminator, False)
        _set_requires_grad(self.discriminator.data_discriminator, True)

        self.opt_d2.zero_grad(set_to_none=True)
        z = self._sample_latent(base.size(0))
        coords = self._expand_coords(base.size(0))
        with torch.no_grad():
            pred_detached = self.generator(base, phi, z=z, coord_field=coords)

        # Pares real/fake usam a mesma semântica de canais [candidato, referência].
        pair_real = torch.cat([ref, ref], dim=1)
        pair_fake = torch.cat([pred_detached.detach(), ref], dim=1)

        # Ruído simétrico (real/fake) evita vazamento de informação por padrão de ruído.
        if float(self.cfg.d2_pair_noise_std) > 0.0:
            pair_real = pair_real + torch.randn_like(pair_real) * float(self.cfg.d2_pair_noise_std)
            pair_fake = pair_fake + torch.randn_like(pair_fake) * float(self.cfg.d2_pair_noise_std)

        d2_real_score = self.discriminator.forward_data(pair_real)
        d2_fake_score = self.discriminator.forward_data(pair_fake)

        gp_weight = self._gp_weight("d2")
        gp = self._maybe_gp(
            self.discriminator.forward_data,
            pair_real,
            pair_fake.detach(),
            gp_weight,
        )
        d2_drift = float(self.cfg.critic_drift) * (
            d2_real_score.pow(2).mean() + d2_fake_score.pow(2).mean()
        )
        d2_gap = d2_real_score.mean() - d2_fake_score.mean()
        d2_gap_excess = torch.relu(d2_gap.abs() - float(self.cfg.max_critic_gap))
        d2_gap_penalty = float(self.cfg.critic_gap_penalty) * d2_gap_excess.pow(2)

        # Objetivo WGAN do discriminador de dados:
        # L_Dd = E[D_d(fake)] - E[D_d(real)] + lambda_gp*GP_d + drift
        d2_loss = (
            (d2_fake_score.mean() - d2_real_score.mean())
            + gp_weight * gp
            + d2_drift
            + d2_gap_penalty
        )
        d2_loss.backward()

        params = list(self.discriminator.data_discriminator.parameters())
        d2_grad = _global_grad_norm(params)
        self._clip(params)
        self.opt_d2.step()

        return {
            "d2_total": float(d2_loss.item()),
            "d2_real": float(d2_real_score.mean().item()),
            "d2_fake": float(d2_fake_score.mean().item()),
            "d2_gap": float(d2_gap.item()),
            "d2_gp": float(gp.item()),
            "d2_gp_lambda": float(gp_weight),
            "d2_drift": float(d2_drift.item()),
            "d2_gap_penalty": float(d2_gap_penalty.item()),
            "d2_grad_norm": float(d2_grad.item()),
        }

    def _update_generator(
        self, base: torch.Tensor, phi: torch.Tensor, ref: torch.Tensor
    ) -> Dict[str, float]:
        """
        Executa um passo de otimização para o gerador (G).

        Otimiza G para satisfazer a PDE e enganar ambos os discriminadores duais.

        Parâmetros:
            base: Extensão das condições de contorno.
            phi: Máscara do domínio.
            ref: Solução de referência.

        Retorno:
            Dicionário com perdas (adv, pde, bc) e métricas de G.
        """
        self.generator.train()
        self.discriminator.eval()
        _set_requires_grad(self.discriminator.physics_discriminator, False)
        _set_requires_grad(self.discriminator.data_discriminator, False)

        self.opt_g.zero_grad(set_to_none=True)

        adv2_enabled = bool(getattr(self.cfg, "use_reference_discriminator", True)) and (
            float(self.cfg.lambda_adv2) > 0.0
        )
        adv_disabled = float(self.cfg.lambda_adv1) == 0.0 and (not adv2_enabled)
        z = self._sample_latent(base.size(0), deterministic=adv_disabled)
        coords = self._expand_coords(base.size(0))

        pred = self.generator(base, phi, z=z, coord_field=coords)
        residual = self.laplacian(pred)
        residual_for_d1 = self._format_residual_for_d1(residual)

        g_adv1 = self._zero_scalar(dtype=pred.dtype)
        if float(self.cfg.lambda_adv1) > 0.0:
            g_adv1 = -self.discriminator.forward_physics(residual_for_d1).mean()
        g_adv2 = self._zero_scalar(dtype=pred.dtype)
        if bool(getattr(self.cfg, "use_reference_discriminator", True)) and (
            float(self.cfg.lambda_adv2) > 0.0
        ):
            g_adv2 = -self.discriminator.forward_data(torch.cat([pred, ref], dim=1)).mean()
        # Mantém métrica PDE normalizada para diagnóstico, mas otimiza o resíduo bruto.
        g_pde = self._pde_mse(residual)
        g_bc = self._boundary_mse(pred, ref)
        g_bc_max = self._boundary_max_abs_error(pred, ref)
        g_residual_l2 = self._residual_l2(residual)
        g_residual_mean_abs = self._residual_mean_abs(residual)
        g_residual_max_abs = self._residual_max_abs(residual)
        g_pred_max_abs = pred.detach().abs().max()

        # Objetivo físico bruto com equivalente de amostragem focada em cantos.
        g_pde_raw = self._pde_training_mean_abs(residual)
        residual_mean_abs_target = max(float(getattr(self.cfg, "residual_mean_abs_target", 0.0)), 0.0)
        g_pde_raw_penalty = g_pde_raw
        if residual_mean_abs_target > 0.0:
            g_pde_raw_excess = torch.relu(g_pde_raw - residual_mean_abs_target)
        else:
            g_pde_raw_excess = g_pde_raw

        # Objetivo do gerador (minimização) a partir da formulação min-max da PI-GAN:
        # L_G = lambda_adv_f*(-E[D_f(fake)])
        #     + lambda_adv_d*(-E[D_d(fake)])
        #     + lambda_pde*L_PDE_raw + lambda_bc*L_BC
        adv_scale = 1.0
        base_adv1 = float(self.cfg.lambda_adv1)
        base_adv2 = float(self.cfg.lambda_adv2)
        lambda_pde_dyn = self._update_lambda_pde_dynamic(float(g_pde_raw.item()))
        weighted_pde = abs(lambda_pde_dyn * float(g_pde_raw_penalty.item())) + 1e-12
        weighted_adv_base = abs(base_adv1 * float(g_adv1.item())) + abs(base_adv2 * float(g_adv2.item()))
        if bool(self.cfg.dynamic_adv_balance):
            target_ratio = max(float(self.cfg.target_adv_over_pde), 1e-6)
            raw_scale = target_ratio * weighted_pde / (weighted_adv_base + 1e-12)
            clipped_scale = float(np.clip(raw_scale, float(self.cfg.adv_scale_min), float(self.cfg.adv_scale_max)))
            ema_beta = float(np.clip(self.cfg.adv_scale_ema_beta, 0.0, 0.9999))
            self._adv_scale_ema = ema_beta * float(self._adv_scale_ema) + (1.0 - ema_beta) * clipped_scale
            adv_scale = float(self._adv_scale_ema)

        adv_warmup = self._adv_warmup_factor()
        adv_residual_gate = self._adv_residual_gate(float(g_residual_l2.item()))
        adv_gate = float(np.clip(adv_warmup * adv_residual_gate, 0.0, 1.0))
        adv_scale = self._apply_adv_stagnation_boost(
            adv_scale=adv_scale,
            residual_mean_abs=float(g_residual_mean_abs.item()),
            adv_gate=adv_gate,
        )
        adv_progressive_multiplier = self._adv_progressive_multiplier()

        eff_adv1 = base_adv1 * adv_scale * adv_gate * adv_progressive_multiplier
        eff_adv2 = base_adv2 * adv_scale * adv_gate * adv_progressive_multiplier
        params = [p for p in self.generator.parameters() if p.requires_grad]
        pde_term = lambda_pde_dyn * g_pde_raw_penalty
        adv_term_base = eff_adv1 * g_adv1 + eff_adv2 * g_adv2
        adv_grad_scale, pde_grad_proxy, adv_grad_proxy = self._gradnorm_adv_scale(
            pde_term=pde_term,
            adv_term=adv_term_base,
            params=params,
        )
        adv_term = adv_grad_scale * adv_term_base
        g_loss = pde_term + adv_term + float(self.cfg.lambda_bc) * g_bc
        self._check_loss_finiteness(
            g_loss=g_loss,
            g_pde_raw=g_pde_raw_penalty,
            g_adv1=g_adv1,
            g_adv2=g_adv2,
        )
        g_loss.backward()

        g_grad = _global_grad_norm(params)
        self._clip(params)
        self.opt_g.step()

        adv_mag = abs(float((adv_term).item()))
        pde_mag = abs(float(pde_term.item())) + 1e-12
        hist_metrics = self._residual_histograms(residual)

        metrics = {
            "g_total": float(g_loss.item()),
            "g_adv1": float(g_adv1.item()),
            "g_adv2": float(g_adv2.item()),
            "g_pde": float(g_pde.item()),
            "g_pde_raw": float(g_pde_raw.item()),
            "g_pde_raw_penalty": float(g_pde_raw_penalty.item()),
            "g_pde_raw_excess": float(g_pde_raw_excess.item()),
            "g_bc": float(g_bc.item()),
            "g_bc_max_abs": float(g_bc_max.item()),
            "g_adv_over_pde": float(adv_mag / pde_mag),
            "g_adv_scale": float(adv_scale),
            "g_adv_boost_multiplier": float(self._last_adv_boost_multiplier),
            "g_adv_progressive_multiplier": float(self._last_adv_progressive_multiplier),
            "g_lambda_adv1_eff": float(eff_adv1),
            "g_lambda_adv2_eff": float(eff_adv2),
            "g_adv_grad_scale": float(adv_grad_scale),
            "g_gradnorm_pde_proxy": float(pde_grad_proxy),
            "g_gradnorm_adv_proxy": float(adv_grad_proxy),
            "g_adv_gate": float(adv_gate),
            "g_adv_gate_enabled": 1.0 if bool(self._adv_gate_enabled) else 0.0,
            "g_adv_warmup": float(adv_warmup),
            "g_adv_residual_gate": float(adv_residual_gate),
            "g_lambda_pde_dyn": float(lambda_pde_dyn),
            "g_lambda_pde_cap_eff": float(self._last_lambda_pde_max_effective),
            "g_pde_scale": float(self._last_pde_scale),
            "g_residual_l2": float(g_residual_l2.item()),
            "g_residual_mean_abs": float(g_residual_mean_abs.item()),
            "g_residual_max_abs": float(g_residual_max_abs.item()),
            "g_pde_corner_mass_effective": float(self._pde_corner_mass_effective),
            "g_pde_corner_weight_factor": float(self._pde_corner_weight_factor),
            "g_pred_max_abs": float(g_pred_max_abs.item()),
            "g_grad_norm": float(g_grad.item()),
            "non_finite_detected": 0.0,
            "activation_overflow": 0.0,
        }
        metrics.update(hist_metrics)
        return metrics

    @staticmethod
    def _merge_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metric_list:
            return {}
        keys = metric_list[0].keys()
        merged: Dict[str, float] = {}
        for key in keys:
            merged[key] = float(np.mean([m[key] for m in metric_list]))
        return merged

    def save_checkpoint(
        self, filepath: Path, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """
        Salva o estado atual do treinamento em um arquivo de checkpoint persistente.

        Parâmetros:
            filepath: Caminho completo do arquivo .pt.
            epoch: Época atual.
            metrics: Métricas para salvar no checkpoint.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": int(epoch),
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_G": self.opt_g.state_dict(),
            "optimizer_D": {
                "physics_discriminator": self.opt_d1.state_dict(),
                "data_discriminator": self.opt_d2.state_dict(),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "config": asdict(self.cfg),
            "rng_state": self._capture_rng_state(self.device),
            "control_state": {
                "adv_scale_ema": float(self._adv_scale_ema),
                "adv_grad_scale_ema": float(self._adv_grad_scale_ema),
                "lambda_pde_dyn": float(self._lambda_pde_dyn),
                "pde_norm_ema": float(self._pde_norm_ema),
                "last_pde_scale": float(self._last_pde_scale),
                "last_d1_gap": float(self._last_d1_gap),
                "last_d2_gap": float(self._last_d2_gap),
                "critics_paused_state": bool(self._critics_paused_state),
                "adv_gate_enabled": bool(self._adv_gate_enabled),
                "best_residual_for_adv": float(self._best_residual_for_adv),
                "adv_stagnation_steps": int(self._adv_stagnation_steps),
                "adv_stagnation_cooldown_counter": int(self._adv_stagnation_cooldown_counter),
                "last_adv_progressive_multiplier": float(self._last_adv_progressive_multiplier),
                "last_n_critic_effective": int(self._last_n_critic_effective),
                "last_lambda_pde_max_effective": float(self._last_lambda_pde_max_effective),
            },
        }
        # Salvamento atômico evita checkpoint parcial em caso de interrupção.
        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(filepath)

    def load_checkpoint(
        self,
        filepath: Path,
        *,
        strict: bool = True,
        load_optimizer_state: bool = True,
        restore_rng_state: bool = True,
    ) -> Tuple[int, Dict[str, float]]:
        """
        Carrega o estado do treinamento a partir de um checkpoint.

        Parâmetros:
            filepath: Caminho do arquivo .pt.
            strict: Se verdadeiro, exige compatibilidade total de nomes/formas.
            load_optimizer_state: Se carrega os estados dos otimizadores.
            restore_rng_state: Se restaura o estado dos geradores de números aleatórios.

        Retorno:
            Tupla contendo (época_carregada, métricas_carregadas).
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint invalido: estrutura esperada dict.")

        required_keys = {"epoch", "generator_state_dict", "discriminator_state_dict"}
        missing = [k for k in required_keys if k not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint invalido: chaves ausentes {missing}.")

        generator_full = self._load_model_state(
            self.generator,
            checkpoint["generator_state_dict"],
            strict=bool(strict),
            model_name="generator",
        )
        discriminator_full = self._load_model_state(
            self.discriminator,
            checkpoint["discriminator_state_dict"],
            strict=bool(strict),
            model_name="discriminator",
        )
        full_model_load = generator_full and discriminator_full

        if load_optimizer_state and full_model_load:
            opt_g = checkpoint.get("optimizer_G")
            if isinstance(opt_g, dict):
                self.opt_g.load_state_dict(opt_g)
            opt_d = checkpoint.get("optimizer_D", {})
            if isinstance(opt_d, dict):
                d1 = opt_d.get("physics_discriminator")
                d2 = opt_d.get("data_discriminator")
                if isinstance(d1, dict):
                    self.opt_d1.load_state_dict(d1)
                if isinstance(d2, dict):
                    self.opt_d2.load_state_dict(d2)
            else:
                # Compatibilidade com checkpoints legados de otimizadores separados.
                d1 = checkpoint.get("optimizer_D1")
                d2 = checkpoint.get("optimizer_D2")
                if isinstance(d1, dict):
                    self.opt_d1.load_state_dict(d1)
                if isinstance(d2, dict):
                    self.opt_d2.load_state_dict(d2)
        elif load_optimizer_state and (not full_model_load) and self.logger:
            self.logger.warning(
                "Estado de otimizadores nao carregado por incompatibilidade parcial de modelo"
            )

        if restore_rng_state:
            rng_state = checkpoint.get("rng_state")
            if isinstance(rng_state, dict):
                self._restore_rng_state(rng_state, self.device)
        control_state = checkpoint.get("control_state")
        if isinstance(control_state, dict):
            self._adv_scale_ema = float(control_state.get("adv_scale_ema", self._adv_scale_ema))
            self._adv_grad_scale_ema = float(
                control_state.get("adv_grad_scale_ema", self._adv_grad_scale_ema)
            )
            self._lambda_pde_dyn = float(control_state.get("lambda_pde_dyn", self._lambda_pde_dyn))
            self._pde_norm_ema = float(control_state.get("pde_norm_ema", self._pde_norm_ema))
            self._last_pde_scale = float(control_state.get("last_pde_scale", self._last_pde_scale))
            self._last_d1_gap = float(control_state.get("last_d1_gap", self._last_d1_gap))
            self._last_d2_gap = float(control_state.get("last_d2_gap", self._last_d2_gap))
            self._critics_paused_state = bool(
                control_state.get("critics_paused_state", self._critics_paused_state)
            )
            self._adv_gate_enabled = bool(
                control_state.get("adv_gate_enabled", self._adv_gate_enabled)
            )
            self._best_residual_for_adv = float(
                control_state.get("best_residual_for_adv", self._best_residual_for_adv)
            )
            self._adv_stagnation_steps = int(
                control_state.get("adv_stagnation_steps", self._adv_stagnation_steps)
            )
            self._adv_stagnation_cooldown_counter = int(
                control_state.get(
                    "adv_stagnation_cooldown_counter",
                    self._adv_stagnation_cooldown_counter,
                )
            )
            self._last_adv_progressive_multiplier = float(
                control_state.get(
                    "last_adv_progressive_multiplier",
                    self._last_adv_progressive_multiplier,
                )
            )
            self._last_n_critic_effective = int(
                control_state.get("last_n_critic_effective", self._last_n_critic_effective)
            )
            self._last_lambda_pde_max_effective = float(
                control_state.get(
                    "last_lambda_pde_max_effective",
                    self._last_lambda_pde_max_effective,
                )
            )

        epoch = int(checkpoint["epoch"])
        metrics_raw = checkpoint.get("metrics", {})
        metrics: Dict[str, float] = {}
        if isinstance(metrics_raw, dict):
            metrics = {str(k): float(v) for k, v in metrics_raw.items()}

        self.last_loaded_epoch = epoch
        self.last_loaded_metrics = metrics
        self.start_epoch = max(1, epoch + 1)
        return epoch, metrics

    def train_step(self) -> Dict[str, float]:
        """
        Executa um passo completo de treino (ou uma iteração do n_critic).

        Inclui a atualização de G e as múltiplas atualizações de D.

        Retorno:
            Métricas agregadas do passo de treinamento.
        """
        batch_size = int(self.cfg.batch_size)
        base = self._expand(self.base_field, batch_size)
        phi = self._expand(self.phi_mask, batch_size)
        ref = self._expand(self.reference_field, batch_size)

        adv1_enabled = float(self.cfg.lambda_adv1) > 0.0
        adv2_enabled = bool(getattr(self.cfg, "use_reference_discriminator", True)) and (
            float(self.cfg.lambda_adv2) > 0.0
        )
        adv_enabled = adv1_enabled or adv2_enabled

        # 1) Atualizações dos críticos: repete n_critic antes de uma atualização de G.
        d1_steps: List[Dict[str, float]] = []
        d2_steps: List[Dict[str, float]] = []
        critics_paused = False
        critics_reduced = False
        disc_update_every = max(1, int(getattr(self.cfg, "disc_update_every", 1)))
        disc_updates_executed = 0.0
        n_critic_effective = self._effective_n_critic()
        update_discriminators = (self._global_step % disc_update_every) == 0
        if adv_enabled:
            self.generator.eval()
            _set_requires_grad(self.generator, False)
            should_pause = self._critics_should_pause()
            if not update_discriminators:
                critics_paused = True
                d1_steps.append(self._zero_d1_metrics())
                d2_steps.append(self._zero_d2_metrics())
            elif should_pause and n_critic_effective > 1:
                # Em over-gap, faz atualização reduzida dos críticos em vez de pausa total.
                # Isso evita colapso adversarial e preserva a estabilidade.
                critics_reduced = True
                self._last_d1_gap *= 0.85
                self._last_d2_gap *= 0.85
                d1_steps.append(self._update_d1(base, phi))
                d2_steps.append(self._update_d2(base, phi, ref))
                disc_updates_executed = 1.0
            else:
                for _ in range(max(1, int(n_critic_effective))):
                    d1_steps.append(self._update_d1(base, phi))
                    d2_steps.append(self._update_d2(base, phi, ref))
                disc_updates_executed = float(max(1, int(n_critic_effective)))
        else:
            self._critics_paused_state = False
            d1_steps.append(self._zero_d1_metrics(gap=0.0))
            d2_steps.append(self._zero_d2_metrics(gap=0.0))

        # 2) Atualização do gerador.
        _set_requires_grad(self.generator, True)
        g_metrics = self._update_generator(base, phi, ref)

        merged = {}
        merged.update(self._merge_metrics(d1_steps))
        merged.update(self._merge_metrics(d2_steps))
        merged.update(g_metrics)
        merged["n_critic"] = float(self.cfg.n_critic)
        merged["n_critic_effective"] = float(n_critic_effective)
        merged["critics_paused"] = 1.0 if critics_paused else 0.0
        merged["critics_reduced"] = 1.0 if critics_reduced else 0.0
        merged["critics_pause_state"] = 1.0 if self._critics_paused_state else 0.0
        merged["disc_updates_executed"] = float(disc_updates_executed)
        self._last_d1_gap = float(merged.get("d1_gap", self._last_d1_gap))
        self._last_d2_gap = float(merged.get("d2_gap", self._last_d2_gap))
        self._global_step += 1

        activation_limit = float(getattr(self.cfg, "activation_abs_limit", 1e6))
        if float(merged.get("g_pred_max_abs", 0.0)) > activation_limit:
            merged["activation_overflow"] = 1.0

        non_finite_metrics = any(
            (not self._is_finite_number(v))
            for v in merged.values()
            if isinstance(v, (float, int, np.floating))
        )
        non_finite_params = self._module_has_non_finite_params(self.generator) or self._module_has_non_finite_params(
            self.discriminator
        )
        if non_finite_metrics or non_finite_params:
            merged["non_finite_detected"] = 1.0
        return merged

    def train(
        self,
        *,
        epoch_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """
        Loop principal de treinamento por múltiplas épocas.

        Parâmetros:
            epoch_callback: Função chamada ao final de cada época com o resumo.

        Retorno:
            Histórico completo de métricas por época.
        """
        history: List[Dict[str, float]] = []
        epochs = int(self.cfg.epochs)
        steps = max(1, int(self.cfg.steps_per_epoch))
        first_epoch = max(1, int(self.start_epoch))
        if first_epoch > epochs:
            if self.logger:
                self.logger.warning(
                    "Checkpoint ja alcanca ou excede epochs alvo; nada a treinar",
                    start_epoch=first_epoch,
                    epochs=epochs,
                )
            return history

        for epoch in range(first_epoch, epochs + 1):
            self._current_epoch = int(epoch)
            epoch_logs: Dict[str, List[float]] = {}
            for _ in range(steps):
                try:
                    step_metrics = self.train_step()
                except FloatingPointError as exc:
                    self._stop_requested = True
                    self._stop_reason = str(exc)
                    step_metrics = {
                        "non_finite_detected": 1.0,
                        "activation_overflow": 0.0,
                        "g_total": float("nan"),
                    }
                for key, value in step_metrics.items():
                    epoch_logs.setdefault(key, []).append(float(value))
                if bool(getattr(self.cfg, "early_stop_on_nonfinite", True)) and (
                    float(step_metrics.get("non_finite_detected", 0.0)) > 0.0
                ):
                    self._stop_requested = True
                    self._stop_reason = "detected NaN/Inf in parameters or metrics"
                    break
                if float(step_metrics.get("activation_overflow", 0.0)) > 0.0:
                    self._stop_requested = True
                    self._stop_reason = "exploding activations detected"
                    break

            summary = {"epoch": int(epoch)}
            for key, values in epoch_logs.items():
                summary[key] = float(np.mean(values))
            summary["lr_g"] = float(self.opt_g.param_groups[0]["lr"])
            summary["lr_d1"] = float(self.opt_d1.param_groups[0]["lr"])
            summary["lr_d2"] = float(self.opt_d2.param_groups[0]["lr"])

            lr_drop_triggered = 0.0
            lr_plateau_triggered = 0.0
            lr_update = self._maybe_reduce_lrs_on_drift(summary)
            if lr_update is not None:
                summary.update(
                    {
                        "lr_g": float(lr_update["lr_g"]),
                        "lr_d1": float(lr_update["lr_d1"]),
                        "lr_d2": float(lr_update["lr_d2"]),
                    }
                )
                lr_drop_triggered = 1.0
            else:
                plateau_update = self._maybe_reduce_lrs_on_plateau(summary)
                if plateau_update is not None:
                    summary.update(
                        {
                            "lr_g": float(plateau_update["lr_g"]),
                            "lr_d1": float(plateau_update["lr_d1"]),
                            "lr_d2": float(plateau_update["lr_d2"]),
                        }
                    )
                    lr_plateau_triggered = 1.0
            summary["lr_drop_triggered"] = float(lr_drop_triggered)
            summary["lr_plateau_triggered"] = float(lr_plateau_triggered)
            history.append(summary)
            if epoch_callback is not None:
                epoch_callback(summary)

            if self.logger and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                self.logger.info(
                    f"[Epoch {epoch}/{epochs}]",
                    g_total=f"{summary.get('g_total', 0.0):.4e}",
                    g_pde=f"{summary.get('g_pde', 0.0):.4e}",
                    g_pde_raw=f"{summary.get('g_pde_raw', 0.0):.4e}",
                    g_bc=f"{summary.get('g_bc', 0.0):.4e}",
                    d1_total=f"{summary.get('d1_total', 0.0):.4e}",
                    d2_total=f"{summary.get('d2_total', 0.0):.4e}",
                    d1_gap=f"{summary.get('d1_gap', 0.0):.4e}",
                    d2_gap=f"{summary.get('d2_gap', 0.0):.4e}",
                    d1_gp_lambda=f"{summary.get('d1_gp_lambda', 0.0):.3f}",
                    d2_gp_lambda=f"{summary.get('d2_gp_lambda', 0.0):.3f}",
                    d1_grad=f"{summary.get('d1_grad_norm', 0.0):.4e}",
                    d2_grad=f"{summary.get('d2_grad_norm', 0.0):.4e}",
                    g_grad=f"{summary.get('g_grad_norm', 0.0):.4e}",
                    adv_over_pde=f"{summary.get('g_adv_over_pde', 0.0):.4e}",
                    adv_gate=f"{summary.get('g_adv_gate', 1.0):.3f}",
                    adv_grad_scale=f"{summary.get('g_adv_grad_scale', 1.0):.3f}",
                    adv_prog_mul=f"{summary.get('g_adv_progressive_multiplier', 1.0):.3f}",
                    lambda_pde=f"{summary.get('g_lambda_pde_dyn', 0.0):.3e}",
                    lambda_pde_cap=f"{summary.get('g_lambda_pde_cap_eff', 0.0):.3e}",
                    pde_raw=f"{summary.get('g_pde_raw_penalty', 0.0):.4e}",
                    residual_l2=f"{summary.get('g_residual_l2', 0.0):.4e}",
                    residual_mean_abs=f"{summary.get('g_residual_mean_abs', 0.0):.4e}",
                    residual_max_abs=f"{summary.get('g_residual_max_abs', 0.0):.4e}",
                    bc_max_abs=f"{summary.get('g_bc_max_abs', 0.0):.4e}",
                    lr_g=f"{summary.get('lr_g', 0.0):.2e}",
                    lr_d1=f"{summary.get('lr_d1', 0.0):.2e}",
                    lr_d2=f"{summary.get('lr_d2', 0.0):.2e}",
                    disc_step=int(summary.get("disc_updates_executed", 0.0)),
                    n_critic_eff=int(summary.get("n_critic_effective", summary.get("n_critic", 1.0))),
                    critics_paused=int(summary.get("critics_paused", 0.0)),
                    critics_reduced=int(summary.get("critics_reduced", 0.0)),
                )

            periodic_save = self.save_frequency > 0 and (epoch % self.save_frequency == 0)
            final_save = self.save_frequency > 0 and epoch == epochs and not periodic_save
            if periodic_save or final_save:
                if self.checkpoint_dir is not None:
                    ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                    self.save_checkpoint(ckpt_path, epoch, summary)
                elif self.logger:
                    self.logger.warning(
                        "Checkpoint nao salvo: checkpoint_dir ausente",
                        epoch=epoch,
                    )
            if self._stop_requested:
                if self.logger:
                    self.logger.warning(
                        "Early stop acionado por seguranca numerica",
                        epoch=epoch,
                        reason=self._stop_reason or "unspecified",
                    )
                break
        if history and self.logger:
            last_residual = float(history[-1].get("g_residual_mean_abs", np.nan))
            target = float(getattr(self.cfg, "residual_tolerance_target", 1e-3))
            if np.isfinite(last_residual) and last_residual > target:
                self.logger.warning(
                    "Meta de tolerancia fisica nao atingida",
                    residual_mean_abs=f"{last_residual:.4e}",
                    tolerance_target=f"{target:.4e}",
                )
        return history

    @torch.no_grad()
    def predict(self, num_samples: int = 1) -> torch.Tensor:
        """
        Gera predições do modelo para o domínio configurado.

        Parâmetros:
            num_samples: Quantidade de amostras a gerar do ensemble stocástico.

        Retorno:
            Tensor formatado [S, 1, H, W] com as predições de temperatura.
        """
        batch_size = max(1, int(num_samples))
        base = self._expand(self.base_field, batch_size)
        phi = self._expand(self.phi_mask, batch_size)

        # Sem termos adversariais, o modelo opera como solucionador determinístico tipo PINN.
        deterministic = (
            str(self.cfg.generator_mode) == "deterministic_adversarial"
            or (
                float(self.cfg.lambda_adv1) == 0.0
                and (
                    float(self.cfg.lambda_adv2) == 0.0
                    or (not bool(getattr(self.cfg, "use_reference_discriminator", True)))
                )
            )
        )
        z = self._sample_latent(batch_size, deterministic=deterministic)

        self.generator.eval()
        coords = self._expand_coords(batch_size)
        return self.generator(base, phi, z=z, coord_field=coords)

