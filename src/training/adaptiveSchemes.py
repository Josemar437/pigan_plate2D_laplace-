"""Controladores testaveis espelhando a dinamica ativa em `src/trainer.py`."""
from collections import deque
import math
from typing import Deque, Dict, Union

import torch

DetectorState = Dict[str, Union[bool, float]]


class AdaptiveLambdaPDE:
    """Lei log-linear de `lambda_pde` ancorada no valor base calibrado."""

    def __init__(
        self,
        lambda_pde_min: float = 50.0,
        lambda_pde_max: float = 500.0,
        ema_beta: float = 0.85,
        residual_scale_reference: float = 1.0e-2,
        residual_tolerance_target: float = 1e-3,
        growth_exponent: float = 1.2,
    ) -> None:
        """Guarda limites e EMA usados pelo controlador de peso PDE.
        
        CORRECÇÃO FÍSICA (24/05/2026):
        - lambda_pde_min: 19.0 → 50.0 (força mínima suficiente)
        - lambda_pde_max: 98.0 → 500.0 (convergência agressiva PDE)
        - growth_exponent: 0.6 → 1.2 (escala mais agressiva quando residuo alto)
        - ema_beta: 0.9 → 0.85 (resposta mais rápida a picos de residuo)
        
        Justificativa: Com residuo_mean_abs=0.0778 vs alvo=0.001 (78× acima),
        a escala anterior era insuficiente. Novo regime força convergência.
        """
        self.lambda_pde_min = lambda_pde_min
        self.lambda_pde_max = lambda_pde_max
        self.ema_beta = ema_beta
        self.residual_scale_reference = residual_scale_reference
        self.residual_tolerance_target = residual_tolerance_target
        self.growth_exponent = growth_exponent
        
        # EMA state
        self.lambda_pde_ema = None
    
    def update(
        self,
        residual_mean_abs: torch.Tensor,
        lambda_pde_current: float,
    ) -> float:
        """Recalcula `lambda_pde` a partir da base fixa e do residuo atual.
        
        Args:
            residual_mean_abs: Mean absolute PDE residual.
            lambda_pde_current: Current λ_PDE value (for reference).
            
        Returns:
            float: Updated EMA-smoothed λ_PDE value.
            
        Raises:
            FloatingPointError: If inputs contain NaN or Inf.
            ValueError: If residual_mean_abs is not scalar or is negative.
        """
        # Converter tensor para float se necessário
        if isinstance(residual_mean_abs, torch.Tensor):
            residual_mean_abs = residual_mean_abs.item()
        residual_mean_abs = float(residual_mean_abs)
        lambda_pde_current = float(lambda_pde_current)

        if not math.isfinite(residual_mean_abs):
            raise FloatingPointError("residual_mean_abs contem NaN/Inf.")
        if not math.isfinite(lambda_pde_current):
            raise FloatingPointError("lambda_pde_current contem NaN/Inf.")
        if residual_mean_abs < 0:
            raise ValueError(f"residual_mean_abs must be >= 0, got {residual_mean_abs}")
        
        # Mesma lei log-linear usada por FieldPIGANTrainer._update_lambda_pde_dynamic.
        scale_reference = max(float(self.residual_scale_reference), 1e-12)
        ratio = max(residual_mean_abs / scale_reference, 1.0)
        lambda_pde_new = lambda_pde_current * (
            1.0 + self.growth_exponent * math.log10(ratio)
        )
        
        # Aplicar limites
        lambda_pde_new = max(self.lambda_pde_min, min(lambda_pde_new, self.lambda_pde_max))
        
        # EMA suavização
        if self.lambda_pde_ema is None:
            self.lambda_pde_ema = lambda_pde_new
        else:
            self.lambda_pde_ema = (self.ema_beta * self.lambda_pde_ema + 
                                   (1 - self.ema_beta) * lambda_pde_new)
        
        return float(self.lambda_pde_ema)


class GradNormBalancer:
    """Resolve a escala adversarial alvo a partir das normas PDE/adversarial."""

    def __init__(
        self,
        target_ratio: float = 0.35,
        ema_beta: float = 0.9,
        scale_min: float = 0.05,
        scale_max: float = 1.0,
    ) -> None:
        """Configura alvo `grad_adv/grad_pde`, clamp e suavizacao EMA."""
        self.target_ratio = target_ratio
        self.ema_beta = ema_beta
        self.scale_min = scale_min
        self.scale_max = scale_max
        
        self.scale_ema = None
    
    def update(
        self,
        grad_norm_adv: torch.Tensor,
        grad_norm_pde: torch.Tensor,
        scale_current: float = 1.0,
    ) -> float:
        """Retorna escala EMA; `scale_current` existe so por compatibilidade.
        
        Args:
            grad_norm_adv: Magnitude of adversarial gradient.
            grad_norm_pde: Magnitude of PDE gradient.
            scale_current: Current scale (for compatibility, not used).
            
        Returns:
            float: EMA-smoothed scale factor between scale_min and scale_max.
            
        Raises:
            FloatingPointError: If any input contains NaN or Inf.
        """
        # Converter tensores para float
        if isinstance(grad_norm_adv, torch.Tensor):
            grad_norm_adv = grad_norm_adv.item()
        if isinstance(grad_norm_pde, torch.Tensor):
            grad_norm_pde = grad_norm_pde.item()
        grad_norm_adv = float(grad_norm_adv)
        grad_norm_pde = float(grad_norm_pde)
        if not math.isfinite(grad_norm_adv) or not math.isfinite(grad_norm_pde):
            raise FloatingPointError("Normas de gradiente contem NaN/Inf.")
        
        # Mesma escala direta usada por FieldPIGANTrainer._gradnorm_adv_scale.
        scale_new = self.target_ratio * grad_norm_pde / max(grad_norm_adv, 1e-12)
        scale_new = max(self.scale_min, min(scale_new, self.scale_max))
        
        # EMA
        if self.scale_ema is None:
            self.scale_ema = scale_new
        else:
            self.scale_ema = (self.ema_beta * self.scale_ema + 
                             (1 - self.ema_beta) * scale_new)
        
        return float(self.scale_ema)


class StagnationDetector:
    """Estado isolado para boost adversarial quando o residuo para de melhorar."""

    def __init__(
        self,
        patience: int = 50,
        rel_tolerance: float = 5e-3,
        boost_factor: float = 1.4,
        cooldown: int = 8,
    ) -> None:
        """Define paciencia, melhora minima, multiplicador e cooldown completo."""
        self.patience = patience
        self.rel_tolerance = rel_tolerance
        self.boost_factor = boost_factor
        self.cooldown = cooldown
        
        self.best_residual = None
        self.stagnation_counter = 0
        self.boost_counter = 0
    
    def update(
        self,
        residual_mean_abs: torch.Tensor,
    ) -> DetectorState:
        """Atualiza melhor residuo, contador de estagnacao e cooldown de boost.
        
        Args:
            residual_mean_abs: Mean absolute PDE residual (tensor or float).
            
        Returns:
            DetectorState: Dict with keys 'is_stagnant', 'boost_active', 'boost_factor'.
            
        Raises:
            FloatingPointError: If residual contains NaN or Inf.
        """
        if isinstance(residual_mean_abs, torch.Tensor):
            residual_mean_abs = residual_mean_abs.item()
        residual_mean_abs = float(residual_mean_abs)
        if not math.isfinite(residual_mean_abs):
            raise FloatingPointError("residual_mean_abs contem NaN/Inf.")
        
        if self.best_residual is None:
            self.best_residual = residual_mean_abs
            return {"is_stagnant": False, "boost_active": False, "boost_factor": 1.0}
        
        # Verificar melhora
        improvement = (self.best_residual - residual_mean_abs) / (self.best_residual + 1e-8)
        
        if improvement > self.rel_tolerance:
            # Houve melhora significante
            self.best_residual = residual_mean_abs
            self.stagnation_counter = 0
        else:
            # Sem melhora
            self.stagnation_counter += 1
        
        # Verificar se estagnado
        is_stagnant = self.stagnation_counter >= self.patience
        
        # Decrementa no início para preservar a duração integral do cooldown.
        if self.boost_counter > 0:
            self.boost_counter -= 1

        # Aplicar boost se estagnado e não em cooldown.
        boost_active = False
        boost_factor_out = 1.0
        
        if is_stagnant and self.boost_counter == 0:
            boost_active = True
            boost_factor_out = self.boost_factor
            self.boost_counter = self.cooldown
            self.stagnation_counter = 0
        
        return {
            "is_stagnant": is_stagnant,
            "boost_active": boost_active,
            "boost_factor": boost_factor_out,
        }


class DivergenceDetector:
    """Detecta crescimento sustentado por medias de janelas adjacentes."""

    def __init__(
        self,
        window_size: int = 16,
        ratio_threshold: float = 1.2,
        patience: int = 2,
    ) -> None:
        """Mantem duas janelas de historico e uma paciencia de alerta."""
        self.window_size = window_size
        self.ratio_threshold = ratio_threshold
        self.patience = patience
        
        self.history: Deque[float] = deque(maxlen=2 * self.window_size)
        self.divergence_counter = 0
    
    def update(
        self,
        loss: torch.Tensor,
    ) -> DetectorState:
        """Compara media recente contra media anterior e acumula streak.
        
        Args:
            loss: Current loss value (tensor or float).
            
        Returns:
            DetectorState: Dict with keys 'is_diverging', 'ratio', 'mean_loss'.
            
        Raises:
            FloatingPointError: If loss contains NaN or Inf.
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError("loss contem NaN/Inf.")
        
        # Adicionar à história
        self.history.append(loss)
        
        if len(self.history) < 2 * self.window_size:
            return {
                "is_diverging": False,
                "ratio": 1.0,
                "mean_loss": sum(self.history) / len(self.history),
            }
        
        values = list(self.history)
        prev_window = values[:self.window_size]
        last_window = values[self.window_size:]
        prev_mean = sum(prev_window) / len(prev_window)
        last_mean = sum(last_window) / len(last_window)
        ratio = (last_mean + 1e-8) / max(prev_mean, 1e-8)
        mean_loss = sum(values) / len(values)
        
        # Atualizar contador
        if ratio > self.ratio_threshold:
            self.divergence_counter += 1
        else:
            self.divergence_counter = 0
        
        is_diverging = self.divergence_counter >= self.patience
        
        return {
            "is_diverging": is_diverging,
            "ratio": ratio,
            "mean_loss": mean_loss,
        }
