"""Esquemas adaptativos para escalonamento dinâmico de pesos durante treinamento."""
from typing import Dict, Optional

import torch


class AdaptiveLambdaPDE:
    """
    Adaptador dinâmico de λ_PDE baseado em resíduo PDE.
    
    Implementa:
    - Escalonamento baseado em escala de resíduo
    - EMA (exponential moving average) para suavização
    - Limites mín/máx para estabilidade
    """

    def __init__(
        self,
        lambda_pde_min: float = 19.0,
        lambda_pde_max: float = 98.0,
        ema_beta: float = 0.9,
        residual_scale_reference: float = 1.0,
        residual_tolerance_target: float = 1e-3,
        growth_exponent: float = 0.6,
    ) -> None:
        """
        Inicializa adaptador de λ_PDE.
        
        Parâmetros:
            lambda_pde_min: Limite inferior de λ_PDE
            lambda_pde_max: Limite superior de λ_PDE
            ema_beta: Fator EMA (0.9 = suave, 0.5 = rápido)
            residual_scale_reference: Escala de referência para normalização
            residual_tolerance_target: Alvo de tolerância de resíduo
            growth_exponent: Expoente para curva de crescimento
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
        epoch: int = 0,
    ) -> float:
        """
        Atualiza λ_PDE baseado em resíduo atual.
        
        Parâmetros:
            residual_mean_abs: Resíduo PDE médio absoluto (escalar ou tensor)
            lambda_pde_current: Valor atual de λ_PDE
            epoch: Época atual (para annealing opcional)
        
        Retorno:
            float: Novo valor de λ_PDE
        """
        # Converter tensor para float se necessário
        if isinstance(residual_mean_abs, torch.Tensor):
            residual_mean_abs = residual_mean_abs.item()
        
        # Computar razão de desempenho
        ratio = residual_mean_abs / (self.residual_scale_reference + 1e-8)
        
        # Aplicar função de crescimento
        growth_factor = (ratio / (self.residual_tolerance_target + 1e-8)) ** self.growth_exponent
        growth_factor = max(0.5, min(growth_factor, 2.0))  # Clip para estabilidade
        
        # Novo λ sem annealing
        lambda_pde_new = lambda_pde_current * growth_factor
        
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
    """
    Balanceador de normas de gradiente para estabilizar treinamento.
    
    Mantém razão entre gradientes de adversarial e PDE em nível alvo.
    """

    def __init__(
        self,
        target_ratio: float = 0.35,
        ema_beta: float = 0.9,
        scale_min: float = 0.05,
        scale_max: float = 1.0,
    ) -> None:
        """
        Inicializa balanceador de normas.
        
        Parâmetros:
            target_ratio: Razão alvo (grad_adv / grad_pde)
            ema_beta: Fator EMA
            scale_min: Escala mínima
            scale_max: Escala máxima
        """
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
        """
        Atualiza escala de balanceamento.
        
        Parâmetros:
            grad_norm_adv: Norma de gradiente do termo adversarial
            grad_norm_pde: Norma de gradiente do termo PDE
            scale_current: Escala atual (para informação)
        
        Retorno:
            float: Nova escala de balanceamento
        """
        # Converter tensores para float
        if isinstance(grad_norm_adv, torch.Tensor):
            grad_norm_adv = grad_norm_adv.item()
        if isinstance(grad_norm_pde, torch.Tensor):
            grad_norm_pde = grad_norm_pde.item()
        
        # Razão observada
        current_ratio = (grad_norm_adv + 1e-8) / (grad_norm_pde + 1e-8)
        
        # Fator de ajuste
        adjustment = self.target_ratio / (current_ratio + 1e-8)
        adjustment = max(0.8, min(adjustment, 1.25))  # Clip para suavidade
        
        # Nova escala
        scale_new = scale_current * adjustment
        scale_new = max(self.scale_min, min(scale_new, self.scale_max))
        
        # EMA
        if self.scale_ema is None:
            self.scale_ema = scale_new
        else:
            self.scale_ema = (self.ema_beta * self.scale_ema + 
                             (1 - self.ema_beta) * scale_new)
        
        return float(self.scale_ema)


class StagnationDetector:
    """
    Detecta estagnação de treinamento e aplica boosters.
    
    Monitora histórico de resíduo e ativa boost quando progresso cessa.
    """

    def __init__(
        self,
        patience: int = 50,
        rel_tolerance: float = 5e-3,
        boost_factor: float = 1.4,
        cooldown: int = 8,
    ) -> None:
        """
        Inicializa detector de estagnação.
        
        Parâmetros:
            patience: Épocas sem melhora antes de boost
            rel_tolerance: Melhora relativa mínima para resetar contador
            boost_factor: Fator multiplicativo de boost
            cooldown: Épocas antes de permitir novo boost
        """
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
    ) -> Dict[str, bool | float]:
        """
        Atualiza estado de estagnação.
        
        Parâmetros:
            residual_mean_abs: Resíduo PDE médio atual
        
        Retorno:
            Dicionário com:
                - is_stagnant: bool, se estagnação foi detectada
                - boost_active: bool, se boost está ativo
                - boost_factor: float, fator a aplicar se boost ativo
        """
        if isinstance(residual_mean_abs, torch.Tensor):
            residual_mean_abs = residual_mean_abs.item()
        
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
        
        # Aplicar boost se estagnado e não em cooldown
        boost_active = False
        boost_factor_out = 1.0
        
        if is_stagnant and self.boost_counter == 0:
            boost_active = True
            boost_factor_out = self.boost_factor
            self.boost_counter = self.cooldown
        
        # Decrementar cooldown
        if self.boost_counter > 0:
            self.boost_counter -= 1
        
        return {
            "is_stagnant": is_stagnant,
            "boost_active": boost_active,
            "boost_factor": boost_factor_out,
        }


class DivergenceDetector:
    """
    Detecta divergência (blow-up) de loss e alerta.
    
    Monitora razão de perdas em janela deslizante.
    """

    def __init__(
        self,
        window_size: int = 16,
        ratio_threshold: float = 1.2,
        patience: int = 2,
    ) -> None:
        """
        Inicializa detector de divergência.
        
        Parâmetros:
            window_size: Tamanho da janela de histórico
            ratio_threshold: Razão max/min para sinalizar divergência
            patience: Épocas consecutivas acima do limiar antes de alertar
        """
        self.window_size = window_size
        self.ratio_threshold = ratio_threshold
        self.patience = patience
        
        self.history = []
        self.divergence_counter = 0
    
    def update(
        self,
        loss: torch.Tensor,
    ) -> Dict[str, bool | float]:
        """
        Atualiza histórico de perdas e detecta divergência.
        
        Parâmetros:
            loss: Valor de loss (escalar)
        
        Retorno:
            Dicionário com:
                - is_diverging: bool
                - ratio: float, razão max/min
                - mean_loss: float
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        
        # Adicionar à história
        self.history.append(loss)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Computar razão se houver histórico suficiente
        if len(self.history) < 4:
            return {
                "is_diverging": False,
                "ratio": 1.0,
                "mean_loss": loss,
            }
        
        min_loss = min(self.history)
        max_loss = max(self.history)
        ratio = (max_loss + 1e-8) / (min_loss + 1e-8)
        mean_loss = sum(self.history) / len(self.history)
        
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
