# Guia de Implementação — Refatoração da PI-GAN

**Data:** 22 de maio de 2026  
**Objetivo:** Executar refatoração em 3 fases sem quebrar P1/P2/P3

---

## FASE 1: Preparação (Sem Mudanças no Código)

### 1.1 Backup e Versionamento

```bash
# Criar branch para refatoração
git checkout -b refactor/architecture-refactoring

# Verificar status atual
python -m pytest tests/ -v
python validate_skill_corrections.py
python verify_hard_constraint.py
```

### 1.2 Checklist de Validação Antes

- [ ] Todos os testes passam
- [ ] P1/P2/P3 validados
- [ ] Último checkpoint carregável
- [ ] Documentação atualizada

---

## FASE 2: Refatoração Estruturada (Modular)

### 2.1 Passo 1: Criar Módulo `src/physics/`

```bash
mkdir -p src/physics
touch src/physics/__init__.py
touch src/physics/pde_residual.py
touch src/physics/domain_metrics.py
touch src/physics/boundary_conditions.py
```

**Arquivo: `src/physics/__init__.py`**

```python
from .pde_residual import PDE_Residual_Computer
from .domain_metrics import compute_physics_metrics
from .boundary_conditions import compute_boundary_loss

__all__ = [
    "PDE_Residual_Computer",
    "compute_physics_metrics",
    "compute_boundary_loss",
]
```

**Arquivo: `src/physics/pde_residual.py`**

```python
import torch
import torch.nn as nn
from typing import Optional

from src.models import LaplacianLayer


class PDE_Residual_Computer:
    """Calcula e pondera resíduos da PDE."""
    
    def __init__(self, laplacian: LaplacianLayer):
        """
        Parâmetros:
            laplacian: Camada de Laplaciano.
        """
        self.laplacian = laplacian
    
    def compute_residual(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computa ∇²T.
        
        Parâmetros:
            field: Campo [B, 1, H, W].
        
        Retorno:
            Resíduo [B, 1, H-2, W-2].
        """
        return self.laplacian(field)
    
    def compute_weighted_loss(
        self,
        residual: torch.Tensor,
        interior_mask: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcula perda ponderada: E[|resíduo| · w].
        
        Parâmetros:
            residual: Resíduo [B, 1, H-2, W-2].
            interior_mask: Máscara binária do interior [B, 1, H, W].
            weight_map: Pesos opcionais (ex: para focus em cantos).
        
        Retorno:
            Perda escalar.
        """
        # Pad residual para match interior_mask
        b, c, h_res, w_res = residual.shape
        padded = torch.zeros_like(interior_mask)
        padded[:, :, 1:-1, 1:-1] = residual
        
        abs_residual = padded.abs()
        
        if weight_map is not None:
            numerator = (abs_residual * weight_map).sum()
            denominator = weight_map.sum()
        else:
            numerator = (abs_residual * interior_mask).sum()
            denominator = interior_mask.sum()
        
        return numerator / (denominator + 1e-12)
```

**Arquivo: `src/physics/domain_metrics.py`**

```python
import torch
from typing import Dict

from src.models import LaplacianLayer


def compute_physics_metrics(
    pred_field: torch.Tensor,
    laplacian: LaplacianLayer,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    base_field: torch.Tensor,
) -> Dict[str, float]:
    """
    Calcula métricas físicas do campo predito.
    
    Parâmetros:
        pred_field: Campo predito [1, 1, H, W].
        laplacian: Camada de Laplaciano.
        interior_mask: Máscara do interior.
        boundary_mask: Máscara da fronteira.
        base_field: Campo base (extensão Dirichlet).
    
    Retorno:
        Dicionário com métricas.
    """
    with torch.no_grad():
        residual = laplacian(pred_field)
        
        # Pad residual
        b, c, h_res, w_res = residual.shape
        padded_residual = torch.zeros_like(pred_field)
        padded_residual[:, :, 1:-1, 1:-1] = residual
        
        # Métricas de resíduo PDE
        pde_residual_mean = (padded_residual.abs() * interior_mask).sum() / (interior_mask.sum() + 1e-12)
        pde_residual_l2 = torch.sqrt(
            ((padded_residual ** 2) * interior_mask).sum() / (interior_mask.sum() + 1e-12)
        )
        pde_residual_max = (padded_residual.abs() * interior_mask).max()
        
        # Erro de contorno
        boundary_error = (
            (pred_field - base_field).abs() * boundary_mask
        ).sum() / (boundary_mask.sum() + 1e-12)
        
    return {
        "pde_residual_mean": float(pde_residual_mean.item()),
        "pde_residual_l2": float(pde_residual_l2.item()),
        "pde_residual_max": float(pde_residual_max.item()),
        "boundary_error": float(boundary_error.item()),
    }


def compute_boundary_loss(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula MSE nas fronteiras.
    
    Parâmetros:
        pred_field: Campo predito.
        target_field: Campo alvo (ex: base_field ou ref_field).
        boundary_mask: Máscara da fronteira.
    
    Retorno:
        Perda escalar.
    """
    diff_sq = (pred_field - target_field) ** 2
    numerator = (diff_sq * boundary_mask).sum()
    denominator = boundary_mask.sum() + 1e-12
    return numerator / denominator
```

---

### 2.2 Passo 2: Criar Módulo `src/training/`

```bash
mkdir -p src/training
touch src/training/__init__.py
touch src/training/loss_functions.py
touch src/training/adaptive_schemes.py
```

**Arquivo: `src/training/__init__.py`**

```python
from .loss_functions import PIGANLossComputation
from .adaptive_schemes import (
    AdaptiveLambdaPDE,
    GradNormBalancer,
    StagnationDetector,
    DivergenceDetector,
)

__all__ = [
    "PIGANLossComputation",
    "AdaptiveLambdaPDE",
    "GradNormBalancer",
    "StagnationDetector",
    "DivergenceDetector",
]
```

**Arquivo: `src/training/loss_functions.py`**

```python
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from src.models import LaplacianLayer
from src.config import ExperimentConfig
from src.physics import PDE_Residual_Computer, compute_boundary_loss


class PIGANLossComputation:
    """Calcula componentes de loss da PI-GAN."""
    
    def __init__(
        self,
        laplacian: LaplacianLayer,
        config: ExperimentConfig,
    ):
        self.pde_computer = PDE_Residual_Computer(laplacian)
        self.config = config
    
    def compute_pde_loss(
        self,
        pred_field: torch.Tensor,
        interior_mask: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcula L_PDE = E[|∇²T|].
        """
        residual = self.pde_computer.compute_residual(pred_field)
        return self.pde_computer.compute_weighted_loss(
            residual,
            interior_mask,
            weight_map,
        )
    
    def compute_adversarial_loss(
        self,
        discriminator: nn.Module,
        pred_field: torch.Tensor,
        ref_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula L_adv = -E[D(T_pred, T_ref)].
        """
        fake_score = discriminator(
            torch.cat([pred_field, ref_field], dim=1)
        )
        return -fake_score.mean()
    
    def compute_boundary_loss(
        self,
        pred_field: torch.Tensor,
        target_field: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula L_BC = MSE na fronteira.
        """
        return compute_boundary_loss(pred_field, target_field, boundary_mask)
    
    def compute_diversity_loss(
        self,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula L_div = -E[||T_i - T_j||].
        
        Parâmetros:
            samples: Múltiplas amostras [N, 1, H, W].
        
        Retorno:
            Perda negativa (para encorajar diversidade).
        """
        n_samples = int(samples.shape[0])
        if n_samples <= 1:
            return torch.tensor(0.0, device=samples.device, dtype=samples.dtype)
        
        # Flatten espacial
        flat = samples.reshape(n_samples, -1)
        spatial_size = max(1, flat.shape[1])
        
        # Média de distâncias L2 par a par
        pairwise_dist = torch.pdist(flat, p=2).mean()
        mean_dist = pairwise_dist / spatial_size
        
        return -mean_dist  # Negativo para maximizar diversidade
    
    def compose_generator_loss(
        self,
        loss_pde: torch.Tensor,
        loss_adv: torch.Tensor,
        loss_bc: torch.Tensor,
        loss_div: torch.Tensor,
        lambda_pde_dyn: float,
        lambda_adv_eff: float,
    ) -> torch.Tensor:
        """
        Compõe loss total do gerador.
        
        L_G = λ_pde_dyn·L_pde + λ_adv_eff·L_adv + λ_bc·L_bc + λ_div·L_div
        """
        loss_g = lambda_pde_dyn * loss_pde
        loss_g = loss_g + lambda_adv_eff * loss_adv
        
        if not self.config.hard_constraint_bc:
            loss_g = loss_g + self.config.lambda_bc * loss_bc
        
        if self.config.latent_dim > 0:
            loss_g = loss_g + self.config.lambda_diversity * loss_div
        
        return loss_g
    
    def compute_discriminator_loss(
        self,
        discriminator: nn.Module,
        pred_field: torch.Tensor,
        ref_field: torch.Tensor,
        gradient_penalty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula L_D para WGAN-GP.
        
        L_D = E[D(fake)] - E[D(real)] + λ_gp·GP + λ_drift·E[D(real)²]
        """
        fake_logit = discriminator(
            torch.cat([pred_field, ref_field], dim=1)
        )
        real_logit = discriminator(
            torch.cat([ref_field, ref_field], dim=1)
        )
        
        loss_d = fake_logit.mean() - real_logit.mean()
        loss_d = loss_d + self.config.lambda_gp * gradient_penalty
        
        if self.config.critic_drift > 0:
            drift_penalty = self.config.critic_drift * (real_logit ** 2).mean()
            loss_d = loss_d + drift_penalty
        
        return loss_d
```

**Arquivo: `src/training/adaptive_schemes.py`**

```python
import numpy as np
import torch
from collections import deque
from typing import Tuple

from src.config import ExperimentConfig


class AdaptiveLambdaPDE:
    """Adaptação dinâmica de λ_PDE via EMA e log-clipping."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ema_value = float(config.lambda_pde)
    
    def update(
        self,
        loss_pde: float,
        reference_scale: float,
    ) -> float:
        """
        Atualiza λ_PDE dinamicamente.
        
        ρ = max(L_PDE / L_0, 1.0)
        λ_des = clip(λ_base · (1 + α·log₁₀(ρ)), λ_min, λ_max)
        λ_PDE ← β·λ_PDE + (1-β)·λ_des
        """
        rho = max(float(loss_pde) / float(reference_scale), 1.0)
        
        log_rho = np.log10(max(rho, 1e-12))
        lambda_desired = (
            self.config.lambda_pde *
            (1.0 + self.config.lambda_pde_growth_exponent * log_rho)
        )
        
        lambda_desired = np.clip(
            lambda_desired,
            float(self.config.lambda_pde_min),
            float(self.config.lambda_pde_max),
        )
        
        beta = float(self.config.lambda_pde_ema_beta)
        self._ema_value = (
            beta * self._ema_value +
            (1.0 - beta) * lambda_desired
        )
        
        return self._ema_value
    
    @property
    def current_value(self) -> float:
        return self._ema_value


class GradNormBalancer:
    """Balanceia escalas de loss via normas de gradiente."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ema_scale = 1.0
    
    def update(
        self,
        grad_norm_adv: float,
        grad_norm_pde: float,
    ) -> float:
        """
        Calcula escala para λ_adv_eff baseada na razão de normas.
        
        ratio = ||∇L_adv|| / ||∇L_pde||
        target = λ_adv_target / λ_pde
        scale = target / ratio
        scale ← β·scale + (1-β)·scale_novo, clipped
        """
        ratio = float(grad_norm_adv) / (float(grad_norm_pde) + 1e-12)
        target_ratio = float(self.config.gradnorm_target_adv_to_pde)
        
        desired_scale = target_ratio / (ratio + 1e-12)
        
        desired_scale = np.clip(
            desired_scale,
            float(self.config.gradnorm_scale_min),
            float(self.config.gradnorm_scale_max),
        )
        
        beta = float(self.config.gradnorm_ema_beta)
        self._ema_scale = (
            beta * self._ema_scale +
            (1.0 - beta) * desired_scale
        )
        
        return self._ema_scale
    
    @property
    def current_scale(self) -> float:
        return self._ema_scale


class StagnationDetector:
    """Detecta estagnação de resíduo e gera boost."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._steps_without_improvement = 0
        self._best_residual = float("inf")
        self._cooldown_counter = 0
    
    def update(self, residual: float) -> Tuple[bool, float]:
        """
        Detecta estagnação e retorna (is_stagnant, boost_factor).
        
        Melhoria considerada se:
        improvement > rel_tol · max(best_residual, 1e-8)
        """
        self._cooldown_counter = max(0, self._cooldown_counter - 1)
        
        improvement = self._best_residual - float(residual)
        threshold = (
            float(self.config.adv_stagnation_rel_tol) *
            max(abs(self._best_residual), 1e-8)
        )
        
        if improvement > threshold:
            self._best_residual = float(residual)
            self._steps_without_improvement = 0
            boost = 1.0
        else:
            self._steps_without_improvement += 1
            
            if (self._steps_without_improvement >= self.config.adv_stagnation_patience
                and self._cooldown_counter == 0):
                boost = float(self.config.adv_stagnation_boost_factor)
                self._cooldown_counter = int(self.config.adv_stagnation_cooldown)
            else:
                boost = 1.0
        
        is_stagnant = (
            self._steps_without_improvement >=
            self.config.adv_stagnation_patience
        )
        
        return is_stagnant, boost
    
    @property
    def steps_without_improvement(self) -> int:
        return self._steps_without_improvement


class DivergenceDetector:
    """Detecta divergência de loss via janela recente."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._loss_window: deque[float] = deque(
            maxlen=max(4, int(config.divergence_window))
        )
    
    def update(self, loss: float) -> bool:
        """
        Detecta divergência se:
        mean(4 perdas recentes) / mean(4 perdas antigas) > threshold
        """
        self._loss_window.append(float(loss))
        
        if len(self._loss_window) < 8:
            return False
        
        window_list = list(self._loss_window)
        recent_mean = np.mean(window_list[-4:])
        old_mean = np.mean(window_list[:4])
        
        ratio = recent_mean / (old_mean + 1e-12)
        threshold = float(self.config.divergence_ratio_threshold)
        
        is_diverging = ratio > threshold
        return is_diverging
    
    @property
    def loss_window(self):
        return list(self._loss_window)
```

---

### 2.3 Passo 3: Testar Novos Módulos

Criar arquivo `tests/test_training_modules.py`:

```python
import pytest
import torch
import numpy as np
from src.config import ExperimentConfig, SystemConfig
from src.models import LaplacianLayer, UNetGenerator2D
from src.training import (
    PIGANLossComputation,
    AdaptiveLambdaPDE,
    GradNormBalancer,
    StagnationDetector,
    DivergenceDetector,
)
from src.physics import PDE_Residual_Computer, compute_physics_metrics


@pytest.fixture
def config():
    return ExperimentConfig()


@pytest.fixture
def laplacian():
    return LaplacianLayer(hx=1.0/31, hy=1.0/31)


def test_pde_residual_computer(laplacian):
    """Verifica cálculo de resíduo PDE."""
    computer = PDE_Residual_Computer(laplacian)
    
    # Campo de zeros → resíduo zero
    field = torch.zeros(1, 1, 32, 32)
    residual = computer.compute_residual(field)
    
    assert residual.shape == (1, 1, 30, 30)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_adaptive_lambda_pde(config):
    """Testa adaptação de λ_PDE."""
    adapter = AdaptiveLambdaPDE(config)
    
    # Inicialmente igual ao config
    assert np.isclose(adapter.current_value, config.lambda_pde, rtol=1e-3)
    
    # Atualizar com loss alto → λ sobe
    lambda1 = adapter.update(loss_pde=10.0, reference_scale=1.0)
    assert lambda1 > config.lambda_pde
    
    # Atualizar com loss baixo → λ desce
    lambda2 = adapter.update(loss_pde=0.1, reference_scale=1.0)
    assert lambda2 < lambda1


def test_stagnation_detector(config):
    """Testa detecção de estagnação."""
    detector = StagnationDetector(config)
    
    # Simulaer melhoria progressiva
    for i in range(5):
        residual = 1.0 - i * 0.15
        is_stagnant, boost = detector.update(residual)
        assert is_stagnant == False
        assert boost == 1.0
    
    # Simular estagnação
    for i in range(60):  # Além da patience
        is_stagnant, boost = detector.update(0.2)  # Residual fixo
        if is_stagnant:
            assert boost >= 1.0


def test_divergence_detector(config):
    """Testa detecção de divergência."""
    detector = DivergenceDetector(config)
    
    # Losses crescentes → divergência
    for i in range(12):
        loss = 1.0 * (1.5 ** i)  # Crescimento exponencial
        is_diverging = detector.update(loss)
    
    # Após janela completa, deve detectar
    assert is_diverging == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Executar:
```bash
python -m pytest tests/test_training_modules.py -v
```

---

### 2.4 Passo 4: Simplificar `trainer.py`

**Em `trainer.py`, substituir `train_step()` por:**

```python
def train_step(
    self,
    base_field: torch.Tensor,
    phi_mask: torch.Tensor,
    ref_field: torch.Tensor,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    coords: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Executa um passo de treinamento (discriminador + gerador).
    
    Retorno:
        Dicionário com métricas.
    """
    metrics = {}
    
    # === SAMPL E LATENT ===
    z = self._sample_latent(self.config.batch_size)
    
    # === FORWARD GENERATOR ===
    pred = self.generator(base_field, phi_mask, z=z, coord_field=coords)
    
    # === UPDATE CRITIC (n_critic times) ===
    for _ in range(self.config.n_critic):
        gp = self._compute_gradient_penalty(
            self.discriminator,
            ref_field,
            pred.detach(),
        )
        loss_d = self.loss_computer.compute_discriminator_loss(
            self.discriminator,
            pred.detach(),
            ref_field,
            gradient_penalty=gp,
        )
        
        self.opt_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.config.max_grad_norm,
        )
        self.opt_d.step()
    
    metrics["loss_d"] = float(loss_d.item())
    
    # === COMPUTE GENERATOR LOSS COMPONENTS ===
    pred = self.generator(base_field, phi_mask, z=z, coord_field=coords)
    
    loss_pde = self.loss_computer.compute_pde_loss(
        pred,
        interior_mask,
        self._pde_train_weight_map,
    )
    loss_adv = self.loss_computer.compute_adversarial_loss(
        self.discriminator,
        pred,
        ref_field,
    )
    loss_bc = self.loss_computer.compute_boundary_loss(
        pred,
        base_field,
        boundary_mask,
    ) if not self.config.hard_constraint_bc else torch.tensor(0.0)
    loss_div = self._compute_diversity_loss() if self.config.latent_dim > 0 else torch.tensor(0.0)
    
    # === ADAPTIVE SCALING ===
    lambda_pde_dyn = self.lambda_pde_adapter.update(
        float(loss_pde),
        self.config.residual_scale_reference,
    )
    
    grad_norm_adv = self._compute_grad_norm([loss_adv])
    grad_norm_pde = self._compute_grad_norm([loss_pde])
    lambda_adv_eff = self.gradnorm_balancer.update(grad_norm_adv, grad_norm_pde)
    
    # === COMPOSE FINAL LOSS ===
    loss_g = self.loss_computer.compose_generator_loss(
        loss_pde,
        loss_adv,
        loss_bc,
        loss_div,
        lambda_pde_dyn,
        lambda_adv_eff * self.config.lambda_adv,
    )
    
    # === UPDATE GENERATOR ===
    self.opt_g.zero_grad()
    loss_g.backward()
    torch.nn.utils.clip_grad_norm_(
        self.generator.parameters(),
        self.config.max_grad_norm,
    )
    self.opt_g.step()
    
    # === ANOMALY DETECTION ===
    is_diverging = self.divergence_detector.update(float(loss_g))
    
    # === LOG METRICS ===
    metrics.update({
        "loss_g": float(loss_g.item()),
        "loss_pde": float(loss_pde.item()),
        "loss_adv": float(loss_adv.item()),
        "loss_bc": float(loss_bc.item()) if loss_bc.ndim > 0 else 0.0,
        "loss_div": float(loss_div.item()) if loss_div.ndim > 0 else 0.0,
        "lambda_pde_dyn": lambda_pde_dyn,
        "lambda_adv_eff": lambda_adv_eff,
        "is_diverging": is_diverging,
    })
    
    return metrics
```

---

### 2.5 Passo 5: Validar P1/P2/P3

```bash
# Executar validações
python validate_skill_corrections.py
python verify_hard_constraint.py

# Executar testes
python -m pytest tests/test_interventions.py -v
python -m pytest tests/test_training_with_interventions.py -v
```

---

## FASE 3: Validação e Limpeza

### 3.1 Testes de Regressão

```bash
# Treinar 50 épocas com configuração padrão
python main.py --epochs 50 --validate-interval 10

# Comparar métricas com run anterior
# Esperado: MAE, RMSE, boundary_error devem ser similares (±2%)
```

### 3.2 Validação de Checkpoints

```bash
# Salvar checkpoint com novo código
python main.py --epochs 10 --save-checkpoints

# Carregar checkpoint com código antigo (fallback)
# Deve funcionar com strict=False
```

### 3.3 Atualizar Documentação

- [ ] Atualizar `README.md` com nova estrutura
- [ ] Adicionar docstrings em novos módulos
- [ ] Criar `MIGRATION_GUIDE.md`
- [ ] Atualizar referências em `pigan-skill/`

---

## Resumo de Mudanças

| Arquivo | Status | Linhas | Mudança |
|---------|--------|-------|--------|
| `config.py` | ✅ Refatorado | -200 | Unificar configs |
| `models.py` | ✅ Sem mudanças | 763 | Compatibilidade |
| `fdm.py` | ✅ Sem mudanças | 120 | Compatibilidade |
| `utils.py` | ✅ Sem mudanças | 224 | Compatibilidade |
| `trainer.py` | ✅ Simplificado | -1300 | 2176 → 800 |
| `pipeline.py` | ✅ Sem mudanças | 2149 | Compatibilidade |
| **Novo:** `physics/pde_residual.py` | ✅ Novo | 70 | Extrair cálculo |
| **Novo:** `physics/domain_metrics.py` | ✅ Novo | 50 | Extrair métricas |
| **Novo:** `training/loss_functions.py` | ✅ Novo | 150 | Extrair loss |
| **Novo:** `training/adaptive_schemes.py` | ✅ Novo | 200 | Extrair adaptação |

**Total:** +3 módulos novos, -1500 linhas de duplicação, +300 linhas de código novo (bem organizado)

---

**Fim do Guia de Implementação**
