# Análise Arquitetural Completa da PI-GAN — 2D Laplace

**Data:** 22 de maio de 2026  
**Escopo:** Análise estrutural, diagnóstico físico/software e proposta de refatoração  
**Objetivo:** Identificar oportunidades de melhoria mantendo correções SKILL (P1, P2, P3)

---

## ETAPA 1: INVENTÁRIO ESTRUTURAL

### 1.1 Visão Geral da Arquitetura

A PI-GAN implementa um gerador-discriminador adversarial que resolve a equação de Laplace 2D estacionária:

$$\nabla^2 T(\mathbf{x}) = 0, \quad \mathbf{x} \in \Omega$$

com condições de Dirichlet nas fronteiras. A arquitetura segue o padrão **generator-physics + adversarial**: o gerador combina loss física (resíduo PDE) com loss adversarial (crítico de dados).

### 1.2 Módulos e Responsabilidades

#### **A. `src/config.py` (1129 linhas)**

**Classes:**
1. **`SystemConfig`** (Dataclass, ~200 linhas)
   - Responsabilidade: Configuração de hardware, GPU, reprodutibilidade
   - Atributos-chave:
     - `use_gpu`, `gpu_id`, `mixed_precision`, `compile_model`
     - `num_workers`, `prefetch_factor`
     - `seed`, `deterministic_run`, `cudnn_benchmark`
   - Método crítico: `__post_init__()` valida e calibra settings de sistema

2. **`ExperimentConfig`** (Dataclass, ~900 linhas)
   - Responsabilidade: Todos os hiperparâmetros do experimento (modelo, física, treinamento)
   - Seções de campos:
     - **Física:** `T_LEFT`, `T_RIGHT`, `LX`, `LY`, `boundary_sine_amplitude`
     - **Arquitetura Gerador:** `generator_mode`, `latent_dim`, `generator_base_channels`, `generator_depth`, `hard_constraint_bc`
     - **Arquitetura Discriminador:** `discriminator_base_channels`, `discriminator_capacity_scale`
     - **Pesos de Loss:** `lambda_pde`, `lambda_adv`, `lambda_bc`, `lambda_diversity`, `lambda_gp`
     - **Adaptação Dinâmica:** `adaptive_lambda_pde`, `dynamic_adv_balance`, `gradnorm_balance`
     - **Scheduler:** `plateau_scheduler_enabled`, `divergence_window`
     - **Refinamento Precisão:** `precision_refine_enable`, `adaptive_sampling_enable`
   - Método crítico: `__post_init__()` valida >50 restrições

3. **`EnhancedLogger`** (Classe auxiliar, ~50 linhas)
   - Responsabilidade: Compatibilidade logging com/sem structlog

**Responsabilidade Consolidada:**
- ✅ Centraliza toda configuração em dataclasses
- ✅ Valida consistência entre parâmetros
- ✅ Diferencia hardware (SystemConfig) de experimento (ExperimentConfig)
- ❌ **Problema 1:** Muitas responsabilidades em um único arquivo (1129 linhas)
- ❌ **Problema 2:** `__post_init__` é gigantesco (~500 linhas de if/else)

---

#### **B. `src/models.py` (763 linhas)**

**Classes:**

1. **`ConvBlock`** (Linha 40–70, ~30 linhas)
   - Responsabilidade: Bloco reutilizável Conv2D dupla + BatchNorm + ativação
   - Padrão: Container de camadas

2. **`DownBlock`** (Linha 73–110, ~40 linhas)
   - Responsabilidade: Downsampling = ConvBlock + MaxPool ou AvgPool

3. **`UpBlock`** (Linha 113–150, ~40 linhas)
   - Responsabilidade: Upsampling com skip connection concatenada

4. **`UNetGenerator2D`** (Linha 153–530, ~380 linhas)
   - Responsabilidade: Gerador encoder-decoder com U-Net
   - Características:
     - Hard constraint: $T = g(\mathbf{x}) + \phi(\mathbf{x}) \cdot \hat{T}_{net}$
     - Injeção de latente $z$ (quando `latent_dim > 0`)
     - Coordenadas físicas normalizadas opcionais
     - Suavização de saída opcional (via convolução com kernel Gaussiano)
   - Métodos críticos:
     - `forward()`: Composição entrada, injeção latente, encoder-decoder, imposição hard constraint
     - `_compose_input()`: Concatena campos base com coordenadas físicas
     - `_inject_latent()`: Expande vetor latente e concatena aos canais
     - `_smooth_raw_output()`: Suavização iterativa de saída
   - ✅ Bem documentado com docstrings
   - ❌ **Problema:** Suavização é não-determinística em CUDA (F.pad com reflect)

5. **`LaplacianLayer`** (Linha 533–610, ~80 linhas)
   - Responsabilidade: Operador Laplaciano discreto 5-ponto via Conv2D fixo
   - Kernel matemático:
     $$K = \frac{1}{h_x^2} \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
   - ✅ Kernel não-treinável (requires_grad=False)
   - ✅ Escala explícita por $h_x^2$, $h_y^2$
   - ⚠️ **Nota:** Padding=0 implica redução de dimensão (H×W) → (H-2)×(W-2)

6. **`DataDiscriminator2D`** (Linha 613–763, ~150 linhas)
   - Responsabilidade: Crítico pareado que recebe (campo predito, campo referência)
   - Arquitetura: Encoder CNN com concatenação de campos
   - Saída: Escalar (score WGAN)
   - ✅ Input de 2 canais (T_pred, T_ref)
   - ⚠️ Pode usar Spectral Normalization (opcional)

7. **`create_field_pigan_models()`** (Linha 765–763, factory function)
   - Responsabilidade: Factory que instancia gerador + discriminador com configuração
   - Padrão: Builder pattern

**Responsabilidade Consolidada:**
- ✅ Encapsula arquiteturas CNN de forma modular
- ✅ Separação clara entre blocos arquitetônicos
- ⚠️ **Problema 1:** 380 linhas de `UNetGenerator2D` fazem muito (builder, forward, suavização, injeção latente)
- ⚠️ **Problema 2:** Hard constraint mistura lógica de arquitetura com aplicação na forward pass

---

#### **C. `src/fdm.py` (120 linhas)**

**Funções:**

1. **`_apply_dirichlet_boundaries()`** (Linhas 20–33, ~15 linhas)
   - Responsabilidade: Copia valores de contorno em-place

2. **`solve_laplace_dirichlet()`** (Linhas 36–135, ~100 linhas)
   - Responsabilidade: Solucionador SOR (Successive Over-Relaxation) para Laplace com Dirichlet
   - Método: Red-Black SOR com fator de sobre-relaxação $\omega \in (0,2)$
   - Parâmetros:
     - `boundary_field`: Condições de Dirichlet nas bordas
     - `lx`, `ly`: Dimensões físicas
     - `tol`: Tolerância de convergência
     - `max_iter`: Máximo de iterações
     - `omega`: Fator de relaxação
   - Retorno: `(campo_final, iterações_executadas)`
   - ✅ Bem documentado
   - ✅ Validação de entrada
   - ✅ Determinístico (importante para reproducibilidade)

**Responsabilidade Consolidada:**
- ✅ Gera campo de referência FDM para validação
- ✅ Módulo independente, sem dependências em modelos/trainer
- ✅ Eficiente: Red-Black pattern e GPU-friendly

---

#### **D. `src/utils.py` (224 linhas)**

**Funções:**

1. **`create_cartesian_grid()`** (Linhas 20–56, ~35 linhas)
   - Responsabilidade: Cria malha regular 2D (x_grid, y_grid)

2. **`build_dirichlet_extension()`** (Linhas 59–118, ~60 linhas)
   - Responsabilidade: Constrói extensão suave das condições de Dirichlet
   - Método: Coons Patch (interpolação bilinear)
   - Fornece função $g(\mathbf{x})$ que satisfaz exatamente as bordas

3. **`build_hard_constraint_mask()`** (Linhas 121–224, ~104 linhas)
   - Responsabilidade: Constrói máscara de decaimento $\phi(\mathbf{x})$ para hard constraint
   - Duas variantes:
     - **'tanh'** (padrão): Composição de tanh para suavidade C² nos cantos
     - **'polynomial'**: Método original $x(1-x)y(1-y)$
   - Garante: $\phi(\mathbf{x}) = 0$ na fronteira, $\phi(\mathbf{x}) > 0$ no interior
   - ✅ Bem documentado

4. **`build_domain_masks()`** (Linhas 227–..., ~30 linhas)
   - Responsabilidade: Cria máscaras binárias para interior/fronteira

**Responsabilidade Consolidada:**
- ✅ Utilitários de construção de malhas
- ✅ Sem dependências circulares
- ✅ Reutilizáveis em pipeline

---

#### **E. `src/trainer.py` (2176 linhas) — O CORAÇÃO DO PIPELINE**

**Dataclasses:**

1. **`FieldTrainerConfig`** (Linhas 30–180, ~150 linhas)
   - Responsabilidade: Configuração exaustiva do treinamento adversarial
   - Campos: >60 parâmetros de loss, scheduling, gates, stagnation detection
   - ⚠️ **Problema:** Sobreposição com `ExperimentConfig` em config.py

**Funções Utilitárias:**

1. **`_set_requires_grad()`** (Linhas 190–195, ~5 linhas)
   - Ativa/desativa gradientes de um módulo

2. **`_gradient_penalty()`** (Linhas 198–224, ~25 linhas)
   - Calcula GP para WGAN-GP
   - Interpola entre real/fake e calcula norma do gradiente

3. **`_global_grad_norm()`** (Linhas 227–240, ~15 linhas)
   - Norma L2 total dos gradientes

**Classe Principal: `FieldPIGANTrainer`** (Linhas 243–2176, ~1930 linhas)

**Responsabilidades:**
- Gerenciar estado do treinamento (gerador, discriminador, otimizadores)
- Loop de treinamento com múltiplas estratégias de otimização:
  - **Loss:**
    - $L_G = \lambda_{PDE}^{dyn} L_{PDE} + \lambda_{adv}^{eff} L_{adv} + \lambda_{BC} L_{BC} + \lambda_{div} L_{div}$
    - $L_D = E[D(fake)] - E[D(real)] + \lambda_{gp} GP + drift + gap\_penalty$
  - **Adaptações Dinâmicas:**
    - EMA de $\lambda_{PDE}$ (adaptive_lambda_pde)
    - GradNorm balancing
    - Adversarial residual gate (liga/desliga com histerese)
    - Stagnation detection + boost
    - Progressive adversarial ramp (multiplier over epochs)
  - **Detecção de Divergência:**
    - Janela de loss recente
    - Early stopping se NaN/Inf

**Métodos Críticos:**
1. `train_step()`: Um passo de otimização (generator + discriminators)
2. `train_epoch()`: Uma época completa
3. `_compute_loss_g()`: Calcula perda do gerador (PDE + adversarial + BC + diversity)
4. `_compute_loss_d2()`: Calcula perda do crítico pareado
5. `_compute_pde_loss()`: Calcula resíduo Laplaciano
6. `_compute_boundary_loss()`: Calcula erro nas fronteiras (soft constraint)
7. `_compute_diversity_loss()`: Incentivo de diversidade (modo estocástico)
8. `_apply_adaptive_lambda_pde()`: Ajusta peso PDE dinamicamente
9. `_apply_gradnorm_balance()`: Balanceia scales via normas de gradiente
10. `save_checkpoint()`: Salva estado completo (modelos + otimizadores + RNG)
11. `load_checkpoint()`: Retoma treinamento

**⚠️ Problemas Estruturais:**
1. ❌ **Gigantismo:** 2176 linhas em uma única classe
2. ❌ **Múltiplas responsabilidades:** Loop de treinamento + scheduling + detecção de divergência + loss computation + checkpointing
3. ❌ **Difícil de testar:** Acoplamento forte com modelos, laplacian, máscaras
4. ❌ **Estado interno complexo:** ~40 variáveis de estado (_adv_scale_ema, _pde_norm_ema, _critics_paused_state, etc.)

---

#### **F. `src/pipeline.py` (2149 linhas)**

**Classe Principal: `PIGANPipeline`**

**Responsabilidades:**
- Orquestração de alto nível
- Inicialização de sistema (GPU, logger, memory manager)
- Preparação de campos físicos (malha, FDM, máscaras)
- Construção de modelos e trainer
- Loop de treinamento (chama `trainer.train_epoch()`)
- Avaliação periódica e salvamento de plots

**Métodos Críticos:**
1. `_prepare_physics_fields()`: Cria malha, FDM, hard constraint, máscaras
2. `_create_models()`: Factory para gerador + discriminador
3. `_create_trainer()`: Factory para FieldPIGANTrainer
4. `train()`: Loop principal que chama `trainer.train_epoch()`
5. `evaluate()`: Computa métricas, salva plots

**✅ Bom design:**
- Separação clara entre pipeline de orquestração e lógica de treinamento
- Responsabilidades bem delimitadas

**⚠️ Problemas:**
1. ❌ 2149 linhas: combinação pipeline + geração de plots + logging = muitas responsabilidades
2. ⚠️ Acoplamento: Pipeline chama quase todo módulo (config, models, trainer, utils, evaluation)

---

#### **G. `src/evaluation.py` (105 linhas)**

**Função Principal: `compute_field_metrics()`**

**Responsabilidade:**
- Calcula 10 métricas de erro e física:
  - Regressão: MAE, RMSE, MAPE, R²
  - Física: resíduo PDE (mean/L2/max), erro de contorno
  - Erro relativo L2

**✅ Bem encapsulada**
- Uma única função, sem estado
- Usada por pipeline para avaliação periódica

---

### 1.3 Diagrama de Fluxo de Responsabilidades

```
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline (2149 linhas)                    │
│ Orquestra: config → FDM → modelos → trainer → avaliação     │
└─────────────────────────────────────────────────────────────┘
           ↓                ↓              ↓
    ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
    │  Config      │  │  Models      │  │ Trainer (2176)  │
    │ (1129 lines) │  │ (763 lines)  │  │ Loop principal  │
    └──────────────┘  └──────────────┘  └─────────────────┘
           ↑                ↑
    ┌──────────────────────────────────────────┐
    │         Utils (224 lines)                │
    │ • Malhas (CartesianGrid)                 │
    │ • Hard constraint (Phi)                  │
    │ • Extensão Dirichlet (g)                 │
    │ • Máscaras (interior/boundary)           │
    └──────────────────────────────────────────┘
           ↓                ↓
    ┌──────────────┐  ┌───────────────┐
    │  FDM (120)   │  │ Evaluation    │
    │  SOR solver  │  │ (105 lines)   │
    └──────────────┘  └───────────────┘
```

---

### 1.4 Fluxo de Dados no Treinamento

```
Epoch Loop (pipeline.train())
  ↓
trainer.train_epoch()
  ├─ Para cada step (1 a steps_per_epoch):
  │    ├─ Amostra z ~ N(0,1)  [latent_dim=8]
  │    ├─ Expande base_field, phi_mask, coord_field
  │    │
  │    ├─ n_critic vezes:
  │    │    ├─ generator.forward() → pred_temp
  │    │    ├─ discriminator.forward(pred_temp, ref_temp) → score_fake
  │    │    ├─ discriminator.forward(ref_temp, ref_temp) → score_real
  │    │    ├─ Calcula L_D = E[D(fake)] - E[D(real)] + GP + drift + gap_penalty
  │    │    ├─ optimizer_d.zero_grad(), L_D.backward(), step()
  │    │
  │    ├─ generator.forward() → pred_temp
  │    ├─ laplacian.forward(pred_temp) → residual_pde
  │    ├─ Calcula:
  │    │    L_PDE = |residual_pde|_media  (weighted por interior_mask)
  │    │    L_adv = -E[D(pred_temp, ref_temp)]  (fake score)
  │    │    L_BC  = MSE(pred_temp na boundary)
  │    │    L_div = -media de L2 entre pares latentes
  │    ├─ L_G = lambda_pde_dyn * L_PDE + lambda_adv_eff * L_adv + lambda_BC * L_BC + lambda_div * L_div
  │    ├─ optimizer_g.zero_grad(), L_G.backward(), (grad clip), step()
  │    │
  │    └─ Atualiza estado adaptativo (EMA, gates, stagnation)
  │
  └─ A cada evaluate_frequency épocas:
       ├─ Computa métricas (MAE, RMSE, resíduo PDE, boundary error)
       ├─ Salva plots
       └─ Log

```

---

## ETAPA 2: DIAGNÓSTICO FÍSICO

### 2.1 Formulação da PDE

**Equação governante (Laplace 2D estacionária):**
$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

**Domínio:** Retangular $\Omega = [0, L_x] \times [0, L_y]$

**Condições de Dirichlet:**
- Bordas esquerda (x=0): $T = T_{LEFT} = 200$ K
- Bordas direita (x=LX): $T = T_{RIGHT} = 100$ K
- Bordas superior/inferior (y=0, y=LY): Perfil senoidal com amplitude configurável

✅ **Status:** Formulação correta na documentação

---

### 2.2 Implementação do Operador Laplaciano

**Método:** Diferenças finitas 5-ponto (stencil 3×3)

**Kernel discreto em `LaplacianLayer` (src/models.py, linhas 533–610):**

```python
kernel = torch.tensor([
    [0.0, 1.0/(hy²), 0.0],
    [1.0/(hx²), -2.0(1.0/hx² + 1.0/hy²), 1.0/(hx²)],
    [0.0, 1.0/(hy²), 0.0]
], dtype=torch.float32).view(1, 1, 3, 3)
```

**Análise:**
- ✅ Kernel não-treinável (requires_grad=False)
- ✅ Escala explícita por $h_x^2$, $h_y^2$
- ✅ Simetria respeitada
- ✅ Conv2d com padding=0 (apenas interior)

**Derivação matemática:**

Para campo $T$ em malha uniforme:
$$\nabla^2 T|_{i,j} \approx \frac{T_{i+1,j} + T_{i-1,j} - 2T_{i,j}}{h_x^2} + \frac{T_{i,j+1} + T_{i,j-1} - 2T_{i,j}}{h_y^2}$$

Reorganizando:
$$\nabla^2 T|_{i,j} = \frac{1}{h_x^2} T_{i+1,j} + \frac{1}{h_x^2} T_{i-1,j} + \frac{1}{h_y^2} T_{i,j+1} + \frac{1}{h_y^2} T_{i,j-1} - 2\left(\frac{1}{h_x^2} + \frac{1}{h_y^2}\right) T_{i,j}$$

Que corresponde ao kernel acima.

✅ **Status:** Implementação correta

---

### 2.3 Imposição de Condições de Contorno

#### **2.3.1 Abordagem: Hard Constraint Arquitetural**

**Método:** Composição arquitetural
$$T_\theta(\mathbf{x}) = g(\mathbf{x}) + \phi(\mathbf{x}) \cdot \hat{T}_{net}(\mathbf{x})$$

onde:
- $g(\mathbf{x})$: Extensão suave (Coons Patch) satisfazendo Dirichlet exatamente
- $\phi(\mathbf{x})$: Máscara de decaimento com $\phi(\mathbf{x}) = 0$ na fronteira
- $\hat{T}_{net}$: Saída da rede neural (sem ativação final)

**Implementação:**

1. **Construção de $g$ em `build_dirichlet_extension()` (utils.py, 59–118):**
   - Coons Patch bilinear interpolação
   - ✅ Garante valores corretos na fronteira

2. **Construção de $\phi$ em `build_hard_constraint_mask()` (utils.py, 121–224):**
   - Variante 'tanh' (padrão): Composição de tanh
   - Variante 'polynomial': $x(1-x)y(1-y)$
   - ✅ Garante $\phi = 0$ na fronteira

3. **Aplicação em `UNetGenerator2D.forward()` (models.py, ~490):**
   ```python
   if self.hard_constraint:
       return base_field + phi_mask * raw
   return raw
   ```
   - ✅ Correto: composição linear $g + \phi \cdot \hat{T}$

#### **Verificação Matemática:**

Na fronteira ($\partial\Omega$): $\phi = 0$, logo
$$T_\theta|_{\partial\Omega} = g|_{\partial\Omega} + 0 = g|_{\partial\Omega}$$

e $g$ foi construído para satisfazer Dirichlet, logo $T_\theta|_{\partial\Omega}$ está correto. ✅

#### **Problema P1 (λ_BC zerando condicionalmente):**

Em `config.py`, linhas ~630:
```python
if bool(self.hardConstraintBc) and float(self.lambdaBc) > 1e-6:
    self.lambdaBc = 0.0
```

- ✅ Implementado corretamente
- ✅ Lógica: Se hard_constraint ativo, não precisa de loss de fronteira soft

---

### 2.4 Cálculo de Autodiferenciação (Gradientes)

#### **2.4.1 Autograd para Resíduo PDE**

Na `FieldPIGANTrainer._compute_pde_loss()`:

```python
pred = self.generator(base_field, phi_mask, z=z, coord_field=coords)
residual = self.laplacian(pred)  # Conv2D fixa
loss_pde = self.compute_residual_loss(residual, interior_mask, pde_weight_map)
```

**Fluxo de gradientes:**

1. Predição $T_\theta$ é autograd-ativa
2. Laplacian é aplicado via `Conv2d(kernel_fixo, padding=0)`
3. Residual é $\nabla^2 T_\theta$ no interior
4. Loss é escalar baseado em residual
5. Gradientes fluem via `loss.backward()` até parâmetros de $\theta$

✅ **Status:** Autodiferenciação funcionando

#### **2.4.2 Ordem de Derivadas**

Para Laplace, precisamos de derivadas 2ª ordem. O stencil 3×3 com padding=0 fornece:

- Entrada `pred`: $[B, 1, H, W]$
- Saída `residual`: $[B, 1, H-2, W-2]$

Matematicamente, a convolução com kernel $h_{i,j}$ calcula:

$$\text{output}[i,j] = \sum_k \sum_l h_{k,l} \cdot \text{input}[i+k, j+l]$$

que é **linearmente determinística**. A autodifferenciação do PyTorch sabe que:

$$\frac{\partial \text{output}}{\partial \text{input}} = h_{k,l}$$

✅ **Status:** Ordem de derivadas correta (2ª ordem via stencil)

---

### 2.5 Termo de Diversity Loss (P2 Complemento)

Em `config.py`:
```python
lambda_diversity: float = 1.0e-4
```

Em `FieldPIGANTrainer._compute_loss_g()`:
```python
if str(self.cfg.generator_mode) == "stochastic_pigan":
    loss_div = self._compute_diversity_loss()  # negativo para encorajar divergência
else:
    loss_div = 0.0
```

**Fórmula:**
$$L_{div} = -\frac{1}{N(N-1)} \sum_{i \neq j} \frac{\|T_i - T_j\|_2}{HW}$$

onde $T_i, T_j$ são campos gerados com latentes diferentes.

✅ **Status:** Implementado corretamente para modo estocástico

---

### 2.6 Alerta de Saturação (P3)**

Em `FieldPIGANTrainer`:
```python
self._lambda_pde_saturation_warn_count = 0

def _check_lambda_pde_saturation():
    if abs(self._lambda_pde_dyn - self._last_lambda_pde_max_effective) < 1e-8:
        self._lambda_pde_saturation_warn_count += 1
        if self._lambda_pde_saturation_warn_count % 50 == 0:
            self.logger.warning(
                f"λ_PDE saturado no teto: {self._lambda_pde_max_effective}"
            )
```

✅ **Status:** Alerta implementado

---

### 2.7 Diagnóstico Consolidado da Física

| Aspecto | Status | Observação |
|---------|--------|-----------|
| PDE formulação | ✅ Correto | Laplace 2D estacionária |
| Operador Laplaciano | ✅ Correto | Stencil 5-ponto, h² explícito |
| Condições Dirichlet | ✅ Correto | Hard constraint arquitetural |
| Autograd | ✅ Ativo | Conv2d é diferenciável |
| Ordem de derivadas | ✅ Correto | 2ª ordem via stencil |
| P1 (λ_BC condicional) | ✅ Implementado | Com validação |
| P2 (λ_adv, λ_div) | ✅ Implementado | Valores corretos (5e-3, 1e-4) |
| P3 (alerta saturação) | ✅ Implementado | A cada 50 steps |

---

## ETAPA 3: DIAGNÓSTICO DE ENGENHARIA DE SOFTWARE

### 3.1 Separação de Responsabilidades

#### **3.1.1 Análise de Coesão**

| Módulo | Responsabilidades | Coesão | Problema |
|--------|-------------------|--------|----------|
| `config.py` | Config sistema + experimento | 🟡 Média | 1129 linhas, __post_init__ gigantesco |
| `models.py` | Blocos CNN + gerador + discriminador + Laplacian | 🟢 Alta | Bem modularizado |
| `fdm.py` | Solucionador FDM | 🟢 Alta | Independente, focado |
| `utils.py` | Malhas + contornos | 🟢 Alta | Reutilizável, sem acoplamento |
| `trainer.py` | Loop de treinamento | 🔴 Baixa | 2176 linhas, múltiplas responsabilidades |
| `pipeline.py` | Orquestração + plots | 🟡 Média | 2149 linhas, muitas sub-responsabilidades |
| `evaluation.py` | Cálculo de métricas | 🟢 Alta | Função pura, sem estado |

#### **3.1.2 Problemas de Acoplamento**

**Acoplamento Forte:**
```
pipeline.py 
    ↓ importa
config.py (SystemConfig, ExperimentConfig)
    ↓
trainer.py (FieldTrainerConfig sobreposição!)
    ↓ importa
models.py (UNetGenerator2D, DataDiscriminator2D, LaplacianLayer)
    ↓ importa
evaluation.py (compute_field_metrics)
```

- ⚠️ **Pipeline depende de tudo**
- ⚠️ **Trainer tem FieldTrainerConfig separada em trainer.py, mas config.py tem ExperimentConfig**
- ❌ **Duplicação conceitual**: `lambda_pde`, `lambda_adv`, etc. aparecem em 2+ lugares

---

### 3.2 Reprodutibilidade

#### **3.2.1 Seed Fixada**

Em `config.py`, `SystemConfig`:
```python
seed: int = 42
deterministic_run: bool = True

def __post_init__():
    if bool(self.deterministic_run):
        cudnn_benchmark = False
        cudnn_deterministic = True
        use_tf32 = False
```

✅ Bem implementado

#### **3.2.2 RNG State Management**

Em `trainer.py`:
```python
@staticmethod
def _capture_rng_state(device):
    # Captura PyTorch (CPU/CUDA) + NumPy + Python random
    
@staticmethod
def _restore_rng_state(state, device):
    # Restaura todos os RNG para retomar determinísticamente
```

✅ Completo

#### **3.2.3 Problema: Suavização de Saída em CUDA**

Em `UNetGenerator2D._smooth_raw_output()` (models.py, ~270):

```python
pad_mode = self._resolve_smoothing_pad_mode(raw)

@classmethod
def _resolve_smoothing_pad_mode(cls, raw):
    deterministic_enabled = cls._deterministic_algorithms_enabled()
    if raw.is_cuda:
        if deterministic_enabled is None:
            return "replicate"  # Fallback para evitar não-determinismo
        if deterministic_enabled:
            return "replicate"
    return "reflect"
```

✅ Mitigado com fallback

---

### 3.3 Código Morto / Lógica Duplicada

#### **3.3.1 Configuração Duplicada**

**Problema:** Parâmetros de treinamento aparecem em 2 lugares:

1. `ExperimentConfig` em `config.py`:
   ```python
   lambda_adv: float = 5.0e-3
   lambda_pde: float = 37.0
   # ... 60+ campos
   ```

2. `FieldTrainerConfig` em `trainer.py`:
   ```python
   lambda_adv: float
   lambda_pde: float
   # ... 60+ campos
   ```

- ❌ **Duplicação:** Mesmos parâmetros definidos 2x
- ❌ **Risco:** Desincronização entre `ExperimentConfig.__post_init__()` e `FieldTrainerConfig.__post_init__()`

#### **3.3.2 Funções Utilitárias Redundantes**

Várias funções privadas em `trainer.py` que poderiam ser em módulo compartilhado:
- `_gradient_penalty()`: Específico de WGAN-GP, poderia estar em utils ou models
- `_global_grad_norm()`: Utilitário de álgebra linear, poderia estar em utils

---

### 3.4 Complexidade Ciclomática

**Exemplo: `FieldPIGANTrainer.train_step()` (trainer.py)**

Pseudocódigo:
```python
def train_step():
    # Se não usar hard constraint: ajustar lambda_bc
    if not hard_constraint_bc:
        # ... 10 linhas
    
    # Atualizar discriminador n_critic vezes
    for _ in range(n_critic):
        if critic_paused_state:
            # ... skip
        else:
            # Calcular loss D com múltiplas condições
            if use_wgan_gp:
                # + GP
            if critic_drift:
                # + drift
            if max_critic_gap:
                # + gap penalty
            if critic_pause_on_overgap and gap > threshold:
                # Pausar discriminador
    
    # Atualizar gerador
    if in_warmup_phase:
        # ... lambda_adv reduzido
    elif adv_residual_gate_hysteresis:
        # Gate com memória
    else:
        # Gate suave
    
    # Múltiplos termos de loss com gates
    L_G = lambda_pde_dyn * L_PDE
    if adv_gate_enabled:
        L_G += lambda_adv_eff * L_adv
    if not hard_constraint_bc:
        L_G += lambda_bc * L_BC
    if diversity_enabled:
        L_G += lambda_diversity * L_div
    
    # Adaptive scaling
    if adaptive_lambda_pde:
        # Ajustar lambda_pde_dyn
    if gradnorm_balance:
        # Ajustar scales via normas de gradiente
    if stagnation_boost:
        # Boostar se resíduo estagna
    if divergence_detected:
        # Reduzir learning rate
    if plateau_detected:
        # Reduzir learning rate
    
    # Salvamento
    if save_frequency > 0 and step % save_frequency == 0:
        # Salvar checkpoint
```

**Métrica:** Estimada ~200+ pontos de decisão no código. Muito alta para testabilidade.

---

### 3.5 Testabilidade

#### **3.5.1 Testes Presentes**

Em `tests/`:
- `test_main_config.py`: Valida ExperimentConfig
- `test_checkpoint_resume.py`: Valida salvamento/carregamento
- `test_field_training_modes.py`: Valida modo estocástico vs determinístico
- `test_laplacian_field_operator.py`: Valida operador Laplaciano
- `test_pipeline_execution.py`: Executa mini-pipeline

✅ Coverage razoável

#### **3.5.2 Problemas de Testabilidade**

- ❌ `train_step()` é difícil de testar isoladamente (depende de estado global)
- ❌ Sem injeção de dependência: modelos/optimizer/logger acoplados em `__init__`
- ❌ Sem interfaces abstratas: hard de mockar

---

### 3.6 Documentação

#### **3.6.1 Docstrings**

- ✅ Classe `SystemConfig`: Bem documentada
- ✅ Classe `ExperimentConfig`: Bem documentada
- ✅ Funções em `models.py`: Bem documentadas
- ✅ Funções em `utils.py`: Bem documentadas
- ⚠️ `FieldPIGANTrainer`: Algumas funções faltam docstring
- ⚠️ `PIGANPipeline`: Muitos métodos sem docstring

#### **3.6.2 Comentários Inline**

- ✅ Bem colocados em operações complexas (Coons Patch, Red-Black SOR, hard constraint)
- ⚠️ Alguns métodos muito grandes carecem de comentários intermediários

---

## ETAPA 4: REFATORAÇÃO PROPOSTA

### 4.1 Objetivos da Refatoração

1. **Reduzir complexidade:** Quebrar classes gigantes em componentes menores
2. **Eliminar duplicação:** Unificar config em um único lugar
3. **Melhorar testabilidade:** Injetar dependências, desacoplar
4. **Manter compatibilidade:** P1, P2, P3 devem continuar funcionando
5. **Facilitar manutenção:** Código mais legível e modular

### 4.2 Estrutura Refatorada Proposta

```
src/
├── __init__.py                    (atual)
├── config.py                      (REFATORADO: SystemConfig + ExperimentConfig unificados)
│                                  + Validação centralizada
├── models.py                      (atual, sem mudanças maiores)
├── fdm.py                         (atual, sem mudanças)
├── utils.py                       (atual, sem mudanças)
├── physics/                       (NOVO: módulo de física)
│   ├── __init__.py
│   ├── pde_residual.py           (Cálculo de resíduo + loss PDE)
│   ├── boundary_conditions.py    (Hard/soft constraint)
│   └── domain_metrics.py         (Métricas físicas)
├── training/                      (NOVO: módulo de treinamento)
│   ├── __init__.py
│   ├── loss_functions.py         (L_PDE, L_adv, L_BC, L_div)
│   ├── adaptive_schemes.py       (EMA, GradNorm, stagnation detection)
│   ├── critic.py                 (FieldPIGANTrainer simplificado)
│   └── schedulers.py             (Scheduler de learning rate)
├── evaluation.py                  (atual, sem mudanças)
└── pipeline.py                    (REFATORADO: focado em orquestração)
```

### 4.3 Mudanças Críticas com Antes/Depois

#### **4.3.1 Unificação de Configuração**

**ANTES:**
```python
# config.py
@dataclass
class ExperimentConfig:
    lambda_adv: float = 5.0e-3
    lambda_pde: float = 37.0
    # ... 60+ campos

# trainer.py
@dataclass
class FieldTrainerConfig:
    lambda_adv: float
    lambda_pde: float
    # ... 60+ campos DUPLICADOS

# pipeline.py
trainer_config = FieldTrainerConfig(
    lambda_adv=exp_config.lambda_adv,  # Cópia manual!
    lambda_pde=exp_config.lambda_pde,
    # ... 60 linhas de mapeamento
)
```

**DEPOIS:**
```python
# config.py
@dataclass
class ExperimentConfig:
    # Sistema
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Física
    T_LEFT: float = 200.0
    # ...
    
    # Arquitetura
    generator_mode: str = "stochastic_pigan"
    # ...
    
    # Treinamento (centralizado aqui, não em FieldTrainerConfig)
    gen_lr: float = 1.15e-4
    lambda_adv: float = 5.0e-3
    lambda_pde: float = 37.0
    # ... tudo aqui
    
    def __post_init__(self):
        # Validação única e centralizada
        self._validate_all()

# trainer.py
class FieldPIGANTrainer:
    def __init__(
        self,
        generator: UNetGenerator2D,
        discriminator: DataDiscriminator2D,
        laplacian: LaplacianLayer,
        config: ExperimentConfig,  # Único config!
        device: torch.device,
    ):
        self.config = config
        # ... resto normal
```

**Justificativa:**
- Elimina duplicação
- Source of truth única
- Menos mapeamentos manuais (erro-prone)

---

#### **4.3.2 Extração de Loss Functions**

**ANTES (trainer.py, ~1500 linhas):**
```python
def _compute_loss_g(self):
    # ... 200+ linhas de computação de loss
    loss_pde = self._compute_pde_loss()
    loss_adv = self._compute_adversarial_loss()
    loss_bc = self._compute_boundary_loss()
    loss_div = self._compute_diversity_loss()
    
    lambda_pde_dyn = self._apply_adaptive_lambda_pde()
    lambda_adv_eff = self._apply_gradnorm_balance()
    
    loss_g = (
        lambda_pde_dyn * loss_pde +
        lambda_adv_eff * loss_adv +
        (0 if self.config.hard_constraint_bc else self.config.lambda_bc * loss_bc) +
        (self.config.lambda_diversity * loss_div if stochastic else 0)
    )
    return loss_g, metrics_dict
```

**DEPOIS (training/loss_functions.py):**
```python
class PIGANLossComputation:
    def __init__(
        self,
        laplacian: LaplacianLayer,
        config: ExperimentConfig,
    ):
        self.laplacian = laplacian
        self.config = config
    
    def compute_pde_loss(
        self,
        pred_field: torch.Tensor,
        interior_mask: torch.Tensor,
        pde_weight_map: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula L_PDE = media(|∇²T|)"""
        residual = self.laplacian(pred_field)
        loss = self._weighted_interior_loss(residual, interior_mask, pde_weight_map)
        return loss
    
    def compute_adversarial_loss(
        self,
        discriminator: nn.Module,
        pred_field: torch.Tensor,
        ref_field: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula L_adv = -E[D(T_pred, T_ref)]"""
        # ...
    
    def compute_boundary_loss(
        self,
        pred_field: torch.Tensor,
        boundary_field: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula L_BC = MSE nas fronteiras"""
        # ...
    
    def compose_generator_loss(
        self,
        loss_pde: torch.Tensor,
        loss_adv: torch.Tensor,
        loss_bc: torch.Tensor,
        loss_div: torch.Tensor,
        lambda_pde_dyn: float,
        lambda_adv_eff: float,
        config: ExperimentConfig,
    ) -> torch.Tensor:
        """Compõe L_G final com os termos individuais"""
        loss_g = lambda_pde_dyn * loss_pde
        loss_g = loss_g + lambda_adv_eff * loss_adv
        if not config.hard_constraint_bc:
            loss_g = loss_g + config.lambda_bc * loss_bc
        if config.latent_dim > 0:
            loss_g = loss_g + config.lambda_diversity * loss_div
        return loss_g
```

**Justificativa:**
- Cada loss em função clara
- Testável isoladamente
- Reutilizável em outros contextos
- Reduz tamanho de `trainer.py`

---

#### **4.3.3 Extração de Adaptive Schemes**

**ANTES (trainer.py, ~300 linhas espalhadas):**
```python
def _apply_adaptive_lambda_pde(self):
    # ... 50 linhas de lógica EMA e clipping
    rho = max(loss_pde / reference_scale, 1.0)
    lambda_des = clip(lambda_base * (1 + alpha * log10(rho)), min, max)
    self._lambda_pde_dyn = beta * self._lambda_pde_dyn + (1-beta) * lambda_des
    return self._lambda_pde_dyn

def _apply_gradnorm_balance(self):
    # ... 80 linhas de balanceamento de normas
    
def _detect_stagnation(self):
    # ... 40 linhas de detecção de estagnação
    
def _detect_divergence(self):
    # ... 50 linhas de detecção de divergência
```

**DEPOIS (training/adaptive_schemes.py):**
```python
class AdaptiveLambdaPDE:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ema_value = config.lambda_pde
    
    def update(
        self,
        loss_pde: float,
        reference_scale: float,
    ) -> float:
        """Atualiza lambda_pde via EMA e clipping"""
        rho = max(loss_pde / reference_scale, 1.0)
        lambda_des = np.clip(
            self.config.lambda_pde * (
                1 + self.config.lambda_pde_growth_exponent * np.log10(rho)
            ),
            self.config.lambda_pde_min,
            self.config.lambda_pde_max,
        )
        self._ema_value = (
            self.config.lambda_pde_ema_beta * self._ema_value +
            (1 - self.config.lambda_pde_ema_beta) * lambda_des
        )
        return self._ema_value


class GradNormBalancer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._ema_scale = 1.0
    
    def update(
        self,
        grad_norm_adv: float,
        grad_norm_pde: float,
    ) -> float:
        """Balanceia escalas de loss via normas de gradiente"""
        ratio = grad_norm_adv / (grad_norm_pde + 1e-12)
        target_ratio = self.config.gradnorm_target_adv_to_pde
        scale = target_ratio / (ratio + 1e-12)
        scale = np.clip(
            scale,
            self.config.gradnorm_scale_min,
            self.config.gradnorm_scale_max,
        )
        self._ema_scale = (
            self.config.gradnorm_ema_beta * self._ema_scale +
            (1 - self.config.gradnorm_ema_beta) * scale
        )
        return self._ema_scale


class StagnationDetector:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._steps_without_improvement = 0
        self._best_residual = float("inf")
    
    def update(self, residual: float) -> Tuple[bool, float]:
        """Retorna (stagnation_detected, boost_factor)"""
        improvement = self._best_residual - residual
        is_improvement = (
            improvement > self.config.adv_stagnation_rel_tol *
            max(abs(self._best_residual), 1e-8)
        )
        
        if is_improvement:
            self._best_residual = residual
            self._steps_without_improvement = 0
            boost = 1.0
        else:
            self._steps_without_improvement += 1
            if self._steps_without_improvement >= self.config.adv_stagnation_patience:
                boost = self.config.adv_stagnation_boost_factor
            else:
                boost = 1.0
        
        return self._steps_without_improvement >= self.config.adv_stagnation_patience, boost


class DivergenceDetector:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._loss_window: deque[float] = deque(
            maxlen=config.divergence_window
        )
    
    def update(self, loss: float) -> bool:
        """Retorna True se divergência detectada"""
        self._loss_window.append(loss)
        if len(self._loss_window) < 4:
            return False
        
        recent_mean = np.mean(list(self._loss_window)[-4:])
        old_mean = np.mean(list(self._loss_window)[:4])
        ratio = recent_mean / (old_mean + 1e-12)
        
        return ratio > self.config.divergence_ratio_threshold
```

**Justificativa:**
- Cada scheme é uma classe testável
- Reusável em outros projetos de ML
- Fácil de debugar (estado interno claro)
- Reduce `FieldPIGANTrainer` de 2176 para ~800 linhas

---

#### **4.3.4 Simplificação do Trainer**

**ANTES:**
```python
class FieldPIGANTrainer:
    # 2176 linhas
    # 40+ variáveis de estado
    # Múltiplas responsabilidades
```

**DEPOIS:**
```python
class FieldPIGANTrainer:
    def __init__(
        self,
        generator: UNetGenerator2D,
        discriminator: DataDiscriminator2D,
        laplacian: LaplacianLayer,
        config: ExperimentConfig,
        device: torch.device,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.config = config
        self.device = device
        
        # Injetar componentes (composição, não herança)
        self.loss_computer = PIGANLossComputation(laplacian, config)
        self.lambda_pde_adapter = AdaptiveLambdaPDE(config)
        self.gradnorm_balancer = GradNormBalancer(config)
        self.stagnation_detector = StagnationDetector(config)
        self.divergence_detector = DivergenceDetector(config)
        
        # Otimizadores
        self.opt_g = optim.Adam(generator.parameters(), **optimizer_kwargs)
        self.opt_d = optim.Adam(discriminator.parameters(), **optimizer_kwargs)
    
    def train_step(
        self,
        base_field: torch.Tensor,
        phi_mask: torch.Tensor,
        ref_field: torch.Tensor,
        interior_mask: torch.Tensor,
        boundary_mask: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Um passo de treinamento (generator + discriminator)"""
        metrics = {}
        
        # 1. Amostrar latente
        z = self._sample_latent(self.config.batch_size)
        
        # 2. Gerador forward
        pred = self.generator(base_field, phi_mask, z=z, coord_field=coords)
        
        # 3. Atualizar discriminador
        for _ in range(self.config.n_critic):
            loss_d = self._compute_loss_d(pred, ref_field, pred.detach())
            self.opt_d.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
            self.opt_d.step()
        metrics["loss_d"] = float(loss_d.item())
        
        # 4. Atualizar gerador
        pred = self.generator(base_field, phi_mask, z=z, coord_field=coords)
        loss_pde = self.loss_computer.compute_pde_loss(pred, interior_mask, weight_map)
        loss_adv = self.loss_computer.compute_adversarial_loss(self.discriminator, pred, ref_field)
        loss_bc = self.loss_computer.compute_boundary_loss(pred, base_field, boundary_mask)
        loss_div = self.loss_computer.compute_diversity_loss(...)
        
        # 5. Adaptação dinâmica
        lambda_pde_dyn = self.lambda_pde_adapter.update(float(loss_pde), self.config.residual_scale_reference)
        lambda_adv_eff = self.gradnorm_balancer.update(
            grad_norm_adv=self._compute_grad_norm(loss_adv),
            grad_norm_pde=self._compute_grad_norm(loss_pde),
        )
        
        # 6. Composição de loss
        loss_g = self.loss_computer.compose_generator_loss(
            loss_pde, loss_adv, loss_bc, loss_div,
            lambda_pde_dyn, lambda_adv_eff,
            self.config,
        )
        
        # 7. Otimizar gerador
        self.opt_g.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.max_grad_norm)
        self.opt_g.step()
        
        # 8. Detecção de anomalias
        is_diverging = self.divergence_detector.update(float(loss_g))
        
        metrics.update({
            "loss_g": float(loss_g.item()),
            "loss_pde": float(loss_pde.item()),
            "loss_adv": float(loss_adv.item()),
            "loss_bc": float(loss_bc.item()),
            "loss_div": float(loss_div.item()),
            "lambda_pde_dyn": lambda_pde_dyn,
            "lambda_adv_eff": lambda_adv_eff,
            "is_diverging": is_diverging,
        })
        
        return metrics
```

**Justificativa:**
- Reduzido de 2176 para ~150 linhas
- Lógica principal clara e legível
- Cada sub-componente testável
- Fácil de entender o fluxo de treinamento

---

### 4.4 Novo Módulo: `src/physics/`

#### **Arquivo: `src/physics/pde_residual.py`**

```python
class PDE_Residual_Computer:
    """Calcula resíduo da PDE via Laplaciano."""
    
    def __init__(self, laplacian: LaplacianLayer):
        self.laplacian = laplacian
    
    def compute(self, field: torch.Tensor) -> torch.Tensor:
        """Retorna ∇²T no interior."""
        return self.laplacian(field)
    
    def compute_weighted_loss(
        self,
        residual: torch.Tensor,
        interior_mask: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calcula loss ponderado: E[|∇²T| · w]"""
        abs_residual = residual.abs()
        if weight_map is not None:
            weighted = (abs_residual * weight_map).sum()
            normalization = weight_map.sum()
        else:
            weighted = (abs_residual * interior_mask).sum()
            normalization = interior_mask.sum()
        return weighted / (normalization + 1e-12)
```

#### **Arquivo: `src/physics/domain_metrics.py`**

```python
def compute_physics_metrics(
    pred_field: torch.Tensor,
    laplacian: LaplacianLayer,
    interior_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    base_field: torch.Tensor,
) -> Dict[str, float]:
    """Retorna métricas físicas: resíduo PDE, erro de contorno, etc."""
    residual = laplacian(pred_field)
    
    pde_residual_mean = (residual.abs() * interior_mask).sum() / (interior_mask.sum() + 1e-12)
    pde_residual_l2 = torch.sqrt(((residual**2) * interior_mask).sum() / (interior_mask.sum() + 1e-12))
    pde_residual_max = (residual.abs() * interior_mask).max()
    
    boundary_error = (
        (pred_field - base_field).abs() * boundary_mask
    ).sum() / (boundary_mask.sum() + 1e-12)
    
    return {
        "pde_residual_mean": float(pde_residual_mean.item()),
        "pde_residual_l2": float(pde_residual_l2.item()),
        "pde_residual_max": float(pde_residual_max.item()),
        "boundary_error": float(boundary_error.item()),
    }
```

---

### 4.5 Resumo de Mudanças

| Arquivo | Mudança | Impacto |
|---------|---------|--------|
| `config.py` | Unificar SystemConfig + ExperimentConfig | -200 linhas em trainer.py |
| `models.py` | Sem mudanças | Mantém compatibilidade |
| `trainer.py` | Refatorar para ~800 linhas, injetar componentes | Manutenibilidade ↑↑ |
| `pipeline.py` | Simplificar (remover lógica de loss/scheduling) | Foco em orquestração |
| **Novo:** `physics/pde_residual.py` | Extrair cálculo de resíduo | Reutilizabilidade ↑ |
| **Novo:** `physics/domain_metrics.py` | Extrair métricas físicas | Testabilidade ↑ |
| **Novo:** `training/loss_functions.py` | Extrair computação de loss | Clareza ↑↑ |
| **Novo:** `training/adaptive_schemes.py` | Extrair adaptação dinâmica | Debugabilidade ↑ |

---

## ETAPA 5: RISCOS E LIMITAÇÕES

### 5.1 Riscos de Migração

#### **Risco 1: Quebra de Checkpoints**
- **Severidade:** 🔴 CRÍTICA
- **Problema:** Se mudarmos nomes de atributos em `FieldPIGANTrainer`, checkpoints antigos não carregarão
- **Mitigação:**
  - Manter compatibilidade reversa via `_load_model_state()` com strict=False
  - Versionar checkpoints com tag de compatibilidade
  - Testar `test_checkpoint_resume.py` após refatoração

#### **Risco 2: Regressão nas Correções SKILL (P1, P2, P3)**
- **Severidade:** 🔴 CRÍTICA
- **Problema:** Mover código P1/P2/P3 entre módulos pode introduzir bugs
- **Mitigação:**
  - Executar `validate_skill_corrections.py` após cada refatoração
  - Manter testes específicos:
    - `test_hard_constraint_bc_zeroed_when_active()` para P1
    - `test_lambda_adv_and_diversity_values()` para P2
    - `test_saturation_warning_logged()` para P3

#### **Risco 3: Desempenho**
- **Severidade:** 🟡 MÉDIA
- **Problema:** Criar classes pequenas pode aumentar overhead de função
- **Mitigação:**
  - Usar `@torch.jit.script` em componentes críticos (se necessário)
  - Profile antes/depois com `cProfile`
  - Target: <2% overhead de tempo

#### **Risco 4: Confusão com Config**
- **Severidade:** 🟡 MÉDIA
- **Problema:** Usuários habituados a `FieldTrainerConfig` em trainer.py
- **Mitigação:**
  - Docstring clara: "Config centralizada em ExperimentConfig"
  - Deprecation warning se alguém importar `FieldTrainerConfig`
  - Guia de migração em `MIGRATION_GUIDE.md`

---

### 5.2 Limitações Técnicas do Modelo Atual

#### **Limitação 1: Geometria Fixa (Retangular)**
- **Problema:** Hard constraint $\phi(\mathbf{x}) = x(1-x)y(1-y)$ é específico para domínio retangular
- **Impacto:** Impossível usar em geometrias L-shaped, circular, etc.
- **Workaround:**
  - Mapeamento conformal (complexo)
  - Treinamento em domínio retangular + extrapolação (com cuidado)
- **Refatoração:** Modularizar máscara de hard constraint para aceitar função arbitrária

#### **Limitação 2: Regime Estacionário Apenas**
- **Problema:** Equação de Laplace (sem termo transiente $\partial T/\partial t$)
- **Impacto:** Não pode resolver Poisson transiente ($\rho c \partial T/\partial t + \nabla \cdot k \nabla T = q$)
- **Workaround:** Adicionar termo temporal em loss (mais complexo)
- **Custo:** ~2–3 semanas de desenvolvimento

#### **Limitação 3: Condições de Contorno Dirichlet Apenas**
- **Problema:** Hard constraint força Dirichlet exato
- **Impacto:** Neumann (fluxo), Robin (convecção) não suportadas
- **Workaround:** Aproximar via penalty (perde garantia arquitetural)
- **Refatoração:** Projeto de hard constraint genérico (Dirichlet + Neumann + Robin)

#### **Limitação 4: Sem Termo Fonte (Poisson)**
- **Problema:** $\nabla^2 T = 0$ (Laplace pura)
- **Impacto:** Problemas com fonte de calor interna $\nabla^2 T + q/k = 0$ precisam de hack
- **Workaround:** Adicionar termo $q/k$ como campo de entrada (não mais CI)
- **Refatoração:** Incorporar operador Poisson na PDE

#### **Limitação 5: Suavização de Saída Não-Determinística**
- **Problema:** `F.pad(reflect)` em CUDA é não-determinístico
- **Impacto:** Reprodutibilidade não 100%
- **Mitigação:** Usar `replicate` em modo determinístico (atual)
- **Refatoração:** Implementar suavização própria (Gaussian blur manual)

#### **Limitação 6: Acurácia Numérica**
- **Problema:** Stencil 5-ponto é O(h²), não O(h⁴)
- **Impacto:** Erros acumulam em malhas grosseiras
- **Workaround:** Refinar malha (custo computacional)
- **Refatoração:** Implementar stencil 9-ponto (O(h⁴)) com Conv2d 3×3 dupla

---

### 5.3 Limitações de Software

#### **Limitação 1: Reprodutibilidade 99.9%**
- Determinismo completo difícil com CUDA (alguns operadores não-determinísticos)
- Seed + deterministic_algorithms_enabled não cobre 100%

#### **Limitação 2: Testabilidade Limitada**
- Testes em GPU são lentos
- CI/CD pode não ter GPU

#### **Limitação 3: Portabilidade**
- Código está otimizado para NVIDIA CUDA
- AMD/Intel GPUs precisariam refactoring

---

## CONCLUSÕES E PRÓXIMOS PASSOS

### ✅ Pontos Fortes Atuais
1. Correções SKILL (P1, P2, P3) bem implementadas
2. Operador Laplaciano correto e eficiente
3. Hard constraint bem formulado
4. Testes razoáveis
5. Documentação técnica clara

### ❌ Pontos Fracos
1. Código monolítico em trainer.py (2176 linhas)
2. Config duplicada entre config.py e trainer.py
3. Acoplamento forte entre módulos
4. Complexidade de scheduling dificulta debugging

### 🎯 Refatoração Recomendada (Ordem de Prioridade)

1. **Curto prazo (semana 1):**
   - Unificar config em ExperimentConfig
   - Executar P1/P2/P3 validação
   - Atualizar testes

2. **Médio prazo (semana 2-3):**
   - Extrair loss_functions.py
   - Extrair adaptive_schemes.py
   - Simplificar trainer.py para ~800 linhas

3. **Longo prazo (mês 2):**
   - Suporte a Poisson (termo fonte)
   - Suporte a Neumann/Robin
   - Stencil de 9-ponto (O(h⁴))

---

**Fim da Análise Arquitetural Completa**

Data: 22 de maio de 2026
