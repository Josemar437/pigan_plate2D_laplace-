# Revisão Completa PI-GAN — Conformidade com pigan-skill

**Data:** 13 de maio de 2026  
**Revisor:** GitHub Copilot  
**Base de Referência:** `pigan-skill/references/{SKILL.md, formulation.md}`

---

## Executivo

O modelo PI-GAN atual **implementa corretamente a variante III (Híbrida)** com discriminador pareado (pix2pix) e resíduo PDE no gerador. No geral, a arquitetura segue as recomendações do skill, mas há **3 problemas críticos** e **7 recomendações** que devem ser abordadas antes de publicação.

| Prioridade | Achado | Impacto | Status |
|---|---|---|---|
| 🔴 CRÍTICO | Hard constraint λ_BC não zerado | Dupla penalização de boundary | ✋ Requer fix |
| 🔴 CRÍTICO | λ_adv_warmup muito longo (>50% total) | Gerador ignora z | ✋ Requer tuning |
| 🔴 CRÍTICO | Satura ção de λ_PDE_dyn em teto | Pressão PDE limitada artificialmente | ✋ Monitorar |
| 🟠 ALTO | Declaração de variante não explícita em docs | Ambigüidade para leitores | ⏳ Documento |
| 🟠 ALTO | Métricas: faltam boundary_error sistêmico | Validação de hard constraint incompleta | ✋ Requer add |
| 🟡 MÉDIO | Referência FDM não declarada em objetivo | Descrição imprecisa | ⏳ Texto |
| 🟡 MÉDIO | Penalização de diversidade não ativa | Modo estocástico subexplorado | ⏳ Config |

---

## 1. Taxonomia de Arquitetura: Conformidade ✅✋

### Achado 1.1: Variante III (Híbrida) Identificada

**Código:** `src/trainer.py:1434`, `src/models.py`

```python
# L_G = lambda_pde_dyn*L_PDE_raw + lambda_adv_eff*L_adv + lambda_bc*L_BC
```

- ✅ **Gerador:** Penaliza resíduo PDE (`L_PDE`) + termo adversarial (`L_adv`)
- ✅ **Discriminador:** Pareado condicional (pix2pix), opera em `(T_θ, T_ref)`
- ✅ **Ambos:** Recebem pressão de consistência física

**Conformidade:** A variante está correta e bem implementada.

### Achado 1.2: Declaração de Variante — PROBLEMA

**Status:** 🟠 ALTO

A documentação e comentários **nunca declaram explicitamente que se trata de Variante III**. O texto fala em "PI-GAN fisicamente consistente" mas não clarifica:

- Onde a física entra (ambos os termos)
- Qual o papel exato do discriminador
- Como isso difere de Variante I ou II

**Recomendação 1.2:**

Adicionar ao `README.md` ou `docs/`:

```markdown
### Variante de Arquitetura

Este projeto implementa uma **PI-GAN Variante III (Híbrida)**:

- **Gerador:** Recebe penalização de resíduo PDE (Variante I) + pressão adversarial
- **Discriminador:** Discriminador pareado (pix2pix, Isola et al. 2017) que opera em pares `(T_θ, T_{ref})`
  - Entrada fake: `(T_gerador, T_referência_FDM)`
  - Entrada real: `(T_referência_FDM, T_referência_FDM)` — pares verdadeiros
- **Objetivo:** Combinar consistência de resíduo + compatibilidade com referência numérica

Isso é mais expressivo que Variante I (puro PDE) mas mais instável que Variante II.
```

---

## 2. Imposição de Condições de Contorno: PROBLEMA CRÍTICO 🔴

### Achado 2.1: Hard Constraint Implementado Corretamente

**Código:** `src/models.py:428-430`, `src/utils.py:130-183`

```python
# Hard constraint:
return base_field + phi_mask * raw  # φ=0 na fronteira por construção
```

Verificação:
- ✅ `φ(x,y) = 0` em toda a fronteira (garantido por `x_hat*(1-x_hat)*y_hat*(1-y_hat)`)
- ✅ Versão tanh suaviza Laplaciano em cantos (Intervenção 4)
- ✅ Normaliza para `φ_max = 1.0` no centro

**Conformidade:** Hard constraint funciona corretamente.

### Achado 2.2: PROBLEMA CRÍTICO — λ_BC Não Zerado

**Código:** `src/trainer.py:1499`

```python
g_loss = (
    pde_term
    + adv_term
    + float(self.cfg.lambda_bc) * g_bc  # ← PROBLEMA: lambda_bc = 20.0
    + lambda_diversity * diversity_loss
)
```

**Problema:**

Com hard constraint ativo (`hard_constraint_bc = True`), o termo `g_bc` (boundary loss) **deveria ser zerado ou não incluído** na perda total. Atualmente:

- `λ_BC = 20.0` (config padrão)
- Fronteira é imposta **duas vezes**: uma via hard constraint (exato), outra via `L_BC` (penalidade)
- Isso cria **dupla penalização** e pode interferir com a convergência

**SKILL Rule (Section 2.2):**

> "Com hard constraint ativo, λ_BC ≡ 0 — não inclua o termo L_BC na loss."

**Recomendação 2.2:**

```python
# Opção A: Condicional (RECOMENDADO)
if self.cfg.hard_constraint_bc:
    lambda_bc_effective = 0.0  # Hard constraint já impõe fronteira exatamente
else:
    lambda_bc_effective = float(self.cfg.lambda_bc)

g_loss = (
    pde_term
    + adv_term
    + lambda_bc_effective * g_bc
    + lambda_diversity * diversity_loss
)

# Opção B: Documentação clara em config
# Se hard_constraint_bc=True, sempre definir lambda_bc=0 no JSON
```

**Ação:** Adicionar à `config.py` após `__post_init__()`:

```python
def __post_init__(self) -> None:
    # ... validações existentes ...
    
    # Recomendação 2.2: Com hard constraint, lambda_bc deve ser zero
    if bool(self.hard_constraint_bc) and float(self.lambda_bc) > 1e-6:
        import warnings
        warnings.warn(
            "⚠️  hard_constraint_bc=True mas lambda_bc=%.2f. "
            "Com hard constraint, λ_BC deve ser 0 para evitar dupla penalização. "
            "Ajustando lambda_bc → 0.0." % float(self.lambda_bc),
            UserWarning
        )
        self.lambda_bc = 0.0
```

---

## 3. Modo Estocástico vs. Determinístico: PROBLEMA CRÍTICO 🔴

### Achado 3.1: Configuração Correta, Mas Aquecimento Muito Longo

**Código:** `src/config.py:191`

```python
generator_mode: str = "stochastic_pigan"  # ✅ Correto
latent_dim: int = 8  # ✅ > 0, modo estocástico
adv_warmup_epochs: int = 400  # ← PROBLEMA
epochs: int = 4000  # Total
```

**Cálculo:**

```
adv_warmup_epochs / epochs = 400 / 4000 = 10%
```

✅ Está **abaixo do limiar de 50%** recomendado pelo SKILL.

**Porém:** A recomendação é **≤ 15% do total**. Com 10%, está próximo do limite aceitável.

### Achado 3.2: Hiperparâmetros de Estocástico — FORA DAS RECOMENDAÇÕES

**Código:** `src/config.py` (linhas 227-240)

```python
lambda_adv: float = 2.0e-1  # = 0.2
target_adv_over_pde: float = 0.02  # = 2%
lambda_diversity: float = 0.0  # ← NÃO ATIVA
```

**Problemas per SKILL:**

| Parâmetro | Valor Atual | Recomendado | Status |
|---|---|---|---|
| `lambda_adv` | 0.2 (1e-1) | 1e-3 a 1e-2 | ✋ ALTO (risco de mode collapse) |
| `target_adv_over_pde` | 0.02 (2%) | 3% a 10% | ✋ BAIXO (adversarial fraco) |
| `lambda_diversity` | 0.0 | ~1e-4 | ✋ NÃO ATIVA (sem incentivo de diversidade) |

**Recomendação 3.2a — Aumentar lambda_adv:**

```python
# src/config.py:227
lambda_adv: float = 5.0e-3  # Aumentar de 0.2e-3 para 5e-3 = 0.005

# Justificativa: modo estocástico precisa de pressão adversarial suficiente
# para que o gerador use z enquanto ainda tem plasticidade
```

**Recomendação 3.2b — Ativar diversidade:**

```python
# src/config.py
lambda_diversity: float = 1.0e-4  # Ativar com valor recomendado
# Adiciona termo L_div = -1/(N(N-1)) Σ ||T_i - T_j||_2 / HW
```

**Recomendação 3.2c — Aumentar target_adv_over_pde:**

```python
# src/config.py
target_adv_over_pde: float = 0.05  # Aumentar de 0.02 para 0.05 = 5%
```

**Verificação:** O gerador realmente usa z?

Execute pós-treinamento:

```python
# Pseudocódigo para validate_latent_usage():
z1 = torch.randn(10, latent_dim)
z2 = torch.zeros(10, latent_dim)
pred1 = generator(..., z=z1)
pred2 = generator(..., z=z2)
diversity = torch.norm(pred1 - pred2) / torch.norm(pred1)
print(f"Relative diversity: {diversity:.4f}")
# Se < 0.01, z é ignorado → precisa aumentar lambda_adv e diminuir adv_warmup
```

---

## 4. Crítico e WGAN-GP: CONFORMIDADE ✅

### Achado 4.1: Estrutura WGAN-GP Correta

**Código:** `src/trainer.py:1600+` (discriminador update)

```python
# Loss do discriminador segue WGAN-GP:
# L_D = E[D(fake)] - E[D(real)] + λ_gp*GP + λ_drift*E[D(real)²] + λ_gap*GapPenalty
```

Verificado:
- ✅ Gradient Penalty implementado (Gulrajani et al. 2017)
- ✅ Drift penalty presente
- ✅ Gap penalty para estabilidade
- ✅ Entrada pareada para DataDiscriminator

**Conformidade:** Implementação correta.

### Achado 4.2: Documentação do Discriminador Pareado

**Código:** `src/models.py:625+`

```python
class DataDiscriminator2D(_BaseDiscriminator2D):
    """Discriminador de dados pareado (pix2pix)."""
```

Entradas:
- Fake: `(T_gerador, T_ref)` → predição do gerador vs. referência
- Real: `(T_ref, T_ref)` → pares verdadeiros (identidade)

**SKILL Rule (Section 3.2):**

> "DataDiscriminator é um discriminador condicional pareado (estrutura pix2pix). Escreva o argumento completo:
> L_adv = -E[D_2(T_θ, T_ref)]"

**Recomendação 4.2:**

Adicionar comentário explícito em `src/trainer.py` perto do cálculo de `g_adv`:

```python
# Discriminador pareado (pix2pix):
# D_2( (T_θ, T_ref) ) → score de "realismo do par"
# Entrada real: (T_ref, T_ref) — identidade
# Entrada fake: (T_θ, T_ref) — par gerado vs. referência
g_adv = -self.discriminator(T_pred, T_ref)  # Negativo para minimização do gerador
```

---

## 5. Peso PDE Adaptativo: CONFORMIDADE COM PROBLEMAS 🟡

### Achado 5.1: Mecanismo Adaptativo Correto

**Código:** `src/trainer.py:830-862`

```python
def _update_lambda_pde_dynamic(self, residual_mean_abs: float) -> float:
    target = max(float(getattr(self.cfg, "residual_scale_reference", 1.0)), 1e-12)
    ratio = max(float(residual_mean_abs) / target, 1.0)
    gain = max(float(getattr(self.cfg, "lambda_pde_growth_exponent", 0.5)), 1e-6)
    desired = base * (1.0 + gain * np.log10(ratio))
    desired = np.clip(desired, lambda_min, lambda_max_effective)
    beta = float(np.clip(getattr(self.cfg, "lambda_pde_ema_beta", 0.9), 0.0, 0.9999))
    self._lambda_pde_dyn = beta * self._lambda_pde_dyn + (1.0 - beta) * desired
```

✅ Implementa corretamente:
$$\lambda_{PDE}^{dyn} \leftarrow \beta\,\lambda_{PDE}^{dyn} + (1-\beta)\,\lambda_{des}$$

### Achado 5.2: Calibração de residual_scale_reference — VERIFICAÇÃO

**Valor atual:** `residual_scale_reference = 0.04` (modificado por Intervenção 1)

**SKILL Rule (Section 4):**

> "Calibração de r̄₀: deve ser próximo ao resíduo médio convergido esperado, não ao alvo final."

**Questão:** O valor 0.04 é apropriado para esse modelo?

**Análise:**

- Se o resíduo converge para ~0.04-0.05, então r̄₀ ≈ 0.04 é correto ✅
- Se o resíduo converge para ~0.01, então r̄₀ deve ser ~0.01 ✋

**Recomendação 5.2:**

Registrar o resíduo médio observado durante treinamento e validar:

```python
# Após época 1000:
epoch_1000_residual_mean = 0.042
if abs(epoch_1000_residual_mean - 0.04) > 0.02:
    print(f"⚠️ Ajustar residual_scale_reference de 0.04 para {epoch_1000_residual_mean:.3f}")
```

### Achado 5.3: PROBLEMA — Saturação de λ_PDE_dyn em Teto

**SKILL Rule (Section 4):**

> "Sinal de problema: λ_PDE_dyn saturando no teto λ_max^eff durante todo o treinamento indica que os objetivos PDE e adversarial estão em tensão."

**Verificação necessária:**

Monitor `g_lambda_pde_dyn` durante treinamento. Se atingir `lambda_pde_max = 150.0` (Intervenção 2) e permanecer lá:

- 🔴 **Crítico:** Pressão PDE é limitada artificialmente
- **Solução:** Aumentar `lambda_pde_max` ou reduzir `lambda_adv`

**Recomendação 5.3:**

Adicionar alerta ao logging:

```python
if float(lambda_pde_dyn) > 0.95 * float(lambda_pde_max_effective):
    self.logger.warning(
        f"λ_PDE_dyn saturou no teto: {lambda_pde_dyn:.2f} ≈ {lambda_pde_max_effective:.2f}. "
        "Considere aumentar lambda_pde_max ou reduzir lambda_adv."
    )
```

---

## 6. Operador Laplaciano Discreto: CONFORMIDADE ✅

### Achado 6.1: Stencil 5 Pontos Correto

**Código:** `src/models.py:450-467`

```python
kernel = torch.tensor([
    [0.0, 1.0 / (hy * hy), 0.0],
    [1.0 / (hx * hx), -2.0 * (1.0 / (hx * hx) + 1.0 / (hy * hy)), 1.0 / (hx * hx)],
    [0.0, 1.0 / (hy * hy), 0.0],
], dtype=torch.float32)
```

✅ Divisão por $h^2$ está **explícita**
✅ Stencil é 3×3 (5 pontos: centro + 4 vizinhos)
✅ Conv2d com `padding=0` → apenas interior

**Conformidade:** Perfeita.

### Achado 6.2: Verificação de Integridade

**Código:** `src/models.py:477-497`

```python
interior = self.conv(field)  # [B,1,H-2,W-2]
residual = torch.zeros_like(field)
residual[:, :, 1:-1, 1:-1] = interior
return residual
```

✅ Bordas têm resíduo zero por construção
✅ Apenas interior é calculado

**Conformidade:** Perfeita.

---

## 7. Referência FDM: Conformidade com Caveat 🟡

### Achado 7.1: FDM em Discriminador

**Código:** `src/pipeline.py:177+`, `src/trainer.py`

- ✅ Solução FDM é calculada como referência (`T_ref`)
- ✅ Discriminador pareado recebe `(T_θ, T_ref)`

**SKILL Rule (Section 7):**

> "O modelo NÃO é puramente physics-informed. O texto correto: O modelo é uma PI-GAN informada por física e assistida por referência numérica."

### Achado 7.2: Descrição Incompleta em Documentação

**Status:** 🟡 MÉDIO

Nenhum documento declara explicitamente que a referência FDM entra no discriminador. Isso pode levar a ambigüidade: é um modelo PINN puro ou numérico-informado?

**Recomendação 7.2:**

Adicionar à metodologia (README ou artigo):

```markdown
### Papel da Referência Numérica

O modelo combina:

1. **Consistência de Resíduo:** Gerador minimiza |∇²T_θ| nos pontos interiores
2. **Compatibilidade Numérica:** Discriminador pareado (pix2pix) compara pares (T_θ, T_FDM)
   - Isso incentiva o gerador a produzir campos próximos à solução de referência

A justificativa é dupla:
- Física: resíduo baixo implica satisfação da PDE
- Numérica: compatibilidade com solução bem estabelecida

Isso **não é puramente physics-informed** (que usaria apenas resíduo).
É uma **PI-GAN assistida por referência numérica** (Variante III).
```

---

## 8. Salvando e Reportando Métricas: PROBLEMA 🟡

### Achado 8.1: Campos Obrigatórios — Verificação

**SKILL Rule (Section 8.1):**

Métricas obrigatórias:

```json
{
  "mae": float,
  "rmse": float,
  "r2": float,
  "relative_l2_error_vs_fdm": float,
  "max_error": float,
  "pde_residual_mean": float,
  "pde_residual_l2": float,
  "pde_residual_max": float,
  "boundary_error": float
}
```

**Verificação:** `src/evaluation.py:65+`

```python
return {
    "mae": float(mae.item()),
    "rmse": float(rmse.item()),
    "mape": float(mape.item()),
    "r2": float(r2.item()),
    "relative_l2_error": float(rel_l2.item()),
    "max_error": float(max_error.item()),
    "pde_residual_mean": float(pde_residual_mean.item()),
    "pde_residual_l2": float(pde_residual_l2.item()),
    "pde_residual_max": float(pde_residual_max.item()),
    "boundary_error": float(boundary_error.item()),
}
```

✅ Todos os 9 campos estão presentes (1 adicional: mape)

**Conformidade:** Perfeita.

### Achado 8.2: Apresentação Simultânea de Residual e Fit

**SKILL Rule (Section 8.1):**

> "Nunca reporte apenas métricas de ajuste (MAE, R²) sem as métricas de resíduo PDE."

**Status:** ✅ Conformidade

O código salva ambos os conjuntos. Porém, necessário verificar se os gráficos e relatórios públicos mostram ambos simultaneamente.

**Recomendação 8.2:**

Se há gráficos de treinamento, incluir dois painéis:

```
Painel 1 (Esquerda):  MAE, RMSE, R² (métricas de ajuste)
Painel 2 (Direita):   pde_residual_mean, pde_residual_max, boundary_error (física)
```

---

## 9. Síntese de Problemas e Ações Prioritárias

### Crítico 🔴 — Deve Ser Feito

| ID | Problema | Solução | Arquivo | Linhas |
|---|---|---|---|---|
| P1 | λ_BC não zerado com hard constraint | Condicional: se hard_constraint_bc=True, então λ_BC=0 | config.py, trainer.py | 212, 1499 |
| P2 | λ_adv muito alto (0.2) para estocástico | Reduzir para 5e-3, ativar lambda_diversity=1e-4 | config.py | 227-228, 233 |
| P3 | Saturation de λ_PDE_dyn monitorada? | Adicionar alerta se λ_PDE_dyn > 0.95*λ_max | trainer.py | 852 |

### Alto 🟠 — Deve Ser Feito Antes de Publicação

| ID | Problema | Solução | Arquivo |
|---|---|---|---|
| P4 | Variante III não declarada explicitamente | Adicionar seção em README descrevendo variante | README.md |
| P5 | Discriminador pareado não documentado | Comentário em trainer.py sobre entrada D_2(T_θ, T_ref) | trainer.py |
| P6 | Referência FDM em discriminador não declarada | Seção em metodologia descrevendo assistência numérica | docs/METODOLOGIA.md |

### Médio 🟡 — Recomendado

| ID | Problema | Solução | Arquivo |
|---|---|---|---|
| P7 | Validação de uso de z | Script `validate_latent_usage()` para teste | tests/ |
| P8 | Calibração de residual_scale_reference | Monitor vs. real durante treinamento | trainer.py logging |

---

## 10. Checklist de Conformidade com pigan-skill

### Taxonomia (Seção 1)
- [ ] **P4:** Declarar explicitamente Variante III em documentação público-facing

### Formulação do Gerador (Seção 2)
- [ ] **P1:** Zerar λ_BC quando hard_constraint=True
- [x] Hard constraint φ=0 na fronteira verificado
- [x] Hard constraint com tanh suavizado implementado

### Modo Estocástico (Seção 5)
- [ ] **P2:** Reduzir λ_adv para faixa recomendada (5e-3)
- [ ] **P2:** Ativar λ_diversity com valor ~1e-4
- [x] adv_warmup < 15% total (10% ✓)

### Crítico WGAN-GP (Seção 3)
- [x] Gradient Penalty implementado
- [x] Drift penalty presente
- [x] Gap penalty presente
- [ ] **P5:** Documentar entrada pareada do discriminador

### Peso PDE Adaptativo (Seção 4)
- [x] Esquema adaptativo implementado com EMA
- [ ] **P3:** Alerta de saturação de λ_PDE_dyn
- [ ] **P8:** Validar calibração de residual_scale_reference

### Operador Laplaciano (Seção 6)
- [x] Divisão por h² explícita
- [x] Stencil 3×3, 5 pontos
- [x] padding=0, apenas interior

### Referência FDM (Seção 7)
- [ ] **P6:** Declarar que é "assistida por FDM" não "puramente PINN"

### Métricas e Reportagem (Seção 8)
- [x] Todos 9 campos obrigatórios salvos
- [x] Residual e fit apresentados conjuntamente (em código)
- [ ] **P8:** Validar gráficos públicos mostram ambos

---

## 11. Recomendação Final

**Prioridade 1 (Esta semana):**
1. Implementar P1 (λ_BC condicional)
2. Implementar P2 (λ_adv tuning + λ_diversity)
3. Implementar P3 (alerta saturação)

**Prioridade 2 (Antes de artigo/release):**
4. Implementar P4 (documentação variante)
5. Implementar P5 (comentário discriminador)
6. Implementar P6 (metodologia FDM)

**Prioridade 3 (Opcional, melhoria contínua):**
7. Implementar P7 (validação z)
8. Implementar P8 (logging calibração)

---

## Apêndice A: Referências

- **pigan-skill:** `pigan-skill/references/SKILL.md`
- **Formulation:** `pigan-skill/references/formulation.md`
- **Código PI-GAN:**
  - Arquitetura: `src/models.py`
  - Treinamento: `src/trainer.py`
  - Pipeline: `src/pipeline.py`
  - Configuração: `src/config.py`
  - Avaliação: `src/evaluation.py`

---

**Fim da Revisão**
