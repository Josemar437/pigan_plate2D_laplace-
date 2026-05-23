#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUIA RÁPIDO: Começar com as Intervenções Implementadas
========================================================

Este arquivo descreve como usar imediatamente as mudanças implementadas.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 🚀 GUIA DE INÍCIO RÁPIDO - INTERVENÇÕES                    ║
╚════════════════════════════════════════════════════════════════════════════╝

MUDANÇAS IMPLEMENTADAS:
═══════════════════════

1. ✅ Recalibração de residual_scale_reference
   - Arquivo: src/config.py:343
   - De: 1.0e-2 (0.01)
   - Para: 0.04
   - Ativo por padrão

2. ✅ Aumento de lambda_pde_max
   - Arquivo: src/config.py:351
   - De: 98.0
   - Para: 150.0
   - Ativo por padrão

3. ✅ Ajuste de precision_refine_lambda_pde_max_scale
   - Arquivo: src/config.py:347
   - De: 0.72
   - Para: 0.80
   - Ativo por padrão

4. ✅ Hard Constraint com Perfil Tanh Suavizado
   - Arquivo: src/utils.py:130
   - Nova opção: hard_constraint_profile = "tanh"
   - Default: "tanh" (suavizado)
   - Alternativo: "polynomial" (original)


COMEÇAR A USAR:
═══════════════

PASSO 1: Verificar que tudo está configurado
─────────────────────────────────────────────
$ python verify_hard_constraint.py

Esperado:
  ✓ Hard Constraint Ativo: hard_constraint_bc = True
  ✓ Perfil de Hard Constraint: hard_constraint_profile = 'tanh'
  ✓ Verificação de Hard Constraint: max erro = 0.00e+00
  ✓ Todas as verificações passaram!


PASSO 2: Treinar com novas configurações
─────────────────────────────────────────
$ python main.py --config scripts/final_4000_best_config.json

Monitorar durante treinamento:
  - boundary_error < 1e-6  (hard constraint funcionando)
  - pde_residual_max       (deve recuar ~40%)
  - pde_residual_mean ≈ 0.04 (convergência esperada)
  - lambda_pde_dyn         (pode atingir ~120 agora)


PASSO 3: Comparar Resultados
─────────────────────────────
Antes (baseline):
  pde_residual_max = 0.824
  lambda_pde_max_effective = 70.56

Esperado após intervenções:
  pde_residual_max ≈ 0.50 (−40%)
  lambda_pde_max_effective = 120.0


USAR PERFIL POLYNOMIAL (ORIGINAL):
═══════════════════════════════════

Se quiser reverter para o comportamento original:

1. Editar src/config.py:
   hard_constraint_profile: str = "polynomial"

2. Impacto:
   - Volta ao Laplaciano O(1) nos cantos
   - Resíduo pode aumentar ~5-10%
   - Suavidade reduzida


REFERÊNCIA RÁPIDA DE CONFIGURAÇÕES:
═══════════════════════════════════

Baseline (antigo):
┌─────────────────────────────────────────────────┐
│ residual_scale_reference = 0.01                 │
│ precision_refine_lambda_pde_max_scale = 0.72   │
│ lambda_pde_max = 98.0                           │
│ hard_constraint_profile = "polynomial"          │
│ ───────────────────────────────────────────────  │
│ Resultado: pde_residual_max ≈ 0.824             │
└─────────────────────────────────────────────────┘

Novo (otimizado):
┌─────────────────────────────────────────────────┐
│ residual_scale_reference = 0.04                 │
│ precision_refine_lambda_pde_max_scale = 0.80   │
│ lambda_pde_max = 150.0                          │
│ hard_constraint_profile = "tanh"                │
│ ───────────────────────────────────────────────  │
│ Esperado: pde_residual_max ≈ 0.50               │
└─────────────────────────────────────────────────┘


TROUBLESHOOTING:
════════════════

❌ Erro: "hard_constraint_profile invalido"
   Solução: Use "tanh" ou "polynomial"

❌ boundary_error muito alto (> 1e-4)
   Motivo: Hard constraint pode estar desativo
   Verificar: config.hard_constraint_bc deve ser True

❌ pde_residual_max não melhora
   Possíveis causas:
   1. lambda_pde_dyn saturando em 150 (era 98)
   2. Necessário mais épocas para convergência
   3. Hotspot em canto necessita Intervenção 3 (amostragem adaptativa)


PRÓXIMAS ETAPAS:
════════════════

1. ✅ Implementadas: Intervenções 1, 2, 4
2. ⏳ Planejada: Intervenção 3 (amostragem adaptativa no hotspot)

Para Intervenção 3, será necessário:
  - Novo método em src/trainer.py: _refine_collocation_weights_adaptive()
  - Novos parâmetros em config.py:
    * hotspot_threshold = 0.3
    * hotspot_refinement_interval = 50
    * hotspot_weight_scale = 2.0


SUPPORT / DEBUG:
════════════════

Para logs detalhados:
  python -c "
  from src.config import ExperimentConfig
  cfg = ExperimentConfig()
  print(f'residual_scale_reference = {cfg.residual_scale_reference}')
  print(f'lambda_pde_max = {cfg.lambda_pde_max}')
  print(f'hard_constraint_profile = {cfg.hard_constraint_profile}')
  "

Para visualizar phi_mask:
  python -c "
  import torch
  from src.utils import create_cartesian_grid, build_hard_constraint_mask
  x, y = create_cartesian_grid(32, 32, 1.0, 1.0, device=torch.device('cpu'))
  phi = build_hard_constraint_mask(x, y, lx=1.0, ly=1.0, smooth_profile='tanh')
  print(f'phi min = {phi.min():.6f}')
  print(f'phi max = {phi.max():.6f}')
  print(f'phi[1,3] = {phi[1,3]:.6f}')  # Hotspot
  "


═════════════════════════════════════════════════════════════════════════════
Última atualização: 12 de Maio de 2026
Status: 3/4 intervenções implementadas e validadas ✅
═════════════════════════════════════════════════════════════════════════════
""")
