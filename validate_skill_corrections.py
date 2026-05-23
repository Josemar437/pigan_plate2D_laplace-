#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validação: Conformidade com Correções SKILL (P1, P2, P3)

Este script valida se as 3 correções críticas foram aplicadas corretamente:
  P1: lambda_bc zerado quando hard_constraint_bc=True
  P2: lambda_adv reduzido para 5e-3 + lambda_diversity ativado
  P3: Alerta quando lambda_pde_dyn saturar no teto

Uso:
    python validate_skill_corrections.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig


def safe_text(value):
    """Converte mensagens para caracteres imprimiveis no console Windows."""
    return str(value).encode("cp1252", errors="replace").decode("cp1252")


def validate_p1_lambda_bc():
    """
    P1: Validar que lambda_bc é zerado com hard constraint
    """
    print("\n" + "="*70)
    print("VALIDAÇÃO P1: lambda_bc condicional (hard constraint)")
    print("="*70)
    
    # Capturar warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = ExperimentConfig()
        
        # Verificar se warning foi lançado
        skill_warnings = [warning for warning in w if "SKILL Conformance" in str(warning.message)]
        
        if skill_warnings:
            print(
                "[PASS] Warning SKILL detectado durante __post_init__: "
                f"{safe_text(skill_warnings[0].message)}"
            )
        
        # Verificar estado final
        if config.hard_constraint_bc and config.lambda_bc == 0.0:
            print(f"[PASS] hard_constraint_bc={config.hard_constraint_bc}, lambda_bc={config.lambda_bc}")
            return True
        else:
            print(f"[FAIL] hard_constraint_bc={config.hard_constraint_bc}, lambda_bc={config.lambda_bc}")
            print(f"  Esperado: lambda_bc=0.0 quando hard_constraint_bc=True")
            return False


def validate_p2_stochastic_params():
    """
    P2: Validar lambda_adv reduzido e lambda_diversity ativado
    """
    print("\n" + "="*70)
    print("VALIDAÇÃO P2: Parâmetros modo estocástico (lambda_adv, lambda_diversity)")
    print("="*70)
    
    config = ExperimentConfig()
    
    # P2a: lambda_adv
    lambda_adv_min_recommended = 1.0e-3
    lambda_adv_max_recommended = 1.0e-2
    lambda_adv_value = float(config.lambda_adv)
    
    print(f"\nP2a: lambda_adv")
    print(f"  Valor atual: {lambda_adv_value:.6e}")
    print(f"  Intervalo SKILL: [{lambda_adv_min_recommended:.2e}, {lambda_adv_max_recommended:.2e}]")
    
    p2a_pass = lambda_adv_min_recommended <= lambda_adv_value <= lambda_adv_max_recommended
    if p2a_pass:
        print(f"  [PASS] Dentro do intervalo recomendado")
    else:
        print(f"  [FAIL] Fora do intervalo recomendado")
        if lambda_adv_value > lambda_adv_max_recommended:
            print(f"    Dica: Reduzir lambda_adv ou seria mode collapse")
    
    # P2b: lambda_diversity
    print(f"\nP2b: lambda_diversity")
    lambda_diversity_value = float(config.lambda_diversity)
    lambda_diversity_recommended = 1.0e-4
    print(f"  Valor atual: {lambda_diversity_value:.6e}")
    print(f"  Valor SKILL recomendado: {lambda_diversity_recommended:.2e}")
    
    p2b_pass = lambda_diversity_value > 0.0
    if p2b_pass:
        print(f"  [PASS] lambda_diversity ativado (valor: {lambda_diversity_value:.2e})")
        if abs(lambda_diversity_value - lambda_diversity_recommended) < 0.5e-4:
            print(f"    Valor bem calibrado com recomendação")
    else:
        print(f"  [FAIL] lambda_diversity nao ativado (zero)")
    
    # P2c: latent_dim
    print(f"\nP2c: latent_dim (para modo estocástico)")
    latent_dim_value = int(config.latent_dim)
    print(f"  Valor atual: {latent_dim_value}")
    
    p2c_pass = latent_dim_value > 0
    if p2c_pass:
        print(f"  [PASS] latent_dim > 0 (modo {config.generator_mode})")
    else:
        print(f"  [FAIL] latent_dim = 0, modo deterministico (sem incerteza)")
    
    return p2a_pass and p2b_pass and p2c_pass


def validate_p3_saturation_warning():
    """
    P3: Validar que alerta de saturação está presente no código
    """
    print("\n" + "="*70)
    print("VALIDAÇÃO P3: Alerta saturação lambda_pde_dyn")
    print("="*70)
    
    # Ler código de trainer.py
    trainer_path = Path(__file__).parent / "src" / "trainer.py"
    trainer_code = trainer_path.read_text(encoding="utf-8")
    
    # Procurar por código de alerta
    search_strings = [
        "lambda_pde_saturation_warn",
        "SKILL Alert P3",
        "> 0.95 * lambda_max_eff"
    ]
    
    found_checks = []
    for search_str in search_strings:
        if search_str in trainer_code:
            found_checks.append(True)
            print(f"  [PASS] Encontrado: '{search_str}'")
        else:
            found_checks.append(False)
            print(f"  [FAIL] Nao encontrado: '{search_str}'")
    
    return all(found_checks)


def validate_config_consistency():
    """
    Validação adicional: Consistência geral de config
    """
    print("\n" + "="*70)
    print("VALIDAÇÃO ADICIONAL: Consistência de Configuração")
    print("="*70)
    
    config = ExperimentConfig()
    
    # Verificar adv_warmup
    adv_warmup_ratio = float(config.adv_warmup_epochs) / float(config.epochs)
    print(f"\nadv_warmup_epochs / epochs = {config.adv_warmup_epochs} / {config.epochs} = {adv_warmup_ratio:.1%}")
    if adv_warmup_ratio <= 0.15:
        print(f"  [PASS] <= 15% (recomendacao SKILL)")
    else:
        print(f"  [WARN] > 15% (pode ignorar z)")
    
    # Verificar target_adv_over_pde
    target_adv = float(config.target_adv_over_pde)
    print(f"\ntarget_adv_over_pde = {target_adv:.2%}")
    if 0.03 <= target_adv <= 0.10:
        print(f"  [PASS] Dentro do intervalo SKILL [3%, 10%]")
    else:
        print(f"  [WARN] Fora do intervalo ideal")
    
    # Resumo valores críticos
    print(f"\nResumo valores críticos:")
    print(f"  - hard_constraint_bc = {config.hard_constraint_bc}")
    print(f"  - hard_constraint_profile = {config.hard_constraint_profile}")
    print(f"  - lambda_pde_max = {config.lambda_pde_max}")
    print(f"  - residual_scale_reference = {config.residual_scale_reference}")
    print(f"  - generator_mode = {config.generator_mode}")


def main():
    print("\n" + "="*70)
    print(" VALIDACAO: CONFORMIDADE SKILL CORRECTIONS (P1, P2, P3)")
    print("="*70)
    
    results = {}
    
    # Executar validações
    results["P1: lambda_bc condicional"] = validate_p1_lambda_bc()
    results["P2: Parâmetros estocásticos"] = validate_p2_stochastic_params()
    results["P3: Alerta saturação"] = validate_p3_saturation_warning()
    
    # Validações adicionais
    validate_config_consistency()
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    
    all_pass = all(results.values())
    for name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} | {name}")
    
    print("\n" + "="*70)
    if all_pass:
        print("[PASS] TODAS AS VALIDACOES PASSARAM")
        print("  O código está em conformidade com as correções SKILL (P1, P2, P3)")
        return 0
    else:
        print("[FAIL] ALGUMAS VALIDACOES FALHARAM")
        print("  Revise as correções implementadas")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
