#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick-start script para testar as intervenções de hotspot.

Uso:
  python test_interventions.py --mode all
  python test_interventions.py --mode phase1
  python test_interventions.py --mode phase2
  python test_interventions.py --mode compare
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig, SystemConfig
from src.pipeline import PIGANPipeline
import json
from datetime import datetime


def run_phase1_baseline():
    """Executa Phase 1: Baseline + Intervenções 1+2+4 (sem amostragem adaptativa)."""
    print("\n" + "="*80)
    print("PHASE 1: Baseline + Intervenções 1+2+4 (sem amostragem adaptativa)")
    print("="*80)
    
    config = ExperimentConfig()
    # Intervenções 1+2+4 já estão ativas por default
    config.adaptive_sampling_enable = False
    config.hard_constraint_profile = "tanh"
    
    print(f"\nConfigurações ativas:")
    print(f"  - residual_scale_reference = {config.residual_scale_reference}")
    print(f"  - lambda_pde_max = {config.lambda_pde_max}")
    print(f"  - precision_refine_lambda_pde_max_scale = {config.precision_refine_lambda_pde_max_scale}")
    print(f"  - hard_constraint_profile = {config.hard_constraint_profile}")
    print(f"  - adaptive_sampling_enable = {config.adaptive_sampling_enable}")
    
    sys_config = SystemConfig()
    pipeline = PIGANPipeline(config, system_config=sys_config)
    
    print("\nIniciando treinamento Phase 1...")
    history = pipeline.run()
    
    return history, "phase1"


def run_phase2_adaptive():
    """Executa Phase 2: Baseline + Todas as 4 Intervenções (com amostragem adaptativa)."""
    print("\n" + "="*80)
    print("PHASE 2: Baseline + Intervenções 1+2+3+4 (com amostragem adaptativa)")
    print("="*80)
    
    config = ExperimentConfig()
    # Ativa Intervenção 3
    config.adaptive_sampling_enable = True
    config.adaptive_sampling_refine_every_epochs = 50
    config.adaptive_sampling_hotspot_threshold = 0.30
    config.adaptive_sampling_weight_scale = 2.0
    config.hard_constraint_profile = "tanh"
    
    print(f"\nConfigurações ativas:")
    print(f"  - residual_scale_reference = {config.residual_scale_reference}")
    print(f"  - lambda_pde_max = {config.lambda_pde_max}")
    print(f"  - precision_refine_lambda_pde_max_scale = {config.precision_refine_lambda_pde_max_scale}")
    print(f"  - hard_constraint_profile = {config.hard_constraint_profile}")
    print(f"  - adaptive_sampling_enable = {config.adaptive_sampling_enable}")
    print(f"  - adaptive_sampling_refine_every_epochs = {config.adaptive_sampling_refine_every_epochs}")
    print(f"  - adaptive_sampling_hotspot_threshold = {config.adaptive_sampling_hotspot_threshold}")
    
    sys_config = SystemConfig()
    pipeline = PIGANPipeline(config, system_config=sys_config)
    
    print("\nIniciando treinamento Phase 2...")
    history = pipeline.run()
    
    return history, "phase2"


def run_compare_polynomial():
    """Executa comparação: polynomial vs tanh hard_constraint."""
    print("\n" + "="*80)
    print("COMPARAÇÃO: polynomial vs tanh hard constraint")
    print("="*80)
    
    print("\n[1] Testando polynomial (método original)...")
    config1 = ExperimentConfig()
    config1.hard_constraint_profile = "polynomial"
    config1.adaptive_sampling_enable = False
    
    sys_config = SystemConfig()
    pipeline1 = PIGANPipeline(config1, system_config=sys_config)
    history1 = pipeline1.run()
    
    print("\n[2] Testando tanh (método novo)...")
    config2 = ExperimentConfig()
    config2.hard_constraint_profile = "tanh"
    config2.adaptive_sampling_enable = False
    
    pipeline2 = PIGANPipeline(config2, system_config=sys_config)
    history2 = pipeline2.run()
    
    # Comparação
    print("\n" + "="*80)
    print("RESULTADO DA COMPARAÇÃO")
    print("="*80)
    
    if history1 and history2:
        final1 = history1[-1] if history1 else {}
        final2 = history2[-1] if history2 else {}
        
        print(f"\nPolynomial (original):")
        print(f"  - pde_residual_max = {final1.get('g_residual_max_abs', 'N/A'):.6f}")
        print(f"  - pde_residual_mean = {final1.get('g_residual_mean_abs', 'N/A'):.6f}")
        
        print(f"\nTanh (novo):")
        print(f"  - pde_residual_max = {final2.get('g_residual_max_abs', 'N/A'):.6f}")
        print(f"  - pde_residual_mean = {final2.get('g_residual_mean_abs', 'N/A'):.6f}")
        
        max_imp = ((final1.get('g_residual_max_abs', 1.0) - final2.get('g_residual_max_abs', 0.0)) /
                   final1.get('g_residual_max_abs', 1.0) * 100)
        mean_imp = ((final1.get('g_residual_mean_abs', 1.0) - final2.get('g_residual_mean_abs', 0.0)) /
                    final1.get('g_residual_mean_abs', 1.0) * 100)
        
        print(f"\nMelhora (tanh vs polynomial):")
        print(f"  - pde_residual_max: {max_imp:.1f}% {'↓' if max_imp > 0 else '↑'}")
        print(f"  - pde_residual_mean: {mean_imp:.1f}% {'↓' if mean_imp > 0 else '↑'}")
    
    return [history1, history2], "compare"


def save_results(history, phase_name):
    """Salva resultados em JSON para análise."""
    results_dir = project_root / "runs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"intervention_test_{phase_name}_{timestamp}.json"
    
    # Converter numpy types para JSON-serializable
    def convert_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], list):
        # Múltiplos históricos (comparação)
        data = {"histories": [convert_types(h) for h in history]}
    else:
        # Histórico único
        data = {"history": convert_types(history)}
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Resultados salvos em: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Testar intervenções para redução de hotspot em PI-GAN"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "phase1", "phase2", "compare"],
        default="phase1",
        help="Modo de teste: all (1+2), phase1 (sem adaptativa), phase2 (com adaptativa), compare (polyn vs tanh)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar resultados em JSON"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("TESTE DE INTERVENÇÕES PARA HOTSPOT (1,3) - PI-GAN LAPLACE 2D")
    print("="*80)
    
    try:
        if args.mode == "phase1":
            history, name = run_phase1_baseline()
            if args.save:
                save_results(history, name)
        
        elif args.mode == "phase2":
            history, name = run_phase2_adaptive()
            if args.save:
                save_results(history, name)
        
        elif args.mode == "compare":
            histories, name = run_compare_polynomial()
            if args.save:
                save_results(histories, name)
        
        elif args.mode == "all":
            print("\nExecutando Phase 1...")
            h1, n1 = run_phase1_baseline()
            if args.save:
                save_results(h1, n1)
            
            print("\nExecutando Phase 2...")
            h2, n2 = run_phase2_adaptive()
            if args.save:
                save_results(h2, n2)
        
        print("\n" + "="*80)
        print("✅ TESTES CONCLUÍDOS COM SUCESSO")
        print("="*80)
    
    except KeyboardInterrupt:
        print("\n⏸️  Teste interrompido pelo usuário.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Erro durante teste: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
