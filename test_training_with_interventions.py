#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Teste: Treinar PI-GAN com Intervenções Implementadas

Este script executa o treinamento com as 3 intervenções já implementadas:
1. residual_scale_reference recalibrado (0.04)
2. lambda_pde_max aumentado (150.0) + precision_refine_scale (0.80)
3. hard_constraint_profile com tanh suavizado

Uso:
    python test_training_with_interventions.py [--epochs 100] [--skip-verify]
"""

import argparse
import sys
from pathlib import Path

import torch

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig, SystemConfig, initialize_system
from src.pipeline import PIGANPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Teste de treinamento com intervenções implementadas"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Número de épocas para treinar (default: 100)",
    )
    parser.add_argument(
        "--skipVerify",
        "--skip_verify",
        dest="skip_verify",
        action="store_true",
        help="Pular verificação de hard constraint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Tamanho do batch (default: do config)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 TESTE DE TREINAMENTO COM INTERVENÇÕES IMPLEMENTADAS")
    print("="*80)
    
    # ============ Verificação Rápida ============
    if not args.skip_verify:
        print("\n[1/3] Verificando hard constraint...")
        try:
            from verify_hard_constraint import verify_hard_constraint
            if not verify_hard_constraint():
                print("\n❌ Verificação falhou! Abortando.")
                return 1
        except Exception as e:
            print(f"\n⚠️ Erro na verificação: {e}")
            print("Continuando mesmo assim...")
    
    # ============ Configuração ============
    print("\n[2/3] Carregando configuração...")
    exp_config = ExperimentConfig()
    sys_config = SystemConfig()
    
    # Ajustes via CLI se necessários
    if args.batch_size:
        exp_config.batch_size = args.batch_size
    
    exp_config.epochs = int(args.epochs)
    
    print(f"\n📊 Configuração Carregada:")
    print(f"   • residual_scale_reference = {exp_config.residual_scale_reference}")
    print(f"   • lambda_pde_max = {exp_config.lambda_pde_max}")
    print(f"   • precision_refine_lambda_pde_max_scale = {exp_config.precision_refine_lambda_pde_max_scale}")
    print(f"   • hard_constraint_profile = {exp_config.hard_constraint_profile}")
    print(f"   • epochs = {exp_config.epochs}")
    print(f"   • batch_size = {exp_config.batch_size}")
    print(f"   • hard_constraint_bc = {exp_config.hard_constraint_bc}")
    
    # Verificar que as intervenções estão ativas
    print(f"\n✅ Verificações de Intervenção:")
    checks = [
        ("Int. 1", exp_config.residual_scale_reference == 0.04),
        ("Int. 2a", exp_config.lambda_pde_max == 150.0),
        ("Int. 2b", exp_config.precision_refine_lambda_pde_max_scale == 0.80),
        ("Int. 4", exp_config.hard_constraint_profile == "tanh"),
    ]
    
    all_ok = True
    for name, check in checks:
        status = "✓" if check else "✗"
        print(f"   [{status}] {name}")
        if not check:
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ AVISO: Algumas intervenções não estão ativas!")
        print("   Verifique src/config.py")
    
    # ============ Pipeline ============
    print("\n[3/3] Inicializando pipeline...")
    try:
        pipeline = PIGANPipeline(exp_config, sys_config)
        print("✅ Pipeline inicializado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar pipeline: {e}")
        return 1
    
    # ============ Treinamento ============
    print("\n" + "="*80)
    print(f"🎯 INICIANDO TREINAMENTO POR {args.epochs} ÉPOCAS")
    print("="*80)
    print("\n📈 Métricas a monitorar:")
    print("   • pde_residual_max (esperado: recuar ~40%)")
    print("   • pde_residual_mean (esperado: ~0.04)")
    print("   • boundary_error (esperado: < 1e-6)")
    print("\n⏱️ Pressione Ctrl+C para interromper\n")
    
    try:
        results = pipeline.run()
        
        print("\n" + "="*80)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO")
        print("="*80)
        
        if results:
            print("\n📊 Resultados Finais:")
            for key, value in sorted(results.items()):
                if isinstance(value, float):
                    print(f"   {key}: {value:.6f}")
                else:
                    print(f"   {key}: {value}")
        
        print("\n💾 Artefatos salvos em:")
        print(f"   • Checkpoints: {pipeline.checkpoint_dir}")
        print(f"   • Plots: {pipeline.plots_dir}")
        print(f"   • Logs: {pipeline.logs_dir}")
        print(f"   • Resultados: {pipeline.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Treinamento interrompido pelo usuário")
        return 1
    except Exception as e:
        print(f"\n❌ Erro durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
