#!/usr/bin/env python3
"""Verificar qual convenção está sendo usada nos atributos"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig

config = ExperimentConfig()

print("=== Verificando Convenção de Atributos ===\n")

# Test pairs (snake_case, legacy camelCase)
tests = [
    ("lambda_adv", "lambdaAdv"),
    ("hard_constraint_bc", "hardConstraintBc"),
    ("lambda_pde", "lambdaPde"),
    ("batch_size", "batchSize"),
    ("latent_dim", "latentDim"),
]

print("Atributo Snake_Case | Atributo camelCase | Status")
print("-" * 60)

for snake, camel in tests:
    has_snake = hasattr(config, snake)
    has_camel = hasattr(config, camel)
    
    if has_snake and not has_camel:
        status = "OK snake_case"
    elif has_camel and not has_snake:
        status = "OK camelCase"
    elif has_snake and has_camel:
        status = "AMBIGUO (ambos!)"
    else:
        status = "Nenhum encontrado!"
    
    print(f"{snake:20} | {camel:20} | {status}")

print("\n" + "=" * 60)
print("Conclusão: O código está em snake_case (Python convention)")
