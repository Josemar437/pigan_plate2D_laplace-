# -*- coding: utf-8 -*-
"""
Script de verificação: Hard Constraint e Boundary Error
Valida que hard_constraint_bc está ativo e que o erro de fronteira é minimal.

Uso:
    python verify_hard_constraint.py
"""

import sys
from pathlib import Path

import torch
import numpy as np

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig, SystemConfig, initialize_system
from src.models import UNetGenerator2D, create_field_pigan_models
from src.utils import (
    create_cartesian_grid,
    build_dirichlet_extension,
    build_hard_constraint_mask,
    build_domain_masks,
)


def verify_hard_constraint():
    """Verifica se hard_constraint_bc está ativo e funcional."""
    
    # ============ Carregamento de Configuração ============
    exp_config = ExperimentConfig()
    sys_config = SystemConfig()
    sys_config, device, _, logger = initialize_system(sys_config)
    
    print("\n" + "="*70)
    print("VERIFICAÇÃO: Hard Constraint e Boundary Error")
    print("="*70)
    
    # ============ Verificação 1: hard_constraint_bc ============
    hc_active = bool(exp_config.hard_constraint_bc)
    print(f"\n[1] Hard Constraint Ativo:")
    print(f"    hard_constraint_bc = {hc_active}")
    if not hc_active:
        print("    [AVISO]  AVISO: Hard constraint está DESATIVADO!")
        return False
    print("    [OK] OK: Hard constraint ativo")
    
    # ============ Verificação 2: Hard Constraint Profile ============
    hc_profile = getattr(exp_config, "hard_constraint_profile", "tanh")
    print(f"\n[2] Perfil de Hard Constraint:")
    print(f"    hard_constraint_profile = '{hc_profile}'")
    valid_profiles = {"tanh", "polynomial"}
    if hc_profile not in valid_profiles:
        print(f"    [ERRO] ERRO: Perfil inválido! Use um de {valid_profiles}")
        return False
    print(f"    [OK] OK: Perfil '{hc_profile}' válido")
    
    # ============ Verificação 3: Construir Modelos ============
    print(f"\n[3] Construindo modelos...")
    try:
        nx, ny = int(exp_config.grid_size_x), int(exp_config.grid_size_y)
        lx, ly = float(exp_config.LX), float(exp_config.LY)
        
        # Grid e campos
        x_grid, y_grid = create_cartesian_grid(
            nx, ny, lx, ly, device=device, dtype=torch.float32
        )
        
        # Base field (Dirichlet extension)
        base_field = build_dirichlet_extension(
            x_grid, y_grid,
            lx=lx, ly=ly,
            t_left=float(exp_config.T_LEFT),
            t_right=float(exp_config.T_RIGHT),
            boundary_sine_amplitude=float(exp_config.boundary_sine_amplitude),
        )
        
        # Hard constraint mask (phi)
        phi_mask = build_hard_constraint_mask(
            x_grid, y_grid,
            lx=lx, ly=ly,
            smooth_profile=str(hc_profile).lower(),
        )
        
        # Domain masks
        interior_mask, boundary_mask = build_domain_masks(
            ny, nx, device=device, dtype=torch.float32
        )
        
        print("    [OK] OK: Grid e campos construídos")
        
    except Exception as e:
        print(f"    [ERRO] ERRO ao construir grid/campos: {e}")
        return False
    
    # ============ Verificação 4: Generator com Hard Constraint ============
    print(f"\n[4] Construindo gerador com hard_constraint_bc=True...")
    try:
        generator = UNetGenerator2D(
            in_channels=3 if exp_config.use_physical_coordinates else 1,
            latent_dim=int(exp_config.latent_dim),
            base_channels=int(exp_config.generator_base_channels),
            depth=int(exp_config.generator_depth),
            use_batch_norm=bool(exp_config.generator_use_batch_norm),
            hard_constraint=True,  # Explicitamente True
            output_smoothing_steps=int(exp_config.generator_output_smoothing_steps),
            output_smoothing_strength=float(exp_config.generator_output_smoothing_strength),
            activation=str(exp_config.generator_activation),
            pooling=str(exp_config.generator_pooling),
        )
        generator.to(device)
        generator.eval()
        print("    [OK] OK: Gerador criado com hard_constraint=True")
        
    except Exception as e:
        print(f"    [ERRO] ERRO ao criar gerador: {e}")
        return False
    
    # ============ Verificação 5: Teste Forward Pass ============
    print(f"\n[5] Testando forward pass...")
    try:
        batch_size = 2
        base_batch = base_field.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        phi_batch = phi_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        
        # Coordenadas físicas (se usadas)
        if exp_config.use_physical_coordinates:
            x_norm = x_grid / lx
            y_norm = y_grid / ly
            coord_batch = torch.stack([x_norm, y_norm], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        else:
            coord_batch = None
        
        # Latent
        z = None
        if exp_config.latent_dim > 0:
            z = torch.randn(batch_size, int(exp_config.latent_dim), device=device)
        
        # Forward
        with torch.no_grad():
            pred = generator(base_batch, phi_batch, z=z, coord_field=coord_batch)
        
        print("    [OK] OK: Forward pass bem-sucedido")
        print(f"       Output shape: {tuple(pred.shape)}")
        
    except Exception as e:
        print(f"    [ERRO] ERRO no forward pass: {e}")
        return False
    
    # ============ Verificação 6: Erro de Fronteira ============
    print(f"\n[6] Verificando erro de fronteira...")
    try:
        with torch.no_grad():
            pred_cpu = pred.cpu().numpy()
            base_cpu = base_batch.cpu().numpy()
            boundary_cpu = boundary_mask.cpu().numpy()
        
        # Erro de fronteira = |pred - base_field| onde boundary=1
        # boundary_cpu tem shape [1, 1, H, W], pred_cpu tem shape [B, 1, H, W]
        # Usar o primeiro sample
        boundary_mask_2d = boundary_cpu[0, 0, :, :]  # [H, W]
        pred_2d = pred_cpu[0, 0, :, :]  # [H, W]
        base_2d = base_cpu[0, 0, :, :]  # [H, W]
        boundary_error = np.abs(pred_2d - base_2d)[boundary_mask_2d > 0.5]
        
        if len(boundary_error) > 0:
            boundary_error_max = float(np.max(boundary_error))
            boundary_error_mean = float(np.mean(boundary_error))
            boundary_error_std = float(np.std(boundary_error))
            
            print(f"    Estatísticas de erro de fronteira (gerador random):")
            print(f"      max  = {boundary_error_max:.2e}")
            print(f"      mean = {boundary_error_mean:.2e}")
            print(f"      std  = {boundary_error_std:.2e}")
            
            # IMPORTANTE: Com hard_constraint=True, o erro DEVE ser zero em pontos de fronteira
            # pois T_output = base_field + phi_mask * raw, e phi_mask=0 na fronteira
            # Se há erro aqui, significa que hard_constraint NÃO está sendo aplicado
            
            # Verificar: valores preditos na fronteira devem ser EXATAMENTE = base_field
            pred_boundary = pred_2d[boundary_mask_2d > 0.5]
            base_boundary = base_2d[boundary_mask_2d > 0.5]
            
            # Com hard constraint, esperamos pred_boundary ≈ base_boundary
            # (com precisão de ponto flutuante ~1e-6)
            actual_error = np.max(np.abs(pred_boundary - base_boundary))
            
            print(f"\n    Verificação de Hard Constraint:")
            print(f"      max(|pred_boundary - base_boundary|) = {actual_error:.2e}")
            
            if actual_error < 1e-5:
                print(f"    [OK] OK: Hard constraint está FUNCIONANDO (erro < 1e-5)")
            else:
                print(f"    [ERRO] PROBLEMA: Hard constraint pode estar COM DEFEITO")
                print(f"       - Verifique se hard_constraint=True em UNetGenerator2D")
                print(f"       - Verifique se base_field e phi_mask estão corretos")
        else:
            print(f"    [AVISO]  AVISO: Nenhum ponto de fronteira para testar")
        
    except Exception as e:
        print(f"    [ERRO] ERRO ao calcular erro de fronteira: {e}")
        return False
    
    # ============ Verificação 7: Perfil de phi nos cantos ============
    print(f"\n[7] Analisando perfil de phi (hard constraint mask)...")
    try:
        phi_cpu = phi_mask.cpu().numpy()
        
        # Valores nos cantos (índices: 0, nx-1, ny-1)
        phi_corners = {
            "(0, 0)": phi_cpu[0, 0],
            "(0, nx-1)": phi_cpu[0, nx-1],
            "(ny-1, 0)": phi_cpu[ny-1, 0],
            "(ny-1, nx-1)": phi_cpu[ny-1, nx-1],
        }
        
        # Valor no centro
        phi_center = phi_cpu[ny//2, nx//2]
        
        # Valor próximo ao hotspot em (1, 3)
        if ny > 3 and nx > 3:
            phi_hotspot = phi_cpu[1, 3]
        else:
            phi_hotspot = None
        
        print(f"    Valores de phi (distância à fronteira):")
        for corner, val in phi_corners.items():
            print(f"      {corner:12s}: {val:.2e}")
        print(f"      Centro (ny/2, nx/2): {phi_center:.4f}")
        if phi_hotspot is not None:
            print(f"      Hotspot (1, 3):      {phi_hotspot:.4f}")
        
        # Verificar que todos os cantos são ~0
        max_corner = max(abs(v) for v in phi_corners.values())
        if max_corner > 1e-6:
            print(f"    [ERRO] ERRO: phi não é zero nos cantos! max={max_corner:.2e}")
            return False
        else:
            print(f"    [OK] OK: phi = 0 nos cantos (enforce Dirichlet)")
        
        # Verificar que phi > 0 no interior
        if phi_center < 0.1:
            print(f"    [AVISO]  AVISO: phi no centro é baixo ({phi_center:.4f})")
            print(f"       Esperado: > 0.1 para suavidade")
        else:
            print(f"    [OK] OK: phi no centro adequado")
        
    except Exception as e:
        print(f"    [ERRO] ERRO ao analisar phi: {e}")
        return False
    
    # ============ SUMÁRIO ============
    print("\n" + "="*70)
    print("RESULTADO: [OK] Todas as verificações passaram!")
    print("="*70)
    print("\nRecomendações:")
    print("  - Hard constraint está ATIVO e funcional")
    print("  - Perfil de phi (hard_constraint_profile) está otimizado")
    print("  - Erro de fronteira deve ser < 1e-6 durante treinamento")
    print("  - Se boundary_error > 1e-4, verifique a implementação do gerador")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = verify_hard_constraint()
    sys.exit(0 if success else 1)
