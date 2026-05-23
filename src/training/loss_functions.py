"""Computação centralizada de funções de perda para PI-GAN."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.physics.pde_residual import PDE_Residual_Computer


class PIGANLossComputation:
    """
    Computador centralizado de todas as perdas PI-GAN.
    
    Encapsula lógica de cálculo de:
    - PDE loss (∇²T = 0)
    - Adversarial loss (discriminador)
    - Boundary condition loss (Dirichlet)
    - Diversity loss (estocástico)
    
    Benefícios:
    - Separação de concerns do trainer
    - Testabilidade: cada componente isolável
    - Reutilização em diferentes contextos
    """

    def __init__(
        self,
        grid_size_x: int,
        grid_size_y: int,
        use_gpu: bool = True,
        pde_kernel_type: str = "centered",
    ) -> None:
        """
        Inicializa computador de perdas.
        
        Parâmetros:
            grid_size_x: Dimensão x da malha
            grid_size_y: Dimensão y da malha
            use_gpu: Usar GPU se disponível
            pde_kernel_type: Tipo de kernel para Laplaciano
        """
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        
        # Inicializar computador PDE
        self.pde_computer = PDE_Residual_Computer(
            grid_size_x=grid_size_x,
            grid_size_y=grid_size_y,
            use_gpu=use_gpu,
            kernel_type=pde_kernel_type,
        )
    
    def compute_pde_loss(
        self,
        prediction: torch.Tensor,
        reference_residual_scale: float = 1.0,
        use_abs: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda PDE: ||∇²T||_2 normalizada.
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            reference_residual_scale: Escala de normalização (padrão: 1.0)
            use_abs: Usar valor absoluto do resíduo
        
        Retorno:
            Tupla:
                - loss: Perda escalar
                - stats: Dicionário com estatísticas intermediárias
        """
        residual, pde_stats = self.pde_computer.compute_pde_residual(
            prediction, 
            use_abs=use_abs
        )
        
        # Normalizar por escala de referência
        loss = residual.mean() / (reference_residual_scale + 1e-8)
        
        stats = {
            "pde_loss_raw": residual.mean(),
            "pde_loss_normalized": loss,
        }
        stats.update(pde_stats)
        
        return loss, stats
    
    def compute_adversarial_loss(
        self,
        discriminator_fake: torch.Tensor,
        discriminator_real: Optional[torch.Tensor] = None,
        loss_type: str = "wasserstein",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda adversarial a partir de scores do discriminador.
        
        Parâmetros:
            discriminator_fake: Scores do discriminador para amostras fake (B,)
            discriminator_real: Scores do discriminador para amostras reais (B,)
            loss_type: Tipo de perda ("wasserstein" ou "hinge")
        
        Retorno:
            Tupla:
                - loss: Perda adversarial
                - stats: Estatísticas
        """
        if loss_type == "wasserstein":
            # Perda Wasserstein-GAN: min -E[D(G(z))]
            loss = -discriminator_fake.mean()
        
        elif loss_type == "hinge":
            # Hinge loss: max(0, 1 - D(G(z)))
            loss = torch.nn.functional.relu(1.0 - discriminator_fake).mean()
        
        else:
            raise ValueError(f"loss_type '{loss_type}' não suportado")
        
        stats = {
            "adv_loss": loss,
            "d_fake_mean": discriminator_fake.mean(),
            "d_fake_std": discriminator_fake.std(),
        }
        
        if discriminator_real is not None:
            stats["d_real_mean"] = discriminator_real.mean()
            stats["d_real_std"] = discriminator_real.std()
            stats["d_gap"] = (discriminator_real.mean() - discriminator_fake.mean()).abs()
        
        return loss, stats
    
    def compute_boundary_loss(
        self,
        prediction: torch.Tensor,
        boundary_target: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
        loss_type: str = "l1",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda de contorno Dirichlet.
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            boundary_target: Valores-alvo na fronteira (B, 1, H, W)
            boundary_mask: Máscara de pontos de fronteira (B, 1, H, W)
            loss_type: "l1", "l2" ou "smooth_l1"
        
        Retorno:
            Tupla:
                - loss: Perda de contorno
                - stats: Estatísticas
        """
        if boundary_mask is None:
            # Máscara padrão: borda
            B, C, H, W = prediction.shape
            boundary_mask = torch.zeros_like(prediction, dtype=torch.bool)
            boundary_mask[:, :, 0, :] = True    # Top
            boundary_mask[:, :, -1, :] = True   # Bottom
            boundary_mask[:, :, :, 0] = True    # Left
            boundary_mask[:, :, :, -1] = True   # Right
        
        # Diferença apenas na fronteira
        diff = prediction - boundary_target
        diff_boundary = diff[boundary_mask]
        
        if loss_type == "l1":
            loss = torch.abs(diff_boundary).mean()
        elif loss_type == "l2":
            loss = (diff_boundary ** 2).mean()
        elif loss_type == "smooth_l1":
            loss = torch.nn.functional.smooth_l1_loss(
                prediction[boundary_mask],
                boundary_target[boundary_mask],
                reduction="mean"
            )
        else:
            raise ValueError(f"loss_type '{loss_type}' não suportado")
        
        stats = {
            "bc_loss": loss,
            "bc_mae": torch.abs(diff_boundary).mean(),
            "bc_max_error": torch.abs(diff_boundary).max(),
        }
        
        return loss, stats
    
    def compute_diversity_loss(
        self,
        latent_samples: torch.Tensor,
        predictions: torch.Tensor,
        loss_type: str = "variance",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda de diversidade para modo estocástico.
        
        Penaliza output pouco variado com diferentes inputs latentes.
        
        Parâmetros:
            latent_samples: Amostras latentes (N, latent_dim)
            predictions: Predições correspondentes (N, 1, H, W)
            loss_type: "variance" ou "pairwise_distance"
        
        Retorno:
            Tupla:
                - loss: Perda de diversidade (negativa = penaliza repetição)
                - stats: Estatísticas
        """
        if loss_type == "variance":
            # Variância das predições em relação à média
            mean_pred = predictions.mean(dim=0, keepdim=True)
            variance = ((predictions - mean_pred) ** 2).mean()
            loss = -variance  # Negativar: quanto maior variance, menor loss
        
        elif loss_type == "pairwise_distance":
            # Distância média entre pares de predições
            # Reshapear para (N, -1)
            N = predictions.shape[0]
            pred_flat = predictions.reshape(N, -1)
            
            # Computar matriz de distâncias
            distances = torch.cdist(pred_flat, pred_flat, p=2)
            # Média excluindo diagonal
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            mean_distance = distances[mask].mean()
            loss = -mean_distance
        
        else:
            raise ValueError(f"loss_type '{loss_type}' não suportado")
        
        stats = {
            "diversity_loss": loss,
            "diversity_magnitude": torch.abs(loss),
        }
        
        return loss, stats
    
    def compose_generator_loss(
        self,
        loss_pde: torch.Tensor,
        loss_adv: torch.Tensor,
        loss_bc: Optional[torch.Tensor] = None,
        loss_diversity: Optional[torch.Tensor] = None,
        lambda_pde: float = 37.0,
        lambda_adv: float = 0.005,
        lambda_bc: float = 0.0,
        lambda_diversity: float = 1e-4,
        apply_gates: bool = False,
        gates: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compõe perda total do gerador com pesos adaptáveis.
        
        Parâmetros:
            loss_pde: Perda PDE
            loss_adv: Perda adversarial
            loss_bc: Perda de contorno (opcional)
            loss_diversity: Perda de diversidade (opcional)
            lambda_pde: Peso para PDE
            lambda_adv: Peso para adversarial
            lambda_bc: Peso para contorno
            lambda_diversity: Peso para diversidade
            apply_gates: Se True, aplicar gates para suprimir términos
            gates: Dicionário com tensores 0-1 para cada termo (opcional)
        
        Retorno:
            Tupla:
                - loss_total: Perda total
                - loss_breakdown: Dicionário detalhando cada componente
        """
        # Inicializar gates se não fornecidos
        if apply_gates and gates is None:
            gates = {
                "pde_gate": torch.tensor(1.0, device=self.device),
                "adv_gate": torch.tensor(1.0, device=self.device),
                "bc_gate": torch.tensor(1.0, device=self.device),
                "diversity_gate": torch.tensor(1.0, device=self.device),
            }
        
        # Compor termos
        pde_term = lambda_pde * loss_pde
        if apply_gates:
            pde_term = pde_term * gates.get("pde_gate", 1.0)
        
        adv_term = lambda_adv * loss_adv
        if apply_gates:
            adv_term = adv_term * gates.get("adv_gate", 1.0)
        
        bc_term = torch.tensor(0.0, device=self.device)
        if loss_bc is not None and lambda_bc > 0:
            bc_term = lambda_bc * loss_bc
            if apply_gates:
                bc_term = bc_term * gates.get("bc_gate", 1.0)
        
        diversity_term = torch.tensor(0.0, device=self.device)
        if loss_diversity is not None and lambda_diversity > 0:
            diversity_term = lambda_diversity * loss_diversity
            if apply_gates:
                diversity_term = diversity_term * gates.get("diversity_gate", 1.0)
        
        # Perda total
        loss_total = pde_term + adv_term + bc_term + diversity_term
        
        loss_breakdown = {
            "loss_total": loss_total,
            "loss_pde_term": pde_term,
            "loss_adv_term": adv_term,
            "loss_bc_term": bc_term,
            "loss_diversity_term": diversity_term,
            "loss_pde_raw": loss_pde,
            "loss_adv_raw": loss_adv,
        }
        
        if loss_bc is not None:
            loss_breakdown["loss_bc_raw"] = loss_bc
        if loss_diversity is not None:
            loss_breakdown["loss_diversity_raw"] = loss_diversity
        
        return loss_total, loss_breakdown
    
    def compute_discriminator_loss(
        self,
        d_real_scores: torch.Tensor,
        d_fake_scores: torch.Tensor,
        gradient_penalty: Optional[torch.Tensor] = None,
        lambda_gp: float = 10.0,
        loss_type: str = "wasserstein",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda do discriminador.
        
        Parâmetros:
            d_real_scores: Scores para amostras reais
            d_fake_scores: Scores para amostras fake
            gradient_penalty: Penalidade de gradiente (opcional, WGAN-GP)
            lambda_gp: Peso da penalidade de gradiente
            loss_type: "wasserstein" ou "hinge"
        
        Retorno:
            Tupla:
                - loss: Perda do discriminador
                - stats: Estatísticas
        """
        if loss_type == "wasserstein":
            # Perda Wasserstein: min E[D(x)] - E[D(G(z))]
            loss = d_fake_scores.mean() - d_real_scores.mean()
        
        elif loss_type == "hinge":
            # Hinge loss
            loss = torch.nn.functional.relu(1.0 - d_real_scores).mean()
            loss += torch.nn.functional.relu(1.0 + d_fake_scores).mean()
        
        else:
            raise ValueError(f"loss_type '{loss_type}' não suportado")
        
        # Adicionar penalidade de gradiente se fornecida
        if gradient_penalty is not None:
            gp_loss = lambda_gp * gradient_penalty
            loss = loss + gp_loss
        else:
            gp_loss = torch.tensor(0.0, device=self.device)
        
        stats = {
            "d_loss": loss,
            "d_loss_base": loss - (lambda_gp * gradient_penalty if gradient_penalty is not None else 0),
            "d_real_mean": d_real_scores.mean(),
            "d_fake_mean": d_fake_scores.mean(),
            "d_gap": (d_real_scores.mean() - d_fake_scores.mean()).abs(),
            "gp_loss": gp_loss,
        }
        
        return loss, stats
