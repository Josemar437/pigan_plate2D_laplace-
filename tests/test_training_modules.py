"""Testes unitários para módulos de treinamento refatorados."""
import sys
from pathlib import Path

import torch
import pytest

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.pde_residual import PDE_Residual_Computer
from src.physics.domain_metrics import Domain_Metrics_Computer
from src.training.loss_functions import PIGANLossComputation
from src.training.adaptive_schemes import (
    AdaptiveLambdaPDE,
    GradNormBalancer,
    StagnationDetector,
    DivergenceDetector,
)


class TestPDEResidualComputer:
    """Testes para PDE_Residual_Computer."""
    
    def test_initialization(self):
        """Teste inicialização do computador PDE."""
        computer = PDE_Residual_Computer(
            grid_size_x=32,
            grid_size_y=32,
            use_gpu=False,
        )
        assert computer.grid_size_x == 32
        assert computer.grid_size_y == 32
        assert computer.laplacian_kernel is not None
    
    def test_laplacian_computation(self):
        """Teste computação do Laplaciano."""
        computer = PDE_Residual_Computer(32, 32, use_gpu=False)
        
        # Campo constante → Laplaciano = 0
        field_const = torch.ones(1, 1, 32, 32)
        lap_const, _ = computer.compute_pde_residual(field_const)
        assert torch.allclose(lap_const, torch.zeros_like(lap_const), atol=1e-4)
    
    def test_pde_residual_stats(self):
        """Teste estatísticas de resíduo PDE."""
        computer = PDE_Residual_Computer(32, 32, use_gpu=False)
        
        # Campo aleatório
        field_random = torch.randn(2, 1, 32, 32)
        residual, stats = computer.compute_pde_residual(field_random)
        
        assert "residual_mean_abs" in stats
        assert "residual_max_abs" in stats
        assert "residual_std" in stats
        assert stats["residual_mean_abs"].shape == (2, 1, 1, 1)


class TestDomainMetricsComputer:
    """Testes para Domain_Metrics_Computer."""
    
    def test_initialization(self):
        """Teste inicialização do computador de métricas."""
        computer = Domain_Metrics_Computer(device=torch.device("cpu"))
        assert computer.device.type == "cpu"
    
    def test_boundary_error(self):
        """Teste computação de erro de contorno."""
        computer = Domain_Metrics_Computer()
        
        # Predição perfeita na fronteira
        pred = torch.zeros(1, 1, 32, 32)
        boundary_target = torch.zeros(1, 1, 32, 32)
        
        error, stats = computer.compute_boundary_error(pred, boundary_target)
        assert error.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_smoothness_metrics(self):
        """Teste métricas de suavidade."""
        computer = Domain_Metrics_Computer()
        
        # Campo suave (linear)
        x = torch.linspace(0, 1, 32).reshape(1, 1, 1, 32).expand(1, 1, 32, 32)
        smoothness, stats = computer.compute_smoothness_metrics(x, order=1)
        
        assert smoothness.item() > 0
        assert "grad_x_mean" in stats


class TestPIGANLossComputation:
    """Testes para PIGANLossComputation."""
    
    def test_initialization(self):
        """Teste inicialização do computador de perdas."""
        computer = PIGANLossComputation(
            grid_size_x=32,
            grid_size_y=32,
            use_gpu=False,
        )
        assert computer.grid_size_x == 32
        assert computer.pde_computer is not None
    
    def test_pde_loss(self):
        """Teste computação de perda PDE."""
        computer = PIGANLossComputation(32, 32, use_gpu=False)
        
        # Campo com ruído
        field = torch.randn(2, 1, 32, 32)
        loss, stats = computer.compute_pde_loss(field)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert "pde_loss_raw" in stats
    
    def test_adversarial_loss(self):
        """Teste computação de perda adversarial."""
        computer = PIGANLossComputation(32, 32, use_gpu=False)
        
        # Scores do discriminador
        d_fake = torch.randn(16)
        d_real = torch.randn(16)
        
        loss, stats = computer.compute_adversarial_loss(
            d_fake, d_real, loss_type="wasserstein"
        )
        
        assert loss.shape == torch.Size([])
        assert "d_gap" in stats
    
    def test_boundary_loss(self):
        """Teste computação de perda de contorno."""
        computer = PIGANLossComputation(32, 32, use_gpu=False)
        
        # Predição com erro de contorno
        pred = torch.randn(1, 1, 32, 32)
        boundary_target = torch.zeros(1, 1, 32, 32)
        
        loss, stats = computer.compute_boundary_loss(
            pred, boundary_target, loss_type="l1"
        )
        
        assert loss.item() > 0
        assert "bc_mae" in stats
    
    def test_compose_generator_loss(self):
        """Teste composição de perda total."""
        computer = PIGANLossComputation(32, 32, use_gpu=False)
        
        loss_pde = torch.tensor(0.5)
        loss_adv = torch.tensor(0.1)
        
        total_loss, breakdown = computer.compose_generator_loss(
            loss_pde, loss_adv,
            lambda_pde=37.0,
            lambda_adv=0.005,
        )
        
        assert total_loss.shape == torch.Size([])
        assert "loss_total" in breakdown
        assert "loss_pde_term" in breakdown


class TestAdaptiveLambdaPDE:
    """Testes para AdaptiveLambdaPDE."""
    
    def test_initialization(self):
        """Teste inicialização."""
        adapter = AdaptiveLambdaPDE()
        assert adapter.lambda_pde_min == 19.0
        assert adapter.lambda_pde_max == 98.0
    
    def test_update_increases_lambda(self):
        """Teste que λ_PDE aumenta quando resíduo é alto."""
        adapter = AdaptiveLambdaPDE()
        
        # Resíduo alto → λ deve aumentar
        residual_high = torch.tensor(1.0)
        lambda_new = adapter.update(residual_high, lambda_pde_current=20.0)
        
        # Nova λ deve ser maior que inicial
        assert lambda_new > 20.0
    
    def test_update_clips_to_bounds(self):
        """Teste que λ_PDE fica dentro de limites."""
        adapter = AdaptiveLambdaPDE(lambda_pde_min=10.0, lambda_pde_max=50.0)
        
        residual_extreme = torch.tensor(100.0)
        lambda_new = adapter.update(residual_extreme, lambda_pde_current=30.0)
        
        assert 10.0 <= lambda_new <= 50.0


class TestGradNormBalancer:
    """Testes para GradNormBalancer."""
    
    def test_initialization(self):
        """Teste inicialização."""
        balancer = GradNormBalancer(target_ratio=0.35)
        assert balancer.target_ratio == 0.35
    
    def test_update_decreases_scale_when_adv_large(self):
        """Teste que escala diminui quando gradiente adversarial é grande."""
        balancer = GradNormBalancer(target_ratio=0.35)
        
        # Gradiente adversarial >> gradiente PDE
        grad_adv = torch.tensor(1.0)
        grad_pde = torch.tensor(0.1)
        
        scale_new = balancer.update(grad_adv, grad_pde, scale_current=1.0)
        
        # Escala deve diminuir para reduzir razão
        assert scale_new < 1.0


class TestStagnationDetector:
    """Testes para StagnationDetector."""
    
    def test_initialization(self):
        """Teste inicialização."""
        detector = StagnationDetector(patience=50)
        assert detector.patience == 50
    
    def test_detects_stagnation(self):
        """Teste detecção de estagnação."""
        detector = StagnationDetector(patience=3, rel_tolerance=0.01)
        
        # Simular resíduo estagnado
        results = []
        for i in range(5):
            result = detector.update(torch.tensor(1.0))
            results.append(result)
        
        # Após 3 iterações sem melhora, deve detectar
        assert any(r["is_stagnant"] for r in results[3:])


class TestDivergenceDetector:
    """Testes para DivergenceDetector."""
    
    def test_initialization(self):
        """Teste inicialização."""
        detector = DivergenceDetector(window_size=16)
        assert detector.window_size == 16
    
    def test_detects_divergence(self):
        """Teste detecção de divergência."""
        detector = DivergenceDetector(ratio_threshold=1.5, patience=1)
        
        # Simular loss divergindo
        losses = [1.0, 1.0, 1.0, 2.5, 2.5, 2.5]  # Salto 2.5x
        
        results = []
        for loss in losses:
            result = detector.update(torch.tensor(loss))
            results.append(result)
        
        # Após salto, deve detectar divergência
        assert any(r["is_diverging"] for r in results[-2:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
