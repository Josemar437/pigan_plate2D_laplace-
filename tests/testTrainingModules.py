"""Testes unitários para módulos de treinamento refatorados."""

import torch
import pytest

from src.physics.pdeResidual import PDE_Residual_Computer
from src.physics.domainMetrics import Domain_Metrics_Computer
from src.training.lossFunctions import PIGANLossComputation
from src.training.adaptiveSchemes import (
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

    def test_laplacian_matches_discrete_polynomial_identities(self):
        """Prova identidades do stencil: Δ_h constante/linear=0 e |Δ_h(i²+j²)|=4."""
        computer = PDE_Residual_Computer(9, 9, use_gpu=False)

        i = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9, 1)
        j = torch.arange(9, dtype=torch.float32).reshape(1, 1, 1, 9)

        linear = 2.0 * i - 3.0 * j + 7.0
        linear_residual, _ = computer.compute_pde_residual(linear)
        assert torch.allclose(linear_residual[:, :, 1:-1, 1:-1], torch.zeros(1, 1, 7, 7), atol=1e-6)

        quadratic = i.pow(2) + j.pow(2)
        quadratic_residual, _ = computer.compute_pde_residual(quadratic)
        expected = torch.full((1, 1, 7, 7), 4.0)
        assert torch.allclose(quadratic_residual[:, :, 1:-1, 1:-1], expected, atol=1e-6)

    def test_signed_laplacian_uses_positive_convention(self):
        """Δ_h(i²+j²) deve ser +4 no interior, igual ao LaplacianLayer usado no treino."""
        computer = PDE_Residual_Computer(9, 9, use_gpu=False)
        i = torch.arange(9, dtype=torch.float32).reshape(1, 1, 9, 1)
        j = torch.arange(9, dtype=torch.float32).reshape(1, 1, 1, 9)

        quadratic = i.pow(2) + j.pow(2)
        signed_residual, _ = computer.compute_pde_residual(quadratic, use_abs=False)

        expected = torch.full((1, 1, 7, 7), 4.0)
        assert torch.allclose(signed_residual[:, :, 1:-1, 1:-1], expected, atol=1e-6)
        assert torch.allclose(signed_residual[:, :, 0, :], torch.zeros(1, 1, 9), atol=1e-6)
    
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

    def test_smoothness_order2_runs_on_common_interior_grid(self):
        """Segunda derivada combina dx²/dy² com shapes compatíveis."""
        computer = Domain_Metrics_Computer()
        pred = torch.randn(2, 1, 8, 7)

        smoothness, stats = computer.compute_smoothness_metrics(pred, order=2)

        assert smoothness.shape == torch.Size([])
        assert torch.isfinite(smoothness)
        assert "laplacian_mean" in stats

    def test_reference_error_r2_uses_same_mask_as_residual(self):
        """R² deve usar a mesma região mascarada no numerador e denominador."""
        computer = Domain_Metrics_Computer()
        reference = torch.zeros(1, 1, 4, 4)
        reference[:, :, 1, 1] = 1.0
        reference[:, :, 1, 2] = 3.0
        prediction = reference.clone()
        prediction[:, :, 1, 1] += 0.5
        prediction[:, :, 1, 2] -= 0.5
        mask = torch.zeros_like(reference, dtype=torch.bool)
        mask[:, :, 1, 1] = True
        mask[:, :, 1, 2] = True

        _, stats = computer.compute_reference_error(prediction, reference, mask_interior=mask)

        expected_ss_res = torch.tensor(0.5)
        expected_ss_tot = torch.tensor(2.0)
        assert stats["r2_score"].item() == pytest.approx(
            (1.0 - expected_ss_res / expected_ss_tot).item(),
            abs=1e-6,
        )


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
        loss_bc = torch.tensor(0.25)
        loss_diversity = torch.tensor(-2.0)
        
        total_loss, breakdown = computer.compose_generator_loss(
            loss_pde,
            loss_adv,
            loss_bc=loss_bc,
            loss_diversity=loss_diversity,
            lambda_pde=37.0,
            lambda_adv=0.005,
            lambda_bc=2.0,
            lambda_diversity=0.1,
        )
        
        expected = 37.0 * loss_pde + 0.005 * loss_adv + 2.0 * loss_bc + 0.1 * loss_diversity
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() == pytest.approx(expected.item(), abs=1e-7)
        assert "loss_total" in breakdown
        assert "loss_pde_term" in breakdown


class TestAdaptiveLambdaPDE:
    """Testes para AdaptiveLambdaPDE."""
    
    def test_initialization(self):
        """Teste inicialização."""
        adapter = AdaptiveLambdaPDE()
        assert adapter.lambda_pde_min == 50.0  # Corrigido (CORR 24/05)
        assert adapter.lambda_pde_max == 500.0  # Corrigido (CORR 24/05)
    
    def test_update_increases_lambda(self):
        """Teste que λ_PDE aumenta quando resíduo é alto."""
        adapter = AdaptiveLambdaPDE(ema_beta=0.0)
        
        # Resíduo alto → λ deve aumentar
        residual_high = torch.tensor(1.0)
        lambda_new = adapter.update(residual_high, lambda_pde_current=20.0)
        
        # Nova λ deve ser maior que inicial
        assert lambda_new > 20.0

    def test_update_is_anchored_to_fixed_base(self):
        """λ_PDE deve ser recalculado a partir da base fixa, não do EMA anterior."""
        adapter = AdaptiveLambdaPDE(
            ema_beta=0.0,
            residual_scale_reference=1.0,
            growth_exponent=0.6,
        )

        first = adapter.update(torch.tensor(10.0), lambda_pde_current=37.0)
        second = adapter.update(torch.tensor(10.0), lambda_pde_current=37.0)

        assert first == pytest.approx(59.2, abs=1e-6)
        assert second == pytest.approx(first, abs=1e-6)

    def test_update_rejects_nonfinite_residual(self):
        """NaN/Inf deve falhar explicitamente em vez de contaminar EMA."""
        adapter = AdaptiveLambdaPDE()

        with pytest.raises(FloatingPointError):
            adapter.update(torch.tensor(float("nan")), lambda_pde_current=37.0)
    
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

    def test_boost_keeps_full_cooldown_and_resets_stagnation_counter(self):
        """Boost não deve consumir um passo de cooldown no mesmo update."""
        detector = StagnationDetector(patience=2, rel_tolerance=0.01, cooldown=5)

        detector.update(torch.tensor(1.0))
        detector.update(torch.tensor(1.0))
        result = detector.update(torch.tensor(1.0))

        assert result["boost_active"] is True
        assert detector.boost_counter == 5
        assert detector.stagnation_counter == 0


class TestDivergenceDetector:
    """Testes para DivergenceDetector."""
    
    def test_initialization(self):
        """Teste inicialização."""
        detector = DivergenceDetector(window_size=16)
        assert detector.window_size == 16
    
    def test_detects_divergence(self):
        """Teste detecção de divergência."""
        detector = DivergenceDetector(window_size=3, ratio_threshold=1.5, patience=1)
        
        # Simular loss divergindo
        losses = [1.0, 1.0, 1.0, 2.5, 2.5, 2.5]  # Salto 2.5x
        
        results = []
        for loss in losses:
            result = detector.update(torch.tensor(loss))
            results.append(result)
        
        # Após salto, deve detectar divergência
        assert any(r["is_diverging"] for r in results[-2:])

    def test_oscillation_is_not_classified_as_divergence(self):
        """Oscilação bounded não deve ser tratada como blow-up monotônico."""
        detector = DivergenceDetector(window_size=8, ratio_threshold=1.2, patience=2)
        losses = [
            1.0,
            1.07,
            1.10,
            1.07,
            1.0,
            0.93,
            0.90,
            0.93,
        ] * 4

        results = [detector.update(torch.tensor(loss)) for loss in losses]

        assert not any(result["is_diverging"] for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

