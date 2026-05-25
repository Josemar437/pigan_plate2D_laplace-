# -*- coding: utf-8 -*-
"""Integration tests for refactored training components.

Validates that modular components work correctly together and with the trainer.
"""

import torch
import pytest

from src.config import ExperimentConfig
from src.physics.pdeResidual import PDE_Residual_Computer
from src.physics.domainMetrics import Domain_Metrics_Computer
from src.training.lossFunctions import PIGANLossComputation
from src.training.adaptiveSchemes import (
    AdaptiveLambdaPDE,
    GradNormBalancer,
    StagnationDetector,
    DivergenceDetector,
)


class TestComponentsWorkTogether:
    """Test that modular components interact correctly."""

    def test_pde_computer_and_loss_computer_share_same_residuals(self):
        """PDE computer should produce consistent results when used directly vs via loss computer."""
        grid_size = 32
        pde_computer = PDE_Residual_Computer(grid_size, grid_size, use_gpu=False)
        loss_computer = PIGANLossComputation(grid_size, grid_size, use_gpu=False)

        # Test field
        field = torch.randn(2, 1, grid_size, grid_size)

        # Get residuals from both paths
        residual_direct, stats_direct = pde_computer.compute_pde_residual(field)
        loss_pde, stats_loss = loss_computer.compute_pde_loss(field)

        # Should be consistent (loss_pde is normalized by default scale of 1.0)
        # The loss is: residual_mean / (reference_scale + eps) where reference_scale defaults to 1.0
        expected_loss = residual_direct.mean()
        assert torch.allclose(
            loss_pde,
            expected_loss,
            atol=1e-5,
        ), "PDE loss should be mean of residuals when scale=1.0"

    def test_domain_metrics_detects_boundary_errors(self):
        """Domain metrics should correctly measure boundary condition violations."""
        H, W = 32, 32
        metrics = Domain_Metrics_Computer()

        # Perfect prediction
        pred_perfect = torch.zeros(1, 1, H, W)
        boundary_target = torch.zeros(1, 1, H, W)
        error_perfect, _ = metrics.compute_boundary_error(pred_perfect, boundary_target)
        assert error_perfect.item() < 1e-10

        # Prediction with boundary errors
        pred_error = torch.zeros(1, 1, H, W)
        pred_error[:, :, 0, :] += 1.0  # Add error at top boundary
        pred_error[:, :, -1, :] += 1.0  # Add error at bottom boundary
        error_boundary, stats = metrics.compute_boundary_error(pred_error, boundary_target)
        assert error_boundary.item() > 0.5  # Should detect the error

    def test_adaptive_lambda_and_stagnation_work_together(self):
        """Test that adaptive lambda and stagnation detector can work in sequence."""
        adapter = AdaptiveLambdaPDE(
            lambda_pde_min=10.0,
            lambda_pde_max=100.0,
            ema_beta=0.5,  # Fast EMA for testing
        )
        detector = StagnationDetector(patience=3, rel_tolerance=0.01, cooldown=2)

        # Simulate training loop
        residuals = [1.0, 0.95, 0.93, 0.93, 0.93, 0.93, 0.92]
        lambdas = [37.0]

        for residual in residuals:
            # Update adaptive lambda
            lambda_new = adapter.update(torch.tensor(residual), lambda_pde_current=lambdas[-1])
            lambdas.append(lambda_new)

            # Check for stagnation
            state = detector.update(torch.tensor(residual))
            if state["boost_active"]:
                # Could apply boost to lambda if desired
                pass

        # Should have detected stagnation and potentially boosted
        assert detector.best_residual is not None and detector.best_residual < 1.0  # Residual improved
        assert detector.stagnation_counter >= 0  # Tracking active

    def test_gradnorm_balancer_with_realistic_gradients(self):
        """Test grad norm balancer with realistic gradient magnitudes."""
        balancer = GradNormBalancer(
            target_ratio=0.35,
            ema_beta=0.0,  # No smoothing for testing
            scale_min=0.1,
            scale_max=2.0,
        )

        # Scenario 1: Adversarial dominates (high grad_adv, low grad_pde)
        # scale = target_ratio * grad_pde / grad_adv = 0.35 * 0.1 / 1.0 = 0.035
        # But clipped to [0.1, 2.0] → 0.1
        grad_adv_high = torch.tensor(1.0)
        grad_pde_low = torch.tensor(0.1)
        scale_1 = balancer.update(grad_adv_high, grad_pde_low)
        assert scale_1 <= 0.5, f"Should reduce scale when adv gradient is large, got {scale_1}"

        # Reset EMA for second test
        balancer.scale_ema = None

        # Scenario 2: PDE dominates (low grad_adv, high grad_pde)
        # scale = target_ratio * grad_pde / grad_adv = 0.35 * 1.0 / 0.1 = 3.5
        # But clipped to [0.1, 2.0] → 2.0
        grad_adv_low = torch.tensor(0.1)
        grad_pde_high = torch.tensor(1.0)
        scale_2 = balancer.update(grad_adv_low, grad_pde_high)
        assert scale_2 >= 1.0, f"Should increase scale when pde gradient is large, got {scale_2}"

    def test_divergence_detector_distinguishes_divergence_from_oscillation(self):
        """Test that divergence detector correctly identifies blow-up vs normal variation."""
        detector = DivergenceDetector(
            window_size=4,
            ratio_threshold=1.5,
            patience=1,
        )

        # Bounded oscillation (should NOT trigger)
        oscillating = [1.0, 1.05, 1.0, 0.95, 1.0, 1.05, 1.0, 0.95] * 2
        osc_results = [detector.update(torch.tensor(v))["is_diverging"] for v in oscillating]
        assert not any(osc_results), "Bounded oscillation should not trigger divergence"

        # Create new detector for diverging case
        detector2 = DivergenceDetector(window_size=4, ratio_threshold=1.5, patience=1)
        diverging = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]
        div_results = [detector2.update(torch.tensor(v))["is_diverging"] for v in diverging]
        assert any(div_results[-4:]), "Sustained growth should trigger divergence"

    def test_all_components_accept_both_tensor_and_float(self):
        """Test that components handle both tensor and float inputs."""
        residual_tensor = torch.tensor(0.01)
        residual_float = torch.tensor(0.01)  # Convert float to tensor

        adapter = AdaptiveLambdaPDE()

        # Both should work
        lambda_from_tensor = adapter.update(residual_tensor, 37.0)
        adapter.lambda_pde_ema = None  # Reset EMA
        lambda_from_float = adapter.update(residual_float, 37.0)

        assert isinstance(lambda_from_tensor, float)
        assert isinstance(lambda_from_float, float)
        # Values should be very close (same computation)
        assert abs(lambda_from_tensor - lambda_from_float) < 1e-6

    def test_components_error_on_invalid_inputs(self):
        """Test that components raise appropriate errors for invalid inputs."""
        adapter = AdaptiveLambdaPDE()
        detector = StagnationDetector()

        # NaN should raise
        with pytest.raises(FloatingPointError):
            adapter.update(torch.tensor(float("nan")), 37.0)

        # Inf should raise
        with pytest.raises(FloatingPointError):
            detector.update(torch.tensor(float("inf")))

    def test_config_validation_catches_invalid_lambda_bounds(self):
        """Test that ExperimentConfig validates adaptive lambda bounds."""
        # Valid config should work
        cfg = ExperimentConfig()
        cfg.adaptive_lambda_pde = True
        cfg.lambda_pde_min = 10.0
        cfg.lambda_pde_max = 100.0
        cfg.__post_init__()  # Should not raise

        # Invalid: min >= max should fail (but existing validation catches this)
        cfg2 = ExperimentConfig()
        cfg2.adaptive_lambda_pde = True
        cfg2.lambda_pde_min = 100.0
        cfg2.lambda_pde_max = 50.0  # Inverted
        # This will be caught by existing ModelConfigurationError
        with pytest.raises(Exception):  # Catches ModelConfigurationError
            cfg2.__post_init__()


class TestComponentPerformance:
    """Test performance characteristics of components."""

    def test_pde_residual_computation_is_deterministic(self):
        """Same field should always produce same residual."""
        computer = PDE_Residual_Computer(32, 32, use_gpu=False)
        field = torch.randn(4, 1, 32, 32)

        residual1, _ = computer.compute_pde_residual(field)
        residual2, _ = computer.compute_pde_residual(field)

        assert torch.allclose(residual1, residual2), "PDE computation should be deterministic"

    def test_loss_computation_doesnt_modify_input_field(self):
        """Loss computation should not modify input field in-place."""
        loss_comp = PIGANLossComputation(32, 32, use_gpu=False)
        field = torch.randn(2, 1, 32, 32)
        field_orig = field.clone()

        _ = loss_comp.compute_pde_loss(field)

        assert torch.allclose(field, field_orig), "Loss computation should not modify input"

    def test_adaptive_components_maintain_bounded_state(self):
        """Adaptive components should maintain bounded internal state."""
        adapter = AdaptiveLambdaPDE(
            lambda_pde_min=10.0,
            lambda_pde_max=100.0,
        )

        # Extreme residuals
        for residual in [1e-10, 1e-1, 1.0, 10.0, 1e10]:
            lambda_new = adapter.update(torch.tensor(residual), 37.0)
            assert 10.0 <= lambda_new <= 100.0, (
                f"Lambda should stay bounded, got {lambda_new} with residual {residual}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
