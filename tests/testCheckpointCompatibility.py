"""Teste de compatibilidade de checkpoint após refatoração."""

import torch
import pytest

from src.config import ExperimentConfig
from src.trainer import FieldPIGANTrainer


class TestCheckpointCompatibility:
    """Validar que refatoração não quebra carregamento de checkpoints."""
    
    def test_trainer_config_is_separate_from_experiment_config(self):
        """Teste que FieldTrainerConfig modela apenas o loop de treino."""
        from src.trainer import FieldTrainerConfig

        cfg = FieldTrainerConfig()
        assert not isinstance(cfg, ExperimentConfig)
        assert cfg.grad_clip > 0.0
        assert cfg.lambda_pde > 0.0
    
    def test_experiment_config_has_all_training_params(self):
        """Teste que ExperimentConfig tem todos os parâmetros necessários."""
        cfg = ExperimentConfig()
        
        # Parâmetros críticos do treinamento
        required_attrs = [
            "epochs", "batch_size", "gen_lr", "disc_lr",
            "lambda_pde", "lambda_bc", "lambda_adv1", "lambda_diversity",
            "hard_constraint_bc", "use_reference_discriminator",
            "adaptive_lambda_pde", "gradnorm_balance",
            "divergence_window", "divergence_ratio_threshold",
        ]
        
        for attr in required_attrs:
            assert hasattr(cfg, attr), f"ExperimentConfig missing {attr}"
            value = getattr(cfg, attr)
            assert value is not None, f"{attr} is None"
    
    def test_p1_correction_automatic(self):
        """Teste que P1 (λ_BC condicional) é aplicada automaticamente."""
        import warnings
        
        # Criar config com hard_constraint_bc=True mas lambda_bc > 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            cfg = ExperimentConfig(
                hard_constraint_bc=True,
                lambda_bc=20.0,  # Alto valor proposital
            )
            
            # Verificar que foi aplicada P1
            assert cfg.lambda_bc == 0.0, "P1 não foi aplicada"
            assert len(w) > 0, "Sem warning de P1"
            assert "P1 CORRECTION" in str(w[-1].message)
    
    def test_p2_lambda_adv_and_diversity(self):
        """Teste que λ_adv e λ_diversity estão configurados."""
        cfg = ExperimentConfig()
        
        # P2a: λ_adv (via lambda_adv1)
        assert hasattr(cfg, "lambda_adv1")
        assert 1e-2 <= cfg.lambda_adv1 <= 1e-1, "lambda_adv1 fora da faixa de D1 ativo"
        assert cfg.lambda_adv2 < cfg.lambda_adv1, "D2 deve ser auxiliar, nao dominante"
        
        # Compatibilidade: property lambda_adv
        assert cfg.lambda_adv == cfg.lambda_adv1
        
        # P2b: λ_diversity
        assert hasattr(cfg, "lambda_diversity")
        assert cfg.lambda_diversity == 1.0e-4, "lambda_diversity não é 1e-4"
        
        # P2c: latent_dim para modo estocástico
        if cfg.generator_mode == "stochastic_pigan":
            assert cfg.latent_dim > 0
    
    def test_trainer_initialization_with_new_components(self):
        """Teste que trainer inicializa novos componentes sem erro."""
        pytest.importorskip("torch")
        
        from src.models import UNetGenerator2D, FieldDualDiscriminator, LaplacianLayer
        
        # Criar config mínima
        cfg = ExperimentConfig()
        device = torch.device("cpu")
        
        # Criar modelos dummy
        H, W = 32, 32
        
        generator = UNetGenerator2D(
            in_channels=1,
            latent_dim=cfg.latent_dim,
            base_channels=12,
            depth=3,
            use_batch_norm=cfg.generator_use_batch_norm,
            hard_constraint=cfg.hard_constraint_bc,
            output_smoothing_steps=cfg.generator_output_smoothing_steps,
            output_smoothing_strength=cfg.generator_output_smoothing_strength,
        )
        
        # Create discriminators
        from src.models import PhysicsDiscriminator2D, DataDiscriminator2D
        physics_disc = PhysicsDiscriminator2D(base_channels=12)
        data_disc = DataDiscriminator2D(base_channels=12)
        discriminator = FieldDualDiscriminator(
            physics_discriminator=physics_disc,
            data_discriminator=data_disc,
        )
        
        laplacian = LaplacianLayer(hx=1.0, hy=1.0)
        
        # Campos auxiliares
        base_field = torch.zeros(1, 1, H, W)
        phi_mask = torch.ones(1, 1, H, W)
        reference_field = torch.zeros(1, 1, H, W)
        interior_mask = torch.ones(1, 1, H, W)
        boundary_mask = torch.zeros(1, 1, H, W)
        boundary_mask[:, :, 0, :] = 1
        boundary_mask[:, :, -1, :] = 1
        boundary_mask[:, :, :, 0] = 1
        boundary_mask[:, :, :, -1] = 1
        
        # Tentar inicializar trainer
        try:
            trainer = FieldPIGANTrainer(
                generator=generator,
                discriminator=discriminator,
                laplacian=laplacian,
                base_field=base_field,
                phi_mask=phi_mask,
                coord_field=None,
                reference_field=reference_field,
                interior_mask=interior_mask,
                boundary_mask=boundary_mask,
                config=cfg,
                device=device,
            )
        except Exception as e:
            pytest.fail(f"Trainer initialization failed: {e}")
        
        # Verificar que novos componentes foram inicializados
        assert hasattr(trainer, "pde_computer")
        assert hasattr(trainer, "domain_metrics_computer")
        assert hasattr(trainer, "loss_computer")
        assert hasattr(trainer, "lambda_pde_adapter")
        assert hasattr(trainer, "gradnorm_balancer")
        assert hasattr(trainer, "stagnation_detector")
        assert hasattr(trainer, "divergence_detector")
        
        # Verificar que são instâncias corretas
        from src.physics.pdeResidual import PDE_Residual_Computer
        from src.physics.domainMetrics import Domain_Metrics_Computer
        from src.training.lossFunctions import PIGANLossComputation
        from src.training.adaptiveSchemes import (
            AdaptiveLambdaPDE,
            GradNormBalancer,
            StagnationDetector,
            DivergenceDetector,
        )
        
        assert isinstance(trainer.pde_computer, PDE_Residual_Computer)
        assert isinstance(trainer.domain_metrics_computer, Domain_Metrics_Computer)
        assert isinstance(trainer.loss_computer, PIGANLossComputation)
        assert isinstance(trainer.lambda_pde_adapter, AdaptiveLambdaPDE)
        assert isinstance(trainer.gradnorm_balancer, GradNormBalancer)
        assert isinstance(trainer.stagnation_detector, StagnationDetector)
        assert isinstance(trainer.divergence_detector, DivergenceDetector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
