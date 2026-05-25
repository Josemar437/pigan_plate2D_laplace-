# -*- coding: utf-8 -*-
"""Training constants calibrated via SKILL Phases 1-2 validation.

These values represent hyperparameter ranges and defaults that have been
empirically validated for the PI-GAN training pipeline on 2D Laplace problems.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class AdaptiveLambdaDefaults:
    """Adaptive PDE weight scheduling parameters.
    
    Based on Phase 1 SKILL validation and empirical tuning.
    """
    # [SKILL P2] λ_PDE range determined by pilot studies on Laplace 2D
    LAMBDA_PDE_MIN: Final[float] = 19.0
    LAMBDA_PDE_MAX: Final[float] = 98.0
    
    # EMA smoothing for stable updates (high beta = aggressive smoothing)
    EMA_BETA: Final[float] = 0.9
    
    # Reference residual scale (normalized against this value)
    RESIDUAL_SCALE_REFERENCE: Final[float] = 1.0e-2
    
    # Physical target for convergence (SKILL Phase 3)
    RESIDUAL_TOLERANCE_TARGET: Final[float] = 1.0e-3
    
    # Growth exponent for log-linear scaling
    GROWTH_EXPONENT: Final[float] = 0.6


@dataclass(frozen=True)
class GradNormBalancerDefaults:
    """Gradient norm balancing for adversarial-physics coupling.
    
    Maintains stable ratio between adversarial and PDE gradient magnitudes.
    """
    # Target ratio: grad_adv / grad_pde
    TARGET_RATIO: Final[float] = 0.35
    
    # EMA smoothing (same as lambda adapter for consistency)
    EMA_BETA: Final[float] = 0.9
    
    # Scale clipping bounds (prevents extreme scaling)
    SCALE_MIN: Final[float] = 0.05
    SCALE_MAX: Final[float] = 1.0


@dataclass(frozen=True)
class StagnationDetectorDefaults:
    """Stagnation detection and recovery parameters.
    
    Triggers adversarial boost when generator residual stops improving.
    """
    # Epochs without improvement before triggering stagnation
    PATIENCE: Final[int] = 50
    
    # Relative improvement threshold (5% improvement required)
    REL_TOLERANCE: Final[float] = 5e-3
    
    # Multiplicative boost factor for adversarial loss
    BOOST_FACTOR: Final[float] = 1.4
    
    # Epochs to wait before allowing next boost
    COOLDOWN: Final[int] = 8


@dataclass(frozen=True)
class DivergenceDetectorDefaults:
    """Loss divergence detection parameters.
    
    Uses rolling window comparison to detect sustained loss growth.
    """
    # Window size for comparison (compare last N epochs vs previous N)
    WINDOW_SIZE: Final[int] = 16
    
    # Ratio threshold for detecting divergence (30% growth)
    RATIO_THRESHOLD: Final[float] = 1.2
    
    # Consecutive epochs of divergence before triggering action
    PATIENCE: Final[int] = 2


# Module-level aliases for easy access
ADAPTIVE_LAMBDA_DEFAULTS = AdaptiveLambdaDefaults()
GRADNORM_BALANCER_DEFAULTS = GradNormBalancerDefaults()
STAGNATION_DETECTOR_DEFAULTS = StagnationDetectorDefaults()
DIVERGENCE_DETECTOR_DEFAULTS = DivergenceDetectorDefaults()
