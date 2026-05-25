# -*- coding: utf-8 -*-
"""Modelos e operadores PI-GAN com notação alinhada ao artigo."""

from src.model.discriminator import (
    DataDiscriminator2D,
    FieldDualDiscriminator,
    PhysicsDiscriminator2D,
    create_field_pigan_models,
)
from src.model.generator import HardConstraintLayer, UNetGenerator2D
from src.model.operators import (
    LaplacianLayer,
    LaplacianOperator,
    build_g_field,
    build_phi_mask,
)

__all__ = [
    "UNetGenerator2D",
    "HardConstraintLayer",
    "LaplacianOperator",
    "LaplacianLayer",
    "PhysicsDiscriminator2D",
    "DataDiscriminator2D",
    "FieldDualDiscriminator",
    "create_field_pigan_models",
    "build_g_field",
    "build_phi_mask",
]
