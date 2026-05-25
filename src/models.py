# -*- coding: utf-8 -*-
"""Shim de compatibilidade — importe de src.model.*."""

from src.model.discriminator import (
    DataDiscriminator2D,
    FieldDualDiscriminator,
    PhysicsDiscriminator2D,
    create_field_pigan_models,
)
from src.model.generator import UNetGenerator2D
from src.model.operators import LaplacianLayer, LaplacianOperator

__all__ = [
    "UNetGenerator2D",
    "LaplacianLayer",
    "LaplacianOperator",
    "PhysicsDiscriminator2D",
    "DataDiscriminator2D",
    "FieldDualDiscriminator",
    "create_field_pigan_models",
]
