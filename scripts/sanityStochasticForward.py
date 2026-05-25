#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sanity check: o gerador estocástico deve mudar a saída quando z muda."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import UNetGenerator2D  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    dtype = torch.float32
    device = torch.device("cpu")

    generator = UNetGenerator2D(
        in_channels=1,
        latent_dim=8,
        base_channels=12,
        depth=3,
        use_batch_norm=False,
        hard_constraint=True,
    ).to(device=device, dtype=dtype)
    generator.eval()

    base = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
    phi = torch.ones_like(base)
    z1 = torch.randn(1, 8, device=device, dtype=dtype)
    z2 = torch.randn(1, 8, device=device, dtype=dtype)

    with torch.no_grad():
        pred1 = generator(base, phi, z=z1)
        pred2 = generator(base, phi, z=z2)

    diff = torch.linalg.vector_norm(pred1 - pred2).item()
    print(f"||pred1 - pred2||_2 = {diff:.6e}")
    assert diff > 1e-6, "Amostras com z diferentes devem gerar saídas diferentes."


if __name__ == "__main__":
    main()
