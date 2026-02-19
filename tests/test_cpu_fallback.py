# -*- coding: utf-8 -*-
import torch

from src.config import SystemConfig, initialize_system


def test_cpu_fallback_with_allow_cpu(monkeypatch):
    monkeypatch.setenv("PIGAN_ALLOW_CPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    system_config = SystemConfig(use_gpu=True, mixed_precision=False, compile_model=False)
    _, device, _, _ = initialize_system(system_config)

    assert device.type == "cpu"
