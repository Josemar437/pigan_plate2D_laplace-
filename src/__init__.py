# -*- coding: utf-8 -*-
"""
Contrato de execução ativo deste projeto.

Esta implementação corresponde a uma PI-GAN híbrida,
definida como rede adversarial generativa informada pela física,
com penalização explícita do resíduo da PDE e regularização adversarial.

Fluxo operacional oficial:
- main.py
- src/pipeline.py
- src/models.py
- src/trainer.py
"""

__all__ = [
    "config",
    "pipeline",
    "models",
    "trainer",
    "utils",
    "fdm",
    "evaluation",
]
