#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auditoria estrutural do projeto PI-GAN.

Este script verifica regras que tornam o repositório auditável sem executar
treinamento: árvore oficial, nomes lowerCamelCase, imports antigos e artefatos
gerados que não devem voltar ao versionamento.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PATHS = {
    ".gitignore",
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "setup.cfg",
    "main.py",
    "inference.py",
    "src/__init__.py",
    "src/config.py",
    "src/evaluation.py",
    "src/fdm.py",
    "src/models.py",
    "src/pipeline.py",
    "src/trainer.py",
    "src/utils.py",
    "src/physics/__init__.py",
    "src/physics/domainMetrics.py",
    "src/physics/pdeResidual.py",
    "src/training/__init__.py",
    "src/training/adaptiveSchemes.py",
    "src/training/lossFunctions.py",
    "scripts/__init__.py",
    "scripts/auditProject.py",
    "scripts/finalBestConfig.json",
    "scripts/optunaCommon.py",
    "scripts/optunaSearch.py",
    "scripts/runFinalBest.py",
    "scripts/runOptunaFinalRefine3050.ps1",
    "scripts/runOptunaFixedBestRefine3050.ps1",
    "scripts/runOptunaOvernight3050.ps1",
    "scripts/sanityStochasticForward.py",
    "scripts/tuneActivePigan.py",
    "scripts/tuneOptunaPhysicsFirst.py",
    "tests/__init__.py",
    "tests/testCheckpointCompatibility.py",
    "tests/testCheckpointResume.py",
    "tests/testCpuFallback.py",
    "tests/testFieldTrainingModes.py",
    "tests/testInferencePoints.py",
    "tests/testLaplacianFieldOperator.py",
    "tests/testMainConfig.py",
    "tests/testPipelineExecution.py",
    "tests/testTrainingModules.py",
}

OPTIONAL_DATA_FILES = {
    "points.csv",
    "pontos.csv",
}

FORBIDDEN_SOURCE_DIRS = {
    ".agents",
    "camelcase-refactor",
    "orphaned",
    "pigan-skill",
    "report",
}

STALE_FILE_REFERENCES = {
    "optuna_common.py",
    "optuna_search.py",
    "run_final_4000_best.py",
    "tune_active_pigan.py",
    "tune_optuna_physics_first.py",
    "pde_residual.py",
    "domain_metrics.py",
    "loss_functions.py",
    "adaptive_schemes.py",
    "run_optuna_final_refine_3050.ps1",
    "run_optuna_fixed_best_refine_3050.ps1",
    "run_optuna_overnight_3050.ps1",
    "final_4000_best_config.json",
}

ALLOWED_ROOT_FILES = EXPECTED_PATHS | OPTIONAL_DATA_FILES
LOWER_CAMEL_RE = re.compile(r"^[a-z][A-Za-z0-9]*$")
TEST_CAMEL_RE = re.compile(r"^test[A-Z][A-Za-z0-9]*$")


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _iter_project_files() -> list[Path]:
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and not any(part in ignored_parts for part in path.parts)
    )


def _is_lower_camel_file(path: Path) -> bool:
    if path.name == "__init__.py":
        return True
    stem = path.stem
    if path.parent.name == "tests" and path.suffix == ".py":
        return bool(TEST_CAMEL_RE.fullmatch(stem))
    return bool(LOWER_CAMEL_RE.fullmatch(stem))


def check_expected_paths(errors: list[str]) -> None:
    missing = sorted(path for path in EXPECTED_PATHS if not (PROJECT_ROOT / path).exists())
    for path in missing:
        errors.append(f"Arquivo oficial ausente: {path}")


def check_unexpected_root_files(errors: list[str]) -> None:
    for path in PROJECT_ROOT.iterdir():
        if path.is_file() and path.name not in {Path(item).name for item in ALLOWED_ROOT_FILES}:
            errors.append(f"Arquivo solto na raiz sem contrato arquitetural: {path.name}")


def check_forbidden_source_dirs(errors: list[str]) -> None:
    for directory in PROJECT_ROOT.rglob("*"):
        if ".venv" in directory.parts:
            continue
        if directory.is_dir() and directory.name in FORBIDDEN_SOURCE_DIRS:
            errors.append(f"Diretorio fora do contrato arquitetural: {_relative(directory)}")


def check_lower_camel_names(errors: list[str]) -> None:
    for path in _iter_project_files():
        rel = _relative(path)
        if path.suffix not in {".py", ".ps1", ".json"}:
            continue
        if path.parent == PROJECT_ROOT:
            continue
        if not _is_lower_camel_file(path):
            errors.append(f"Nome fora do lowerCamelCase/testCamelCase: {rel}")


def check_stale_references(errors: list[str]) -> None:
    searchable_suffixes = {".py", ".ps1", ".md", ".txt", ".json", ".ini", ".cfg"}
    for path in _iter_project_files():
        if path.suffix not in searchable_suffixes:
            continue
        if _relative(path) == "scripts/auditProject.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for old_name in STALE_FILE_REFERENCES:
            if old_name in text:
                errors.append(f"Referencia antiga '{old_name}' encontrada em {_relative(path)}")


def check_import_boundaries(errors: list[str]) -> None:
    for path in _iter_project_files():
        if path.suffix != ".py":
            continue
        rel = _relative(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "sys.path.insert" in text and not rel.startswith("scripts/"):
            errors.append(f"sys.path.insert fora de scripts/: {rel}")


def run_audit() -> list[str]:
    errors: list[str] = []
    check_expected_paths(errors)
    check_unexpected_root_files(errors)
    check_forbidden_source_dirs(errors)
    check_lower_camel_names(errors)
    check_stale_references(errors)
    check_import_boundaries(errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audita estrutura e convenções do projeto.")
    parser.add_argument("--quiet", action="store_true", help="Mostra apenas erros.")
    args = parser.parse_args()

    errors = run_audit()
    if errors:
        print("AUDIT FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    if not args.quiet:
        print("AUDIT PASSED")
        print(f"Arquivos oficiais verificados: {len(EXPECTED_PATHS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
