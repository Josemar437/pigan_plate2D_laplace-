# -*- coding: utf-8 -*-
"""Inferência de temperatura em pontos (x,y) para a placa 2D.

Uso típico:
    python inference.py --points_csv pontos.csv \
        --field_npy runs/results/temperature_pred.npy

Também é possível inferir diretamente de checkpoint:
    python inference.py --points_csv pontos.csv \
        --checkpoint runs/checkpoints/checkpoint_epoch_4000.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def load_points_csv(points_csv: Path) -> np.ndarray:
    """Lê CSV com colunas x,y e retorna array [N,2]."""
    with open(points_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("CSV de pontos sem cabeçalho.")

        field_map: Dict[str, str] = {
            str(name).strip().lower(): str(name) for name in reader.fieldnames
        }
        if "x" not in field_map or "y" not in field_map:
            raise ValueError(
                "CSV de pontos deve conter colunas 'x' e 'y'. "
                f"Recebidas: {reader.fieldnames}"
            )
        x_key = field_map["x"]
        y_key = field_map["y"]

        points = []
        for line_idx, row in enumerate(reader, start=2):
            x_raw = row.get(x_key, "")
            y_raw = row.get(y_key, "")
            try:
                x_val = _parse_number_token(x_raw)
                y_val = _parse_number_token(y_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Valor inválido em x/y na linha {line_idx}: "
                    f"x={x_raw!r}, y={y_raw!r}."
                ) from exc
            points.append([x_val, y_val])

    if not points:
        raise ValueError("CSV de pontos está vazio (sem linhas de dados).")
    return np.asarray(points, dtype=np.float64)


def save_temperature_csv(
    output_csv: Path,
    points_xy: np.ndarray,
    temperatures: np.ndarray,
    *,
    delimiter: str = ",",
    decimal_comma: bool = False,
) -> None:
    """Salva CSV de saída com colunas x,y,temperature."""
    if delimiter not in {",", ";"}:
        raise ValueError("delimiter deve ser ',' ou ';'.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["x", "y", "temperature"])
        for idx in range(points_xy.shape[0]):
            writer.writerow(
                [
                    _format_float(points_xy[idx, 0], decimal_comma),
                    _format_float(points_xy[idx, 1], decimal_comma),
                    _format_float(temperatures[idx], decimal_comma),
                ]
            )


def save_checkpoint_temperature_csv(
    output_csv: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    delimiter: str = ",",
    decimal_comma: bool = False,
) -> None:
    """Salva temperaturas de um ou mais checkpoints em formato longo."""
    if delimiter not in {",", ";"}:
        raise ValueError("delimiter deve ser ',' ou ';'.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["checkpoint", "epoch", "x", "y", "temperature"])
        for row in rows:
            writer.writerow(
                [
                    row["checkpoint"],
                    row["epoch"],
                    _format_float(row["x"], decimal_comma),
                    _format_float(row["y"], decimal_comma),
                    _format_float(row["temperature"], decimal_comma),
                ]
            )


def print_temperature_table(
    points_xy: np.ndarray,
    temperatures: np.ndarray,
    *,
    checkpoint: Optional[Path] = None,
) -> None:
    """Imprime uma tabela compacta com x, y e temperatura."""
    if checkpoint is not None:
        print(f"\nCheckpoint: {checkpoint}")
    print("idx,x,y,temperature")
    for idx in range(points_xy.shape[0]):
        print(
            ",".join(
                [
                    str(idx),
                    _format_float(points_xy[idx, 0], False),
                    _format_float(points_xy[idx, 1], False),
                    _format_float(temperatures[idx], False),
                ]
            )
        )


def _format_indices(indices: Iterable[int], max_items: int = 8) -> str:
    idx_list = list(indices)
    if len(idx_list) <= max_items:
        return ", ".join(str(i) for i in idx_list)
    shown = ", ".join(str(i) for i in idx_list[:max_items])
    return f"{shown}, ... (+{len(idx_list) - max_items} mais)"


def _parse_number_token(token: Any) -> float:
    text = str(token).strip().replace(" ", "")
    if text == "":
        raise ValueError("token numérico vazio")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            # Ex.: 1.234,56 -> 1234.56
            text = text.replace(".", "").replace(",", ".")
        else:
            # Ex.: 1,234.56 -> 1234.56
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return float(text)


def _format_float(value: Any, decimal_comma: bool) -> str:
    text = f"{float(value):.12g}"
    if decimal_comma:
        text = text.replace(".", ",")
    return text


def interpolate_points_from_field(
    field: np.ndarray,
    points_xy: np.ndarray,
    *,
    lx: float,
    ly: float,
    method: str = "bilinear",
    allow_outside: bool = False,
) -> np.ndarray:
    """Interpola temperatura da malha para pontos (x,y) no domínio."""
    if field.ndim != 2:
        raise ValueError("field deve ter formato 2D [H,W].")
    h, w = int(field.shape[0]), int(field.shape[1])
    if h < 2 or w < 2:
        raise ValueError("field deve ter resolução mínima 2x2.")
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy deve ter formato [N,2].")
    if not np.isfinite(points_xy).all():
        raise ValueError("points_xy contém NaN/Inf.")
    if float(lx) <= 0.0 or float(ly) <= 0.0:
        raise ValueError("lx e ly devem ser positivos.")

    x = points_xy[:, 0].astype(np.float64, copy=False)
    y = points_xy[:, 1].astype(np.float64, copy=False)

    tol = 1e-12
    outside_mask = (
        (x < -tol)
        | (x > float(lx) + tol)
        | (y < -tol)
        | (y > float(ly) + tol)
    )
    if bool(np.any(outside_mask)) and (not bool(allow_outside)):
        outside_idx = np.where(outside_mask)[0].tolist()
        raise ValueError(
            "Há pontos fora do domínio [0,LX]x[0,LY]. "
            f"Índices: {_format_indices(outside_idx)}."
        )

    x = np.clip(x, 0.0, float(lx))
    y = np.clip(y, 0.0, float(ly))

    gx = x * (float(w - 1) / float(lx))
    gy = y * (float(h - 1) / float(ly))

    method_key = str(method).strip().lower()
    if method_key == "nearest":
        ix = np.rint(gx).astype(np.int64)
        iy = np.rint(gy).astype(np.int64)
        ix = np.clip(ix, 0, w - 1)
        iy = np.clip(iy, 0, h - 1)
        return field[iy, ix].astype(np.float64, copy=False)

    if method_key != "bilinear":
        raise ValueError(
            "Método inválido. Use 'bilinear' ou 'nearest'."
        )

    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    tx = gx - x0.astype(np.float64)
    ty = gy - y0.astype(np.float64)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]

    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    ).astype(np.float64, copy=False)


def _load_field_npy(field_npy: Path) -> np.ndarray:
    """Carrega campo de temperatura de arquivo .npy."""
    raw = np.load(field_npy)
    if raw.ndim == 2:
        field = raw
    elif raw.ndim == 3 and raw.shape[0] == 1:
        field = raw[0]
    elif raw.ndim == 4 and raw.shape[0] == 1 and raw.shape[1] == 1:
        field = raw[0, 0]
    else:
        raise ValueError(
            "Arquivo .npy inválido para campo de temperatura. "
            f"Esperado [H,W], [1,H,W] ou [1,1,H,W], recebido {raw.shape}."
        )
    if field.ndim != 2:
        raise ValueError("Campo carregado não é 2D.")
    return np.asarray(field, dtype=np.float64)


def _extract_2d_field(prediction: Any) -> np.ndarray:
    """Normaliza saída do gerador para campo 2D [H,W]."""
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().cpu().numpy()
    field = np.asarray(prediction, dtype=np.float64)
    if field.ndim == 4:
        field = field.mean(axis=0)
    elif field.ndim == 3 and field.shape[0] != 1:
        field = field.mean(axis=0)
    while field.ndim > 2 and field.shape[0] == 1:
        field = field[0]
    if field.ndim != 2:
        raise ValueError(
            "Predição do checkpoint não pôde ser convertida para campo 2D. "
            f"Formato recebido: {np.asarray(prediction).shape}."
        )
    return field.astype(np.float64, copy=False)


def _load_config_payload(config_json: Optional[Path]) -> Dict[str, Any]:
    if config_json is None:
        return {}
    with open(config_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("config_json deve conter um objeto JSON.")
    return payload


def _to_snake_case(name: str) -> str:
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def _resolve_domain_lengths(
    payload: Dict[str, Any],
    *,
    lx_override: Optional[float],
    ly_override: Optional[float],
) -> tuple[float, float]:
    lx = (
        float(lx_override)
        if lx_override is not None
        else float(payload.get("LX", 1.0))
    )
    ly = (
        float(ly_override)
        if ly_override is not None
        else float(payload.get("LY", 1.0))
    )
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("LX e LY devem ser positivos.")
    return lx, ly


def _build_experiment_config_from_payload(
    payload: Dict[str, Any],
    *,
    lx: float,
    ly: float,
) -> Any:
    from src.config import ExperimentConfig

    cfg = ExperimentConfig()
    valid_fields = set(cfg.__dataclass_fields__.keys())
    for key, value in payload.items():
        canonical_key = str(key)
        if canonical_key not in valid_fields:
            snake_key = _to_snake_case(canonical_key)
            upper_key = canonical_key.upper()
            if snake_key in valid_fields:
                canonical_key = snake_key
            elif upper_key in valid_fields:
                canonical_key = upper_key
        if canonical_key in valid_fields:
            setattr(cfg, canonical_key, value)
    cfg.LX = float(lx)
    cfg.LY = float(ly)
    cfg.__post_init__()
    return cfg


def _infer_field_from_checkpoint(
    checkpoint_path: Path,
    *,
    experiment_config: Any,
    runs_dir: Path,
    num_samples: int,
    strict_checkpoint: bool,
    use_gpu: bool,
    seed: int,
) -> np.ndarray:
    import torch

    from src.config import SystemConfig
    from src.pipeline import PIGANPipeline

    sys_cfg = SystemConfig(
        use_gpu=bool(use_gpu),
        mixed_precision=False,
        compile_model=False,
        dynamic_batch_size=False,
        seed=int(seed),
        log_file=None,
        deterministic_run=True,
        deterministic_warn_only=False,
    )
    pipeline = PIGANPipeline(experiment_config, sys_cfg, runs_dir=runs_dir)
    hx, hy = pipeline._prepare_physics_fields()
    pipeline.generator, pipeline.discriminator = pipeline._create_models()
    pipeline.trainer = pipeline._create_trainer(hx, hy)

    pipeline.trainer.load_checkpoint(
        checkpoint_path,
        strict=bool(strict_checkpoint),
        load_optimizer_state=False,
        restore_rng_state=False,
    )
    with torch.no_grad():
        samples = max(1, int(num_samples))
        pred = pipeline.trainer.predict(num_samples=samples)
    return _extract_2d_field(pred)


def _default_output_path(points_csv: Path) -> Path:
    return points_csv.with_name(f"{points_csv.stem}_temperature.csv")


def _default_checkpoint_output_path(points_csv: Path) -> Path:
    return points_csv.with_name(f"{points_csv.stem}_temperature_checkpoints.csv")


def _checkpoint_epoch(path: Path) -> Optional[int]:
    match = re.search(r"checkpoint_epoch_(\d+)", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _sort_checkpoint_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: (
            _checkpoint_epoch(path) is None,
            _checkpoint_epoch(path) if _checkpoint_epoch(path) is not None else 0,
            str(path),
        ),
    )


def _resolve_checkpoint_glob(pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    parent = pattern_path.parent if str(pattern_path.parent) else Path(".")
    matches = _sort_checkpoint_paths(parent.glob(pattern_path.name))
    return [path for path in matches if path.is_file()]


def _checkpoint_rows(
    checkpoint: Path,
    points_xy: np.ndarray,
    temperatures: np.ndarray,
) -> list[Dict[str, Any]]:
    epoch = _checkpoint_epoch(checkpoint)
    return [
        {
            "checkpoint": checkpoint.name,
            "epoch": "" if epoch is None else epoch,
            "x": points_xy[idx, 0],
            "y": points_xy[idx, 1],
            "temperature": temperatures[idx],
        }
        for idx in range(points_xy.shape[0])
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inferência de temperatura em pontos (x,y) para placa 2D."
        )
    )
    parser.add_argument(
        "--points-csv",
        "--points_csv",
        type=Path,
        required=True,
        help="CSV de entrada com colunas x,y.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--field-npy",
        "--field_npy",
        type=Path,
        default=None,
        help=(
            "Arquivo .npy com campo de temperatura (ex.: "
            "runs/results/temperature_pred.npy)."
        ),
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt para inferência via gerador.",
    )
    source.add_argument(
        "--checkpoint-glob",
        "--checkpoint_glob",
        type=str,
        default=None,
        help=(
            "Padrão glob para inferir vários checkpoints "
            "(ex.: runs/checkpoints/checkpoint_epoch_*.pt)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        "--output_csv",
        type=Path,
        default=None,
        help="CSV de saída (padrão: <points>_temperature.csv).",
    )
    parser.add_argument(
        "--output-delimiter",
        "--output_delimiter",
        type=str,
        choices=[",", ";"],
        default=",",
        help="Separador do CSV de saída.",
    )
    parser.add_argument(
        "--output-decimal-comma",
        "--output_decimal_comma",
        action="store_true",
        help="Formata números com vírgula decimal no CSV de saída.",
    )
    parser.add_argument(
        "--print-values",
        "--print_values",
        action="store_true",
        help="Imprime x,y,temperature no terminal.",
    )
    parser.add_argument(
        "--excelPtbr",
        "--excel-ptbr",
        "--excel_ptbr",
        dest="excel_ptbr",
        action="store_true",
        help=(
            "Atalho para saída compatível com Excel PT-BR "
            "(delimiter=';' e decimal com vírgula)."
        ),
    )
    parser.add_argument(
        "--config-json",
        "--config_json",
        type=Path,
        default=None,
        help=(
            "JSON opcional com parâmetros do ExperimentConfig usados no treino. "
            "Útil para compatibilidade de checkpoint."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Diretório base para a pipeline ao inferir via checkpoint.",
    )
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        default=1,
        help="Número de amostras para média do campo (modo estocástico).",
    )
    parser.add_argument(
        "--strict-checkpoint",
        "--strict_checkpoint",
        dest="strict_checkpoint",
        action="store_true",
        help="Ativa carregamento estrito do checkpoint (default).",
    )
    parser.add_argument(
        "--no-strict-checkpoint",
        "--no_strict_checkpoint",
        dest="strict_checkpoint",
        action="store_false",
        help="Permite carregamento parcial do checkpoint.",
    )
    parser.set_defaults(strict_checkpoint=True)
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Método de interpolação para os pontos de consulta.",
    )
    parser.add_argument(
        "--allow-outside",
        "--allow_outside",
        action="store_true",
        help="Permite pontos fora do domínio com clamp para a borda.",
    )
    parser.add_argument(
        "--lx",
        type=float,
        default=None,
        help="Comprimento LX do domínio (override do config).",
    )
    parser.add_argument(
        "--ly",
        type=float,
        default=None,
        help="Comprimento LY do domínio (override do config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para inferência via checkpoint.",
    )
    parser.add_argument(
        "--use-gpu",
        "--use_gpu",
        action="store_true",
        help="Solicita GPU para inferência via checkpoint.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    points_csv = Path(args.points_csv)
    if not points_csv.exists():
        raise FileNotFoundError(f"Arquivo de pontos não encontrado: {points_csv}")

    payload = _load_config_payload(
        Path(args.config_json) if args.config_json is not None else None
    )
    domain_lx, domain_ly = _resolve_domain_lengths(
        payload,
        lx_override=args.lx,
        ly_override=args.ly,
    )
    points_xy = load_points_csv(points_csv)
    output_delimiter = str(args.output_delimiter)
    output_decimal_comma = bool(args.output_decimal_comma)
    if bool(args.excel_ptbr):
        output_delimiter = ";"
        output_decimal_comma = True

    if args.field_npy is not None:
        field = _load_field_npy(Path(args.field_npy))
        temperatures = interpolate_points_from_field(
            field,
            points_xy,
            lx=float(domain_lx),
            ly=float(domain_ly),
            method=str(args.interpolation),
            allow_outside=bool(args.allow_outside),
        )
        output_csv = (
            Path(args.output_csv)
            if args.output_csv is not None
            else _default_output_path(points_csv)
        )
        save_temperature_csv(
            output_csv,
            points_xy,
            temperatures,
            delimiter=output_delimiter,
            decimal_comma=output_decimal_comma,
        )
        if bool(args.print_values):
            print_temperature_table(points_xy, temperatures)
        print(f"Pontos processados: {points_xy.shape[0]}")
        print(f"Campo utilizado: {field.shape[0]}x{field.shape[1]}")
        print(f"Dominio: [0,{float(domain_lx)}] x [0,{float(domain_ly)}]")
        print(f"Saida: {output_csv}")
        return

    if args.checkpoint_glob is not None:
        checkpoints = _resolve_checkpoint_glob(str(args.checkpoint_glob))
        if not checkpoints:
            raise FileNotFoundError(
                f"Nenhum checkpoint encontrado para: {args.checkpoint_glob}"
            )
    else:
        checkpoints = [Path(args.checkpoint)]

    if args.checkpoint_glob is None and len(checkpoints) == 1:
        exp_cfg = _build_experiment_config_from_payload(
            payload,
            lx=domain_lx,
            ly=domain_ly,
        )
        checkpoint_path = checkpoints[0]
        field = _infer_field_from_checkpoint(
            checkpoint_path,
            experiment_config=exp_cfg,
            runs_dir=Path(args.runs_dir),
            num_samples=max(1, int(args.num_samples)),
            strict_checkpoint=bool(args.strict_checkpoint),
            use_gpu=bool(args.use_gpu),
            seed=int(args.seed),
        )
        temperatures = interpolate_points_from_field(
            field,
            points_xy,
            lx=float(domain_lx),
            ly=float(domain_ly),
            method=str(args.interpolation),
            allow_outside=bool(args.allow_outside),
        )
        output_csv = (
            Path(args.output_csv)
            if args.output_csv is not None
            else _default_output_path(points_csv)
        )
        save_temperature_csv(
            output_csv,
            points_xy,
            temperatures,
            delimiter=output_delimiter,
            decimal_comma=output_decimal_comma,
        )
        if bool(args.print_values):
            print_temperature_table(
                points_xy,
                temperatures,
                checkpoint=checkpoint_path,
            )
        print(f"Pontos processados: {points_xy.shape[0]}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Campo utilizado: {field.shape[0]}x{field.shape[1]}")
        print(f"Dominio: [0,{float(domain_lx)}] x [0,{float(domain_ly)}]")
        print(f"Saida: {output_csv}")
        return

    exp_cfg = _build_experiment_config_from_payload(
        payload,
        lx=domain_lx,
        ly=domain_ly,
    )
    all_rows: list[Dict[str, Any]] = []
    last_field_shape: Optional[tuple[int, int]] = None
    for checkpoint_path in checkpoints:
        field = _infer_field_from_checkpoint(
            checkpoint_path,
            experiment_config=exp_cfg,
            runs_dir=Path(args.runs_dir),
            num_samples=max(1, int(args.num_samples)),
            strict_checkpoint=bool(args.strict_checkpoint),
            use_gpu=bool(args.use_gpu),
            seed=int(args.seed),
        )
        last_field_shape = (int(field.shape[0]), int(field.shape[1]))
        temperatures = interpolate_points_from_field(
            field,
            points_xy,
            lx=float(domain_lx),
            ly=float(domain_ly),
            method=str(args.interpolation),
            allow_outside=bool(args.allow_outside),
        )
        if bool(args.print_values):
            print_temperature_table(
                points_xy,
                temperatures,
                checkpoint=checkpoint_path,
            )
        all_rows.extend(_checkpoint_rows(checkpoint_path, points_xy, temperatures))

    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else _default_checkpoint_output_path(points_csv)
    )
    save_checkpoint_temperature_csv(
        output_csv,
        all_rows,
        delimiter=output_delimiter,
        decimal_comma=output_decimal_comma,
    )

    print(f"Pontos processados por checkpoint: {points_xy.shape[0]}")
    print(f"Checkpoints processados: {len(checkpoints)}")
    if last_field_shape is not None:
        print(f"Campo utilizado: {last_field_shape[0]}x{last_field_shape[1]}")
    print(f"Dominio: [0,{float(domain_lx)}] x [0,{float(domain_ly)}]")
    print(f"Saida: {output_csv}")


if __name__ == "__main__":
    main()
