# -*- coding: utf-8 -*-
import numpy as np
import pytest

from inference import (
    _extract_2d_field,
    interpolate_points_from_field,
    load_points_csv,
    save_checkpoint_temperature_csv,
    save_temperature_csv,
)


def test_interpolate_bilinear_matches_linear_field():
    lx = 1.0
    ly = 1.0
    x = np.linspace(0.0, lx, 5)
    y = np.linspace(0.0, ly, 4)
    xx, yy = np.meshgrid(x, y)
    field = 10.0 * xx + yy

    points = np.array(
        [[0.25, 0.75], [0.90, 0.10], [0.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    values = interpolate_points_from_field(
        field, points, lx=lx, ly=ly, method="bilinear"
    )

    expected = 10.0 * points[:, 0] + points[:, 1]
    assert np.allclose(values, expected, atol=1e-12)


def test_interpolate_nearest_uses_grid_cell_closest_center():
    field = np.array(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ],
        dtype=np.float64,
    )
    points = np.array([[0.49, 0.51], [0.99, 0.99]], dtype=np.float64)
    values = interpolate_points_from_field(
        field, points, lx=1.0, ly=1.0, method="nearest"
    )

    assert values[0] == pytest.approx(11.0)
    assert values[1] == pytest.approx(22.0)


def test_interpolate_outside_domain_raises_by_default():
    field = np.ones((4, 4), dtype=np.float64)
    points = np.array([[1.1, 0.2]], dtype=np.float64)
    with pytest.raises(ValueError, match="fora do domínio"):
        interpolate_points_from_field(
            field,
            points,
            lx=1.0,
            ly=1.0,
            method="bilinear",
            allow_outside=False,
        )


def test_load_and_save_points_csv_roundtrip(tmp_path):
    tmp_path = tmp_path
    points_csv = tmp_path / "points.csv"
    points_csv.write_text("X,Y\n0.1,0.2\n0.5,0.6\n", encoding="utf-8")
    points = load_points_csv(points_csv)

    out_csv = tmp_path / "out.csv"
    temps = np.array([123.0, 456.0], dtype=np.float64)
    save_temperature_csv(out_csv, points, temps)

    saved = out_csv.read_text(encoding="utf-8")
    assert "x,y,temperature" in saved
    assert "0.1,0.2,123" in saved
    assert "0.5,0.6,456" in saved


def test_load_points_csv_semicolon_and_decimal_comma(tmp_path):
    tmp_path = tmp_path
    points_csv = tmp_path / "pontos_ptbr.csv"
    points_csv.write_text("x;y\n0,10;0,20\n0,50;0,75\n", encoding="utf-8")
    points = load_points_csv(points_csv)
    expected = np.array([[0.10, 0.20], [0.50, 0.75]], dtype=np.float64)
    assert np.allclose(points, expected, atol=0.0, rtol=0.0)


def test_save_temperature_csv_excel_ptbr_format(tmp_path):
    tmp_path = tmp_path
    out_csv = tmp_path / "saida_ptbr.csv"
    points = np.array([[0.1, 0.2]], dtype=np.float64)
    temps = np.array([190.146261304], dtype=np.float64)

    save_temperature_csv(
        out_csv,
        points,
        temps,
        delimiter=";",
        decimal_comma=True,
    )
    saved = out_csv.read_text(encoding="utf-8")
    assert "x;y;temperature" in saved
    assert "0,1;0,2;190,146261304" in saved


def test_save_checkpoint_temperature_csv_includes_checkpoint_metadata(tmp_path):
    out_csv = tmp_path / "saida_checkpoints.csv"
    rows = [
        {
            "checkpoint": "checkpoint_epoch_250.pt",
            "epoch": 250,
            "x": 0.1,
            "y": 0.2,
            "temperature": 190.146261304,
        }
    ]

    save_checkpoint_temperature_csv(out_csv, rows)

    saved = out_csv.read_text(encoding="utf-8")
    assert "checkpoint,epoch,x,y,temperature" in saved
    assert "checkpoint_epoch_250.pt,250,0.1,0.2,190.146261304" in saved


def test_extract_2d_field_from_generator_samples():
    samples = np.arange(2 * 1 * 3 * 4, dtype=np.float64).reshape(2, 1, 3, 4)

    field = _extract_2d_field(samples)

    expected = samples.mean(axis=0)[0]
    assert field.shape == (3, 4)
    assert np.allclose(field, expected)
