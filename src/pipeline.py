# -*- coding: utf-8 -*-
"""Pipeline principal PI-GAN fisicamente consistente para a equação de Laplace 2D."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import asdict

from src.config import (
    ExperimentConfig,
    SystemConfig,
    initialize_system,
    log_hyperparameters_at_start,
)
from src.evaluation import compute_field_metrics
from src.fdm import solve_laplace_mixed_dirichlet_neumann
from src.utils import (
    build_dirichlet_extension,
    build_hard_constraint_mask,
    build_mixed_boundary_masks,
    create_cartesian_grid,
)
from src.models import (
    FieldDualDiscriminator,
    LaplacianLayer,
    UNetGenerator2D,
    create_field_pigan_models,
)
from src.trainer import FieldPIGANTrainer, FieldTrainerConfig


class PIGANPipeline:
    """
    Pipeline completo: configuração -> referência FDM -> modelos -> treinamento
    adversarial PI-GAN -> métricas.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        system_config: Optional[SystemConfig] = None,
        runs_dir: Path = Path("runs"),
    ) -> None:
        self.exp_config = experiment_config
        self.sys_config = system_config or SystemConfig()
        self.sys_config, self.device, self.memory_manager, self.logger = initialize_system(
            self.sys_config
        )

        self.runs_dir = Path(runs_dir)
        self.results_dir = self.runs_dir / "results"
        configured_checkpoint_dir = getattr(self.exp_config, "checkpoint_dir", None)
        if configured_checkpoint_dir:
            checkpoint_dir = Path(str(configured_checkpoint_dir))
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = self.runs_dir / checkpoint_dir
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = self.runs_dir / "checkpoints"
        self.plots_dir = self.runs_dir / "plots"
        self.logs_dir = self.runs_dir / "logs"
        for d in (self.results_dir, self.checkpoint_dir, self.plots_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.generator: Optional[UNetGenerator2D] = None
        self.discriminator: Optional[FieldDualDiscriminator] = None
        self.trainer: Optional[FieldPIGANTrainer] = None
        self.laplacian: Optional[LaplacianLayer] = None
        self._last_eval_pred: Optional[torch.Tensor] = None

        self.base_field: Optional[torch.Tensor] = None
        self.phi_mask: Optional[torch.Tensor] = None
        self.coord_field: Optional[torch.Tensor] = None
        self.reference_field: Optional[torch.Tensor] = None
        self.interior_mask: Optional[torch.Tensor] = None
        self.boundary_mask: Optional[torch.Tensor] = None
        self.neumann_mask: Optional[torch.Tensor] = None
        self._baseline_metrics: Optional[Dict[str, float]] = None

        if self.logger:
            self.logger.info(
                "Pipeline PI-GAN 2D inicializado",
                device=str(self.device),
                grid=f"{self.exp_config.grid_size_y}x{self.exp_config.grid_size_x}",
                hard_constraint=self.exp_config.hard_constraint_bc,
                latent_dim=self.exp_config.latent_dim,
                generator_mode=str(self.exp_config.generator_mode),
                use_physical_coordinates=bool(
                    getattr(self.exp_config, "use_physical_coordinates", True)
                ),
                use_reference_discriminator=bool(
                    getattr(self.exp_config, "use_reference_discriminator", True)
                ),
            )
            self.logger.info(
                "Formula PI-GAN ativa",
                objective=(
                    "min_G max_Df,Dd: "
                    "L_G=lambda_pde*L_PDE_raw + lambda_adv_f*(-E[D_f(fake)]) "
                    "+ lambda_adv_d*(-E[D_d(fake)]) + lambda_bc*L_BC "
                    "+ lambda_neumann*L_N; "
                    "L_Df=E[D_f(fake)]-E[D_f(real)] + lambda_gp*GP_f; "
                    "L_Dd=E[D_d(fake)]-E[D_d(real)] + lambda_gp*GP_d"
                ),
            )

    def _prepare_physics_fields(self) -> Tuple[float, float]:
        """
        Prepara a malha, extensão de Dirichlet, referência FDM e máscaras do domínio.

        Retorno:
            Uma tupla (hx, hy) com os tamanhos dos passos espaciais.
        """
        nx = int(self.exp_config.grid_size_x)
        ny = int(self.exp_config.grid_size_y)
        lx = float(self.exp_config.LX)
        ly = float(self.exp_config.LY)
        field_dtype = torch.float64 if bool(getattr(self.sys_config, "use_double", False)) else torch.float32

        x_grid, y_grid = create_cartesian_grid(
            nx,
            ny,
            lx,
            ly,
            device=self.device,
            dtype=field_dtype,
        )
        x_norm = x_grid / (float(lx) + 1e-12)
        y_norm = y_grid / (float(ly) + 1e-12)
        coord = torch.stack([x_norm, y_norm], dim=0)  # [2,H,W]

        g_field = build_dirichlet_extension(
            x_grid,
            y_grid,
            lx=lx,
            ly=ly,
            t_left=float(self.exp_config.T_LEFT),
            t_right=float(self.exp_config.T_RIGHT),
            boundary_sine_amplitude=float(self.exp_config.boundary_sine_amplitude),
        )
        phi = build_hard_constraint_mask(x_grid, y_grid, lx=lx, ly=ly)

        t_ref, fdm_iters = solve_laplace_mixed_dirichlet_neumann(
            g_field,
            lx=lx,
            ly=ly,
            t_left=float(self.exp_config.T_LEFT),
            t_right=float(self.exp_config.T_RIGHT),
            tol=float(self.exp_config.fdm_tol),
            max_iter=int(self.exp_config.fdm_max_iter),
            omega=float(self.exp_config.fdm_omega),
        )

        interior_mask, boundary_mask, neumann_mask = build_mixed_boundary_masks(
            ny, nx, device=self.device, dtype=field_dtype
        )

        self.base_field = g_field.unsqueeze(0).unsqueeze(0)
        self.phi_mask = phi.unsqueeze(0).unsqueeze(0)
        self.coord_field = coord.unsqueeze(0)
        self.reference_field = t_ref.unsqueeze(0).unsqueeze(0)
        self.interior_mask = interior_mask
        self.boundary_mask = boundary_mask
        self.neumann_mask = neumann_mask

        if self.logger:
            self.logger.info(
                "Campo de referencia FDM preparado",
                fdm_iterations=fdm_iters,
                fdm_tol=self.exp_config.fdm_tol,
                boundary_sine_amplitude=self.exp_config.boundary_sine_amplitude,
            )

        hx = lx / float(nx - 1)
        hy = ly / float(ny - 1)
        return hx, hy

    def _create_models(self) -> Tuple[UNetGenerator2D, FieldDualDiscriminator]:
        """
        Instancia o gerador e os discriminadores baseados na configuração.

        Retorno:
            Uma tupla (gerador, discriminador_dual).
        """
        use_coords = bool(getattr(self.exp_config, "use_physical_coordinates", True))
        generator_in_channels = 1 + (2 if use_coords else 0)
        generator_config = {
            "in_channels": int(generator_in_channels),
            "latent_dim": int(self.exp_config.latent_dim),
            "base_channels": int(self.exp_config.generator_base_channels),
            "depth": int(self.exp_config.generator_depth),
            "use_batch_norm": bool(self.exp_config.generator_use_batch_norm),
            "hard_constraint": bool(self.exp_config.hard_constraint_bc),
            "zero_init_final": bool(getattr(self.exp_config, "generator_zero_init_final", True)),
            "output_smoothing_steps": int(
                getattr(self.exp_config, "generator_output_smoothing_steps", 0)
            ),
            "output_smoothing_strength": float(
                getattr(self.exp_config, "generator_output_smoothing_strength", 0.0)
            ),
            "activation": str(getattr(self.exp_config, "generator_activation", "silu")),
            "pooling": str(getattr(self.exp_config, "generator_pooling", "avg")),
        }
        discriminator_config = {
            "base_channels": int(self.exp_config.discriminator_base_channels),
            "capacity_scale": float(getattr(self.exp_config, "discriminator_capacity_scale", 1.0)),
            "dropout": float(getattr(self.exp_config, "discriminator_dropout", 0.0)),
            "use_spectral_norm": bool(getattr(self.exp_config, "discriminator_spectral_norm", False)),
            "init": str(getattr(self.exp_config, "discriminator_init", "kaiming")),
        }
        generator_config["init"] = str(getattr(self.exp_config, "generator_init", "kaiming"))
        return create_field_pigan_models(
            generator_config=generator_config,
            discriminator_config=discriminator_config,
            device=self.device,
            logger=self.logger,
        )

    def _create_trainer(self, hx: float, hy: float) -> FieldPIGANTrainer:
        """
        Cria a instância do treinador PI-GAN com os modelos e configurações.

        Parâmetros:
            hx: Tamanho do passo em X.
            hy: Tamanho do passo em Y.

        Retorno:
            Instância do treinador configurado.

        Exceções:
            RuntimeError: Se os modelos ou campos não foram inicializados.
        """
        if self.generator is None or self.discriminator is None:
            raise RuntimeError("Modelos devem ser inicializados antes de criar o trainer.")
        use_coords = bool(getattr(self.exp_config, "use_physical_coordinates", True))
        if any(
            tensor is None
            for tensor in (
                self.base_field,
                self.phi_mask,
                self.reference_field,
                self.interior_mask,
                self.boundary_mask,
                self.neumann_mask,
            )
        ):
            raise RuntimeError("Campos físicos devem ser preparados antes de criar o trainer.")
        if use_coords and self.coord_field is None:
            raise RuntimeError(
                "Campo de coordenadas deve ser preparado quando use_physical_coordinates=True."
            )

        self.laplacian = LaplacianLayer(hx=hx, hy=hy).to(self.device)

        lambda_bc = float(self.exp_config.lambda_bc)
        if bool(self.exp_config.hard_constraint_bc) and lambda_bc != 0.0:
            if self.logger:
                self.logger.info(
                    "Hard constraint ativo: termo de contorno da loss desativado",
                    previous_lambda_bc=lambda_bc,
                )
            lambda_bc = 0.0

        trainer_cfg = FieldTrainerConfig(
            epochs=int(self.exp_config.epochs),
            steps_per_epoch=max(1, int(self.exp_config.steps_per_epoch)),
            batch_size=max(1, int(self.exp_config.batch_size)),
            n_critic=max(1, int(self.exp_config.n_critic)),
            disc_update_every=max(1, int(getattr(self.exp_config, "disc_update_every", 1))),
            gen_lr=float(self.exp_config.gen_lr),
            disc_lr=float(self.exp_config.disc_lr),
            betas=tuple(self.exp_config.betas),
            weight_decay=float(self.exp_config.weight_decay),
            lambda_adv1=float(self.exp_config.lambda_adv1),
            lambda_adv2=float(self.exp_config.lambda_adv2),
            lambda_pde=float(self.exp_config.lambda_pde),
            lambda_bc=lambda_bc,
            lambda_neumann=float(getattr(self.exp_config, "lambda_neumann", 0.0)),
            lambda_gp=float(self.exp_config.lambda_gp),
            use_wgan_gp=bool(self.exp_config.use_wgan_gp),
            d1_real_noise_std=float(self.exp_config.d1_real_noise_std),
            d2_pair_noise_std=float(self.exp_config.d2_pair_noise_std),
            critic_drift=float(self.exp_config.critic_drift),
            max_critic_gap=float(getattr(self.exp_config, "max_critic_gap", 8.0)),
            critic_gap_penalty=float(getattr(self.exp_config, "critic_gap_penalty", 0.05)),
            residual_tanh_scale=float(self.exp_config.residual_tanh_scale),
            use_tanh_on_residual=bool(self.exp_config.use_tanh_on_residual),
            dynamic_adv_balance=bool(getattr(self.exp_config, "dynamic_adv_balance", True)),
            target_adv_over_pde=float(getattr(self.exp_config, "target_adv_over_pde", 0.25)),
            adv_scale_ema_beta=float(getattr(self.exp_config, "adv_scale_ema_beta", 0.9)),
            adv_scale_min=float(getattr(self.exp_config, "adv_scale_min", 0.25)),
            adv_scale_max=float(getattr(self.exp_config, "adv_scale_max", 50.0)),
            pde_norm_ema_beta=float(getattr(self.exp_config, "pde_norm_ema_beta", 0.99)),
            grad_clip=float(self.exp_config.max_grad_norm),
            generator_mode=str(self.exp_config.generator_mode),
            use_reference_discriminator=bool(
                getattr(self.exp_config, "use_reference_discriminator", True)
            ),
            save_frequency=max(0, int(getattr(self.exp_config, "save_frequency", 0))),
            checkpoint_dir=str(self.checkpoint_dir),
            disc_lr_d1=getattr(self.exp_config, "disc_lr_d1", None),
            disc_lr_d2=getattr(self.exp_config, "disc_lr_d2", None),
            lambda_gp_d1=getattr(self.exp_config, "lambda_gp_d1", None),
            lambda_gp_d2=getattr(self.exp_config, "lambda_gp_d2", None),
            d1_real_residual_mode=str(getattr(self.exp_config, "d1_real_residual_mode", "reference")),
            lambda_pde_raw=float(getattr(self.exp_config, "lambda_pde_raw", 0.0)),
            residual_mean_abs_target=float(getattr(self.exp_config, "residual_mean_abs_target", 0.0)),
            adv_residual_gate_target=float(getattr(self.exp_config, "adv_residual_gate_target", 0.0)),
            adv_residual_gate_min=float(getattr(self.exp_config, "adv_residual_gate_min", 0.1)),
            adv_residual_gate_hysteresis=bool(
                getattr(self.exp_config, "adv_residual_gate_hysteresis", False)
            ),
            adv_residual_gate_off_threshold=float(
                getattr(self.exp_config, "adv_residual_gate_off_threshold", 0.0)
            ),
            adv_residual_gate_power=float(
                getattr(self.exp_config, "adv_residual_gate_power", 1.0)
            ),
            adv_warmup_epochs=max(0, int(getattr(self.exp_config, "adv_warmup_epochs", 0))),
            critic_pause_on_overgap=bool(getattr(self.exp_config, "critic_pause_on_overgap", False)),
            critic_pause_gap_factor=float(getattr(self.exp_config, "critic_pause_gap_factor", 1.25)),
            critic_resume_gap_factor=float(
                getattr(self.exp_config, "critic_resume_gap_factor", 0.75)
            ),
            adv_stagnation_boost=bool(getattr(self.exp_config, "adv_stagnation_boost", False)),
            adv_stagnation_patience=max(
                1, int(getattr(self.exp_config, "adv_stagnation_patience", 40))
            ),
            adv_stagnation_rel_tol=max(
                0.0, float(getattr(self.exp_config, "adv_stagnation_rel_tol", 1e-3))
            ),
            adv_stagnation_boost_factor=float(
                getattr(self.exp_config, "adv_stagnation_boost_factor", 1.15)
            ),
            adv_stagnation_min_gate=float(
                getattr(self.exp_config, "adv_stagnation_min_gate", 0.5)
            ),
            adv_stagnation_cooldown=max(
                0, int(getattr(self.exp_config, "adv_stagnation_cooldown", 5))
            ),
            adv_progressive_enable=False,
            adv_progressive_start_epoch=0,
            adv_progressive_ramp_epochs=max(
                1, int(getattr(self.exp_config, "adv_progressive_ramp_epochs", 80))
            ),
            adv_progressive_max_multiplier=max(
                1.0,
                float(getattr(self.exp_config, "adv_progressive_max_multiplier", 2.5)),
            ),
            adv_progressive_power=max(
                1e-6,
                float(getattr(self.exp_config, "adv_progressive_power", 1.0)),
            ),
            pde_corner_sampling_ratio=float(
                np.clip(getattr(self.exp_config, "pde_corner_sampling_ratio", 0.0), 0.0, 0.95)
            ),
            pde_corner_band_points=max(
                1, int(getattr(self.exp_config, "pde_corner_band_points", 2))
            ),
            precision_refine_enable=False,
            precision_refine_start_epoch=max(
                0, int(getattr(self.exp_config, "precision_refine_start_epoch", 0))
            ),
            precision_refine_n_critic=max(
                1, int(getattr(self.exp_config, "precision_refine_n_critic", self.exp_config.n_critic))
            ),
            precision_refine_n_critic_ramp_epochs=max(
                1, int(getattr(self.exp_config, "precision_refine_n_critic_ramp_epochs", 1))
            ),
            precision_refine_lambda_pde_max_scale=float(
                np.clip(
                    getattr(self.exp_config, "precision_refine_lambda_pde_max_scale", 1.0),
                    1e-6,
                    1.0,
                )
            ),
            adaptive_lambda_pde=bool(getattr(self.exp_config, "adaptive_lambda_pde", True)),
            residual_tolerance_target=float(getattr(self.exp_config, "residual_tolerance_target", 1e-3)),
            residual_scale_reference=float(getattr(self.exp_config, "residual_scale_reference", 1.0)),
            lambda_pde_growth_exponent=float(getattr(self.exp_config, "lambda_pde_growth_exponent", 0.5)),
            lambda_pde_min=float(getattr(self.exp_config, "lambda_pde_min", 1.0)),
            lambda_pde_max=float(getattr(self.exp_config, "lambda_pde_max", 500.0)),
            lambda_pde_ema_beta=float(getattr(self.exp_config, "lambda_pde_ema_beta", 0.9)),
            gradnorm_balance=bool(getattr(self.exp_config, "gradnorm_balance", True)),
            gradnorm_target_adv_to_pde=float(getattr(self.exp_config, "gradnorm_target_adv_to_pde", 0.25)),
            gradnorm_ema_beta=float(getattr(self.exp_config, "gradnorm_ema_beta", 0.9)),
            gradnorm_scale_min=float(getattr(self.exp_config, "gradnorm_scale_min", 0.05)),
            gradnorm_scale_max=float(getattr(self.exp_config, "gradnorm_scale_max", 1.0)),
            divergence_window=max(4, int(getattr(self.exp_config, "divergence_window", 20))),
            divergence_ratio_threshold=float(getattr(self.exp_config, "divergence_ratio_threshold", 1.3)),
            divergence_patience=max(1, int(getattr(self.exp_config, "divergence_patience", 2))),
            lr_drop_factor=float(getattr(self.exp_config, "lr_drop_factor", 0.1)),
            max_lr_drops=max(0, int(getattr(self.exp_config, "max_lr_drops", 3))),
            plateau_scheduler_enabled=bool(getattr(self.exp_config, "plateau_scheduler_enabled", False)),
            plateau_metric_key=str(getattr(self.exp_config, "plateau_metric_key", "g_residual_mean_abs")),
            plateau_mode=str(getattr(self.exp_config, "plateau_mode", "min")),
            plateau_patience=max(1, int(getattr(self.exp_config, "plateau_patience", 25))),
            plateau_factor=float(getattr(self.exp_config, "plateau_factor", 0.5)),
            plateau_min_delta=max(0.0, float(getattr(self.exp_config, "plateau_min_delta", 1e-4))),
            plateau_cooldown=max(0, int(getattr(self.exp_config, "plateau_cooldown", 5))),
            plateau_max_drops=max(0, int(getattr(self.exp_config, "plateau_max_drops", 4))),
            plateau_reduce_discriminators=bool(getattr(self.exp_config, "plateau_reduce_discriminators", False)),
            activation_abs_limit=float(getattr(self.exp_config, "activation_abs_limit", 1e6)),
            residual_hist_bins=max(4, int(getattr(self.exp_config, "residual_hist_bins", 12))),
            physics_refine_enable=bool(getattr(self.exp_config, "physics_refine_enable", False)),
            physics_refine_min_train_epochs=max(
                0, int(getattr(self.exp_config, "physics_refine_min_train_epochs", 50))
            ),
            physics_refine_steps=max(0, int(getattr(self.exp_config, "physics_refine_steps", 0))),
            physics_refine_lr=float(getattr(self.exp_config, "physics_refine_lr", 2.0e-5)),
            physics_refine_batch_size=max(
                1, int(getattr(self.exp_config, "physics_refine_batch_size", 8))
            ),
            physics_refine_lambda_data=max(
                0.0, float(getattr(self.exp_config, "physics_refine_lambda_data", 1.0e-6))
            ),
            physics_refine_lambda_bc=max(
                0.0, float(getattr(self.exp_config, "physics_refine_lambda_bc", 1.0))
            ),
            physics_refine_lambda_neumann=max(
                0.0, float(getattr(self.exp_config, "physics_refine_lambda_neumann", 1.0))
            ),
            physics_refine_patience=max(
                0, int(getattr(self.exp_config, "physics_refine_patience", 120))
            ),
            physics_refine_min_delta=max(
                0.0, float(getattr(self.exp_config, "physics_refine_min_delta", 1.0e-5))
            ),
            neumann_dy=hy,
            early_stop_on_nonfinite=bool(getattr(self.exp_config, "early_stop_on_nonfinite", True)),
        )

        if str(self.exp_config.generator_mode) == "deterministic_adversarial":
            # Em modo determinístico, lotes muito grandes e duplicados não trazem ganho.
            trainer_cfg.batch_size = max(2, min(8, trainer_cfg.batch_size))
            if self.logger:
                self.logger.info(
                    "Batch ajustado para modo deterministico",
                    batch_size=trainer_cfg.batch_size,
                )

        trainer = FieldPIGANTrainer(
            generator=self.generator,
            discriminator=self.discriminator,
            laplacian=self.laplacian,
            base_field=self.base_field,  # type: ignore[arg-type]
            phi_mask=self.phi_mask,  # type: ignore[arg-type]
            coord_field=self.coord_field if use_coords else None,  # type: ignore[arg-type]
            reference_field=self.reference_field,  # type: ignore[arg-type]
            interior_mask=self.interior_mask,  # type: ignore[arg-type]
            boundary_mask=self.boundary_mask,  # type: ignore[arg-type]
            config=trainer_cfg,
            device=self.device,
            logger=self.logger,
            neumann_mask=self.neumann_mask,  # type: ignore[arg-type]
        )
        return trainer

    def _evaluate(self) -> Dict[str, float]:
        """
        Avalia o modelo atual comparando a predição média com a referência FDM.

        Retorno:
            Dicionário com métricas calculadas (MAE, RMSE, R2, PDE residual e
            erro L2 relativo final contra a referência FDM no grid atual).

        Exceções:
            RuntimeError: Se o treinador ou campos necessários não estão prontos.
        """
        if self.trainer is None or self.laplacian is None:
            raise RuntimeError("Trainer e laplaciano são obrigatórios para avaliação.")
        if (
            self.reference_field is None
            or self.interior_mask is None
            or self.boundary_mask is None
            or self.neumann_mask is None
        ):
            raise RuntimeError("Referência e máscaras são obrigatórias para avaliação.")

        num_eval_samples = 1
        if int(self.exp_config.latent_dim) > 0:
            num_eval_samples = max(1, min(32, int(self.exp_config.analysis_num_samples)))

        preds = self.trainer.predict(num_samples=num_eval_samples)
        pred_mean = preds.mean(dim=0, keepdim=True)
        self._last_eval_pred = pred_mean.detach()
        metrics = compute_field_metrics(
            pred_mean,
            self.reference_field,
            self.laplacian,
            self.interior_mask,
            self.boundary_mask,
            self.neumann_mask,
            neumann_dy=float(self.exp_config.LY) / float(int(self.exp_config.grid_size_y) - 1),
        )
        metrics["relative_l2_error_vs_fdm"] = float(
            metrics.get("relative_l2_error", 0.0)
        )
        return metrics

    @staticmethod
    def _compute_adversarial_health(history: List[Dict[str, float]]) -> Dict[str, float]:
        if not history:
            return {}

        def _series(key: str) -> np.ndarray:
            return np.asarray([float(item.get(key, np.nan)) for item in history], dtype=float)

        def _finite_ratio(arr: np.ndarray, predicate: Any) -> float:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 0.0
            return float(np.mean(predicate(finite)))

        paused = _series("critics_paused")
        reduced = _series("critics_reduced")
        disc_updates = _series("disc_updates_executed")
        d1_total = _series("d1_total")
        d2_total = _series("d2_total")
        adv_gate = _series("g_adv_gate")
        adv_gate_enabled = _series("g_adv_gate_enabled")
        adv_boost = _series("g_adv_boost_multiplier")
        adv_progressive = _series("g_adv_progressive_multiplier")
        adv_over_pde = _series("g_adv_over_pde")
        residual_mean = _series("g_residual_mean_abs")

        paused_ratio_flag = _finite_ratio(paused, lambda x: x > 0.5)
        paused_ratio_no_update = _finite_ratio(disc_updates, lambda x: x <= 0.0)
        disc_updates_finite = disc_updates[np.isfinite(disc_updates)]

        health = {
            "epochs": int(len(history)),
            "critics_paused_ratio": paused_ratio_no_update,
            "critics_pause_flag_ratio": paused_ratio_flag,
            "critics_reduced_ratio": _finite_ratio(reduced, lambda x: x > 0.5),
            "d1_nonzero_ratio": _finite_ratio(d1_total, lambda x: np.abs(x) > 1e-12),
            "d2_nonzero_ratio": _finite_ratio(d2_total, lambda x: np.abs(x) > 1e-12),
            "disc_update_ratio": _finite_ratio(disc_updates, lambda x: x > 0.0),
            "disc_updates_mean": float(np.mean(disc_updates_finite)) if disc_updates_finite.size else 0.0,
            "adv_gate_enabled_ratio": _finite_ratio(adv_gate_enabled, lambda x: x > 0.5),
            "adv_gate_open_ratio": _finite_ratio(adv_gate, lambda x: x >= 0.95),
            "adv_boost_ratio": _finite_ratio(adv_boost, lambda x: x > 1.001),
            "adv_progressive_ratio": _finite_ratio(adv_progressive, lambda x: x > 1.001),
            "adv_gate_start": float(adv_gate[0]) if adv_gate.size else 0.0,
            "adv_gate_end": float(adv_gate[-1]) if adv_gate.size else 0.0,
            "adv_progressive_start": float(adv_progressive[0]) if adv_progressive.size else 1.0,
            "adv_progressive_end": float(adv_progressive[-1]) if adv_progressive.size else 1.0,
            "adv_over_pde_start": float(adv_over_pde[0]) if adv_over_pde.size else 0.0,
            "adv_over_pde_end": float(adv_over_pde[-1]) if adv_over_pde.size else 0.0,
        }
        finite_gate_mask = np.isfinite(adv_gate)
        finite_gate = adv_gate[finite_gate_mask]
        if finite_gate.size > 0:
            finite_epochs = _series("epoch")[finite_gate_mask]
            open_mask = finite_gate >= 0.95
            partial_mask = finite_gate > 1e-6
            health["adv_gate_closed_ratio"] = float(np.mean(finite_gate <= 1e-6))
            health["adv_gate_mean"] = float(np.mean(finite_gate))
            health["adv_gate_min"] = float(np.min(finite_gate))
            health["adv_gate_max"] = float(np.max(finite_gate))
            health["adv_gate_ever_opened"] = bool(np.any(open_mask))
            health["adv_gate_never_opened"] = bool(not np.any(open_mask))
            health["adv_gate_first_open_epoch"] = (
                float(finite_epochs[open_mask][0]) if np.any(open_mask) else None
            )
            health["adv_gate_first_nonzero_epoch"] = (
                float(finite_epochs[partial_mask][0]) if np.any(partial_mask) else None
            )
        if residual_mean.size > 0:
            health["g_residual_mean_abs_start"] = float(residual_mean[0])
            health["g_residual_mean_abs_end"] = float(residual_mean[-1])
        return health

    @staticmethod
    def _compute_training_gain(
        *,
        baseline: Dict[str, float],
        final: Dict[str, float],
    ) -> Dict[str, float]:
        monitored = ("pde_residual_mean", "pde_residual_max", "rmse", "relative_l2_error")
        gain: Dict[str, float] = {}
        for key in monitored:
            base_val = float(baseline.get(key, np.nan))
            final_val = float(final.get(key, np.nan))
            gain[f"{key}_baseline"] = base_val
            gain[f"{key}_final"] = final_val
            if np.isfinite(base_val) and np.isfinite(final_val):
                gain[f"{key}_improvement_factor"] = float(base_val / max(final_val, 1e-16))
            else:
                gain[f"{key}_improvement_factor"] = float("nan")
        return gain

    @staticmethod
    def _detect_stagnation_epoch_from_history(
        history: List[Dict[str, Any]],
        *,
        metric_key: str,
        patience: int,
        rel_tol: float,
        min_epoch: int,
    ) -> Optional[int]:
        if not history:
            return None

        best = float("inf")
        stale_count = 0
        stale_epochs: List[int] = []
        patience = max(1, int(patience))
        rel_tol = max(0.0, float(rel_tol))
        min_epoch = max(1, int(min_epoch))

        for idx, item in enumerate(history):
            raw_metric = item.get(metric_key, np.nan)
            try:
                metric = float(raw_metric)
            except Exception:
                continue
            if not np.isfinite(metric):
                continue

            epoch = int(item.get("epoch", idx + 1))
            if epoch < min_epoch:
                continue

            if not np.isfinite(best):
                best = metric
                stale_count = 0
                stale_epochs.clear()
                continue

            improve_threshold = best * (1.0 - rel_tol)
            if metric < improve_threshold:
                best = metric
                stale_count = 0
                stale_epochs.clear()
                continue

            stale_count += 1
            stale_epochs.append(epoch)
            if stale_count >= patience:
                return int(stale_epochs[0])
        return None

    def _configure_adv_progressive_from_history(self) -> None:
        if self.trainer is None:
            return
        if not bool(getattr(self.exp_config, "adv_progressive_from_history", False)):
            self.trainer.cfg.adv_progressive_enable = False
            return

        configured_history = getattr(self.exp_config, "adv_progressive_history_path", None)
        if configured_history:
            history_path = Path(str(configured_history))
            if not history_path.is_absolute():
                history_path = self.runs_dir / history_path
        else:
            history_path = self.results_dir / "training_history.json"

        if not history_path.exists():
            self.trainer.cfg.adv_progressive_enable = False
            if self.logger:
                self.logger.info(
                    "Lambda_adv progressivo desativado (historico nao encontrado)",
                    history_path=str(history_path),
                )
            return

        try:
            raw = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.trainer.cfg.adv_progressive_enable = False
            if self.logger:
                self.logger.warning(
                    "Falha ao carregar training_history para lambda_adv progressivo",
                    history_path=str(history_path),
                    error=str(exc),
                )
            return

        if not isinstance(raw, list):
            self.trainer.cfg.adv_progressive_enable = False
            if self.logger:
                self.logger.warning(
                    "Formato invalido de training_history para lambda_adv progressivo",
                    history_path=str(history_path),
                    expected="list[dict]",
                )
            return

        history: List[Dict[str, Any]] = [item for item in raw if isinstance(item, dict)]
        metric_key = str(getattr(self.exp_config, "adv_progressive_metric_key", "g_residual_mean_abs"))
        patience = int(
            getattr(
                self.exp_config,
                "adv_progressive_stagnation_patience",
                getattr(self.exp_config, "adv_stagnation_patience", 40),
            )
        )
        rel_tol = float(
            getattr(
                self.exp_config,
                "adv_progressive_stagnation_rel_tol",
                getattr(self.exp_config, "adv_stagnation_rel_tol", 1e-3),
            )
        )
        min_epoch = int(getattr(self.exp_config, "adv_progressive_min_epoch", 1))
        stagnation_epoch = self._detect_stagnation_epoch_from_history(
            history,
            metric_key=metric_key,
            patience=patience,
            rel_tol=rel_tol,
            min_epoch=min_epoch,
        )

        if stagnation_epoch is None:
            self.trainer.cfg.adv_progressive_enable = False
            if self.logger:
                self.logger.info(
                    "Lambda_adv progressivo desativado (sem estagnacao detectada)",
                    history_path=str(history_path),
                    metric_key=metric_key,
                    patience=patience,
                    rel_tol=rel_tol,
                )
            return

        shift = int(getattr(self.exp_config, "adv_progressive_epoch_shift", 0))
        start_epoch = max(1, int(stagnation_epoch + shift))
        ramp_epochs = max(1, int(getattr(self.exp_config, "adv_progressive_ramp_epochs", 80)))
        max_multiplier = max(
            1.0,
            float(getattr(self.exp_config, "adv_progressive_max_multiplier", 2.5)),
        )
        power = max(1e-6, float(getattr(self.exp_config, "adv_progressive_power", 1.0)))

        self.trainer.cfg.adv_progressive_enable = True
        self.trainer.cfg.adv_progressive_start_epoch = int(start_epoch)
        self.trainer.cfg.adv_progressive_ramp_epochs = int(ramp_epochs)
        self.trainer.cfg.adv_progressive_max_multiplier = float(max_multiplier)
        self.trainer.cfg.adv_progressive_power = float(power)

        if self.logger:
            self.logger.info(
                "Lambda_adv progressivo configurado por estagnacao do historico",
                history_path=str(history_path),
                metric_key=metric_key,
                stagnation_epoch=int(stagnation_epoch),
                start_epoch=int(start_epoch),
                ramp_epochs=int(ramp_epochs),
                max_multiplier=float(max_multiplier),
                power=float(power),
            )

    def _configure_precision_refine(self) -> None:
        if self.trainer is None:
            return

        if not bool(getattr(self.exp_config, "precision_refine_enable", False)):
            self.trainer.cfg.precision_refine_enable = False
            return

        start_epoch = max(0, int(getattr(self.exp_config, "precision_refine_start_epoch", 0)))
        if start_epoch <= 0 and bool(
            getattr(self.exp_config, "precision_refine_use_adv_progressive_start", True)
        ):
            start_epoch = max(0, int(getattr(self.trainer.cfg, "adv_progressive_start_epoch", 0)))
        if start_epoch <= 0:
            start_epoch = max(1, int(round(0.65 * float(self.exp_config.epochs))))

        n_critic_target = max(
            int(self.trainer.cfg.n_critic),
            int(getattr(self.exp_config, "precision_refine_n_critic", self.trainer.cfg.n_critic)),
        )
        ramp_epochs = max(
            1,
            int(getattr(self.exp_config, "precision_refine_n_critic_ramp_epochs", 1)),
        )
        pde_max_scale = float(
            np.clip(
                getattr(self.exp_config, "precision_refine_lambda_pde_max_scale", 1.0),
                1e-6,
                1.0,
            )
        )

        self.trainer.cfg.precision_refine_enable = True
        self.trainer.cfg.precision_refine_start_epoch = int(start_epoch)
        self.trainer.cfg.precision_refine_n_critic = int(n_critic_target)
        self.trainer.cfg.precision_refine_n_critic_ramp_epochs = int(ramp_epochs)
        self.trainer.cfg.precision_refine_lambda_pde_max_scale = float(pde_max_scale)

        if self.logger:
            self.logger.info(
                "Precision refine configurado",
                start_epoch=int(start_epoch),
                n_critic_base=int(self.trainer.cfg.n_critic),
                n_critic_target=int(n_critic_target),
                n_critic_ramp_epochs=int(ramp_epochs),
                lambda_pde_max_scale=float(pde_max_scale),
            )

    def _save_results(
        self, metrics: Dict[str, float], history: List[Dict[str, float]]
    ) -> None:
        """
        Salva os tensores de resultado, métricas JSON, histórico e gera plots.

        Parâmetros:
            metrics: Dicionário de métricas finais.
            history: Lista de dicionários com o histórico de treinamento por época.
        """
        if self.reference_field is None:
            return
        if self.trainer is None or self.laplacian is None:
            return

        if self._last_eval_pred is not None:
            pred_tensor = self._last_eval_pred.to(self.device)
        else:
            num_eval_samples = 1
            if int(self.exp_config.latent_dim) > 0:
                num_eval_samples = max(1, min(32, int(self.exp_config.analysis_num_samples)))
            preds = self.trainer.predict(num_samples=num_eval_samples)
            pred_tensor = preds.mean(dim=0, keepdim=True)

        pred = pred_tensor[0, 0].detach().cpu().numpy()
        ref = self.reference_field[0, 0].detach().cpu().numpy()
        residual_tensor = self.laplacian(pred_tensor) * self.interior_mask
        residual = residual_tensor[0, 0].detach().cpu().numpy()

        np.save(self.results_dir / "temperature_pred.npy", pred)
        np.save(self.results_dir / "temperature_ref_fdm.npy", ref)
        np.save(self.results_dir / "pde_residual.npy", residual)
        l2_vs_fdm = float(
            metrics.get("relative_l2_error_vs_fdm", metrics.get("relative_l2_error", 0.0))
        )
        with open(self.results_dir / "l2_relative_vs_fdm.txt", "w", encoding="utf-8") as f:
            f.write(f"{l2_vs_fdm:.16e}\n")

        with open(self.results_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(self.results_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        health = self._compute_adversarial_health(history)
        if health:
            with open(self.results_dir / "adversarial_health.json", "w", encoding="utf-8") as f:
                json.dump(health, f, indent=2)
        if self._baseline_metrics is not None:
            with open(self.results_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
                json.dump(self._baseline_metrics, f, indent=2)
            gain = self._compute_training_gain(
                baseline=self._baseline_metrics,
                final=metrics,
            )
            with open(self.results_dir / "training_gain.json", "w", encoding="utf-8") as f:
                json.dump(gain, f, indent=2)

        if bool(getattr(self.exp_config, "generate_plots", True)):
            self._save_plots(pred, ref, residual, history, metrics)
        elif self.logger:
            self.logger.info("Geracao de plots basicos desativada por configuracao")

        if bool(getattr(self.exp_config, "generate_extended_plots", True)):
            self._save_requested_plots(history)
        elif self.logger:
            self.logger.info("Geracao de plots adicionais desativada por configuracao")

    def _collect_ensemble_data(self) -> Dict[str, np.ndarray]:
        """
        Gera amostras do modelo e calcula estatísticas (média, incerteza, quantis).

        Calcula a média, desvio padrão, intervalo de confiança de 95% e
        coeficiente de variação para o campo e resíduos.

        Retorno:
            Dicionário com arrays NumPy contendo os dados estatísticos espaciais.

        Exceções:
            RuntimeError: Se os modelos base não foram preparados.
        """
        if self.trainer is None or self.laplacian is None or self.reference_field is None:
            raise RuntimeError("Missing trainer/laplacian/reference for ensemble plots.")

        n_samples = max(10, min(60, int(self.exp_config.analysis_num_samples)))
        preds = self.trainer.predict(num_samples=n_samples)  # [S,1,H,W]
        residuals = self.laplacian(preds)  # [S,1,H,W]

        mean_pred = preds.mean(dim=0, keepdim=True)  # [1,1,H,W]
        std_pred = preds.std(dim=0, keepdim=True, unbiased=False)
        q025 = torch.quantile(preds, q=0.025, dim=0, keepdim=True)
        q975 = torch.quantile(preds, q=0.975, dim=0, keepdim=True)
        interval95 = q975 - q025
        cv = std_pred / (mean_pred.abs() + 1e-8)

        res_mean = residuals.mean(dim=0, keepdim=True)
        res_std = residuals.std(dim=0, keepdim=True, unbiased=False)

        s, _, h, w = preds.shape
        center_samples = preds[:, 0, h // 2, w // 2]

        x_coords = np.linspace(0.0, float(self.exp_config.LX), w, dtype=np.float32)
        y_coords = np.linspace(0.0, float(self.exp_config.LY), h, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        return {
            "pred_samples": preds[:, 0].detach().cpu().numpy(),      # [S,H,W]
            "res_samples": residuals[:, 0].detach().cpu().numpy(),   # [S,H,W]
            "pred_mean": mean_pred[0, 0].detach().cpu().numpy(),     # [H,W]
            "pred_std": std_pred[0, 0].detach().cpu().numpy(),
            "pred_cv": cv[0, 0].detach().cpu().numpy(),
            "interval95": interval95[0, 0].detach().cpu().numpy(),
            "res_mean": res_mean[0, 0].detach().cpu().numpy(),
            "res_std": res_std[0, 0].detach().cpu().numpy(),
            "center_samples": center_samples.detach().cpu().numpy(),  # [S]
            "ref": self.reference_field[0, 0].detach().cpu().numpy(),
            "xx": xx,
            "yy": yy,
        }

    @staticmethod
    def _configure_spatial_axis(ax: Any) -> None:
        """
        Configura os eixos para visualização espacial (aspect ratio e labels).

        Parâmetros:
            ax: Eixo do Matplotlib a ser configurado.
        """
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _plot_spatial_heatmap(
        self,
        *,
        ax: Any,
        values: np.ndarray,
        title: str,
        cmap: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Cria um mapa de calor (heatmap) 2D com barra de cores.

        Parâmetros:
            ax: Eixo do Matplotlib.
            values: Matriz de valores 2D.
            title: Título do gráfico.
            cmap: Mapa de cores (colormap).
            vmin: Valor mínimo para escala.
            vmax: Valor máximo para escala.
        """
        im = ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        self._configure_spatial_axis(ax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_spatial_scatter(
        self,
        *,
        ax: Any,
        xx: np.ndarray,
        yy: np.ndarray,
        values: np.ndarray,
        title: str,
        cmap: str,
        marker_size: float = 7.0,
        alpha: float = 0.8,
    ) -> None:
        """
        Cria um gráfico de dispersão (scatter) colorido pelas coordenadas.

        Parâmetros:
            ax: Eixo do Matplotlib.
            xx: Coordenadas X.
            yy: Coordenadas Y.
            values: Valores para a cor dos pontos.
            title: Título do gráfico.
            cmap: Mapa de cores.
            marker_size: Tamanho do marcador.
            alpha: Transparência.
        """
        sc = ax.scatter(
            xx.flatten(),
            yy.flatten(),
            c=values.flatten(),
            s=marker_size,
            cmap=cmap,
            alpha=alpha,
        )
        ax.set_title(title)
        self._configure_spatial_axis(ax)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    def _save_plot_ensemble_predictions(self, data: Dict[str, np.ndarray]) -> None:
        """
        Gera e salva o quadro de predições do ensemble (média, erro, incerteza).

        Parâmetros:
            data: Dicionário com arrays coletados pelo ensemble.
        """
        mean_pred = data["pred_mean"]
        std_pred = data["pred_std"]
        cv = data["pred_cv"]
        ref = data["ref"]
        center_samples = data["center_samples"]
        error = np.abs(mean_pred - ref)

        vmin_t = float(min(np.min(mean_pred), np.min(ref)))
        vmax_t = float(max(np.max(mean_pred), np.max(ref)))

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("ensemble_predictions", fontsize=16, fontweight="bold")

        self._plot_spatial_heatmap(
            ax=axes[0, 0],
            values=ref,
            title="Solucao FVM (referencia numerica)",
            cmap="plasma",
            vmin=vmin_t,
            vmax=vmax_t,
        )
        self._plot_spatial_heatmap(
            ax=axes[0, 1],
            values=mean_pred,
            title="Predicao media PI-GAN",
            cmap="plasma",
            vmin=vmin_t,
            vmax=vmax_t,
        )
        self._plot_spatial_heatmap(
            ax=axes[0, 2],
            values=error,
            title="Erro absoluto PI-GAN",
            cmap="Reds",
        )
        self._plot_spatial_heatmap(
            ax=axes[1, 0],
            values=std_pred,
            title="Incerteza epistemica PI-GAN",
            cmap="viridis",
        )
        self._plot_spatial_heatmap(
            ax=axes[1, 1],
            values=cv,
            title="Coeficiente de variacao PI-GAN",
            cmap="magma",
        )

        center_samples_hist = np.asarray(center_samples, dtype=np.float64)
        finite_center = center_samples_hist[np.isfinite(center_samples_hist)]
        if finite_center.size == 0:
            finite_center = np.array([0.0], dtype=np.float64)
        hist_bins = min(30, max(1, int(finite_center.size)))
        axes[1, 2].hist(finite_center, bins=hist_bins, color="#2E86AB", alpha=0.75, edgecolor="black")
        axes[1, 2].axvline(ref[ref.shape[0] // 2, ref.shape[1] // 2], color="red", linestyle="--", linewidth=2)
        axes[1, 2].set_title("Distribuicao no centro")
        axes[1, 2].set_xlabel("Temperatura")
        axes[1, 2].set_ylabel("Frequencia")
        axes[1, 2].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plots_dir / "ensemble_predictions.png", dpi=int(self.exp_config.dpi))
        plt.close(fig)

    def _save_plot_gan_quality_metrics(self, data: Dict[str, np.ndarray]) -> None:
        """
        Gera e salva métricas de qualidade típicas de GAN (diversidade, convergência).

        Parâmetros:
            data: Dicionário com amostras e estatísticas do ensemble.
        """
        pred_samples = data["pred_samples"]  # [S,H,W]
        res_samples = data["res_samples"]    # [S,H,W]
        pred_std = data["pred_std"]
        xx = data["xx"]
        yy = data["yy"]

        s = pred_samples.shape[0]
        diversity = []
        for k in range(2, s + 1):
            diversity.append(float(np.var(pred_samples[:k], axis=0).mean()))
        diversity_x = np.arange(2, s + 1)

        residual_unc = np.std(res_samples, axis=0)
        all_predictions = pred_samples.reshape(-1)

        # Convergência estatística em 4 pontos
        h, w = pred_samples.shape[1], pred_samples.shape[2]
        points = [(h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
        cum_curves = []
        for py, px in points:
            vals = pred_samples[:, py, px]
            cum = np.cumsum(vals) / np.arange(1, s + 1)
            cum_curves.append(cum)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("gan_quality_metrics", fontsize=16, fontweight="bold")

        axes[0, 0].plot(diversity_x, diversity, color="#1f77b4", linewidth=2)
        axes[0, 0].set_title("Evolução da Diversidade PI-GAN")
        axes[0, 0].set_xlabel("Número de amostras")
        axes[0, 0].set_ylabel("Variância média")
        axes[0, 0].grid(True, alpha=0.3)

        self._plot_spatial_heatmap(
            ax=axes[0, 1],
            values=residual_unc,
            title="Incerteza Residual PI-GAN",
            cmap="inferno",
        )

        axes[0, 2].hist(all_predictions, bins=50, color="#A23B72", alpha=0.75, edgecolor="black")
        axes[0, 2].set_title("Distribuição das Predições PI-GAN")
        axes[0, 2].set_xlabel("Temperatura")
        axes[0, 2].set_ylabel("Frequência")
        axes[0, 2].grid(True, alpha=0.3)

        self._plot_spatial_scatter(
            ax=axes[1, 0],
            xx=xx,
            yy=yy,
            values=pred_std,
            title="Variabilidade Espacial PI-GAN",
            cmap="viridis",
        )

        for i, curve in enumerate(cum_curves):
            axes[1, 1].plot(np.arange(1, s + 1), curve, linewidth=2, label=f"Ponto {i+1}")
        axes[1, 1].set_title("Convergência Estatística PI-GAN")
        axes[1, 1].set_xlabel("Número de amostras")
        axes[1, 1].set_ylabel("Média acumulada")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        axes[1, 2].axis("off")

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plots_dir / "gan_quality_metrics.png", dpi=int(self.exp_config.dpi))
        plt.close(fig)

    def _save_plot_physics_consistency(self, data: Dict[str, np.ndarray]) -> None:
        """
        Gera e salva plots de consistência física (resíduos PDE e gradientes).

        Parâmetros:
            data: Dicionário com dados de resíduos e campos preditos.
        """
        res_mean = data["res_mean"]
        res_std = data["res_std"]
        pred_mean = data["pred_mean"]
        res_samples = data["res_samples"]

        dy = float(self.exp_config.LY) / max(1, pred_mean.shape[0] - 1)
        dx = float(self.exp_config.LX) / max(1, pred_mean.shape[1] - 1)
        gy, gx = np.gradient(pred_mean, dy, dx)
        grad_mag = np.sqrt(gx**2 + gy**2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("physics_consistency", fontsize=16, fontweight="bold")

        self._plot_spatial_heatmap(
            ax=axes[0, 0],
            values=np.abs(res_mean),
            title="Resíduo PDE médio PI-GAN",
            cmap="Reds",
        )

        self._plot_spatial_heatmap(
            ax=axes[0, 1],
            values=res_std,
            title="Incerteza do Residuo PDE PI-GAN",
            cmap="magma",
        )

        self._plot_spatial_heatmap(
            ax=axes[1, 0],
            values=grad_mag,
            title="Gradiente medio PI-GAN",
            cmap="viridis",
        )

        axes[1, 1].hist(res_samples.reshape(-1), bins=60, color="#E74C3C", alpha=0.75, edgecolor="black")
        axes[1, 1].set_title("Distribuicao dos residuos PDE (PI-GAN)")
        axes[1, 1].set_xlabel("Resíduo PDE")
        axes[1, 1].set_ylabel("Frequencia")
        axes[1, 1].grid(True, alpha=0.3)

        # Histograma auxiliar em log10 para monitorar cauda de distribuição.
        log_res = np.log10(np.abs(res_samples.reshape(-1)) + 1e-12)
        inset = axes[1, 1].inset_axes([0.55, 0.55, 0.42, 0.40])
        inset.hist(log_res, bins=40, color="#8E44AD", alpha=0.75, edgecolor="black")
        inset.set_title("log10|R|", fontsize=9)
        inset.set_xlabel("log10|Resíduo|", fontsize=8)
        inset.tick_params(axis="both", labelsize=8)
        inset.grid(True, alpha=0.25)

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plots_dir / "physics_consistency.png", dpi=int(self.exp_config.dpi))
        plt.close(fig)

    def _save_plot_training_history_detailed(
        self, history: List[Dict[str, float]]
    ) -> None:
        """
        Gera e salva o histórico detalhado de perdas e métricas durante o treino.

        Parâmetros:
            history: Lista de históricos por época.
        """
        if not history:
            return
        epochs = np.array([h.get("epoch", i + 1) for i, h in enumerate(history)], dtype=float)

        def _series(key: str) -> np.ndarray:
            return np.array([h.get(key, np.nan) for h in history], dtype=float)

        def _amplitude(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return 0.0
            return float(np.max(np.abs(finite)))

        def _plot_if_relevant(
            ax: Any,
            key: str,
            label: str,
            *,
            min_amp: float = 0.0,
            **plot_kwargs: Any,
        ) -> bool:
            vals = _series(key)
            if not np.isfinite(vals).any():
                return False
            if _amplitude(vals) < float(min_amp):
                return False
            ax.plot(epochs, vals, linewidth=1.8, label=label, **plot_kwargs)
            return True

        def _set_epoch_ticks(ax: Any) -> None:
            if epochs.size <= 12:
                ax.set_xticks(np.unique(epochs.astype(int)))

        def _maybe_set_symlog(ax: Any, *keys: str) -> None:
            series = []
            for key in keys:
                vals = _series(key)
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    series.append(np.abs(finite))
            if not series:
                return
            merged = np.concatenate(series)
            positive = merged[merged > 0.0]
            if positive.size < 8:
                return
            vmax = float(np.max(positive))
            vmed = float(np.median(positive))
            if vmax > 200.0 * max(vmed, 1e-12):
                linthresh = max(1e-6, 0.25 * vmed)
                ax.set_yscale("symlog", linthresh=linthresh)

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("training_history", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        g_total_amp = max(_amplitude(_series("g_total")), 1e-8)
        g_min_amp = max(1e-8, 0.01 * g_total_amp)
        g_has_lines = False
        g_has_lines |= _plot_if_relevant(ax, "g_total", "G total")
        g_has_lines |= _plot_if_relevant(ax, "g_adv1", "G adv D1", min_amp=g_min_amp)
        g_has_lines |= _plot_if_relevant(ax, "g_adv2", "G adv D2", min_amp=g_min_amp)
        g_has_lines |= _plot_if_relevant(ax, "g_pde", "G PDE", min_amp=g_min_amp)
        g_has_lines |= _plot_if_relevant(ax, "g_pde_raw", "G PDE bruto", min_amp=g_min_amp)
        g_has_lines |= _plot_if_relevant(ax, "g_bc", "G BC", min_amp=g_min_amp)
        ax.set_title("Perdas do Gerador PI-GAN")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        _set_epoch_ticks(ax)
        _maybe_set_symlog(ax, "g_total", "g_adv1", "g_adv2", "g_pde", "g_pde_raw", "g_bc")
        if g_has_lines:
            ax.legend()

        ax = axes[0, 1]
        d1_total_amp = max(_amplitude(_series("d1_total")), 1e-10)
        d1_min_amp = max(1e-10, 0.01 * d1_total_amp)
        d1_has_lines = False
        d1_has_lines |= _plot_if_relevant(ax, "d1_total", "D1 total")
        d1_has_lines |= _plot_if_relevant(ax, "d1_gap", "D1 gap", min_amp=d1_min_amp)
        d1_has_lines |= _plot_if_relevant(ax, "d1_gp", "D1 GP", min_amp=d1_min_amp)
        d1_has_lines |= _plot_if_relevant(ax, "d1_drift", "D1 drift", min_amp=d1_min_amp)
        ax.set_title("Perdas do Discriminador (D1)")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        _set_epoch_ticks(ax)
        if d1_has_lines:
            ax.legend()

        ax = axes[1, 0]
        d2_total_amp = max(_amplitude(_series("d2_total")), 1e-10)
        d2_min_amp = max(1e-10, 0.01 * d2_total_amp)
        d2_has_lines = False
        d2_has_lines |= _plot_if_relevant(ax, "d2_total", "D2 total")
        d2_has_lines |= _plot_if_relevant(ax, "d2_gap", "D2 gap", min_amp=d2_min_amp)
        d2_has_lines |= _plot_if_relevant(ax, "d2_gp", "D2 GP", min_amp=d2_min_amp)
        d2_has_lines |= _plot_if_relevant(ax, "d2_drift", "D2 drift", min_amp=d2_min_amp)
        ax.set_title("Perda do discriminador (D2)")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        _set_epoch_ticks(ax)
        if d2_has_lines:
            ax.legend()

        ax_g = axes[1, 1]
        ax_d = ax_g.twinx()
        g_vs_d_has = False
        if _plot_if_relevant(ax_g, "g_total", "Gerador", color="#1f77b4"):
            g_vs_d_has = True
        if _plot_if_relevant(ax_d, "d1_total", "D1", color="#ff7f0e"):
            g_vs_d_has = True
        if _plot_if_relevant(ax_d, "d2_total", "D2", color="#2ca02c"):
            g_vs_d_has = True
        ax_g.set_title("Gerador vs Discriminadores (escala separada)")
        ax_g.set_xlabel("Epoca")
        ax_g.set_ylabel("Loss G")
        ax_d.set_ylabel("Loss D")
        ax_g.grid(True, alpha=0.3)
        _set_epoch_ticks(ax_g)
        if g_vs_d_has:
            h1, l1 = ax_g.get_legend_handles_labels()
            h2, l2 = ax_d.get_legend_handles_labels()
            ax_g.legend(h1 + h2, l1 + l2)

        ax = axes[2, 0]
        g_pde = _series("g_pde")
        phys_has_lines = False
        phys_has_lines |= _plot_if_relevant(ax, "g_pde", "G PDE")
        phys_has_lines |= _plot_if_relevant(
            ax,
            "g_pde_raw_penalty",
            "PDE bruto (penalizado)",
            linestyle="-.",
            min_amp=max(1e-8, 0.01 * _amplitude(g_pde)),
        )
        phys_has_lines |= _plot_if_relevant(ax, "g_bc", "G BC", min_amp=max(1e-8, 0.01 * _amplitude(g_pde)))
        phys_has_lines |= _plot_if_relevant(
            ax,
            "g_residual_l2",
            "Resíduo L2 (bruto)",
            linestyle="--",
            min_amp=max(1e-8, 0.01 * _amplitude(g_pde)),
        )
        phys_has_lines |= _plot_if_relevant(
            ax,
            "g_residual_mean_abs",
            "Resíduo médio absoluto (bruto)",
            linestyle=":",
            min_amp=max(1e-8, 0.01 * _amplitude(g_pde)),
        )
        phys_has_lines |= _plot_if_relevant(
            ax,
            "g_residual_max_abs",
            "Resíduo máximo absoluto (bruto)",
            linestyle="-",
            min_amp=max(1e-8, 0.01 * _amplitude(g_pde)),
        )
        phys_has_lines |= _plot_if_relevant(
            ax,
            "g_lambda_pde_dyn",
            "lambda_PDE dinamico",
            linestyle="--",
            min_amp=1e-8,
        )
        g_total_vals = _series("g_total")
        if np.isfinite(g_pde).any() and np.isfinite(g_total_vals).any():
            ratio = g_pde / (np.abs(g_total_vals) + 1e-8)
            ax.plot(epochs, ratio, linewidth=2, linestyle="--", label="PDE/|G_total|")
            phys_has_lines = True
        ax.set_title("Componente fisica")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Valor")
        ax.grid(True, alpha=0.3)
        _set_epoch_ticks(ax)
        _maybe_set_symlog(
            ax,
            "g_pde",
            "g_pde_raw_penalty",
            "g_bc",
            "g_residual_l2",
            "g_residual_mean_abs",
            "g_residual_max_abs",
            "g_lambda_pde_dyn",
        )
        if phys_has_lines:
            ax.legend()

        ax = axes[2, 1]
        res_log_has_lines = False
        if _plot_if_relevant(ax, "g_residual_mean_abs", "Resíduo médio absoluto", color="#d62728"):
            res_log_has_lines = True
        if _plot_if_relevant(ax, "g_residual_max_abs", "Resíduo máximo absoluto", color="#9467bd"):
            res_log_has_lines = True
        if _plot_if_relevant(ax, "g_residual_l2", "Resíduo L2", color="#2ca02c"):
            res_log_has_lines = True
        ax.set_title("Resíduo bruto (escala log)")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Resíduo")
        ax.grid(True, alpha=0.3)
        _set_epoch_ticks(ax)
        ax.set_yscale("log")
        if res_log_has_lines:
            ax.legend()

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plots_dir / "training_history.png", dpi=int(self.exp_config.dpi))
        plt.close(fig)

    def _save_plot_uncertainty_analysis(self, data: Dict[str, np.ndarray]) -> None:
        """
        Gera e salva a análise de incerteza da predição PI-GAN.

        Parâmetros:
            data: Dicionário com dados coletados pelo ensemble.
        """
        mean_pred = data["pred_mean"]
        std_pred = data["pred_std"]
        cv = data["pred_cv"]
        interval95 = data["interval95"]
        xx = data["xx"]
        yy = data["yy"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("uncertainty_analysis", fontsize=16, fontweight="bold")

        self._plot_spatial_heatmap(
            ax=axes[0, 0],
            values=mean_pred,
            title="Temperatura media PI-GAN",
            cmap="plasma",
        )

        self._plot_spatial_heatmap(
            ax=axes[0, 1],
            values=std_pred,
            title="Incerteza epistemica PI-GAN",
            cmap="viridis",
        )

        self._plot_spatial_heatmap(
            ax=axes[0, 2],
            values=cv,
            title="Coeficiente de variacao PI-GAN",
            cmap="magma",
        )

        self._plot_spatial_heatmap(
            ax=axes[1, 0],
            values=interval95,
            title="Largura do Intervalo 95% PI-GAN",
            cmap="coolwarm",
        )

        axes[1, 1].hist(std_pred.reshape(-1), bins=50, color="#8E44AD", alpha=0.75, edgecolor="black")
        axes[1, 1].set_title("Distribuicao de incerteza PI-GAN")
        axes[1, 1].set_xlabel("Desvio padrao")
        axes[1, 1].set_ylabel("Frequencia")
        axes[1, 1].grid(True, alpha=0.3)

        self._plot_spatial_scatter(
            ax=axes[1, 2],
            xx=xx,
            yy=yy,
            values=std_pred,
            title="Mapa Espacial de Incerteza PI-GAN",
            cmap="viridis",
        )

        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(self.plots_dir / "uncertainty_analysis.png", dpi=int(self.exp_config.dpi))
        plt.close(fig)

    def _save_requested_plots(self, history: List[Dict[str, float]]) -> None:
        """
        Orquestra a geração de todos os plots analíticos adicionais.

        Parâmetros:
            history: Histórico de treinamento por época.
        """
        try:
            data = self._collect_ensemble_data()
            self._save_plot_ensemble_predictions(data)
            self._save_plot_gan_quality_metrics(data)
            self._save_plot_physics_consistency(data)
            self._save_plot_training_history_detailed(history)
            self._save_plot_uncertainty_analysis(data)

            if self.logger:
                self.logger.info(
                    "Plots adicionais salvos",
                    ensemble=str(self.plots_dir / "ensemble_predictions.png"),
                    gan_quality=str(self.plots_dir / "gan_quality_metrics.png"),
                    physics_consistency=str(self.plots_dir / "physics_consistency.png"),
                    training_history=str(self.plots_dir / "training_history.png"),
                    uncertainty=str(self.plots_dir / "uncertainty_analysis.png"),
                )
        except Exception as exc:
            if self.logger:
                self.logger.warning("Falha ao gerar plots adicionais", error=str(exc))

    def _save_plots(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        residual: np.ndarray,
        history: List[Dict[str, float]],
        metrics: Dict[str, float],
    ) -> None:
        """
        Gera e salva os plots básicos de comparação de campo e curvas de treino.

        Parâmetros:
            pred: Campo predito (NumPy).
            ref: Campo de referência FDM (NumPy).
            residual: Resíduo PDE (NumPy).
            history: Histórico de treinamento.
            metrics: Métricas calculadas.
        """
        try:
            error = np.abs(pred - ref)

            vmin_t = float(min(np.min(pred), np.min(ref)))
            vmax_t = float(max(np.max(pred), np.max(ref)))

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle("PI-GAN 2D Laplace: Campo e Consistencia Fisica", fontsize=13, fontweight="bold")

            im0 = axes[0, 0].imshow(ref, origin="lower", cmap="plasma", vmin=vmin_t, vmax=vmax_t)
            axes[0, 0].set_title("Referencia FDM")
            plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

            im1 = axes[0, 1].imshow(pred, origin="lower", cmap="plasma", vmin=vmin_t, vmax=vmax_t)
            axes[0, 1].set_title("Predicao PI-GAN")
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

            im2 = axes[1, 0].imshow(error, origin="lower", cmap="Reds")
            axes[1, 0].set_title("|Erro|")
            plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

            im3 = axes[1, 1].imshow(residual, origin="lower", cmap="coolwarm")
            axes[1, 1].set_title("Residuo PDE (Laplaciano)")
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

            for ax in axes.flat:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal")

            metrics_text = (
                f"MAE: {metrics.get('mae', 0.0):.3e}\n"
                f"RMSE: {metrics.get('rmse', 0.0):.3e}\n"
                f"MAPE (%): {metrics.get('mape', 0.0):.3e}\n"
                f"R2: {metrics.get('r2', 0.0):.6f}\n"
                f"L2 rel vs FDM: {metrics.get('relative_l2_error_vs_fdm', metrics.get('relative_l2_error', 0.0)):.3e}\n"
                f"Erro max: {metrics.get('max_error', 0.0):.3e}\n"
                f"Residuo medio: {metrics.get('pde_residual_mean', 0.0):.3e}\n"
                f"Residuo max: {metrics.get('pde_residual_max', 0.0):.3e}\n"
                f"Erro fronteira: {metrics.get('boundary_error', 0.0):.3e}"
            )
            fig.text(0.02, 0.02, metrics_text, fontsize=10, family="monospace")
            fig.tight_layout(rect=[0, 0.06, 1, 0.96])
            fig.savefig(self.plots_dir / "field_comparison.png", dpi=int(self.exp_config.dpi))
            plt.close(fig)

            if history:
                epochs = np.array([h.get("epoch", i + 1) for i, h in enumerate(history)], dtype=float)

                fig2, ax2 = plt.subplots(1, 1, figsize=(11, 5))
                plotted = False
                plotted_series: List[np.ndarray] = []
                curves = [
                    ("g_total", "G total"),
                    ("g_pde", "G PDE"),
                    ("g_pde_raw", "G PDE bruto"),
                    ("g_bc", "G BC"),
                    ("g_residual_l2", "Resíduo L2 (bruto)"),
                    ("g_residual_mean_abs", "Resíduo médio absoluto (bruto)"),
                    ("g_residual_max_abs", "Resíduo máximo absoluto (bruto)"),
                    ("d1_total", "D1 total"),
                    ("d2_total", "D2 total"),
                ]
                for key, label in curves:
                    values = np.array([h.get(key, np.nan) for h in history], dtype=float)
                    if np.isfinite(values).any():
                        ax2.plot(epochs, values, linewidth=1.8, label=label)
                        plotted_series.append(values)
                        plotted = True

                if plotted:
                    finite_abs = np.concatenate(
                        [np.abs(v[np.isfinite(v)]) for v in plotted_series if np.isfinite(v).any()]
                    )
                    positive = finite_abs[finite_abs > 0.0]
                    if positive.size >= 8:
                        vmax = float(np.max(positive))
                        vmed = float(np.median(positive))
                        if vmax > 200.0 * max(vmed, 1e-12):
                            ax2.set_yscale("symlog", linthresh=max(1e-6, 0.25 * vmed))
                    ax2.set_title("Historico de Treinamento")
                    ax2.set_xlabel("Epoca")
                    ax2.set_ylabel("Loss")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    fig2.tight_layout()
                    fig2.savefig(self.plots_dir / "training_curves.png", dpi=int(self.exp_config.dpi))
                plt.close(fig2)

            if self.logger:
                self.logger.info(
                    "Plots salvos",
                    field_plot=str(self.plots_dir / "field_comparison.png"),
                    training_plot=str(self.plots_dir / "training_curves.png"),
                )

        except Exception as exc:
            if self.logger:
                self.logger.warning("Falha ao gerar plots", error=str(exc))

    def run(
        self,
    ) -> Tuple[
        UNetGenerator2D,
        FieldDualDiscriminator,
        Dict[str, float],
        List[Dict[str, float]],
    ]:
        """
        Executa o pipeline completo: do setup à avaliação final.

        Retorno:
            Uma tupla (gerador, discriminador_dual, métricas_finais, histórico).

        Exceções:
            Exception: Se ocorrer qualquer erro durante a execução.
        """
        start = time.time()
        try:
            hx, hy = self._prepare_physics_fields()
            self.generator, self.discriminator = self._create_models()
            self.trainer = self._create_trainer(hx, hy)
            self.trainer.experiment_config = asdict(self.exp_config)
            hp_path = log_hyperparameters_at_start(
                self.results_dir,
                experiment_config=self.exp_config,
                system_config=self.sys_config,
                trainer_hyperparameters=self.trainer._paper_hyperparameters(),
            )
            if self.logger:
                self.logger.info("Hiperparametros registrados", path=str(hp_path))
            self._baseline_metrics = None

            resume_path = getattr(self.exp_config, "resume_checkpoint", None)
            if resume_path:
                checkpoint_file = Path(str(resume_path))
                if not checkpoint_file.is_absolute():
                    local_candidate = self.runs_dir / checkpoint_file
                    if local_candidate.exists():
                        checkpoint_file = local_candidate
                if not checkpoint_file.exists():
                    raise FileNotFoundError(
                        f"Checkpoint de retomada nao encontrado: {checkpoint_file}"
                    )
                strict_resume = bool(getattr(self.exp_config, "strict_checkpoint_loading", True))
                load_optimizer_state = bool(
                    getattr(self.exp_config, "load_checkpoint_optimizer_state", True)
                )
                restore_rng_state = bool(
                    getattr(self.exp_config, "restore_checkpoint_rng_state", True)
                )
                loaded_epoch, loaded_metrics = self.trainer.load_checkpoint(
                    checkpoint_file,
                    strict=strict_resume,
                    load_optimizer_state=load_optimizer_state,
                    restore_rng_state=restore_rng_state,
                )
                if self.logger:
                    self.logger.info(
                        "Treino retomado de checkpoint",
                        checkpoint=str(checkpoint_file),
                        loaded_epoch=loaded_epoch,
                        resume_from_epoch=self.trainer.start_epoch,
                        strict=strict_resume,
                        loaded_metric_keys=sorted(list(loaded_metrics.keys())),
                    )
            else:
                self._baseline_metrics = self._evaluate()
                if self.logger:
                    self.logger.info(
                        "Baseline pre-treino (modelo nao treinado)",
                        rmse=f"{self._baseline_metrics.get('rmse', 0.0):.4e}",
                        relative_l2=f"{self._baseline_metrics.get('relative_l2_error', 0.0):.4e}",
                        pde_residual_mean=f"{self._baseline_metrics.get('pde_residual_mean', 0.0):.4e}",
                        pde_residual_max=f"{self._baseline_metrics.get('pde_residual_max', 0.0):.4e}",
                    )
                    if float(self._baseline_metrics.get("pde_residual_mean", float("inf"))) < 1e-6:
                        self.logger.warning(
                            "Baseline fisico muito baixo antes do treino; "
                            "considere aumentar a complexidade das condicoes de contorno",
                            boundary_sine_amplitude=float(
                                getattr(self.exp_config, "boundary_sine_amplitude", 0.0)
                            ),
                        )

            self._configure_adv_progressive_from_history()
            self._configure_precision_refine()
            history = self.trainer.train()
            refine_history: List[Dict[str, float]] = []
            if (
                bool(getattr(self.exp_config, "physics_refine_enable", False))
                and int(getattr(self.exp_config, "physics_refine_steps", 0)) > 0
                and int(self.exp_config.epochs)
                >= int(getattr(self.exp_config, "physics_refine_min_train_epochs", 50))
            ):
                refine_history = self.trainer.refine_physics()
                if refine_history:
                    history.extend(refine_history)
                    if self.logger:
                        self.logger.info(
                            "Refinamento fisico final concluido",
                            steps=len(refine_history),
                            residual_mean_abs=f"{refine_history[-1].get('physics_refine_residual_mean_abs', 0.0):.4e}",
                        )
            elif self.logger and bool(getattr(self.exp_config, "physics_refine_enable", False)):
                self.logger.info(
                    "Refinamento fisico final ignorado",
                    epochs=int(self.exp_config.epochs),
                    min_train_epochs=int(
                        getattr(self.exp_config, "physics_refine_min_train_epochs", 50)
                    ),
                    steps=int(getattr(self.exp_config, "physics_refine_steps", 0)),
                )
            metrics = self._evaluate()
            self._save_results(metrics, history)

            if self.logger:
                self.logger.info(
                    "Treinamento PI-GAN concluido",
                    mae=f"{metrics.get('mae', 0.0):.4e}",
                    rmse=f"{metrics.get('rmse', 0.0):.4e}",
                    mape=f"{metrics.get('mape', 0.0):.4e}",
                    r2=f"{metrics.get('r2', 0.0):.6f}",
                    relative_l2=f"{metrics['relative_l2_error']:.4e}",
                    relative_l2_vs_fdm=f"{metrics.get('relative_l2_error_vs_fdm', metrics['relative_l2_error']):.4e}",
                    max_error=f"{metrics['max_error']:.4e}",
                    pde_residual_mean=f"{metrics['pde_residual_mean']:.4e}",
                    pde_residual_max=f"{metrics.get('pde_residual_max', 0.0):.4e}",
                    boundary_error=f"{metrics['boundary_error']:.4e}",
                    elapsed_s=f"{time.time() - start:.1f}",
                )
                tolerance = float(getattr(self.exp_config, "residual_tolerance_target", 1e-3))
                if float(metrics.get("pde_residual_mean", float("inf"))) > tolerance:
                    self.logger.warning(
                        "Tolerancia fisica alvo nao atingida na avaliacao final",
                        residual_mean=f"{metrics.get('pde_residual_mean', 0.0):.4e}",
                        tolerance_target=f"{tolerance:.4e}",
                    )

            return self.generator, self.discriminator, metrics, history
        finally:
            if self.device.type == "cuda":
                self.memory_manager.clear_cache()


__all__ = ["PIGANPipeline"]

