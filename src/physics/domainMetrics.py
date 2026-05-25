"""Metricas de campo para predicoes Laplace 2D ja materializadas."""
from typing import Dict, Optional, Tuple

import torch


class DomainMetricsComputer:
    """Metricas auxiliares; o pipeline ativo ainda usa `src/evaluation.py`.
    
    Computes field-level metrics (boundary error, smoothness, range, reference error)
    for monitoring and validation of PI-GAN predictions.
    
    Attributes:
        device: Torch device for tensor operations (cuda or cpu).
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Guarda o device usado para tensores escalares auxiliares.
        
        Args:
            device: Torch device (defaults to CPU).
        """
        self.device = device or torch.device("cpu")
    
    def compute_boundary_error(
        self,
        prediction: torch.Tensor,
        boundary_values: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """MAE de Dirichlet sobre a borda padrao ou `boundary_mask` informado.
        
        Args:
            prediction: Predicted field [B, C, H, W].
            boundary_values: Target boundary values [B, C, H, W].
            boundary_mask: Optional mask for boundary region. Defaults to standard borders.
            
        Returns:
            Tuple of (error, stats_dict).
        """
        if boundary_mask is None:
            # Usar fronteira padrão (borda)
            B, C, H, W = prediction.shape
            boundary_mask = torch.zeros_like(prediction, dtype=torch.bool)
            boundary_mask[:, :, 0, :] = True    # Top
            boundary_mask[:, :, -1, :] = True   # Bottom
            boundary_mask[:, :, :, 0] = True    # Left
            boundary_mask[:, :, :, -1] = True   # Right
        
        # Diferença apenas na fronteira
        diff = torch.abs(prediction - boundary_values)
        diff_boundary = diff[boundary_mask]
        
        error = diff_boundary.mean()
        
        stats = {
            "boundary_mae": error,
            "boundary_max": diff_boundary.max(),
            "boundary_std": diff_boundary.std(),
            "boundary_n_points": torch.tensor(diff_boundary.numel(), device=self.device),
        }
        
        return error, stats
    
    def compute_smoothness_metrics(
        self,
        prediction: torch.Tensor,
        order: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mede gradiente (`order=1`) ou segunda derivada em grade comum (`order=2`).
        
        Args:
            prediction: Field tensor [B, C, H, W].
            order: Derivative order (1 or 2). Default is 1.
            
        Returns:
            Tuple of (smoothness_metric, stats_dict).
            
        Raises:
            ValueError: If order is not 1 or 2.
        """
        B, C, H, W = prediction.shape
        
        if order == 1:
            # Gradiente de primeira ordem (não tomar abs antes de combinar para manter sinais)
            dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]  # shape (B, C, H, W-1)
            dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]  # shape (B, C, H-1, W)

            # Para combinar dx e dy em uma grade comum, cortar a última linha/coluna
            # resultando em shape (B, C, H-1, W-1)
            gx = dx[:, :, :-1, :]   # remove última linha -> (B, C, H-1, W-1)
            gy = dy[:, :, :, :-1]   # remove última coluna -> (B, C, H-1, W-1)

            grad_magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
            smoothness = grad_magnitude.mean()

            stats = {
                "grad_x_mean": dx.abs().mean(),
                "grad_y_mean": dy.abs().mean(),
                "grad_magnitude_mean": grad_magnitude.mean(),
                "grad_magnitude_max": grad_magnitude.max(),
            }
        
        elif order == 2:
            # Segunda derivada em uma grade comum: pontos interiores [1:-1, 1:-1].
            dx2 = torch.abs(prediction[:, :, :, 2:] - 2 * prediction[:, :, :, 1:-1] + prediction[:, :, :, :-2])
            dy2 = torch.abs(prediction[:, :, 2:, :] - 2 * prediction[:, :, 1:-1, :] + prediction[:, :, :-2, :])
            d2x_common = dx2[:, :, 1:-1, :]
            d2y_common = dy2[:, :, :, 1:-1]
            laplacian = torch.sqrt(d2x_common ** 2 + d2y_common ** 2 + 1e-8)
            smoothness = laplacian.mean()
            
            stats = {
                "d2x_mean": dx2.mean(),
                "d2y_mean": dy2.mean(),
                "laplacian_mean": laplacian.mean(),
                "laplacian_max": laplacian.max(),
            }
        
        else:
            raise ValueError(f"order={order} não suportado")
        
        return smoothness, stats
    
    def compute_field_range(
        self,
        prediction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Resumo escalar de faixa e quartis do campo predito."""
        stats = {
            "min": prediction.min(),
            "max": prediction.max(),
            "mean": prediction.mean(),
            "std": prediction.std(),
            "median": torch.median(prediction),
            "q25": torch.quantile(prediction, 0.25),
            "q75": torch.quantile(prediction, 0.75),
        }
        
        return stats
    
    def compute_reference_error(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        mask_interior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compara predicao e `T_ref` na mesma regiao usada para MAE/MSE/R2."""
        diff = prediction - reference
        
        if mask_interior is not None:
            if mask_interior.shape != diff.shape:
                raise ValueError("mask_interior deve ter o mesmo shape de prediction/reference.")
            diff_interior = diff[mask_interior]
            reference_interior = reference[mask_interior]
        else:
            # Usar interior padrão (remover borda)
            diff_interior = diff[:, :, 1:-1, 1:-1]
            reference_interior = reference[:, :, 1:-1, 1:-1]
        
        mae = torch.abs(diff_interior).mean()
        mse = (diff_interior ** 2).mean()
        rmse = torch.sqrt(mse)
        
        # R² score
        ss_res = (diff_interior ** 2).sum()
        ss_tot = ((reference_interior - reference_interior.mean()) ** 2).sum()
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        
        stats = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "max_abs_error": torch.abs(diff_interior).max(),
        }
        
        return mae, stats
    
    def compute_all_metrics(
        self,
        prediction: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        boundary_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Agrega faixa, suavidade, contorno e erro vs FDM quando disponiveis."""
        all_metrics = {}
        
        # Métricas de intervalo
        all_metrics.update(self.compute_field_range(prediction))
        
        # Métricas de suavidade
        _, smooth_stats = self.compute_smoothness_metrics(prediction, order=1)
        all_metrics.update(smooth_stats)
        
        # Métricas de contorno
        if boundary_values is not None:
            _, bc_stats = self.compute_boundary_error(prediction, boundary_values)
            all_metrics.update(bc_stats)
        
        # Métricas vs referência
        if reference is not None:
            _, ref_stats = self.compute_reference_error(prediction, reference)
            all_metrics.update(ref_stats)
        
        return all_metrics


# Compatibilidade com imports existentes durante a migração de nomes.
Domain_Metrics_Computer = DomainMetricsComputer
