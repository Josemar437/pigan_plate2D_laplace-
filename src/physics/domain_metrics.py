"""Métricas de qualidade e validação em domínio físico."""
from typing import Dict, Optional, Tuple

import torch


class Domain_Metrics_Computer:
    """
    Computador de métricas em domínio físico para campos 2D.
    
    Valida que predições satisfazem:
    - Condições de contorno de Dirichlet
    - Suavidade do campo
    - Consistência com referência FDM
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Inicializa computador de métricas de domínio.
        
        Parâmetros:
            device: Dispositivo CUDA ou CPU
        """
        self.device = device or torch.device("cpu")
    
    def compute_boundary_error(
        self,
        prediction: torch.Tensor,
        boundary_values: torch.Tensor,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa erro em condições de contorno de Dirichlet.
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            boundary_values: Valores esperados na fronteira (B, 1, H, W)
            boundary_mask: Máscara booleana de pixels de fronteira (B, 1, H, W)
        
        Retorno:
            Tupla:
                - error: Erro médio absoluto na fronteira
                - stats: Estatísticas do erro
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
        """
        Computa métricas de suavidade (regularidade) do campo.
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            order: Ordem da derivada (1 ou 2)
        
        Retorno:
            Tupla:
                - smoothness: Medida de suavidade (menor = mais suave)
                - stats: Estatísticas de derivadas
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
            # Laplaciano (segunda derivada)
            dx2 = torch.abs(prediction[:, :, :, 2:] - 2 * prediction[:, :, :, 1:-1] + prediction[:, :, :, :-2])
            dy2 = torch.abs(prediction[:, :, 2:, :] - 2 * prediction[:, :, 1:-1, :] + prediction[:, :, :-2, :])
            laplacian = torch.sqrt(dx2[:, :, :, :-1] ** 2 + dy2[:, :, :-1, :] ** 2 + 1e-8)
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
        """
        Computa intervalo e distribuição de valores do campo.
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
        
        Retorno:
            Dicionário com estatísticas do intervalo
        """
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
        """
        Computa erro relativo à solução de referência (FDM).
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            reference: Campo de referência FDM (B, 1, H, W)
            mask_interior: Máscara opcional de pontos interiores
        
        Retorno:
            Tupla:
                - error: Erro normalizxo
                - stats: Estatísticas de erro
        """
        diff = prediction - reference
        
        if mask_interior is not None:
            diff_interior = diff[mask_interior]
        else:
            # Usar interior padrão (remover borda)
            diff_interior = diff[:, :, 1:-1, 1:-1]
        
        mae = torch.abs(diff_interior).mean()
        mse = (diff_interior ** 2).mean()
        rmse = torch.sqrt(mse)
        
        # R² score
        ss_res = (diff_interior ** 2).sum()
        ss_tot = ((reference[:, :, 1:-1, 1:-1] - reference[:, :, 1:-1, 1:-1].mean()) ** 2).sum()
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
        """
        Computa todas as métricas de domínio de uma vez.
        
        Parâmetros:
            prediction: Campo predito
            reference: Campo de referência (opcional)
            boundary_values: Valores de fronteira esperados (opcional)
        
        Retorno:
            Dicionário unificado com todas as métricas
        """
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
