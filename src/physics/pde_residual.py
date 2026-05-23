"""Computação de resíduos PDE para equação de Laplace 2D."""
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class PDE_Residual_Computer:
    """
    Computador eficiente de resíduos PDE usando operador Laplaciano 5-pontos.
    
    Implementa ∇²T = 0 usando stencil de diferenças finitas centrais com
    coeficientes normalizados por h².
    
    Atributos:
        grid_size_x (int): Dimensão x da malha
        grid_size_y (int): Dimensão y da malha
        laplacian_kernel (torch.Tensor): Kernel 5-pontos normalizado
        device (torch.device): Dispositivo CUDA ou CPU
    """

    def __init__(
        self, 
        grid_size_x: int, 
        grid_size_y: int,
        use_gpu: bool = True,
        kernel_type: str = "centered",
    ) -> None:
        """
        Inicializa o computador de resíduos PDE.
        
        Parâmetros:
            grid_size_x: Dimensão x da malha
            grid_size_y: Dimensão y da malha  
            use_gpu: Se True, usa CUDA quando disponível
            kernel_type: Tipo de kernel ("centered" ou "forward")
        """
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.kernel_type = kernel_type
        
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        
        # Construir kernel Laplaciano 5-pontos normalizado
        # Stencil: [-1, -4, 20, -4, -1] / 6h² (O(2) accurado)
        # Para h=1 (malha normalizada): [-1, -4, 20, -4, -1] / 6
        self.laplacian_kernel = self._build_laplacian_kernel()
        
    def _build_laplacian_kernel(self) -> torch.Tensor:
        """
        Constrói kernel 5-pontos para o Laplaciano.
        
        Retorno:
            torch.Tensor: Kernel de shape (1, 1, 3, 3) normalizado
        """
        if self.kernel_type == "centered":
            # Stencil 5-pontos centrado (O(2) acurado)
            kernel = torch.tensor(
                [[0.0, -1.0, 0.0],
                 [-1.0, 4.0, -1.0],
                 [0.0, -1.0, 0.0]],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"kernel_type '{self.kernel_type}' não suportado")
        
        return kernel
    
    def compute_laplacian(
        self, 
        field: torch.Tensor,
        use_reflect_padding: bool = True,
    ) -> torch.Tensor:
        """
        Computa Laplaciano de um campo usando convolução.
        
        Parâmetros:
            field: Tensor de shape (B, 1, H, W) ou (B, H, W)
            use_reflect_padding: Se True, usa reflect padding para determinismo
        
        Retorno:
            torch.Tensor: Laplaciano de shape (B, 1, H, W)
        """
        # Garantir shape (B, 1, H, W)
        if field.ndim == 3:
            field = field.unsqueeze(1)
        
        # Padding: reflect para determinismo em CUDA
        if use_reflect_padding:
            # Usar reflect padding quando em modo determinístico
            padded = F.pad(field, (1, 1, 1, 1), mode='reflect')
        else:
            # Fallback: replicate padding
            padded = F.pad(field, (1, 1, 1, 1), mode='replicate')
        
        # Convolução com kernel normalizado
        laplacian = F.conv2d(padded, self.laplacian_kernel, padding=0)
        
        return laplacian
    
    def compute_pde_residual(
        self, 
        prediction: torch.Tensor,
        use_abs: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa resíduo PDE: |∇²T| onde T = prediction.
        
        Parâmetros:
            prediction: Campo predito de shape (B, 1, H, W)
            use_abs: Se True, retorna valor absoluto do resíduo
        
        Retorno:
            Tupla:
                - residual: Resíduo de shape (B, 1, H-2, W-2)
                - stats: Dicionário com estatísticas do resíduo
        """
        laplacian = self.compute_laplacian(prediction)
        
        if use_abs:
            residual = torch.abs(laplacian)
        else:
            residual = laplacian
        
        # Computar estatísticas
        stats = {
            "residual_mean_abs": residual.mean(dim=(2, 3), keepdim=True),  # (B, 1, 1, 1)
            "residual_max_abs": residual.amax(dim=(2, 3), keepdim=True),   # (B, 1, 1, 1)
            "residual_std": residual.std(dim=(2, 3), keepdim=True),        # (B, 1, 1, 1)
            "laplacian": laplacian,
        }
        
        return residual, stats
    
    def compute_corner_residual(
        self,
        prediction: torch.Tensor,
        corner_band_points: int = 2,
        return_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computa resíduo PDE em região de cantos/bordas (para reforço).
        
        Parâmetros:
            prediction: Campo predito (B, 1, H, W)
            corner_band_points: Espessura da banda de cantos
            return_mask: Se True, retorna máscara de cantos
        
        Retorno:
            Tupla:
                - residual_corners: Resíduo apenas em cantos
                - mask (opcional): Máscara binária de cantos
        """
        residual, _ = self.compute_pde_residual(prediction, use_abs=True)
        
        B, C, H, W = residual.shape
        
        # Criar máscara de cantos
        mask = torch.zeros_like(residual, dtype=torch.bool)
        
        # Top-left corner
        mask[:, :, :corner_band_points, :corner_band_points] = True
        # Top-right corner
        mask[:, :, :corner_band_points, -corner_band_points:] = True
        # Bottom-left corner
        mask[:, :, -corner_band_points:, :corner_band_points] = True
        # Bottom-right corner
        mask[:, :, -corner_band_points:, -corner_band_points:] = True
        
        residual_corners = residual[mask].reshape(B, C, -1).mean(dim=2, keepdim=True)
        
        if return_mask:
            return residual_corners, mask
        return residual_corners, None
    
    def compute_weighted_pde_loss(
        self,
        prediction: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        corner_weight: float = 1.0,
        corner_band_points: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computa perda PDE com peso adicional em cantos.
        
        Parâmetros:
            prediction: Campo predito
            reference: Campo de referência (não usado neste método base)
            corner_weight: Peso multiplicativo para resíduo em cantos
            corner_band_points: Espessura da banda
        
        Retorno:
            Tupla:
                - loss: Perda ponderada
                - stats: Estatísticas intermediárias
        """
        # Resíduo geral
        residual, stats = self.compute_pde_residual(prediction, use_abs=True)
        interior_loss = residual.mean()
        
        # Resíduo em cantos (reforçado)
        corner_residual, _ = self.compute_corner_residual(
            prediction, 
            corner_band_points=corner_band_points
        )
        corner_loss = corner_residual.mean()
        
        # Perda combinada
        total_loss = interior_loss + corner_weight * corner_loss
        
        stats.update({
            "interior_loss": interior_loss,
            "corner_loss": corner_loss,
            "total_pde_loss": total_loss,
        })
        
        return total_loss, stats
