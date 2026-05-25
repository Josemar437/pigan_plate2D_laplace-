"""Laplaciano discreto reutilizavel para a placa Laplace 2D."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class PDEResidualComputer:
    """Mesmo stencil 5-pontos de `LaplacianLayer`, preparado para integracao futura."""

    def __init__(
        self, 
        grid_size_x: int, 
        grid_size_y: int,
        use_gpu: bool = True,
        kernel_type: str = "centered",
        hx: float = 1.0,
        hy: Optional[float] = None,
    ) -> None:
        """Configura kernel `+nabla^2` com espacamentos fisicos `hx/hy`."""
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.kernel_type = kernel_type
        self.hx = float(hx)
        self.hy = float(hy if hy is not None else hx)
        if self.hx <= 0.0 or self.hy <= 0.0:
            raise ValueError("hx e hy devem ser positivos.")
        
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        
        # Construir kernel Laplaciano 5-pontos normalizado
        self.laplacian_kernel = self._build_laplacian_kernel()
        
    def _build_laplacian_kernel(self) -> torch.Tensor:
        """Cria kernel `[1,1,3,3]`; apenas `centered` e suportado."""
        if self.kernel_type == "centered":
            # Stencil 5-pontos centrado de segunda ordem para +∇²T.
            kernel = torch.tensor(
                [
                    [0.0, 1.0 / (self.hy * self.hy), 0.0],
                    [
                        1.0 / (self.hx * self.hx),
                        -2.0 * (1.0 / (self.hx * self.hx) + 1.0 / (self.hy * self.hy)),
                        1.0 / (self.hx * self.hx),
                    ],
                    [0.0, 1.0 / (self.hy * self.hy), 0.0],
                ],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("kernel_type inválido. Use 'centered'.")
        
        return kernel
    
    def compute_laplacian(
        self, 
        field: torch.Tensor,
        use_reflect_padding: bool = False,
    ) -> torch.Tensor:
        """Retorna `[B,1,H,W]`; por padrao calcula interior e zera a borda."""
        # Garantir shape (B, 1, H, W)
        if field.ndim == 3:
            field = field.unsqueeze(1)
        
        if use_reflect_padding:
            padded = F.pad(field, (1, 1, 1, 1), mode='reflect')
            laplacian = F.conv2d(padded, self.laplacian_kernel, padding=0)
        else:
            interior = F.conv2d(field, self.laplacian_kernel, padding=0)
            laplacian = torch.zeros_like(field)
            laplacian[:, :, 1:-1, 1:-1] = interior
        
        return laplacian
    
    def compute_pde_residual(
        self, 
        prediction: torch.Tensor,
        use_abs: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calcula residuo assinado ou absoluto mantendo shape `[B,1,H,W]`."""
        laplacian = self.compute_laplacian(prediction)
        
        if use_abs:
            residual = torch.abs(laplacian)
        else:
            residual = laplacian
        
        abs_residual = torch.abs(laplacian)
        stats = {
            "residual_mean_abs": abs_residual.mean(dim=(2, 3), keepdim=True),
            "residual_max_abs": abs_residual.amax(dim=(2, 3), keepdim=True),
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
        """Media o residuo absoluto nas quatro bandas de canto."""
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
        """Combina residuo medio global com reforco opcional nos cantos."""
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


# Compatibilidade com imports existentes durante a migração de nomes.
PDE_Residual_Computer = PDEResidualComputer
