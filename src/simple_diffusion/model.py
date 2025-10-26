import torch
import torch.nn.functional as F
from torch import nn


class DiffusionModel(nn.Module):
    def __init__(self, image_size: int = 28, channels: int = 1, time_dim: int = 128) -> None:
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # U-Net architecture
        self.conv1 = nn.Conv2d(channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, channels, 3, padding=1)

        self.time_proj1 = nn.Linear(time_dim, 64)
        self.time_proj2 = nn.Linear(time_dim, 128)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the diffusion model.

        Args:
            x (torch.Tensor): Noisy input image tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step tensor of shape (B, 1).

        Returns:
            torch.Tensor: Denoised image tensor of shape (B, C, H, W).
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # U-Net forward pass
        h1 = F.silu(self.conv1(x))  # [B, 64, H, W]
        h1 += self.time_proj1(t_emb)[:, :, None, None]

        h2 = F.silu(self.conv2(h1))  # [B, 128, H, W]
        h2 += self.time_proj2(t_emb)[:, :, None, None]

        h3 = F.silu(self.conv3(h2))
        h4 = F.silu(self.conv4(h3))

        noise_pred = self.conv_out(h4)
        return noise_pred
