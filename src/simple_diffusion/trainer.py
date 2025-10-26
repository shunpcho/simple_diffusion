from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from simple_diffusion.dataset import get_data_loaders
from simple_diffusion.model import DiffusionModel


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader[tuple[torch.Tensor, int]],
        val_dataloader: DataLoader[tuple[torch.Tensor, int]],
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        optimizer: torch.optim.Optimizer | None = None,
        output_dir: Path = Path("./results"),
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_timesteps = num_timesteps
        self.optimizer = optimizer

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = self.model.to(self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to the input image x0 at time step t."""
        batch_size = x0.size(0)
        noise = torch.randn_like(x0).to(self.device)

        # Get the corresponding alpha_bar for time step t
        alpha_bar_t = self.alpha_bars[t].reshape(batch_size, 1, 1, 1).to(self.device)

        noisy_x = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise

    def train_step(self) -> float:
        total_loss = 0.0
        self.model.train()
        for batch in self.train_dataloader:
            train_images, _ = batch
            train_images = train_images.to(self.device)

            # Sample random time steps for each image in the batch
            t = torch.randint(0, self.num_timesteps, (train_images.size(0),)).to(self.device)
            noisy_x, noise = self.add_noise(train_images, t)

            t_input = t.float().unsqueeze(1) / self.num_timesteps
            noise_pred = self.model(noisy_x, t_input)

            loss = F.mse_loss(noise_pred, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    @torch.no_grad()
    def sample(self, num_samples: int = 8) -> torch.Tensor:
        """Generate samples from the diffusion model."""
        self.model.eval()

        # Start from pure noise
        x = torch.randn(num_samples, self.model.channels, self.model.image_size, self.model.image_size).to(self.device)

        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            t_input = t_batch.float().unsqueeze(1) / self.num_timesteps

            noise_pred = self.model(x, t_input)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred)

            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(self.betas[t]) * noise

            # Save intermediate steps periodically
            if t % 100 == 0:
                save_image(x, self.output_dir / f"sample_t{t}.png", normalize=True)

        save_image(generated_images, self.output_dir / "final_samples.png", normalize=True, nrow=2)
        self.model.train()
        return x


if __name__ == "__main__":
    # Initialize a simple model
    model = DiffusionModel(image_size=32, channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Reduce batch size to save memory
    train_loader, val_loader = get_data_loaders(batch_size=4)
    trainer = DiffusionTrainer(model, train_loader, val_loader, num_timesteps=1000, optimizer=optimizer)

    print("DiffusionTrainer initialized successfully.")
    print(f"Using device: {trainer.device}")

    print("Starting a training step...")
    for epoch in range(3):
        loss = trainer.train_step()
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        # Clear cache to free up memory
        torch.cuda.empty_cache()

    print("Generating images...")
    generated_images = trainer.sample(num_samples=4)
    print("Generated images shape:", generated_images.shape)
