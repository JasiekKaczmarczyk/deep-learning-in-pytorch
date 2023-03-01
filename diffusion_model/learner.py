import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from PIL import Image

from diffusion_model import DiffusionModel
from forward_diffusion import ForwardDiffusionModule
from reverse_diffusion import UNet
from utils import Transforms

class Learner(pl.LightningModule):
    def __init__(self, diffusion_model: DiffusionModel, lr: float, model_filepath: str, sampled_filepath: str):
        super().__init__()

        self.save_hyperparameters()

        self.diffusion_model = diffusion_model
        self.num_steps = self.diffusion_model.forward_diffusion_model.num_steps

        self.transforms = Transforms()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.reverse_diffusion_model(x, t)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.diffusion_model.reverse_diffusion_model.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.shape[0]
        t = torch.randint(1, self.num_steps, size=(batch_size,), device=self.device, dtype=torch.long)

        x_noised, noise = self.diffusion_model.forward_diffusion(x, t)
        predicted_noise = self.diffusion_model.reverse_diffusion(x_noised, t)

        loss = F.mse_loss(noise, predicted_noise)

        self.log_dict(
            {
                "loss": loss
            }
        )

        return loss

    def training_epoch_end(self, outputs):
        # save sampled images
        noise = torch.randn((4, 3, 64, 64), device=self.device)

        sampled_images = self.diffusion_model.sample(noise)
        grid = torchvision.utils.make_grid(sampled_images)

        # transformed_grid = self.transforms.t2i(grid)
        torchvision.utils.save_image(grid, fp=f"{self.sampled_filepath}/sampled_{self.current_epoch}.jpg")

        # save model
        self.diffusion_model.save(self.save_filepath)
    

