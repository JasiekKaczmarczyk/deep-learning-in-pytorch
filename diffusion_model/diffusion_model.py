import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm


class DiffusionModel:
    def __init__(self, forward_diffusion_model: nn.Module, reverse_diffusion_model: nn.Module):
        super().__init__()

        self.forward_diffusion_model = forward_diffusion_model
        self.reverse_diffusion_model = reverse_diffusion_model
        self.num_steps = forward_diffusion_model.num_steps

    def save(self, filepath="diffusion_model.pt"):
        torch.save(
            {
                self.forward_diffusion_model,
                self.reverse_diffusion_model
            },
            filepath
        )

    @property
    def load(self, filepath="diffusion_model.pt"):
        loaded = torch.load(filepath)

        return DiffusionModel(loaded["forward_diffusion_model"], loaded["reverse_diffusion_model"])

    def forward_diffusion(self, x: torch.Tensor, t: torch.Tensor):
        x_noised, noise = self.forward_diffusion_model(x, t)

        return x_noised, noise

    def reverse_diffusion(self, x: torch.Tensor, t: torch.Tensor):
        predicted_noise = self.reverse_diffusion_model(x, t)

        return predicted_noise

    @torch.no_grad()
    def sample(self, x: torch.Tensor):
        # set model to eval mode
        self.reverse_diffusion_model.eval()

        # reversing diffusion process
        for i in tqdm(reversed(1, self.num_steps)):
            # generating timestep tensor of size (batch_size, )
            t = (torch.ones(x.shape[0], device=x.device) * i).long()

            # predict noise
            predicted_noise = self.reverse_diffusion_model(x, t)

            # get params diffusion params for timestep
            alpha = self.forward_diffusion_model.alphas[t][:, None, None, None]
            alpha_hat = self.forward_diffusion_model.alphas_hat[t][:, None, None, None]
            beta = self.forward_diffusion_model.betas[t][:, None, None, None]

            if i > 1:
                noise = torch.randn_like(x, device=x.device)
            else:
                noise = torch.zeros_like(x, device=x.device)

            # subtracting noise from image
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        # set model to training mode
        self.reverse_diffusion_model.train()

        return x
