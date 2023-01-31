import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class ForwardDiffusionModule(nn.Module):
    def __init__(self, num_steps: int, beta_start: float, beta_end: float, schedule_type: str = "linear"):
        super().__init__()

        betas = self.beta_schedule(beta_start, beta_end, num_steps, schedule_type)
        alphas = 1 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_hat", alphas_hat)

    def beta_schedule(self, beta_start, beta_end, num_steps, schedule_type):
        def linear_schedule():
            return torch.linspace(beta_start, beta_end, num_steps)

        def quadratic_schedule():
            return torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2

        def sigmoid_schedule():
            betas = torch.linspace(-6, 6, num_steps)
            return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

        def cosine_schedule():
            t = torch.linspace(0, num_steps, num_steps+1)
            x = torch.cos(((t/num_steps) + 0.008) / (1+0.008) * torch.pi * 0.5) ** 2
            alphas_hat = x / x[0]
            betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])
            return torch.clip(betas, 0.0001, 0.9999)

        if schedule_type == "linear":
            schedule = linear_schedule
        elif schedule_type == "quadratic":
            schedule = quadratic_schedule
        elif schedule_type == "sigmoid":
            schedule = sigmoid_schedule
        elif schedule_type == "cosine":
            schedule = cosine_schedule
        else:
            raise ValueError(schedule_type)
        
        return schedule()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x, device=x.device)

        # get alphas_hat for each batch
        alphas_hat_t = self.alphas_hat[t][:, None, None, None]

        mu = torch.sqrt(alphas_hat_t) * x
        sigma = torch.sqrt(1 - alphas_hat_t) * noise

        return mu + sigma, noise