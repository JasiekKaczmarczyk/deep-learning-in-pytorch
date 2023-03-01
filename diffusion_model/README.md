# Diffusion Model
Implementation of Diffusion Model

## Usage
```python
from diffusion_model import DiffusionModel
from forward_diffusion import ForwardDiffusionModule
from reverse_diffusion import UNet


forward_diffusion = ForwardDiffusionModule(num_steps = 256, beta_start = 0.0001, beta_end = 0.2, schedule_type = "cosine")
reverse_diffusion = UNet(c_in = 3, c_out = 3, time_dim = 256)
diffusion_model = DiffusionModel(forward_diffusion_model = forward_diffusion, reverse_diffusion_model = reverse_diffusion)

# shapes: [batch_size, channels, height, width]
z = torch.randn((1, 3, 256, 256))
sampled_img = diffusion_model.sample(z)
```
## Citations
```bibtex
@article{ho2020,
    title  = {Denoising Diffusion Probabilistic Models},
    author = {Jonathan Ho, Ajay Jain and Pieter Abbeel},
    year   = {2020},
    url = {https://arxiv.org/abs/2006.11239}
}
```

```bibtex
@article{nichol2021,
    title  = {Improved Denoising Diffusion Probabilistic Models},
    author = {Alex Nichol and Prafulla Dhariwal},
    year   = {2021},
    url = {https://arxiv.org/abs/2102.09672}
}
```