import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from hyena import HyenaOperator

class FeedForward(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)
    
class Block(nn.Module):
    def __init__(self, embedding_size: int, order: int, ffn_hidden_size: int):
        super().__init__()

        self.hyena_operator = nn.Sequential(
            nn.LayerNorm(embedding_size),
            HyenaOperator(embedding_size, order, causal=False)
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_size),
            FeedForward(embedding_size, ffn_hidden_size)
        )

    def forward(self, x: torch.Tensor):
        x = self.hyena_operator(x) + x
        x = self.ffn(x) + x

        return x
    
class HyenaViT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass