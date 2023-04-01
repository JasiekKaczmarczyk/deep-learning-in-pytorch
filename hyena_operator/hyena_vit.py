import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from hyena import HyenaOperator

class PositionalEncoding1D(nn.Module):
    def __init__(self, embedding_size: int):
        """
        Positional Encoding layer: creates sinusoidal positional encoding for input sequence `x`

        Args:
            embedding_size (int): size of embedding dim
        """
        super().__init__()

        self.embedding_size = embedding_size

        inv_freq = 1.0 / (10000 ** (torch.arange(0, embedding_size, 2).float() / embedding_size))
        self.register_buffer("inv_freq", inv_freq)

        # cache for storing encoding if already calculated
        self.cache = None

    def forward(self, x: torch.Tensor):
        if x.ndim != 3:
            raise RuntimeError("Input should have dims (batch_size, seq_len, embedding_size)")

        if self.cache is not None and self.cache.ndim == x.ndim:
            return self.cache

        _, seq_len, _ = x.shape
        position = torch.arange(seq_len, device=x.device).float()
        pos_emb = torch.einsum("i,j->ij", position, self.inv_freq)
        
        pe = torch.zeros(seq_len, self.embedding_size, device=x.device)
        pe[:, 0::2] = torch.sin(pos_emb)
        pe[:, 1::2] = torch.cos(pos_emb)
        self.cache = pe

        return pe
    
class Img2Patch(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()

        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        # patch image
        return einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)

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
    
class HyenaViTBlock(nn.Module):
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
    def __init__(self, embedding_size: int, in_channels: int, patch_size: int, num_classes: int, num_hyena_blocks: int, hyena_order: int, ffn_hidden_size: int):
        super().__init__()

        # initial processing blocks
        self.img2patch = Img2Patch(patch_size)
        self.patch2emb = nn.Linear(in_channels*(patch_size**2), embedding_size)
        self.pe = PositionalEncoding1D(embedding_size)

        self.hyena_transformer = nn.Sequential(
            *[HyenaViTBlock(embedding_size, hyena_order, ffn_hidden_size) for _ in range(num_hyena_blocks)]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, num_classes)
        )

    def forward(self, x: torch.Tensor):
        # shapes: [batch_size, channels, height, width] -> [batch_size, num_patches, channels*(patch_size**2)]
        x = self.img2patch(x)
        # shapes: [batch_size, num_patches, channels*(patch_size**2)] -> [batch_size, num_patches, embedding_size]
        x = self.patch2emb(x)

        # adding positional encoding
        x = x + self.pe(x)

        x = self.hyena_transformer(x)

        return self.mlp_head(x)
    

if __name__=="__main__":
    x = torch.randn((1, 3, 256, 256))

    model = HyenaViT(32, 3, 32, 4, 2, 2, 64)

    print(model(x).shape)

