import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# https://github.com/fkodom/fft-conv-pytorch
from fft_conv_pytorch import fft_conv


class HyenaProjection(nn.Module):
    def __init__(self, embeding_size: int, hyena_order: int):
        """
        Hyena Projection layer: processess input data to value and `hyena_order` number of gates

        Args:
            embeding_size (int): size of embedding dim
            hyena_order (int): number of projections for hyena operator
        """
        super().__init__()

        self.hyena_order = hyena_order
        out_size = (hyena_order+1)*embeding_size

        self.linear = nn.Linear(embeding_size, out_size)
        self.dephtwise_conv = nn.Conv1d(out_size, out_size, kernel_size=3, padding=1, groups=out_size)

    def forward(self, x: torch.Tensor):
        # projection across seq_len: [batch_size, seq_len, embedding_size] -> [batch_size, seq_len, (order+1)*embedding_size]
        z_hat = self.linear(x)

        # rearranging [batch_size, seq_len, embedding] -> [batch_size, embedding, seq_len]
        z_hat = einops.rearrange(z_hat, "b l e -> b e l")

        # depthwise conv across embedding: [batch_size, embedding, seq_len] -> [batch_size, embedding, seq_len]
        z = self.dephtwise_conv(z_hat)

        # rearranging output: [batch_size, (order+1)*embedding_size, seq_len] -> [(order+1), batch_size, embedding_size, seq_len]
        out = einops.rearrange(z, "b (n e) l -> n b e l", n=(self.hyena_order+1))

        return out
    
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

class WindowFunc(nn.Module):
    def __init__(self, scaling_factor: float, bias: float):
        """
        Window function applied to conv filters

        Args:
            scaling_factor (float): slope of window
            bias (float): additional bias term
        """
        super().__init__()

        self.scaling_factor = scaling_factor
        self.bias = bias

    def forward(self, x: torch.Tensor):
        _, seq_len, _ = x.shape

        position = torch.arange(seq_len, device=x.device).float()
        return torch.exp(-self.scaling_factor * position + self.bias)

class HyenaFilter(nn.Module):
    def __init__(self, embedding_size: int, order: int, window_scaling_factor: float, window_bias: float):
        """
        Hyena Filter layer: generates implicit convolution filters for hyena operator

        Args:
            embedding_size (int): size of embedding dim
            order (int): number of hyena filters
            window_scaling_factor (float): scaling factor for window function
            window_bias (float): bias term for window function
        """
        super().__init__()

        self.order = order

        self.positional_encoding = PositionalEncoding1D(embedding_size)
        self.filter_proj = nn.Linear(embedding_size, order*embedding_size)
        self.window = WindowFunc(window_scaling_factor, window_bias)

    def forward(self, x: torch.Tensor):
        # generating positional encodings
        pe = self.positional_encoding(x)

        # generating filters
        # shape: [seq_len, order * embedding]
        h_hat = self.filter_proj(pe)
        # out shape: [order, embedding, seq_len]
        h_hat = einops.rearrange(h_hat, "l (o e) -> o e l", o=self.order)

        # applying window function
        h = h_hat * self.window(x)

        return h

class HyenaOperator(nn.Module):
    def __init__(self, embedding_size: int, order: int, window_scaling_factor: float, window_bias: float, causal: bool = False):
        """
        Hyena Operator: applies Hyena Operator to input sequence

        Args:
            embedding_size (int): size of embedding dim
            order (int): number of hyena filters applied
            window_scaling_factor (float): scaling factor for window function
            window_bias (float): bias term for window function
            causal (bool, optional): is model autoregressive or not, if True applies masking to filters before convolution. Defaults to False.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.order = order

        self.window_scaling_factor = window_scaling_factor
        self.window_bias = window_bias
        self.causal = causal

        # projection of data to gates and value
        self.data_proj = HyenaProjection(embedding_size, order)
        self.filter_proj = HyenaFilter(embedding_size, order, window_scaling_factor, window_bias)

    def forward(self, x: torch.Tensor):
        _, seq_len, _ = x.shape

        # projection of input of shape: [order+1, batch_size, embedding, seq_len]
        # dim=0 corresponds to gates: projection[1], projection[2], ..., projection[order-1] and value: projection[-1]
        projection = self.data_proj(x)
        # filters shape: [order, embedding, seq_len]
        filters = self.filter_proj(x)

        # if causal future values are masked with 0 
        if self.causal:
            filters[:, :, (seq_len//2)+1:] = 0.0

        # grabbing value
        value = projection[-1]

        for i in range(self.order):
            gate = projection[i]
            kernel = filters[i]
            
            value = gate * fft_conv(value, kernel)

        # rearranging the output
        out = einops.rearrange(value, "b e l -> b l e")

        return out





