import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, heads: int):
        """
        Multi Head Attention Layer

        Args:
            embedding_size (int): size of embedding dim
            heads (int): number of attention heads
        """

        super().__init__()

        self.heads = heads
        self.head_dim = embedding_size // heads
        self.embedding_size = embedding_size

        assert (self.head_dim * heads == embedding_size), "Embedding size needs to be dividable by heads"

        self.values_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.keys_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        self.queries_proj = nn.Linear(embedding_size, embedding_size, bias=False)

        self.fc_out = nn.Linear(embedding_size, embedding_size)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):

        # applying linear projections
        q = self.queries_proj(q)
        k = self.keys_proj(k)
        v = self.values_proj(v)

        # rearranging q, k, v from [batch_size, seq_len, embedding_size] -> [batch_size, seq_len, heads, head_size]
        q = einops.rearrange(q, "n l (h d) -> n l h d", h=self.heads)
        k = einops.rearrange(k, "n l (h d) -> n l h d", h=self.heads)
        v = einops.rearrange(v, "n l (h d) -> n l h d", h=self.heads)

        # shapes
        # query: (N, query_len, heads, head_dim)
        # keys: (N, key_len, heads, head_dim)
        # output: (N, heads, query_len, key_len)
        qk = torch.einsum("n q h d, n k h d -> n h q k", [q, k])

        if mask is not None:
            # setting masked values to -inf so that softmax will give then probability zero
            qk = qk.masked_fill(mask == 0, -float("-inf"))

        # applying softmax over key dimension to calculate attention scores
        attn = torch.softmax(qk * (self.embedding_size ** -0.5), dim=3)

        # shapes
        # attn: (N, heads, query_len, key_len)
        # values: (N, values_len, heads, head_dim)
        # output: (N, query_len, heads, head_dim)
        out = torch.einsum("n h q l, n l h d -> n q h d", [attn, v])

        # concatenation of heads
        out = einops.rearrange(out, "n l h d -> n l (h d)")

        return self.fc_out(out)



class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, heads: int, ffn_expansion: int, dropout_rate: float = 0.3):
        """
        Transformer Block

        Args:
            embedding_size (int): size of embedding dim
            heads (int): number of attention heads
            ffn_expansion (int): scaling factor for hidden dim expansion in feed forward layer
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        super().__init__()

        # expanded dimension for feed forward
        hidden_dim = embedding_size * ffn_expansion




        self.ln1 = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadAttention(embedding_size, heads)

        self.ln2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # skip connection
        skip = x

        # normalization
        x = self.ln1(x)

        # calculating attention
        x = self.attention(x, x, x, mask)

        # residual connection with input x
        x = x + skip
        skip = x

        # dropout
        x = self.dropout(x)

        # normalization
        x = self.ln2(x)

        # passing to feedforward layer
        x = self.feed_forward(x)

        # residual connection
        x = x + skip

        # dropout
        x = self.dropout(x)

        return x
    
    
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

class GPT(nn.Module):
    def __init__(self, embedding_size: int, seq_len: int, out_classes: int, num_blocks: int, heads: int, ffn_expansion: int, dropout_rate: float = 0.3):
        """
        GPT-1 Architecture

        Args:
            embedding_size (int): size of embedding dim
            seq_len (int): length of input sequence, used to construct mask
            out_classes (int): number of output classes
            num_blocks (int): number of transformer decoder blocks
            heads (int): number of attention heads in each block
            ffn_expansion (int): scaling factor for hidden dim expansion in feed forward layer
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
        """
        super().__init__()
        self.pos_enc = PositionalEncoding1D(embedding_size)

        self.gpt = nn.ModuleList(
            [
                TransformerBlock(embedding_size, heads, ffn_expansion, dropout_rate) for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(embedding_size, out_classes)

        mask = torch.tril(torch.ones((seq_len, seq_len)))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        pe = self.pos_enc(x)
        x = x + pe

        for layer in self.gpt:
            x = layer(x, self.mask)

        return self.classifier(x)