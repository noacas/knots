import pfrl
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Reformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:x.size(0), :]


class LSHSelfAttention(nn.Module):
    """Simplified Locality-Sensitive Hashing (LSH) Self-Attention module."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, n_hashes: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_hashes = n_hashes
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _lsh_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simplified implementation of LSH attention."""
        batch_size, seq_len, _ = q.size()

        # Simple dot-product attention as a placeholder for LSH attention
        # In a full implementation, we would use LSH bucketing here
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        return context

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # Project inputs to queries, keys and values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        context = self._lsh_attention(q, k, v, mask)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)

        return output


class ReversibleBlock(nn.Module):
    """Reversible block with two functions F and G."""

    def __init__(self, f_block: nn.Module, g_block: nn.Module):
        super().__init__()
        self.f_block = f_block
        self.g_block = g_block

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(y1)
        return y1, y2

    def backward_pass(self, y: Tuple[torch.Tensor, torch.Tensor],
                      dy: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom backward pass for memory efficiency."""
        y1, y2 = y
        dy1, dy2 = dy

        # Reconstruct x2
        x2 = y2 - self.g_block(y1)
        # Compute dx2
        dx2 = dy2 + dy1 @ self.f_block.weight

        # Reconstruct x1
        x1 = y1 - self.f_block(x2)
        # Compute dx1
        dx1 = dy1 + dx2 @ self.g_block.weight

        return (x1, x2), (dx1, dx2)


class FeedForward(nn.Module):
    """Simple feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ReformerBlock(nn.Module):
    """Single Reformer layer block."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 256,
                 dropout: float = 0.1, n_hashes: int = 4):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)

        self.attention = LSHSelfAttention(d_model, n_heads, dropout, n_hashes)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention block
        residual = x
        x = self.attn_layer_norm(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.ff_layer_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x

        return x


class ReformerNetwork(nn.Module):
    """Reformer-based network for RL policy and value functions."""

    def __init__(self, input_dim: int, output_dim: int, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4, d_ff: int = 256,
                 dropout: float = 0.1, n_hashes: int = 4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)

        self.reformer_blocks = nn.ModuleList([
            ReformerBlock(d_model, n_heads, d_ff, dropout, n_hashes)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input if it's not batched
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        # We treat the input as a sequence of length 1
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        pos = self.pos_encoding(torch.zeros(1, batch_size, x.size(-1))).transpose(0, 1)
        x = x + pos

        # Apply Reformer blocks
        for block in self.reformer_blocks:
            x = block(x)

        # Output projection (using the "sequence" final state)
        x = self.output_norm(x).squeeze(1)
        x = self.output_proj(x)

        return x


def create_reformer_policy(obs_size: int, action_size: int) -> nn.Module:
    """Create a policy network using Reformer architecture."""
    policy_network = ReformerNetwork(
        input_dim=obs_size,
        output_dim=action_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        n_hashes=4
    )

    return nn.Sequential(
        policy_network,
        pfrl.policies.SoftmaxCategoricalHead(),
    )


def create_reformer_vf(obs_size: int) -> nn.Module:
    """Create a value function network using Reformer architecture."""
    return ReformerNetwork(
        input_dim=obs_size,
        output_dim=1,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        n_hashes=4
    )


def ortho_init(layer, gain):
    """Initialize weights using orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def initialize_reformer_network(network):
    """Initialize a Reformer network."""
    for name, param in network.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.orthogonal_(param, gain=1.0)
        elif 'bias' in name:
            nn.init.zeros_(param)