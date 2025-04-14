import torch.nn as nn
import pfrl
from reformer_pytorch import Reformer
from reformer_pytorch.reformer_pytorch import Always, default, AbsolutePositionalEmbedding
from torch.nn import Identity


class ReformerKnots(nn.Module):
    def __init__(self, dim, depth, max_seq_len, output_dim, heads=8, dim_head=64, bucket_size=64, n_hashes=4,
                 ff_chunks=100, attn_chunks=1, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_mult=4,
                 ff_activation=None, ff_glu=False, layer_dropout=0.,
                 random_rotations_per_head=False, use_scale_norm=False, use_rezero=False, use_full_attn=False,
                 full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False, emb_dim=None,
                 n_local_attn_heads=0, pooling='mean'):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.pooling = pooling  # How to aggregate sequence into single representation

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
        self.layer_pos_emb = Always(None)

        self.reformer = Reformer(dim, depth, heads=heads, dim_head=dim_head, bucket_size=bucket_size, n_hashes=n_hashes,
                                 ff_chunks=ff_chunks, attn_chunks=attn_chunks, causal=causal, weight_tie=weight_tie,
                                 lsh_dropout=lsh_dropout, ff_mult=ff_mult, ff_activation=ff_activation, ff_glu=ff_glu,
                                 ff_dropout=ff_dropout, post_attn_dropout=0., layer_dropout=layer_dropout,
                                 random_rotations_per_head=random_rotations_per_head, use_scale_norm=use_scale_norm,
                                 use_rezero=use_rezero, use_full_attn=use_full_attn, full_attn_thres=full_attn_thres,
                                 reverse_thres=reverse_thres, num_mem_kv=num_mem_kv, one_value_head=one_value_head,
                                 n_local_attn_heads=n_local_attn_heads)
        self.norm = nn.LayerNorm(dim)

        self.out = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, output_dim)
        )

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]

        # Reshape x to be [batch_size, seq_len, dim]
        # For braids, seq_len is the number of elements in the braid

        if len(x.shape) == 2:
            # x is [batch_size, braid_elements]
            # Reshape to [batch_size, braid_elements, 1]
            x = x.unsqueeze(-1)

            # If your elements need to be embedded into a higher dimension
            # You might want to add an embedding layer here
            # For example: x = self.embedding(x.long())

        # If your input is already [batch_size, braid_elements, feature_dim]
        # but feature_dim != self.dim, we need to project
        if x.shape[2] != self.dim:
            # Project to model dimension
            x = self.to_model_dim(x)

        # Add positional embeddings
        x = x + self.pos_emb(x)

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.reformer(x, pos_emb=layer_pos_emb, **kwargs)
        x = self.norm(x)

        # Aggregate sequence representation into a single vector for the entire braid
        if self.pooling == 'mean':
            # Mean pooling across sequence dimension
            x = x.mean(dim=1)  # [batch_size, dim]
        elif self.pooling == 'max':
            # Max pooling across sequence dimension
            x = x.max(dim=1)[0]  # [batch_size, dim]
        elif self.pooling == 'first':
            # Take representation of first token
            x = x[:, 0, :]  # [batch_size, dim]
        elif self.pooling == 'last':
            # Take representation of last token
            x = x[:, -1, :]  # [batch_size, dim]

        # Output prediction for the entire braid
        return self.out(x)  # [batch_size, output_dim]


class PolicyWrapper(nn.Module):
    def __init__(self, reformer_model, softmax_head):
        super().__init__()
        self.reformer = reformer_model
        self.softmax_head = softmax_head

    def forward(self, x):
        # Process the braid input to get a single representation for the entire braid
        action_logits = self.reformer(x)  # [batch_size, output_dim]

        # Apply softmax head for action probabilities
        action_probs = self.softmax_head(action_logits)
        return action_probs


def create_reformer_policy(obs_size: int, action_size: int, pooling='mean') -> nn.Module:
    """Create a policy network using Reformer architecture.

    Args:
        obs_size: Size of the observation (braid element count)
        action_size: Number of possible actions
        pooling: Method to aggregate sequence ('mean', 'max', 'first', or 'last')
    """
    # For braids, we want to set the dimension to a reasonable embedding size
    embedding_dim = 64

    reformer = ReformerKnots(
        dim=embedding_dim,  # Dimension of each token embedding
        depth=6,
        max_seq_len=obs_size,  # The maximum braid length
        heads=8,
        bucket_size=min(64, obs_size // 2 if obs_size > 4 else obs_size),
        n_hashes=4,
        ff_chunks=10,
        lsh_dropout=0.1,
        causal=False,  # Non-causal since we want each element to attend to all other elements
        n_local_attn_heads=2,
        use_full_attn=(obs_size <= 64),  # Use full attention for small braids
        output_dim=action_size,
        pooling=pooling,  # How to pool sequence into a single representation
    )
    return PolicyWrapper(reformer, pfrl.policies.SoftmaxCategoricalHead())


def create_reformer_vf(obs_size: int, pooling='mean') -> nn.Module:
    """Create a value function network using Reformer architecture."""
    # Embedding dimension for each braid element
    embedding_dim = 64

    return ReformerKnots(
        dim=embedding_dim,
        depth=6,
        max_seq_len=obs_size,
        heads=8,
        bucket_size=min(64, obs_size // 2 if obs_size > 4 else obs_size),
        n_hashes=4,
        ff_chunks=10,
        lsh_dropout=0.1,
        causal=False,  # Non-causal for braid self-attention
        n_local_attn_heads=2,
        use_full_attn=(obs_size <= 64),  # Use full attention for small braids
        output_dim=1,
        pooling=pooling,  # How to pool sequence into a single representation
    )