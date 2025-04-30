import gymnasium as gym
import torch
import torch.nn as nn
from reformer_pytorch import Reformer
from reformer_pytorch.reformer_pytorch import Always, default, AbsolutePositionalEmbedding
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import Identity


class ReformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, depth, features_dim, heads=8, dim_head=64, n_hashes=4,
                 ff_chunks=100, attn_chunks=1, causal=False, weight_tie=False, lsh_dropout=0., ff_dropout=0., ff_mult=4,
                 ff_activation=None, ff_glu=False, layer_dropout=0.,
                 random_rotations_per_head=False, use_scale_norm=False, use_rezero=False, use_full_attn=False,
                 full_attn_thres=0, reverse_thres=0, num_mem_kv=0, one_value_head=False, emb_dim=None,
                 n_local_attn_heads=0, pooling='mean'):

        super(ReformerFeatureExtractor, self).__init__(observation_space, features_dim)

        dim = features_dim
        max_seq_len = observation_space.shape[0]

        bucket_size=min(64, dim // 2 if dim > 4 else dim)

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
            nn.Linear(emb_dim, features_dim)
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
