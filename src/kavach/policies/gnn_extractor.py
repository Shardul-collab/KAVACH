"""
gnn_extractor.py — custom stable-baselines3 features extractor that replaces
the default "MlpPolicy" flat-vector processing with actual message-passing
graph convolution, for use with QKDRoutingEnvGNN's Dict observation space.

Why this shape, not a library (torch_geometric etc.): the topology is FIXED
for a given training run (edge_index is identical every step, only edge/node
FEATURES change), which means the usual reason to reach for a sparse-graph
library — variable graph structure per sample requiring batched sparse
indexing — doesn't apply here. Every sample in a batch shares the same
edge_index, so plain torch.index_add_ over a shared index array is both
simpler and correct; no PyG dependency needed.

Architecture: edge-conditioned message passing (not vanilla GCN, since
vanilla GCN's propagation rule only uses binary adjacency and ignores edge
features — and the whole point here is that per-edge SKR/buffer-level/
feasibility/coexistence data should inform routing, the same information
your hand-engineered candidate summary already captured, just now consumed
via learned message passing over the real graph instead of pre-selected by
k_shortest_candidates before the network ever sees it).

Per layer, for each directed edge (u -> v):
    message  = MLP([h_u, edge_feat(u,v)])
    h_v_new  = ReLU(W_self @ h_v + mean_{u in N(v)} message)

Output features_dim = k*hidden_dim (per-candidate pooled embeddings)
                     + hidden_dim   (global mean-pooled graph embedding)
                     + 1            (demand_feat)

hidden_dim=384, n_layers=3 -> ~1.34M params in this extractor alone,
~1.59M total once SB3's default actor/critic heads (net_arch=[64,64]) are
added on top — matches train_cifarnet.py's CIFARNet (1,588,618 params)
closely, by design, for a size comparison between the two paradigms.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 384, n_layers: int = 3):
        node_f = observation_space["node_feat"].shape[1]
        edge_f = observation_space["edge_feat"].shape[1]
        k = observation_space["candidate_masks"].shape[0]

        features_dim = k * hidden_dim + hidden_dim + 1
        super().__init__(observation_space, features_dim=features_dim)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.k = k

        self.node_embed = nn.Linear(node_f, hidden_dim)
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + edge_f, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.self_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        node_feat = obs["node_feat"]                      # (B, N, node_f)
        edge_index = obs["edge_index"].long()              # (B, 2, E)
        edge_feat = obs["edge_feat"]                        # (B, E, edge_f)
        node_mask = obs["node_mask"]                        # (B, N)
        cand_masks = obs["candidate_masks"]                  # (B, k, N)
        demand_feat = obs["demand_feat"]                     # (B, 1)

        B, N, _ = node_feat.shape
        E = edge_feat.shape[1]

        # Topology is fixed per training run, so edge_index is identical across
        # every sample in the batch — take the first sample's structure.
        src_idx = edge_index[0, 0]  # (E,)
        dst_idx = edge_index[0, 1]  # (E,)

        h = F.relu(self.node_embed(node_feat))  # (B, N, H)

        deg = torch.zeros(N, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst_idx, torch.ones(E, device=h.device, dtype=h.dtype))
        deg = deg.clamp(min=1.0)

        for msg_mlp, self_transform in zip(self.msg_mlps, self.self_transforms):
            h_src = h[:, src_idx, :]                          # (B, E, H)
            msg_in = torch.cat([h_src, edge_feat], dim=-1)      # (B, E, H+edge_f)
            msg = msg_mlp(msg_in)                                # (B, E, H)

            agg = torch.zeros(B, N, self.hidden_dim, device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_idx, msg)
            agg = agg / deg.view(1, N, 1)

            h = F.relu(self_transform(h) + agg)
            h = h * node_mask.unsqueeze(-1)

        cand_counts = cand_masks.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, k, 1)
        cand_embed = torch.bmm(cand_masks, h) / cand_counts                 # (B, k, H)
        cand_embed = cand_embed.reshape(B, self.k * self.hidden_dim)

        node_counts = node_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)   # (B, 1)
        global_embed = (h * node_mask.unsqueeze(-1)).sum(dim=1) / node_counts  # (B, H)

        return torch.cat([cand_embed, global_embed, demand_feat], dim=-1)


if __name__ == "__main__":
    # Architecture-only sanity check (shapes + param count), not a training run.
    from gymnasium.spaces import Box, Dict as DictSpace
    import numpy as np

    max_nodes, max_edges_dir, k = 300, 600, 4
    obs_space = DictSpace({
        "node_feat": Box(-1, 1, (max_nodes, 4), dtype=np.float32),
        "edge_index": Box(0, max_nodes - 1, (2, max_edges_dir), dtype=np.int64),
        "edge_feat": Box(-1, 1, (max_edges_dir, 4), dtype=np.float32),
        "node_mask": Box(0, 1, (max_nodes,), dtype=np.float32),
        "candidate_masks": Box(0, 1, (k, max_nodes), dtype=np.float32),
        "demand_feat": Box(0, 1, (1,), dtype=np.float32),
    })
    extractor = GNNFeaturesExtractor(obs_space, hidden_dim=384, n_layers=3)
    n_params = sum(p.numel() for p in extractor.parameters())
    print(f"extractor params: {n_params:,}  (features_dim={extractor.features_dim})")

    B = 8
    dummy = {
        "node_feat": torch.randn(B, max_nodes, 4),
        "edge_index": torch.randint(0, max_nodes, (B, 2, max_edges_dir)),
        "edge_feat": torch.randn(B, max_edges_dir, 4),
        "node_mask": torch.ones(B, max_nodes),
        "candidate_masks": torch.zeros(B, k, max_nodes),
        "demand_feat": torch.rand(B, 1),
    }
    out = extractor(dummy)
    print(f"output shape: {tuple(out.shape)}")
