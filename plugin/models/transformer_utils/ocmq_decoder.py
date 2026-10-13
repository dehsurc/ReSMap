"""Object-Centric Cooperative Multi-modal Query (OCMQ) decoder (Sec. 3.3)."""
import torch
import torch.nn as nn

from .CustomMSDeformableAttention import CustomMSDeformableAttention

MODALITIES = ('cam', 'sat', 'fuse')


class OCMQDecoderLayer(nn.Module):
    """One OCMQ layer.

    1. Each object query is cloned into camera / satellite / fused copies by
       adding a learned modality embedding.
    2. Self-attention within each modality group (Eq. 4).
    3. Each copy cross-attends to its own BEV with deformable attention (Eq. 5).
    4. A softmax gate merges the three evidence tokens back into one query (Eq. 6).
    5. Self-attention over the merged queries, then an FFN.
    """

    def __init__(self, embed_dims=512, num_heads=8, num_points=20, ffn_dims=1024,
                 dropout=0.1):
        super().__init__()
        self.modality_embed = nn.Embedding(3, embed_dims)

        self.cam_intra_sa = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.sat_intra_sa = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.fuse_intra_sa = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.cam_intra_norm = nn.LayerNorm(embed_dims)
        self.sat_intra_norm = nn.LayerNorm(embed_dims)
        self.fuse_intra_norm = nn.LayerNorm(embed_dims)
        self.intra_dropout = nn.Dropout(dropout)

        self.cam_cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1,
            num_points=num_points, dropout=dropout, batch_first=True)
        self.sat_cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1,
            num_points=num_points, dropout=dropout, batch_first=True)
        self.fuse_cross_attn = CustomMSDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1,
            num_points=num_points, dropout=dropout, batch_first=True)
        self.cam_cross_norm = nn.LayerNorm(embed_dims)
        self.sat_cross_norm = nn.LayerNorm(embed_dims)
        self.fuse_cross_norm = nn.LayerNorm(embed_dims)

        self.modality_gate = nn.Linear(embed_dims, 1)
        self.fusion_norm = nn.LayerNorm(embed_dims)

        self.post_sa = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.post_sa_norm = nn.LayerNorm(embed_dims)
        self.post_sa_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dims)

    @staticmethod
    def drop_mask(drop_groups, B, device):
        """(B, 3) mask of the evidence tokens to exclude from the gate."""
        if drop_groups is None:
            return None
        mask = torch.zeros(B, 3, dtype=torch.bool, device=device)
        for b, group in enumerate(drop_groups):
            if group is not None:
                mask[b, MODALITIES.index(group)] = True
        return mask if mask.any() else None

    def forward(self, query, ref_pts, bev_inputs, drop_groups=None):
        """
        Args:
            query: (B, N, D) object queries.
            ref_pts: (B, N, P, 2) reference polylines in [0, 1].
            bev_inputs: (tokens, spatial_shapes, level_start) for the camera,
                satellite and fused BEV, in that order.
        """
        B = query.shape[0]
        intra_sa = (self.cam_intra_sa, self.sat_intra_sa, self.fuse_intra_sa)
        intra_norm = (self.cam_intra_norm, self.sat_intra_norm, self.fuse_intra_norm)
        cross_attn = (self.cam_cross_attn, self.sat_cross_attn, self.fuse_cross_attn)
        cross_norm = (self.cam_cross_norm, self.sat_cross_norm, self.fuse_cross_norm)

        # BEV rows run top-down, reference points bottom-up.
        flipped_ref_pts = ref_pts.clone()
        flipped_ref_pts[..., 1:2] = 1.0 - ref_pts[..., 1:2]

        group_queries = [query + self.modality_embed.weight[m].view(1, 1, -1) for m in range(3)]
        group_sa = [intra_sa[m](q, q, q)[0] for m, q in enumerate(group_queries)]
        group_queries = [intra_norm[m](q + self.intra_dropout(sa))
                         for m, (q, sa) in enumerate(zip(group_queries, group_sa))]

        evidence = []
        for m, (tokens, shapes, level_start) in enumerate(bev_inputs):
            q = group_queries[m]
            cross = cross_attn[m](
                query=q, value=tokens, identity=torch.zeros_like(q),
                reference_points=flipped_ref_pts, spatial_shapes=shapes,
                level_start_index=level_start)
            evidence.append(cross_norm[m](q + cross))
        evidence = torch.stack(evidence, dim=2)  # (B, N, 3, D)

        gate_logits = self.modality_gate(evidence)
        mask = self.drop_mask(drop_groups, B, query.device)
        if mask is not None:
            gate_logits = gate_logits.masked_fill(mask[:, None, :, None], float('-inf'))
        alpha = torch.softmax(gate_logits, dim=2)
        query = self.fusion_norm((alpha * evidence).sum(dim=2))

        post_sa, _ = self.post_sa(query, query, query)
        query = self.post_sa_norm(query + self.post_sa_dropout(post_sa))
        return self.ffn_norm(query + self.ffn(query))


class OCMQDecoder(nn.Module):
    """Stack of OCMQ layers with iterative reference-polyline refinement."""

    def __init__(self, num_layers=6, embed_dims=512, num_heads=8, num_points=20,
                 ffn_dims=1024, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_points = num_points
        self.layers = nn.ModuleList([
            OCMQDecoderLayer(embed_dims, num_heads, num_points, ffn_dims, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, ref_pts, reg_branches, cam_inputs, sat_inputs, fuse_inputs,
                drop_groups=None):
        """Returns per-layer queries and refined reference polylines."""
        bev_inputs = (cam_inputs, sat_inputs, fuse_inputs)
        inter_queries = []
        inter_ref_pts = []
        for layer_idx, layer in enumerate(self.layers):
            query = layer(query, ref_pts, bev_inputs, drop_groups=drop_groups)
            new_ref_pts = reg_branches[layer_idx](query).view(
                -1, query.shape[1], self.num_points, 2).sigmoid()
            inter_queries.append(query)
            inter_ref_pts.append(new_ref_pts)
            ref_pts = new_ref_pts.detach()
        return inter_queries, inter_ref_pts
