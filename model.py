import math
import torch
import torch.nn as nn

from modules import (
    VisibilityAwareMultiScaleSpatialModeling,
    ReliabilityGuidedTemporalSelection,
    VisibilityConstrainedCollaborativeFusion,
    masked_average_pooling,
)


class PositionalEncooding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class OACSFMranch(nn.Module):

    def __init__(self, in_dim, embed_dim, num_classes, num_heads=8, num_groups=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.position = PositionalEncoding(embed_dim)
        self.vams = VisibilityAwareMultiScaleSpatialModeling(
            dim=embed_dim,
            num_groups=num_groups,
            dropout=dropout
        )
        self.rtsm = ReliabilityGuidedTemporalSelection(
            dim=embed_dim,
            temporal_kernel=7,
            dropout=dropout
        )
        self.vccf = VisibilityConstrainedCollaborativeFusion(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        x = self.proj(x)
        x = self.position(x)

        spatial_feat, visibility = self.vams(x, mask=mask)
        temporal_feat, temporal_weight, reliability = self.rtsm(spatial_feat, visibility=visibility, mask=mask)
        fused_feat = self.vccf(
            spatial_feat=spatial_feat,
            temporal_feat=temporal_feat,
            visibility=visibility,
            reliability=reliability,
            mask=mask
        )

        pooled = masked_average_pooling(fused_feat, mask=mask)
        logits = self.classifier(pooled)
        aux = {
            'visibility': visibility,
            'temporal_weight': temporal_weight,
            'reliability': reliability,
            'fused_feature': fused_feat,
        }
        return logits, aux


class OACSFM(nn.Module):


    def __init__(self, in_dim=1408, embed_dim=512, num_classes=7, num_heads=8, num_groups=4, dropout=0.1):
        super().__init__()
        self.shared_branch = OACSFMBranch(
            in_dim=in_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            num_groups=num_groups,
            dropout=dropout,
        )

    def forward(self, inputs_high, inputs_middle, inputs_low):
        x_high, mask_high = inputs_high
        x_middle, mask_middle = inputs_middle
        x_low, mask_low = inputs_low

        logits_high, aux_high = self.shared_branch(x_high, mask_high)
        logits_middle, aux_middle = self.shared_branch(x_middle, mask_middle)
        logits_low, aux_low = self.shared_branch(x_low, mask_low)

        logits_tuple = (logits_high, logits_middle, logits_low)
        aux_tuple = (aux_high, aux_middle, aux_low)
        return logits_tuple, aux_tuple
