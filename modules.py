import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityAwareMutiScaleSpatialModeling(nn.Module):


    def __init__(self, dim, num_groups=4, visibility_hidden_ratio=4, dropout=0.0):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(f"embed dim {dim} must be divisible by num_groups {num_groups}")

        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups

        kernels = [1, 3, 5, 7]
        if num_groups > len(kernels):
            kernels = kernels + [7] * (num_groups - len(kernels))
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(self.group_dim, self.group_dim, kernel_size=kernels[i],
                      padding=kernels[i] // 2, groups=self.group_dim)
            for i in range(num_groups)
        ])
        self.context_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.group_dim, self.group_dim, kernel_size=3, padding=1, groups=self.group_dim),
                nn.GELU(),
                nn.Conv1d(self.group_dim, self.group_dim, kernel_size=1)
            )
            for _ in range(num_groups)
        ])

        hidden = max(dim // visibility_hidden_ratio, 16)
        self.visibility_estimator = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        if x.dim() != 3:
            raise ValueError(f"VAMS expects input shape (B,T,C), got {tuple(x.shape)}")

        visibility = self.visibility_estimator(x).squeeze(-1)  # (B, T)
        if mask is not None:
            visibility = visibility * mask.float()

        x_t = x.transpose(1, 2)  # (B, C, T)
        chunks = torch.chunk(x_t, self.num_groups, dim=1)
        v = visibility.unsqueeze(1)  # (B, 1, T)

        scale_outputs = []
        for chunk, scale_conv, context_conv in zip(chunks, self.scale_convs, self.context_convs):
            local_response = scale_conv(chunk)
            context_response = context_conv(local_response)
            scale_outputs.append(v * local_response + (1.0 - v) * context_response)

        fused = torch.cat(scale_outputs, dim=1)
        fused = self.fusion(fused).transpose(1, 2)  # (B, T, C)
        enhanced = self.norm(x + self.dropout(fused))
        return enhanced, visibility


class ReliabilityGuidedTeporalSelection(nn.Module):


    def __init__(self, dim, reduction=16, temporal_kernel=7, dropout=0.0):
        super().__init__()
        hidden = max(dim // reduction, 16)
        padding = temporal_kernel // 2

        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=temporal_kernel, padding=padding),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _masked_pool(self, x, mask, mode='mean'):
        if mask is None:
            return x.mean(dim=1) if mode == 'mean' else x.max(dim=1).values

        mask_f = mask.unsqueeze(-1).float()
        if mode == 'mean':
            return (x * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)

        x_masked = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
        pooled = x_masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

    def forward(self, x, visibility=None, mask=None):

        avg_pool = self._masked_pool(x, mask, mode='mean')
        max_pool = self._masked_pool(x, mask, mode='max')
        channel_weight = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)).unsqueeze(1)
        x_channel = x * channel_weight

        avg_t = x_channel.mean(dim=-1, keepdim=True)
        max_t = x_channel.max(dim=-1, keepdim=True).values
        temporal_stat = torch.cat([avg_t, max_t], dim=-1).transpose(1, 2)  # (B, 2, T)
        temporal_weight = self.temporal_conv(temporal_stat).squeeze(1)  # (B, T)

        if visibility is None:
            reliability = temporal_weight
        else:
            reliability = temporal_weight * visibility

        if mask is not None:
            temporal_weight = temporal_weight * mask.float()
            reliability = reliability * mask.float()

        selected = x_channel * reliability.unsqueeze(-1)
        selected = self.norm(x + self.dropout(selected))
        return selected, temporal_weight, reliability


class VisibilityConstrinedCllaborativeFusion(nn.Module):


    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(1, num_heads)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        b, t, c = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, spatial_feat, temporal_feat, visibility=None, reliability=None, mask=None):
        b, tq, _ = spatial_feat.shape
        tk = temporal_feat.shape[1]

        q = self._reshape_heads(self.q_proj(spatial_feat))
        k = self._reshape_heads(self.k_proj(temporal_feat))
        v = self._reshape_heads(self.v_proj(temporal_feat))

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, Tq, Tk)

        if visibility is not None and reliability is not None:
            gate = visibility.unsqueeze(-1) * reliability.unsqueeze(1)  # (B, Tq, Tk)
            gate_bias = self.gate_proj(gate.unsqueeze(-1)).permute(0, 3, 1, 2)
            attn_logits = attn_logits + gate_bias

        if mask is not None:
            key_mask = ~mask.bool()
            attn_logits = attn_logits.masked_fill(key_mask[:, None, None, :], torch.finfo(attn_logits.dtype).min)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, tq, self.dim)
        out = self.out_proj(out)
        return self.norm(spatial_feat + self.dropout(out))


def masked_average_pooling(x, mask=None):
    if mask is None:
        return x.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
