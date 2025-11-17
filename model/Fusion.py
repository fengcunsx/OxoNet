import torch.nn as nn
import torch
import torch.nn.functional as F

from model.signallayer import ResidualConnect


def masked_mean_pooling(x, mask):
    """
        x: (B, L, D)
        mask: (B, L)
        """
    if mask is None:
        _, count, _ = x.shape
        sum_x = x.sum(dim=1)
        return sum_x / count
    mask = mask.unsqueeze(-1).float()  # (B, L, 1)
    masked_x = x * mask  # 清零 padding 区域
    sum_x = masked_x.sum(dim=1)  # (B, D)
    count = mask.sum(dim=1).clamp(min=1e-6)  # 防止除以 0
    return sum_x / count  # (B, D)


class PredHead(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.pool = masked_mean_pooling
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, sig_feat, sig_mask=None):
        pooled = self.pool(sig_feat, sig_mask)
        embed = F.normalize(pooled, p=2, dim=-1)
        return embed, self.fc(pooled)


class Fusion(nn.Module):
    def __init__(self, dim=128, mha_dim=4, dropout=0.3, seq_l=5):
        super(Fusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=mha_dim, dropout=dropout,
                                                     batch_first=True)
        self.mid = seq_l // 2
        self.norm = nn.LayerNorm(dim)
        self.dim = dim * seq_l
        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )
        self.pred_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, sig_feat, seq_feat, sig_mask=None):
        mask = (~sig_mask) if (sig_mask is not None) else None
        attn_out = self.norm(
            self.cross_attention(query=seq_feat, key=sig_feat, value=sig_feat, key_padding_mask=mask)[0])
        feat = self.ffn(attn_out)
        embed = F.normalize(feat, p=2, dim=-1)
        # feat = attn_out[:, self.mid, :]
        # embed = F.normalize(feat, p=2, dim=-1)
        prob = self.pred_head(feat)
        return embed, prob
    # feat = F.normalize(attn_out[:, self.mid, :], p=2, dim=-1)
    # prob = self.pred_head(self.ffn(attn_out))

# class CrossAttention(nn.Module):
#     def __init__(self, dim=128, mha_dim=4, dropout=0.3, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=mha_dim, dropout=dropout,
#                                                      batch_first=True)
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, sig_feat, seq_feat, sig_mask=None):
#         mask = (~sig_mask) if (sig_mask is not None) else None
#         attn_out = self.cross_attention(query=seq_feat, key=sig_feat, value=sig_feat, key_padding_mask=mask)[0]
#
#         return self.norm(attn_out)
#
#
# class FusionBlock(nn.Module):
#     def __init__(self, dim=128, mha_dim=4, dropout=0.3, weight=1.0):
#         super(FusionBlock, self).__init__()
#         self.weight = weight
#         self.cross_attention = CrossAttention(dim=dim, mha_dim=mha_dim, dropout=dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * 2, dim),
#             nn.SiLU(),
#             nn.LayerNorm(dim)
#         )
#
#     def forward(self, sig_feat, seq_feat, sig_mask=None):
#         mask = (~sig_mask) if (sig_mask is not None) else None
#         attn_out = self.cross_attention(sig_feat, seq_feat, mask)
#
#         attn_out = attn_out * self.weight + seq_feat
#
#         ffn_out = self.ffn(attn_out) * self.weight + attn_out
#         return ffn_out
#
#
# class Fusion(nn.Module):
#     def __init__(self, dim=128, mha_dim=4, dropout=0.3, seq_l=5, n_block=2):
#         super(Fusion, self).__init__()
#         self.decoder = nn.Sequential(*[
#             FusionBlock(dim=dim, mha_dim=mha_dim, dropout=dropout, weight=1) for _ in range(n_block)
#         ])
#         self.mid = seq_l // 2
#         self.pred_head = nn.Sequential(
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, sig_feat, seq_feat, sig_mask=None):
#         for layer in self.decoder:
#             seq_feat = layer(sig_feat, seq_feat, sig_mask=sig_mask)
#         feat = sig_feat[:, self.mid, :]
#         embed = F.normalize(feat, p=2, dim=-1)
#         prob = self.pred_head(feat)
#         return embed, prob
#     # feat = F.normalize(attn_out[:, self.mid, :], p=2, dim=-1)
#     # prob = self.pred_head(self.ffn(attn_out))
