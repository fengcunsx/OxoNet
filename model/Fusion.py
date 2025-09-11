import torch.nn as nn
import torch


class Fusion(nn.Module):
    def __init__(self, dim=128, mha_dim=4, dropout=0.3, seq_l=5):
        super(Fusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=mha_dim, dropout=dropout,
                                                     batch_first=True)
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

    def forward(self, sig_feat, seq_feat,sig_mask=None):
        mask = (~sig_mask) if (sig_mask is not None) else None
        attn_out = self.norm(self.cross_attention(query=seq_feat, key=sig_feat, value=sig_feat,key_padding_mask=mask)[0])
        feat = self.ffn(attn_out)
        prob = self.pred_head(feat)
        return feat, prob
