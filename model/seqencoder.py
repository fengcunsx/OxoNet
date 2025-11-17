import torch.nn as nn
import torch

from model.seqlayer import BaseResidualBlock


class SeqEncoder(nn.Module):
    def __init__(self, base_embedding_dim=16, out_channel=128):
        super(SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(4, base_embedding_dim)
        self.conv_block = nn.Sequential(
            BaseResidualBlock(16, out_channel // 4),
            BaseResidualBlock(out_channel // 4, out_channel // 2),
            BaseResidualBlock(out_channel // 2, out_channel),
        )

    # def forward(self, base_stat, seq):
    #     seq_info = self.embedding(seq)
    #     # feature = torch.cat((base_stat, seq_info), dim=-1).permute(0, 2, 1)
    #     # feature = self.conv_block(feature)
    #     feature = seq_info .permute(0, 2, 1)
    #     feature = self.conv_block(feature)
    #     return feature.permute(0, 2, 1)

    def forward(self, seq):
        seq_info = self.embedding(seq)
        feature = seq_info .permute(0, 2, 1)
        feature = self.conv_block(feature)
        return feature.permute(0, 2, 1)
