import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k, stride=1, padding=0, bias=True, norm=True, pool=None):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, k, stride=stride, padding=padding, bias=bias)
        self.activation = nn.SiLU()
        self.normal = nn.BatchNorm1d(out_dim) if norm else None
        self.pool = pool

    def forward(self, x, mask):
        h = self.conv(x)  # (B, C, L)
        if mask is not None:
            h = h * mask.unsqueeze(1).float()
        if self.normal is not None:
            h = self.normal(h)  # BN 按通道归一化
        if self.activation is not None:
            h = self.activation(h)
        if self.pool is not None:
            h = self.pool(h)
        return h


class SignalConvBlock(nn.Module):
    def __init__(self, in_dim=1, out_dim=128):
        super().__init__()
        self.convs = nn.Sequential(
            ConvolutionLayer(in_dim, out_dim // 4, 3, 2, 1),  # 175 -> ~88
            ConvolutionLayer(out_dim // 4, out_dim // 2, 3, 2, 1),  # 88 -> ~44
            ConvolutionLayer(out_dim // 2, out_dim, 3, 1, 1)  # 44 -> 44
        )
        # 保存各层的池化参数（用于掩码处理）
        self.pool_params = [
            {"kernel_size": 3, "stride": 2, 'padding': 1},  # 第一层后 175 -> ?
            {"kernel_size": 3, "stride": 2, 'padding': 1},  # 第二层后 ?
            None,  # 第三层无池化
        ]

    def forward(self, x, mask=None):

        for i, layer in enumerate(self.convs):
            if mask is not None and self.pool_params[i] is not None:
                mask = self.pool_mask(mask, **self.pool_params[i])
            x = layer(x, mask)
        return x, mask

    def pool_mask(self, mask, kernel_size, stride, padding):
        mask = F.pad(mask, (padding, padding), value=False)  # 填充 False
        mask_chunks = mask.unfold(-1, kernel_size, stride)  # 滑窗
        mask_out, _ = mask_chunks.max(dim=-1)  # 最大池化
        return mask_out


class ResidualConnect(nn.Module):
    def __init__(self, module: nn.Module, module_weight: float = 1.0, input_weight: float = 1.0):
        super(ResidualConnect, self).__init__()
        self.module = module
        self.module_weight = module_weight
        self.input_weight = input_weight

    def forward(self, x, mask=None):
        out = (self.module(x, mask) * self.module_weight) + (x * self.input_weight)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


class FFN(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128, dropout: float = 0.3, expand: int = 2):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=input_dim * expand, bias=False),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=input_dim * expand, out_features=output_dim, bias=False),
            nn.Dropout(p=dropout)
        )

    def forward(self, x, mask=None):
        x = self.model(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class SignalConv(nn.Module):
    """
    受inception块的启发，使用不同尺寸的卷积核，之后使用1*1初步融合

    """

    def __init__(self, in_channel: int, out_channel: int, mid_channel=None, bias=False, norm=True):
        super(SignalConv, self).__init__()
        if mid_channel is None:
            mid_channel = out_channel
        self.conv_3 = ConvolutionLayer(in_channel, mid_channel, 3, stride=1, padding=1, bias=bias,
                                       norm=True)
        self.conv_5 = ConvolutionLayer(in_channel, mid_channel, 5, stride=1, padding=2, bias=bias,
                                       norm=True)
        self.conv_7 = ConvolutionLayer(in_channel, mid_channel, 7, stride=1, padding=3, bias=bias,
                                       norm=True)

        self.conv_1 = ConvolutionLayer(mid_channel * 3, out_channel, 1, stride=1, padding=0, bias=bias,
                                       norm=norm)

    def forward(self, x, mask=None):
        feature_3 = self.conv_3(x, mask)
        feature_5 = self.conv_5(x, mask)
        feature_7 = self.conv_7(x, mask)
        feature = torch.cat((feature_3, feature_5, feature_7), 1)
        return self.conv_1(feature, mask)


class SignalBlockConvModule(nn.Module):

    def __init__(
            self,
            in_channels: int,
            dropout_p: float = 0.3,
    ) -> None:
        super(SignalBlockConvModule, self).__init__()

        self.conv = SignalConv(in_channel=in_channels, out_channel=in_channels, mid_channel=in_channels // 2, bias=True,
                               norm=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv1_1 = ConvolutionLayer(in_channels, in_channels, 1, stride=1, padding=0, bias=True,
                                        norm=True)

    def forward(self, x, mask=None):
        out = self.conv(x.transpose(1, 2))
        out = self.conv1_1(out, mask)
        out = self.dropout(out).transpose(1, 2)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        # return self.sequential(x).transpose(1, 2)
        return out


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int = 55, dim: int = 128):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))  # 可学习位置编码

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # 加上前 L 个位置编码
        return x


class LongRangeSignal(nn.Module):
    def __init__(self, input_dim, mha_head, dp=0.3):
        super(LongRangeSignal, self).__init__()
        self.pos = LearnablePositionalEncoding()
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=mha_head, dropout=dp, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        x = self.pos(x)
        mha_mask = ~mask
        if mask is not None:
            out = self.norm(self.mha(x, x, x, key_padding_mask=mha_mask, need_weights=False)[0])
            out = out * mask.unsqueeze(-1).float()
        else:
            out = self.norm(self.mha(x, x, x, need_weights=False)[0])
        return out


class SignalBlock(nn.Module):

    def __init__(
            self,
            input_dim: int = 128,
            ffn_expand: int = 2,
            ffn_dp=0.3,

            mha_head=8,
            mha_dp=0.3,

            conv_dropout_p: float = 0.3,
            ffn_residual: float = 0.5,
    ):
        super(SignalBlock, self).__init__()

        self.global_sig = ResidualConnect(
            LongRangeSignal(input_dim=input_dim, mha_head=mha_head, dp=mha_dp),
        )

        self.local_sig = ResidualConnect(
            module=SignalBlockConvModule(
                in_channels=input_dim,
                dropout_p=conv_dropout_p,
            ),
        )
        self.ffn = ResidualConnect(
            module=FFN(
                input_dim=input_dim,
                expand=ffn_expand,
                dropout=ffn_dp,
                output_dim=input_dim,
            ),
            module_weight=ffn_residual,
        )
        self.normal = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        out = self.local_sig(x, mask)
        out = self.global_sig(out, mask.bool())
        out = self.ffn(out, mask)
        out = self.normal(out)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


if __name__ == "__main__":
    B = 4
    x = torch.randn(B, 1, 125)
    mask = torch.ones(B, 125).bool()
    stem = SignalConvBlock()
    y, m = stem(x, mask)
    print(y.shape)  # torch.Size([4, 128, 32])
    print(m.shape)  # torch.Size([4, 32])
