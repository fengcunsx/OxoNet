import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class DeformConv1D(nn.Module):
    """
    1D Deformable Convolution using torch.grid_sample (stable version)
    - 支持 modulation (mask)
    - 使用 kernel-center 对齐
    - 正确处理 grid_sample 的 (x,y) 顺序：x 对应 width（这里为 1），y 对应 height（序列长度）
    - align_corners=True，对归一化做精确映射
    注意：当前实现假设 groups==1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 modulation=True, max_offset=None):
        super().__init__()
        assert groups == 1, "当前实现暂不支持 groups>1"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.modulation = modulation
        self.groups = groups
        self.max_offset = max_offset if max_offset is not None else kernel_size // 2

        # 主卷积权重（未实际用 F.conv1d，而是展开后 matmul）
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        # 初始化主权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # offset 分支 —— 初始化为 0（初期等效普通卷积）
        self.offset_conv = nn.Conv1d(in_channels, kernel_size,
                                     kernel_size=kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # modulation mask 分支（可选）—— bias 设为 0.2，使 sigmoid 初期略大于 0.5
        if modulation:
            self.mask_conv = nn.Conv1d(in_channels, kernel_size,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, bias=True)
            nn.init.constant_(self.mask_conv.weight, 0.)
            nn.init.constant_(self.mask_conv.bias, 0.2)
        else:
            self.mask_conv = None

    def forward(self, x):
        """
        x: (N, C, L)
        return: (N, out_channels, L_out)
        """
        N, C, L = x.shape
        k = self.kernel_size

        # 1) 预测 offset & mask
        offset = self.offset_conv(x)  # (N, k, L_out)
        # 限制 offset 范围（平滑）
        offset = torch.tanh(offset) * float(self.max_offset)

        if self.modulation:
            mask = torch.sigmoid(self.mask_conv(x))  # (N, k, L_out)
        else:
            mask = torch.ones_like(offset, device=x.device, dtype=x.dtype)

        L_out = offset.shape[-1]

        # 2) 生成基础采样位置：以 kernel center 对齐
        # base kernel positions (centered)
        # example k=5 -> [-2, -1, 0, 1, 2] (times dilation)
        kernel_idx = torch.arange(-(k - 1) // 2, (k - 1) // 2 + 1.0, device=x.device, dtype=x.dtype)
        kernel_idx = kernel_idx.view(1, k, 1) * float(self.dilation)  # (1, k, 1)

        # output positions in input coords: pos = i*stride (+ maybe -padding correction)
        out_pos = torch.arange(0, L_out, device=x.device, dtype=x.dtype) * float(self.stride)
        out_pos = out_pos.view(1, 1, L_out)  # (1,1,L_out)

        # combine: (N, k, L_out)
        base_grid = out_pos + kernel_idx  # (1, k, L_out)
        base_grid = base_grid.expand(N, -1, -1)  # (N, k, L_out)
        pos = base_grid + offset  # (N, k, L_out)

        # 3) clamp 到输入范围（避免 grid_sample 采样到 nan）
        pos = pos.clamp(0.0, float(max(0, L - 1)))

        # 4) 归一化到 grid_sample 的坐标系 [-1, 1]
        #  grid_sample 对 (N,C,H,W) 的 grid : grid[...,0] = x (width) in [-1,1]
        #                                         grid[...,1] = y (height) in [-1,1]
        # 我们把输入扩展为 (N, C, H=L, W=1)，因此 x 维只有一个点 -> 固定为 0.0 (中心)
        # y 维对应序列索引，需要按照 align_corners=True 的公式归一化：
        #   y_norm = (2 * pos / (L-1)) - 1
        if L == 1:
            # 防止除0
            y_norm = torch.zeros_like(pos)
        else:
            y_norm = (pos / (L - 1.0)) * 2.0 - 1.0  # (N, k, L_out)

        # reshape为 grid_sample 需要的形状 (N, H_out, W_out, 2)
        # 我们希望 H_out = L_out, W_out = k
        # current y_norm: (N, k, L_out) -> transpose to (N, L_out, k)
        y_norm_t = y_norm.transpose(1, 2).contiguous()  # (N, L_out, k)
        x_norm_t = torch.zeros_like(y_norm_t)  # (N, L_out, k)  # x coord = 0 (center)

        grid = torch.stack([x_norm_t, y_norm_t], dim=-1)  # (N, L_out, k, 2)  注意 (x,y) 顺序

        # 5) 扩展输入为 2D image 格式： (N, C, H=L, W=1)
        x_img = x.unsqueeze(-1)  # (N, C, L, 1)

        # 6) 使用 grid_sample：输出 (N, C, H_out=L_out, W_out=k)
        sampled = F.grid_sample(x_img, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        # sampled: (N, C, L_out, k)

        # 7) 应用 mask（broadcast到channel）
        # mask: (N, k, L_out) ->  permute -> (N, L_out, k) -> unsqueeze channel dim -> (N,1,L_out,k)
        mask_t = mask.permute(0, 2, 1).unsqueeze(1)  # (N,1,L_out,k)
        sampled = sampled * mask_t  # (N, C, L_out, k)

        # 8) 重排与矩阵乘法
        # (N, C, L_out, k) -> (N, L_out, C, k)
        sampled = sampled.permute(0, 2, 1, 3).contiguous()
        sampled = sampled.view(N, L_out, C * k)  # (N, L_out, C*k)

        weight_flat = self.weight.view(self.out_channels, -1)  # (out, C*k)

        # out: (N, L_out, out) -> permute -> (N, out, L_out)
        out = torch.einsum("nlc,oc->nlo", sampled, weight_flat).permute(0, 2, 1).contiguous()

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        return out  # (N, out_channels, L_out)


class DeformConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_offset=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, modulation=True, norm=True, pool=None):
        super().__init__()
        self.conv = DeformConv1D(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                                 max_offset=max_offset, dilation=dilation, groups=groups, modulation=modulation)
        self.activation = nn.SiLU()
        self.normal = nn.BatchNorm1d(out_channels) if norm else None
        self.pool = pool

    def forward(self, x, mask=None):
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


class DeformableConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_offset=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, modulation=True, norm=True, pool=None, dropout_p=0.3):
        super().__init__()
        self.conv = DeformConvolutionLayer(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           bias=bias,
                                           max_offset=max_offset, dilation=dilation, groups=groups,
                                           modulation=modulation, norm=norm, pool=pool)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask=None):
        out = self.conv(x.transpose(1, 2))
        out = self.dropout(out).transpose(1, 2)
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        # return self.sequential(x).transpose(1, 2)
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

        self.post_conv = ResidualConnect(DeformableConvModule(in_channels=input_dim,
                                                              out_channels=input_dim,
                                                              kernel_size=5,
                                                              modulation=True,
                                                              max_offset=1,
                                                              stride=1,
                                                              padding=2,
                                                              bias=True,
                                                              norm=True,
                                                              dropout_p=conv_dropout_p))

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

        # out = self.global_sig(x, mask.bool())

        out = self.post_conv(out, mask)
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
