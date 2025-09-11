import torch.nn as nn
import torch.nn.functional as F
import torch

def get_activation(activation):
    if activation is None or activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()


class ConvolutionLayer(nn.Module):

    def __init__(self, in_dim, out_dim, k, stride=1, padding=0, bias=True, activation=None, norm=True, pool=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_dim, out_dim, k, stride=stride, padding=padding, bias=bias)
        self.activation = get_activation(activation)
        self.normal = nn.BatchNorm1d(out_dim) if norm else None
        self.pool = pool

    def forward(self, x):
        h = self.conv(x)
        if self.normal is not None:
            h = self.normal(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.pool is not None:
            h = self.pool(h)
        return h


class BaseConv(nn.Module):
    """
        conv layer,kernel = 3
    """

    def __init__(self, in_channel: int, out_channel: int, bias=False, activation='silu', norm=True):
        super().__init__()
        self.conv_3 = ConvolutionLayer(in_channel, out_channel, 3, stride=1, padding=1, bias=bias,
                                       activation=activation, norm=False)
        self.conv_1 = ConvolutionLayer(out_channel, out_channel, 1, stride=1, padding=0, bias=bias,
                                       activation=activation, norm=norm)

    def forward(self, x):
        feature = self.conv_3(x)
        return self.conv_1(feature)


class BaseResidual(nn.Module):
    """
    residual layer
    """

    def __init__(self, in_channels, out_channels, skip_conv=False, stride=1):
        super(BaseResidual, self).__init__()
        self.conv1 = BaseConv(in_channels, out_channels)
        self.conv2 = BaseConv(out_channels, out_channels)
        if skip_conv:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip_conv = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.skip_conv:
            x = self.skip_conv(x)
        return F.relu(x + y)


class BaseResidualBlock(nn.Module):
    """
    residual block
    """

    def __init__(self, in_channels, out_channels, num_residuals=2, dropout=0.3):
        super(BaseResidualBlock, self).__init__()
        self.residualBlock = nn.Sequential(
            BaseResidual(in_channels, out_channels, skip_conv=True, stride=1),
            *[BaseResidual(out_channels, out_channels) for _ in range(num_residuals - 1)],
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.residualBlock(x)
