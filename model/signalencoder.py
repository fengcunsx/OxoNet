import torch.nn as nn
import torch

from model.signallayer import SignalConvBlock, SignalBlock


class SignalEncoder(nn.Module):

    def __init__(self, in_channel=1, out_channel=128, block_count=4, dropout=0.3, sig_l=125, pos_mode="rope"):
        super(SignalEncoder, self).__init__()
        self.sig_len = sig_l
        self.conv = SignalConvBlock(in_dim=in_channel, out_dim=out_channel)
        self.blocks = nn.Sequential(
            *[SignalBlock(input_dim=out_channel, mha_dp=dropout, ffn_dp=dropout, conv_dropout_p=dropout,
                          pos_mode=pos_mode) for _ in range(block_count)])

    def creat_mask(self, sig_l, max_l=None):
        if max_l is None:
            max_l = self.sig_len
        b = sig_l.shape[0]
        range_tensor = torch.arange(max_l, dtype=torch.long, device=sig_l.device).expand(b, max_l)

        sig_ls = sig_l.unsqueeze(1)

        return range_tensor < sig_ls

    def forward(self, signal, sig_l=None):
        mask = None
        if sig_l is not None:
            mask = self.creat_mask(sig_l)
        signal = signal.permute(0, 2, 1).contiguous()
        signal, mask = self.conv(signal, mask)
        signal = signal.permute(0, 2, 1).contiguous()
        for block in self.blocks:
            signal = block(signal, mask)
        return signal, mask
