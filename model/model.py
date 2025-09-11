import torch.nn as nn
import torch

from model.Fusion import Fusion
from model.seqencoder import SeqEncoder
from model.signalencoder import SignalEncoder


class DetectModel(nn.Module):
    def __init__(self, dim=128, dropout=0.3, sig_blocks=4, sig_l=125, seq_l=5):
        super().__init__()
        self.signal_encoder = SignalEncoder(out_channel=dim, dropout=dropout, block_count=sig_blocks, sig_l=sig_l)
        self.seq_encoder = SeqEncoder(out_channel=dim)
        self.connected = Fusion(dim=dim, dropout=dropout, seq_l=seq_l)

    def forward(self, signal, mean, std, dwell, seq, sig_l):
        signal, mask = self.signal_encoder(signal, sig_l)
        base_stat = torch.cat((mean, std, dwell), dim=-1)
        seq = self.seq_encoder(base_stat, seq)
        feat, prob = self.connected(signal, seq, mask)
        return feat, prob
