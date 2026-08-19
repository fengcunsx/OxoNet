import torch.nn as nn
import torch

from model.Fusion import Fusion, PredHead
from model.seqencoder import SeqEncoder
from model.signalencoder import SignalEncoder


class DetectModel(nn.Module):
    def __init__(self, dim=128, dropout=0.3, sig_blocks=4, sig_l=125, seq_l=5, pos_mode="rope"):
        super().__init__()
        self.signal_encoder = SignalEncoder(out_channel=dim, dropout=dropout, block_count=sig_blocks, sig_l=sig_l,
                                            pos_mode=pos_mode)
        self.seq_encoder = SeqEncoder(out_channel=dim)
        self.connected = Fusion(dim=dim, dropout=dropout, seq_l=seq_l)
        # self.connected = PredHead(input_dim=dim)

    def forward(self, signal, seq, sig_l):
        signal, mask = self.signal_encoder(signal, sig_l)
        seq = self.seq_encoder(seq)
        feat, prob = self.connected(signal, seq, mask)
        # feat, prob = self.connected(signal, mask)
        return feat, prob
