"""Sequence models for continuous sign language recognition.

Both variants share a dilated temporal-convolution encoder and differ only in
the temporal head, which isolates the architecture comparison (RQ3):

    TCN encoder  ->  2-layer BiLSTM        ->  linear  ->  log-softmax
    TCN encoder  ->  Transformer encoder   ->  linear  ->  log-softmax

The encoder applies one stride-2 pooling step, so output length is T // 2.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNEncoder(nn.Module):
    """Linear projection followed by residual dilated Conv1d blocks."""

    def __init__(self, in_dim: int, d_model: int = 256, dropout: float = 0.2,
                 dilations: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
            nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=d, dilation=d),
                nn.BatchNorm1d(d_model), nn.ReLU(inplace=True), nn.Dropout(dropout))
            for d in dilations])
        self.pool = nn.MaxPool1d(2, ceil_mode=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, T, D_in)
        h = self.proj(x).transpose(1, 2)                     # (B, d, T)
        for blk in self.blocks:
            h = h + blk(h)
        return self.pool(h).transpose(1, 2)                  # (B, T//2, d)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CSLRModel(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, head: str = "bilstm",
                 d_model: int = 256, hidden: int = 256, layers: int = 2,
                 nhead: int = 8, dropout: float = 0.2):
        super().__init__()
        self.head_type = head
        self.encoder = TCNEncoder(in_dim, d_model, dropout)
        if head == "bilstm":
            self.rnn = nn.LSTM(d_model, hidden, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=dropout if layers > 1 else 0.0)
            out_dim = hidden * 2
        elif head == "transformer":
            self.pos = PositionalEncoding(d_model)
            layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, dropout,
                                               batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(layer, num_layers=max(layers, 4))
            out_dim = d_model
        else:
            raise ValueError(f"unknown head '{head}'")
        self.classifier = nn.Linear(out_dim, vocab_size + 1)   # +1 for CTC blank

    @staticmethod
    def out_lengths(in_lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(in_lengths, 2, rounding_mode="floor").clamp(min=1)

    def forward(self, x: torch.Tensor, in_lengths: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)                                   # (B, T2, d)
        out_len = self.out_lengths(in_lengths)
        if self.head_type == "bilstm":
            h, _ = self.rnn(h)
        else:
            mask = (torch.arange(h.size(1), device=h.device)[None, :] >= out_len[:, None].to(h.device))
            h = self.tr(self.pos(h), src_key_padding_mask=mask)
            h = torch.nan_to_num(h)
        return F.log_softmax(self.classifier(h), dim=-1)      # (B, T2, V+1)


def build_model(cfg: dict, in_dim: int, vocab_size: int) -> CSLRModel:
    return CSLRModel(in_dim, vocab_size, head=cfg.get("head", "bilstm"),
                     d_model=cfg.get("d_model", 256), hidden=cfg.get("hidden", 256),
                     layers=cfg.get("layers", 2), dropout=cfg.get("dropout", 0.2))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
