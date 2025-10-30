# python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class LSTMTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_labels: int,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0.0
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_out_dim, num_labels)

    def forward(self, x):
        emb = self.embedding(x)                 # [B, T, E]
        out, (h_n, c_n) = self.lstm(emb)        # h_n: [num_layers * num_directions, B, H]
        if self.lstm.bidirectional:
            # 取最后一层的正、反向隐状态拼接
            h_fwd = h_n[-2]                     # [B, H]
            h_bwd = h_n[-1]                     # [B, H]
            feat = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            feat = h_n[-1]                      # [B, H]
        feat = self.dropout(feat)
        logits = self.fc(feat)                  # [B, C]
        return logits


class CNNTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_labels: int,
        dropout_rate: float = 0.5,
        kernel_sizes=(3, 4, 5),
        num_filters=100
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=ks)
            for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, x):
        emb = self.embedding(x)                 # [B, T, E]
        emb = emb.transpose(1, 2)               # [B, E, T]
        conv_outs = [F.relu(conv(emb)) for conv in self.convs]          # list of [B, F, T']
        pooled = [F.max_pool1d(co, kernel_size=co.size(2)).squeeze(2) for co in conv_outs]  # [B, F]
        feat = torch.cat(pooled, dim=1)         # [B, F*len(K)]
        feat = self.dropout(feat)
        logits = self.fc(feat)                  # [B, C]
        return logits