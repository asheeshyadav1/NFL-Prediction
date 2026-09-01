"""The projection network.

An LSTM over the player's last N game lines; its final hidden state joins
pre-kickoff context (matchup, rest, role) through a small head. Kept small on
purpose -- 34k training player-weeks does not support more.
"""

from __future__ import annotations

import torch
from torch import nn


class ProjectionNet(nn.Module):
    def __init__(
        self,
        n_seq_features: int,
        n_ctx_features: int,
        hidden: int = 64,
        layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_seq_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + n_ctx_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        out = self.head(torch.cat([h_n[-1], ctx], dim=1)).squeeze(-1)
        # Points floor near zero; softplus avoids hard-clipping the gradient.
        return nn.functional.softplus(out)
