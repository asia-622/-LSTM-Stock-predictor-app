# ============================================================
# model.py — PyTorch LSTM model definition
# ============================================================

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Multi-layer LSTM with dropout for univariate time-series prediction.

    Architecture
    ------------
    Input  →  LSTM (num_layers, hidden_size)
           →  Dropout
           →  Fully-Connected (hidden_size → 1)
    Output →  Single next-step price prediction
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        output_size: int = 1,
    ):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Core LSTM — batch_first=True expects (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Regularisation between LSTM and FC head
        self.dropout = nn.Dropout(dropout)

        # Projection to scalar prediction
        self.fc = nn.Linear(hidden_size, output_size)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size, seq_len, input_size)

        Returns
        -------
        Tensor of shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # Initialise hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))   # out: (batch, seq, hidden)

        # Take the last time-step output
        out = out[:, -1, :]               # (batch, hidden)
        out = self.dropout(out)
        out = self.fc(out)                # (batch, output_size)
        return out

    # ----------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity-check ───────────────────────────────────────
if __name__ == "__main__":
    model = StockLSTM(input_size=1, hidden_size=128, num_layers=2, dropout=0.2)
    print(model)
    print(f"Trainable parameters: {model.count_parameters():,}")

    # Dummy forward pass
    dummy = torch.randn(16, 60, 1)   # batch=16, seq=60, features=1
    output = model(dummy)
    print(f"Output shape: {output.shape}")   # Expected: (16, 1)
