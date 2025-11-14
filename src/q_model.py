import torch
import torch.nn as nn
from src.config import Config, State, QValues


class QModel(nn.Module):
    """
    Flatten (price_seq, pos) -> Q-values.
    Input:
        state: (price_seq, pos), where
            price_seq: (B, T)
            pos: (B,)
    Output:
        q_values: (B, A)
    """
    # def __init__(self, window_size: int, n_actions: int, hidden_dim: int = 32):
    def __init__(self, conf: Config):
        super().__init__()
        # ===== Use configuration parameters =====
        self.net = nn.Sequential(
            nn.Linear(conf.window_size + 1, conf.hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hidden_dim, conf.hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hidden_dim, conf.num_actions)
        )

    def forward(self, state: State) -> QValues:
        price_seq, pos = state
        x = torch.cat([price_seq, pos.unsqueeze(-1)], dim=-1)
        q_values = self.net(x)
        return q_values
