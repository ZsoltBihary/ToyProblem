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


if __name__ == "__main__":
    from src.environment import Environment
    from src.config import Config
    from torchinfo import summary

    # --- Create config and environment ---
    conf = Config()
    env = Environment(conf)
    state = env.get_state()  # (price_seq, pos)

    # --- Create the model ---
    model = QModel(conf)
    output = model(state)
    print(f"Output shape: {output.shape}")

    # --- Wrap model in a tiny nn.Module for torchinfo ---
    class QModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, price_seq, pos):
            return self.model((price_seq, pos))

    wrapper = QModelWrapper(model)

    # --- Run summary ---
    summary(
        wrapper,
        input_data=(state[0], state[1]),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        verbose=1
    )
