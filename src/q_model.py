import torch
import torch.nn as nn
from src.config import Config, State, QValues


# ==============================================================
#   Residual Block (MLP)
# ==============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, use_layernorm: bool = False):
        super().__init__()

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = nn.ReLU()

    def forward(self, x):
        """x → LN → FC → ReLU → LN → FC → +x"""
        h = x

        if self.use_layernorm:
            h = self.norm1(h)
        h = self.act(self.fc1(h))

        if self.use_layernorm:
            h = self.norm2(h)
        h = self.fc2(h)

        # Residual connection
        return self.act(x + h)


# ==============================================================
#   Q-model with configurable residual blocks
# ==============================================================

class QModel(nn.Module):
    """
    Flatten (price_seq, pos) -> Q-values.
    Input:
        state: (price_seq, pos), shapes (B,T) and (B,)
    Output:
        q_values: (B, A)
    """

    def __init__(self, conf: Config):
        super().__init__()

        input_dim = conf.window_size + 1
        hidden = conf.hidden_dim
        num_blocks = getattr(conf, "num_blocks", 4)  # default 4 if not in conf
        use_layernorm = getattr(conf, "use_layernorm", False)

        # Initial projection to hidden dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )

        # Residual backbone
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden, use_layernorm))

        self.residual_stack = nn.Sequential(*blocks)

        # Final projection to action Q-values
        self.output_layer = nn.Linear(hidden, conf.num_actions)

    # ----------------------------------------------------------

    def forward(self, state: State) -> QValues:
        price_seq, pos = state

        x = torch.cat([price_seq, pos.unsqueeze(-1)], dim=-1)

        x = self.input_layer(x)
        x = self.residual_stack(x)
        q_values = self.output_layer(x)

        return q_values

#
# class QModel(nn.Module):
#     """
#     Flatten (price_seq, pos) -> Q-values.
#     Input:
#         state: (price_seq, pos), where
#             price_seq: (B, T)
#             pos: (B,)
#     Output:
#         q_values: (B, A)
#     """
#     # def __init__(self, window_size: int, n_actions: int, hidden_dim: int = 32):
#     def __init__(self, conf: Config):
#         super().__init__()
#         # ===== Use configuration parameters =====
#         self.net = nn.Sequential(
#             nn.Linear(conf.window_size + 1, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.num_actions)
#         )
#
#     def forward(self, state: State) -> QValues:
#         price_seq, pos = state
#         x = torch.cat([price_seq, pos.unsqueeze(-1)], dim=-1)
#         q_values = self.net(x)
#         return q_values


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
