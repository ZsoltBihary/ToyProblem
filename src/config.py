import torch
from torch import Tensor

# ===== Aliases to make type-setting more expressive =====
PriceSeq = Tensor       # shape: (B, T)
PosSeq = Tensor         # shape: (B, T)
State = tuple[PriceSeq, PosSeq]
Action = Tensor         # shape: (B,)
Reward = Tensor         # shape: (B,)

# ===== For possible GPU acceleration in selected parts of the algorithm =====
GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def action_2_pos(action: Tensor) -> Tensor:
    """ Convert action [0, 1, 2] to position [-1.0, 0.0, 1.0] """
    return (action - 1).to(dtype=torch.float32)


def pos_2_action(pos: Tensor) -> Tensor:
    """ Convert position [-1.0, 0.0, 1.0] to action [0, 1, 2] """
    return pos.to(dtype=torch.long) + 1


class Config:
    # === Action interpretation as positions ===
    num_actions: int = 3  # A

    def __init__(
            self,
            # ===== Data tensor sizes =====
            batch_size: int = 128,  # B, number of parallel simulations
            window_size: int = 10,  # T, lookback window size

            # ===== Price dynamics =====
            S_mean: float = 100.0,
            volatility: float = 2.0,  # daily volatility
            mean_reversion: float = 0.1,  # inverse mean-reversion timescale

            # ===== Reward specification =====
            discount_factor: float = 0.99,  # One-period discount factor used in PV(reward)
            half_bidask: float = 1.0,  # bid-ask trading friction parameter
            risk_aversion: float = 0.02,  # weight on variance in mean-variance utility

            # ===== Epsilon-soft action selection =====
            epsilon: float = 0.1,  # for epsilon-greedy action selection
            temperature: float = 0.1,  # for soft action selection
    ):
        # ===== Data tensor sizes
        self.batch_size = batch_size
        self.window_size = window_size
        # ===== PriceSeq dynamics
        self.S_mean = S_mean
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        # ===== Reward specification
        self.discount_factor = discount_factor
        self.half_bidask = half_bidask
        self.risk_aversion = risk_aversion
        # ===== Epsilon-soft action selection
        self.epsilon = epsilon
        self.temperature = temperature
