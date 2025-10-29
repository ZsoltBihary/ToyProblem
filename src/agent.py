import torch
from torch import Tensor
import torch.nn.functional as F
from src.config import Config, PriceSeq, PosSeq, State, Action, pos_2_action


class Agent:
    def __init__(self, conf: Config):
        self.conf = conf
        # ===== Consume configuration parameters =====
        # === Data tensor sizes
        self.B = conf.batch_size
        self.T = conf.window_size
        self.A = conf.num_actions
        # === Price dynamics
        self.S_mean = conf.S_mean
        self.vol = conf.volatility
        self.kappa = conf.mean_reversion
        # === Reward specification
        self.disc_f = conf.discount_factor
        self.half_ba = conf.half_bidask
        self.risk_av = conf.risk_aversion
        # ===== Epsilon-soft action selection
        self.eps = conf.epsilon
        self.temp = conf.temperature

    def act(self, state: State) -> Action:
        # For future reference:
        # encoded = self.encode(state)
        # model_logits = self.model(encoded)
        # action = self.select_action(model_logits)

        # For now, let us implement a trivial example - always Long
        action = 2 * torch.ones(self.B, dtype=torch.long)
        return action

    def encode(self, state: State):
        # Trivial example - no encoding
        return state

    def model(self, encoded):
        # Trivial example - this leads to uniform random action
        model_logits = torch.zeros((self.B, self.A), dtype=torch.float32)
        return model_logits

    def select_action(self, model_logits) -> Action:
        """
        Epsilon-soft strategy, robust version.
        Handles large logits, small temperatures, and avoids NaNs.
        """
        # --- Step 1: Scale by temperature ---
        logits = model_logits / self.temp

        # --- Step 2: Numerical stabilization ---
        # Subtract max along action dimension to prevent overflow in softmax
        logits = logits - logits.max(dim=-1, keepdim=True)[0]

        # --- Step 3: Softmax ---
        probs = F.softmax(logits, dim=-1)

        # --- Step 4: Epsilon-greedy smoothing ---
        probs = (1.0 - self.eps) * probs + self.eps / probs.size(-1)

        # --- Step 5: Clamp for safety ---
        probs = torch.clamp(probs, min=1e-8, max=1.0)

        # --- Step 6: Sample action ---
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action


if __name__ == "__main__":
    conf = Config(batch_size=4)
    trader = Agent(conf)
    price_seq = torch.zeros((trader.B, trader.T), dtype=torch.float32)
    pos_seq = torch.zeros((trader.B, trader.T), dtype=torch.float32)
    state = price_seq, pos_seq
    action = trader.act(state)
    print("Selected action:\n", action)
    print("Sanity check passed.")
