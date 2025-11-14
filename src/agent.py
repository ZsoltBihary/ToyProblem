import torch
import torch.nn.functional as F
from src.config import Config, State, Action, QValues
from src.q_model import QModel


class DQNAgent:
    def __init__(self, conf: Config):
        self.conf = conf
        # ===== Consume configuration parameters =====
        # === Data tensor sizes
        self.B = conf.batch_size
        self.T = conf.window_size
        self.A = conf.num_actions
        # ===== Reward specification
        self.gamma = conf.gamma
        # ===== Exploration parameters
        self.epsilon = conf.epsilon
        self.temperature = conf.temperature
        # ===== Training cycle control
        self.lr = conf.learning_rate
        self.use_ddqn = conf.use_ddqn

        # ===== Set up models, optimization =====
        self.online_model = QModel(conf)
        self.target_model = QModel(conf)
        self.update_target_model()
        self.optimizer = torch.optim.AdamW(
            self.online_model.parameters(),
            lr=self.lr
        )

    def act(self, state: State) -> Action:
        q_values = self.online_model(state)
        action = self.select_action(q_values)
        return action

    # ---------------------------------------------------------
    # ACTION SELECTION BASED ON Q-VALUES
    # ---------------------------------------------------------

    def select_action(self, q_values: QValues) -> Action:
        """
        Epsilon-soft strategy, robust version.
        Handles large logits, small temperatures, and avoids NaNs.
        """
        # --- Step 1: Scale by temperature ---
        inv_temp = 1.0 / max(self.temperature, 1e-6)
        logits = q_values * inv_temp

        # --- Step 2: Numerical stabilization ---
        # Subtract max along action dimension to prevent overflow in softmax
        logits = logits - logits.max(dim=-1, keepdim=True)[0]

        # --- Step 3: Softmax ---
        probs = F.softmax(logits, dim=-1)

        # --- Step 4: Epsilon-greedy smoothing ---
        probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)

        # --- Step 5: Sample action ---
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action

    # ---------------------------------------------------------
    # TRAINING STEP (supports DQN and DDQN)
    # ---------------------------------------------------------
    def train_step(self, batch):
        prices, positions, actions, rewards, next_prices, next_positions = batch

        with torch.no_grad():
            # ---- VANILLA DQN target ----
            if not self.use_ddqn:
                next_q_target = self.target_model((next_prices, next_positions))
                max_next_q = next_q_target.max(1, keepdim=True)[0]
                target = rewards.unsqueeze(1) + self.gamma * max_next_q
            # ---- DOUBLE DQN target ----
            else:
                # action selection: online model
                next_q_online = self.online_model((next_prices, next_positions))
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                # action evaluation: target model
                next_q_target = self.target_model((next_prices, next_positions))
                max_next_q = next_q_target.gather(1, next_actions)
                target = rewards.unsqueeze(1) + self.gamma * max_next_q

        # loss and optimization
        q_values = self.online_model((prices, positions))
        chosen_q = q_values.gather(1, actions.unsqueeze(1))  # shape (B,1)
        loss = F.mse_loss(chosen_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.online_model.state_dict())


if __name__ == "__main__":
    conf = Config(batch_size=4)
    trader = DQNAgent(conf)
    price_seq = torch.zeros((trader.B, trader.T), dtype=torch.float32)
    pos = torch.zeros(trader.B, dtype=torch.float32)
    state = price_seq, pos
    action = trader.act(state)
    print("Selected action:\n", action)
    print("Sanity check passed.")
