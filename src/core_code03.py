"""
batched_dqn_trading_single_asset.py

Simplified batched Deep Q-Learning scaffold for a continuing trading environment.

Changes from the multi-asset version:
- Single asset only (no M dimension)
- Use 'prices' instead of 'price_window' naming
- Tensor shapes:
    prices: (B, T)
    positions: (B,)
- Actions: discrete {0,1,2} mapped to {-1,0,+1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from typing import Tuple


# -----------------------------------------------------------------------------
# 1) BatchedTradingEnv (single asset)
# -----------------------------------------------------------------------------
class BatchedTradingEnv:
    """
    Batched continuous trading environment for one asset.

    Args:
        batch_size (B): number of parallel universes
        window_size (T): lookback window size
        total_len: total synthetic price length
        device: torch device
    State:
        (prices: (B, T), positions: (B,))
    Action:
        actions_idx: (B,) in {0,1,2} -> {-1,0,+1}
    Reward:
        reward: (B,)
    """
    def __init__(self, batch_size: int, window_size: int, total_len: int = 50_000, device='cpu'):
        assert total_len > window_size
        self.device = device
        self.B = batch_size
        self.T = window_size
        self.total_len = total_len

        # Synthetic price paths
        self.prices_all = torch.randn(self.B, self.total_len, device=self.device).cumsum(dim=-1)
        self.ptr = torch.full((self.B,), fill_value=self.T, dtype=torch.long, device=self.device)
        self.positions = torch.zeros(self.B, device=self.device)

    def reset(self):
        self.ptr[:] = self.T
        self.positions.zero_()
        return self._get_state()

    def _action_index_to_change(self, actions_idx: torch.Tensor) -> torch.Tensor:
        """Map discrete {0,1,2} -> {-1,0,+1}"""
        return (actions_idx.to(torch.int64) - 1).float()  # (B,)

    def step(self, actions_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Perform one step for all universes."""
        assert actions_idx.shape == (self.B,)

        idx = self.ptr
        price_t = self.prices_all[torch.arange(self.B, device=self.device), idx - 1]
        price_tp1 = self.prices_all[torch.arange(self.B, device=self.device), idx]
        price_diff = price_tp1 - price_t

        change = self._action_index_to_change(actions_idx)
        self.positions = torch.tanh(self.positions + change)

        reward = self.positions * price_diff
        self.ptr = (self.ptr + 1) % self.total_len

        next_state = self._get_state()
        return next_state, reward

    def _get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current (prices, positions)"""
        idx_base = self.ptr.unsqueeze(-1) - torch.arange(self.T, device=self.device)
        idx = idx_base % self.total_len
        prices = self.prices_all.gather(dim=-1, index=idx)
        return prices, self.positions.clone()


# -----------------------------------------------------------------------------
# 2) QNetwork
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Flatten (prices, positions) -> Q-values.
    Input:
        prices: (B, T)
        positions: (B,)
    Output:
        q_values: (B, n_actions)
    """
    def __init__(self, window_size: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        state_dim = window_size + 1
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state_tuple: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        prices, positions = state_tuple
        x = torch.cat([prices, positions.unsqueeze(-1)], dim=-1)
        return self.net(x)


# -----------------------------------------------------------------------------
# 3) ReplayBufferDataset
# -----------------------------------------------------------------------------
class ReplayBufferDataset(Dataset):
    """
    Circular replay buffer for individual transitions.
    Each push() adds a batch of B transitions.
    """
    def __init__(self, capacity: int, window_size: int, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.T = window_size

        self.prices = torch.zeros((capacity, window_size), device=self.device)
        self.positions = torch.zeros(capacity, device=self.device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        self.rewards = torch.zeros(capacity, device=self.device)
        self.next_prices = torch.zeros((capacity, window_size), device=self.device)
        self.next_positions = torch.zeros(capacity, device=self.device)

    def push(self, state, actions_idx, rewards, next_state):
        prices, positions = state
        next_prices, next_positions = next_state
        B = prices.shape[0]

        write_end = self.ptr + B
        if write_end <= self.capacity:
            self._write_slice(self.ptr, write_end,
                              prices, positions, actions_idx, rewards, next_prices, next_positions)
            self.ptr = write_end % self.capacity
        else:
            first = self.capacity - self.ptr
            second = B - first
            self._write_slice(self.ptr, self.capacity,
                              prices[:first], positions[:first],
                              actions_idx[:first], rewards[:first],
                              next_prices[:first], next_positions[:first])
            self._write_slice(0, second,
                              prices[first:], positions[first:],
                              actions_idx[first:], rewards[first:],
                              next_prices[first:], next_positions[first:])
            self.ptr = second
        self.size = min(self.size + B, self.capacity)

    def _write_slice(self, start, end, prices, positions, actions_idx, rewards, next_prices, next_positions):
        self.prices[start:end] = prices
        self.positions[start:end] = positions
        self.actions[start:end] = actions_idx
        self.rewards[start:end] = rewards
        self.next_prices[start:end] = next_prices
        self.next_positions[start:end] = next_positions

    def __len__(self): return self.size

    def __getitem__(self, idx: int):
        return (self.prices[idx],
                self.positions[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_prices[idx],
                self.next_positions[idx])


# -----------------------------------------------------------------------------
# 4) DQNAgent
# -----------------------------------------------------------------------------
class DQNAgent:
    """Standard DQN Agent"""
    def __init__(self, window_size: int, n_actions: int = 3, lr: float = 1e-3,
                 gamma: float = 0.99, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions

        self.q_net = QNetwork(window_size, n_actions).to(device)
        self.target_net = QNetwork(window_size, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.update_target_network()

    def select_action(self, state_tuple, epsilon: float) -> torch.Tensor:
        prices, positions = state_tuple
        B = prices.shape[0]
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (B,), device=self.device)
        with torch.no_grad():
            q = self.q_net(state_tuple)
            return torch.argmax(q, dim=1)

    def train_step(self, batch):
        prices, positions, actions_idx, rewards, next_prices, next_positions = batch
        prices, positions, actions_idx, rewards, next_prices, next_positions = \
            [x.to(self.device) for x in (prices, positions, actions_idx, rewards, next_prices, next_positions)]

        q_values = self.q_net((prices, positions))
        chosen_q = q_values.gather(1, actions_idx.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_net((next_prices, next_positions))
            target = rewards.unsqueeze(1) + self.gamma * next_q.max(1, keepdim=True)[0]

        loss = F.mse_loss(chosen_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# -----------------------------------------------------------------------------
# 5) Training cycle
# -----------------------------------------------------------------------------
def train_rollout_cycle(env, agent, buffer, rollout_steps, epochs, dataloader_batch_size, verbose=True):
    """One rollout+train cycle"""
    state = env._get_state()
    for _ in range(rollout_steps):
        actions_idx = agent.select_action(state, epsilon=0.1)
        next_state, rewards = env.step(actions_idx)
        buffer.push(state, actions_idx, rewards, next_state)
        state = next_state

    if len(buffer) == 0:
        return

    dataloader = DataLoader(buffer, batch_size=dataloader_batch_size, shuffle=True)
    losses = []
    for _ in range(epochs):
        for batch in dataloader:
            loss = agent.train_step(batch)
            losses.append(loss)

    agent.update_target_network()

    if verbose:
        print(f"Cycle done | buffer {len(buffer)}/{buffer.capacity} | "
              f"avg_loss={sum(losses)/len(losses):.6f}")


# -----------------------------------------------------------------------------
# 6) Example main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = 128
    T = 100
    R = 100
    E = 2
    capacity = B * 200
    dataloader_bs = 256
    cycles = 10

    torch.manual_seed(0)
    random.seed(0)

    env = BatchedTradingEnv(batch_size=B, window_size=T, total_len=50_000, device=device)
    agent = DQNAgent(window_size=T, n_actions=3, lr=1e-3, gamma=0.99, device=device)
    buffer = ReplayBufferDataset(capacity=capacity, window_size=T, device=device)
    env.reset()

    for _ in range(cycles):
        train_rollout_cycle(env, agent, buffer, R, E, dataloader_bs)

    print("Done demo training cycles.")
