import torch
import matplotlib.pyplot as plt
from src.config import Config

# Consume configuration
conf = Config()
delta = conf.gamma
kappa = conf.mean_reversion * 1.0
beta = conf.half_bidask * 1.0
sigma = conf.volatility
gamma = conf.risk_aversion * sigma * sigma
mean = conf.S_mean
# Create s-grids, and pos-grids
num_s = 2501
min_s, max_s = 50.0, 150.0
d_s = (max_s - min_s) / (num_s - 1.0)
s = min_s + d_s * torch.arange(num_s)
pos = torch.tensor([-1.0, 0.0, 1.0])
# Create transition matrix
mean_z = s - kappa * (s - mean)
P = torch.exp(-torch.pow(s[None, :] - mean_z[:, None], 2.0) / (2.0 * sigma * sigma))
P /= P.sum(dim=1, keepdim=True).clamp_min(1e-12)
print(torch.round(100 * P[0:3, 0:100]).int())
# Sanity check for expected z
Ez = torch.einsum('ij, j->i', P, s)
print(Ez)

# Let us define the value, action-value, and action tensors V(a, s), Q(a, s; b), and action(a, s)
V = torch.zeros((3, num_s), dtype=torch.float32)
Q = torch.zeros((3, num_s, 3), dtype=torch.float32)
action = torch.zeros((3, num_s), dtype=torch.long)
# Let us perform the Bell-iteration ...
for t in range(5000):
    EQ1 = torch.einsum('sz, bz -> sb', P, V)
    Rew = kappa * (mean - s[None, :, None]) * pos[None, None, :]  # Expected return
    Rew = Rew - beta * torch.abs(pos[None, None, :] - pos[:, None, None])  # Bid-ask friction
    Rew = Rew - gamma * torch.pow(pos[None, None, :], 2.0)  # Risk aversion
    Q1 = delta * EQ1[None, :, :] + (1.0-delta) * Rew
    V1, action1 = torch.max(Q, dim=2)
    V_error = (V - V1).abs().max()
    Q_error = (Q - Q1).abs().max()
    action_error = (action - action1).abs().max()
    print(f"{t}  V err: {V_error.item():.6f}  "
          f"Q err: {Q_error.item():.6f}  "
          f"action err: {action_error.item()}")
    Q = Q1
    V = V1
    action = action1

print("Starting from the mean price, with 0 position, value = ", V[1, num_s//2].item())

s_np = s[1000:1501].cpu().numpy()
pos_np = pos[action[:, 1000:1501]].cpu().numpy()   # shape (3, 1001)

plt.figure(figsize=(8, 5))
plt.plot(s_np, pos_np[0], label="from -1")
plt.plot(s_np, pos_np[1], label="from 0")
plt.plot(s_np, pos_np[2], label="from 1")

plt.xlabel("s")
plt.ylabel("action(s)")
plt.legend()
plt.grid(True)
plt.show()

# s_np = s.cpu().numpy()
# V_np = V.cpu().numpy()   # shape (3, 1001)
#
# plt.figure(figsize=(8, 5))
# for i in range(3):
#     plt.plot(s_np, V_np[i], label=f"V[{i}]")
#
# plt.xlabel("s")
# plt.ylabel("V(s)")
# plt.legend()
# plt.grid(True)
# plt.show()

finish = 0
