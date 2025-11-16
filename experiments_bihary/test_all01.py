import torch
import matplotlib.pyplot as plt
from src.config import Config
from src.environment import Environment
from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.trainer import DQNTrainer
# from line_profiler_pycharm import profile

# conf = Config()
conf = Config(
    batch_size=512,
    window_size=1,
    rollout_steps=50,
    buffer_mult=1,
    learning_rate=0.005,

    use_ddqn=True,
    dueling=True,
    use_layernorm=False,
    dropout=0.0,
    num_blocks=2,
    hidden_dim=32
)

train_env = Environment(conf)
eval_env = Environment(conf)
agent = DQNAgent(conf)
buffer = ReplayBuffer(conf)

trainer = DQNTrainer(conf, train_env, eval_env, agent, buffer)
trainer.run(n_cycles=50)
print("Training finished.")

# Create s-grids, and pos-grids
mean = conf.S_mean
num_s = 1001
min_s, max_s = mean * 0.7, mean * 1.3
d_s = (max_s - min_s) / (num_s - 1.0)
s = (min_s + d_s * torch.arange(num_s)).unsqueeze(1)
pos = torch.tensor([-1.0, 0.0, 1.0]).repeat(num_s, 1)   # shape (1001, 3)
action = torch.zeros((num_s, 3), dtype=torch.long)      # shape (1001, 3)

for a in range(3):
    # pos_a = pos[a] * torch.ones(num_s)
    state = (s, pos[:, a])
    action[:, a] = agent.act(state, greedy=True)

off = num_s // 3 + 50
s_np = s[off: -off].cpu().numpy()
pos1d = torch.tensor([-1.0, 0.0, 1.0])
pos_np = pos1d[action[off: -off, :]].cpu().numpy()   # shape (3, 1001)

plt.figure(figsize=(8, 5))
plt.plot(s_np, pos_np[:, 0], label="from -1")
plt.plot(s_np, pos_np[:, 1], label="from 0")
plt.plot(s_np, pos_np[:, 2], label="from 1")

plt.xlabel("s")
plt.ylabel("action(s)")
plt.legend()
plt.grid(True)
plt.show()
