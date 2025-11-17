import torch
import matplotlib.pyplot as plt
from src.config import Config
from src.environment import Environment
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.dqn_trainer import DQNTrainer
from policy_plotter import PolicyPlotter

# conf = Config()
conf = Config(
    batch_size=128,
    window_size=1,
    rollout_steps=100,
    buffer_mult=2,
    learning_rate=0.005,

    use_ddqn=True,
    dueling=True,
    use_layernorm=False,
    dropout=0.0,
    num_blocks=5,
    hidden_dim=32
)
n_cycles = 50

train_env = Environment(conf)
eval_env = Environment(conf)
agent = DQNAgent(conf)
buffer = ReplayBuffer(conf)
trainer = DQNTrainer(conf, train_env, eval_env, agent, buffer)
plotter = PolicyPlotter(agent)

for i in range(1, n_cycles + 1):
    # Update scheduled parameters
    progress = i / n_cycles
    trainer.agent.lr = max(trainer.lr0 / 100.0, trainer.lr0 * (1.0 - 1.5 * progress))
    trainer.agent.epsilon = max(trainer.eps0 / 5.0, trainer.eps0 * (1.0 - 1.2 * progress))
    trainer.agent.temperature = max(trainer.temp0 / 10.0, trainer.temp0 * (1.0 - progress))
    # One full cycle of rollout + training + evaluation
    avg_loss, avg_ro_rew, avg_ev_rew = trainer.cycle(i)

    print(
        f"[{i:03d}/{n_cycles}] "
        f"buffer {len(trainer.buffer)}/{trainer.buffer.capacity} | "
        f"lr= {trainer.agent.lr:.6f} | "
        f"eps= {trainer.agent.epsilon:.3f} | "
        f"temp= {trainer.agent.temperature:.4f} | "
        f"avg_loss= {avg_loss:.4f} | "
        f"avg_rollout_rew= {avg_ro_rew:.4f} | "
        f"avg_eval_rew= {avg_ev_rew:.4f} "
    )
    # ---- Plot current policy ----
    if i % 1 == 0:    # update movie frame
        plotter.update()

print("Training finished.")
plt.ioff()       # disable interactive mode
plt.show()       # window stays open
