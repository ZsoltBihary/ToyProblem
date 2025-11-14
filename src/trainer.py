from torch.utils.data import DataLoader
from src.config import Config
from src.environment import Environment
from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from line_profiler_pycharm import profile


class DQNTrainer:
    def __init__(self, conf: Config, env: Environment, agent: DQNAgent, buffer: ReplayBuffer):
        self.conf = conf
        self.env = env
        self.agent = agent
        self.buffer = buffer

        # Extract config parameters
        self.rollout_steps = conf.rollout_steps
        self.num_epochs = conf.num_epochs
        self.minibatch_size = conf.minibatch_size
        self.target_update = conf.target_update

    # ------------------------------------------------------------
    # Rollout (collect experience)
    # ------------------------------------------------------------
    @profile
    def rollout(self):
        state = self.env.get_state()

        for _ in range(self.rollout_steps):
            action = self.agent.act(state)
            next_state, reward = self.env.step(action)

            self.buffer.push(
                state,
                action,
                reward,
                next_state
            )
            state = next_state

    # ------------------------------------------------------------
    # Training (optimize Q network)
    # ------------------------------------------------------------
    @profile
    def train(self):
        if len(self.buffer) == 0:
            return 0.0

        dataloader = DataLoader(
            self.buffer,
            batch_size=self.minibatch_size,
            shuffle=True
        )

        losses = []
        for _ in range(self.num_epochs):
            for batch in dataloader:
                loss = self.agent.train_step(batch)
                losses.append(loss)

        return sum(losses) / len(losses)

    # ------------------------------------------------------------
    # One full cycle of rollout + training
    # ------------------------------------------------------------
    @profile
    def cycle(self, cycle_idx):
        self.rollout()
        avg_loss = self.train()

        if cycle_idx % self.target_update == 0:
            self.agent.update_target_model()

        return avg_loss

    # ------------------------------------------------------------
    # Train for N cycles
    # ------------------------------------------------------------
    @profile
    def run(self, n_cycles, verbose=True):
        for i in range(1, n_cycles + 1):
            avg_loss = self.cycle(i)
            if verbose:
                print(
                    f"[Cycle {i:04d}] "
                    f"buffer {len(self.buffer)}/{self.buffer.capacity} | "
                    f"avg_loss={avg_loss:.6f}"
                )


if __name__ == "__main__":
    conf = Config(
        batch_size=128,
        window_size=10,
        rollout_steps=100,
        buffer_mult=2,
        learning_rate=1e-3,
        use_ddqn=True
    )

    env = Environment(conf)
    agent = DQNAgent(conf)
    buffer = ReplayBuffer(conf)

    trainer = DQNTrainer(conf, env, agent, buffer)

    trainer.run(n_cycles=5)
    print("Training finished.")
