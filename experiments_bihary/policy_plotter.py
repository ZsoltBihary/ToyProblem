import torch
import matplotlib.pyplot as plt


class PolicyPlotter:
    def __init__(self, agent, num_s=501, smin=90.0, smax=110.0, device="cpu"):
        self.agent = agent
        self.device = device

        self.num_s = num_s
        self.s = torch.linspace(smin, smax, num_s, device=device).unsqueeze(1)
        self.pos_vals = torch.tensor([-1.0, 0.0, 1.0], device=device)

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

    def update(self):
        actions = torch.zeros((self.num_s, 3), dtype=torch.long, device=self.device)

        for i, pos in enumerate(self.pos_vals):
            pos_vec = pos.expand(self.num_s)
            state = (self.s, pos_vec)
            actions[:, i] = self.agent.act(state, greedy=True)

        pos_np = self.pos_vals[actions].cpu().numpy()
        s_np = self.s.cpu().numpy()

        self.ax.clear()
        self.ax.plot(s_np, pos_np[:, 0], label="from -1")
        self.ax.plot(s_np, pos_np[:, 1], label="from 0")
        self.ax.plot(s_np, pos_np[:, 2], label="from 1")

        self.ax.set_xlabel("s")
        self.ax.set_ylabel("action(s)")
        self.ax.legend()
        self.ax.grid(True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# class PolicyPlotter:
#     """
#     Plots greedy policies π(s) = argmax_a Q(s,a)
#     for last_price ∈ [min_s, max_s] and pos ∈ {-1,0,+1}.
#     Can overlay multiple curves (movie during training).
#     """
#
#     def __init__(self, min_s=90.0, max_s=110.0, num_s=501,
#                  pos_vals=(-1.0, 0.0, 1.0),
#                  device="cpu",
#                  overlay_alpha=0.25):
#         """
#         overlay_alpha: transparency for older policies in movie mode
#         """
#
#         self.min_s = min_s
#         self.max_s = max_s
#         self.num_s = num_s
#         self.pos_vals = torch.tensor(pos_vals, dtype=torch.float32, device=device)
#         self.overlay_alpha = overlay_alpha
#         self.device = device
#
#         # Precompute the price grid
#         d_s = (max_s - min_s) / (num_s - 1)
#         self.s = (min_s + d_s * torch.arange(num_s, device=device)).unsqueeze(1)
#
#         # Prepare interactive plot
#         plt.ion()
#         self.fig, self.ax = plt.subplots(figsize=(8, 5))
#
#     # -------------------------------------------------------------
#
#     @torch.no_grad()
#     def compute_policy(self, agent):
#         """
#         Compute greedy policy for each pos and each price s.
#         Returns: actions_np shape (num_s, num_pos)
#         """
#         num_pos = len(self.pos_vals)
#         actions = torch.zeros((self.num_s, num_pos), dtype=torch.long, device=self.device)
#
#         for i, pos in enumerate(self.pos_vals):
#             # pos_vec = torch.tensor([-1.0, 0.0, 1.0]).repeat(self.num_s, 1)
#             # pos_vec = pos.expand(self.num_s)
#             pos_vec = torch.full((self.num_s,), pos, device=self.device)
#             state = (self.s, pos_vec)
#             actions[:, i] = agent.act(state, greedy=True)
#
#         return actions.cpu().numpy()  # shape: (num_s, 3)
#
#     # -------------------------------------------------------------
#
#     def plot_policy(self, agent, overlay=False, label=None):
#         """
#         Plot one policy. If overlay=False, clear the axes first.
#         If overlay=True, draw semi-transparent curves over previous ones.
#         """
#
#         actions_np = self.compute_policy(agent)     # (num_s, 3)
#         pos_np = np.array(self.pos_vals.cpu())      # e.g. [-1, 0, +1]
#         s_np = self.s.cpu().numpy().flatten()       # 1D array
#
#         if not overlay:
#             self.ax.cla()   # start fresh
#
#         # Colors for pos = -1, 0, +1
#         colors = ["red", "black", "blue"]
#
#         for i, color in enumerate(colors):
#             # Convert action index → new_pos using pos_vals array
#             next_pos = pos_np[actions_np[:, i]]
#             self.ax.plot(
#                 s_np,
#                 next_pos,
#                 color=color,
#                 alpha=self.overlay_alpha if overlay else 1.0,
#                 label=(None if overlay else f"from {pos_np[i]:.0f}")
#             )
#
#         if not overlay:
#             self.ax.set_title("Greedy Policy")
#             self.ax.set_xlabel("s")
#             self.ax.set_ylabel("action(s) → next position")
#             self.ax.grid(True)
#             self.ax.legend()
#
#         if label is not None:
#             # annotate training step on the chart
#             self.ax.text(0.02, 0.95, label, transform=self.ax.transAxes,
#                          fontsize=10, color="green")
#
#         self.fig.canvas.draw()
#         plt.pause(0.001)
#
#     # -------------------------------------------------------------
#
#     def close(self):
#         """Turn off interactive mode and show the final figure."""
#         plt.ioff()
#         plt.show()
