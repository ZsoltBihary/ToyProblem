# import torch
# from src.config import Config
# from src.environment import Environment
# from src.agent import Agent
#
# conf = Config()
# market = Environment(conf)
# trader = Agent(conf)
#
# n_show = 2
# print("\n ***** Initial state of market environment:")
# market.print_env(n=n_show)
# # action = 1 * torch.ones(conf.batch_size, dtype=torch.long)  # Short
# state = market.price_seq.detach().clone(), market.pos_seq.detach().clone()
# for t in range(3):
#     action = trader.act(state)
#     (state, reward) = market.step(action)
#
# print("\n ***** Final state of market environment:")
# market.print_env(n=n_show)
