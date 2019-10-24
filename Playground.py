import environment as Env
from environment import Environment
from dqn_model import DQN
from agent_dqn import Agent_DQN

import argparse
from test import test
import torchvision
import torchvision.transforms as transforms
from collections import namedtuple
import matplotlib.pyplot as plt
import torch
from PIL import Image

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

args = parse()
env = Environment('BreakoutNoFrameskip-v4', "", atari_wrapper=True, test=False)
n = env.action_space
state = env.reset()
device = torch.device("cpu")
input = torch.tensor(state,device=device)
agent = Agent_DQN(env,args)
dqn = DQN()
torch.save(dqn.state_dict(),"checkpoint.pth")
state_dict = torch.load("checkpoint.pth")

agent.train()
# Experience = namedtuple(
#             'Experience',
#             ('state','action','next_state','reward')
#         )
# e = Experience(state,action,next_state,reward)
# print(e)
# agent.train()











# print("shape:",input.shape)
#
# input = input.unsqueeze(0).to(device)
# input = input.permute(0, 3, 1, 2)
# input = input.to(device=device, dtype=torch.float)
#
# print("shape:",input.shape)
# print(input.dtype)
#
# Q = d(input)
# action = Q.argmax(dim = 1).item()
# action = 1
# observation, reward, done, info = env.step(action)
# memory = []
# memory.append([frame,observation,action,Q[0]])
# print(memory)


